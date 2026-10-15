"""DualVLN driving for the B2W + Z1, shared by vln_drive.py (navigation tests, live demo)
and fetch_box.py --vln (language -> walk to the table -> scripted pick).

Import AFTER the AppLauncher has started (it imports Isaac Lab).

    import vln_nav
    vln_nav.add_cameras(cfg)                         # before gym.make
    nav = vln_nav.Navigator(robot, cam, cam_down, dt, sim_step, server)
    log = nav.run("Walk to the wooden table and stop in front of it.", out_dir, max_time=60)

`sim_step(cmd)` must advance the simulation by one control step with the base velocity
command cmd = (vx, vy, wz) in the robot frame (model_2600 drives the legs and wheels).

Two cameras at the head camera's spot (0.42 m ahead of base_link, 0.29 m up), set up as
DualVLN was trained in Habitat: 640x480, 79 deg horizontal FOV, one level and one pitched
30 deg down (Habitat's LOOKDOWN tilts 2 x 15 deg; the model asks for it, action 5, before
it picks a pixel goal). Both stream RGB + depth to the DualVLN server
(Isaac/vln/dualvln_server.py, its own container). DualVLN answers with a short trajectory
in the robot frame (x forward, y left, metres) -- or a discrete action: 0 STOP, 1 forward,
2 left, 3 right. The trajectory is fixed in the world with the pose at which the image was
taken and followed by pure pursuit; discrete moves run to completion before the next
request, as in Habitat. A depth bumper stops the body short of what is in front of it.
"""

import io
import json
import math
import os
import threading
import time

import numpy as np
import requests
import torch
from PIL import Image, ImageDraw

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg

MIN_SPEED, MIN_TURN = 0.25, 0.45  # model_2600's dead zones (see fetch_box.py)
LOOKAHEAD = 0.6                    # m, pure pursuit
ACC_V, DEC_V, ACC_W = 0.8, 1.5, 2.5  # m/s^2, m/s^2, rad/s^2: command slew limits (smoothing)
HFOV, LOOK_DOWN = 79.0, 30.0       # deg: DualVLN's Habitat camera and its look-down tilt
# The head camera is a RealSense D455's colour camera (OmniVision OV9782), with the lens of Isaac Sim's
# own rsd455.usd: focal 1.93 mm on a 3.896 x 2.453 mm sensor = 90 x 65 deg, 16:10 (1280 x 800 native).
# DualVLN still gets what it was trained on: the exact 79 deg, 4:3 centre crop of it, at 640 x 480.
FRONT_MODEL = "RealSense D455 colour (OV9782)"
D455_FOCAL, D455_APERTURE_H = 1.93, 3.896  # mm, rsd455.usd Camera_OmniVision_OV9782_Color
CAM_W, CAM_H = 640, 400                    # rendered (half the D455's 1280 x 800)
VLN_W, VLN_H = 640, 480                    # what DualVLN gets
CAM_POS = (0.42, 0.0, 0.29)        # base_link frame: the head camera (f_oc_link/front_camera)
# depth bumper: DualVLN was trained for a human-height agent and drives the B2W's head into
# table tops; the cameras sit at the front of the body, so their depth is the clearance
BUMPER_STOP, BUMPER_SLOW = 0.35, 0.8  # m from the camera: stop / start slowing (contact at ~0.18: the body reaches ahead of it; ~0.07 overshoot)
BODY_HALF_W = 0.287 + 0.05           # m: hip half width + margin (fetch_box.HIP_HALF_WIDTH)
BLOCKED_END = 3.0                    # s held by the bumper while the model still drives on -> end "BLOCKED"
TABLE_RECT = (4.0, 4.6, 0.5, 1.5)    # x0, x1, y0, y1 of the fetch scene's table top (fetch_env_cfg TABLE_*)
AT_TABLE = 1.0                       # m from base centre to the table edge: the body reaches ~0.45 m ahead


def add_cameras(cfg):
    """Add vln_cam (level: the D455 head camera) and vln_cam_down (the same camera pitched 30 deg down,
    for DualVLN's look-down view) to the scene config. "world" convention = x forward, z up, so pitching
    down is +LOOK_DOWN about y."""
    lens = sim_utils.PinholeCameraCfg(focal_length=D455_FOCAL, horizontal_aperture=D455_APERTURE_H,
                                      clipping_range=(0.05, 30.0))
    for name, pitch in (("vln_cam", 0.0), ("vln_cam_down", LOOK_DOWN)):
        setattr(cfg.scene, name, CameraCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/base_link/{name}", spawn=lens,
            offset=CameraCfg.OffsetCfg(pos=CAM_POS, rot=(math.cos(math.radians(pitch) / 2), 0.0,
                                                         math.sin(math.radians(pitch) / 2), 0.0), convention="world"),
            width=CAM_W, height=CAM_H, data_types=["rgb", "distance_to_image_plane"], update_period=0.0,
        ))


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def table_edge_dist(x, y, rect=None):
    """Distance (m) from a point to the table top's rectangle (0 inside); default: the module's TABLE_RECT
    (looked up at call time -- fetch_box sets it to the target table in the three-table scene)."""
    x0, x1, y0, y1 = rect if rect is not None else TABLE_RECT
    return math.hypot(max(x0 - x, 0.0, x - x1), max(y0 - y, 0.0, y - y1))


def normalise_instruction(text):
    """DualVLN was trained on R2R-style sentences: "walk to the table" sends it wandering
    where "Walk to the table." does not."""
    text = text.strip()
    text = text[0].upper() + text[1:]
    return text if text[-1] in ".!?" else text + "."


def parse_command(line):
    """A typed line -> ("quit",) / ("stop",) / ("home",) / ("go", sentence) / None."""
    text = line.strip()
    if not text:
        return None
    word = text.lower().rstrip(".!")
    if word in ("q", "quit", "exit"):
        return ("quit",)
    if word in ("stop", "home"):
        return (word,)
    return ("go", text)


def start_command_readers(commands, watch_path, say=print):
    """Feed parse_command() results into the queue `commands` from stdin and from the file
    watch_path (every time the file CHANGES, its last line; what is there at start is stale)."""
    import sys

    def stdin_reader():
        for line in sys.stdin:
            cmd = parse_command(line)
            if cmd:
                commands.put(cmd)

    def file_watcher():
        last = os.path.getmtime(watch_path) if os.path.exists(watch_path) else None
        while True:
            try:
                if os.path.exists(watch_path) and os.path.getmtime(watch_path) != last:
                    last = os.path.getmtime(watch_path)
                    with open(watch_path) as f:
                        lines = f.read().strip().splitlines()
                    cmd = parse_command(lines[-1]) if lines else None
                    if cmd:
                        say(f"[file] {lines[-1].strip()}")
                        commands.put(cmd)
            except Exception as exc:  # noqa: BLE001
                say(f"watch error: {exc}")
            time.sleep(0.1)

    threading.Thread(target=stdin_reader, daemon=True).start()
    threading.Thread(target=file_watcher, daemon=True).start()


def png(a):
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG")
    return buf.getvalue()


def depth_png(depth):
    d = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    return png(np.clip(d * 10000.0, 0, 65535).astype(np.uint16))  # 0.1 mm units


def grab(c):
    return (c.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8),
            c.data.output["distance_to_image_plane"][0, ..., 0].cpu().numpy())


def vln_intrinsic():
    """DualVLN's camera: 640 x 480, 79 deg, centred (4 x 4, as the server takes it)."""
    K = np.eye(4)
    K[0, 0] = K[1, 1] = VLN_W / 2 / math.tan(math.radians(HFOV / 2))
    K[0, 2], K[1, 2] = VLN_W / 2, VLN_H / 2
    return K


def to_vln(rgb, depth, K):
    """The exact 79 deg, 4:3 centre crop of a wider pinhole image (intrinsics K), resized to 640 x 480:
    a pinhole image cropped about its principal point is a narrower pinhole image."""
    import cv2
    fx, cx, cy = K[0, 0], K[0, 2], K[1, 2]
    hw = fx * math.tan(math.radians(HFOV / 2))  # half width of the crop, px
    hh = hw * VLN_H / VLN_W
    x0, x1, y0, y1 = int(round(cx - hw)), int(round(cx + hw)), int(round(cy - hh)), int(round(cy + hh))
    assert x0 >= 0 and y0 >= 0 and x1 <= rgb.shape[1] and y1 <= rgb.shape[0], "camera narrower than DualVLN's 79 deg"
    rgb = cv2.resize(rgb[y0:y1, x0:x1], (VLN_W, VLN_H), interpolation=cv2.INTER_LINEAR)
    depth = cv2.resize(np.ascontiguousarray(depth[y0:y1, x0:x1]), (VLN_W, VLN_H), interpolation=cv2.INTER_NEAREST)
    return rgb, depth


def ask(server, rgb, depth, rgb_down, depth_down, instruction, intrinsic, reset, rid, s2_blocking, session="default"):
    """POST one observation (level + look-down view); returns the server's JSON reply."""
    files = {"image": ("rgb.png", png(rgb), "image/png"), "depth": ("depth.png", depth_png(depth), "image/png"),
             "image_down": ("rgb_down.png", png(rgb_down), "image/png"),
             "depth_down": ("depth_down.png", depth_png(depth_down), "image/png")}
    form = {"json": json.dumps({"reset": reset, "instruction": instruction, "intrinsic": intrinsic, "rid": rid,
                                "s2_blocking": s2_blocking, "session": session})}
    r = requests.post(f"{server}/eval_dual", files=files, data=form, timeout=120)
    r.raise_for_status()
    return r.json()


def follow(path_w, x, y, yaw):
    """Pure pursuit on a world-frame polyline -> (vx, 0, wz) for model_2600."""
    d = np.hypot(path_w[:, 0] - x, path_w[:, 1] - y)
    i0 = int(np.argmin(d))
    rest = path_w[i0:]
    seg = np.r_[0.0, np.cumsum(np.hypot(np.diff(rest[:, 0]), np.diff(rest[:, 1])))]
    j = int(np.searchsorted(seg, LOOKAHEAD))
    tgt = rest[min(j, len(rest) - 1)]
    to_end = float(np.hypot(path_w[-1, 0] - x, path_w[-1, 1] - y))
    if to_end < 0.12:
        return (0.0, 0.0, 0.0), True
    herr = wrap(math.atan2(tgt[1] - y, tgt[0] - x) - yaw)
    wz = max(-0.8, min(0.8, 1.6 * herr))
    vx = min(0.6, 0.9 * to_end) * max(0.0, math.cos(herr)) ** 2
    if abs(herr) > 1.0:           # mostly a turn: in place, at least the policy's minimum
        vx, wz = 0.0, math.copysign(max(abs(wz), MIN_TURN), wz)
    elif vx < MIN_SPEED:          # below the policy's dead zone: it would not move at all
        vx = MIN_SPEED
    return (vx, 0.0, wz), False


def to_pose(goal, x, y, yaw):
    """Discrete actions (Habitat's 0.25 m steps / 15 deg turns) -> drive to the goal pose:
    head for its position, then turn to its yaw. Returns (command, reached). A pure turn
    servoes the heading only: turning in place creeps forward, and chasing the start
    point again would spin the robot round."""
    gx, gy, gyaw, moves = goal
    dist = math.hypot(gx - x, gy - y)
    if moves and dist > 0.08:
        herr = wrap(math.atan2(gy - y, gx - x) - yaw)
        if abs(herr) > 0.35:
            return (0.0, 0.0, math.copysign(max(MIN_TURN, min(0.8, 1.6 * abs(herr))), herr)), False
        return (max(MIN_SPEED, min(0.5, dist)), 0.0, max(-0.8, min(0.8, 1.6 * herr))), False
    yerr = wrap(gyaw - yaw)
    if abs(yerr) > math.radians(4):
        return (0.0, 0.0, math.copysign(max(MIN_TURN, min(0.8, 1.6 * abs(yerr))), yerr)), False
    return (0.0, 0.0, 0.0), True


def motion_stats(m, dt):
    """How smooth the drive was: measured from the base velocity, not the commands."""
    if len(m) < 3:
        return {}
    v, w = m[:, 3], m[:, 4]
    moving = np.abs(v) > 0.15
    stops = int(np.sum(moving[:-1] & ~moving[1:]))  # moving -> stopped transitions
    acc = np.diff(v) / dt
    jerk = np.diff(acc) / dt
    wacc = np.diff(w) / dt
    return {"stops": stops, "rms_acc": round(float(np.sqrt(np.mean(acc ** 2))), 2),
            "rms_jerk": round(float(np.sqrt(np.mean(jerk ** 2))), 1),
            "rms_ang_acc": round(float(np.sqrt(np.mean(wacc ** 2))), 2),
            "mean_speed": round(float(np.mean(np.abs(v))), 3), "moving_frac": round(float(np.mean(moving)), 2)}


class Smoother:
    """Slew-rate limit on (vx, wz). The follower's raw commands jump: a new path every
    0.4 s, the 0.25 m/s dead-zone floor, turn-in-place switching. Braking is allowed to
    be quicker than speeding up; a stop from below 0.3 m/s is passed through at once."""

    def __init__(self, dt, enabled=True):
        self.dt, self.enabled = dt, enabled
        self.v = self.w = 0.0

    def __call__(self, cmd):
        vx, _, wz = cmd
        if not self.enabled or (vx == 0.0 and wz == 0.0 and abs(self.v) < 0.3):
            self.v, self.w = vx, wz
            return cmd
        dv = vx - self.v
        lim = (ACC_V if abs(vx) > abs(self.v) else DEC_V) * self.dt
        self.v += max(-lim, min(lim, dv))
        self.w += max(-ACC_W * self.dt, min(ACC_W * self.dt, wz - self.w))
        v = self.v
        if 0.0 < abs(v) < MIN_SPEED and abs(vx) >= MIN_SPEED:  # ramping up through the dead zone: skip it
            v = self.v = math.copysign(MIN_SPEED, v)
        return (v, 0.0, self.w)


class Planner:
    """One DualVLN request at a time, on a thread, so the simulation can keep stepping
    (async); with sync=True the request is waited for."""

    def __init__(self, sync):
        self.sync = sync
        self.th, self.done = None, None

    def busy(self):
        return self.th is not None

    def start(self, capture, **kw):
        def run():
            try:
                self.done = (capture, ask(**kw), None)
            except Exception as e:  # noqa: BLE001
                self.done = (capture, None, e)
        self.done = None
        self.th = threading.Thread(target=run, daemon=True)
        self.th.start()
        if self.sync:
            self.th.join()

    def poll(self):
        if self.th is not None and not self.th.is_alive():
            self.th = None
            return self.done
        return None


class Navigator:
    """Drives the robot with DualVLN until it says STOP, the bumper holds it for
    BLOCKED_END s, max_time runs out, or should_stop() is true.

    sync: the simulation waits for each answer (default). Async keeps driving the last
    path while System 2 thinks, as on a real robot, but System 2 then plans on 2-3 s old
    images: 0/5 at the table vs 4/5 (2026-10-05)."""

    def __init__(self, robot, cam, cam_down, dt, sim_step, server, sync=True, smooth=True, bumper=True,
                 vln_period=0.4, say=print, should_stop=None, on_step=None, driver=None):
        self.robot, self.cam, self.cam_down, self.dt = robot, cam, cam_down, dt
        self.sim_step, self.server, self.sync = sim_step, server, sync
        self.smooth_on, self.bumper_on, self.period = smooth, bumper, vln_period
        self.say, self.should_stop, self.on_step = say, should_stop or (lambda: False), on_step
        self.intrinsic = None
        self._rays = None
        # driver: None -> our pure pursuit follows DualVLN's trajectory; otherwise an object with
        # new_path(path_w) (each new trajectory, world frame), command() -> (vx, vy, wz), stop()
        # -- fetch_box's Nav2 drivers (FollowPath / NavigateToPose)
        self.driver = driver
        self.current_path = None  # DualVLN's latest trajectory in the world (the recorder's plan label)
        import uuid
        self.session = uuid.uuid4().hex[:8]  # the server keeps one DualVLN memory per session

    def base_pose(self):
        p = self.robot.data.root_pos_w[0]
        q = self.robot.data.root_quat_w[0]
        yaw = math.atan2(2 * (float(q[0]) * float(q[3]) + float(q[1]) * float(q[2])),
                         1 - 2 * (float(q[2]) ** 2 + float(q[3]) ** 2))
        return float(p[0]), float(p[1]), yaw

    def clearance(self):
        """Free distance ahead of the body: the nearest depth point, from both cameras, inside
        the body's corridor (|y| < BODY_HALF_W) and height band (8 cm above the floor to 20 cm
        above the camera). The level camera alone loses a table top's edge inside ~0.35 m."""
        zc = CAM_POS[2] + float(self.robot.data.root_pos_w[0, 2])  # camera height above the floor
        best = math.inf
        for c, pitch in ((self.cam, 0.0), (self.cam_down, math.radians(LOOK_DOWN))):
            d = c.data.output["distance_to_image_plane"][0, ..., 0]
            if self._rays is None:
                h, w = d.shape
                v_, u_ = torch.meshgrid(torch.arange(h, device=d.device), torch.arange(w, device=d.device), indexing="ij")
                fx, cx, cy = self.intrinsic[0, 0], self.intrinsic[0, 2], self.intrinsic[1, 2]
                self._rays = (-(u_ - cx) / fx, -(v_ - cy) / fx)  # left, up per metre of depth
            left, up = self._rays
            f = torch.nan_to_num(d, nan=0.0, posinf=0.0)
            X = f * math.cos(pitch) + up * f * math.sin(pitch)
            Z = zc - f * math.sin(pitch) + up * f * math.cos(pitch)
            ok = (f > 0.02) & (X < 1.5) & ((left * f).abs() < BODY_HALF_W) & (Z > 0.08) & (Z < zc + 0.2)
            if bool(ok.any()):
                best = min(best, float(X[ok].min()))
        return best

    def bumper(self, cmd, clear):
        """Cap forward speed by the clearance. Returns (command, held)."""
        vx, vy, wz = cmd
        if not self.bumper_on or vx <= 0.0 or clear >= BUMPER_SLOW:
            return cmd, False
        if clear < BUMPER_STOP:
            return (0.0, vy, wz), True
        cap = MIN_SPEED + (0.6 - MIN_SPEED) * (clear - BUMPER_STOP) / (BUMPER_SLOW - BUMPER_STOP)
        return (min(vx, cap), vy, wz), False

    def run(self, instruction, out, max_time=60.0):
        """One instruction. Writes out/log.json, out/frames/ (level | look-down image per
        request, trajectory in green) and out/goals/ (each System 2 pixel goal, red, on the
        image it was picked in). Returns the log; log["camvid"] holds the frame images
        (not written to the JSON)."""
        if self.intrinsic is None:  # the camera's own (bumper); DualVLN gets vln_intrinsic() crops
            self.intrinsic = np.eye(4)
            self.intrinsic[:3, :3] = self.cam.data.intrinsic_matrices[0].cpu().numpy()
        os.makedirs(os.path.join(out, "frames"), exist_ok=True)
        os.makedirs(os.path.join(out, "goals"), exist_ok=True)
        self.say(f"instruction: {instruction!r}")
        dt, robot, intrinsic = self.dt, self.robot, vln_intrinsic()
        x, y, yaw = self.base_pose()
        log = {"instruction": instruction, "start": [round(x, 3), round(y, 3), round(math.degrees(yaw), 1)],
               "requests": [], "path": []}
        camvid = []
        path_w, stopped, reason, k, t_sim, t_wall = None, False, "timeout", 0, 0.0, time.time()
        goal, goal_deadline, next_req = None, 0.0, 0.0
        planner, smooth, frozen, captures = Planner(self.sync), Smoother(dt, self.smooth_on), 0.0, []
        clear, blocked = math.inf, 0.0
        log["motion"] = []  # per control step: t, commanded vx wz, measured vx wz, clearance (m, 9 = none)
        while t_sim < max_time:
            if self.should_stop():
                reason = "interrupted"
                break
            x, y, yaw = self.base_pose()
            if not planner.busy() and goal is None and t_sim >= next_req - 1e-6:
                # discrete moves run to completion before the next request, as in Habitat
                next_req = t_sim + self.period
                (rgb, depth), (rgb_down, depth_down) = (to_vln(*grab(c), self.intrinsic) for c in (self.cam, self.cam_down))
                t_req, rid = time.time(), len(captures)
                captures.append(dict(rid=rid, x=x, y=y, yaw=yaw, t=t_sim, rgb=rgb, rgb_down=rgb_down))
                if rid >= 60:
                    captures[rid - 60] = None  # System 2 never lags that far; free the images
                planner.start(captures[rid], server=self.server, rgb=rgb, depth=depth, rgb_down=rgb_down,
                              depth_down=depth_down, instruction=instruction, intrinsic=intrinsic.tolist(),
                              reset=(k == 0), rid=rid, s2_blocking=self.sync, session=self.session)
                if self.sync:
                    frozen += time.time() - t_req
            done = planner.poll()
            if done is not None:
                cap, rep, err = done
                if err is not None:
                    self.say(f"request failed: {err}")
                    reason = f"server error: {err}"
                    break
                # the answer is about the scene at capture time: anchor it to that pose
                cx, cy, cyaw = cap["x"], cap["y"], cap["yaw"]
                entry = {"rid": cap["rid"], "t": round(cap["t"], 2), "t_answer": round(t_sim, 2),
                         "pose": [round(cx, 3), round(cy, 3), round(math.degrees(cyaw), 1)], "infer_s": rep.get("infer_s")}
                s2cap = captures[rep["s2_rid"]] if rep.get("s2_rid") is not None and captures[rep["s2_rid"]] else None
                if s2cap is not None:  # System 2 just finished, on an earlier image
                    entry.update(s2_rid=rep["s2_rid"], s2_s=rep.get("s2_s"), pixel_goal=rep.get("pixel_goal"),
                                 looked_down=rep.get("looked_down"))
                    if rep.get("pixel_goal"):
                        g = Image.fromarray(np.concatenate([s2cap["rgb"], s2cap["rgb_down"]], axis=1))
                        r_, c_ = rep["pixel_goal"]  # (row, col) in the look-down image (or the level one)
                        u_ = (640 if rep.get("looked_down") else 0) + c_
                        ImageDraw.Draw(g).ellipse([u_ - 9, r_ - 9, u_ + 9, r_ + 9], outline=(255, 40, 40), width=3)
                        g.save(os.path.join(out, "goals", f"{rep['s2_rid']:04d}.jpg"), quality=90)
                img = Image.fromarray(np.concatenate([cap["rgb"], cap["rgb_down"]], axis=1))  # level | look-down
                draw = ImageDraw.Draw(img)
                if rep.get("wait"):
                    entry.update(kind="wait")
                elif "trajectory" in rep:
                    traj = np.asarray(rep["trajectory"], dtype=np.float64)[:, :2]
                    c, s = math.cos(cyaw), math.sin(cyaw)
                    path_w = np.c_[cx + c * traj[:, 0] - s * traj[:, 1], cy + s * traj[:, 0] + c * traj[:, 1]]
                    path_w = path_w[3:] if len(path_w) > 4 else path_w  # the upstream client skips the first 3 too
                    self.current_path = path_w
                    if self.driver is not None and len(path_w) >= 2:
                        self.driver.new_path(path_w)
                    entry.update(kind="trajectory", len_m=round(float(np.hypot(*traj[-1])), 2),
                                 end_robot=[round(float(v), 2) for v in traj[-1]], traj=np.round(traj, 3).tolist())
                    for px_, py_ in traj[::3]:  # the trajectory, projected into the level image
                        if px_ > 0.3:
                            zc = CAM_POS[2] + float(robot.data.root_pos_w[0, 2])  # camera height above the floor
                            fx = intrinsic[0, 0]
                            u_, v_ = 320 - fx * py_ / px_, 240 + fx * zc / px_
                            if 0 <= u_ < 640 and 0 <= v_ < 480:
                                draw.ellipse([u_ - 3, v_ - 3, u_ + 3, v_ + 3], fill=(40, 255, 40))
                else:
                    acts = rep.get("discrete_action", [])
                    entry.update(kind="discrete", actions=acts)
                    if 0 in acts:
                        stopped, reason = True, "STOP"
                    else:  # 0.25 m steps / 15-deg turns, composed into one goal pose (the upstream client's PID mode)
                        # turns are relative to the heading System 2 saw; steps start from where the robot is now
                        px, py, pyaw = x, y, (s2cap or cap)["yaw"]
                        for a in acts:
                            if a == 1:
                                px, py = px + 0.25 * math.cos(pyaw), py + 0.25 * math.sin(pyaw)
                            elif a in (2, 3):
                                pyaw += math.radians(15 if a == 2 else -15)
                        path_w = None
                        if any(a in (1, 2, 3) for a in acts):
                            goal, goal_deadline = (px, py, pyaw, 1 in acts), t_sim + 1.0 + 1.0 * len(acts)
                draw.text((8, 8), f"t={cap['t']:5.1f}s  {entry['kind']}", fill=(255, 255, 0))
                draw.text((8, 460), instruction[:100], fill=(255, 255, 255))
                img.save(os.path.join(out, "frames", f"{len(log['requests']):04d}.jpg"), quality=90)
                camvid.append(np.asarray(img))
                log["requests"].append(entry)
                self.say(f"t={cap['t']:5.1f}s pos ({cx:+.2f},{cy:+.2f}) yaw {math.degrees(cyaw):+6.1f} -> {entry['kind']} "
                         f"{entry.get('actions', entry.get('end_robot'))} pixel {entry.get('pixel_goal')} ({entry['infer_s']} s)")
                if stopped:
                    break
            if goal is not None:
                cmd, reached = to_pose(goal, x, y, yaw)
                if reached or t_sim > goal_deadline:
                    goal, next_req = None, t_sim
            elif path_w is not None and len(path_w) >= 2:
                cmd = self.driver.command() if self.driver is not None else follow(path_w, x, y, yaw)[0]
            else:
                cmd = (0.0, 0.0, 0.0)
            if k % 5 == 0:
                clear = self.clearance()
            cmd, held = self.bumper(cmd, clear)
            blocked = blocked + dt if held else 0.0
            if blocked > BLOCKED_END:
                reason = "BLOCKED"  # the model keeps driving into what is in front: as far as the body goes
                break
            if self.driver is None:  # Nav2 smooths its own commands (velocity_smoother)
                cmd = smooth(cmd)
            self.sim_step(cmd)
            lv, av = robot.data.root_lin_vel_b[0], robot.data.root_ang_vel_b[0]
            log["motion"].append([round(t_sim, 3), round(cmd[0], 3), round(cmd[2], 3),
                                  round(float(lv[0]), 3), round(float(av[2]), 3), round(min(clear, 9.0), 3)])
            if self.on_step is not None:
                self.on_step(k)
            if k % 10 == 0:
                log["path"].append([round(x, 3), round(y, 3)])
            k += 1
            t_sim += dt

        if self.driver is not None:
            self.driver.stop()
        self.current_path = None
        self.sim_step((0.0, 0.0, 0.0))
        x, y, yaw = self.base_pose()
        path = np.array(log["path"])
        log.update(motion_stats(np.array(log["motion"]).reshape(-1, 6), dt), frozen_wall_s=round(frozen, 1),
                   mode="sync" if self.sync else "async", smoothing=self.smooth_on, bumper=self.bumper_on,
                   follower=getattr(self.driver, "name", "pure_pursuit"),
                   end=[round(x, 3), round(y, 3), round(math.degrees(yaw), 1)], reason=reason, sim_time_s=round(t_sim, 1),
                   wall_time_s=round(time.time() - t_wall, 1), table_edge_m=round(table_edge_dist(x, y), 2),
                   path_len_m=round(float(np.sum(np.hypot(*np.diff(path, axis=0).T))) if len(path) > 1 else 0.0, 2))
        with open(os.path.join(out, "log.json"), "w") as f:
            json.dump(log, f, indent=1)
        log["camvid"] = camvid
        self.say(f"motion: {log.get('stops')} stops, rms acc {log.get('rms_acc')} m/s2, rms ang acc "
                 f"{log.get('rms_ang_acc')} rad/s2, mean speed {log.get('mean_speed')} m/s, frozen {log['frozen_wall_s']} s "
                 f"({log['mode']})")
        self.say(f"END {reason}: at ({x:+.2f},{y:+.2f}) yaw {math.degrees(yaw):+.0f}, {log['table_edge_m']:.2f} m from the "
                 f"table edge, {log['path_len_m']} m driven in {t_sim:.1f} s sim ({log['wall_time_s']} s wall), "
                 f"{len(log['requests'])} requests -> {out}")
        return log
