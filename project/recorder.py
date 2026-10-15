"""Episode recorder for the whole-body VLA dataset (Phase 5) -- raw format, version 2.

The simulation writes a simple raw format; tools/raw_to_lerobot.py converts it to LeRobot
v2.1 (parquet + mp4) outside the container (pyarrow is not in the Isaac image).

    rec = recorder.Recorder(env, robot, front_cam, wrist_cam, root="/workspace/datasets/raw/<run>", fps=10)
    rec.begin(meta)                                   # language, target, start pose, seed, ...
    rec.step(cmd, arm_q, grip, phase, plan_w=None)    # EVERY control step; keeps every (50/fps)-th
    rec.end(outcome)                                  # -> ep_XXXXXX/{front.mp4, wrist.mp4, data.npz, meta.json}

What is recorded (the whole-body VLA paper's inputs and outputs), per frame at fps (10 Hz):

  VISION        front RGB (the head camera: a RealSense D455 colour camera, 90 x 65 deg, 16:10 -> 320 x 200;
                DualVLN sees a 79 deg crop of it) and wrist RGB (a RealSense D405 in its holder, 87 x 56 deg,
                16:9 -> 320 x 180), as h264 mp4 -- 320 wide, the camera's own aspect; front and wrist DEPTH,
                distance to the image plane, uint16 millimetres (0 = no return / > 10 m / below the D405's
                7 cm minimum), at half the image size, in data.npz; meta.json["cameras"] has each camera's
                model, size and field of view (older recordings: 320 x 240, 79 deg front, 60 deg wrist)
  LANGUAGE      meta.json: the task sentence ("Pick up the red box on the green table.") and the
                navigation sentence DualVLN was given ("Walk to the green table ...")
  PROPRIOCEPTION  joint positions AND velocities of all 23 joints (legs 12, wheels 4, arm 6, gripper 1),
                base linear / angular velocity and projected gravity (base frame), base height,
                end-effector pose in the base frame (pos 3 + quat wxyz 4)
  ACTION        base velocity command (vx, vy, wz) as sent to the locomotion controller; end-effector
                target = the EE pose 1/fps later, in the base frame AT THIS FRAME; arm joint targets (6)
                and gripper target (1) as sent to the arm drives
                -- the previous action and longer histories / future chunks are the neighbouring frames
                (the converter adds observation.prev_action; LeRobot's delta_timestamps build windows)
  LABELS        phase per frame (PHASES), the planned path ahead in the base frame (10 points every
                0.2 m: DualVLN's trajectory while walking, Nav2's plan while it drives, the line to the
                pick point while docking; NaN when none), base pose in the world (evaluation only);
                per episode: target table / box, success, delivered, DualVLN arrived / right table
"""

import json
import math
import os

import numpy as np

FORMAT = 2
PHASES = ["idle", "walk", "backoff", "dock", "pick", "carry", "deliver", "nav2", "nav2_dock"]
IMG = (320, 240)    # w, h: the cameras' 4:3 at half the VLN resolution
DEPTH = (160, 120)  # w, h of the stored depth
DEPTH_MAX = 10.0    # m
PLAN_N, PLAN_STEP = 10, 0.2  # plan points, spacing (m)
EE_BODY = "gripperStator"
JOINT_GROUPS = {
    "legs": [f"{l}_{p}_joint" for l in ("FL", "FR", "RL", "RR") for p in ("hip", "thigh", "calf")],
    "wheels": [f"{l}_foot_joint" for l in ("FL", "FR", "RL", "RR")],
    "arm": [f"joint{k}" for k in range(1, 7)],
    "gripper": ["jointGripper"],
}
JOINT_NAMES = [n for g in JOINT_GROUPS.values() for n in g]  # 23, the order of joint_pos / joint_vel


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _quat_rot(q, v):
    qv = np.concatenate([[0.0], v])
    return _quat_mul(_quat_mul(q, qv), q * np.array([1, -1, -1, -1]))[1:]


def to_base(root_pos, root_quat, p_w, q_w):
    """World pose -> pose in the base frame (pos, quat wxyz)."""
    qi = root_quat * np.array([1, -1, -1, -1])
    p_b = _quat_rot(qi, p_w - root_pos)
    q_b = _quat_mul(qi, q_w)
    return np.concatenate([p_b, q_b / np.linalg.norm(q_b)])


WRIST = {"model": "generic", "size_wh": IMG, "depth_wh": DEPTH, "min_depth_m": 0.0}  # set by add_wrist_camera
D405_MIN_DEPTH = 0.07  # m: the real D405 returns no depth closer than this


def wrist_camera_info(usd_path):
    """The wrist camera as the robot asset defines it: a RealSense D405 (tools/wrist_d405.py) or the bare
    camera; image size keeps the camera's aspect (the D405 is 16:9)."""
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(usd_path)
    cam = next((p for p in stage.Traverse() if p.IsA(UsdGeom.Camera) and p.GetName() == "end_effector_camera"), None)
    d405 = any(p.GetName() == "wrist_d405" for p in stage.Traverse())
    info = {"model": "RealSense D405" if d405 else "generic", "size_wh": IMG, "depth_wh": DEPTH,
            "min_depth_m": D405_MIN_DEPTH if d405 else 0.0}
    if cam is not None and d405:  # the bare camera keeps the 320 x 240 every earlier dataset has
        c = UsdGeom.Camera(cam)
        f, ha, va = c.GetFocalLengthAttr().Get(), c.GetHorizontalApertureAttr().Get(), c.GetVerticalApertureAttr().Get()
        h = int(round(IMG[0] * va / ha / 2)) * 2
        info.update(size_wh=(IMG[0], h), depth_wh=(DEPTH[0], h // 2),
                    hfov_deg=round(math.degrees(2 * math.atan(ha / 2 / f)), 1),
                    vfov_deg=round(math.degrees(2 * math.atan(va / 2 / f)), 1))
    return info


def add_wrist_camera(cfg):
    """The robot's end-effector camera as scene.wrist_cam, RGB + depth, at its own aspect ratio."""
    from isaaclab.sensors import CameraCfg
    WRIST.update(wrist_camera_info(cfg.scene.robot.spawn.usd_path))
    w, h = WRIST["size_wh"]
    cfg.scene.wrist_cam = CameraCfg(prim_path="{ENV_REGEX_NS}/Robot/z1/link06/end_effector_camera", spawn=None,
                                    width=w, height=h, data_types=["rgb", "distance_to_image_plane"],
                                    update_period=0.0)


def plan_ahead(plan_w, x, y, yaw):
    """A world-frame path [(x, y), ...] -> PLAN_N points every PLAN_STEP m from its point nearest the
    robot, in the base frame (x forward, y left); the end repeats once the path runs out."""
    out = np.full((PLAN_N, 2), np.nan, np.float32)
    if plan_w is None or len(plan_w) < 2:
        return out
    p = np.asarray(plan_w, np.float64)[:, :2]
    i0 = int(np.argmin(np.hypot(p[:, 0] - x, p[:, 1] - y)))
    p = p[i0:]
    if len(p) < 2:
        p = np.vstack([p, p])
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    q = np.stack([np.interp(np.arange(1, PLAN_N + 1) * PLAN_STEP, s, p[:, k]) for k in (0, 1)], axis=1)
    c, sn = math.cos(yaw), math.sin(yaw)
    dx, dy = q[:, 0] - x, q[:, 1] - y
    out[:, 0], out[:, 1] = c * dx + sn * dy, -sn * dx + c * dy
    return out


def rec_size(cam, width=IMG[0]):
    """Recorded image size: 320 wide, the camera's own aspect (even height, for h264)."""
    w, h = cam.cfg.width, cam.cfg.height
    hh = int(round(width * h / w / 2)) * 2
    return (width, hh), (width // 2, hh // 2)


class Recorder:
    def __init__(self, env, robot, front_cam, wrist_cam, root, fps=10, front_model="head camera"):
        self.env, self.robot, self.front, self.wrist = env, robot, front_cam, wrist_cam
        self.cams = {}
        for name, cam, model, min_d in (("front", front_cam, front_model, 0.0),
                                        ("wrist", wrist_cam, WRIST["model"], WRIST["min_depth_m"])):
            size, dsize = rec_size(cam)
            self.cams[name] = {"model": model, "size_wh": size, "depth_wh": dsize, "min_depth_m": min_d,
                               "render_wh": (cam.cfg.width, cam.cfg.height)}
        self.root, self.fps = root, fps
        self.every = max(1, int(round(1.0 / (fps * env.step_dt))))
        os.makedirs(root, exist_ok=True)
        try:
            os.chmod(root, 0o777)
        except OSError:
            pass
        self.n = len([d for d in os.listdir(root) if d.startswith("ep_")])
        self.joints = robot.find_joints(JOINT_NAMES, preserve_order=True)[0]
        assert len(self.joints) == 23, robot.joint_names
        self.ee = robot.find_bodies([EE_BODY])[0][0]
        self.on = False

    def begin(self, meta):
        self.meta = dict(meta)
        self.k = 0
        self.frames = {"front": [], "wrist": []}
        self.cols = {k: [] for k in ("joint_pos", "joint_vel", "base", "ee_pose", "depth_front", "depth_wrist",
                                     "action_base", "action_arm", "ee_w", "phase", "plan", "world", "t")}
        self.on = True

    def _img(self, cam, size=IMG):
        import cv2
        a = cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        return a if a.shape[1::-1] == tuple(size) else cv2.resize(a, tuple(size), interpolation=cv2.INTER_AREA)

    def _depth(self, cam, size=DEPTH, min_m=0.0):
        import cv2
        d = cam.data.output["distance_to_image_plane"][0, ..., 0].cpu().numpy().astype(np.float32)
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        d[(d <= min_m) | (d > DEPTH_MAX)] = 0.0
        d = cv2.resize(d, tuple(size), interpolation=cv2.INTER_NEAREST)
        return np.round(d * 1000.0).astype(np.uint16)

    def step(self, cmd, arm_q, grip, phase, plan_w=None):
        """Call after every env step (control rate); records every self.every-th."""
        if not self.on:
            return
        k = self.k
        self.k += 1
        if k % self.every:
            return
        d = self.robot.data
        rp, rq = d.root_pos_w[0].cpu().numpy(), d.root_quat_w[0].cpu().numpy()
        ep_w, eq_w = d.body_pos_w[0, self.ee].cpu().numpy(), d.body_quat_w[0, self.ee].cpu().numpy()
        jp, jv = d.joint_pos[0].cpu().numpy(), d.joint_vel[0].cpu().numpy()
        yaw = math.atan2(2 * (rq[0] * rq[3] + rq[1] * rq[2]), 1 - 2 * (rq[2] ** 2 + rq[3] ** 2))
        self.cols["joint_pos"].append(jp[self.joints].astype(np.float32))
        self.cols["joint_vel"].append(jv[self.joints].astype(np.float32))
        self.cols["base"].append(np.concatenate([d.root_lin_vel_b[0].cpu().numpy(), d.root_ang_vel_b[0].cpu().numpy(),
                                                 d.projected_gravity_b[0].cpu().numpy(), [rp[2]]]).astype(np.float32))
        self.cols["ee_pose"].append(to_base(rp, rq, ep_w, eq_w).astype(np.float32))
        self.cols["action_base"].append(np.asarray(cmd, np.float32))
        self.cols["action_arm"].append(np.concatenate([np.asarray(arm_q, np.float32).ravel()[:6], [float(grip)]]).astype(np.float32))
        self.cols["ee_w"].append(np.concatenate([rp, rq, ep_w, eq_w]).astype(np.float32))
        self.cols["phase"].append(PHASES.index(phase) if phase in PHASES else 0)
        self.cols["plan"].append(plan_ahead(plan_w, rp[0], rp[1], yaw))
        self.cols["world"].append(np.array([rp[0], rp[1], yaw], np.float32))
        self.cols["t"].append(k * self.env.step_dt)
        for name, cam in (("front", self.front), ("wrist", self.wrist)):
            ci = self.cams[name]
            if "hfov_deg" not in ci:  # from the rendered camera's own intrinsics
                K = cam.data.intrinsic_matrices[0].cpu().numpy()
                w, h = ci["render_wh"]
                ci["hfov_deg"] = round(math.degrees(2 * math.atan(w / 2 / K[0, 0])), 1)
                ci["vfov_deg"] = round(math.degrees(2 * math.atan(h / 2 / K[1, 1])), 1)
            self.frames[name].append(self._img(cam, ci["size_wh"]))
            self.cols[f"depth_{name}"].append(self._depth(cam, ci["depth_wh"], ci["min_depth_m"]))

    def end(self, outcome):
        """Write the episode. The EE target of frame i is the EE pose at frame i+1, expressed in the base
        frame of frame i (the last frame repeats its own pose)."""
        if not self.on:
            return None
        self.on = False
        c = self.cols
        n = len(c["t"])
        if n < 2:
            return None
        ee = np.stack(c["ee_w"])  # root pos 3, root quat 4, ee pos 3, ee quat 4 (world)
        ee_target = np.stack([to_base(ee[i, 0:3], ee[i, 3:7], ee[min(i + 1, n - 1), 7:10], ee[min(i + 1, n - 1), 10:14])
                              for i in range(n)]).astype(np.float32)
        out = os.path.join(self.root, f"ep_{self.n:06d}")
        os.makedirs(out, exist_ok=True)
        np.savez_compressed(
            os.path.join(out, "data.npz"), format=FORMAT, t=np.asarray(c["t"], np.float32),
            joint_pos=np.stack(c["joint_pos"]), joint_vel=np.stack(c["joint_vel"]), base=np.stack(c["base"]),
            ee_pose=np.stack(c["ee_pose"]), depth_front=np.stack(c["depth_front"]), depth_wrist=np.stack(c["depth_wrist"]),
            action_base=np.stack(c["action_base"]), action_arm=np.stack(c["action_arm"]), action_ee=ee_target,
            phase=np.asarray(c["phase"], np.int8), plan=np.stack(c["plan"]), world=np.stack(c["world"]))
        import imageio.v2 as imageio
        for cam, fr in self.frames.items():
            imageio.mimsave(os.path.join(out, f"{cam}.mp4"), fr, fps=self.fps, codec="libx264", quality=5,
                            macro_block_size=2)  # 8 would pad the D405's 180 rows to 184
        meta = {**self.meta, **outcome, "format": FORMAT, "frames": n, "fps": self.fps, "phases": PHASES,
                "joint_names": JOINT_NAMES, "base_names": BASE_NAMES, "ee_names": EE_NAMES,
                "action_names": ACTION_NAMES, "depth": {"size_wh": DEPTH, "unit": "mm", "max_m": DEPTH_MAX},
                "cameras": {k: {kk: vv for kk, vv in v.items() if kk != "render_wh"} for k, v in self.cams.items()},
                "plan": {"points": PLAN_N, "step_m": PLAN_STEP, "frame": "base (x forward, y left)"}}
        with open(os.path.join(out, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        for fn in [out] + [os.path.join(out, x) for x in os.listdir(out)]:  # the container runs as uid 1234:
            os.chmod(fn, 0o777 if os.path.isdir(fn) else 0o666)          # let the host user annotate/clean up
        self.n += 1
        return out


BASE_NAMES = [f"base_lin_vel_{a}" for a in "xyz"] + [f"base_ang_vel_{a}" for a in "xyz"] + \
    [f"gravity_{a}" for a in "xyz"] + ["base_height"]
EE_NAMES = [f"ee_pos_{a}" for a in "xyz"] + [f"ee_quat_{a}" for a in "wxyz"]
ACTION_NAMES = {"base": ["vx", "vy", "wz"], "arm": [f"joint{k}" for k in range(1, 7)] + ["gripper"], "ee": EE_NAMES}
