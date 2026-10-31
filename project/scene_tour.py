"""Drive the B2W once round a reconstructed scene and film it: third-person chase camera + front camera.

--robot b2w (default): the bare B2W with ETH's original policy and a camera at the front of the body.
--robot b2w_z1: the B2W + Z1 + gripper of the fetch recordings (task ...Fetch...Gripper, its table and box left out),
arm stowed, the stance fine-tune of ETH's policy (eth_loco.FINETUNED, trained on this robot) and the head
camera of the recordings (RealSense D455 lens, level; vln_nav.CAM_POS); 0.6 m/s.

The scene is a splat + collider stage (e.g. record3d_2laps M1) used as the ground of the B2W velocity task
("usd" terrain). ETH's locomotion policy (eth_loco.py) walks; a pure-pursuit controller follows a closed route
made by reconstruction_with_mesh/tools/robot_route.py (corridor centre lines from the collider mesh). The chase camera rides on
the route 2.4 m behind the robot, so it stays inside the corridors on turns.

    ./isaac/run_script.sh project/scene_tour.py --stage /workspace/scenes/.../isaac/M1_robot_ground.usda \
        --route /workspace/scenes/.../robot/route.json --out /workspace/renders/m1_tour
    (ISAAC_IMAGE=isaac-lab:2.3.2)

Writes <out>/side_by_side.mp4 (chase | front, with a minimap), chase.mp4, front.mp4, track.json (pose, command,
cross-track error and body contact force per step) and a few stills. Prints TOUR| lines.

--layout three (b2w_z1): the chase view fills the left half, full height; the right half is the head camera on top and
the end-effector camera (the D405 on the wrist, the asset's end_effector_camera) below it; also writes wrist.mp4.
--view_width/--view_height are then one right-hand view (default 960 x 540, so the video is 1920 x 1080).

The NuRec render settings (tone mapping) of the splat stage are copied into the renderer: the task references the
stage, and Isaac applies render settings only from the opened stage's root layer, so the splat would look washed out.
"""

import argparse
import json
import math
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--stage", required=True, help="ground USD (container path)")
parser.add_argument("--route", required=True, help="route.json from robot_route.py (container path)")
parser.add_argument("--out", required=True)
parser.add_argument("--robot", choices=("b2w", "b2w_z1"), default="b2w")
parser.add_argument("--task", default=None, help="default: by --robot")
parser.add_argument("--policy", default=None,
                    help="default: ETH's original for b2w (it was made for the bare B2W), the stance fine-tune for b2w_z1")
parser.add_argument("--vmax", type=float, default=None, help="m/s on straights (default 0.8 b2w, 0.6 b2w_z1)")
parser.add_argument("--alat", type=float, default=0.35, help="m/s^2 lateral acceleration allowed in curves")
parser.add_argument("--laps", type=float, default=1.0)
parser.add_argument("--max_time", type=float, default=240.0)
parser.add_argument("--fps", type=int, default=25)
parser.add_argument("--view_width", type=int, default=None, help="per view (default 960 x 720 b2w; 1024 x 640 = the D455's 16:10 for b2w_z1)")
parser.add_argument("--view_height", type=int, default=None)  # not --width/--height: AppLauncher takes those
parser.add_argument("--layout", choices=("pair", "three"), default="pair",
                    help="pair: chase | front; three (b2w_z1): chase | front over end-effector camera")
parser.add_argument("--chase_back", type=float, default=2.4, help="m behind the robot, along the route")
parser.add_argument("--chase_height", type=float, default=1.5)
parser.add_argument("--chase_hfov", type=float, default=None, help="deg (default 75 b2w, 80 b2w_z1, 65 for the tall view of --layout three)")
parser.add_argument("--front_pitch", type=float, default=None, help="deg down (default 8 b2w; 0 b2w_z1 = level, as recorded)")
parser.add_argument("--look_z", type=float, default=None, help="chase camera aim height (default 0.4 b2w, 0.5 b2w_z1)")
parser.add_argument("--quick", type=float, default=0.0, help="test: stop after this many seconds of driving")
parser.add_argument("--arm_partners", action="store_true", help="b2w_z1 diagnostic: which robot parts the arm's contacts are with")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
Z1 = args.robot == "b2w_z1"
THREE = args.layout == "three"
assert Z1 or not THREE, "--layout three needs the end-effector camera of --robot b2w_z1"
if THREE:
    args.view_width, args.view_height = args.view_width or 960, args.view_height or 540
    args.chase_hfov = args.chase_hfov or 65.0
for k, v_b2w, v_z1 in (("task", "RobotLab-Isaac-Velocity-Warehouse-Unitree-B2W-v0", "RobotLab-Isaac-Fetch-Warehouse-Unitree-B2W-Z1-Gripper-v0"),
                       ("policy", "/workspace/eth/policy_force_new.onnx", "/workspace/eth/policy_stance_v5_600.pt"),
                       ("vmax", 0.8, 0.6), ("view_width", 960, 1024), ("view_height", 720, 640), ("front_pitch", 8.0, 0.0),
                       ("look_z", 0.4, 0.5), ("chase_hfov", 75.0, 80.0)):
    if getattr(args, k) is None:
        setattr(args, k, v_z1 if Z1 else v_b2w)
args.enable_cameras = True
args.headless = True
app = AppLauncher(args).app

import omni.kit.app  # noqa: E402

omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("omni.usd.schema.omni_nurec_types", True)

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.envs import mdp as base_mdp  # noqa: E402
from isaaclab.managers import EventTermCfg as EventTerm  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.sensors import CameraCfg, ContactSensorCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import robot_lab.tasks  # noqa: E402,F401
import eth_loco  # noqa: E402
import vln_nav  # noqa: E402  (the head camera's lens and mount)
from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.unitree_b2w.arm_cargo_env_cfg import ARM_STOW  # noqa: E402
from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.unitree_b2w.fetch_env_cfg import GRIPPER_CLOSED  # noqa: E402


def say(*a):
    print("TOUR|", *a, flush=True)


os.makedirs(args.out, exist_ok=True)
R = json.load(open(args.route))
P = np.asarray(R["points"], float)
assert R["closed"], "the tour expects a closed route"
seg = np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1)
S = np.r_[0.0, np.cumsum(seg)[:-1]]
LOOP = float(seg.sum())
N = len(P)
CL = np.asarray(R["clearance"], float)                           # clearance of each route point (m)
_g = np.load(os.path.splitext(args.route)[0] + ".grid.npz")
GRID_CAM, GRID_X0, GRID_Y0, GRID_RES = _g["clear_cam"], float(_g["x0"]), float(_g["y0"]), float(_g["res"])
# path curvature (1/m) from the heading change over +-0.3 m
hd = np.unwrap(np.arctan2(*(np.roll(P, -1, axis=0) - P).T[::-1]))
k6 = int(round(0.3 / np.median(seg)))
KAPPA = np.abs((np.roll(hd, -k6) - np.roll(hd, k6) + np.pi) % (2 * np.pi) - np.pi) / (2 * k6 * np.median(seg))


def at(s):
    """Point on the closed route at arc length s (any real s)."""
    s = s % LOOP
    i = int(np.searchsorted(S, s, side="right") - 1)
    t = (s - S[i]) / seg[i]
    return P[i] + t * (P[(i + 1) % N] - P[i])


# ---------------------------------------------------------------- env
cfg = parse_env_cfg(args.task, device=args.device or "cpu", num_envs=1)
cfg.scene.terrain.usd_path = args.stage
cfg.scene.sky_light = None  # the stage has its own dome light
for name in list(vars(cfg.terminations)):
    if not name.startswith("_"):
        setattr(cfg.terminations, name, None)
cfg.episode_length_s = 1e5
for ev in ("randomize_rigid_body_mass_base", "randomize_rigid_body_mass_others", "randomize_com_positions",
           "randomize_apply_external_force_torque", "randomize_actuator_gains", "randomize_push_robot",
           "randomize_reset_joints"):
    if hasattr(cfg.events, ev):
        setattr(cfg.events, ev, None)
if getattr(cfg.events, "randomize_rigid_body_material", None) is not None:
    cfg.events.randomize_rigid_body_material.params.update(static_friction_range=(1.0, 1.0), dynamic_friction_range=(1.0, 1.0),
                                                           restitution_range=(0.0, 0.0), num_buckets=1)
if Z1:
    # the fetch task's warehouse props stay out of the apartment
    for name in ("table_top", "table_leg0", "table_leg1", "table_leg2", "table_leg3", "box", "hip_contacts"):
        setattr(cfg.scene, name, None)
    cfg.events.reset_box = None
    # contact_forces (Robot/.*) only sees the robot's direct children; the arm's links (those with colliders) sit under z1/
    ARM_LINKS = ("link00", "link02", "link03", "link04", "link06", "gripperStator", "gripperMover")
    PARTNERS = [f"{{ENV_REGEX_NS}}/Robot/{n}" for n in ("base_link", "lidar_link", "head_Link")] + \
               [f"{{ENV_REGEX_NS}}/Robot/z1/{n}" for n in ARM_LINKS]
    cfg.scene.arm_contacts = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/z1/(" + "|".join(ARM_LINKS) + ")",
                                              filter_prim_paths_expr=PARTNERS if args.arm_partners else [])
    # the fetch task drops the robot material event (the jaws get their own); the wheels get the bare run's 1.0
    cfg.events.tour_wheel_friction = EventTerm(
        func=base_mdp.randomize_rigid_body_material, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_foot"]), "static_friction_range": (1.0, 1.0),
                "dynamic_friction_range": (1.0, 1.0), "restitution_range": (0.0, 0.0), "num_buckets": 1})
still = (0.0, 0.0)
cfg.events.randomize_reset_base.params = {
    "pose_range": {k: still for k in ("x", "y", "z", "roll", "pitch", "yaw")},
    "velocity_range": {k: still for k in ("x", "y", "z", "roll", "pitch", "yaw")},
}
cfg.commands.base_velocity.debug_vis = False
eth_loco.patch_cfg(cfg)
sx, sy, syaw = R["start"]
cfg.scene.robot = cfg.scene.robot.replace(init_state=cfg.scene.robot.init_state.replace(
    pos=(sx, sy, eth_loco.SPAWN_Z), rot=(math.cos(syaw / 2), 0.0, 0.0, math.sin(syaw / 2))))
cfg.sim.render_interval = cfg.decimation * max(1, round(1.0 / (args.fps * cfg.sim.dt * cfg.decimation)))
cfg.sim.render.rendering_mode = "quality"
W, H = args.view_width, args.view_height
CH = 2 * H if THREE else H                                       # the chase view: full video height in --layout three
ap_h = 20.955
fx_chase = W / 2 / math.tan(math.radians(args.chase_hfov) / 2)
pitch = math.radians(args.front_pitch)
if Z1:  # the recordings' head camera: the D455 colour lens (90 x 65 deg), level, at the head
    FRONT_LABEL = "front camera (D455 on the head)" if THREE else "head camera (D455, as in the recordings)"
    front_lens = sim_utils.PinholeCameraCfg(focal_length=vln_nav.D455_FOCAL, horizontal_aperture=vln_nav.D455_APERTURE_H,
                                            clipping_range=(0.05, 60.0))
    front_pos = vln_nav.CAM_POS
else:
    FRONT_LABEL = "front camera (on the robot)"
    fx_front = W / 2 / math.tan(math.radians(87.0) / 2)        # D455-like 87 deg horizontal
    front_lens = sim_utils.PinholeCameraCfg(focal_length=fx_front * ap_h / W, horizontal_aperture=ap_h, clipping_range=(0.05, 60.0))
    # the real B2W's front camera sits at (0.42, 0.025, 0.062) on base_link (URDF joint_f_dc)
    front_pos = (0.43, 0.0, 0.07)
cfg.scene.front_cam = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base_link/tour_front_cam", width=W, height=H, data_types=["rgb"], update_period=0.0,
    spawn=front_lens,  # "world" convention = x forward, z up: pitching down is + about y
    offset=CameraCfg.OffsetCfg(pos=front_pos, rot=(math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0), convention="world"))
cfg.scene.chase_cam = CameraCfg(
    prim_path="{ENV_REGEX_NS}/TourChaseCam", width=W, height=CH, data_types=["rgb"], update_period=0.0,
    spawn=sim_utils.PinholeCameraCfg(focal_length=fx_chase * ap_h / W, horizontal_aperture=ap_h, clipping_range=(0.05, 60.0)))
if THREE:  # the asset's own end-effector camera (the D405 in its holder on link06, 87 x 56 deg = 16:9), as recorded
    cfg.scene.wrist_cam = CameraCfg(prim_path="{ENV_REGEX_NS}/Robot/z1/link06/end_effector_camera", spawn=None,
                                    width=W, height=H, data_types=["rgb"], update_period=0.0)

env = gym.make(args.task, cfg=cfg)
u = env.unwrapped
dt = u.step_dt
robot = u.scene["robot"]
front, chase = u.scene["front_cam"], u.scene["chase_cam"]
wrist = u.scene["wrist_cam"] if THREE else None
# the splat's NuRec render settings (tone mapping that passes its colours through as captured) live in its root layer,
# which Isaac only reads from the opened stage; the task references the stage, so copy them into the renderer
import carb.settings  # noqa: E402
from pxr import Usd  # noqa: E402

_rs, _scene = {}, Usd.Stage.Open(args.stage)                  # keep the stage: its layers die with it
for _layer in _scene.GetUsedLayers():                           # the scene's layers only (the robot's assets carry some too)
    _rs.update(_layer.customLayerData.get("renderSettings", {}))
del _scene
for _k, _v in _rs.items():
    carb.settings.get_settings().set("/" + _k.replace(":", "/"), _v)
say(f"render settings from the stage: {_rs}")
contact = u.scene["contact_forces"]
body_ids = [i for i, n in enumerate(contact.body_names) if not n.endswith("_foot")]   # anything but the wheels
arm = u.scene["arm_contacts"] if Z1 else None
render_every = cfg.sim.render_interval // cfg.decimation
say(f"dt {dt * 1000:.0f} ms, render every {render_every} steps ({1 / (dt * render_every):.0f} fps); route {LOOP:.1f} m, "
    f"{N} points; contact bodies {[contact.body_names[i] for i in body_ids] + (arm.body_names if arm else [])}")
driver = eth_loco.Driver(u, onnx_path=args.policy)
# arm and gripper (b2w_z1): absolute joint targets, held at the stow every step (unset, they would go to 0)
HOLD = []
if Z1:
    _sl, _st = {}, 0
    for _n, _w in zip(u.action_manager.active_terms, u.action_manager.action_term_dim):
        _sl[_n], _st = slice(_st, _st + _w), _st + _w
    HOLD = [(_sl["arm_action"], torch.tensor([ARM_STOW[j] for j in u.action_manager.get_term("arm_action")._joint_names])),
            (_sl["gripper_action"], torch.tensor([GRIPPER_CLOSED]))]

# ---------------------------------------------------------------- video
writers = {k: imageio.get_writer(os.path.join(args.out, f"{k}.mp4"), fps=args.fps, codec="libx264", quality=8,
                                 macro_block_size=8, ffmpeg_log_level="error")
           for k in ("chase", "front", "side_by_side") + (("wrist",) if THREE else ())}
try:  # the Isaac image has no system fonts; Kit ships a few
    FONT = ImageFont.truetype("/isaac-sim/kit/resources/fonts/OpenSans-SemiBold.ttf", 24)
except OSError:
    FONT = ImageFont.load_default()
MAP = 230                                                        # minimap size (px)
lo, hi = P.min(0) - 1.0, P.max(0) + 1.0
msc = (MAP - 20) / float(max(hi - lo))
to_map = lambda xy: (10 + (xy[0] - lo[0]) * msc, MAP - 10 - (xy[1] - lo[1]) * msc)
route_px = [to_map(p) for p in P] + [to_map(P[0])]


def label(img, lines, font):
    d = ImageDraw.Draw(img)
    y = 8
    for ln in lines:
        d.text((10, y), ln, font=font, fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0))
        y += font.size + 6


def compose(ch, fr, info, wr=None):
    a, b = Image.fromarray(ch), Image.fromarray(fr)
    label(a, ["third-person view"] + info, FONT)
    label(b, [FRONT_LABEL], FONT)
    if wr is not None:  # --layout three: front camera over the end-effector camera, right of the chase view
        c = Image.fromarray(wr)
        label(c, ["end-effector camera (D405 on the wrist)"], FONT)
        b = Image.fromarray(np.vstack([np.asarray(b), np.asarray(c)]))
    m = Image.new("RGBA", (MAP, MAP), (30, 30, 30, 200))
    dm = ImageDraw.Draw(m)
    dm.line(route_px, fill=(255, 80, 80, 255), width=2)
    x, y, yaw = pose()
    cx, cy = to_map((x, y))
    dm.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=(80, 170, 255, 255))
    dm.line([cx, cy, cx + 14 * math.cos(yaw), cy - 14 * math.sin(yaw)], fill=(80, 170, 255, 255), width=3)
    a = a.convert("RGBA")
    a.alpha_composite(m, (W - MAP - 10, CH - MAP - 10))
    return np.hstack([np.asarray(a.convert("RGB")), np.asarray(b)])


def pose():
    p = robot.data.root_pos_w[0].cpu().numpy()
    return float(p[0]), float(p[1]), float(robot.data.heading_w[0])


# ---------------------------------------------------------------- drive
s_prog = 0.0           # unwrapped arc length reached on the route
idx = 0
cmd_v = 0.0
eye_f = None
track = []
frames = 0
WALL0 = time.time()
t = 0.0
step_k = 0


def nearest(x, y):
    """Closest route point near the current progress (window -0.5 m .. +3 m): progress never jumps a corridor."""
    global idx
    w = np.arange(idx - 10, idx + 60) % N
    d = np.hypot(P[w, 0] - x, P[w, 1] - y)
    j = int(np.argmin(d))
    return int(w[j]), float(d[j])


def control(x, y, yaw, slow=False):
    global idx, s_prog, cmd_v
    j, e = nearest(x, y)
    ds = (S[j] - S[idx] + LOOP / 2) % LOOP - LOOP / 2
    s_prog += ds
    idx = j
    # signed cross-track error (+ = left of the route)
    tng = P[(j + 1) % N] - P[j]
    tng = tng / (np.linalg.norm(tng) + 1e-9)
    e_signed = float(np.cross(tng, np.array([x, y]) - P[j]))
    step_m = float(np.median(seg))
    ahead = np.arange(j, j + int(2.0 / step_m)) % N               # curvature in the next 2 m
    v_cap = min(args.vmax, math.sqrt(args.alat / max(float(KAPPA[ahead].max()), 1e-3)))
    # narrow passages (doorways): slow down from 1.5 m before until the body (~0.6 m behind the base) is through
    near = np.arange(j - int(0.6 / step_m), j + int(1.5 / step_m)) % N
    v_cap = min(v_cap, float(np.interp(CL[near].min(), [0.35, 0.42, 0.55, 0.8], [0.18, 0.25, 0.45, args.vmax])))
    if slow:
        v_cap = min(v_cap, 0.2)
    remaining = args.laps * LOOP - s_prog
    v_cap = min(v_cap, max(0.0, 0.6 * remaining))               # roll to a stop at the end
    Ld = float(np.clip(0.35 + 0.6 * cmd_v, 0.35, 1.0))
    tx, ty = at(S[j] + Ld)
    c, s = math.cos(yaw), math.sin(yaw)
    lx, ly = c * (tx - x) + s * (ty - y), -s * (tx - x) + c * (ty - y)
    alpha = math.atan2(ly, lx)
    if abs(alpha) > 0.7:                                          # facing away: turn first
        v_cap = min(v_cap, 0.1)
    acc = 0.5 * dt                                                # m/s per step: smooth for the camera
    cmd_v = float(np.clip(v_cap, cmd_v - 2 * acc, cmd_v + acc))
    wz = float(np.clip(2.0 * max(cmd_v, 0.25) * math.sin(alpha) / Ld, -1.0, 1.0))
    # sideways correction of the cross-track error (ETH's policy follows small vy well): the
    # wheel-legged body is ~0.6 m wide and some doorways leave it only ~5 cm a side
    vy = float(np.clip(-1.5 * e_signed, -0.2, 0.2))
    return (cmd_v, vy, wz), e_signed, remaining


def grid_at(G, q):
    r = np.clip(((q[..., 1] - GRID_Y0) / GRID_RES).astype(int), 0, G.shape[0] - 1)
    c = np.clip(((q[..., 0] - GRID_X0) / GRID_RES).astype(int), 0, G.shape[1] - 1)
    return G[r, c]


def los_min(a, b, skip_end=0.35):
    """Smallest camera-height clearance along the segment a -> b, leaving out the last skip_end m (the robot)."""
    L = float(np.hypot(*(b - a)))
    f = np.linspace(0, max(0.0, 1 - skip_end / max(L, 1e-6)), max(2, int(L / GRID_RES * 2)))
    return float(grid_at(GRID_CAM, a[None] + f[:, None] * (b - a)[None]).min())


def chase_view(x, y, yaw):
    """Eye on the route behind the robot (1.2 .. --chase_back m): the candidate with a clear line of sight that
    is farthest back and farthest from walls and furniture (a camera next to a cabinet or door frame sees a
    blur of close-up splats); smoothed."""
    global eye_f
    me = np.array([x, y])
    best, best_score = None, -1e9
    for d in np.arange(args.chase_back, 1.19, -0.15):
        e = at(S[idx] - d)
        if los_min(e, me) < 0.15:
            continue
        score = min(float(grid_at(GRID_CAM, e)), 0.7) + 0.25 * d
        if score > best_score:
            best, best_score = e, score
    e = best if best is not None else at(S[idx] - 1.2)
    eye = np.array([e[0], e[1], args.chase_height])
    look = np.array([x + 0.3 * math.cos(yaw), y + 0.3 * math.sin(yaw), args.look_z])   # the robot itself stays centred on turns
    eye_f = eye if eye_f is None else eye_f + 0.05 * (eye - eye_f)
    chase.set_world_poses_from_view(torch.tensor(eye_f[None], dtype=torch.float32, device=u.device),
                                    torch.tensor(look[None], dtype=torch.float32, device=u.device))


def step(cmd, info_extra=""):
    global frames, t, step_k
    with torch.inference_mode():
        a = torch.zeros(1, u.action_manager.total_action_dim, device=u.device)
        driver.fill(a, list(cmd))
        for sl, val in HOLD:
            a[0, sl] = val.to(a.device)
        x, y, yaw = pose()
        chase_view(x, y, yaw)
        env.step(a)
    t += dt
    step_k += 1
    if step_k % render_every == 0:
        ch = chase.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        fr = front.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        wr = wrist.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8) if wrist is not None else None
        v = float(torch.norm(robot.data.root_lin_vel_b[0, :2]))
        info = [f"t {t:5.1f} s   speed {v:.2f} m/s", f"loop {100 * min(s_prog / (args.laps * LOOP), 1):3.0f} %  "
                f"({s_prog:.1f} / {args.laps * LOOP:.0f} m){info_extra}"]
        writers["chase"].append_data(ch)
        writers["front"].append_data(fr)
        if wr is not None:
            writers["wrist"].append_data(wr)
        sbs = compose(ch, fr, info, wr)
        writers["side_by_side"].append_data(sbs)
        if frames % 250 == 0:
            say(f"frame {frames}: {(time.time() - WALL0) / max(frames, 1):.2f} s wall per frame")
        if frames % 250 == 0 or frames == 75:
            Image.fromarray(sbs).save(os.path.join(args.out, f"still_{frames:05d}.jpg"), quality=88)
        frames += 1


with torch.inference_mode():
    env.reset()
driver.reset()
x, y, yaw = pose()
say(f"spawned at ({x:.2f}, {y:.2f}) yaw {math.degrees(yaw):.0f} deg, base z {float(robot.data.root_pos_w[0, 2]):.2f}")
for _ in range(int(2.0 / dt)):                                    # land and settle
    step((0.0, 0.0, 0.0))
say(f"settled: base z {float(robot.data.root_pos_w[0, 2]):.3f}, tilt {eth_loco.tilt_deg(robot):.1f} deg")
_rel = (robot.data.body_pos_w[0] - robot.data.root_pos_w[0]).cpu().numpy()
_lat = -math.sin(yaw) * _rel[:, 0] + math.cos(yaw) * _rel[:, 1]
say("link frames, lateral offset from the base (m): " + ", ".join(f"{n} {v:+.2f}" for n, v in zip(robot.body_names, _lat)
                                                                 if n.endswith(("_hip", "_foot"))))
_fwd = math.cos(yaw) * _rel[:, 0] + math.sin(yaw) * _rel[:, 1]
_z = robot.data.body_pos_w[0, :, 2].cpu().numpy()
say("link frames (forward, lateral, height above the floor, m): " + ", ".join(
    f"{n} ({f:+.2f}, {v:+.2f}, {h:.2f})" for n, f, v, h in zip(robot.body_names, _fwd, _lat, _z)))
f_max, e_max, stuck_t, status = 0.0, 0.0, 0.0, "DONE"
t_drive0 = t
recoveries, slow_until, contacts = 0, -1.0, {}
while True:
    x, y, yaw = pose()
    cmd, e, remaining = control(x, y, yaw, slow=s_prog < slow_until)
    step(cmd)
    fb = torch.norm(contact.data.net_forces_w[0, body_ids], dim=-1)
    names = [contact.body_names[i] for i in body_ids]
    if arm is not None:
        fb = torch.cat([fb, torch.norm(arm.data.net_forces_w[0], dim=-1)])
        names = names + arm.body_names
    f = float(fb.max())
    if arm is not None and args.arm_partners:
        fm = torch.norm(arm.data.force_matrix_w[0], dim=-1)          # (arm link, partner)
        for i, j in zip(*torch.nonzero(fm > 1.0, as_tuple=True)):
            k = f"{arm.body_names[int(i)]}<->{PARTNERS[int(j)].split('/')[-1]}"
            contacts[k] = contacts.get(k, 0) + 1
        # what the arm touches outside these robot parts (scene): net force minus the robot-part partners
        rest = torch.norm(arm.data.net_forces_w[0] - arm.data.force_matrix_w[0].sum(dim=1), dim=-1)
        if float(rest.max()) > 1.0:
            k = f"{arm.body_names[int(rest.argmax())]}<->other"
            contacts[k] = contacts.get(k, 0) + 1
    if f > 1.0:
        b = names[int(fb.argmax())]
        contacts[b] = contacts.get(b, 0) + 1
    f_max, e_max = max(f_max, f), max(e_max, abs(e))
    v = float(torch.norm(robot.data.root_lin_vel_b[0, :2]))
    track.append([round(t, 3), round(x, 3), round(y, 3), round(yaw, 3), round(cmd[0], 3), round(cmd[2], 3), round(v, 3),
                  round(e, 3), round(f, 1), round(float(robot.data.root_pos_w[0, 2]), 3), round(eth_loco.tilt_deg(robot), 1),
                  round(cmd[1], 3)])
    if step_k % int(5.0 / dt) == 0:
        say(f"t {t:6.1f}  pos ({x:6.2f}, {y:6.2f})  prog {s_prog:5.1f}/{args.laps * LOOP:.1f} m  v {v:.2f}  "
            f"xtrack {e:+.2f}  contact {f:.0f} N  tilt {eth_loco.tilt_deg(robot):.1f}")
    stuck_t = stuck_t + dt if (cmd[0] > 0.1 and v < 0.04) else 0.0
    if eth_loco.tilt_deg(robot) > 45:
        status = "FELL"
        break
    if stuck_t > 2.5:
        if recoveries >= 3:
            status = "STUCK"
            break
        recoveries += 1
        say(f"stuck at ({x:.2f}, {y:.2f}), prog {s_prog:.1f} m, contacts {contacts}: backing up (recovery {recoveries})")
        for _ in range(int(1.5 / dt)):                             # back up along the route, re-centring
            x, y, yaw = pose()
            _, e, _ = control(x, y, yaw, slow=True)
            step((-0.2, float(np.clip(-1.5 * e, -0.2, 0.2)), 0.0))
        cmd_v, stuck_t, slow_until = 0.0, 0.0, s_prog + 2.5
        continue
    if remaining < 0.05 and v < 0.05:
        break
    if t > args.max_time or (args.quick and t - t_drive0 > args.quick):
        status = "TIMEOUT" if not args.quick else "QUICK"
        break
for _ in range(int(1.5 / dt)):                                    # hold the last view
    step((0.0, 0.0, 0.0))
for w in writers.values():
    w.close()
tr = np.asarray(track)
summary = {"status": status, "robot": args.robot, "policy": args.policy, "vmax": args.vmax, "route_m": LOOP, "driven_m": s_prog, "drive_time_s": t - t_drive0, "frames": frames,
           "fps": args.fps, "max_cross_track_m": e_max, "p95_cross_track_m": float(np.percentile(np.abs(tr[:, 7]), 95)),
           "max_body_contact_N": f_max, "steps_with_body_contact": int((tr[:, 8] > 1.0).sum()),
           "max_tilt_deg": float(tr[:, 10].max()), "mean_speed": float(tr[:, 6].mean()),
           "recoveries": recoveries, "contact_steps_by_body": contacts,
           "columns": ["t", "x", "y", "yaw", "cmd_vx", "cmd_wz", "speed", "xtrack", "body_contact_N", "base_z", "tilt_deg",
                       "cmd_vy"],
           "track": track}
json.dump(summary, open(os.path.join(args.out, "track.json"), "w"))
say(f"{status}: {s_prog:.1f} of {LOOP:.1f} m in {t - t_drive0:.1f} s, {frames} frames; cross-track max {e_max:.2f} m, "
    f"p95 {summary['p95_cross_track_m']:.2f} m; body contact max {f_max:.0f} N ({summary['steps_with_body_contact']} steps); "
    f"tilt max {summary['max_tilt_deg']:.1f} deg; recoveries {recoveries}; contacts by body {contacts}")
env.close()
app.close()
