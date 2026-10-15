"""Demo video: a visible target in the room, the robot drives there and puts its hand on it,
crouching when the target is low (Phase 4's whole-body result, policy wb9).

Each target in --targets is its own trial from the start pose (wb9 tipped over turning
sharply with the arm up after a reach, so the robot is reset between targets): a red sphere
marks it; NAVIGATE drives to a standoff short of it (the waypoint control law of
demo_fetch.py, turning at most --max_turn); REACH switches the world-frame hand target on
and holds it. The overlay shows the phase, the hand-to-target distance and the body
height (the crouch). Two views are written: following the robot, and a fixed side view.

    ISAAC_IMAGE=isaac-lab:2.3.2 ./run_script.sh demo_reach_video.py \\
        --targets "2.5,0,0.3; 2.5,1.8,0.6; 0.8,2.6,0.35"
Output: renders/demo_reach/{follow,side}.mp4, side_by_side.mp4, summary.json
"""

import argparse
import json
import math
import os

from isaaclab.app import AppLauncher

LOGS = "/workspace/policies"  # trained checkpoints (policies/)
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="RobotLab-Isaac-WholeBody-WorldTarget-Unitree-B2W-Z1-v0")
parser.add_argument("--checkpoint", default=f"{LOGS}/b2w_z1_wholebody_worldtarget/2026-09-23_13-25-50_wb9/model_7700.pt")
parser.add_argument("--targets", default="2.5,0,0.3; 2.5,1.8,0.6; 0.8,2.6,0.35", help="x,y,z; ... (m, env-local)")
parser.add_argument("--standoff", type=float, default=0.45)
parser.add_argument("--vmax", type=float, default=0.3, help="wb9 was trained to +/-0.3 m/s")
parser.add_argument("--reach_time", type=float, default=5.0)
parser.add_argument("--timeout", type=float, default=40.0)
parser.add_argument("--max_turn", type=float, default=0.6, help="rad/s while driving")
parser.add_argument("--out", default="/workspace/renders/demo_reach")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless, args.enable_cameras = True, True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import robot_lab.tasks  # noqa: E402,F401


def say(*a):
    print("DEMO|", *a, flush=True)


TARGETS = [tuple(float(v) for v in t.split(",")) for t in args.targets.split(";") if t.strip()]

cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1)
cfg.seed = 42
cfg.episode_length_s = 1.0e4
cfg.terminations.time_out = None
cfg.observations.policy.enable_corruption = False
bv = cfg.commands.base_velocity
bv.heading_command, bv.rel_standing_envs, bv.rel_heading_envs = False, 0.0, 0.0
bv.resampling_time_range, bv.ranges.heading, bv.debug_vis = (1.0e6, 1.0e6), None, False
cfg.commands.ee_pose.resampling_time_range = (1.0e6, 1.0e6)
cfg.commands.ee_pose.debug_vis = False
still = (0.0, 0.0)
cfg.events.randomize_reset_base.params = {
    "pose_range": {k: still for k in ("x", "y", "z", "roll", "pitch", "yaw")},
    "velocity_range": {k: still for k in ("x", "y", "z", "roll", "pitch", "yaw")},
}
cfg.viewer.origin_type = "asset_root"
cfg.viewer.asset_name = "robot"
cfg.viewer.eye = (2.2, 2.2, 1.4)
cfg.viewer.lookat = (0.3, 0.0, 0.3)
cfg.viewer.resolution = (1280, 720)
# fixed side camera: in the scene from the start (a sensor made after sim start never initialises)
cfg.scene.side_cam = CameraCfg(prim_path="{ENV_REGEX_NS}/DemoSideCam", width=1280, height=720, data_types=["rgb"],
                               spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.1, 50.0)))

agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
env = RslRlVecEnvWrapper(gym.make(args.task, cfg=cfg, render_mode="rgb_array"), clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)
u = env.unwrapped
robot = u.scene["robot"]
vel = u.command_manager.get_term("base_velocity")
ee = u.command_manager.get_term("ee_pose")
hand_idx = ee.body_idx
dt = u.step_dt
origin = u.scene.env_origins[0]
IDENT = torch.tensor([1.0, 0.0, 0.0, 0.0], device=u.device)

# the targets: red spheres (the active one), grey for the others
markers = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/DemoTargets", markers={
    "active": sim_utils.SphereCfg(radius=0.05, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.1, 0.1),
                                                                                          emissive_color=(0.4, 0.0, 0.0))),
    "other": sim_utils.SphereCfg(radius=0.04, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.6, 0.6))),
    "done": sim_utils.SphereCfg(radius=0.04, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.2))),
}))
tw = torch.tensor([[origin[0] + x, origin[1] + y, origin[2] + z] for x, y, z in TARGETS], device=u.device)

# fixed side camera, looking across the area of the targets
xs, ys = [t[0] for t in TARGETS] + [0.0], [t[1] for t in TARGETS] + [0.0]
cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
side = u.scene["side_cam"]
eye = torch.tensor([[float(origin[0]) + cx + 0.5, float(origin[1]) + cy - 4.0, 1.3]], device=u.device)
look = torch.tensor([[float(origin[0]) + cx, float(origin[1]) + cy, 0.35]], device=u.device)

try:
    FONT = ImageFont.truetype("/isaac-sim/kit/resources/fonts/OpenSans-SemiBold.ttf", 30)
    SMALL = ImageFont.truetype("/isaac-sim/kit/resources/fonts/roboto_medium.ttf", 24)
except OSError:
    FONT = SMALL = ImageFont.load_default()

frames = {"follow": [], "side": []}
log = []


def overlay(img, lines):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 52 + 32 * (len(lines) - 1)], fill=(0, 0, 0))
    d.text((18, 8), lines[0], fill=(255, 255, 255), font=FONT)
    for i, s in enumerate(lines[1:]):
        d.text((18, 50 + 32 * i), s, fill=(255, 220, 90), font=SMALL)
    return np.asarray(im)


def show_markers(active):
    pos = tw.clone()
    idx = [0 if i == active else (2 if i < active else 1) for i in range(len(TARGETS))]
    markers.visualize(translations=pos, marker_indices=torch.tensor(idx, device=u.device))


def park_arm():
    d = robot.data
    fwd = torch.stack([torch.cos(d.heading_w), torch.sin(d.heading_w)], dim=1)
    ee.pose_command_w[:, 0:2] = d.root_pos_w[:, 0:2] + 0.40 * fwd
    ee.pose_command_w[:, 2] = origin[2] + 0.60
    ee.pose_command_w[:, 3:] = IDENT


def step(obs, k, title, info):
    with torch.inference_mode():
        obs, *_ = env.step(policy(obs))
    if k % 2 == 0:  # 25 fps
        lines = [title] + info
        frames["follow"].append(overlay(u.render(), lines))
        frames["side"].append(overlay(side.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8), lines))
    return obs


def status(i):
    d = robot.data
    hand = d.body_pos_w[0, hand_idx]
    err = float(torch.norm(hand - tw[i]))
    base_h = float(d.root_pos_w[0, 2] - origin[2])
    return err, base_h


obs = env.get_observations()
side.set_world_poses_from_view(eye, look)
k = 0
for _ in range(int(1.0 / dt)):  # settle, then a moment to show the targets
    show_markers(0)
    vel.vel_command_b[:] = 0.0
    park_arm()
    err, bh = status(0)
    obs = step(obs, k, "Whole-body reach (Phase 4 policy wb9)", [f"{len(TARGETS)} targets in the room (red = next)",
                                                                 f"body height {bh:.2f} m"])
    k += 1
for i, (x, y, z) in enumerate(TARGETS):
    show_markers(i)
    title = f"Trial {i + 1}/{len(TARGETS)}: target ({x:.1f}, {y:.1f}) m, height {z:.2f} m"
    if i > 0:  # each target is a trial from the start pose
        with torch.inference_mode():
            obs, _ = env.reset()
        for _ in range(int(1.5 / dt)):
            vel.vel_command_b[:] = 0.0
            park_arm()
            err, bh = status(i)
            obs = step(obs, k, title, ["new trial: robot back at the start", f"body height {bh:.2f} m"])
            k += 1
    goal = tw[i:i + 1, :2]
    t, arrived = 0.0, False
    while t < args.timeout and not arrived:  # 1. navigate
        d = robot.data
        e2 = goal - d.root_pos_w[:, :2]
        dist = torch.norm(e2, dim=1)
        c, s = torch.cos(d.heading_w), torch.sin(d.heading_w)
        ex, ey = c * e2[:, 0] + s * e2[:, 1], -s * e2[:, 0] + c * e2[:, 1]
        bearing = torch.atan2(ey, ex)
        speed = torch.clamp((dist - args.standoff) / 0.8, min=0.0, max=args.vmax)
        speed = torch.where(bearing.abs() > 0.6, torch.zeros_like(speed), speed)
        vel.vel_command_b[:, 0], vel.vel_command_b[:, 1] = speed, 0.0
        vel.vel_command_b[:, 2] = torch.clamp(2.0 * bearing, -args.max_turn, args.max_turn)
        park_arm()
        err, bh = status(i)
        obs = step(obs, k, title, ["phase: DRIVE to the target", f"distance {float(dist):.2f} m   body height {bh:.2f} m"])
        k += 1
        t += dt
        arrived = bool((dist <= args.standoff + 0.05).all())
    t_drive = t
    best, bh_min = 9.0, 9.0
    for _ in range(int(round(args.reach_time / dt))):  # 2. reach
        vel.vel_command_b[:] = 0.0
        ee.pose_command_w[:, :3] = tw[i:i + 1]
        ee.pose_command_w[:, 3:] = IDENT
        err, bh = status(i)
        best, bh_min = min(best, err), min(bh_min, bh)
        crouch = "  <- CROUCHING" if bh < 0.5 else ""
        obs = step(obs, k, title, ["phase: REACH (hand on the target)",
                                   f"hand to target {err * 100:.1f} cm   body height {bh:.2f} m{crouch}"])
        k += 1
    log.append({"target": [x, y, z], "drive_s": round(t_drive, 1), "arrived": arrived,
                "final_hand_err_cm": round(err * 100, 1), "best_hand_err_cm": round(best * 100, 1),
                "min_body_height_m": round(bh_min, 3)})
    say(f"target {i + 1}: {log[-1]}")
    for _ in range(int(1.0 / dt)):  # back to the carry pose before the next target
        vel.vel_command_b[:] = 0.0
        park_arm()
        err, bh = status(i)
        obs = step(obs, k, title, ["done: arm back", f"body height {bh:.2f} m"])
        k += 1
show_markers(len(TARGETS))

import imageio.v2 as imageio  # noqa: E402

os.makedirs(args.out, exist_ok=True)
for name, fr in frames.items():
    imageio.mimsave(os.path.join(args.out, f"{name}.mp4"), fr, fps=25, codec="libx264", quality=8, macro_block_size=8)
both = [np.concatenate([a[::2, ::2], b[::2, ::2]], axis=1) for a, b in zip(frames["follow"], frames["side"])]
imageio.mimsave(os.path.join(args.out, "side_by_side.mp4"), both, fps=25, codec="libx264", quality=8, macro_block_size=8)
with open(os.path.join(args.out, "summary.json"), "w") as f:
    json.dump({"task": args.task, "checkpoint": args.checkpoint, "targets": log}, f, indent=1)
say(f"wrote {args.out}/follow.mp4, side.mp4, side_by_side.mp4 ({len(frames['follow'])} frames)")
env.close()
app.close()
