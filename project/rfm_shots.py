"""Stills of an RFM whole-body policy reaching a scripted sequence of 6D targets, headless (report Fig. 5).

The same policy, task and viewer settings as `live.sh goto_mine` (rfm_goto.py), but the targets are scripted and every
--every seconds a frame is rendered from a camera fixed for that target, so the arrival can be picked afterwards.

    ISAAC_IMAGE=isaac-lab:2.3.2 ./run_script.sh rfm_shots.py \\
        --task RobotLab-Isaac-RFM-Delta-Sphere-Unitree-B2W-Z1-v0 \\
        --checkpoint /workspace/policies/b2w_z1_rfm/2026-10-08_19-58-51_rfm19_delta_sphere/model_37600.pt \\
        --step "low:1.1,0.35,0.42:8:2.7,-1.9,1.3:0.7,0.15,0.35" --step "far:-1.8,1.8,0.95:11:1.2,4.0,1.7:-1.4,1.5,0.5"
Step spec  name : target x,y,z[,roll,pitch,yaw] : seconds : camera eye x,y,z : camera look-at x,y,z  (world, env 0)
Out: renders/report/rfm/<name>_<t>.png, <name>.json (hand error, base height and tilt per frame)
"""

import argparse
import json
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--step", action="append", required=True)
parser.add_argument("--every", type=float, default=0.5, help="seconds between saved frames")
parser.add_argument("--res", default="1600x1200")
parser.add_argument("--out", default="/workspace/renders/report/rfm")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402
from isaaclab.utils.math import quat_from_euler_xyz  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import robot_lab.tasks  # noqa: E402,F401

from rfm_helpers import restart_schedule  # noqa: E402


def say(*a):
    print("SHOTS|", *a, flush=True)


# --- the env exactly as rfm_goto.py sets it up (no terminations, no corruption, no shoves, still reset pose)
cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1)
cfg.seed = 42
cfg.episode_length_s = 1.0e9
for _name in list(vars(cfg.terminations)):
    if not _name.startswith("_"):
        setattr(cfg.terminations, _name, None)
cfg.observations.policy.enable_corruption = False
for _name in ("randomize_push_robot", "randomize_apply_external_force_torque"):
    if hasattr(cfg.events, _name):
        setattr(cfg.events, _name, None)
if getattr(cfg.commands, "base_velocity", None) is not None:
    cfg.commands.base_velocity.debug_vis = False
cfg.commands.ee_pose.resampling_time_range = (1.0e9, 1.0e9)
if hasattr(cfg.commands.ee_pose, "debug_vis"):
    cfg.commands.ee_pose.debug_vis = False  # the red sphere below marks the target
still = (0.0, 0.0)
cfg.events.randomize_reset_base.params = {
    "pose_range": {k: still for k in ("x", "y", "z", "roll", "pitch", "yaw")},
    "velocity_range": {k: still for k in ("x", "y", "z", "roll", "pitch", "yaw")},
}
cfg.viewer.origin_type = "world"
cfg.viewer.resolution = tuple(int(v) for v in args.res.split("x"))

agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
env = RslRlVecEnvWrapper(gym.make(args.task, cfg=cfg, render_mode="rgb_array"), clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)
u = env.unwrapped
robot = u.scene["robot"]
ee = u.command_manager.get_term("ee_pose")
dt = u.step_dt
MARK = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/RfmTarget", markers={
    "target": sim_utils.SphereCfg(radius=0.05, visual_material=sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.95, 0.1, 0.1), emissive_color=(0.4, 0.0, 0.0)))}))
MARK.set_visibility(False)

with torch.inference_mode():  # hold still where it stands (a reset samples a random target)
    ee.pose_command_w[:, :3] = robot.data.body_pos_w[:, ee.body_idx]
    ee.pose_command_w[:, 3:] = robot.data.body_quat_w[:, ee.body_idx]
restart_schedule(ee, robot)
obs = env.get_observations()
os.makedirs(args.out, exist_ok=True)
u.render()  # the first call only creates the viewport capture (and returns an empty image)


def grab():
    u.render()
    return u.render()[..., :3]


for spec in args.step:
    name, tgt, secs, eye, look = spec.split(":")
    t = [float(v) for v in tgt.split(",")] + [0.0, 0.0, 0.0]
    x, y, z, rr, pp, yy = t[:6]
    pos = torch.tensor([[x, y, z]], device=u.device) + u.scene.env_origins
    quat = quat_from_euler_xyz(torch.tensor([rr], device=u.device), torch.tensor([pp], device=u.device),
                               torch.tensor([yy], device=u.device))
    with torch.inference_mode():
        ee.pose_command_w[:, :3] = pos
        ee.pose_command_w[:, 3:] = quat
    restart_schedule(ee, robot)
    MARK.set_visibility(True)
    MARK.visualize(translations=pos)
    u.viewport_camera_controller.update_view_location([float(v) for v in eye.split(",")],
                                                      [float(v) for v in look.split(",")])
    log = []
    n = int(round(float(secs) / dt))
    every = max(1, int(round(args.every / dt)))
    say(f"{name}: target ({x:.2f}, {y:.2f}, {z:.2f}), {secs} s")
    for k in range(1, n + 1):
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))
        if k % every == 0 or k == n:
            d = robot.data
            perr = float(ee.metrics["position_error"].mean())
            h = float(d.root_pos_w[0, 2] - u.scene.env_origins[0, 2])
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(-d.projected_gravity_b[0, 2])))))
            tt = round(k * dt, 2)
            Image.fromarray(grab()).save(f"{args.out}/{name}_{tt:05.2f}.png")
            log.append({"t": tt, "hand_err_cm": round(perr * 100, 1), "base_h": round(h, 3), "tilt_deg": round(tilt, 1),
                        "base_xy": [round(float(v), 3) for v in d.root_pos_w[0, :2] - u.scene.env_origins[0, :2]]})
    json.dump(log, open(f"{args.out}/{name}.json", "w"), indent=1)
    best = min(log, key=lambda r: r["hand_err_cm"])
    say(f"{name}: final hand error {log[-1]['hand_err_cm']} cm, best {best['hand_err_cm']} cm at {best['t']} s, "
        f"base height {log[-1]['base_h']} m, tilt {log[-1]['tilt_deg']} deg")
env.close()
app.close()
