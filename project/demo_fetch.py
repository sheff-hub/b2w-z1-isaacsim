"""End-to-end demo: name a point in the room, the robot drives there and reaches it.

Puts the two Phase 4 pieces together on one policy:

  1. NAVIGATE -- the waypoint follower's control law drives the robot to a
     standoff position short of the target;
  2. REACH    -- the world-frame hand target is switched on and the robot places
     its hand on the point, crouching if the point is low.

The target is a WORLD coordinate, which is the whole point: "(2.5, 0, 0.25)"
means a spot in the room, not a spot relative to the robot. Needs a world-target
policy (wb9); a body-frame policy cannot represent the goal at all.

    /isaac-sim/python.sh /workspace/project/demo_fetch.py --task <id> --checkpoint <path> \
        --target 2.5,0,0.25 [--real-time]
"""

import argparse
import json
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--target", default="2.5,0,0.25", help="x,y,z in metres, relative to the env origin")
parser.add_argument("--standoff", type=float, default=0.45, help="how far short of the target to stop, m")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--vmax", type=float, default=0.3, help="wb9 was trained to +/-0.3 m/s")
parser.add_argument("--tracking", type=float, default=0.8)
parser.add_argument("--reach_time", type=float, default=5.0)
parser.add_argument("--timeout", type=float, default=60.0)
parser.add_argument("--real-time", action="store_true")
parser.add_argument("--out", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import robot_lab.tasks  # noqa: E402,F401

from free_camera import CameraToggle  # noqa: E402


def say(*a):
    print("DEMO|", *a, flush=True)


TX, TY, TZ = (float(v) for v in args.target.split(","))

cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
cfg.seed = 42
cfg.episode_length_s = 1.0e4
cfg.terminations.time_out = None
cfg.observations.policy.enable_corruption = False
# No training disturbances in a viewer: the random shove (body velocity set to up
# to +-0.5 m/s every 10-15 s) and the random constant force/torque on the body
# (up to 30 N / 10 N m, held all episode). They make the policy robust in training;
# here they only add jolts and lean that are not the policy's own behaviour.
for _name in ("randomize_push_robot", "randomize_apply_external_force_torque"):
    if hasattr(cfg.events, _name):
        setattr(cfg.events, _name, None)
cfg.commands.base_velocity.heading_command = False
cfg.commands.base_velocity.rel_standing_envs = 0.0
cfg.commands.base_velocity.rel_heading_envs = 0.0
cfg.commands.base_velocity.resampling_time_range = (1.0e6, 1.0e6)
cfg.commands.base_velocity.ranges.heading = None
cfg.commands.base_velocity.debug_vis = False
cfg.commands.ee_pose.resampling_time_range = (1.0e6, 1.0e6)
still = (0.0, 0.0)
cfg.events.randomize_reset_base.params = {
    "pose_range": {k: still for k in ("x", "y", "z", "roll", "pitch", "yaw")},
    "velocity_range": {k: still for k in ("x", "y", "z", "roll", "pitch", "yaw")},
}
cfg.viewer.origin_type = "asset_root"
cfg.viewer.asset_name = "robot"
cfg.viewer.eye = (2.5, 2.5, 1.5)

agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
env = RslRlVecEnvWrapper(gym.make(args.task, cfg=cfg), clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)

u = env.unwrapped
robot = u.scene["robot"]
vel = u.command_manager.get_term("base_velocity")
ee = u.command_manager.get_term("ee_pose")
if type(ee).__name__ != "WorldPoseCommand":
    say("WARNING: this task's hand target is body-frame, so the world target below is "
        "not what the robot will actually chase.")
hand_idx = ee.body_idx
dt = u.step_dt
cam = CameraToggle(u, following=True)  # V releases the camera, C snaps it back
target_w = u.scene.env_origins + torch.tensor([TX, TY, TZ], device=u.device)
say(f"target (env-local): ({TX}, {TY}, {TZ}) m; standoff {args.standoff} m; vmax {args.vmax} m/s")

obs = env.get_observations()
IDENT = torch.tensor([1.0, 0.0, 0.0, 0.0], device=u.device)


def park_arm():
    """Hold the hand at a neutral spot that travels with the robot while driving."""
    d = robot.data
    fwd = torch.stack([torch.cos(d.heading_w), torch.sin(d.heading_w)], dim=1)
    ee.pose_command_w[:, 0:2] = d.root_pos_w[:, 0:2] + 0.40 * fwd
    ee.pose_command_w[:, 2] = u.scene.env_origins[:, 2] + 0.60
    ee.pose_command_w[:, 3:] = IDENT


# ---------------------------------------------------------------- 1. navigate
goal = target_w[:, :2].clone()
say(f"debug: env_origin {u.scene.env_origins[0].tolist()}, robot start "
    f"{robot.data.root_pos_w[0, :2].tolist()}, goal {goal[0].tolist()}, "
    f"initial distance {float(torch.norm(goal - robot.data.root_pos_w[:, :2], dim=1).mean()):.2f} m, dt {dt}")
t, arrived, last_report = 0.0, False, 0.0
while t < args.timeout and not arrived:
    t0 = time.time()
    d = robot.data
    err = goal - d.root_pos_w[:, :2]
    dist = torch.norm(err, dim=1)
    c, s = torch.cos(d.heading_w), torch.sin(d.heading_w)
    ex, ey = c * err[:, 0] + s * err[:, 1], -s * err[:, 0] + c * err[:, 1]
    bearing = torch.atan2(ey, ex)
    speed = torch.clamp((dist - args.standoff) / args.tracking, min=0.0, max=args.vmax)
    speed = torch.where(bearing.abs() > 0.6, torch.zeros_like(speed), speed)
    vel.vel_command_b[:, 0] = speed
    vel.vel_command_b[:, 1] = 0.0
    vel.vel_command_b[:, 2] = torch.clamp(2.0 * bearing, -1.0, 1.0)
    park_arm()
    with torch.inference_mode():
        obs, *_ = env.step(policy(obs))
    t += dt
    if t - last_report >= 1.0:
        last_report = t
        say(f"debug: t={t:5.1f}s dist={float(dist.mean()):.2f} m "
            f"cmd_vx={float(speed.mean()):.2f} actual_vx={float(d.root_lin_vel_b[:, 0].mean()):+.2f}")
    arrived = bool((dist <= args.standoff + 0.05).all())
    if args.real_time:
        time.sleep(max(0.0, dt - (time.time() - t0)))

approach = float(torch.norm(goal - robot.data.root_pos_w[:, :2], dim=1).mean())
say(f"navigate: {'arrived' if arrived else 'TIMED OUT'} after {t:.1f}s, "
    f"standing {approach:.2f} m from the target")

# ------------------------------------------------------------------- 2. reach
for _ in range(int(round(args.reach_time / dt))):
    t0 = time.time()
    vel.vel_command_b[:] = 0.0
    ee.pose_command_w[:, :3] = target_w
    ee.pose_command_w[:, 3:] = IDENT
    with torch.inference_mode():
        obs, *_ = env.step(policy(obs))
    if args.real_time:
        time.sleep(max(0.0, dt - (time.time() - t0)))

d = robot.data
err = float(ee.metrics["position_error"].mean())
hand_z = float(d.body_pos_w[:, hand_idx, 2].mean() - u.scene.env_origins[:, 2].mean())
base_z = float(d.root_pos_w[:, 2].mean() - u.scene.env_origins[:, 2].mean())
say(f"reach: hand error {err * 100:.1f} cm | hand height {hand_z:.3f} m (target {TZ}) | "
    f"base height {base_z:.3f} m")
say(f"=> {'SUCCESS' if err < 0.05 and arrived else 'not within 5 cm'}")
if args.out:
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"task": args.task, "checkpoint": args.checkpoint, "target": [TX, TY, TZ],
                   "navigate_s": t, "arrived": arrived, "standoff_m": approach,
                   "hand_err_m": err, "hand_z": hand_z, "base_z": base_z}, f, indent=2)
    say(f"wrote {args.out}")
say("done")
cam.close()          # release the camera before teardown
env.close()
app.close()
