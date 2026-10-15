"""Drive the robot from A to B: a waypoint follower on top of a velocity policy.

Phase 4's stated goal. The policy only understands velocities, so this converts
"go to (x, y)" into the [vx, vy, yaw_rate] command it does understand:

    error       = goal - position, rotated into the robot's own frame
    bearing     = atan2(e_y, e_x)          how far off straight-ahead the goal is
    |bearing| large  -> turn on the spot   (driving forward would arc away)
    |bearing| small  -> drive forward, still correcting heading

Two details that come from our own measurements:

  * SPEED UNDERSHOOT. Every policy trained here reaches only ~75-80% of a
    commanded forward speed (see PHASE4_OVERNIGHT.md), so the requested speed is
    divided by --tracking (default 0.8). Without this the robot creeps in slower
    than asked and the arrival time is misleading.
  * The commanded speed is capped at --vmax, which must stay inside the range
    the policy was trained on (+/-1.0 m/s) or it is extrapolating.

Reports per waypoint: time to arrive, path length against the straight-line
distance, and the final distance. A waypoint that times out is reported as such
rather than silently skipped.

    /isaac-sim/python.sh /workspace/project/waypoint.py --task <id> --checkpoint <path> \
        --waypoints "3,0; 3,3; 0,3; 0,0" [--stream --real-time]
"""

import argparse
import json
import math
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--waypoints", default="3,0; 3,3; 0,3; 0,0", help='"x,y; x,y; ..." metres, env-local')
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--radius", type=float, default=0.30, help="arrival radius, m")
parser.add_argument("--vmax", type=float, default=0.9, help="max commanded speed (trained range is 1.0)")
parser.add_argument("--wmax", type=float, default=1.0, help="max commanded turn rate, rad/s")
parser.add_argument("--tracking", type=float, default=0.8, help="measured fraction of commanded speed achieved")
parser.add_argument("--turn_first", type=float, default=0.6, help="bearing beyond which it turns on the spot, rad")
parser.add_argument("--timeout", type=float, default=30.0, help="seconds allowed per waypoint")
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
    print("WP|", *a, flush=True)


GOALS = [tuple(float(v) for v in p.split(",")) for p in args.waypoints.split(";") if p.strip()]

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
if hasattr(cfg.commands, "ee_pose"):
    cfg.commands.ee_pose.resampling_time_range = (1.0e6, 1.0e6)
still = (0.0, 0.0)
cfg.events.randomize_reset_base.params = {
    "pose_range": {k: still for k in ("x", "y", "z", "roll", "pitch", "yaw")},
    "velocity_range": {k: still for k in ("x", "y", "z", "roll", "pitch", "yaw")},
}
cfg.viewer.origin_type = "asset_root"
cfg.viewer.asset_name = "robot"
cfg.viewer.eye = (3.0, 3.0, 2.0)

agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
env = RslRlVecEnvWrapper(gym.make(args.task, cfg=cfg), clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)

u = env.unwrapped
robot = u.scene["robot"]
vel = u.command_manager.get_term("base_velocity")
ee = u.command_manager.get_term("ee_pose") if "ee_pose" in u.command_manager.active_terms else None
dt = u.step_dt
cam = CameraToggle(u, following=True)  # V releases the camera, C snaps it back
say(f"{args.task}: {len(GOALS)} waypoints, {args.num_envs} envs, arrival radius {args.radius} m")


def hold_arm():
    """Keep the arm at one steady, reachable pose so it is not a variable here."""
    if ee is None:
        return
    d = robot.data
    if type(ee).__name__ == "WorldPoseCommand":
        fwd = torch.stack([torch.cos(d.heading_w), torch.sin(d.heading_w)], dim=1)
        ee.pose_command_w[:, 0:2] = d.root_pos_w[:, 0:2] + 0.45 * fwd
        ee.pose_command_w[:, 2] = u.scene.env_origins[:, 2] + 0.55
        ee.pose_command_w[:, 3:] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=u.device)
    else:
        ee.pose_command_b[:, :3] = torch.tensor([0.45, 0.0, 0.40], device=u.device)
        ee.pose_command_b[:, 3:] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=u.device)


obs = env.get_observations()
results = []
for gx, gy in GOALS:
    goal = u.scene.env_origins[:, :2] + torch.tensor([gx, gy], device=u.device)
    start = robot.data.root_pos_w[:, :2].clone()
    straight = float(torch.norm(goal - start, dim=1).mean())
    travelled = torch.zeros(u.num_envs, device=u.device)
    prev = start.clone()
    arrived = torch.zeros(u.num_envs, dtype=torch.bool, device=u.device)
    arrive_t = torch.full((u.num_envs,), float("nan"), device=u.device)
    t = 0.0
    while t < args.timeout and not bool(arrived.all()):
        t0 = time.time()
        d = robot.data
        err_w = goal - d.root_pos_w[:, :2]
        dist = torch.norm(err_w, dim=1)
        # rotate the error into the robot's frame
        c, s = torch.cos(d.heading_w), torch.sin(d.heading_w)
        ex = c * err_w[:, 0] + s * err_w[:, 1]
        ey = -s * err_w[:, 0] + c * err_w[:, 1]
        bearing = torch.atan2(ey, ex)

        wz = torch.clamp(2.0 * bearing, -args.wmax, args.wmax)
        speed = torch.clamp(dist / args.tracking, max=args.vmax)      # undershoot compensation
        speed = torch.where(bearing.abs() > args.turn_first, torch.zeros_like(speed), speed)
        speed = torch.where(arrived, torch.zeros_like(speed), speed)
        wz = torch.where(arrived, torch.zeros_like(wz), wz)

        vel.vel_command_b[:, 0] = speed
        vel.vel_command_b[:, 1] = 0.0
        vel.vel_command_b[:, 2] = wz
        hold_arm()
        with torch.inference_mode():
            obs, *_ = env.step(policy(obs))

        now = d.root_pos_w[:, :2]
        travelled += torch.norm(now - prev, dim=1)
        prev = now.clone()
        newly = (~arrived) & (dist < args.radius)
        arrive_t = torch.where(newly, torch.full_like(arrive_t, t), arrive_t)
        arrived |= dist < args.radius
        t += dt
        if args.real_time:
            time.sleep(max(0.0, dt - (time.time() - t0)))

    final = float(torch.norm(goal - robot.data.root_pos_w[:, :2], dim=1).mean())
    ok = int(arrived.sum())
    mean_t = float(arrive_t[~torch.isnan(arrive_t)].mean()) if ok else float("nan")
    path = float(travelled.mean())
    results.append({"goal": [gx, gy], "reached": ok, "of": u.num_envs, "time_s": mean_t,
                    "straight_m": straight, "path_m": path, "final_err_m": final})
    say(f"goal ({gx:+.1f},{gy:+.1f}) | reached {ok}/{u.num_envs} | t {mean_t:5.1f}s | "
        f"path {path:5.2f} m vs straight {straight:5.2f} m | final err {final:.2f} m"
        + ("" if ok else "  TIMED OUT"))

tot = sum(r["path_m"] for r in results)
reached = sum(r["reached"] for r in results)
say(f"total: {reached}/{len(GOALS) * u.num_envs} waypoint arrivals, {tot:.1f} m driven")
if args.out:
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"task": args.task, "checkpoint": args.checkpoint, "vmax": args.vmax,
                   "tracking": args.tracking, "radius": args.radius, "waypoints": results}, f, indent=2)
    say(f"wrote {args.out}")
say("done")
cam.close()          # release the camera before teardown
env.close()
app.close()
