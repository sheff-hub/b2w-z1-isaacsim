"""Does the body help the arm? Sweep hand-target HEIGHT and watch the body.

The question only has meaning for a policy whose target is fixed in the world
(wb9): there, lowering the body genuinely brings the hand closer to a low
target. For a body-frame policy (wb1..wb8) the target rides along with the
robot, so the body cannot buy reach -- those runs are the control group, and a
flat line is the expected, correct result.

For each target height the robot is held at a zero drive command, given that one
target for `--settle` seconds, and then measured:

  * base height      -- the whole point: does it crouch for low targets?
  * hand height      -- where the hand actually ended up
  * hand error       -- how close to the target it got
  * leg joint travel -- how far the legs moved from their stance

    /isaac-sim/python.sh /workspace/evaluation/crouch_test.py --task <id> --checkpoint <path>

Reported as a table, and as JSON with --out. Compare the base-height column of a
world-target policy against a body-frame one.
"""

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--reach", type=float, default=0.45, help="target distance in front, m")
parser.add_argument("--settle", type=float, default=4.0, help="seconds per target")
parser.add_argument("--heights", type=str, default="0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95")
parser.add_argument("--out", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import robot_lab.tasks  # noqa: E402,F401


def say(*a):
    print("CROUCH|", *a, flush=True)


cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
cfg.seed = 42
cfg.episode_length_s = 1.0e4
cfg.observations.policy.enable_corruption = False

# the script drives both commands itself; nothing may resample underneath it
cfg.commands.base_velocity.heading_command = False
cfg.commands.base_velocity.rel_standing_envs = 0.0
cfg.commands.base_velocity.rel_heading_envs = 0.0
cfg.commands.base_velocity.resampling_time_range = (1.0e6, 1.0e6)
cfg.commands.base_velocity.ranges.heading = None
cfg.commands.ee_pose.resampling_time_range = (1.0e6, 1.0e6)

for name in ("randomize_rigid_body_material", "randomize_rigid_body_mass_base",
             "randomize_rigid_body_mass_others", "randomize_com_positions",
             "randomize_apply_external_force_torque", "randomize_actuator_gains",
             "randomize_push_robot"):
    if hasattr(cfg.events, name):
        setattr(cfg.events, name, None)
still = (0.0, 0.0)
cfg.events.randomize_reset_base.params = {
    "pose_range": {k: still for k in ("x", "y", "z", "roll", "pitch", "yaw")},
    "velocity_range": {k: still for k in ("x", "y", "z", "roll", "pitch", "yaw")},
}

agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
env = RslRlVecEnvWrapper(gym.make(args.task, cfg=cfg), clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)

u = env.unwrapped
robot = u.scene["robot"]
ee = u.command_manager.get_term("ee_pose")
vel = u.command_manager.get_term("base_velocity")
hand_idx = ee.body_idx
legs = [i for i, n in enumerate(robot.joint_names) if n.endswith(("_hip_joint", "_thigh_joint", "_calf_joint"))]
default_legs = robot.data.default_joint_pos[:, legs].clone()
world_frame = hasattr(ee, "pose_command_w") and type(ee).__name__ == "WorldPoseCommand"
say(f"task {args.task}")
say(f"target frame: {'WORLD (body can help)' if world_frame else 'BODY (body cannot help -- control group)'}")

dt = u.step_dt
steps = int(round(args.settle / dt))
heights = [float(h) for h in args.heights.split(",")]
rows = []
obs = env.get_observations()
for h in heights:
    with torch.inference_mode():
        for k in range(steps):
            vel.vel_command_b[:] = 0.0
            d = robot.data
            if world_frame:
                # absolute point: args.reach ahead in the heading frame, height h
                fwd = torch.stack([torch.cos(d.heading_w), torch.sin(d.heading_w)], dim=1)
                ee.pose_command_w[:, 0:2] = d.root_pos_w[:, 0:2] + args.reach * fwd
                ee.pose_command_w[:, 2] = u.scene.env_origins[:, 2] + h
                ee.pose_command_w[:, 3:] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=u.device)
            else:
                # the body-frame equivalent of the same world height, measured
                # from the stance height the robot starts at
                ee.pose_command_b[:, 0] = args.reach
                ee.pose_command_b[:, 1] = 0.0
                ee.pose_command_b[:, 2] = h - 0.61
                ee.pose_command_b[:, 3:] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=u.device)
            obs, *_ = env.step(policy(obs))
            if k == steps - int(round(1.0 / dt)):   # start averaging over the last second
                acc = {"base": 0.0, "hand": 0.0, "err": 0.0, "legs": 0.0, "n": 0}
            if k >= steps - int(round(1.0 / dt)):
                acc["base"] += float(d.root_pos_w[:, 2].mean())
                acc["hand"] += float(d.body_pos_w[:, hand_idx, 2].mean())
                acc["err"] += float(ee.metrics["position_error"].mean())
                acc["legs"] += float((d.joint_pos[:, legs] - default_legs).abs().mean())
                acc["n"] += 1
    n = acc["n"]
    rows.append({"target_h": h, "base_h": acc["base"] / n, "hand_h": acc["hand"] / n,
                 "err": acc["err"] / n, "leg_travel": acc["legs"] / n})
    r = rows[-1]
    say(f"target {h:.2f} m | base {r['base_h']:.3f} | hand {r['hand_h']:.3f} | "
        f"err {r['err'] * 100:5.1f} cm | leg travel {r['leg_travel']:.3f} rad")

lo, hi = rows[0], rows[-1]
span = hi["base_h"] - lo["base_h"]
say(f"base height change from lowest to highest target: {span * 100:+.1f} cm")
say("=> the body HELPS the arm" if abs(span) > 0.03 else "=> the body does NOT move with target height")
if args.out:
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"task": args.task, "checkpoint": args.checkpoint, "world_frame": world_frame,
                   "reach": args.reach, "rows": rows, "base_span": span}, f, indent=2)
    say(f"wrote {args.out}")
say("done")
env.close()
app.close()
