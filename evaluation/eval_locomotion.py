"""Measure how well a velocity-tracking policy drives, under a fixed command script.

Runs one checkpoint on one task through the same sequence of commands every time
(forward, brake, reverse, turns, sideways, arc) and reports, per phase:

  * tracking   -- mean |actual - commanded| for forward speed, sideways speed and
                  turn rate, measured after the first 1 s of each phase settles
  * tilt       -- worst nose-down pitch (the braking dip shows here), mean and
                  worst |roll|
  * height     -- mean base height
  * leg effort -- mean |torque| over the 12 leg joints: what carrying costs
  * falls      -- environments that tipped past 60 deg at any point
  * hand       -- if the task also commands a hand target (ee_pose), its error
                  once the arm has had 1.5 s on the current target. Lets a
                  whole-body policy be compared with the two-policy setup
                  measured by loco_reach.py on exactly the same driving script.

For a clean A/B comparison every source of randomness in the task is switched
off: friction, mass and centre-of-mass randomization, external forces, pushes,
actuator-gain randomization, observation noise. Robots start upright and still.
So two runs differ only in what the task itself differs in (here: the arm).

    /isaac-sim/python.sh /workspace/evaluation/eval_locomotion.py \
        --task <gym id> --checkpoint <path> [--num_envs 8] [--out <json>]
"""

import argparse
import json
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--speed", type=float, default=1.0,
                    help="scale every commanded velocity. The phases below ask for 1.0 m/s, but "
                         "the world-target policies (wb9, wb12, wb14) are trained only to "
                         "+/-0.3 m/s -- at 1.0 they are extrapolating, and they fall. Use "
                         "--speed 0.3 to test inside the envelope they were trained for.")
parser.add_argument("--out", default=None, help="write results as JSON here")
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

import robot_lab.tasks  # noqa: E402,F401  (registers the robot_lab tasks)


def say(*a):
    print("EVAL|", *a, flush=True)


# (name, seconds, vx, vy, wz). Within the trained ranges (all +/-1.0).
PHASES = [
    ("settle",  3.0,  0.0, 0.0,  0.0),
    ("forward", 4.0,  1.0, 0.0,  0.0),
    ("brake",   2.0,  0.0, 0.0,  0.0),
    ("reverse", 3.0, -1.0, 0.0,  0.0),
    ("stop",    2.0,  0.0, 0.0,  0.0),
    ("turn_L",  3.0,  0.0, 0.0,  1.0),
    ("turn_R",  3.0,  0.0, 0.0, -1.0),
    ("strafe",  3.0,  0.0, 0.5,  0.0),
    ("arc",     3.0,  0.8, 0.0,  0.6),
    ("final",   2.0,  0.0, 0.0,  0.0),
]
SETTLE_S = 1.0  # tracking is measured only after this much of each phase

# ---------------------------------------------------------------- task setup
cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
cfg.seed = 42
cfg.episode_length_s = 1.0e4  # no time-out resets mid-script

cmd = cfg.commands.base_velocity
cmd.heading_command = False          # otherwise yaw rate is recomputed from heading
cmd.rel_standing_envs = 0.0
cmd.rel_heading_envs = 0.0
cmd.resampling_time_range = (1.0e6, 1.0e6)
cmd.debug_vis = False
cmd.ranges.heading = None

cfg.observations.policy.enable_corruption = False
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
command = u.command_manager.get_term("base_velocity")
legs = [i for i, n in enumerate(robot.joint_names) if n.endswith(("_hip_joint", "_thigh_joint", "_calf_joint"))]
ee = u.command_manager.get_term("ee_pose") if "ee_pose" in u.command_manager.active_terms else None
ee_period = cfg.commands.ee_pose.resampling_time_range[0] if ee is not None else 0.0
dt = u.step_dt
say(f"task {args.task}: {robot.num_joints} joints, {args.num_envs} envs, dt {dt}")

# ------------------------------------------------------------------ the run
results = []
fell = torch.zeros(args.num_envs, dtype=torch.bool, device=u.device)
obs = env.get_observations()
for name, seconds, vx, vy, wz in PHASES:
    vx, vy, wz = vx * args.speed, vy * args.speed, wz * args.speed
    target = torch.tensor([vx, vy, wz], device=u.device)
    n = int(round(seconds / dt))
    skip = int(round(SETTLE_S / dt))
    err = torch.zeros(3, device=u.device)
    speed = torch.zeros(3, device=u.device)
    counted = 0
    pitch_max = -1e9
    roll_abs_sum, roll_abs_max, height_sum, torque_sum = 0.0, 0.0, 0.0, 0.0
    hand_sum, hand_n = 0.0, 0
    for k in range(n):
        command.vel_command_b[:] = target
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))

        d = robot.data
        actual = torch.stack([d.root_lin_vel_b[:, 0], d.root_lin_vel_b[:, 1], d.root_ang_vel_b[:, 2]], dim=1)
        g = d.projected_gravity_b
        pitch = torch.rad2deg(torch.atan2(g[:, 0], -g[:, 2]))   # + = nose down
        roll = torch.rad2deg(torch.atan2(-g[:, 1], -g[:, 2]))   # + = right side down
        fell |= g[:, 2] > -0.5                                  # tilted past 60 deg

        pitch_max = max(pitch_max, float(pitch.max()))
        roll_abs_sum += float(roll.abs().mean())
        roll_abs_max = max(roll_abs_max, float(roll.abs().max()))
        height_sum += float(d.root_pos_w[:, 2].mean())
        torque_sum += float(d.applied_torque[:, legs].abs().mean())
        if ee is not None:
            settled = ee.time_left <= (ee_period - 1.5)   # 1.5 s on this target
            if bool(settled.any()):
                hand_sum += float(ee.metrics["position_error"][settled].mean())
                hand_n += 1
        if k >= skip:
            err += (actual - target).abs().mean(dim=0)
            speed += actual.mean(dim=0)
            counted += 1

    row = {
        "phase": name, "cmd": [vx, vy, wz],
        "err_vx": float(err[0] / counted), "err_vy": float(err[1] / counted), "err_wz": float(err[2] / counted),
        "act_vx": float(speed[0] / counted), "act_vy": float(speed[1] / counted), "act_wz": float(speed[2] / counted),
        "pitch_max": pitch_max, "roll_mean": roll_abs_sum / n, "roll_max": roll_abs_max,
        "height": height_sum / n, "leg_torque": torque_sum / n,
        "hand_err": (hand_sum / hand_n) if hand_n else None,
    }
    results.append(row)
    say(f"{name:8s} cmd {vx:+.1f} {vy:+.1f} {wz:+.1f} | act {row['act_vx']:+.2f} {row['act_vy']:+.2f} "
        f"{row['act_wz']:+.2f} | err {row['err_vx']:.3f} {row['err_vy']:.3f} {row['err_wz']:.3f} | "
        f"pitch_max {pitch_max:+.1f} roll {row['roll_mean']:.1f}/{roll_abs_max:.1f} | "
        f"h {row['height']:.3f} | tau {row['leg_torque']:.1f}"
        + (f" | hand {row['hand_err'] * 100:.1f} cm" if row["hand_err"] is not None else ""))

summary = {"task": args.task, "checkpoint": args.checkpoint, "num_envs": args.num_envs,
           "falls": int(fell.sum()), "phases": results}
say(f"falls: {summary['falls']} / {args.num_envs}")
if args.out:
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    say(f"wrote {args.out}")
say("done")
env.close()
app.close()
