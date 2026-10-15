"""Which world targets can this policy actually reach? A map, not a single number.

crouch_test.py sweeps target HEIGHT at one fixed distance and answers "does the
body help?". This answers the larger question the workspace measurement raises:
of the whole volume the robot could reach in principle, which parts does the
trained policy reach in practice?

Every environment is given its OWN target from a grid of (forward distance,
height) -- so a 7 x 13 grid is measured in one 4-second settle, not 91 of them.
For each cell it reports:

  * hand error       -- reached if within --tol (default 5 cm)
  * base height      -- how far the body crouched or stood to serve that cell
  * body pitch       -- leaning is the other way the body buys reach, and the
                        lowest targets need it: the Z1's joint2/joint3 cannot
                        swing back past the shoulder, so a LEVEL body bottoms out
                        around 0.24 m however far it crouches
  * arm self-contact -- whether the arm was touching the robot while it did it
                        (only meaningful for a task with self-collisions on)

Printed as a map so the reachable region is visible at a glance, plus a summary
line that is comparable between policies.

    /isaac-sim/python.sh /workspace/evaluation/reach_map.py --task <id> --checkpoint <path> \
        [--x 0.25,0.75,6] [--z 0.10,1.40,14] [--out ...]
"""

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--x", default="0.25,0.75,6", help="forward distance: start,stop,count")
parser.add_argument("--z", default="0.10,1.40,14", help="world height: start,stop,count")
parser.add_argument("--settle", type=float, default=5.0, help="seconds per target")
parser.add_argument("--tol", type=float, default=0.05, help="counts as reached, m")
parser.add_argument("--fixed_targets", action="store_true",
                    help="Place each target ONCE in the world instead of re-deriving it from the "
                         "robot's pose every step. Required for any policy that drives: the "
                         "step-by-step placement silently makes the target ride along with the "
                         "robot, so a policy that approaches can never arrive and scores ~0.")
parser.add_argument("--out", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True


def grid(spec):
    a, b, n = spec.split(",")
    a, b, n = float(a), float(b), int(n)
    return [a + (b - a) * i / max(1, n - 1) for i in range(n)]


XS, ZS = grid(args.x), grid(args.z)
CELLS = [(x, z) for z in ZS for x in XS]
args.num_envs = len(CELLS)

app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import robot_lab.tasks  # noqa: E402,F401


def say(*a):
    print("MAP|", *a, flush=True)


cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
cfg.seed = 42
cfg.episode_length_s = 1.0e4
cfg.observations.policy.enable_corruption = False
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
arm_sensor = u.scene.sensors.get("arm_contacts") if hasattr(u.scene, "sensors") else None
world_frame = type(ee).__name__ == "WorldPoseCommand"
say(f"task {args.task}")
say(f"target frame: {'WORLD' if world_frame else 'BODY (the map is not meaningful)'}")
say(f"grid: {len(XS)} x {len(ZS)} = {len(CELLS)} targets, one per environment, "
    f"{args.settle}s settle, tolerance {args.tol * 100:.0f} cm")
say(f"self-collision sensing: {'on' if arm_sensor is not None else 'absent (task has none)'}")

dt = u.step_dt
steps = int(round(args.settle / dt))
avg_from = steps - int(round(1.0 / dt))
tx = torch.tensor([c[0] for c in CELLS], device=u.device)
tz = torch.tensor([c[1] for c in CELLS], device=u.device)
IDENT = torch.tensor([1.0, 0.0, 0.0, 0.0], device=u.device)

# How far the arm intrudes into the robot's own volume.
#
# The contact sensor cannot answer this: with self-collisions off it only ever
# sees the arm touching the GROUND, and with them on the asset jams (see
# PHASE4_WORKSPACE.md section 5). So intrusion is measured geometrically, the
# same way the training penalty defines it -- the depth to which link03..link06
# enter the measured body and lidar-mast boxes. This is the before/after number
# for "the arm passes through the lidar mount".
from isaaclab.utils.math import quat_apply_inverse  # noqa: E402

from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.unitree_b2w.self_clearance import (  # noqa: E402
    KEEPOUT_BOXES,
)

arm_ids, arm_names = robot.find_bodies(["link0[3-6]"], preserve_order=True)
# WHO IS DOING THE REACHING -- the arm, or the whole robot shuffling?
# A policy can satisfy a hand-position target by driving and rotating the base
# with the arm held still. That scores well and looks wrong, so measure both.
arm_joint_ids, _ = robot.find_joints(["joint[1-6]"], preserve_order=True)
arm_stow_q = robot.data.default_joint_pos[:, arm_joint_ids].clone()
say(f"intrusion measured on {arm_names}")


BOX_NAMES = ("body", "lidar mast")


def intrusion(per_box=False):
    """Max depth (m) any monitored arm link is inside the body/mast volumes."""
    d = robot.data
    pts_w = d.body_pos_w[:, arm_ids, :]
    rel = (pts_w - d.root_pos_w.unsqueeze(1)).reshape(-1, 3)
    quat = d.root_quat_w.repeat_interleave(len(arm_ids), dim=0)
    pts_b = quat_apply_inverse(quat, rel).reshape(-1, len(arm_ids), 3)
    out = []
    for centre, half in KEEPOUT_BOXES:
        c = torch.tensor(centre, device=u.device)
        h = torch.tensor(half, device=u.device)
        depth = (h - (pts_b - c).abs()).min(dim=-1).values.clamp(min=0.0)
        out.append(depth.max(dim=1).values)
    return torch.stack(out, dim=1) if per_box else torch.stack(out, dim=1).max(dim=1).values


obs = env.get_observations()
acc = {k: torch.zeros(len(CELLS), device=u.device) for k in ("err", "base", "pitch", "hand", "ori", "armq")}
yaw_travel = torch.zeros(len(CELLS), device=u.device)
prev_yaw = robot.data.heading_w.clone()
contact = torch.zeros(len(CELLS), device=u.device)
intrude = torch.zeros(len(CELLS), device=u.device)
intrude_box = torch.zeros(len(CELLS), len(KEEPOUT_BOXES), device=u.device)
n = 0
with torch.inference_mode():
    for k in range(steps):
        vel.vel_command_b[:] = 0.0
        d = robot.data
        if not (args.fixed_targets and k > 0):
            fwd = torch.stack([torch.cos(d.heading_w), torch.sin(d.heading_w)], dim=1)
            ee.pose_command_w[:, 0:2] = d.root_pos_w[:, 0:2] + tx.unsqueeze(1) * fwd
            ee.pose_command_w[:, 2] = u.scene.env_origins[:, 2] + tz
            ee.pose_command_w[:, 3:] = IDENT
        obs, *_ = env.step(policy(obs))
        dy = torch.atan2(torch.sin(robot.data.heading_w - prev_yaw),
                         torch.cos(robot.data.heading_w - prev_yaw))
        yaw_travel += dy.abs()
        prev_yaw = robot.data.heading_w.clone()
        per = intrusion(per_box=True)
        intrude_box = torch.maximum(intrude_box, per)
        intrude = torch.maximum(intrude, per.max(dim=1).values)
        if arm_sensor is not None:
            f = torch.max(torch.norm(arm_sensor.data.net_forces_w_history, dim=-1), dim=1)[0]
            contact = torch.maximum(contact, f.max(dim=1)[0])
        if k >= avg_from:
            g = d.projected_gravity_b
            acc["err"] += ee.metrics["position_error"]
            acc["ori"] += ee.metrics["orientation_error"]
            acc["armq"] += (d.joint_pos[:, arm_joint_ids] - arm_stow_q).abs().sum(dim=1)
            acc["base"] += d.root_pos_w[:, 2] - u.scene.env_origins[:, 2]
            acc["pitch"] += torch.rad2deg(torch.atan2(-g[:, 0], -g[:, 2]))
            acc["hand"] += d.body_pos_w[:, hand_idx, 2] - u.scene.env_origins[:, 2]
            n += 1

for key in acc:
    acc[key] /= max(1, n)
err = acc["err"].cpu().numpy()
base = acc["base"].cpu().numpy()
pitch = acc["pitch"].cpu().numpy()
touch = contact.cpu().numpy()
intr = intrude.cpu().numpy()

nx = len(XS)
say("")
say("=== hand error, cm (. = reached within tolerance, X = missed) ===")
say("      " + "".join(f"{x:7.2f}" for x in XS) + "   <- forward distance, m")
rows = []
for j, z in enumerate(ZS):
    cells = []
    for i in range(nx):
        e = err[j * nx + i] * 100
        cells.append(f"{e:6.1f}{'.' if e <= args.tol * 100 else 'X'}")
    say(f"z {z:4.2f}" + "".join(cells))
    rows.append([float(err[j * nx + i]) for i in range(nx)])

say("")
say("=== base height, m (how far the body moved to serve each row) ===")
say("      " + "".join(f"{x:7.2f}" for x in XS))
for j, z in enumerate(ZS):
    say(f"z {z:4.2f}" + "".join(f"{base[j * nx + i]:7.3f}" for i in range(nx)))

say("")
say("=== body pitch, deg (negative = nose down; how the lowest targets are won) ===")
say("      " + "".join(f"{x:7.2f}" for x in XS))
for j, z in enumerate(ZS):
    say(f"z {z:4.2f}" + "".join(f"{pitch[j * nx + i]:7.1f}" for i in range(nx)))

reached = err <= args.tol
say("")
say(f"reached {int(reached.sum())}/{len(CELLS)} cells ({100 * reached.mean():.0f}%) "
    f"within {args.tol * 100:.0f} cm")
if reached.any():
    zr = [CELLS[i][1] for i in range(len(CELLS)) if reached[i]]
    xr = [CELLS[i][0] for i in range(len(CELLS)) if reached[i]]
    say(f"reachable span: height {min(zr):.2f} - {max(zr):.2f} m "
        f"({(max(zr) - min(zr)) * 100:.0f} cm), distance {min(xr):.2f} - {max(xr):.2f} m")
    say(f"base height used: {base.min():.3f} - {base.max():.3f} m "
        f"({(base.max() - base.min()) * 100:.1f} cm of body travel)")
    say(f"body pitch used:  {pitch.min():+.1f} to {pitch.max():+.1f} deg")
say(f"median orientation error: {float(sorted(acc['ori'].cpu().numpy())[len(CELLS) // 2]):.3f} rad "
    f"(only meaningful for a task that commands orientation)")
say(f"median hand error over reached cells: "
    f"{100 * float(sorted(err[reached])[len(err[reached]) // 2]) if reached.any() else float('nan'):.1f} cm")
if arm_sensor is not None:
    hit = touch > 1.0
    say(f"arm touched the ground in {int(hit.sum())}/{len(CELLS)} cells "
        f"(peak force {touch.max():.0f} N; with self-collisions off this cannot "
        f"see arm-vs-body contact)")

say("")
say(f"ARM vs BASE: mean arm displacement from stow {float(acc['armq'].mean()):.2f} rad "
    f"(summed over 6 joints) | mean base yaw travel {float(yaw_travel.mean()):.2f} rad")
say("  a policy that reaches with its arm shows a large arm number and a small yaw one")

bad = intr > 0.0
say(f"ARM INTO THE ROBOT'S OWN VOLUME: {int(bad.sum())}/{len(CELLS)} cells "
    f"({100 * bad.mean():.0f}%), worst depth {100 * intr.max():.1f} cm, "
    f"mean over offending cells {100 * (intr[bad].mean() if bad.any() else 0):.1f} cm")
for bi, bname in enumerate(BOX_NAMES):
    col = intrude_box[:, bi].cpu().numpy()
    nb = (col > 0.0).sum()
    say(f"    into the {bname:11s}: {int(nb):2d}/{len(CELLS)} cells, worst {100 * col.max():.1f} cm")

if args.out:
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"task": args.task, "checkpoint": args.checkpoint, "xs": XS, "zs": ZS,
                   "tol": args.tol, "err_m": rows,
                   "base_h": [float(v) for v in base], "pitch_deg": [float(v) for v in pitch],
                   "contact_N": [float(v) for v in touch],
                   "intrusion_m": [float(v) for v in intr],
                   "reached": int(reached.sum()), "cells": len(CELLS)}, f, indent=2)
    say(f"wrote {args.out}")
say("done")
env.close()
app.close()
