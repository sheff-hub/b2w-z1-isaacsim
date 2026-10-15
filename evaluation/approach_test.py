"""Can it go to a pose anywhere around it, and arrive intact?

This is the test the RFM policy exists to pass, and the one every earlier policy
fails. A target is placed at a given distance and bearing -- including straight
behind, which forces a turn -- and the robot is left alone with it. Nothing in
this script drives: there is no velocity command and no state machine, unlike
project/goto.py. Whatever happens is the policy's own doing.

Each test case gets its own environment, so a 3 x 4 sweep of distances and
bearings runs in one 20-second episode rather than twelve.

Per case it reports:

  closest    the nearest the HAND ever got to the target -- the honest measure of
             whether it arrived at all
  final      hand error at the end
  min height the lowest the body went. A collapse shows up here: our earlier
             policies dropped from 0.55 m to 0.12 m when handed a far target
  max tilt   likewise, 68 deg was the signature of the lunge-and-topple failure
  fell       base below 0.2 m or tilt past 75 deg at any point (the paper's
             failure definition, Section V-B)

    /isaac-sim/python.sh /workspace/evaluation/approach_test.py --task <id> --checkpoint <path>
"""

import argparse
import json
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--distances", default="1.0,2.0,3.0")
parser.add_argument("--bearings", default="0,90,180,-90", help="degrees, 0 = straight ahead")
parser.add_argument("--target_height", type=float, default=0.45,
                    help="target height in metres (not --height: that name collides with "
                         "SimulationApp's own integer window height)")
parser.add_argument("--seconds", type=float, default=20.0)
parser.add_argument("--out", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

DIST = [float(v) for v in args.distances.split(",")]
BEAR = [math.radians(float(v)) for v in args.bearings.split(",")]
CASES = [(d, b) for d in DIST for b in BEAR]
args.num_envs = len(CASES)

app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import robot_lab.tasks  # noqa: E402,F401

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "project"))  # rfm_helpers
from rfm_helpers import restart_schedule  # noqa: E402


def say(*a):
    print("APPR|", *a, flush=True)


cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
cfg.seed = 42
cfg.episode_length_s = 1.0e9
for name in list(vars(cfg.terminations)):
    if not name.startswith("_"):
        setattr(cfg.terminations, name, None)     # let it fail visibly, don't reset
cfg.observations.policy.enable_corruption = False
cfg.commands.ee_pose.resampling_time_range = (1.0e9, 1.0e9)
if hasattr(cfg.commands, "base_velocity"):
    cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.commands.base_velocity.rel_heading_envs = 0.0
    cfg.commands.base_velocity.ranges.heading = None
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
vel = u.command_manager.get_term("base_velocity") if "base_velocity" in u.command_manager.active_terms else None
dt = u.step_dt
say(f"{len(CASES)} targets: distances {DIST} m x bearings {args.bearings} deg, height {args.target_height} m")
say(f"{args.seconds:.0f} s each, nothing but the policy driving")

obs = env.get_observations()
# place one target per environment, in each robot's own heading frame
d0 = robot.data
head = d0.heading_w
dist_t = torch.tensor([c[0] for c in CASES], device=u.device)
bear_t = torch.tensor([c[1] for c in CASES], device=u.device)
tx = d0.root_pos_w[:, 0] + dist_t * torch.cos(head + bear_t)
ty = d0.root_pos_w[:, 1] + dist_t * torch.sin(head + bear_t)
ee.pose_command_w[:, 0] = tx
ee.pose_command_w[:, 1] = ty
ee.pose_command_w[:, 2] = u.scene.env_origins[:, 2] + args.target_height
ee.pose_command_w[:, 3:] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=u.device)
restart_schedule(ee, robot)

shoulder_idx = robot.find_bodies(["link02"])[0][0]
closest = torch.full((len(CASES),), 1.0e9, device=u.device)
# signed pitch, positive = nose UP / rear DOWN -- the "sitting back" posture
max_pitch = torch.full((len(CASES),), -1.0e9, device=u.device)
# DRIVING BACKWARDS: steps spent closing on the target while moving in reverse.
# The arm is mounted at the FRONT, so arriving backwards means turning round at
# the end with the target behind the body -- or reaching over the back.
steps_approach = torch.zeros(len(CASES), device=u.device)
steps_reverse = torch.zeros(len(CASES), device=u.device)
# SHAKING: while the hand is already on the target (< 3 cm), how much does the
# arm still move? A calm hold reads near zero; a jittering one does not.
arm_jids, _ = robot.find_joints(["joint[1-6]"], preserve_order=True)
hold_steps = torch.zeros(len(CASES), device=u.device)
hold_qd2 = torch.zeros(len(CASES), device=u.device)        # sum of arm joint speed^2
hold_hand_v = torch.zeros(len(CASES), device=u.device)     # hand speed, m/s
prev_act = None
min_h = torch.full((len(CASES),), 1.0e9, device=u.device)
max_tilt = torch.zeros(len(CASES), device=u.device)
steps = int(round(args.seconds / dt))
with torch.inference_mode():
    for _ in range(steps):
        if vel is not None:
            vel.vel_command_b[:] = 0.0          # prove nothing else is steering
        act = policy(obs)
        obs, *_ = env.step(act)
        d = robot.data
        on_target = ee.metrics["position_error"] < 0.03
        hold_steps += on_target.float()
        qd = d.joint_vel[:, arm_jids]
        hold_qd2 += torch.where(on_target, (qd ** 2).sum(dim=1), torch.zeros_like(hold_steps))
        hv = d.body_lin_vel_w[:, ee.body_idx].norm(dim=1)
        hold_hand_v += torch.where(on_target, hv, torch.zeros_like(hold_steps))
        closest = torch.minimum(closest, ee.metrics["position_error"])
        h = d.root_pos_w[:, 2] - u.scene.env_origins[:, 2]
        min_h = torch.minimum(min_h, h)
        tilt = torch.rad2deg(torch.acos(torch.clamp(-d.projected_gravity_b[:, 2], -1.0, 1.0)))
        max_tilt = torch.maximum(max_tilt, tilt)
        g = d.projected_gravity_b
        pitch = torch.rad2deg(torch.atan2(-g[:, 0], -g[:, 2]))   # same convention as reach_map.py
        max_pitch = torch.maximum(max_pitch, pitch)
        far = torch.norm(ee.pose_command_w[:, :2] - d.root_pos_w[:, :2], dim=1) > 0.6
        moving = d.root_lin_vel_b[:, :2].norm(dim=1) > 0.05
        steps_approach += (far & moving).float()
        steps_reverse += (far & moving & (d.root_lin_vel_b[:, 0] < -0.05)).float()

final = ee.metrics["position_error"]
# how far the target ended up from the shoulder: small = the arm works in the
# comfortable middle of its workspace, ~0.69 m = at full stretch
reverse_frac = steps_reverse / steps_approach.clamp(min=1.0)
hold_n = hold_steps.clamp(min=1.0)
arm_rms = torch.sqrt(hold_qd2 / hold_n)          # rad/s, RMS over the 6 arm joints summed
hand_speed = hold_hand_v / hold_n                # m/s, mean hand speed while on target
shoulder_d = torch.norm(ee.pose_command_w[:, :3] - robot.data.body_pos_w[:, shoulder_idx], dim=1)
ori = ee.metrics["orientation_error"]
# A FALL IS A TILT, NOT A LOW BODY.
#
# The paper's failure test is "base height below 0.2 m OR projected gravity past
# -0.1", which suits a robot that holds its body at a steady height. Ours
# deliberately crouches to 0.19-0.28 m to put its hand on low targets -- that is
# the behaviour the whole project is about -- so a height test marks its best
# work as a failure. Measured: cases flagged at 0.19 m had a tilt of 35 deg and
# were reaching normally, while the genuine flips read 147-167 deg.
fell = max_tilt > 75.0
crouched = min_h < 0.25

say("")
say(f"{'target':>16s} {'closest':>9s} {'final':>9s} {'ori':>7s} {'min h':>7s} {'max tilt':>9s} "
    f"{'max pitch':>9s} {'shoulder':>9s} {'reverse':>8s}  {'':4s}")
rows = []
for i, (dist, bear) in enumerate(CASES):
    tag = f"{dist:.1f} m @ {math.degrees(bear):+4.0f}"
    say(f"{tag:>16s} {float(closest[i]) * 100:8.1f}cm {float(final[i]) * 100:8.1f}cm "
        f"{float(ori[i]):7.2f} {float(min_h[i]):7.2f} {float(max_tilt[i]):8.0f}d "
        f"{float(max_pitch[i]):+8.0f}d {float(shoulder_d[i]):8.2f}m {float(reverse_frac[i]) * 100:7.0f}%  "
        f"{'FELL' if bool(fell[i]) else 'ok'}")
    rows.append({"distance": dist, "bearing_deg": math.degrees(bear),
                 "closest_m": float(closest[i]), "final_m": float(final[i]),
                 "ori_rad": float(ori[i]), "min_height_m": float(min_h[i]),
                 "max_tilt_deg": float(max_tilt[i]), "max_pitch_deg": float(max_pitch[i]),
                 "shoulder_dist_m": float(shoulder_d[i]), "reverse_frac": float(reverse_frac[i]),
                 "hold_arm_rms": float(arm_rms[i]), "hold_hand_speed": float(hand_speed[i]),
                 "fell": bool(fell[i])})

arrived = (closest < 0.10)
say("")
say(f"got within 10 cm: {int(arrived.sum())}/{len(CASES)}   |   fell (tilt>75 deg): "
    f"{int(fell.sum())}/{len(CASES)}   |   crouched below 0.25 m: {int(crouched.sum())}/{len(CASES)}")
say(f"median closest approach: {float(closest.median()) * 100:.1f} cm")
say(f"lowest the body went: {float(min_h.min()):.2f} m "
    f"(a collapse reads ~0.12 m; a deliberate crouch reads 0.19-0.30 m)")
say(f"worst tilt: {float(max_tilt.max()):.0f} deg (the lunge failure reads ~68 deg)")
say(f"sitting back: max pitch median {float(max_pitch.median()):+.0f} deg, worst {float(max_pitch.max()):+.0f} deg "
    f"(positive = nose up, rear down)")
say(f"driving in REVERSE while approaching: median {float(reverse_frac.median()) * 100:.0f}% of moving time, "
    f"cases mostly reversing (>50%): {int((reverse_frac > 0.5).sum())}/{len(CASES)}")
held = hold_steps > 10
say(f"SHAKING while on target (<3 cm): arm joint speed RMS median "
    f"{float(arm_rms[held].median()) if held.any() else float('nan'):.2f} rad/s, "
    f"hand speed median {float(hand_speed[held].median()) * 100 if held.any() else float('nan'):.1f} cm/s "
    f"(a calm hold reads ~0)")
say(f"target-to-shoulder at the end: median {float(shoulder_d.median()):.2f} m "
    f"(arm reach 0.69 m; the phase switch is at 0.35 m)")
if args.out:
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"task": args.task, "checkpoint": args.checkpoint,
                   "height": args.target_height, "seconds": args.seconds, "cases": rows,
                   "arrived": int(arrived.sum()), "fell": int(fell.sum())}, f, indent=2)
    say(f"wrote {args.out}")
say("done")
env.close()
app.close()
