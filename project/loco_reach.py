"""Drive the B2W and reach with the Z1 at once, using the two existing policies.

Phase 4, step 3, option A: no new training. The driving policy (model_2600) and
the reaching policy (reach v2) each get their own observation group of the
LocoReach task and control their own joints.

  keyboard mode   drive with the arrows / Z X (as in play.py), the arm reaching
                  a new target every 4 s; F toggles the arm between reaching and
                  folded, so the difference can be felt directly.

  eval mode       the fixed driving script of eval_locomotion.py, run twice --
                  arm folded, then arm reaching -- reporting per phase the
                  driving numbers (tracking, tilt, height, falls) and the hand's
                  error to its target once it has had 1.5 s to get there.

    /isaac-sim/python.sh /workspace/project/loco_reach.py --mode eval --out <json>
    /isaac-sim/python.sh /workspace/project/loco_reach.py --mode keyboard --real-time --livestream 1

Each policy is rebuilt from its training checkpoint (the actor: Linear/ELU
layers, no observation normalization in either task) and its input width is
checked against its observation group before anything runs.
"""

import argparse
import json
import os
import time

from isaaclab.app import AppLauncher

LOGS = "/workspace/policies"  # trained checkpoints (policies/)
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="RobotLab-Isaac-LocoReach-Flat-Unitree-B2W-Z1-v0")
parser.add_argument("--drive_ckpt", default=f"{LOGS}/unitree_b2w_flat/reference_dda1510/model_2600.pt")
parser.add_argument("--arm_ckpt", default=f"{LOGS}/b2w_z1_reach/2026-09-22_16-02-34_v2/model_999.pt")
parser.add_argument("--mode", choices=("keyboard", "eval"), default="keyboard")
parser.add_argument("--num_envs", type=int, default=None, help="default: 1 for keyboard, 8 for eval")
parser.add_argument("--real-time", action="store_true")
parser.add_argument("--out", default=None, help="eval: write results as JSON here")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.mode == "eval":
    args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import robot_lab.tasks  # noqa: E402,F401


def say(*a):
    print("LR|", *a, flush=True)


# ------------------------------------------------------------------ policies
def load_actor(path, device):
    """Rebuild an RSL-RL actor (Linear, ELU, ..., Linear) from its checkpoint."""
    sd = torch.load(path, map_location=device, weights_only=False)["model_state_dict"]
    idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith("actor.") and k.endswith(".weight")})
    layers = []
    for n, i in enumerate(idx):
        w, b = sd[f"actor.{i}.weight"], sd[f"actor.{i}.bias"]
        lin = nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data.copy_(w)
        lin.bias.data.copy_(b)
        layers.append(lin)
        if n < len(idx) - 1:
            layers.append(nn.ELU())
    net = nn.Sequential(*layers).to(device).eval()
    return net, sd[f"actor.{idx[0]}.weight"].shape[1], sd[f"actor.{idx[-1]}.weight"].shape[0]


# ---------------------------------------------------------------- task setup
n_envs = args.num_envs or (1 if args.mode == "keyboard" else 8)
cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=n_envs)
cfg.seed = 42
cfg.episode_length_s = 1.0e4
cfg.terminations.time_out = None

for group in (cfg.observations.policy, cfg.observations.arm):
    group.enable_corruption = False

cmd = cfg.commands.base_velocity   # set by keyboard / the script, never resampled
cmd.heading_command = False
cmd.rel_standing_envs = 0.0
cmd.rel_heading_envs = 0.0
cmd.resampling_time_range = (1.0e6, 1.0e6)
cmd.debug_vis = False
cmd.ranges.heading = None

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
if args.mode == "keyboard":
    cfg.viewer.origin_type = "asset_root"   # camera follows the robot
    cfg.viewer.asset_name = "robot"
    cfg.viewer.eye = (2.5, 2.5, 1.8)
    cfg.viewer.lookat = (0.0, 0.0, 0.3)

env = gym.make(args.task, cfg=cfg)
u = env.unwrapped
dev = u.device
robot = u.scene["robot"]
am = u.action_manager
ee = u.command_manager.get_term("ee_pose")
vel = u.command_manager.get_term("base_velocity")

drive, d_in, d_out = load_actor(args.drive_ckpt, dev)
arm, a_in, a_out = load_actor(args.arm_ckpt, dev)
obs_dims = u.observation_manager.group_obs_dim
assert (d_in,) == tuple(obs_dims["policy"]), f"driving policy expects {d_in} inputs, group has {obs_dims['policy']}"
assert (a_in,) == tuple(obs_dims["arm"]), f"arm policy expects {a_in} inputs, group has {obs_dims['arm']}"

# where each policy's outputs go in the full action vector, by term name
slices, start = {}, 0
for name, width in zip(am.active_terms, am.action_term_dim):
    slices[name] = slice(start, start + width)
    start += width
assert slices["joint_pos"].stop - slices["joint_pos"].start + slices["joint_vel"].stop - slices["joint_vel"].start == d_out
assert slices["arm_action"].stop - slices["arm_action"].start == a_out
say(f"drive {d_in}->{d_out}, arm {a_in}->{a_out}, action terms {list(slices)} ({start} total), {n_envs} env(s)")

arm_on = {"on": True}


def act(obs):
    a = torch.zeros(u.num_envs, am.total_action_dim, device=dev)
    d = drive(obs["policy"])
    n_pos = slices["joint_pos"].stop - slices["joint_pos"].start
    a[:, slices["joint_pos"]] = d[:, :n_pos]
    a[:, slices["joint_vel"]] = d[:, n_pos:]
    if arm_on["on"]:
        a[:, slices["arm_action"]] = arm(obs["arm"])
    # else: arm action 0 = the default (folded) pose, as in the cargo task
    return a


dt = u.step_dt
obs, _ = env.reset()

# ------------------------------------------------------------------ keyboard
if args.mode == "keyboard":
    from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg

    kb = Se2Keyboard(Se2KeyboardCfg(v_x_sensitivity=1.0, v_y_sensitivity=1.0, omega_z_sensitivity=1.0))

    def toggle():
        arm_on["on"] = not arm_on["on"]
        say(f"arm {'REACHING' if arm_on['on'] else 'FOLDED'}")

    kb.add_callback("F", toggle)
    say("arrows drive, Z/X turn, L stop, F toggles the arm (reaching <-> folded)")
    while app.is_running():
        t0 = time.time()
        vel.vel_command_b[:] = torch.tensor(kb.advance(), dtype=torch.float32, device=dev)
        with torch.inference_mode():
            obs, *_ = env.step(act(obs))
        if args.real_time:
            time.sleep(max(0.0, dt - (time.time() - t0)))

# ---------------------------------------------------------------------- eval
else:
    PHASES = [("settle", 3.0, 0.0, 0.0, 0.0), ("forward", 4.0, 1.0, 0.0, 0.0),
              ("brake", 2.0, 0.0, 0.0, 0.0), ("reverse", 3.0, -1.0, 0.0, 0.0),
              ("stop", 2.0, 0.0, 0.0, 0.0), ("turn_L", 3.0, 0.0, 0.0, 1.0),
              ("turn_R", 3.0, 0.0, 0.0, -1.0), ("strafe", 3.0, 0.0, 0.5, 0.0),
              ("arc", 3.0, 0.8, 0.0, 0.6), ("final", 2.0, 0.0, 0.0, 0.0)]
    SETTLE = int(round(1.0 / dt))
    period = cfg.commands.ee_pose.resampling_time_range[0]
    legs = [i for i, n in enumerate(robot.joint_names) if n.endswith(("_hip_joint", "_thigh_joint", "_calf_joint"))]

    summary = {}
    for mode in ("folded", "reaching"):
        arm_on["on"] = mode == "reaching"
        # Reset inside inference mode: the sim's tensors were last written there,
        # and an in-place write from outside it is refused by PyTorch.
        with torch.inference_mode():
            obs, _ = env.reset()
        fell = torch.zeros(u.num_envs, dtype=torch.bool, device=dev)
        rows = []
        for name, seconds, vx, vy, wz in PHASES:
            target = torch.tensor([vx, vy, wz], device=dev)
            n = int(round(seconds / dt))
            err = torch.zeros(3, device=dev)
            got = torch.zeros(3, device=dev)
            counted, pitch_max, roll_max, h, tau = 0, -1e9, 0.0, 0.0, 0.0
            hand, hand_n = 0.0, 0
            for k in range(n):
                vel.vel_command_b[:] = target
                with torch.inference_mode():
                    obs, *_ = env.step(act(obs))
                d = robot.data
                actual = torch.stack([d.root_lin_vel_b[:, 0], d.root_lin_vel_b[:, 1], d.root_ang_vel_b[:, 2]], 1)
                g = d.projected_gravity_b
                fell |= g[:, 2] > -0.5
                pitch_max = max(pitch_max, float(torch.rad2deg(torch.atan2(g[:, 0], -g[:, 2])).max()))
                roll_max = max(roll_max, float(torch.rad2deg(torch.atan2(-g[:, 1], -g[:, 2])).abs().max()))
                h += float(d.root_pos_w[:, 2].mean())
                tau += float(d.applied_torque[:, legs].abs().mean())
                if k >= SETTLE:
                    err += (actual - target).abs().mean(0)
                    got += actual.mean(0)
                    counted += 1
                # hand error only once the arm has had 1.5 s on its current target
                settled = ee.time_left <= (period - 1.5)
                if arm_on["on"] and bool(settled.any()):
                    hand += float(ee.metrics["position_error"][settled].mean())
                    hand_n += 1
            rows.append({"phase": name, "cmd": [vx, vy, wz],
                         "act": (got / counted).tolist(), "err": (err / counted).tolist(),
                         "pitch_max": pitch_max, "roll_max": roll_max, "height": h / n,
                         "leg_torque": tau / n, "hand_err": (hand / hand_n) if hand_n else None})
            r = rows[-1]
            say(f"{mode:8s} {name:8s} act {r['act'][0]:+.2f} {r['act'][1]:+.2f} {r['act'][2]:+.2f} | "
                f"err {r['err'][0]:.3f} {r['err'][1]:.3f} {r['err'][2]:.3f} | pitch {pitch_max:+5.1f} "
                f"roll {roll_max:4.1f} | h {r['height']:.3f} | tau {r['leg_torque']:5.1f} | hand "
                + (f"{r['hand_err'] * 100:5.1f} cm" if r["hand_err"] is not None else "   --"))
        summary[mode] = {"falls": int(fell.sum()), "num_envs": u.num_envs, "phases": rows}
        say(f"{mode}: falls {int(fell.sum())}/{u.num_envs}")
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        say(f"wrote {args.out}")

say("done")
env.close()
app.close()
