"""Why does the hand shake on target? A measurement, not a demo.

Every case gets its own environment and the robot is left alone with one fixed
6D target. After it has had time to get there, the last --window seconds are
scored:

  pos / ori    mean hand error over the window -- is it actually on target?
  amp          RMS of the hand position about its own mean, mm. This is the
               shake you SEE: 1-2 mm is invisible, 10 mm is obvious
  hand v       mean hand speed, cm/s (a still hand reads ~0)
  arm rms      arm joint speed RMS, rad/s (same measure as approach_test.py)
  da           RMS step-to-step change of the ARM ACTIONS -- whether the
               policy's own output chatters, as opposed to the motors ringing
  f peak       dominant frequency of the hand motion, Hz. 2-3 Hz would be the
               arm's PD ringing; 10-25 Hz is the policy flipping its output
               every other step (25 Hz is the Nyquist limit at 50 Hz control)
  base v       base planar speed, cm/s -- is the body wandering underneath?

Cases: three reachable points in front x four orientations (one of them, hand
pointing BACK at the robot, deliberately impossible), plus driven cases with
rpy = 0 -- which is exactly what rfm_goto.py sends when you type only "x y z".

Play-time interventions, to separate "the policy chatters" from "the arm rings":
  --ema A       low-pass the arm actions: a = A * a_new + (1 - A) * a_prev
  --arm_kd K    override the arm joints' damping (trained with 4.0)
  --latch       freeze the arm's targets once on target, release when it drifts

    ./run_script.sh shake_diag.py --task <id> --checkpoint <path> [--ema 0.3]
"""

import argparse
import json
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--seconds", type=float, default=24.0)
parser.add_argument("--window", type=float, default=8.0)
parser.add_argument("--ema", type=float, default=1.0)
parser.add_argument("--arm_kd", type=float, default=None)
parser.add_argument("--arm_kp", type=float, default=None)
parser.add_argument("--latch", action="store_true")
parser.add_argument("--stochastic", action="store_true", help="sample actions instead of the mean")
parser.add_argument("--cases", default="all", help="all | near | drive")
parser.add_argument("--out", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

PI = math.pi
ORIENTS = {           # rpy in the robot's starting heading frame (it starts facing +x)
    "fwd": (0.0, 0.0, 0.0),        # = rpy 0, what typing "x y z" sends
    "down": (0.0, 1.0, 0.0),       # pitched down, inside the training range
    "side": (0.4, 0.3, 0.9),       # roll + yaw near the edge of the training range
    "back": (0.0, 0.0, PI),        # hand pointing back at the robot: impossible
}
NEAR = {"mid": (0.75, 0.0, 0.80), "low": (0.70, 0.25, 0.55), "high": (0.70, -0.25, 1.05)}
CASES = []
if args.cases in ("all", "near"):
    for pn, p in NEAR.items():
        for on, o in ORIENTS.items():
            CASES.append((f"{pn}/{on}", p, o))
if args.cases in ("all", "drive"):
    # rpy 0 in the WORLD, like rfm_goto.py -- the approach side is the policy's call
    for name, (dist, bear, z) in {"2m@+90": (2.0, 90, 0.8), "2m@180": (2.0, 180, 0.8),
                                  "2m@0": (2.0, 0, 0.8), "2m@-90/low": (2.0, -90, 0.5)}.items():
        b = math.radians(bear)
        CASES.append((f"drive {name}", (dist * math.cos(b), dist * math.sin(b), z), (0.0, 0.0, 0.0)))
args.num_envs = len(CASES)

app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.utils.math import quat_from_euler_xyz  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import robot_lab.tasks  # noqa: E402,F401

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "project"))  # rfm_helpers
from rfm_helpers import restart_schedule  # noqa: E402


def say(*a):
    print("SHAKE|", *a, flush=True)


cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
cfg.seed = 42
cfg.episode_length_s = 1.0e9
for name in list(vars(cfg.terminations)):
    if not name.startswith("_"):
        setattr(cfg.terminations, name, None)
cfg.observations.policy.enable_corruption = False
cfg.commands.ee_pose.resampling_time_range = (1.0e9, 1.0e9)
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
if args.stochastic:
    _ac = getattr(runner.alg, "policy", None) or runner.alg.actor_critic
    policy = lambda o: _ac.act(o)  # noqa: E731
else:
    policy = runner.get_inference_policy(device=env.unwrapped.device)

u = env.unwrapped
robot = u.scene["robot"]
ee = u.command_manager.get_term("ee_pose")
dt = u.step_dt
N = len(CASES)

# where the arm's 6 numbers sit in the 22-wide action
off, arm_sl = 0, None
for name, term in zip(u.action_manager.active_terms, u.action_manager._terms.values()):
    say(f"action term {name}: [{off}:{off + term.action_dim}]")
    if name == "arm_action":
        arm_sl = slice(off, off + term.action_dim)
    off += term.action_dim
assert arm_sl is not None
arm_jids, arm_names = robot.find_joints(["joint[1-6]"], preserve_order=True)
say(f"arm joints {arm_names}")
say(f"arm gains in sim: kp {robot.data.joint_stiffness[0, arm_jids].tolist()} "
    f"kd {robot.data.joint_damping[0, arm_jids].tolist()}")
if args.arm_kd is not None or args.arm_kp is not None:
    if args.arm_kd is not None:
        robot.write_joint_damping_to_sim(torch.full((N, 6), args.arm_kd, device=u.device), joint_ids=arm_jids)
    if args.arm_kp is not None:
        robot.write_joint_stiffness_to_sim(torch.full((N, 6), args.arm_kp, device=u.device), joint_ids=arm_jids)
    say(f"OVERRIDE arm gains: kp {robot.data.joint_stiffness[0, arm_jids].tolist()} "
        f"kd {robot.data.joint_damping[0, arm_jids].tolist()}")
say(f"interventions: ema {args.ema}  arm_kd {args.arm_kd}  arm_kp {args.arm_kp}  latch {args.latch}  "
    f"stochastic {args.stochastic}")

obs = env.get_observations()
d0 = robot.data
head = d0.heading_w
c, s = torch.cos(head), torch.sin(head)
rel = torch.tensor([cs[1] for cs in CASES], device=u.device)
rpy = torch.tensor([cs[2] for cs in CASES], device=u.device)
ee.pose_command_w[:, 0] = d0.root_pos_w[:, 0] + c * rel[:, 0] - s * rel[:, 1]
ee.pose_command_w[:, 1] = d0.root_pos_w[:, 1] + s * rel[:, 0] + c * rel[:, 1]
ee.pose_command_w[:, 2] = u.scene.env_origins[:, 2] + rel[:, 2]
ee.pose_command_w[:, 3:] = quat_from_euler_xyz(rpy[:, 0], rpy[:, 1], rpy[:, 2] + head)
restart_schedule(ee, robot)

steps = int(round(args.seconds / dt))
win = int(round(args.window / dt))
pos_hist = torch.zeros(win, N, 3, device=u.device)
# the body under the arm: where is the shoulder, and how much of the hand's
# motion is just the body moving it?
sh_idx = robot.find_bodies(["link02"])[0][0]
sh_hist = torch.zeros(win, N, 3, device=u.device)
# hand position in the BODY frame: the arm's own contribution
hb_hist = torch.zeros(win, N, 3, device=u.device)
da_leg2 = torch.zeros(N, device=u.device)
da_whl2 = torch.zeros(N, device=u.device)
prev_full = None
perr = torch.zeros(N, device=u.device)
oerr = torch.zeros(N, device=u.device)
hand_v = torch.zeros(N, device=u.device)
qd2 = torch.zeros(N, device=u.device)
da2 = torch.zeros(N, device=u.device)
base_v = torch.zeros(N, device=u.device)
latched_frac = torch.zeros(N, device=u.device)
prev_arm = None
latched = torch.zeros(N, dtype=torch.bool, device=u.device)
latch_val = torch.zeros(N, 6, device=u.device)

with torch.inference_mode():
    for k in range(steps):
        act = policy(obs).clone()
        a = act[:, arm_sl]
        if prev_arm is not None and args.ema < 1.0:
            a = args.ema * a + (1.0 - args.ema) * prev_arm
        if args.latch:
            pe, oe = ee.metrics["position_error"], ee.metrics["orientation_error"]
            grab = (~latched) & (pe < 0.02) & (oe < 0.15)
            free = latched & ((pe > 0.04) | (oe > 0.3))
            latch_val[grab] = (prev_arm if prev_arm is not None else a)[grab]
            latched = (latched | grab) & ~free
            a = torch.where(latched.unsqueeze(1), latch_val, a)
        act[:, arm_sl] = a
        j = k - (steps - win)
        if j >= 0 and prev_arm is not None:
            da2 += ((a - prev_arm) ** 2).sum(dim=1)
        prev_arm = a.clone()
        if j >= 0 and prev_full is not None:
            da_leg2 += ((act[:, 0:12] - prev_full[:, 0:12]) ** 2).sum(dim=1)
            da_whl2 += ((act[:, 12:16] - prev_full[:, 12:16]) ** 2).sum(dim=1)
        prev_full = act.clone()
        obs, *_ = env.step(act)
        if j >= 0:
            d = robot.data
            pos_hist[j] = d.body_pos_w[:, ee.body_idx]
            sh_hist[j] = d.body_pos_w[:, sh_idx]
            from isaaclab.utils.math import quat_apply_inverse
            hb_hist[j] = quat_apply_inverse(d.root_quat_w, d.body_pos_w[:, ee.body_idx] - d.root_pos_w)
            perr += ee.metrics["position_error"]
            oerr += ee.metrics["orientation_error"]
            hand_v += d.body_lin_vel_w[:, ee.body_idx].norm(dim=1)
            qd2 += (d.joint_vel[:, arm_jids] ** 2).sum(dim=1)
            base_v += d.root_lin_vel_b[:, :2].norm(dim=1)
            latched_frac += latched.float()

perr /= win; oerr /= win; hand_v /= win; base_v /= win; latched_frac /= win
arm_rms = torch.sqrt(qd2 / win)
da = torch.sqrt(da2 / win)
centred = pos_hist - pos_hist.mean(dim=0, keepdim=True)
amp = torch.sqrt((centred ** 2).sum(dim=2).mean(dim=0))           # m


def _amp(h):
    c_ = h - h.mean(dim=0, keepdim=True)
    return torch.sqrt((c_ ** 2).sum(dim=2).mean(dim=0))


amp_sh = _amp(sh_hist)      # how much the shoulder (the arm's base) moves
amp_arm = _amp(hb_hist)     # hand motion relative to the body: the arm's own share
da_leg = torch.sqrt(da_leg2 / win)
da_whl = torch.sqrt(da_whl2 / win)
spec = torch.fft.rfft(centred, dim=0).abs().pow(2).sum(dim=2)       # (freq, N)
freqs = torch.fft.rfftfreq(win, d=dt).to(u.device)
spec[0] = 0.0
f_peak = freqs[spec.argmax(dim=0)]
# share of the motion's energy above 5 Hz: fast chatter vs slow sway
hi_share = spec[freqs > 5.0].sum(dim=0) / spec.sum(dim=0).clamp(min=1e-12)

say("")
say(f"{'case':>18s} {'pos':>7s} {'ori':>6s} {'amp':>7s} {'hand v':>8s} {'arm rms':>8s} "
    f"{'da':>6s} {'f peak':>7s} {'>5Hz':>5s} {'base v':>7s} {'shoulder':>8s} {'arm-own':>8s} {'da leg':>6s} {'da whl':>6s}"
    + ("  latched" if args.latch else ""))
rows = []
for i, (name, p, o) in enumerate(CASES):
    say(f"{name:>18s} {float(perr[i]) * 100:6.1f}c {float(oerr[i]):6.2f} {float(amp[i]) * 1000:6.1f}m "
        f"{float(hand_v[i]) * 100:7.1f}c {float(arm_rms[i]):8.2f} {float(da[i]):6.3f} "
        f"{float(f_peak[i]):6.1f}H {float(hi_share[i]) * 100:4.0f}% {float(base_v[i]) * 100:6.1f}c "
        f"{float(amp_sh[i]) * 1000:7.1f}m {float(amp_arm[i]) * 1000:7.1f}m {float(da_leg[i]):6.3f} {float(da_whl[i]):6.3f}"
        + (f"  {float(latched_frac[i]) * 100:5.0f}%" if args.latch else ""))
    rows.append({"case": name, "pos_m": float(perr[i]), "ori_rad": float(oerr[i]),
                 "amp_m": float(amp[i]), "hand_v": float(hand_v[i]), "arm_rms": float(arm_rms[i]),
                 "da": float(da[i]), "f_peak": float(f_peak[i]), "hi_share": float(hi_share[i]),
                 "base_v": float(base_v[i]), "latched": float(latched_frac[i]),
                 "amp_shoulder": float(amp_sh[i]), "amp_arm_own": float(amp_arm[i]),
                 "da_leg": float(da_leg[i]), "da_wheel": float(da_whl[i])})

on = perr < 0.05
say("")
say(f"cases on target (<5 cm): {int(on.sum())}/{N}")
if on.any():
    say(f"ON-TARGET medians: amp {float(amp[on].median()) * 1000:.1f} mm | hand v "
        f"{float(hand_v[on].median()) * 100:.1f} cm/s | arm rms {float(arm_rms[on].median()):.2f} rad/s | "
        f"da {float(da[on].median()):.3f} | f peak {float(f_peak[on].median()):.1f} Hz | "
        f">5Hz {float(hi_share[on].median()) * 100:.0f}% | ori {float(oerr[on].median()):.2f} rad")
    say(f"ON-TARGET split: shoulder moves {float(amp_sh[on].median()) * 1000:.1f} mm, hand relative to body "
        f"{float(amp_arm[on].median()) * 1000:.1f} mm | da legs {float(da_leg[on].median()):.3f} "
        f"wheels {float(da_whl[on].median()):.3f}")
if args.out:
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"args": vars(args) | {"cases": args.cases}, "rows": rows}, f, indent=2, default=str)
    say(f"wrote {args.out}")
say("done")
env.close()
app.close()
