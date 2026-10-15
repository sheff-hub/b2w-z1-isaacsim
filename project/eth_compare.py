"""A/B: ETH's SRU B2W locomotion controller vs our model_2600, on OUR robot (B2W + Z1 + gripper).

ETH (eth_controller/, leggedrobotics/sru-robot-deployment, IJRR 2025): the real-robot
deployment policy policy_force_new.onnx -- an LSTM (hidden/cell 256), 60 observations
(base lin vel, base ang vel, projected gravity, [vx vy wz], joint pos - default wrapped to
[-2pi, 2pi), joint vel, last action; joints grouped hip/thigh/calf/foot x FL/FR/RL/RR),
16 actions: leg target = default + 0.5 a, wheel velocity = 5 a, 50 Hz. Run as trained:
its leg/wheel gains (implicit PD 100/3.5, wheels damping 3.0, effort 200/200/320/20) and
its taller stance (thigh 0.4, calf -1.3, spawn 0.75 m). The arm and gripper are ours.

OURS: model_2600 (robot_lab B2W flat, MLP, 57 obs), with the fetch task's settings.

Tests (each from a fresh reset in the open aisle; measured over the second half of the
last phase): following of forward / sideways / diagonal / small commands from rest, the
docking stall case (slowing from (0.3, -0.05) into (0.15, -0.22)), a 180-deg turn in place
(creep), braking from 1 m/s, and 10 s standing with the arm stretched out to the side.

    ./run_script.sh eth_compare.py --device cpu --controller eth  [--video]
    ./run_script.sh eth_compare.py --device cpu --controller ours [--video]
    live window (see live.sh eth):  --gui --real-time [--keyboard]
Output: renders/eth_compare/<controller>.json (+ .mp4 with --video).
"""

import argparse
import json
import math
import os
import time

from isaaclab.app import AppLauncher

LOGS = "/workspace/policies"  # trained checkpoints (policies/)
parser = argparse.ArgumentParser()
parser.add_argument("--controller", choices=("eth", "ours"), required=True)
parser.add_argument("--task", default="RobotLab-Isaac-Fetch-Warehouse-Unitree-B2W-Z1-Gripper-v0")
parser.add_argument("--drive_ckpt", default=f"{LOGS}/unitree_b2w_flat/reference_dda1510/model_2600.pt")
parser.add_argument("--eth_onnx", default="/workspace/eth/policy_force_new.onnx")
parser.add_argument("--tests", default="", help="comma-separated test names (default: all)")
parser.add_argument("--video", action="store_true")
parser.add_argument("--gui", action="store_true", help="Isaac window (needs DISPLAY; see live.sh eth)")
parser.add_argument("--keyboard", action="store_true", help="with --gui: drive with W/S A/D Q/E instead of the tests")
parser.add_argument("--real-time", action="store_true")
parser.add_argument("--out", default="/workspace/renders/eth_compare")
parser.add_argument("--tag", default="", help="suffix of the output files (e.g. a fine-tuned checkpoint's name)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video:
    args.enable_cameras = True
if not args.gui:
    args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import robot_lab.tasks  # noqa: E402,F401


def say(*a):
    print("ETHCMP|", *a, flush=True)


# ---------------------------------------------------------------- ETH contract (eth_controller/src/eth_controller/policy.py)
ETH_JOINTS = [f"{leg}_{part}_joint" for part in ("hip", "thigh", "calf", "foot") for leg in ("FL", "FR", "RL", "RR")]
ETH_DEFAULT = np.asarray([0.0] * 4 + [0.4] * 4 + [-1.3] * 4 + [0.0] * 4, dtype=np.float32)
ETH_STANCE = {"hip": 0.0, "thigh": 0.4, "calf": -1.3, "foot": 0.0}
ETH_SPAWN_Z = 0.75
READY_SIDE = [-1.57, 2.0, -1.35, 0.45, 0.0, -1.45]  # fetch_box.py: the arm out to the right for the side pick


class EthPolicy:
    def __init__(self, path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        ins = {i.name: i.shape for i in self.sess.get_inputs()}
        assert ins == {"obs": [1, 60], "h_in": [1, 1, 256], "c_in": [1, 1, 256]}, ins
        self.reset()

    def reset(self):
        self.h = np.zeros((1, 1, 256), np.float32)
        self.c = np.zeros((1, 1, 256), np.float32)
        self.last = np.zeros(16, np.float32)

    def __call__(self, obs):
        a, self.h, self.c = self.sess.run(["actions", "h_out", "c_out"], {"obs": obs[None].astype(np.float32),
                                                                            "h_in": self.h, "c_in": self.c})
        self.last = a[0].astype(np.float32)
        return self.last


def load_mlp(path, device):
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
    return nn.Sequential(*layers).to(device).eval()


# ---------------------------------------------------------------- env
cfg = parse_env_cfg(args.task, device=args.device or "cuda:0", num_envs=1)
for name in list(vars(cfg.terminations)):
    if not name.startswith("_"):
        setattr(cfg.terminations, name, None)
cfg.events.reset_box = None  # the box is not part of this test
if args.controller == "eth":
    r = cfg.scene.robot
    acts = dict(r.actuators)
    for k in ("hip", "thigh", "calf", "wheel"):
        assert k in acts, f"actuator group {k} missing: {list(acts)}"
    acts["hip"] = ImplicitActuatorCfg(joint_names_expr=[".*_hip_joint"], effort_limit_sim=200.0, velocity_limit_sim=23.0,
                                      stiffness=100.0, damping=3.5, friction=0.0)
    acts["thigh"] = ImplicitActuatorCfg(joint_names_expr=[".*_thigh_joint"], effort_limit_sim=200.0,
                                        velocity_limit_sim=23.0, stiffness=100.0, damping=3.5, friction=0.0)
    acts["calf"] = ImplicitActuatorCfg(joint_names_expr=[".*_calf_joint"], effort_limit_sim=320.0,
                                       velocity_limit_sim=14.0, stiffness=100.0, damping=3.5, friction=0.0)
    acts["wheel"] = ImplicitActuatorCfg(joint_names_expr=[".*_foot_joint"], effort_limit_sim=20.0,
                                        velocity_limit_sim=50.0, stiffness=0.0, damping=3.0, friction=0.0)
    jp = dict(r.init_state.joint_pos)
    for k in list(jp):
        for part, v in ETH_STANCE.items():
            if k.endswith(f"_{part}_joint"):
                del jp[k]
    jp.update({f".*_{part}_joint": v for part, v in ETH_STANCE.items()})
    cfg.scene.robot = r.replace(actuators=acts, soft_joint_pos_limit_factor=0.95,
                                init_state=r.init_state.replace(pos=(r.init_state.pos[0], r.init_state.pos[1], ETH_SPAWN_Z),
                                                                joint_pos=jp))
    # the env's action terms then ARE the ETH mapping: target = default (ETH stance) + 0.5 a, wheel vel = 5 a
    cfg.actions.joint_pos.scale = 0.5
    cfg.actions.joint_vel.scale = 5.0
env = gym.make(args.task, cfg=cfg, render_mode="rgb_array" if args.video else None)
u = env.unwrapped
dev, dt = u.device, u.step_dt
robot, am = u.scene["robot"], u.action_manager
vel = u.command_manager.get_term("base_velocity")
slices, start = {}, 0
for name, width in zip(am.active_terms, am.action_term_dim):
    slices[name] = slice(start, start + width)
    start += width
ARM = robot.find_joints([f"joint{k}" for k in range(1, 7)], preserve_order=True)[0]
ETH_IDS = robot.find_joints(ETH_JOINTS, preserve_order=True)[0]
leg_term, wheel_term = am.get_term("joint_pos"), am.get_term("joint_vel")
# ETH action index -> position within our action vector
ETH_TO_ACT = []
for n in ETH_JOINTS:
    if n in leg_term._joint_names:
        ETH_TO_ACT.append(slices["joint_pos"].start + leg_term._joint_names.index(n))
    else:
        ETH_TO_ACT.append(slices["joint_vel"].start + wheel_term._joint_names.index(n))
say(f"controller {args.controller}, dt {dt * 1000:.0f} ms, leg term joints {leg_term._joint_names}")
if args.controller == "eth":
    eth = EthPolicy(args.eth_onnx)
    import eth_loco  # the shared driver: ETH's ONNX, or a fine-tuned checkpoint via ETH_POLICY
    eth_driver = eth_loco.Driver(u)
    dflt = robot.data.default_joint_pos[0, ETH_IDS].cpu().numpy()
    assert np.allclose(dflt, ETH_DEFAULT, atol=1e-4), f"stance not ETH's: {dflt}"
else:
    mlp = load_mlp(args.drive_ckpt, dev)
hold = {"arm": None}
cmd_now = np.zeros(3, np.float32)


def eth_obs():
    d = robot.data
    q = d.joint_pos[0, ETH_IDS].cpu().numpy() - ETH_DEFAULT
    q = np.mod(q + 2 * np.pi, 4 * np.pi) - 2 * np.pi
    return np.concatenate([d.root_lin_vel_b[0].cpu().numpy(), d.root_ang_vel_b[0].cpu().numpy(),
                           d.projected_gravity_b[0].cpu().numpy(), cmd_now, q,
                           d.joint_vel[0, ETH_IDS].cpu().numpy(), eth.last]).astype(np.float32)


frames = []


def step(obs, cmd, label=""):
    global cmd_now
    cmd_now = np.asarray(cmd, np.float32)
    vel.vel_command_b[:] = torch.tensor(cmd_now, device=dev)
    with torch.inference_mode():
        a = torch.zeros(1, am.total_action_dim, device=dev)
        if args.controller == "eth":
            eth_driver.fill(a, [float(v) for v in cmd_now])
        else:
            d = mlp(obs["policy"])
            n_pos = slices["joint_pos"].stop - slices["joint_pos"].start
            a[:, slices["joint_pos"]], a[:, slices["joint_vel"]] = d[:, :n_pos], d[:, n_pos:]
        a[:, slices["arm_action"]] = hold["arm"]
        a[:, slices["gripper_action"]] = 0.0
        t0 = time.time()
        obs, *_ = env.step(a)
    if args.video and step.k % 2 == 0:
        img = Image.fromarray(u.render())
        ImageDraw.Draw(img).text((12, 10), f"{args.controller.upper()}  {label}  cmd ({cmd[0]:+.2f}, {cmd[1]:+.2f}, "
                                 f"{cmd[2]:+.2f})", fill=(255, 255, 0))
        frames.append(np.asarray(img))
    step.k += 1
    if args.real_time:
        time.sleep(max(0.0, dt - (time.time() - t0)))
    return obs


step.k = 0


def place(x=1.0, y=-1.5, yaw_deg=0.0, arm=None):
    with torch.inference_mode():
        st = robot.data.default_root_state
        st[:, 0], st[:, 1] = x, y
        st[:, 3:7] = torch.tensor([math.cos(math.radians(yaw_deg) / 2), 0, 0, math.sin(math.radians(yaw_deg) / 2)], device=dev)
        obs, _ = env.reset()
    if args.controller == "eth":
        eth.reset()
    hold["arm"] = robot.data.default_joint_pos[0, ARM].clone()
    if arm is not None:
        hold["arm"] = torch.tensor(arm, device=dev)
    return obs


def tilt_deg():
    g = robot.data.projected_gravity_b[0]
    return math.degrees(math.acos(max(-1.0, min(1.0, float(-g[2])))))


def yaw_now():
    q = robot.data.root_quat_w[0]
    return math.atan2(2 * (float(q[0]) * float(q[3]) + float(q[1]) * float(q[2])), 1 - 2 * (float(q[2]) ** 2 + float(q[3]) ** 2))


# name: (phases [(cmd, seconds)], what to report)
TESTS = {
    "fwd_0.3": ([((0.3, 0, 0), 4.0)], "follow"),
    "fwd_1.0": ([((1.0, 0, 0), 5.0)], "follow"),
    "small_fwd_0.1": ([((0.1, 0, 0), 4.0)], "follow"),
    "small_fwd_0.15": ([((0.15, 0, 0), 4.0)], "follow"),
    "back_0.3": ([((-0.3, 0, 0), 4.0)], "follow"),
    "left_0.2": ([((0, 0.2, 0), 4.0)], "follow"),
    "right_0.2": ([((0, -0.2, 0), 4.0)], "follow"),
    "right_0.3": ([((0, -0.3, 0), 4.0)], "follow"),
    "diag_0.15_-0.22": ([((0.15, -0.22, 0), 4.0)], "follow"),
    "slowdown_into_diag": ([((0.3, -0.05, 0), 3.0), ((0.15, -0.22, 0), 5.0)], "follow"),
    "slowdown_into_right": ([((0.3, 0, 0), 3.0), ((0, -0.3, 0), 5.0)], "follow"),
    "yaw_0.5": ([((0, 0, 0.5), 4.0)], "follow"),
    "turn_180": ([((0, 0, 0.6), math.pi / 0.6), ((0, 0, 0), 2.0)], "creep"),
    "brake_from_1.0": ([((1.0, 0, 0), 4.0), ((0, 0, 0), 3.0)], "brake"),
    "hold_arm_out_10s": ([((0, 0, 0), 10.0)], "drift"),
}


def run_test(name):
    phases, kind = TESTS[name]
    obs = place(arm=READY_SIDE if name == "hold_arm_out_10s" else None)
    for _ in range(int(1.5 / dt)):  # settle (ETH's LSTM warms up)
        obs = step(obs, (0, 0, 0), f"{name}: settle")
    p0 = robot.data.root_pos_w[0, :2].clone()
    yaw0 = yaw_now()
    max_tilt, min_h, samples, brake_start = 0.0, 9.0, [], None
    for i_ph, (cmd, sec) in enumerate(phases):
        last = i_ph == len(phases) - 1
        n = int(round(sec / dt))
        if kind == "brake" and last:
            brake_start = robot.data.root_pos_w[0, :2].clone()
        for k in range(n):
            obs = step(obs, cmd, name)
            max_tilt = max(max_tilt, tilt_deg())
            min_h = min(min_h, float(robot.data.root_pos_w[0, 2]))
            if last and k >= n // 2:
                samples.append([float(v) for v in robot.data.root_lin_vel_b[0, :2]] + [float(robot.data.root_ang_vel_b[0, 2])])
    m = np.mean(samples, axis=0)
    cmd = np.asarray(phases[-1][0], dtype=float)
    res = {"test": name, "cmd": cmd.tolist(), "measured": [round(float(v), 3) for v in m],
           "max_tilt_deg": round(max_tilt, 1), "min_height_m": round(min_h, 3), "fell": max_tilt > 45}
    if kind == "follow":
        lin = np.asarray(cmd[:2])
        if np.linalg.norm(lin) > 1e-6:
            res["followed_pct"] = round(100 * float(m[:2] @ lin) / float(lin @ lin), 0)
            res["off_axis"] = round(float(np.linalg.norm(m[:2] - lin * (m[:2] @ lin) / (lin @ lin))), 3)
        else:
            res["followed_pct"] = round(100 * float(m[2]) / float(cmd[2]), 0)
    elif kind == "creep":
        res["turned_deg"] = round(math.degrees(math.atan2(math.sin(yaw_now() - yaw0), math.cos(yaw_now() - yaw0))), 1)
        res["creep_m"] = round(float((robot.data.root_pos_w[0, :2] - p0).norm()), 3)
    elif kind == "brake":
        res["stop_dist_m"] = round(float((robot.data.root_pos_w[0, :2] - brake_start).norm()), 3)
    elif kind == "drift":
        res["drift_m"] = round(float((robot.data.root_pos_w[0, :2] - p0).norm()), 3)
    say(f"{name:22s} {res}")
    return res


os.makedirs(args.out, exist_ok=True)
if args.keyboard:
    from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
    kb = Se2Keyboard(Se2KeyboardCfg(v_x_sensitivity=0.8, v_y_sensitivity=0.3, omega_z_sensitivity=0.6, sim_device="cpu"))
    # W/S A/D Q/E as in eth_controller/src/eth_controller/app.py (Se2Keyboard's own keys are arrows / Z X)
    kb._INPUT_KEY_MAPPING.update({"W": np.asarray([0.8, 0.0, 0.0]), "S": np.asarray([-0.8, 0.0, 0.0]),
                                  "A": np.asarray([0.0, 0.3, 0.0]), "D": np.asarray([0.0, -0.3, 0.0]),
                                  "Q": np.asarray([0.0, 0.0, 0.6]), "E": np.asarray([0.0, 0.0, -0.6])})
    say("keyboard (click the Isaac window first): hold W/S forward/back, A/D left/right, Q/E turn; "
        "close the window to quit")
    obs = place()
    while app.is_running():
        c = kb.advance().cpu().numpy()
        obs = step(obs, (float(c[0]), float(c[1]), float(c[2])), "keyboard")
else:
    names = [t for t in args.tests.split(",") if t] or list(TESTS)
    results = [run_test(n) for n in names]
    path = os.path.join(args.out, f"{args.controller}{'_' + args.tag if args.tag else ''}.json")
    with open(path, "w") as f:
        json.dump({"controller": args.controller, "results": results}, f, indent=1)
    say(f"wrote {path}")
    if args.video and frames:
        import imageio.v2 as imageio
        imageio.mimsave(os.path.join(args.out, f"{args.controller}{'_' + args.tag if args.tag else ''}.mp4"), frames, fps=int(round(1 / (2 * dt))),
                        macro_block_size=8)
        say(f"video {os.path.join(args.out, args.controller + '.mp4')} ({len(frames)} frames)")
env.close()
app.close()
