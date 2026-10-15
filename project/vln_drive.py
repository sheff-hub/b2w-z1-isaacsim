"""Drive the B2W + Z1 with DualVLN from a language instruction (Stage-2 VLN test).

The DualVLN driving itself (cameras, client, path following, bumper) is in vln_nav.py,
shared with fetch_box.py --vln. This script runs it: one instruction, a batch of starts
(--starts/--repeats, with a success summary), or live (--interactive: type instructions;
see live.sh). The arm stays stowed.

    # 1) server:  Isaac/vln/run_dualvln_server.sh            (full precision; --quant 8bit: 6 GB less)
    # 2) robot:
    ISAAC_IMAGE=isaac-lab:2.3.2 ./run_script.sh vln_drive.py --device cpu \\
        --instruction "walk to the wooden table and stop in front of it"

Output (renders/vln/<run>/): log.json (every request/answer and the path),
frames/ (level | look-down image per request, trajectory in green), goals/ (each
System 2 pixel goal, red, on the image it was picked in), camera.mp4, wide.mp4.
"""

import argparse
import json
import math
import os
import queue
import time

from isaaclab.app import AppLauncher

LOGS = "/workspace/policies"  # trained checkpoints (policies/)
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="RobotLab-Isaac-Fetch-Warehouse-Unitree-B2W-Z1-Gripper-v0")
parser.add_argument("--drive_ckpt", default=f"{LOGS}/unitree_b2w_flat/reference_dda1510/model_2600.pt")
parser.add_argument("--instruction", default="", help="one-shot run; omit with --interactive")
parser.add_argument("--interactive", action="store_true",
                    help="live: wait for instructions typed in the terminal or written to --watch")
parser.add_argument("--watch", default="/workspace/project/instruction.txt")
parser.add_argument("--real-time", action="store_true", help="pace the simulation to the wall clock (for watching)")
parser.add_argument("--async_s2", action="store_true",
                    help="keep driving the last path while System 2 thinks, as on a real robot (implies "
                         "--real-time). Default: the simulation waits for each answer, which costs no simulated "
                         "time; async lets System 2 plan on 2-3 s old images and reached the table 0/5 vs 4/5")
parser.add_argument("--no_smooth", action="store_true", help="send the follower's commands unfiltered")
parser.add_argument("--no_bumper", action="store_true", help="no depth-based stop in front of obstacles")
parser.add_argument("--gui", action="store_true", help="open the Isaac Sim window (needs DISPLAY; see live.sh)")
parser.add_argument("--server", default="http://127.0.0.1:5801")
parser.add_argument("--loco", choices=("ours", "eth"), default="eth",
                    help="base controller: model_2600, or ETH's SRU policy (eth_loco.py; 15 cm taller stance)")
parser.add_argument("--max_time", type=float, default=60.0, help="s of simulated time")
parser.add_argument("--vln_period", type=float, default=0.4, help="s of simulated time between requests")
parser.add_argument("--start", default="1.0,-1.5,0", help="robot start x,y,yaw(deg)")
parser.add_argument("--starts", default="", help="batch: \"x,y,yaw; x,y,yaw; ...\" -- one episode per start "
                    "(x --repeats), in one simulator; writes <out>/<tag>_summary.json")
parser.add_argument("--repeats", type=int, default=1)
parser.add_argument("--out", default="/workspace/renders/vln")
parser.add_argument("--tag", default="")
parser.add_argument("--no_video", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.sync = not args.async_s2
if args.async_s2:
    args.real_time = True  # asynchronous planning: model latency must cost robot time, as on hardware
if not args.interactive and not args.instruction:
    parser.error("--instruction is required unless --interactive")
LIVE = args.gui or (args.livestream or 0) > 0  # livestream's default is -1
args.enable_cameras = True
if not LIVE:
    args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import robot_lab.tasks  # noqa: E402,F401
import vln_nav  # noqa: E402

if args.loco == "eth":
    import eth_loco  # noqa: E402
    vln_nav.MIN_SPEED, vln_nav.MIN_TURN = eth_loco.MIN_SPEED, eth_loco.MIN_TURN


def say(*a):
    print("VLN|", *a, flush=True)


def load_actor(path, device):
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


# ---------------------------------------------------------------- task + camera
cfg = parse_env_cfg(args.task, device=args.device or "cuda:0", num_envs=1)
for name in list(vars(cfg.terminations)):
    if not name.startswith("_"):
        setattr(cfg.terminations, name, None)
sx, sy, syaw = (float(v) for v in args.start.split(","))
cfg.scene.robot.init_state.pos = (sx, sy, cfg.scene.robot.init_state.pos[2])
cfg.scene.robot.init_state.rot = (math.cos(math.radians(syaw) / 2), 0.0, 0.0, math.sin(math.radians(syaw) / 2))
vln_nav.add_cameras(cfg)
if args.loco == "eth":
    eth_loco.patch_cfg(cfg)
cfg.viewer.origin_type = "world"
cfg.viewer.eye = (3.6, -5.2, 7.0)
cfg.viewer.lookat = (4.0, 1.0, 0.0)

RECORD = not (args.no_video or LIVE or args.interactive)  # live: the window is the video
env = gym.make(args.task, cfg=cfg, render_mode="rgb_array" if RECORD else None)
u = env.unwrapped
dev = u.device
dt = u.step_dt
robot = u.scene["robot"]
am = u.action_manager
vel = u.command_manager.get_term("base_velocity")
drive = load_actor(args.drive_ckpt, dev) if args.loco == "ours" else None
slices, start = {}, 0
for name, width in zip(am.active_terms, am.action_term_dim):
    slices[name] = slice(start, start + width)
    start += width
ARM = robot.find_joints([f"joint{k}" for k in range(1, 7)], preserve_order=True)[0]
hold_arm = {"q": None}
eth_driver = eth_loco.Driver(u) if args.loco == "eth" else None


def act(obs):
    a = torch.zeros(1, am.total_action_dim, device=dev)
    if args.loco == "eth":
        eth_driver.fill(a, vel.vel_command_b[0].tolist())
    else:
        d = drive(obs["policy"])
        n_pos = slices["joint_pos"].stop - slices["joint_pos"].start
        a[:, slices["joint_pos"]] = d[:, :n_pos]
        a[:, slices["joint_vel"]] = d[:, n_pos:]
    a[:, slices["arm_action"]] = hold_arm["q"]
    a[:, slices["gripper_action"]] = 0.0
    return a


# ---------------------------------------------------------------- run
commands = queue.Queue()  # interactive: instructions and control words, from stdin and --watch


def sim_step(cmd):
    global obs
    t0 = time.time()
    vel.vel_command_b[:] = torch.tensor(cmd, device=dev)
    with torch.inference_mode():
        obs, *_ = env.step(act(obs))
    if args.real_time:
        time.sleep(max(0.0, dt - (time.time() - t0)))


obs, _ = env.reset()
hold_arm["q"] = robot.data.joint_pos[0, ARM].clone()
for _ in range(int(1.0 / dt)):  # settle
    sim_step((0.0, 0.0, 0.0))

wide = []


def on_step(k):
    if RECORD and k % 2 == 0:
        wide.append(u.render())


nav = vln_nav.Navigator(robot, u.scene["vln_cam"], u.scene["vln_cam_down"], dt, sim_step, args.server,
                        sync=args.sync, smooth=not args.no_smooth, bumper=not args.no_bumper,
                        vln_period=args.vln_period, say=say,
                        should_stop=lambda: args.interactive and not commands.empty(), on_step=on_step)


def episode(instruction):
    """One instruction, until DualVLN says STOP, --max_time runs out, or (interactive) a new command arrives."""
    out = os.path.join(args.out, f"{time.strftime('%m%d-%H%M%S')}{'_' + args.tag if args.tag else ''}")
    wide.clear()
    log = nav.run(instruction, out, args.max_time)
    camvid = log.pop("camvid")
    if RECORD:
        import imageio.v2 as imageio
        if wide:
            imageio.mimsave(os.path.join(out, "wide.mp4"), wide, fps=int(round(1 / (2 * dt))), macro_block_size=8)
        if camvid:
            imageio.mimsave(os.path.join(out, "camera.mp4"), camvid, fps=max(1, int(round(1 / args.vln_period))),
                            macro_block_size=8)
    return log


def place(x, y, yaw_deg):
    """Reset the episode with the robot at (x, y, yaw): the task resets the root to its default state."""
    global obs
    with torch.inference_mode():
        st = robot.data.default_root_state
        st[:, 0], st[:, 1] = x, y
        st[:, 3:7] = torch.tensor([math.cos(math.radians(yaw_deg) / 2), 0.0, 0.0, math.sin(math.radians(yaw_deg) / 2)],
                                  device=dev)
        obs, _ = env.reset()
    for _ in range(int(1.0 / dt)):  # settle
        sim_step((0.0, 0.0, 0.0))


def batch():
    starts = [tuple(float(v) for v in st.split(",")) for st in args.starts.split(";") if st.strip()]
    rows = []
    for rep_i in range(args.repeats):
        for st in starts:
            place(*st)
            L = episode(args.instruction)
            edge = L["table_edge_m"]
            rows.append({"start": list(st), "repeat": rep_i, "reason": L["reason"], "edge_m": edge,
                         "success": L["reason"] == "STOP" and edge <= vln_nav.AT_TABLE,
                         "arrived": L["reason"] in ("STOP", "BLOCKED") and edge <= vln_nav.AT_TABLE,
                         **{k: L.get(k) for k in ("sim_time_s", "wall_time_s", "frozen_wall_s", "path_len_m", "stops",
                                                  "rms_acc", "rms_ang_acc", "mean_speed", "moving_frac")}})
            say(f"[{len(rows)}/{len(starts) * args.repeats}] success {rows[-1]['success']} "
                f"(edge {edge:.2f} m, {L['reason']})")
    n = len(rows)
    agg = {"episodes": n, "success": sum(r["success"] for r in rows), "arrived": sum(r["arrived"] for r in rows),
           "blocked": sum(r["reason"] == "BLOCKED" for r in rows),
           **{k: round(float(np.mean([r[k] for r in rows if r[k] is not None])), 3)
              for k in ("edge_m", "sim_time_s", "frozen_wall_s", "stops", "rms_acc", "rms_ang_acc", "mean_speed")}}
    summary = {"instruction": args.instruction, "mode": "sync" if args.sync else "async", "smoothing": not args.no_smooth,
               "bumper": not args.no_bumper, "aggregate": agg, "episodes": rows}
    path = os.path.join(args.out, f"{args.tag or 'batch'}_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=1)
    say(f"SUMMARY {agg['success']}/{n} stopped at the table ({agg['arrived']}/{n} incl. bumper-held) | "
        + ", ".join(f"{k} {v}" for k, v in agg.items() if k not in ("episodes", "success", "arrived")) + f" -> {path}")


if args.starts:
    batch()
elif not args.interactive:
    episode(args.instruction)
else:
    vln_nav.start_command_readers(commands, args.watch, say)
    say("READY. Type an instruction, e.g.  walk to the wooden table and stop in front of it")
    say("      stop = halt   home = back to the start   q = quit")
    say(f"      (or from another terminal:  ./live.sh say \"...\"  -> {args.watch})")
    while app.is_running():
        if commands.empty():
            sim_step((0.0, 0.0, 0.0))  # stand still, waiting
            continue
        cmd = commands.get()
        if cmd[0] == "quit":
            break
        if cmd[0] == "home":
            with torch.inference_mode():  # reset() outside it raises "Inplace update to inference tensor"
                obs, _ = env.reset()
            say("back at the start")
        elif cmd[0] == "stop":
            say("stopped, waiting")
        else:
            episode(vln_nav.normalise_instruction(cmd[1]))
            say("waiting for the next instruction")
env.close()
app.close()
