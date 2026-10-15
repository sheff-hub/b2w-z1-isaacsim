"""Give the robot a 6D pose anywhere in the room. It decides how to get there.

This is what the RFM policy (wb16) buys over everything before it. There is no
velocity command and no state machine here -- no "drive until 0.45 m, then
switch to reaching", which is what project/goto.py had to do. The policy is handed
one 6D end-effector target and works out for itself whether that means turning
round, driving three metres, crouching, or just extending the arm.

    PLAY_SCRIPT=/workspace/project/rfm_goto.py ./run_train.sh --play --stream \
        --task RobotLab-Isaac-RFM-Unitree-B2W-Z1-v0 --checkpoint <path>

Type, or write into Isaac/target.txt from another terminal:

    2.5 0 0.3               position only, hand level
    -2 1 0.8  0 0.6 0       position + roll/pitch/yaw, radians
    stop                    hold still where it is
    home                    put the robot back at the origin
    q                       quit

Reported every second: how far the target still is, the position and orientation
error, the body height and tilt. Watch the phase column -- it is the policy's own
sense of "this is still a driving problem" (D near 1) shading into "this is a
reaching problem" (D near 0), which is the mechanism the whole task is built on.
"""

import argparse
import math
import os
import queue
import sys
import threading
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--target", default=None, help="initial target 'x,y,z[,r,p,y]'")
parser.add_argument("--watch", default="/workspace/project/target.txt")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--report_every", type=float, default=1.0)
parser.add_argument("--real-time", action="store_true", default=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.utils.math import quat_from_euler_xyz  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import robot_lab.tasks  # noqa: E402,F401

from free_camera import CameraToggle  # noqa: E402
from rfm_helpers import restart_schedule  # noqa: E402


def say(*a):
    print("RFM|", *a, flush=True)


commands: "queue.Queue[tuple]" = queue.Queue()


def parse_line(text):
    text = text.strip().lower().replace(",", " ")
    if not text:
        return None
    if text in ("q", "quit", "exit"):
        return ("quit",)
    if text in ("home", "reset", "r"):
        return ("home",)
    if text in ("s", "stop", "hold"):
        return ("stop",)
    try:
        n = [float(v) for v in text.split()]
    except ValueError:
        say(f"cannot read '{text}' -- expected 'x y z' or 'x y z roll pitch yaw'")
        return None
    if len(n) == 3:
        n += [0.0, 0.0, 0.0]
    if len(n) != 6:
        say("expected 3 numbers (position) or 6 (position + rpy)")
        return None
    return ("goto", *n)


def stdin_reader():
    for line in sys.stdin:
        c = parse_line(line)
        if c:
            commands.put(c)


def file_watcher(path):
    last = os.path.getmtime(path) if os.path.exists(path) else None
    while True:
        try:
            if os.path.exists(path):
                stamp = os.path.getmtime(path)
                if stamp != last:
                    last = stamp
                    lines = open(path).read().strip().splitlines()
                    if lines:
                        c = parse_line(lines[-1])
                        if c:
                            say(f"[file] {lines[-1].strip()}")
                            commands.put(c)
        except Exception as exc:
            say(f"watch error: {exc}")
        time.sleep(0.1)


cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
cfg.seed = 42
cfg.episode_length_s = 1.0e9
# No terminations at all in a viewer. During training a tip-over or a low body
# ends the episode and RESETS the robot -- which on screen looks like it
# teleporting home and setting off for a random point. If it falls here, you
# should see it fall; type `home` to start again.
for _name in list(vars(cfg.terminations)):
    if not _name.startswith("_"):
        setattr(cfg.terminations, _name, None)
cfg.observations.policy.enable_corruption = False
# No training disturbances in a viewer: the random shove (body velocity set to up
# to +-0.5 m/s every 10-15 s) and the random constant force/torque on the body
# (up to 30 N / 10 N m, held all episode). They make the policy robust in training;
# here they only add jolts and lean that are not the policy's own behaviour.
for _name in ("randomize_push_robot", "randomize_apply_external_force_torque"):
    if hasattr(cfg.events, _name):
        setattr(cfg.events, _name, None)
# Hide the inherited velocity-command arrows (green = commanded, blue = actual base
# velocity, 0.5 m above the body). RFM has no velocity command -- it is fixed at
# zero -- and at rest the blue arrow shrinks to a disk spinning with sim noise.
if getattr(cfg.commands, "base_velocity", None) is not None:
    cfg.commands.base_velocity.debug_vis = False
cfg.commands.ee_pose.resampling_time_range = (1.0e9, 1.0e9)
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
rfm = u.reward_manager.get_term_cfg("rfm").func if "rfm" in u.reward_manager.active_terms else None
dt = u.step_dt
cam = CameraToggle(u, following=True)

# The target as a red sphere, as in goto.py.
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402

TARGET_MARKER = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/RfmTarget", markers={
    "target": sim_utils.SphereCfg(radius=0.05, visual_material=sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.95, 0.1, 0.1), emissive_color=(0.4, 0.0, 0.0)))}))
TARGET_MARKER.set_visibility(False)


def hold_here():
    """Make the CURRENT hand pose the target, fixed in the world.

    These policies have no velocity command: whatever sits in the hand-target
    buffer IS the instruction. A reset samples a random target up to 3 m away,
    so without this the robot sets off for a point nobody asked for -- on launch
    and after every `home`. Fixed in the world, NOT re-placed relative to the
    body each step: a target that rides along with the robot is a carrot on a
    stick, and a world-target policy chases it with its whole body.
    """
    with torch.inference_mode():
        ee.pose_command_w[:, :3] = robot.data.body_pos_w[:, ee.body_idx]
        ee.pose_command_w[:, 3:] = robot.data.body_quat_w[:, ee.body_idx]
    restart_schedule(ee, robot)
    TARGET_MARKER.set_visibility(False)


hold_here()
say(f"policy {os.path.basename(args.checkpoint)}")
say("type:  x y z  |  x y z roll pitch yaw  |  stop  |  home  |  q")

threading.Thread(target=stdin_reader, daemon=True).start()
if args.watch:
    threading.Thread(target=file_watcher, args=(args.watch,), daemon=True).start()
    say(f"watching {args.watch}")

if args.target:
    commands.put(parse_line(args.target.replace(",", " ")))

obs = env.get_observations()
step, last_report, quit_now, have_target = 0, 0.0, False, False

while app.is_running() and not quit_now:
    t0 = time.time()
    while not commands.empty():
        cmd = commands.get()
        if cmd[0] == "quit":
            quit_now = True
        elif cmd[0] == "stop":
            hold_here()
            have_target = False
            say("holding still where it is")
        elif cmd[0] == "home":
            with torch.inference_mode():
                obs = env.reset()[0]
            hold_here()
            have_target = False
            say("reset to the origin, holding still")
        elif cmd[0] == "goto":
            x, y, z, rr, pp, yy = cmd[1:]
            pos = torch.tensor([x, y, z], device=u.device).unsqueeze(0) + u.scene.env_origins
            quat = quat_from_euler_xyz(
                torch.tensor([rr], device=u.device),
                torch.tensor([pp], device=u.device),
                torch.tensor([yy], device=u.device),
            )
            # Written straight into the command buffer, and NOT re-parked or
            # re-aimed every step: the policy owns the approach from here.
            ee.pose_command_w[:, :3] = pos
            ee.pose_command_w[:, 3:] = quat
            restart_schedule(ee, robot)
            TARGET_MARKER.set_visibility(True)
            TARGET_MARKER.visualize(translations=pos)
            have_target = True
            d0 = float(torch.norm(pos[:, :2] - robot.data.root_pos_w[:, :2], dim=1).mean())
            say(f"target ({x:.2f}, {y:.2f}, {z:.2f}) rpy ({rr:.2f}, {pp:.2f}, {yy:.2f}) "
                f"-- {d0:.2f} m away")

    with torch.inference_mode():
        obs, _, _, _ = env.step(policy(obs))
    step += 1

    now = step * dt
    if have_target and now - last_report >= args.report_every:
        last_report = now
        d = robot.data
        gap = float(torch.norm(ee.pose_command_w[:, :2] - d.root_pos_w[:, :2], dim=1).mean())
        perr = float(ee.metrics["position_error"].mean())
        oerr = float(ee.metrics["orientation_error"].mean())
        h = float(d.root_pos_w[:, 2].mean() - u.scene.env_origins[:, 2].mean())
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(-d.projected_gravity_b[:, 2].mean())))))
        ph = float(rfm.last["D"].mean()) if rfm is not None and hasattr(rfm, "last") else float("nan")
        mode = "driving" if ph > 0.5 else "reaching"
        say(f"{mode:8s} phase {ph:4.2f} | base {gap:5.2f} m from it | hand {perr * 100:5.1f} cm "
            f"| ori {oerr:4.2f} rad | h {h:.2f} tilt {tilt:3.0f} deg")

    if args.real_time:
        time.sleep(max(0.0, dt - (time.time() - t0)))

say("done")
cam.close()
env.close()
app.close()
