"""Type a coordinate, watch the robot go there and put its hand on it.

The interactive version of demo_fetch.py: instead of one target fixed on the
command line, targets arrive WHILE the simulation runs, so you can try a dozen
points in one session without paying the ~90 s Isaac start-up each time.

    PLAY_SCRIPT=/workspace/project/goto.py ./run_train.sh --play --stream \
        --task <id> --checkpoint <path>

Then just type coordinates at the prompt:

    2.5 0 0.3       drive to (2.5, 0) and put the hand at z = 0.30  (it crouches)
    home            put the robot back at the origin (it drifts while idle)
    2 1.5 1.1       a high one                                     (it stands tall)
    0 0 0.6         come back to the origin
    stop            stay put, hold the current pose
    q               quit

TWO WAYS IN, because a livestreamed run is not always attached to a terminal:

  * the PROMPT, when the container has a TTY (run it yourself with --stream);
  * a FILE, always: write a line into Isaac/target.txt from any other terminal
        echo "2.5 0 0.3" > project/target.txt
    and it is picked up within a tenth of a second. The file is the same
    single-writer, last-value-wins contract a ROS 2 topic gives, which is what
    this becomes in Phase 5 -- the state machine below does not care where the
    numbers came from.

WHICH POLICY TO RUN IT WITH
===========================
Use **wb9** (b2w_z1_wholebody_worldtarget/*_wb9/model_7700.pt). Measured on the
"fwd 2.0 0.3" test: drives 2.0 -> 0.49 m, then crouches to 0.37 m and holds the
hand at 0.7-1.3 cm.

NOT wb15/wb12. They reach better standing still (and keep the arm out of the
lidar mast), but they turn at only ~30 % of the commanded rate, and this control
law needs the robot to face its goal: wb15 gets within ~1.3 m, drifts off
heading, and then spins its wheels without closing. That is the documented
posture-for-turning trade (PHASE4_WORKSPACE.md section 9), not a fault here.

HOW IT DRIVES, AND WHY IT SWITCHES MODE
=======================================
A world-frame target further away than the arm can reach is a NAVIGATION problem;
once the robot is standing next to it, it is a REACHING problem. The policy can do
both but not at once -- these policies are trained to +/-0.3 m/s, and a target is
left behind within a second at full speed (wb10 learned to ignore the arm
entirely because of exactly that). So:

    DRIVE   far away  -> turn toward it, drive, arm parked at a neutral pose
    REACH   within --standoff -> stop the base, hand goes to the world point
    HOLD    reached, or asked to stop -> keep the pose until a new target

which is the two-phase split demo_fetch.py already proved, made continuous.
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
parser.add_argument("--target", default=None, help="initial target 'x,y,z' (optional)")
parser.add_argument("--watch", default="/workspace/project/target.txt",
                    help="file polled for new targets; empty string disables")
parser.add_argument("--standoff", type=float, default=0.45,
                    help="stop the base this far from the target and start reaching, m")
parser.add_argument("--vmax", type=float, default=0.3, help="max drive speed (trained range)")
parser.add_argument("--drive_arm", choices=("park", "far"), default="park",
                    help="what the hand is told to do WHILE DRIVING. 'park' holds it at a "
                         "comfortable pose that travels with the robot (in distribution: the "
                         "policy only ever saw targets 0.25-0.65 m away). 'far' points it at the "
                         "goal itself, which is far outside that range -- the arm stretches to "
                         "full extension and can topple the robot.")
parser.add_argument("--turn_first", type=float, default=0.25,
                    help="bearing (rad) beyond which the robot turns on the spot instead of "
                         "driving. Tight on purpose: these policies turn at only ~30 %% of the "
                         "commanded rate, so driving while misaligned arcs around the goal "
                         "faster than the yaw command can correct it.")
parser.add_argument("--wmax", type=float, default=0.5,
                    help="max commanded yaw rate. These policies are trained to +/-0.5 rad/s and "
                         "track only ~30 %% of it, so asking for more is extrapolation, not speed.")
parser.add_argument("--tracking", type=float, default=0.8,
                    help="measured fraction of a commanded speed actually achieved")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--report_every", type=float, default=1.0)
parser.add_argument("--real-time", action="store_true", default=True)
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
    print("GOTO|", *a, flush=True)


# --------------------------------------------------------------------- input
commands: "queue.Queue[tuple]" = queue.Queue()


def parse_line(text):
    """'2.5 0 0.3' or '2.5,0,0.3' -> ('goto', x, y, z); also 'stop' and 'q'."""
    text = text.strip().lower().replace(",", " ")
    if not text:
        return None
    if text in ("q", "quit", "exit"):
        return ("quit",)
    if text in ("s", "stop", "hold"):
        return ("stop",)
    if text in ("home", "reset", "r"):
        return ("home",)
    if text.startswith("fwd"):
        # "fwd 1.5 [z]" -- a target that many metres straight ahead of the robot,
        # which needs no knowledge of where the robot currently is.
        bits = text.split()[1:]
        try:
            d = float(bits[0]) if bits else 1.5
            z = float(bits[1]) if len(bits) > 1 else 0.3
        except ValueError:
            say("usage: fwd <metres> [height]")
            return None
        return ("fwd", d, z)
    parts = text.split()
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        say(f"cannot read '{text}' -- expected 'x y z', or stop, or q")
        return None
    if len(nums) == 2:
        nums.append(0.6)          # a sensible default height: about shoulder level
    if len(nums) != 3:
        say(f"expected 2 or 3 numbers, got {len(nums)}")
        return None
    return ("goto", *nums)


def stdin_reader():
    """Read typed lines without blocking the simulation loop."""
    for line in sys.stdin:
        cmd = parse_line(line)
        if cmd:
            commands.put(cmd)
    say("stdin closed -- the file interface still works")


def file_watcher(path):
    """Poll a file; every time its contents CHANGE, treat it as a new command.

    Whatever is already in the file when we start is ignored on purpose. It is
    left over from the previous session, and acting on it means a stale 'q'
    quits the simulator before you have typed anything -- which is exactly what
    happened the first time this was used.
    """
    last = os.path.getmtime(path) if os.path.exists(path) else None
    if last is not None:
        try:
            with open(path) as f:
                stale = f.read().strip().splitlines()
            if stale:
                say(f"ignoring what was already in {os.path.basename(path)}: '{stale[-1].strip()}'")
        except Exception:
            pass
    while True:
        try:
            if os.path.exists(path):
                stamp = os.path.getmtime(path)
                if stamp != last:
                    last = stamp
                    with open(path) as f:
                        text = f.read().strip().splitlines()
                    if text:
                        cmd = parse_line(text[-1])
                        if cmd:
                            say(f"[file] {text[-1].strip()}")
                            commands.put(cmd)
        except Exception as exc:
            say(f"watch error: {exc}")
        time.sleep(0.1)


# ----------------------------------------------------------------------- env
cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
cfg.seed = 42
cfg.episode_length_s = 1.0e9
if hasattr(cfg.terminations, "time_out"):
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
cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
cfg.commands.base_velocity.ranges.heading = None
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
vel = u.command_manager.get_term("base_velocity")
ee = u.command_manager.get_term("ee_pose")
dt = u.step_dt
cam = CameraToggle(u, following=True)
wheel_ids, _ = robot.find_joints([".*_foot_joint"])
IDENT = torch.tensor([1.0, 0.0, 0.0, 0.0], device=u.device)

# the current target as a red sphere (rendered in the window / stream; hidden when holding)
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402

TARGET_MARKER = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/GotoTarget", markers={
    "target": sim_utils.SphereCfg(radius=0.05, visual_material=sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.95, 0.1, 0.1), emissive_color=(0.4, 0.0, 0.0)))}))
TARGET_MARKER.set_visibility(False)


def show_target(t):
    if t is None:
        TARGET_MARKER.set_visibility(False)
    else:
        TARGET_MARKER.set_visibility(True)
        TARGET_MARKER.visualize(translations=t[:1])
world_frame = type(ee).__name__ == "WorldPoseCommand"
if not world_frame:
    say("WARNING: this task's hand target is BODY-frame, so a world coordinate "
        "cannot be expressed. Use a world-target task (wb9/wb12/wb15).")

say(f"policy {os.path.basename(args.checkpoint)} on {args.task}")
say(f"drive limit {args.vmax} m/s, standoff {args.standoff} m")
say("type:  x y z   |   stop   |   q      (or: echo '2.5 0 0.3' > project/target.txt)")

threading.Thread(target=stdin_reader, daemon=True).start()
if args.watch:
    threading.Thread(target=file_watcher, args=(args.watch,), daemon=True).start()
    say(f"watching {args.watch}")

# --------------------------------------------------------------------- loop
target = None
shown = {"t": None}
state = "HOLD"
if args.target:
    commands.put(parse_line(args.target))

obs = env.get_observations()
step, last_report, quit_now = 0, 0.0, False


def park_arm():
    """A neutral hand pose for when there is nothing to reach for.

    NOT used while driving -- see the note in the drive branch.
    """
    d = robot.data
    fwd = torch.stack([torch.cos(d.heading_w), torch.sin(d.heading_w)], dim=1)
    ee.pose_command_w[:, 0:2] = d.root_pos_w[:, 0:2] + 0.40 * fwd
    ee.pose_command_w[:, 2] = u.scene.env_origins[:, 2] + 0.60
    ee.pose_command_w[:, 3:] = IDENT


while app.is_running() and not quit_now:
    t0 = time.time()

    while not commands.empty():
        cmd = commands.get()
        if cmd[0] == "quit":
            quit_now = True
        elif cmd[0] == "stop":
            state, target = "HOLD", None
            say("holding")
        elif cmd[0] == "home":
            # Standing still is not the same as staying still: with a zero
            # velocity command these policies still leak ~0.15 m/s, which over a
            # minute of idling carries the robot metres from where it started.
            # A reset puts it back on its env origin so a typed coordinate means
            # what you expect.
            # reset() must run inside inference_mode, or Isaac Lab raises
            # "Inplace update to inference tensor outside InferenceMode".
            with torch.inference_mode():
                obs = env.reset()[0]
            state, target = "HOLD", None
            say("reset to the origin")
        elif cmd[0] == "fwd":
            d0 = robot.data
            ahead = torch.stack([torch.cos(d0.heading_w), torch.sin(d0.heading_w)], dim=1)[0]
            pos = d0.root_pos_w[0]
            target = torch.tensor([float(pos[0] + cmd[1] * ahead[0]),
                                   float(pos[1] + cmd[1] * ahead[1]),
                                   float(u.scene.env_origins[0, 2]) + cmd[2]],
                                  device=u.device).unsqueeze(0)
            state = "DRIVE"
            say(f"target {cmd[1]:.1f} m straight ahead at z={cmd[2]:.2f}")
        elif cmd[0] == "goto":
            target = torch.tensor(cmd[1:], device=u.device).unsqueeze(0) + u.scene.env_origins
            say(f"debug: env_origin {[round(v, 2) for v in u.scene.env_origins[0].tolist()]} "
                f"robot {[round(v, 2) for v in robot.data.root_pos_w[0, :3].tolist()]} "
                f"target_w {[round(v, 2) for v in target[0].tolist()]}")
            state = "DRIVE"
            d0 = float(torch.norm(target[:, :2] - robot.data.root_pos_w[:, :2], dim=1).mean())
            say(f"target ({cmd[1]:.2f}, {cmd[2]:.2f}, {cmd[3]:.2f}) -- {d0:.2f} m away")

    if target is not shown["t"]:
        show_target(target)
        shown["t"] = target
    d = robot.data
    if state == "HOLD" or target is None:
        vel.vel_command_b[:] = 0.0
        if target is not None:
            ee.pose_command_w[:, :3] = target
            ee.pose_command_w[:, 3:] = IDENT
        else:
            park_arm()
    else:
        err = target[:, :2] - d.root_pos_w[:, :2]
        dist = torch.norm(err, dim=1)
        c, s = torch.cos(d.heading_w), torch.sin(d.heading_w)
        ex = c * err[:, 0] + s * err[:, 1]
        ey = -s * err[:, 0] + c * err[:, 1]
        bearing = torch.atan2(ey, ex)

        if state == "DRIVE":
            speed = torch.clamp((dist - args.standoff) / args.tracking, min=0.0, max=args.vmax)
            # Face the goal before driving, and ease off the throttle in
            # proportion to how far off heading we still are -- driving at full
            # speed while 25 deg off arcs around the target instead of closing on
            # it, which is what a weak-turning policy does if you let it.
            speed = speed * torch.clamp(torch.cos(bearing), min=0.0)
            speed = torch.where(bearing.abs() > args.turn_first, torch.zeros_like(speed), speed)
            vel.vel_command_b[:, 0] = speed
            vel.vel_command_b[:, 1] = 0.0
            vel.vel_command_b[:, 2] = torch.clamp(2.0 * bearing, -args.wmax, args.wmax)
            # What the hand does while driving is a real trade, see --drive_arm.
            if args.drive_arm == "far":
                ee.pose_command_w[:, :3] = target
                ee.pose_command_w[:, 3:] = IDENT
            else:
                park_arm()
            if bool((dist <= args.standoff + 0.05).all()):
                state = "REACH"
                say(f"arrived, {float(dist.mean()):.2f} m from it -- reaching")
        else:  # REACH
            vel.vel_command_b[:] = 0.0
            ee.pose_command_w[:, :3] = target
            ee.pose_command_w[:, 3:] = IDENT

    with torch.inference_mode():
        obs, _, _, _ = env.step(policy(obs))
    step += 1

    now = step * dt
    if target is not None and now - last_report >= args.report_every:
        last_report = now
        hand_err = float(ee.metrics["position_error"].mean())
        base_z = float(d.root_pos_w[:, 2].mean() - u.scene.env_origins[:, 2].mean())
        if state == "DRIVE":
            err2 = target[:, :2] - d.root_pos_w[:, :2]
            c2, s2 = torch.cos(d.heading_w), torch.sin(d.heading_w)
            brg = torch.atan2(-s2 * err2[:, 0] + c2 * err2[:, 1], c2 * err2[:, 0] + s2 * err2[:, 1])
            say(f"{state}  {float(torch.norm(err2, dim=1).mean()):5.2f} m to go | "
                f"bearing {math.degrees(float(brg.mean())):+6.1f} deg | "
                f"cmd vx {float(vel.vel_command_b[:, 0].mean()):+.2f} wz {float(vel.vel_command_b[:, 2].mean()):+.2f} | "
                f"actual vx {float(d.root_lin_vel_b[:, 0].mean()):+.2f} | "
                f"h {float(d.root_pos_w[:, 2].mean() - u.scene.env_origins[:, 2].mean()):.2f} "
                f"tilt {math.degrees(math.acos(max(-1.0, min(1.0, float(-d.projected_gravity_b[:, 2].mean()))))):4.0f} deg"
                + ("  (turning to face it)" if abs(float(brg.mean())) > 0.6 else ""))
        else:
            say(f"{state}  hand error {hand_err * 100:5.1f} cm | base height {base_z:.3f} m"
                + ("   <- crouching" if base_z < 0.45 else ""))

    if args.real_time:
        time.sleep(max(0.0, dt - (time.time() - t0)))

say("done")
cam.close()          # release the camera before teardown
env.close()
app.close()
