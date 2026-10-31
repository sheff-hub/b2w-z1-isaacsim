"""Phase 5, step 1: drive to the table and pick up the box -- scripted, oracle perception.

Task RobotLab-Isaac-Fetch-Warehouse-Unitree-B2W-Z1-Gripper-v0 (fetch_env_cfg.py).
The job is split in two, and each half uses the tool that already works:

  LOCOMOTION    model_2600.pt (B2W flat policy) drives the legs and wheels. A
                go-to-pose controller turns "reach the standoff point in front of
                the box, facing the table" into its velocity commands.
  MANIPULATION  the base holds still (zero command). The arm unfolds in JOINT
                space to a ready pose (elbow high, above the lidar mast), then
                short damped-least-squares IK moves on the point between the
                fingertips: pre-grasp -> open -> descend -> close -> lift, and
                back to the ready pose to carry. The box pose comes from the
                simulator (the oracle).

Why the arm is not one differential-IK controller from start to finish: from the
folded stow pose, differential IK toward a far, rotated target wanders into
joint limits (joint4 pinned at 87 deg, joint2 on its stop) and stalls ~40 cm
away. It only converges locally, so it is only used near the box.

Every episode is logged per stage and judged:
  success = box lifted >= 8 cm off the table AND still held after carrying
            it back to the ready pose (within 6 cm of the grasp point).

    ISAAC_IMAGE=isaac-lab:2.3.2 ./run_script.sh fetch_box.py --device cpu --episodes 5 --video
    (--device cpu: PhysX on the CPU; ~2x real time for one robot, and immune to a
    busy GPU -- on the shared GPU it ran at 0.02x)

LANGUAGE (--vln): the task sentence (--task_instruction, kept for the dataset) is done
in three phases. 1) DualVLN (vln_nav.py; server Isaac/vln/run_dualvln_server.sh) walks
to the table from the camera alone, on the navigation part of the sentence (--vln).
2) Scripted: back away from the table until there is room to turn, then the usual
docking -- routed around the table if DualVLN stopped on its far side. 3) The scripted
pick. Every phase is logged, so a failure is attributed to walking, parking or picking.

    ./run_script.sh fetch_box.py --device cpu --deliver --episodes 10 \\
        --vln "Walk to the wooden table and stop in front of it." \\
        --starts "1.0,-1.5,30; 0.5,1.0,0; 6.0,1.0,0"
"""

import argparse
import json
import math
import os
import queue
import re
import time

from isaaclab.app import AppLauncher

LOGS = "/workspace/policies"  # trained checkpoints (policies/)
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="RobotLab-Isaac-Fetch-Warehouse-Unitree-B2W-Z1-Gripper-v0")
parser.add_argument("--drive_ckpt", default=f"{LOGS}/unitree_b2w_flat/reference_dda1510/model_2600.pt")
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--video", action="store_true", help="record episodes (--video_episodes) to --video_dir")
parser.add_argument("--video_episodes", default="0", help="comma-separated episode indices to record")
parser.add_argument("--video_dir", default="/workspace/renders/fetch")
parser.add_argument("--out", default="/workspace/renders/fetch/results.json")
parser.add_argument("--real-time", action="store_true")
parser.add_argument("--gui", action="store_true", help="open the Isaac Sim window (needs DISPLAY; see live.sh)")
parser.add_argument("--probe", action="store_true", help="no driving: test the ready pose and IK waypoints")
parser.add_argument("--probe_n", type=int, default=4, help="probe attempts")
parser.add_argument("--probe_stance", default="", help="--loco eth: hold these stance values in turn (1 = crouched "
                    "6 cm, 0 = ETH's default, < 0 = taller, extrapolated) and report base height and tilt; no driving")
parser.add_argument("--probe_yaw", default="0", help="probe: comma-separated heading offsets (deg) of the dock pose")
parser.add_argument("--probe_lateral", default="0", help="probe: comma-separated offsets (m) of the dock pose toward the table (side) / forward (front)")
parser.add_argument("--nav2", action="store_true",
                    help="Nav2 (start nav2/run_nav2.sh first) picks the side and drives to the entry point E "
                         "collision-free; the scripted part then drives E -> P -> D and picks. Replaces the "
                         "scripted side choice and corner route; falls back to them if Nav2 fails")
parser.add_argument("--no_progress", action="store_true", help="do not write renders/progress/progress.jsonl (./process.sh)")
parser.add_argument("--nav2_sensor", choices=("os0", "scan", "both"), default="os0",
                    help="what Nav2 sees: os0 = the robot's own RTX lidar (PointCloud2 /points, needs rendering), "
                         "scan = the 3-height ray-cast /scan, both")
parser.add_argument("--nav2_goal", choices=("dock", "entry"), default="dock",
                    help="dock: Nav2 drives through E all the way to the pick point D, in its heading, and the pick "
                         "starts there; entry: Nav2 only to E, the script drives E -> P -> D (the first setup)")
parser.add_argument("--target", default="auto",
                    help="three-table scene (task ...Fetch3...): which box, as table/box colours, e.g. green/red; "
                         "auto = a random one per episode (seeded). DualVLN is told the table, the task names both. "
                         "Shelf scenes: shelf/colour/table or a colour. A comma list takes them in turn per episode")
parser.add_argument("--vln_follower", choices=("pp", "follow", "goal"), default="pp",
                    help="who drives DualVLN's trajectories: pp = our pure pursuit (vln_nav.follow); follow = Nav2's "
                         "controller follows each trajectory (FollowPath, needs --nav2); goal = Nav2 plans and drives to "
                         "each trajectory's end (NavigateToPose, needs --nav2)")
parser.add_argument("--nav2_timeout", type=float, default=90.0, help="s of sim time for the Nav2 drive")
parser.add_argument("--ik_gain", type=float, default=0.1,
                    help="fraction of the IK step added to the arm targets per control step. 0.3 made the arm "
                         "shake at 2.3 Hz (hand +-5 cm, growing) whenever it held a pose by IK -- above the box "
                         "and after the lift: integral action on targets the PD drive follows ~6 deg behind")
parser.add_argument("--ik_deadband", type=float, default=0.0,
                    help="m: hold the arm targets while the grasp point is this close to its setpoint")
parser.add_argument("--pitch", type=float, default=60.0, help="grasp approach, degrees below horizontal (80: arm hits the lidar mast)")
parser.add_argument("--no_self_collision", action="store_true")
parser.add_argument("--approach", choices=("side", "front"), default="side",
                    help="side: drive along the table and pick to the robot's right (default); front: face the table")
parser.add_argument("--side", choices=("auto", "right", "left"), default="auto",
                    help="side pick: park with the table on the robot's right (facing +y) or left (facing -y); "
                         "auto = whichever is cheaper to reach (drive + turn) from where the robot is")
parser.add_argument("--roll_left", type=float, default=None, help="left-side ready-pose wrist roll (default: READY_ROLL)")
parser.add_argument("--arm", choices=("moveit", "ik"), default="moveit",
                    help="moveit: MoveIt plans every arm motion of the pick around the table, the boxes and the robot's "
                         "own body (start it first: python3 run.py moveit start); ik: the scripted damped "
                         "least-squares pick")
parser.add_argument("--moveit_server", default="http://127.0.0.1:5810", help="MoveIt bridge (moveit/moveit_server.py)")
parser.add_argument("--moveit_speed", type=float, default=0.3,
                    help="MoveIt velocity and acceleration scaling of the free motions (0..1); straight-line moves "
                         "near the box use half of it")
parser.add_argument("--deliver", action="store_true", help="after the pick, carry the box to DROP_ZONE and check it is still held")
parser.add_argument("--deliver_frac", type=float, default=None,
                    help="per episode: deliver with this probability, else pick only (overrides --deliver; the task "
                         "sentence follows: pick-only wordings never say 'bring', delivery wordings always do)")
parser.add_argument("--video_res", default="", help="video resolution WxH (default: the viewer's 1280x720)")
parser.add_argument("--video_quality", type=float, default=5.0, help="imageio's x264 quality 0-10: 5 = crf 25 (small "
                    "files), 7 = crf 15 (near lossless, for presentations)")
parser.add_argument("--stills", type=int, default=0, help="with --video: also save every N-th video frame as PNG "
                    "(lossless, for figures) to <video_dir>/stills_seed<S>_ep<E>/")
parser.add_argument("--cam_eye", default="5.1,-0.2,0.95", help="--cam custom: camera position x,y,z (world)")
parser.add_argument("--cam_lookat", default="3.9,1.3,0.55", help="--cam custom: look-at point x,y,z (world)")
parser.add_argument("--cam", choices=("wide", "follow", "center", "close", "hero", "overview", "custom", "scene", "scene_follow"),
                    default="wide",
                    help="video camera: fixed wide shot of aisle and table, follow the robot, follow it with the "
                         "whole robot centred (wheels in view, for the stance), or close on the gripper; scene / "
                         "scene_follow: the scene file's camera (fixed / following the robot)")
parser.add_argument("--loco", choices=("ours", "eth"), default="eth",
                    help="base controller: ours = model_2600 (robot_lab); eth = ETH's SRU policy (eth_loco.py), "
                         "with its drives and its stance, 15 cm taller")
parser.add_argument("--vln", default="", help="walk to the table with DualVLN on this navigation instruction first")
parser.add_argument("--walk_only", action="store_true", help="door scene: end each episode after DualVLN's walk "
                    "(to compare walk sentences and starts quickly)")
parser.add_argument("--door_unturn_s", type=float, default=4.0, help="door scene: once the door is pushed open, "
                    "turn the lever back up under the hand over this long before the jaws let go of it (0: let go of "
                    "it pressed, as before: its spring snaps it up into the jaws and the arm swings)")
parser.add_argument("--door_ease_s", type=float, default=1.0, help="door scene: then ease the arm off the lever over "
                    "this long, the jaws still closed: its joint targets to where it only holds its own weight (0: no, "
                    "the targets as the IK left them)")
parser.add_argument("--task_instruction", default="Pick up the box on the wooden table.",
                    help="the whole task in words, recorded with each episode")
parser.add_argument("--vln_server", default="http://127.0.0.1:5801")
parser.add_argument("--vln_max_time", type=float, default=60.0, help="s of simulated time for the DualVLN phase")
parser.add_argument("--vln_out", default="/workspace/renders/fetch_vln", help="DualVLN logs/frames per episode")
parser.add_argument("--live_vln", action="store_true",
                    help="--interactive in a shelf scene: DualVLN walks to the named shelf first (default: Nav2 drives "
                         "there from the start)")
parser.add_argument("--interactive", action="store_true",
                    help="live: type tasks, e.g. \"go to the wooden table and pick the box\" (see live.sh fetch)")
parser.add_argument("--watch", default="/workspace/project/instruction.txt", help="--interactive: also read tasks from this file")
parser.add_argument("--wheel_brake", choices=("auto", "on", "off"), default="auto",
                    help="zero the wheel velocity targets while the arm works (the base-hold phase). Default off: "
                         "tried for --loco eth (its base drifts with the arm out), the policy then pushed the body "
                         "around on its legs instead -- 62 cm in one descent, 0/10")
parser.add_argument("--crouch", choices=("auto", "on", "off"), default="auto",
                    help="--loco eth: lower the body to ETH's crouched profile (~6 cm) while the arm picks; auto = on")
parser.add_argument("--record", default="", help="record every episode for the VLA dataset (recorder.py) under this "
                    "directory, e.g. /workspace/datasets/raw/<run>")
parser.add_argument("--record_fps", type=float, default=10.0)
parser.add_argument("--random_start", default="", help="x0,x1,y0,y1[,yaw0,yaw1]: a random start per episode in "
                    "this box (any heading, or in yaw0..yaw1 deg), at least 1 m from the tables -- dataset variety; "
                    "overrides --starts")
parser.add_argument("--task_variants", action="store_true",
                    help="a random wording of the task per episode (TASK_TEXTS; with --deliver the bring-back ones)")
parser.add_argument("--starts", default="", help="robot start poses \"x,y,yaw; ...\" (deg), used in turn per episode")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.nav2:
    # the ROS 2 bridge must be enabled at start-up: enabled later (enable_extension) it reloads
    # omni.replicator, whose graph then cannot be rebuilt -- Isaac Lab's env.seed() / reset(seed) fail
    args.kit_args = ((getattr(args, "kit_args", "") or "") + " --enable isaacsim.ros2.bridge").strip()
    if args.nav2_sensor in ("os0", "both"):  # the RTX lidar: its annotators, and rendering every step
        args.kit_args += " --enable isaacsim.sensors.rtx"
        args.enable_cameras = True
assert args.vln_follower == "pp" or args.nav2, "--vln_follower follow / goal drive with Nav2: add --nav2"
SHELF_TASK = "Shelf" in args.task or "Door" in args.task  # shelf / door scene: live tasks need no DualVLN walk
#                                                              (where the shelf and the doors are is known; --live_vln)
USE_VLN = bool(args.vln) or (args.interactive and (args.live_vln or not SHELF_TASK))
CAMS = USE_VLN or bool(args.record)  # the VLN cameras double as the recorded front camera
if args.video or CAMS:
    args.enable_cameras = True
if not args.gui and not (args.livestream or 0) > 0:  # the default is -1 (truthy)
    args.headless = True  # AppLauncher renders offscreen (videos) only when headless AND enable_cameras
launcher = AppLauncher(args)
app = launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import isaaclab.utils.math as mu  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import robot_lab.tasks  # noqa: E402,F401
import progress as progress_mod  # noqa: E402  (./process.sh reads what it writes)
from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.unitree_b2w import fetch_env_cfg as F  # noqa: E402

if CAMS or args.interactive:
    import vln_nav  # noqa: E402  (Isaac Lab must be up first; --interactive reads typed tasks with it)
if args.nav2:
    import nav2_link  # noqa: E402
if args.record:
    import recorder  # noqa: E402
if args.loco == "eth":
    import eth_loco  # noqa: E402


def say(*a):
    print("FETCH|", *a, flush=True)


# ------------------------------------------------------------------ settings
STANDOFF = 0.56          # base centre to the table's near edge (m): where v6 actually ended up (nominal 0.62, overrun
                         # ~6 cm) -- now reached directly by the closed-loop trim. At 0.62 the box sat further out
                         # and the carry pose held it lower (8/72 lifted < 8 cm)
PREDOCK = 0.60           # the turn to face the table happens this far before the standoff point (m)
MIN_SPEED = 0.25         # m/s, as a (vx, vy) norm: robot_lab's command generator zeroed every linear
                         # command with norm <= 0.2 in training, so model_2600 stands still below that
TRIM_SPEED = 0.3         # m/s, distance-trim pulses
STOP_LEAD = 0.12         # m: stop this early; after a zero command it coasts on -- 36 episodes at 0.08 overran by 3-18 cm
PRE_DIST = 0.12          # pre-grasp: backed off along the approach axis (m)
GRASP_RAISE = 0.015      # grasp point above the box centre. 0.005: the jaw root / palm clipped the box's near top edge while descending (9/72 knocked); 0.02-0.03 with the old stiff grip squeezed it out
CLOSE_TIME = 1.0         # s to close the gripper: slamming it shut popped the box out
LIFT = 0.06              # short IK lift after closing (m); the carry move lifts the rest
CARRY_LIFT = 0.12        # m: box bottom above the table when carried (topped up by IK if short)
NAV_TIMEOUT = 40.0       # s
BACKOFF_CLEAR = 0.8      # m, base centre to the table top's edge: room to turn without sweeping the body into it
ROUTE_BLOCK = 0.6        # m: a straight drive closer than this to the table is routed round a corner...
ROUTE_CORNER = 0.75      # m: ...via the corners of the table grown by this much
ENTRY_BACK = 0.8         # m: turn to face the pre-dock point this far behind it, then drive there heading the dock way
STAGING_BACK = 2.1       # m: --nav2 dock: Nav2's staging pose S this far behind the pick point (docking server staging_x_offset)
# --nav2 dock: park 4 cm further from the table than the script. At 5 cm the footprint (0.30 + 0.01 m) sat
# 2.7 cm from the table, inside the costmap's 5 cm cell at the edge that the lidar marks, and the docking
# controller refused ("Failed to get control"). The arm picks 3 and 5 cm further out (4/4 probes).
NAV2_DOCK_EXTRA = 0.04
STALL_S, KICK_S = 1.0, 0.6  # s: commanded to move but still this long -> kick for this long
BREAK_S, BREAK_HOLD = 0.3, 0.5  # --nav2: still this long under a Nav2 command -> breakaway command for this long
BREAK_V, BREAK_W = 0.28, 0.30   # m/s, rad/s: the smallest linear / turn commands the ETH controller follows from rest
DZ_VY, DZ_WZ = 0.12, 0.10       # m/s, rad/s: --nav2 dead-zone inverse for sideways and turn commands
KICK, KICK_MAX = 1.8, 0.5   # the kick multiplies the linear command, up to this norm (m/s)
DROP_ZONE = F.DROP_ZONE  # world xy (m): the delivery point (scene file drop_zone; the three-table one below)
# Docking tolerance. The arm targets are computed from the box in the base frame,
# so the pick does not need a precise stop -- only a box within reach.
DOCK_POS = 0.05          # m, per axis
DOCK_YAW = math.radians(8.0)
DOCK_PATIENCE = 4.0      # s docking, after which a stop within DOCK_GOOD is accepted
DOCK_GOOD = (0.12, math.radians(15.0))
MIN_TURN = 0.45          # rad/s: model_2600 does not turn for 0.25 (dead zone)
# Ready pose (joint1..6, rad): upper arm swung forward and up so the elbow is
# above the lidar mast, forearm pointing forward-down. joint6 rolls the gripper
# so its jaws close sideways; READY_ROLL's sign is chosen by --probe.
READY = [0.0, 2.3, -1.9, 0.8, 0.0]
READY_ROLL = -1.45       # the roll of a measured good grasp (joint6 -1.44); +90 deg sent IK the long way round
# Carry pose: like READY with the forearm folded further up -- lifted a held box 16 cm.
CARRY_Q = [0.0, 2.2, -2.2, 1.0, 0.0]
# SIDE approach: drive along the table's near edge, heading +y, table on the robot's
# right; the arm works to that side (joint1 -90 deg), so it never passes over the lidar
# mount at the front, and the braking overshoot lands ALONG the table, where the arm
# absorbs it. The widest parts at table-top height are the hip housings, |y| 0.287 m
# (probe_robot_extent.py); SIDE_GAP is their clearance to the table edge.
HIP_HALF_WIDTH = 0.287
SIDE_GAP = 0.15         # m (0.12 + up to 5.6 cm docking error left 6-8 cm, and the box too close for IK)
SIDE_ABEAM = 0.08        # m: stop with the box this far ahead of the base centre (arm base at +0.056)
SIDE_PREDOCK_BACK = 1.3  # m: turn to drive along the table this far before the stop point...
SIDE_PREDOCK_OUT = 0.20  # ...and this much further from the table (the turn creeps and sweeps)
# Ready pose for the side pick, near the pre-grasp joints that successful side picks
# reached (joint1 ~ -1.5, 1.9, -1.15, 0.35): from the high forward READY rotated to the
# side, IK toward a box 5 cm closer than planned swung joint1 to -2.3 and failed (5/8).
READY_SIDE = [-1.57, 2.0, -1.35, 0.45, 0.0]
# left-side wrist roll: the SAME as the right side's (3/3 probes). Mirrored (+1.45), IK swung the
# wrist round to 2.65 and closed the jaws along the wrong axis (2/3 failed).
READY_ROLL_LEFT = args.roll_left if args.roll_left is not None else READY_ROLL
# IK
IK_LAMBDA = 0.05         # damping
IK_GAIN = args.ik_gain   # fraction of the DLS step added to the joint TARGETS each control step
IK_MAX_STEP = 0.05       # rad per control step
IK_WINDUP = 0.25         # rad: targets may lead the measured joints by at most this
IK_NULL_GAIN = 0.1       # null-space pull toward the nominal posture (fraction per step, before IK_GAIN)
if args.loco == "eth":   # ETH's policy follows small commands: lower floors (eth_compare.py)
    MIN_SPEED, MIN_TURN = eth_loco.MIN_SPEED, eth_loco.MIN_TURN
    # ETH's stance is 15 cm taller: the hip housings ride above the 0.6 m table top, and the
    # box sits 15 cm further below the arm's shoulder. From the old dock pose IK hit joint2's
    # limit (1/3 probes); 10 cm closer, 3/3 with no contact (--probe --probe_lateral 0.05,0.10)
    SIDE_GAP = 0.05
    # (READY_SIDE stays: one nearer the converged pre-grasp joints, (-1.57, 2.25, -1.0, -0.2),
    # swept the gripper into the box on the way, 12-37 cm in 6/6 probes)
    if USE_VLN:
        vln_nav.MIN_SPEED, vln_nav.MIN_TURN = eth_loco.MIN_SPEED, eth_loco.MIN_TURN


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
    return nn.Sequential(*layers).to(device).eval(), sd[f"actor.{idx[0]}.weight"].shape[1]


# ---------------------------------------------------------------- task setup
cfg = parse_env_cfg(args.task, device=args.device or "cuda:0", num_envs=1)
cfg.seed = args.seed
for name in list(vars(cfg.terminations)):  # the script judges falls itself
    if not name.startswith("_"):
        setattr(cfg.terminations, name, None)
if args.no_self_collision:
    cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
if CAMS:
    vln_nav.add_cameras(cfg)
if args.record:
    recorder.add_wrist_camera(cfg)
if args.loco == "eth":
    eth_loco.patch_cfg(cfg)
if args.nav2:
    nav2_link.enable_queries(cfg)  # the /scan ray casts and the map need PhysX scene queries
if args.cam == "wide":  # fixed, world frame: sees the spawn, the aisle and the table
    cfg.viewer.origin_type = "world"
    cfg.viewer.eye = (6.2, -2.6, 2.3)
    cfg.viewer.lookat = (2.9, 0.3, 0.45)
elif args.cam == "center":  # follows the root (world axes); "follow" aims ahead of it and crops the wheels
    cfg.viewer.eye = (-2.4, -2.2, 1.6)
    cfg.viewer.lookat = (0.0, 0.0, 0.2)
elif args.cam == "close" and args.approach == "side":  # across the table, looking back at the arm
    cfg.viewer.eye = (1.9, -0.2, 0.9)
    cfg.viewer.lookat = (0.45, 0.05, 0.05)
elif args.cam == "close":  # beside the pick, looking at the box (offsets from the robot root, world axes)
    cfg.viewer.eye = (0.55, -0.75, 0.55)
    cfg.viewer.lookat = (0.62, 0.0, 0.08)
elif args.cam == "hero":  # follows the root (world axes), front three-quarter view for figures
    cfg.viewer.eye = (2.3, -1.9, 1.45)
    cfg.viewer.lookat = (0.15, 0.0, 0.35)
elif args.cam == "custom":  # fixed, world frame: --cam_eye / --cam_lookat
    cfg.viewer.origin_type = "world"
    cfg.viewer.eye = tuple(float(v) for v in args.cam_eye.split(","))
    cfg.viewer.lookat = tuple(float(v) for v in args.cam_lookat.split(","))
elif args.cam in ("scene", "scene_follow"):  # the scene file's views (scenes/*.yaml, camera:)
    _sc = F.SCENE_DOORS if "Door" in args.task else F.SCENE_SHELF if "Shelf" in args.task else (
        F.SCENE_MULTI if "Fetch3" in args.task else F.SCENE_ONE)
    _cam = _sc["camera"]["fixed" if args.cam == "scene" else "follow"]
    cfg.viewer.origin_type = "world" if args.cam == "scene" else "asset_root"
    cfg.viewer.eye, cfg.viewer.lookat = tuple(_cam["eye"]), tuple(_cam["lookat"])
elif args.cam == "overview":  # fixed, high: the whole three-table scene
    cfg.viewer.origin_type = "world"
    cfg.viewer.eye = (-3.6, 1.0, 7.8)
    cfg.viewer.lookat = (4.2, 1.0, -0.4)
if args.video_res:
    cfg.viewer.resolution = tuple(int(v) for v in args.video_res.split("x"))

env = gym.make(args.task, cfg=cfg, render_mode="rgb_array" if args.video else None)
u = env.unwrapped
VIDEO_EPS = {int(v) for v in args.video_episodes.split(",")} if args.video else set()


def render_plate(sd):
    """--stills: the video camera's view without the robot (the background of a chronophotograph), and the camera
    (eye, look-at, field of view, resolution) so world positions can be projected into the stills."""
    import imageio.v2 as imageio_
    import omni.usd
    from pxr import UsdGeom
    os.makedirs(sd, exist_ok=True)
    stage = omni.usd.get_context().get_stage()
    vis = UsdGeom.Imageable(stage.GetPrimAtPath("/World/envs/env_0/Robot"))
    vis.MakeInvisible()
    u.render()  # the first call only creates the viewport capture (and returns an empty image)
    for _ in range(6):  # the renderer needs a few frames to settle
        u.sim.render()
    imageio_.imwrite(f"{sd}/plate.png", u.render())
    vis.MakeVisible()
    for _ in range(3):
        u.sim.render()
    cam = UsdGeom.Camera(stage.GetPrimAtPath(cfg.viewer.cam_prim_path))
    f, ha = cam.GetFocalLengthAttr().Get(), cam.GetHorizontalApertureAttr().Get()
    with open(f"{sd}/camera.json", "w") as fh:
        json.dump({"eye": list(cfg.viewer.eye), "lookat": list(cfg.viewer.lookat), "origin_type": cfg.viewer.origin_type,
                   "resolution": list(cfg.viewer.resolution), "focal_mm": f, "h_aperture_mm": ha,
                   "hfov_deg": math.degrees(2 * math.atan(ha / 2 / f))}, fh, indent=1)
    say(f"plate and camera -> {sd}")
frames = {"on": False, "writer": None, "n": 0}


def video_start():
    """Stream the episode's frames into an mp4 as they come: ~5000 HD frames held in memory (the old way) are
    ~14 GB, and the kernel killed a run for it on the shared machine."""
    import imageio.v2 as imageio_
    os.makedirs(args.video_dir, exist_ok=True)
    import uuid
    frames["tmp"] = f"{args.video_dir}/.recording_{uuid.uuid4().hex[:8]}.mp4"  # (pids repeat across containers)
    frames["writer"] = imageio_.get_writer(frames["tmp"], fps=int(round(1 / dt)), macro_block_size=8,
                                           quality=args.video_quality)
    frames["n"], frames["on"] = 0, True


def video_end(name):
    """Close the stream and give it its final name (the outcome is known only now)."""
    frames["on"] = False
    if frames["writer"] is None:
        return
    frames["writer"].close()
    frames["writer"] = None
    if frames["n"]:
        os.replace(frames["tmp"], name)
        say(f"video {name} ({frames['n']} frames)")
    else:
        os.remove(frames["tmp"])
dev = u.device
if args.video:
    import carb
    say(f"render: enable_cameras={args.enable_cameras} launcher offscreen={launcher._offscreen_render} "
        f"carb={carb.settings.get_settings().get('/isaaclab/render/offscreen')} sim mode={u.sim.render_mode}")
dt = u.step_dt
robot = u.scene["robot"]
SHELF = any(k.startswith("shelfbox_") for k in u.scene.keys())  # shelves and tables (fetch_env_cfg ...Shelf...)
DOOR = any(k.startswith("door_") for k in u.scene.keys())  # hinged doors (fetch_env_cfg ...Door...)
MULTI = "box" not in u.scene.keys() and not SHELF and not DOOR  # the three-table scene (fetch_env_cfg ...Fetch3...)
if DOOR:
    DOOR_ARTS = {k: u.scene[F.door_name(k)] for k in F.DOORS}
    box = None  # nothing to pick: the door episode never asks for it
    assert args.arm == "moveit", "the door scene is opened by MoveIt: --arm moveit"
elif SHELF:
    BOXES = {(s, b[0]): u.scene[F.shelf_box_name(s, b[0])] for s in F.SHELVES for b in F.shelf_boxes(s)}
    box = next(iter(BOXES.values()))  # set_target() per episode
    assert args.arm == "moveit", "the shelf scene is picked and placed by MoveIt: --arm moveit"
elif MULTI:
    TABLE_COLORS = F.COLORS3
    BOXES = {(t, c): u.scene[F.box_name(t, c)] for t in F.COLORS3 for c in F.BOX_COLORS}
    box = BOXES[("yellow", "red")] if ("yellow", "red") in BOXES else next(iter(BOXES.values()))  # set_target() per episode
    DROP_ZONE = F.DROP_ZONE3
else:
    box = u.scene["box"]
# the current target: its table (or shelf) and box. near_x: the edge the robot docks at, face: the robot works from
# its -x (-1) or +x (+1) side, top: the surface the box stands on (or goes to); the rest tunes the arm (shelf scene)
TGT = {"table": None, "box": None, "near_x": F.TABLE_NEAR_EDGE_X, "face": -1, "top": F.TABLE_HEIGHT, "sides": None,
       "pitch": None, "pre_dist": None, "lift": None, "retreat": 0.0, "crouch": True, "flip": False, "carry": None,
       "abeam": None, "gap": None, "stance": None, "place_depth": None}
am = u.action_manager
vel = u.command_manager.get_term("base_velocity")

if args.loco == "eth":
    eth_driver = eth_loco.Driver(u)
else:
    drive, d_in = load_actor(args.drive_ckpt, dev)
    assert (d_in,) == tuple(u.observation_manager.group_obs_dim["policy"]), "drive policy / obs group mismatch"
slices, start = {}, 0
for name, width in zip(am.active_terms, am.action_term_dim):
    slices[name] = slice(start, start + width)
    start += width
say(f"action terms {[(k, v.stop - v.start) for k, v in slices.items()]}, step {dt * 1000:.0f} ms, device {dev}")
if args.nav2:
    _LID = robot.find_bodies(["lidar_link"])[0][0]
    NAV2_LINK = nav2_link.Nav2Link(  # attached to NAV2 below, once NAV2 exists
        sensor=args.nav2_sensor,
        lidar_link_pose=lambda: (robot.data.body_pos_w[0, _LID].cpu().numpy(), robot.data.body_quat_w[0, _LID].cpu().numpy()),
        # the robot's links and the box it may be carrying, for the lidar self filter
        self_points=lambda: np.concatenate([robot.data.body_pos_w[0].cpu().numpy()]
                                           + ([box.data.root_pos_w[0:1].cpu().numpy()] if box is not None else [])))
PROG = progress_mod.Progress(enabled=not args.no_progress)
PROG.event("sim_ready", loco=args.loco, nav2=args.nav2, vln=args.vln, deliver=args.deliver,
           episodes=args.episodes, interactive=args.interactive)

ARM = robot.find_joints([f"joint{k}" for k in range(1, 7)], preserve_order=True)[0]
GRIP = robot.find_joints(["jointGripper"])[0][0]
BODY = robot.find_bodies(["gripperStator"])[0][0]
JAC_COLS = [i + 6 for i in ARM]  # floating base: the Jacobian's first 6 columns are the base
LIM = robot.data.soft_joint_pos_limits[0, ARM]
OFFSET = torch.tensor([F.GRASP_POINT], device=dev)

state = {"arm": None, "grip": F.GRIPPER_CLOSED}  # absolute joint targets (rad)
cur = {"obs": None}  # the latest observation, for the DualVLN phase (vln_nav steps through sim_step)
commands = queue.Queue()  # --interactive: typed tasks and control words


def sim_step(cmd):
    vel.vel_command_b[:] = torch.tensor(cmd, device=dev)
    cur["obs"] = step(cur["obs"])


class Nav2VlnDriver:
    """Nav2 drives DualVLN's trajectories (vln_nav.Navigator's driver): "follow" streams each trajectory to
    the controller server (FollowPath: Nav2's controller and collision checking on DualVLN's own path);
    "goal" streams each trajectory's end as a NavigateToPose goal (Nav2 plans its own path there). The
    newest trajectory replaces the previous goal (every ~0.4 s of sim time)."""

    STALL_S, STALL_M = 3.0, 0.05  # not moved this far in this long while following a path: clear the costmaps

    def __init__(self, mode):
        self.mode = mode
        self.name = {"follow": "nav2_followpath", "goal": "nav2_navigate_to_end"}[mode]
        self.kick = None  # new_kick() is defined further down
        self.still = None  # (sim time, x, y) since when the robot has not moved
        self.clears = 0

    def new_path(self, path_w):
        pts = [(float(a), float(b)) for a, b in path_w]
        if self.mode == "follow":
            NAV2["link"].stream_path(pts)
        else:
            (x0, y0), (x1, y1) = pts[-2], pts[-1]
            NAV2["link"].stream_goal((x1, y1, math.atan2(y1 - y0, x1 - x0)))

    def command(self):
        if self.kick is None:
            self.kick = new_kick()
        p, _ = base_pose()
        x, y, t = float(p[0]), float(p[1]), NAV2["t"]
        if self.still is None or math.hypot(x - self.still[1], y - self.still[2]) > self.STALL_M:
            self.still = (t, x, y)
        elif t - self.still[0] > self.STALL_S:  # the controller refuses to move ("collision ahead" on stale marks)
            self.clears += NAV2["link"].clear_costmaps(("local",)) > 0
            self.still = (t, x, y)
        return tuple(shape_cmd(list(NAV2["link"].command()), self.kick))

    def stop(self):
        NAV2["link"].stop_stream()
        vel.vel_command_b[:] = 0


WALK_AWAY_M = 3.0  # m: a door walk ends once the robot is this much further from the door than where it set off
WALK_AWAY = {"d0": None}  # that starting distance (reset by vln_walk)


def vln_should_stop():
    """End DualVLN's walk: a new command typed (live), or -- shelf scenes -- the robot within AT_TABLE of the named
    shelf. Nav2 knows where the shelf is and parks at its box side; left to itself DualVLN walked on round the
    see-through shelf in the hall (to its far corner, 2 of 2) instead of stopping in front of it."""
    if args.interactive and not commands.empty():
        return True
    if DOOR and DOOR_T["name"] is not None:  # the door: the same 1 m; and walking away from it ends the walk too
        p, _ = base_pose()
        d = vln_nav.table_edge_dist(float(p[0]), float(p[1]), door_rect(DOOR_T["name"]))
        if d <= vln_nav.AT_TABLE:
            return "near"
        WALK_AWAY["d0"] = d if WALK_AWAY["d0"] is None else WALK_AWAY["d0"]
        if d > WALK_AWAY["d0"] + WALK_AWAY_M:  # DualVLN once walked 11 m the other way, off Nav2's map: hand over
            return "away"                     # to Nav2 while it is still in the hall
        return False
    if SHELF and TGT.get("table") in F.SHELVES:
        p, _ = base_pose()
        if vln_nav.table_edge_dist(float(p[0]), float(p[1]), F.SHELVES[TGT["table"]]["rect"]) <= vln_nav.AT_TABLE:
            return "near"
    return False


nav = vln_nav.Navigator(robot, u.scene["vln_cam"], u.scene["vln_cam_down"], dt, sim_step, args.vln_server,
                        say=lambda *a: say("  vln", *a),
                        should_stop=vln_should_stop,
                        driver=Nav2VlnDriver(args.vln_follower) if args.vln_follower != "pp" else None) if USE_VLN else None
REC = recorder.Recorder(u, robot, u.scene["vln_cam"], u.scene["wrist_cam"], args.record, args.record_fps,
                        front_model=vln_nav.FRONT_MODEL) \
    if args.record else None


def record_begin(rec, nav_text):
    if REC is None:
        return
    PHASE["name"] = "idle"
    REC.begin({"task": rec["task_instruction"], "nav_instruction": nav_text or "", "loco": args.loco,
               "seed": args.seed, "episode": rec["episode"], "start": rec.get("start"),
               "box_start_w": [round(float(v), 3) for v in box.data.root_pos_w[0]], "approach": args.approach,
               "arm": args.arm, "target": rec.get("target"), "scene": "three_tables" if MULTI else (F.SCENE_SHELF["name"] if SHELF else "one_table"),
               "nav2": args.nav2, "vln_follower": args.vln_follower if USE_VLN else None,
               "deliver_task": bool(args.deliver)})


def record_end(rec):
    if REC is None:
        return
    vln = rec["stages"].get("vln", {})
    right_table = None
    if (MULTI or SHELF) and vln.get("end") is not None and rec.get("target"):  # did DualVLN head for the named one?
        ex, ey = vln["end"][0], vln["end"][1]
        rects = {k: tuple(v["rect"]) for k, v in F.SHELVES.items()} if SHELF else {t: F.table_rect3(t) for t in F.COLORS3}
        near = min(rects, key=lambda k: vln_nav.table_edge_dist(ex, ey, rects[k]))
        right_table = near == rec["target"]["table"]
    out = REC.end({"success": bool(rec.get("success")), "delivered": rec.get("delivered"), "placed": rec.get("placed"),
                   "vln_arrived": vln.get("arrived"), "vln_reason": vln.get("reason"), "vln_right_table": right_table,
                   "nav2_docked": rec["stages"].get("nav2_dock", {}).get("success"), "fell": rec.get("fell"),
                   "side": rec["stages"].get("side", {}).get("chosen"),
                   "time_s": round(u.episode_length_buf[0].item() * dt, 1)})
    PHASE["name"] = "idle"
    say(f"ep {rec['episode']} recorded -> {out}")
IK_NOMINAL = {"q": None}  # set to the ready pose (with its wrist roll) when the arm unfolds
ARM_YAW = 0.0 if args.approach == "front" else -math.pi / 2  # arm working direction, base frame (set per episode)
SIDE = {"name": "right"}  # the side pick's side this episode: table on the robot's right or left


def set_side(name):
    """Table on the robot's right (arm out to -90 deg) or left (+90 deg): everything the arm
    does is mirrored about the robot's x axis."""
    global ARM_YAW
    SIDE["name"] = name
    if args.approach == "side":
        ARM_YAW = -math.pi / 2 if name == "right" else math.pi / 2


def ready_side():
    """(joints 1-5, roll) of the side ready pose for the current side: the right one mirrored
    for the left (joint1 and the wrist yaw joint5 flip sign; the roll does not, see READY_ROLL_LEFT)."""
    if SIDE["name"] == "right":
        return READY_SIDE, READY_ROLL
    q = [-READY_SIDE[0], READY_SIDE[1], READY_SIDE[2], READY_SIDE[3], -READY_SIDE[4]]
    return q, READY_ROLL_LEFT
clock = {"t": None}


# ------------------------------------------------------------------ helpers
def act(obs):
    a = torch.zeros(1, am.total_action_dim, device=dev)
    if args.loco == "eth":
        eth_driver.fill(a, vel.vel_command_b[0].tolist())
        if WHEEL_BRAKE and BASE_HOLD["xy"] is not None:  # the legs balance, the wheels hold
            a[:, slices["joint_vel"]] = 0.0
    else:
        d = drive(obs["policy"])
        n_pos = slices["joint_pos"].stop - slices["joint_pos"].start
        a[:, slices["joint_pos"]] = d[:, :n_pos]
        a[:, slices["joint_vel"]] = d[:, n_pos:]
    a[:, slices["arm_action"]] = state["arm"]
    a[:, slices["gripper_action"]] = state["grip"]
    return a


watch = {"on": False, "trace": [], "k": 0}


BASE_HOLD = {"xy": None, "active": False, "engaged_steps": 0}
WHEEL_BRAKE = args.wheel_brake == "on"
HOLD_GAIN, HOLD_VMAX = 2.5, 0.25  # --loco eth: proportional hold, 1/s and m/s
HOLD_ENGAGE, HOLD_RELEASE = 0.03, 0.01  # m: hysteresis of the manipulation-time position hold (0.05 still let the descend miss by 4.7 cm)


CROUCH = args.loco == "eth" and args.crouch in ("auto", "on")


def set_crouch(obs, target, seconds):
    """Ramp ETH's stance toward crouched (1) or default (0) over `seconds`."""
    c0, n = eth_driver.crouch, max(1, int(seconds / dt))
    for k in range(n):
        eth_driver.set_crouch(c0 + (k + 1) / n * (target - c0))
        obs = step(obs)
    return obs


def base_hold_command():
    """While the arm works, keep the base where it docked. Swinging the arm out to the side
    shifts the centre of mass, and the drive policy (trained without an arm) sometimes rolls
    away with a ZERO command -- 54-61 cm along the table in 2 of 8 episodes, so the arm chased
    a box that kept moving. Engage past HOLD_ENGAGE, steer back at the smallest speed the policy
    follows, release inside HOLD_RELEASE."""
    p, yaw = base_pose()
    e = BASE_HOLD["xy"] - p[:2]
    d = float(e.norm())
    if args.loco == "eth":
        # ETH's policy follows small commands (0.1 m/s at 104 %), so hold continuously and in
        # proportion, heading included: its base drifts with the arm out (up to 17.7 cm in one
        # descent under the hysteresis hold below, which knocked the box in 3 of 4 failures)
        BASE_HOLD["engaged_steps"] += int(d > 0.01)
        ex = math.cos(yaw) * float(e[0]) + math.sin(yaw) * float(e[1])
        ey = -math.sin(yaw) * float(e[0]) + math.cos(yaw) * float(e[1])
        k = min(HOLD_GAIN, HOLD_VMAX / max(d, 1e-6))
        yerr = math.atan2(math.sin(BASE_HOLD["yaw"] - yaw), math.cos(BASE_HOLD["yaw"] - yaw))
        return (k * ex, k * ey, max(-0.3, min(0.3, 2.0 * yerr)))
    if d > HOLD_ENGAGE:
        BASE_HOLD["active"] = True
    elif d < HOLD_RELEASE:
        BASE_HOLD["active"] = False
    if not BASE_HOLD["active"]:
        return (0.0, 0.0, 0.0)
    BASE_HOLD["engaged_steps"] += 1
    ex = math.cos(yaw) * float(e[0]) + math.sin(yaw) * float(e[1])
    ey = -math.sin(yaw) * float(e[0]) + math.cos(yaw) * float(e[1])
    k = MIN_SPEED / max(d, 1e-6)
    return (k * ex, k * ey, 0.0)


PHASE = {"name": "idle"}  # recorded with every frame: walk / backoff / dock / pick / deliver / idle
SIMCLK = {"t": 0.0}  # sim seconds since start (progress events)
NAV2 = {"link": NAV2_LINK if args.nav2 else None, "t": 0.0, "wall0": 0.0, "sim0": 0.0, "pace": False,
        "ready": False}  # --nav2: the ROS 2 link and the sim clock
# FETCH_TRACE=<file.npz>: per-step trace of the pick (arm, hand, base, hold command) for diagnosis
TRACE = {"path": os.environ.get("FETCH_TRACE", ""), "rows": [], "sub": "", "ep": 0}


def step(obs):
    if BASE_HOLD["xy"] is not None:  # braked: no drive command either, the policy just stands
        vel.vel_command_b[:] = torch.tensor((0.0, 0.0, 0.0) if WHEEL_BRAKE else base_hold_command(), device=dev)
    with torch.inference_mode():
        obs, *_ = env.step(act(obs))
    SIMCLK["t"] += dt
    if DOOR:
        door_latch_update()
    PROG.tick(SIMCLK["t"], PHASE["name"] + (f":{TRACE['sub']}" if PHASE["name"] in ("pick", "place") and TRACE["sub"] else ""),
              progress_beat)
    if NAV2["link"] is not None:
        nav2_tick()
    if TRACE["path"] and PHASE["name"] == "pick":
        d = robot.data
        hand_w = d.body_pos_w[0, BODY] + mu.quat_apply(d.body_quat_w[0:1, BODY], OFFSET[None] if OFFSET.dim() == 1 else OFFSET)[0]
        TRACE["rows"].append((TRACE["ep"], TRACE["sub"], *hand_w.tolist(), *d.root_pos_w[0].tolist(),
                              *d.root_lin_vel_b[0].tolist(), *d.root_ang_vel_b[0].tolist(),
                              *vel.vel_command_b[0].tolist(), *state["arm"].tolist(), *d.joint_pos[0, ARM].tolist(),
                              *d.joint_vel[0, ARM].tolist()))
    if REC is not None:
        REC.step(vel.vel_command_b[0].tolist(), state["arm"].cpu().numpy(), state["grip"], PHASE["name"], current_plan())
    if frames["on"]:
        img = u.render()
        frames["writer"].append_data(img)
        frames["n"] += 1
        if args.stills and frames["n"] % args.stills == 1:
            import imageio.v2 as imageio_
            sd = f"{args.video_dir}/stills_seed{args.seed}_ep{TRACE['ep']}_{args.cam}"
            os.makedirs(sd, exist_ok=True)
            imageio_.imwrite(f"{sd}/{frames['n']:05d}_{PHASE['name']}.png", img)
            rp, rq = robot.data.root_pos_w[0], robot.data.root_quat_w[0]
            hand = robot.data.body_pos_w[0, BODY]
            with open(f"{sd}/poses.jsonl", "a") as fh:  # where the robot was in each still (chronophotographs)
                fh.write(json.dumps({"file": f"{frames['n']:05d}_{PHASE['name']}.png", "frame": frames["n"],
                                     "phase": PHASE["name"], "x": round(float(rp[0]), 4), "y": round(float(rp[1]), 4),
                                     "z": round(float(rp[2]), 4),
                                     "yaw": round(math.atan2(2 * float(rq[0] * rq[3] + rq[1] * rq[2]),
                                                             1 - 2 * float(rq[2] ** 2 + rq[3] ** 2)), 4),
                                     "hand": [round(float(v), 4) for v in hand]}) + "\n")
    if watch["on"]:
        watch["k"] += 1
        if watch["k"] % 10 == 0:  # every 0.2 s: box offset from the grasp point, gripper angle, jaw forces
            f = touching(0.5)
            watch["trace"].append((round(float((box_in_base()[0] - ee_pose()[0]).norm()) * 100, 1),
                                   round(float(robot.data.joint_pos[0, GRIP]), 2),
                                   f.get("gripperStator", 0), f.get("gripperMover", 0)))
    if args.real_time:
        time.sleep(dt)
    return obs


def current_plan():
    """The path the robot is following now, world frame (the recorder's plan label): DualVLN's trajectory
    while walking, Nav2's global plan while Nav2 drives (to the staging pose, delivering), the straight
    line to the pick point while docking; None otherwise."""
    ph = PHASE["name"]
    if ph == "walk" and nav is not None:
        return nav.current_path
    if NAV2["link"] is None:
        return None
    if ph == "nav2_dock" and NAV2.get("goal") is not None:
        p, _ = base_pose()
        return [(float(p[0]), float(p[1])), tuple(NAV2["goal"][:2])]
    if ph in ("nav2", "deliver") and NAV2["pace"]:
        return NAV2["link"].plan_points()
    return None


def progress_beat():
    """Every 2 s of sim time, for ./process.sh: where the robot is and how far it still has to go."""
    p, yaw = base_pose()
    v = robot.data.root_lin_vel_b[0]
    out = {"x": float(p[0]), "y": float(p[1]), "yaw_deg": round(math.degrees(yaw), 1),
           "speed": float(math.hypot(float(v[0]), float(v[1]))), "table_m": vln_nav_edge(float(p[0]), float(p[1]))}
    goal = NAV2.get("goal")
    if goal is not None and PHASE["name"] == "nav2":
        out["goal_m"] = math.hypot(float(p[0]) - goal[0], float(p[1]) - goal[1])
        out["goal_yaw_err_deg"] = round(math.degrees(math.atan2(math.sin(goal[2] - yaw), math.cos(goal[2] - yaw))), 1)
    return out


def nav2_tick():
    """Publish clock / TF / odom / scan for Nav2; while Nav2 drives, keep sim time from running ahead of
    the wall clock (Nav2's timers run on sim time, and CPU physics runs ~3x real time)."""
    NAV2["t"] += dt
    p, yaw = base_pose()
    v = robot.data.root_lin_vel_b[0]
    NAV2["link"].tick(NAV2["t"], float(p[0]), float(p[1]), yaw, (float(v[0]), float(v[1])),
                      float(robot.data.root_ang_vel_b[0, 2]))
    if NAV2["pace"]:
        ahead = (NAV2["t"] - NAV2["sim0"]) - (time.time() - NAV2["wall0"])
        if ahead > 0:
            time.sleep(ahead)


def grip_move(obs, target, seconds, hold_s):
    """Ramp the gripper target linearly to `target` (rad), then hold."""
    g0, n = state["grip"], max(1, int(seconds / dt))
    for k in range(n):
        state["grip"] = g0 + (k + 1) / n * (target - g0)
        obs = step(obs)
    return hold(obs, hold_s)


def hold(obs, seconds):
    for _ in range(int(seconds / dt)):
        obs = step(obs)
    return obs


def q_arm():
    return robot.data.joint_pos[0, ARM].clone()


def base_pose():
    p = robot.data.root_pos_w[0]
    yaw = float(mu.euler_xyz_from_quat(robot.data.root_quat_w)[2][0])
    return p, math.atan2(math.sin(yaw), math.cos(yaw))


def tilt_deg():
    g = robot.data.projected_gravity_b[0]
    return math.degrees(math.acos(max(-1.0, min(1.0, float(-g[2])))))


def quat_inv_tilt():
    """The inverse of the body's roll and pitch (its orientation without the heading), (1, 4): a hand orientation
    meant in the level, heading-aligned frame, premultiplied by it, is that orientation in the base frame. After
    docking the body leaned 3-4 deg towards the shelf, and a nearly level approach to the high tray became one
    3-4 deg steeper, out of reach."""
    q = robot.data.root_quat_w[0]
    w, x, y, z = (float(v) for v in q)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    q_yaw = torch.tensor([[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]], device=dev)
    q_tilt = mu.quat_mul(mu.quat_inv(q_yaw), q[None])
    return mu.quat_inv(q_tilt)


def box_in_base():
    """Box centre and yaw, in the robot's base frame."""
    pos_b, quat_b = mu.subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w,
                                                 box.data.root_pos_w, box.data.root_quat_w)
    return pos_b[0], float(mu.euler_xyz_from_quat(quat_b)[2][0])


def ee_pose():
    """Grasp point (between the fingertips) and gripper orientation, in the base frame."""
    p_b, q_b = mu.subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w,
                                            robot.data.body_pos_w[:, BODY], robot.data.body_quat_w[:, BODY])
    return (p_b + mu.quat_apply(q_b, OFFSET))[0], q_b[0]


def jacobian():
    """6x6 Jacobian of the grasp point in the base frame. The offset is rotated into
    the base frame before its lever arm is applied (Isaac Lab's DifferentialIK action
    uses the offset in the body frame, which is only right when the two align)."""
    j = robot.root_physx_view.get_jacobians()[:, BODY, :, JAC_COLS].clone()
    r_inv = mu.matrix_from_quat(mu.quat_inv(robot.data.root_quat_w))
    j[:, :3] = torch.bmm(r_inv, j[:, :3])
    j[:, 3:] = torch.bmm(r_inv, j[:, 3:])
    _, q_b = mu.subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w,
                                          robot.data.body_pos_w[:, BODY], robot.data.body_quat_w[:, BODY])
    lever = mu.quat_apply(q_b, OFFSET)
    j[:, :3] += torch.bmm(-mu.skew_symmetric_matrix(lever), j[:, 3:])
    return j[0]


def ik_target(p_goal, q_goal):
    """One damped-least-squares step toward the goal pose, ACCUMULATED on the joint
    targets (integral action). Adding the step to the MEASURED joints instead (as Isaac
    Lab's IK action does) leaves a steady error: the PD drive needs a standing joint
    error to hold the arm against gravity, which showed up as ~3 cm at the fingertips.

    Secondary task in the null space: pull the joints toward the nominal (ready) posture
    and away from the limits. Without it, a plain 12 cm descend drifted into a folded
    solution against joint2/joint3's limits (joint1 swung 40 deg back) and missed by
    23-29 cm (2 of 8 side episodes). The pull cannot move the hand: N = I - J+ J."""
    p, q = ee_pose()
    pe, re = mu.compute_pose_error(p[None], q[None], p_goal[None], q_goal[None], rot_error_type="axis_angle")
    err = torch.cat([pe[0], re[0]])
    if args.ik_deadband and float(pe[0].norm()) < args.ik_deadband and float(re[0].norm()) < 4 * args.ik_deadband:
        return state["arm"].clone()
    j = jacobian()
    j_pinv = j.T @ torch.linalg.inv(j @ j.T + IK_LAMBDA ** 2 * torch.eye(6, device=dev))
    qm = q_arm()
    q_nom = torch.tensor(IK_NOMINAL["q"], device=dev) if IK_NOMINAL["q"] is not None else qm
    span = LIM[:, 1] - LIM[:, 0]
    to_nom = (q_nom - qm) / span                                        # posture, normalised by range
    margin = 0.15 * span
    push_off = torch.clamp(LIM[:, 0] + margin - qm, min=0) - torch.clamp(qm - (LIM[:, 1] - margin), min=0)
    secondary = IK_NULL_GAIN * to_nom * span + push_off
    dq = j_pinv @ err + (torch.eye(6, device=dev) - j_pinv @ j) @ secondary
    dq = (IK_GAIN * dq).clamp(-IK_MAX_STEP, IK_MAX_STEP)
    target = (state["arm"] + dq).clamp(qm - IK_WINDUP, qm + IK_WINDUP)
    return target.clamp(LIM[:, 0], LIM[:, 1])


def pose_err(p_goal, q_goal):
    p, q = ee_pose()
    return float((p - p_goal).norm()), float(mu.quat_error_magnitude(q[None], q_goal[None])[0])


def touching(min_n=2.0):
    """Robot links (other than wheels) feeling a contact force, in N."""
    cs = u.scene.sensors.get("contact_forces")
    if cs is None:
        return {}
    f = cs.data.net_forces_w[0].norm(dim=-1)
    return {n: round(float(v), 1) for n, v in zip(cs.body_names, f)
            if float(v) > min_n and not n.endswith("_foot")}


MOVER = robot.find_bodies(["gripperMover"])[0][0]


def fingers_in_box():
    """Fixed-jaw tip, moving-jaw tip and grasp point, in the BOX frame (cm): x, y
    horizontal along the box sides, z up from the box centre."""
    out = {}
    pts = {"fixed_tip": (BODY, (0.14, 0.0, -0.02)), "mover_tip": (MOVER, (0.09, 0.0, 0.0)),
           "grasp_pt": (BODY, tuple(F.GRASP_POINT))}
    for name, (b, off) in pts.items():
        pw = robot.data.body_pos_w[:, b] + mu.quat_apply(robot.data.body_quat_w[:, b], torch.tensor([off], device=dev))
        pb, _ = mu.subtract_frame_transforms(box.data.root_pos_w, box.data.root_quat_w, pw)
        out[name] = [round(float(v) * 100, 1) for v in pb[0]]
    stator_z_w = mu.quat_apply(robot.data.body_quat_w[:, BODY], torch.tensor([[0.0, 0.0, 1.0]], device=dev))
    out["closing_axis_in_box"] = [round(float(v), 2) for v in mu.quat_apply(mu.quat_inv(box.data.root_quat_w), stator_z_w)[0]]
    return out


def hip_partners():
    """Front-hip contact force per partner (N), from the filtered diagnostic sensor."""
    cs = u.scene.sensors.get("hip_contacts")
    if cs is None or cs.data.force_matrix_w is None:
        return {}
    fm = cs.data.force_matrix_w[0].norm(dim=-1)  # (hips, partners)
    out0 = {"net": round(float(cs.data.net_forces_w[0].norm(dim=-1).max()), 1), "matrix_sum": round(float(fm.sum()), 1)}
    names = [p.split("/")[-1] for p in cfg.scene.hip_contacts.filter_prim_paths_expr]
    out = dict(out0)
    for i, hip in enumerate(cs.body_names):
        for j in range(fm.shape[1]):
            if float(fm[i, j]) > 1.0:
                out[f"{hip}-{names[j] if j < len(names) else j}"] = round(float(fm[i, j]), 1)
    return out


def box_lift():
    """Box bottom above the surface it stood on (m): the table top, or the shelf tray."""
    return float(box.data.root_pos_w[0, 2]) - (TGT["top"] + F.BOX_SIZE[2] / 2)


def grasp_quat(yaw_rel, pitch_deg, flip=None):
    """Approach axis (stator x) pitched pitch_deg below horizontal, pointing along the arm's
    working direction ARM_YAW in the base frame (0 = ahead, -pi/2 = to the right); jaws
    closing horizontally across it (stator z), so the pitch keeps them parallel to the
    box's side faces. Turned to the box's yaw (the box is symmetric under 90 deg turns:
    use the equivalent yaw nearest the working direction). flip (default TGT["flip"]): the gripper rolled 180 deg
    about the approach -- the same grasp, but the wrist RealSense hangs down and back instead of riding up and
    forward, where a shelf's tray above stops it."""
    flip = TGT["flip"] if flip is None else flip
    y = ARM_YAW + ((yaw_rel - ARM_YAW + math.pi / 4) % (math.pi / 2) - math.pi / 4)
    ph = math.radians(pitch_deg)
    a = torch.tensor([math.cos(ph), 0.0, -math.sin(ph)], device=dev)
    c = torch.tensor([0.0, -1.0 if flip else 1.0, 0.0], device=dev)
    rot = torch.stack([a, torch.cross(c, a, dim=0), c], dim=1)  # columns = stator x, y, z
    cz, sz = math.cos(y), math.sin(y)
    rz = torch.tensor([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], device=dev)
    return mu.quat_from_matrix(rz @ rot)


def joint_move(obs, q_goal, seconds, hold_s=0.3):
    """Min-jerk in joint space from the current target to q_goal."""
    q0 = state["arm"].clone()
    q_goal = torch.as_tensor(q_goal, device=dev, dtype=torch.float32).clamp(LIM[:, 0], LIM[:, 1])
    n = max(1, int(seconds / dt))
    for k in range(n):
        s = (k + 1) / n
        s = s * s * s * (10 - 15 * s + 6 * s * s)
        state["arm"] = q0 + s * (q_goal - q0)
        obs = step(obs)
    return hold(obs, hold_s)


def slerp(q0, q1, s):
    d = float(torch.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + s * (q1 - q0)
    else:
        th = math.acos(d)
        q = (math.sin((1 - s) * th) * q0 + math.sin(s * th) * q1) / math.sin(th)
    return q / q.norm()


def ik_move(obs, goal_fn, seconds, hold_s=0.5, tol=0.005, settle_max=1.5):
    """Cartesian min-jerk from the current grasp-point pose to goal_fn() (re-evaluated each
    step, so a drifting base or moving box is tracked), IK every step, then hold -- and
    keep servoing (up to settle_max s more) until within tol: a fixed hold left one
    wrist orientation 1.8 cm short, enough for the box to escape the jaws."""
    p0, q0 = ee_pose()
    n, n_hold = max(1, int(seconds / dt)), int(hold_s / dt)
    for k in range(n + n_hold):
        gp, gq = goal_fn()
        s = min(1.0, (k + 1) / n)
        s = s * s * s * (10 - 15 * s + 6 * s * s)
        state["arm"] = ik_target(p0 + s * (gp - p0), slerp(q0, gq, s))
        obs = step(obs)
    for _ in range(int(settle_max / dt)):
        if pose_err(*goal_fn())[0] < tol:
            break
        state["arm"] = ik_target(*goal_fn())
        obs = step(obs)
    return (obs,) + pose_err(*goal_fn())


# ------------------------------------------------------------------ MoveIt
# MoveIt plans in the arm's base frame (link00, fixed on the body); the simulator executes the trajectories.
LINK00 = robot.find_bodies(["link00"])[0][0]
MOVEIT = None
if args.arm == "moveit":
    import moveit_link
    MOVEIT = moveit_link.MoveIt(args.moveit_server)
    if not MOVEIT.health():
        raise SystemExit(f"MoveIt is not answering at {args.moveit_server}: start it (python3 run.py moveit start "
                         "or moveit/run_moveit.sh), or pick with --arm ik")
KI_ARM, KI_SETTLE, INT_MAX = 0.03, 0.08, 0.15  # joint-space integral (per control step) and its bound (rad)
MI = {"int": None}  # the integral term: the standing joint error the PD drives need against gravity


def l00_frame():
    return robot.data.body_pos_w[:, LINK00], robot.data.body_quat_w[:, LINK00]


def world_to_l00(p_w, q_w):
    p, q = mu.subtract_frame_transforms(*l00_frame(), p_w.reshape(1, 3), q_w.reshape(1, 4))
    return [float(v) for v in p[0]] + [float(v) for v in q[0]]


def base_to_l00(p_b, q_b):
    """A pose in the B2W base frame -> link00 (via the world: base -> world -> link00)."""
    p_w, q_w = mu.combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, p_b.reshape(1, 3),
                                           q_b.reshape(1, 4))
    return world_to_l00(p_w[0], q_w[0])


def grasp_point_w():
    p = robot.data.body_pos_w[0, BODY] + mu.quat_apply(robot.data.body_quat_w[:, BODY], OFFSET)[0]
    return p, robot.data.body_quat_w[0, BODY]


def moveit_scene(with_target=True, attach_target=False):
    """The planning scene in link00: every table top, the boxes (the target unless it is in the gripper or about
    to be), the robot's own body and the floor; with attach_target, the box rides on the grasp point."""
    yaw0 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev)
    objs = []
    if SHELF:  # the shelf's trays, lips and frame; the table and what stands on it (scene file, world boxes)
        for kind, group in (("shelf", F.SHELVES), ("table", F.TABLES)):
            for gname, grp in group.items():
                for name, (x0, y0, z0, x1, y1, z1) in grp["collision"].items():
                    c = torch.tensor([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2], device=dev)
                    objs.append((f"{gname}_{kind}_{name}", world_to_l00(c, yaw0), (x1 - x0, y1 - y0, z1 - z0)))
    else:
        for i, (x0, x1, y0, y1) in enumerate(TABLE_RECTS):
            c = torch.tensor([(x0 + x1) / 2, (y0 + y1) / 2, F.TABLE_HEIGHT - F.TABLE_TOP_THICKNESS / 2], device=dev)
            objs.append((f"table{i}", world_to_l00(c, yaw0), (x1 - x0, y1 - y0, F.TABLE_TOP_THICKNESS)))
    others = list(BOXES.values()) if (MULTI or SHELF) else [box]
    for i, b in enumerate(others):
        if b is box and (attach_target or not with_target):
            continue
        objs.append((f"box{i}", world_to_l00(b.data.root_pos_w[0], b.data.root_quat_w[0]), F.BOX_SIZE))
    for name, c, size in moveit_link.BODY_BOXES:
        objs.append((name, base_to_l00(torch.tensor(c, device=dev), yaw0), size))
    floor = torch.tensor([float(robot.data.root_pos_w[0, 0]), float(robot.data.root_pos_w[0, 1]), -0.05], device=dev)
    objs.append(("floor", world_to_l00(floor, yaw0), (6.0, 6.0, 0.1)))
    attached = []
    if attach_target:
        gp, gq = grasp_point_w()
        p, q = mu.subtract_frame_transforms(gp[None], gq[None], box.data.root_pos_w, box.data.root_quat_w)
        attached.append(("held_box", "grasp_point", [float(v) for v in p[0]] + [float(v) for v in q[0]], F.BOX_SIZE))
    r = MOVEIT.scene(objs, attached)
    if not r.get("ok"):
        raise RuntimeError(f"MoveIt planning scene: {r.get('error')}")


def run_traj(obs, reply, settle_tol=0.004, settle_max=1.5, hold_s=0.2):
    """Follow a MoveIt joint trajectory (time-parameterised) at the control rate, linear between its points.

    The arm's PD drives need a standing joint error to hold the arm against gravity (~3 cm at the fingertips on
    the IK pick), so the targets carry an integral term, as a robot's joint trajectory controller would; at the
    end the arm settles on the last point (up to settle_max s). Returns (obs, worst joint error, rad)."""
    pts = reply["points"]
    times = [p[0] for p in pts]
    qs = torch.tensor([p[1:] for p in pts], device=dev, dtype=torch.float32)
    if MI["int"] is None:
        MI["int"] = torch.zeros(6, device=dev)

    def drive(q_ref, ki):
        MI["int"] = (MI["int"] + ki * (q_ref - q_arm())).clamp(-INT_MAX, INT_MAX)
        state["arm"] = (q_ref + MI["int"]).clamp(LIM[:, 0], LIM[:, 1])

    k_seg, n = 1, max(1, math.ceil(times[-1] / dt))
    for k in range(n + 1):
        t = min(k * dt, times[-1])
        while k_seg < len(times) - 1 and times[k_seg] < t:
            k_seg += 1
        if len(times) == 1:
            q_ref = qs[0]
        else:
            t0, t1 = times[k_seg - 1], times[k_seg]
            s = 1.0 if t1 <= t0 else min(1.0, max(0.0, (t - t0) / (t1 - t0)))
            q_ref = qs[k_seg - 1] + s * (qs[k_seg] - qs[k_seg - 1])
        drive(q_ref, KI_ARM)
        obs = step(obs)
    q_ref = qs[-1]
    for _ in range(int(settle_max / dt)):
        if float((q_ref - q_arm()).abs().max()) < settle_tol:
            break
        drive(q_ref, KI_SETTLE)
        obs = step(obs)
    obs = hold(obs, hold_s)
    return obs, float((q_ref - q_arm()).abs().max())


def moveit_call(st, stage, fn, *a, **kw):
    """One MoveIt request, timed and logged under st['moveit'][stage]; None if it failed."""
    r = fn(*a, **kw)
    info = {k: r.get(k) for k in ("ok", "code", "fraction", "plan_time", "tries", "tried", "found", "server_s", "error")
            if k in r}
    if r.get("points"):
        info["duration_s"] = round(r["points"][-1][0], 2)
        info["points"] = len(r["points"])
    st.setdefault("moveit", {})[stage] = info
    if not r.get("ok"):
        if r.get("code") == 99999 and a:  # failed at once: usually the start state is in collision -- with what?
            c = MOVEIT.check(a[0], kw.get("gripper", 0.0))
            info["start_contacts"] = c.get("contacts")
        say(f"MoveIt {stage} failed: {info}")
        return None
    return r


def pick_moveit(obs, rec, pitch):
    """The side pick with MoveIt: crouch; MoveIt plans the arm from wherever it is to the pre-grasp pose around the
    table, the boxes and the robot's body; open; a straight-line MoveIt approach to the grasp; close; a straight
    lift; then MoveIt to the carry pose with the box attached to the gripper. The robot base is held as in the
    IK pick, and the same measurements are recorded, so the two can be compared (evaluation/summarize_fetch.py).

    From a shelf (scene layout "shelf"), the scene file sets the approach pitch, the pre-grasp distance and the lift,
    and after the lift a straight pull-out (the box attached) takes the box out under the tray above; no crouch."""
    PHASE["name"] = "pick"
    st = rec["stages"]
    speed = args.moveit_speed
    pitches = [pitch] if TGT["pitch"] is None else ([TGT["pitch"]] if not isinstance(TGT["pitch"], list) else TGT["pitch"])
    pitch = pitches[0]
    GRASPED["pitch"], GRASPED["flip"], GRASPED["carry"] = pitch, TGT["flip"], TGT["carry"]
    GRASPED["place_depth"] = TGT["place_depth"]
    pre_dist = TGT["pre_dist"] or PRE_DIST
    lift_h = TGT["lift"] or LIFT

    def goal(back, up):
        bpos, byaw = box_in_base()
        q = grasp_quat(byaw, pitch)
        if SHELF:  # pitched below the level, not below the body (which leans a few degrees after docking)
            q = mu.quat_mul(quat_inv_tilt(), q[None])[0]
        approach = mu.quat_apply(q[None], torch.tensor([[1.0, 0.0, 0.0]], device=dev))[0]
        return bpos + torch.tensor([0.0, 0.0, GRASP_RAISE + up], device=dev) - back * approach, q

    def failed(stage):
        """No plan: fill the stages the summary reads, so the episode counts as a failed pick, not a crash."""
        st["moveit"]["failed_at"] = stage
        e, r_ = pose_err(*goal(0.0, 0.0))
        st.setdefault("descend", {"pos_err_cm": round(e * 100, 1), "rot_err_deg": round(math.degrees(r_), 1),
                                  "box_moved_cm": 0.0})
        st.setdefault("close", {"gripper_rad": round(float(robot.data.joint_pos[0, GRIP]), 3)})
        st["carry"] = {"box_lift_cm": round(box_lift() * 100, 1),
                       "box_to_grasp_cm": round(float((box_in_base()[0] - ee_pose()[0]).norm()) * 100, 1)}
        rec["success"] = False
        BASE_HOLD["xy"] = None
        vel.vel_command_b[:] = 0
        return obs

    BASE_HOLD["xy"], BASE_HOLD["active"], BASE_HOLD["engaged_steps"] = robot.data.root_pos_w[0, :2].clone(), False, 0
    BASE_HOLD["yaw"] = base_pose()[1]
    st["moveit"] = {}
    MI["int"] = None  # the gravity offset of the last episode's carry pose does not fit the stowed arm
    TRACE["sub"] = "crouch"  # the body's height for this box: crouched (1) for a low one, taller (< 0) for a high one
    stance = (TGT["stance"] if TGT["stance"] is not None else (1.0 if TGT["crouch"] else 0.0)) if CROUCH else 0.0
    if stance:
        obs = set_crouch(obs, stance, 1.5)
    st["stance"] = stance
    base0, yaw_b0 = robot.data.root_pos_w[0, :2].clone(), base_pose()[1]
    # MoveIt's model against the simulator's: the grasp point for the measured joints, both in link00 (mm)
    fk = MOVEIT.fk(q_arm().tolist(), float(state["grip"]))
    if fk.get("ok"):
        gp, gq = grasp_point_w()
        sim = world_to_l00(gp, gq)
        st["moveit"]["fk_check_mm"] = round(1000 * math.dist(fk["pose"][:3], sim[:3]), 2)

    TRACE["sub"] = "pre_grasp"  # free motion: from the stowed arm, around the body and the table
    gp_b, gq_b = goal(pre_dist, 0.0)
    if SHELF:  # first the pre-grasp posture from which the straight approach into the shelf works, then plan there;
        moveit_scene(with_target=False)  # the scene's pitches in turn (a turned box can be out of reach at one)
        bp_b, byaw_b = box_in_base()  # where the arm sees the box: what decides reach at a high tray
        rq = robot.data.root_quat_w[0]
        roll_, pitch_, _ = (math.degrees(float(v)) for v in mu.euler_xyz_from_quat(rq[None]))
        st["moveit"]["approach_ctx"] = {"box_b": [round(float(v), 3) for v in bp_b],
                                        "box_yaw_rel_deg": round(math.degrees(float(byaw_b) - ARM_YAW), 1),
                                        "base_z": round(float(robot.data.root_pos_w[0, 2]), 3),
                                        "roll_deg": round((roll_ + 180) % 360 - 180, 2),
                                        "pitch_deg": round((pitch_ + 180) % 360 - 180, 2)}
        for pitch in pitches:
            gp_b, gq_b = goal(pre_dist, 0.0)
            g_b, g_q = goal(0.0, 0.0)
            ap = moveit_call(st, "approach_ik", MOVEIT.approach, q_arm().tolist(), base_to_l00(gp_b, gq_b),
                             base_to_l00(g_b, g_q), gripper=float(state["grip"]), approach_gripper=F.GRIPPER_OPEN, tries=32)
            if ap is not None:
                break
        if ap is None:
            return failed("approach_ik")
        GRASPED["pitch"] = pitch
        st["moveit"]["pitch"] = pitch
        moveit_scene(with_target=True)
        r = moveit_call(st, "pre_grasp", MOVEIT.plan_joints, q_arm().tolist(), ap["joints"], gripper=float(state["grip"]),
                        time=5.0, attempts=5, vel=speed, acc=speed)
    else:
        moveit_scene(with_target=True)
        r = moveit_call(st, "pre_grasp", MOVEIT.plan_pose, q_arm().tolist(), base_to_l00(gp_b, gq_b),
                        gripper=float(state["grip"]), pos_tol=0.003, rot_tol=0.03, time=5.0, attempts=5,
                        vel=speed, acc=speed)
    if r is None:
        return failed("pre_grasp")
    obs, jerr = run_traj(obs, r, hold_s=0.5)
    e, rr = pose_err(*goal(pre_dist, 0.0))
    st["pre_grasp"] = {"pos_err_cm": round(e * 100, 1), "rot_err_deg": round(math.degrees(rr), 1),
                       "joint_err_deg": round(math.degrees(jerr), 2), "joints": [round(float(v), 2) for v in q_arm()]}
    TRACE["sub"] = "open"
    obs = grip_move(obs, F.GRIPPER_OPEN, 0.6, 0.3)
    st["fingers_in_box_frame_cm"] = fingers_in_box()
    st["open"] = {"gripper_rad": round(float(robot.data.joint_pos[0, GRIP]), 3)}
    box0 = box.data.root_pos_w[0].clone()
    yaw0 = box_in_base()[1]

    TRACE["sub"] = "descend"  # straight line to the grasp; the target box is out of the scene (the jaws close on it)
    moveit_scene(with_target=False)
    gp_b, gq_b = goal(0.0, 0.0)
    r = moveit_call(st, "descend", MOVEIT.cartesian, q_arm().tolist(), [base_to_l00(gp_b, gq_b)],
                    gripper=float(state["grip"]), max_step=0.005, avoid_collisions=True, vel=speed / 2, acc=speed / 2)
    if r is None and SHELF and ap is not None:
        # no first step: the arm stopped a fraction of a degree off the pre-pose /approach checked, and MoveIt now
        # finds it touching the shelf (a green box, steeply under the 0.9 m tray: 1 of 6). /approach checked this
        # straight line from the planned joints; within tracking error of them, follow it unchecked
        dq = max(abs(float(a) - float(b)) for a, b in zip(q_arm().tolist(), ap["joints"]))
        chk = MOVEIT.check(q_arm().tolist(), gripper=float(state["grip"]))
        st["moveit"]["descend_start"] = {"valid": chk.get("valid"), "contacts": chk.get("contacts"),
                                         "off_planned_deg": round(math.degrees(dq), 2)}
        if dq < math.radians(1.0):
            r = moveit_call(st, "descend_unchecked", MOVEIT.cartesian, q_arm().tolist(), [base_to_l00(gp_b, gq_b)],
                            gripper=float(state["grip"]), max_step=0.005, avoid_collisions=False, vel=speed / 2,
                            acc=speed / 2)
    if r is None:
        return failed("descend")
    obs, jerr = run_traj(obs, r, hold_s=0.5)
    # the descent is planned once; if the base drifted meanwhile, correct with short straight MoveIt moves
    # (what the IK pick does continuously)
    st["moveit"]["corrections"] = []
    for _ in range(2):
        e, _ = pose_err(*goal(0.0, 0.0))
        if e < 0.004:
            break
        gp_b, gq_b = goal(0.0, 0.0)
        r = moveit_call(st, "descend_fix", MOVEIT.cartesian, q_arm().tolist(), [base_to_l00(gp_b, gq_b)],
                        gripper=float(state["grip"]), max_step=0.002, avoid_collisions=True, vel=speed / 3,
                        acc=speed / 3)
        if r is None:
            break
        obs, jerr = run_traj(obs, r, hold_s=0.3)
        st["moveit"]["corrections"].append(round(e * 100, 2))
    e, rr = pose_err(*goal(0.0, 0.0))
    st["hip_partners_N"] = hip_partners()
    st["descend"] = {"pos_err_cm": round(e * 100, 1), "rot_err_deg": round(math.degrees(rr), 1),
                     "joint_err_deg": round(math.degrees(jerr), 2),
                     "joints": [round(float(v), 2) for v in q_arm()], "touching": touching(),
                     "base_moved_cm": round(float((robot.data.root_pos_w[0, :2] - base0).norm()) * 100, 1),
                     "base_turned_deg": round(math.degrees(base_pose()[1] - yaw_b0), 1),
                     "box_moved_cm": round(float((box.data.root_pos_w[0] - box0).norm()) * 100, 1),
                     "box_yaw_deg": [round(math.degrees(yaw0), 1), round(math.degrees(box_in_base()[1]), 1)]}
    st["at_grasp_box_frame_cm"] = fingers_in_box()
    TRACE["sub"] = "close"
    obs = grip_move(obs, F.GRIPPER_CLOSED, CLOSE_TIME, 0.5)
    st["close"] = {"gripper_rad": round(float(robot.data.joint_pos[0, GRIP]), 3), "touching": touching(),
                   "box_to_grasp_cm": round(float((box_in_base()[0] - ee_pose()[0]).norm()) * 100, 1)}
    watch["on"] = True

    TRACE["sub"] = "lift"  # straight up; the box still rests on the table at the start, so no collision check
    p_now, q_now = ee_pose()
    up = base_to_l00(p_now + torch.tensor([0.0, 0.0, lift_h], device=dev), q_now)
    r = MOVEIT.cartesian(q_arm().tolist(), [up], gripper=float(state["grip"]), max_step=0.005, avoid_collisions=False,
                         vel=speed / 2, acc=speed / 2)
    st["moveit"]["lift"] = {k: r.get(k) for k in ("ok", "code", "fraction", "server_s", "error") if k in r}
    if not r.get("ok"):  # out of reach before the top (a high tray, the hand nearly level): enough if the box clears
        if not (SHELF and r.get("points") and r.get("fraction", 0.0) * lift_h >= SHELF_LIP + 0.004):  # the lip
            print(f"FETCH| MoveIt lift failed: {st['moveit']['lift']}", flush=True)
            return failed("lift")
    obs, _ = run_traj(obs, r, hold_s=0.3)
    st["lift"] = {"pos_err_cm": round(float((ee_pose()[0] - (p_now + torch.tensor([0.0, 0.0, lift_h], device=dev))).norm())
                                       * 100, 1), "box_lift_cm": round(box_lift() * 100, 1)}
    if TGT["retreat"] > 0:  # out of the shelf: straight back, level, the box attached (it must clear the lip)
        TRACE["sub"] = "retreat"
        moveit_scene(with_target=False, attach_target=True)
        p_now, q_now = ee_pose()
        a = mu.quat_apply(q_now[None], torch.tensor([[1.0, 0.0, 0.0]], device=dev))[0]
        a[2] = 0.0
        back = base_to_l00(p_now - TGT["retreat"] * a / a.norm(), q_now)
        # the arm may run out of reach before the end of the line: most of it (the box well clear) will do
        r = None
        for stage, avoid in (("retreat", True), ("retreat_unchecked", False)):
            r_ = MOVEIT.cartesian(q_arm().tolist(), [back], gripper=float(state["grip"]), max_step=0.005,
                                  avoid_collisions=avoid, vel=speed / 2, acc=speed / 2)
            st["moveit"][stage] = {k: r_.get(k) for k in ("ok", "code", "fraction", "server_s", "error") if k in r_}
            if r_.get("points") and r_.get("fraction", 0.0) >= 0.6:
                r = r_
                break
        if r is None:  # (unchecked: the box hangs a little low in the jaws and starts "in" the lip)
            return failed("retreat")
        obs, _ = run_traj(obs, r, hold_s=0.3)
        bw = box.data.root_pos_w[0]
        st["retreat"] = {"box_out_of_shelf_cm": round(100 * TGT["face"] * (float(bw[0]) - TGT["near_x"]), 1),
                         "box_lift_cm": round(box_lift() * 100, 1),
                         "box_to_grasp_cm": round(float((box_in_base()[0] - ee_pose()[0]).norm()) * 100, 1)}
    TRACE["sub"] = "standup"
    if stance:
        obs = set_crouch(obs, 0.0, 1.5)

    TRACE["sub"] = "carry_move"  # free motion to the carry pose, the box attached to the gripper
    moveit_scene(with_target=False, attach_target=True)
    cq = TGT["carry"] or CARRY_Q  # a shelf scene's own: the table pick's holds the box out to the side, into the shelf
    carry = [cq[0] + ARM_YAW] + list(cq[1:]) + [float(q_arm()[5])]
    r = moveit_call(st, "carry_move", MOVEIT.plan_joints, q_arm().tolist(), carry, gripper=float(state["grip"]),
                    time=5.0, attempts=5, vel=speed, acc=speed)
    for _ in range(2 if SHELF else 0):  # out of a shelf the hand can end low by the robot (the wrist beside the body
        if r is not None:                # boxes, from the low tray): up 6 cm at a time -- the box is out -- and again
                                         # (18 cm reached the tray above; unchecked, so no further)
            break
        p_now, q_now = ee_pose()
        r = moveit_call(st, "carry_raise", MOVEIT.cartesian, q_arm().tolist(),
                        [base_to_l00(p_now + torch.tensor([0.0, 0.0, 0.06], device=dev), q_now)],
                        gripper=float(state["grip"]), max_step=0.005, avoid_collisions=False, vel=speed / 2, acc=speed / 2)
        if r is None:
            break
        obs, _ = run_traj(obs, r, hold_s=0.2)
        st["moveit"]["carry_raised_cm"] = st["moveit"].get("carry_raised_cm", 0) + 6
        r = moveit_call(st, "carry_move", MOVEIT.plan_joints, q_arm().tolist(), carry, gripper=float(state["grip"]),
                        time=5.0, attempts=5, vel=speed, acc=speed)
    if r is None:
        return failed("carry_move")
    obs, _ = run_traj(obs, r, hold_s=0.5)
    short = CARRY_LIFT - box_lift()  # as in the IK pick: the carry pose holds the box as high as it sits in the jaws
    if short > 0.01 and not SHELF:  # (a box from a high tray is carried lower than it stood: nothing to top up)
        p_up, q_up = ee_pose()
        up = base_to_l00(p_up + torch.tensor([0.0, 0.0, short], device=dev), q_up)
        r = moveit_call(st, "carry_topup", MOVEIT.cartesian, q_arm().tolist(), [up], gripper=float(state["grip"]),
                        max_step=0.005, avoid_collisions=True, vel=speed / 2, acc=speed / 2)
        if r is not None:
            obs, _ = run_traj(obs, r, hold_s=0.3, settle_max=0.5)
        st["carry_topup"] = {"asked_cm": round(short * 100, 1)}
    TRACE["sub"] = "carry_hold"
    obs = hold(obs, 1.5)
    watch["on"] = False
    st["slip_trace"] = watch["trace"][:]
    watch["trace"].clear()
    bpos, _ = box_in_base()
    held = float((bpos - ee_pose()[0]).norm())
    lifted = box_lift()
    st["carry"] = {"box_lift_cm": round(lifted * 100, 1), "box_to_grasp_cm": round(held * 100, 1)}
    rec["success"] = bool(lifted >= 0.08 and held <= 0.06)
    if SHELF:  # held and out of the shelf (a box from a high tray is carried lower than it stood)
        st["carry"]["box_out_cm"] = round(100 * TGT["face"] * (float(box.data.root_pos_w[0, 0]) - TGT["near_x"]), 1)
        rec["success"] = bool(held <= 0.06 and st["carry"]["box_out_cm"] > 3.0)
    st["base_hold"] = {"engaged_s": round(BASE_HOLD["engaged_steps"] * dt, 1),
                       "drift_end_cm": round(float((robot.data.root_pos_w[0, :2] - BASE_HOLD["xy"]).norm()) * 100, 1)}
    BASE_HOLD["xy"] = None
    vel.vel_command_b[:] = 0
    return obs


def pick(obs, rec, pitch):
    rec["stages"]["arm"] = args.arm
    return (pick_moveit if args.arm == "moveit" else pick_ik)(obs, rec, pitch)


# ------------------------------------------------------------------ manipulation
def pick_ik(obs, rec, pitch):
    PHASE["name"] = "pick"
    st = rec["stages"]
    ph = math.radians(pitch)

    def goal(back, up):
        bpos, byaw = box_in_base()
        q = grasp_quat(byaw, pitch)
        approach = mu.quat_apply(q[None], torch.tensor([[1.0, 0.0, 0.0]], device=dev))[0]
        return bpos + torch.tensor([0.0, 0.0, GRASP_RAISE + up], device=dev) - back * approach, q

    BASE_HOLD["xy"], BASE_HOLD["active"], BASE_HOLD["engaged_steps"] = robot.data.root_pos_w[0, :2].clone(), False, 0
    BASE_HOLD["yaw"] = base_pose()[1]
    TRACE["sub"] = "crouch"
    if CROUCH:  # the box sits 15 cm further below the shoulder at ETH's stance: win back ~6 cm
        obs = set_crouch(obs, 1.0, 1.5)
    if args.approach == "side":
        ready, roll = ready_side()
    else:
        ready, roll = [READY[0] + ARM_YAW] + READY[1:], READY_ROLL
    TRACE["sub"] = "unfold"
    obs = joint_move(obs, list(ready) + [roll], 3.0)
    IK_NOMINAL["q"] = list(ready) + [roll]
    base0, yaw_b0 = robot.data.root_pos_w[0, :2].clone(), base_pose()[1]
    st["unfold"] = {"joints": [round(float(v), 2) for v in q_arm()], "tilt_deg": round(tilt_deg(), 1)}
    TRACE["sub"] = "pre_grasp"
    obs, e, r = ik_move(obs, lambda: goal(PRE_DIST, 0.0), 3.0, hold_s=1.0)
    st["pre_grasp"] = {"pos_err_cm": round(e * 100, 1), "rot_err_deg": round(math.degrees(r), 1),
                       "joints": [round(float(v), 2) for v in q_arm()]}
    TRACE["sub"] = "open"
    obs = grip_move(obs, F.GRIPPER_OPEN, 0.6, 0.3)
    st["fingers_in_box_frame_cm"] = fingers_in_box()
    st["open"] = {"gripper_rad": round(float(robot.data.joint_pos[0, GRIP]), 3)}
    box0 = box.data.root_pos_w[0].clone()
    yaw0 = box_in_base()[1]
    TRACE["sub"] = "descend"
    obs, e, r = ik_move(obs, lambda: goal(0.0, 0.0), 3.0, hold_s=0.8)
    st["hip_partners_N"] = hip_partners()
    st["descend"] = {"pos_err_cm": round(e * 100, 1), "rot_err_deg": round(math.degrees(r), 1),
                     "joints": [round(float(v), 2) for v in q_arm()], "touching": touching(),
                     "base_moved_cm": round(float((robot.data.root_pos_w[0, :2] - base0).norm()) * 100, 1),
                     "base_turned_deg": round(math.degrees(base_pose()[1] - yaw_b0), 1),
                     "box_moved_cm": round(float((box.data.root_pos_w[0] - box0).norm()) * 100, 1),
                     "box_yaw_deg": [round(math.degrees(yaw0), 1), round(math.degrees(box_in_base()[1]), 1)]}
    st["at_grasp_box_frame_cm"] = fingers_in_box()
    TRACE["sub"] = "close"
    obs = grip_move(obs, F.GRIPPER_CLOSED, CLOSE_TIME, 0.5)
    st["close"] = {"gripper_rad": round(float(robot.data.joint_pos[0, GRIP]), 3), "touching": touching(),
                   "box_to_grasp_cm": round(float((box_in_base()[0] - ee_pose()[0]).norm()) * 100, 1)}
    watch["on"] = True
    p_lift, q_lift = ee_pose()
    p_lift = p_lift + torch.tensor([0.0, 0.0, LIFT], device=dev)
    TRACE["sub"] = "lift"
    obs, e, r = ik_move(obs, lambda: (p_lift, q_lift), 1.2, hold_s=0.5)
    st["lift"] = {"pos_err_cm": round(e * 100, 1), "box_lift_cm": round(box_lift() * 100, 1)}
    TRACE["sub"] = "standup"
    if CROUCH:  # stand up again before carrying (and driving)
        obs = set_crouch(obs, 0.0, 1.5)
    TRACE["sub"] = "carry_move"
    obs = joint_move(obs, [CARRY_Q[0] + ARM_YAW] + CARRY_Q[1:] + [float(q_arm()[5])], 2.5, hold_s=0.5)
    # The carry pose is a fixed joint configuration, so how high it holds the box depends
    # on where the box sits in the gripper (with the robot docked at the right distance it
    # held two boxes only ~6 cm up). Top up to CARRY_LIFT with a straight IK raise.
    short = CARRY_LIFT - box_lift()
    if short > 0.01:
        p_up, q_up = ee_pose()
        p_up = p_up + torch.tensor([0.0, 0.0, short], device=dev)
        obs, e_up, _ = ik_move(obs, lambda: (p_up, q_up), 1.5, hold_s=0.3, settle_max=0.5)
        st["carry_topup"] = {"asked_cm": round(short * 100, 1), "pos_err_cm": round(e_up * 100, 1)}
    TRACE["sub"] = "carry_hold"
    obs = hold(obs, 1.5)
    watch["on"] = False
    st["slip_trace"] = watch["trace"][:]
    watch["trace"].clear()
    bpos, _ = box_in_base()
    held = float((bpos - ee_pose()[0]).norm())
    lifted = box_lift()
    st["carry"] = {"box_lift_cm": round(lifted * 100, 1), "box_to_grasp_cm": round(held * 100, 1)}
    rec["success"] = bool(lifted >= 0.08 and held <= 0.06)
    st["base_hold"] = {"engaged_s": round(BASE_HOLD["engaged_steps"] * dt, 1),
                       "drift_end_cm": round(float((robot.data.root_pos_w[0, :2] - BASE_HOLD["xy"]).norm()) * 100, 1)}
    BASE_HOLD["xy"] = None
    vel.vel_command_b[:] = 0
    return obs


# ------------------------------------------------------------------ locomotion
def dock_pose(bp_w):
    """(goal xy, pre-dock xy, heading) of the base for picking the box at bp_w."""
    if args.approach == "front":
        goal = torch.tensor([TGT["near_x"] - STANDOFF, float(bp_w[1])], device=dev)
        return goal, goal - torch.tensor([PREDOCK, 0.0], device=dev), 0.0
    gap = (TGT["gap"] if TGT["gap"] is not None else SIDE_GAP) + (NAV2_DOCK_EXTRA if args.nav2 and args.nav2_goal == "dock" else 0.0)
    face = TGT["face"]  # -1: the robot stands at -x of the edge (the tables, the shelf); +1: at +x (the side table)
    x_c = TGT["near_x"] + face * (HIP_HALF_WIDTH + gap)
    # the surface on the robot's right: facing +y when it is at +x of the robot (face -1), -y when at -x (face +1)
    sgn = (1.0 if SIDE["name"] == "right" else -1.0) * -face
    goal = torch.tensor([x_c, float(bp_w[1]) - sgn * (TGT["abeam"] or SIDE_ABEAM)], device=dev)
    predock = goal + torch.tensor([face * SIDE_PREDOCK_OUT, -sgn * SIDE_PREDOCK_BACK], device=dev)
    return goal, predock, sgn * math.pi / 2


TABLE_RECT = (F.TABLE_NEAR_EDGE_X, F.TABLE_NEAR_EDGE_X + F.TABLE_SIZE[0],
              F.TABLE_CENTER[1] - F.TABLE_SIZE[1] / 2, F.TABLE_CENTER[1] + F.TABLE_SIZE[1] / 2)
# every table top (the three-table scene has three; TABLE_RECT is the target's, set by set_target)
TABLE_RECTS = [F.table_rect3(t) for t in F.COLORS3] if MULTI else [TABLE_RECT]
PLACE_T = {"name": None}  # shelf scenes: the table the box goes on this episode
if SHELF:  # everything the scripted drive must keep clear of; the target shelf / table is the "table" (TABLE_RECT)
    STAGING_BACK = F.SCENE_SHELF.get("nav2_staging", STAGING_BACK)  # must match Nav2's docking server (make_params.py)
    TABLE_RECTS = [tuple(v) for v in F.SCENE_SHELF["obstacles"].values()]
    TABLE_RECT = tuple(next(iter(F.SHELVES.values()))["rect"])
elif DOOR:  # the obstacles keep random starts clear; the walk's goal is the target door (paint_doors)
    STAGING_BACK = F.SCENE_DOORS.get("nav2_staging", STAGING_BACK)
    TABLE_RECTS = [tuple(v) for v in F.SCENE_DOORS["obstacles"].values()]
if CAMS:
    vln_nav.TABLE_RECT = TABLE_RECT


def use_surface(s):
    """Shelf scene: dock at, and work the arm on, a shelf (F.SHELVES) or the table the box goes on (F.TABLES)."""
    global TABLE_RECT
    TGT.update(near_x=s["near_x"], face=s["face"], top=s["top"], sides=s.get("sides"), pitch=s.get("pitch"),
               pre_dist=s.get("pre_dist"), lift=s.get("lift"), retreat=s.get("retreat", 0.0), crouch=s.get("crouch", False),
               flip=s.get("jaw_flip", False), carry=s.get("carry"), abeam=s.get("abeam"),
               gap=s.get("gap"), stance=s.get("stance"), place_depth=s.get("place_depth"))
    TABLE_RECT = tuple(s["rect"])
    if CAMS:
        vln_nav.TABLE_RECT = TABLE_RECT


def shelf_surface(shelf, color):
    """A shelf's settings for one box: the shelf's, with those of the box's tray (its `levels`) on top."""
    sh = F.SHELVES[shelf]
    return {**sh, **sh.get("levels", {}).get(color, {})}


def sides_allowed():
    """The sides the robot may park on: the scene's (a shelf with another one right behind it: one side), else both
    (or --side)."""
    if TGT["sides"]:
        return tuple(TGT["sides"])
    return ("right", "left") if args.side == "auto" else (args.side,)


def nearest_table_rect(x, y):
    return min(TABLE_RECTS, key=lambda r: math.hypot(max(r[0] - x, 0.0, x - r[1]), max(r[2] - y, 0.0, y - r[3])))


def set_target(table, color, place=None):
    """Three-table scene (or a shelf scene: table = the shelf's name, place = the table the box goes on): the box to
    fetch -- every table- and box-dependent function follows it."""
    global box, TABLE_RECT
    if SHELF:
        TGT.update(table=table, box=color)
        box = BOXES[(table, color)]
        PLACE_T["name"] = place or PLACE_T["name"] or next(iter(F.TABLES))
        use_surface(shelf_surface(table, color))
        return
    TGT.update(table=table, box=color, near_x=F.table_rect3(table)[0])
    box = BOXES[(table, color)]
    TABLE_RECT = F.table_rect3(table)
    if CAMS:
        vln_nav.TABLE_RECT = TABLE_RECT


if SHELF:
    set_target(*next(iter(BOXES)))


def target_texts(table, color, rng=None):
    """(DualVLN's walking sentence, the task sentence) for a target."""
    nav = f"Walk to the {table} table and stop in front of it."
    if args.deliver:  # every wording says the box comes back (parse_task's DELIVER matches each)
        choices = [f"Bring me the {color} box from the {table} table.",
                   f"Pick up the {color} box on the {table} table and bring it to me.",
                   f"Go to the {table} table, grab the {color} box and bring it back.",
                   f"Fetch the {color} box from the {table} table and carry it to the drop zone."]
    else:  # pick only: no wording implies bringing it back ("fetch" does, so it is not used)
        choices = [f"Pick up the {color} box on the {table} table.",
                   f"Go to the {table} table and pick up the {color} box.",
                   f"Grab the {color} box from the {table} table.",
                   f"Lift the {color} box off the {table} table."]
    task = str(rng.choice(choices)) if (rng is not None and args.task_variants) else choices[0]
    return nav, task


def shelf_texts(shelf, color, table, rng=None):
    """(DualVLN's walking sentence, the task sentence) for a shelf scene: the shelf's and the table's labels from the
    scene file ("white shelf", "wooden table"; the one-shelf scene: "shelf", "side table")."""
    sh, tb = F.SHELVES[shelf]["label"], F.TABLES[table]["label"]
    nav = F.SHELVES[shelf].get("walk") or f"Walk to the {sh} and stop in front of it."  # the scene's own wording
    choices = [f"Take the {color} box from the {sh} and put it on the {tb}.",
               f"Bring the {color} box from the {sh} to the {tb}.",
               f"Move the {color} box from the {sh} to the {tb}.",
               f"Get the {color} box off the {sh} and place it on the {tb}."]
    task = str(rng.choice(choices)) if (rng is not None and args.task_variants) else choices[0]
    return nav, task


def parse_shelf_task(text):
    """'take the green box from the shelf and put it on the table' -> (shelf, colour, table); None for
    what the sentence does not say (a scene with one shelf or one table needs no name for it)."""
    low = text.lower()

    def named(group, noun):
        if len(group) == 1:
            return next(iter(group))
        for k, v in group.items():
            if v["label"].lower() in low or re.search(rf"\b{re.escape(k)}\s+(medicine\s+|side\s+)?{noun}\b", low):
                return k
        return None

    mc = re.search(rf"\b({'|'.join(map(re.escape, F.SHELF_COLORS))})\s+(box|cube|block)\b", low) or \
        re.search(rf"\b({'|'.join(map(re.escape, F.SHELF_COLORS))})\b", low)
    shelf, colour = named(F.SHELVES, "shelf"), (mc.group(1) if mc else None)
    if colour is None and shelf is not None:  # no colour: the box said by its tray ("the one on the top shelf")
        by_h = sorted(F.shelf_boxes(shelf), key=lambda b: b[3])
        if len({b[3] for b in by_h}) > 1:
            for words, i in (("top|upper|highest|high", -1), ("bottom|lowest|lower|low", 0), ("middle", len(by_h) // 2)):
                if re.search(rf"\b({words})\b", low):
                    colour = by_h[i][0]
                    break
    return shelf, colour, named(F.TABLES, "table")


def _grown(m, rect=None):
    x0, x1, y0, y1 = rect if rect is not None else TABLE_RECT
    return x0 - m, x1 + m, y0 - m, y1 + m


def _inside(p, rect):
    return rect[0] <= p[0] <= rect[1] and rect[2] <= p[1] <= rect[3]


def _hits(a, b, rect):
    """Does the segment a->b cross the rectangle? (Liang-Barsky clipping)"""
    x0, x1, y0, y1 = rect
    dx, dy = b[0] - a[0], b[1] - a[1]
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, a[0] - x0), (dx, x1 - a[0]), (-dy, a[1] - y0), (dy, y1 - a[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return False
        else:
            t = q / p
            if p < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                return False
    return True


def route_around_table(start, goal):
    """Waypoints from start to goal (both xy) that keep the body clear of the tables: the shortest path
    over the corners of every table grown by ROUTE_CORNER (a visibility graph), no straight leg closer
    than ROUTE_BLOCK to any table. Only the scripted docking needs it. With three tables, routing round
    the target table alone drove into the one in between (2 of 12 episodes, FR thigh 52 N). A table
    whose grown block contains an endpoint is not checked against that leg (the leg cannot avoid it)."""
    nodes = [tuple(start)]
    for rect in TABLE_RECTS:
        x0, x1, y0, y1 = _grown(ROUTE_CORNER, rect)
        nodes += [(x0, y0), (x0, y1), (x1, y0), (x1, y1)]
    nodes.append(tuple(goal))
    blocks = [_grown(ROUTE_BLOCK, rect) for rect in TABLE_RECTS]

    def free(a, b):
        return not any(_hits(a, b, blk) for blk in blocks if not (_inside(a, blk) or _inside(b, blk)))

    dist, prev, todo = {0: 0.0}, {}, set(range(len(nodes)))
    while todo:
        i = min((n for n in todo if n in dist), key=lambda n: dist[n], default=None)
        if i is None:
            break
        todo.discard(i)
        for j in todo:
            if free(nodes[i], nodes[j]):
                d = dist[i] + math.dist(nodes[i], nodes[j])
                if d < dist.get(j, math.inf):
                    dist[j], prev[j] = d, i
    if len(nodes) - 1 not in dist:
        return [tuple(goal)]
    path, n = [], len(nodes) - 1
    while n != 0:
        path.append(nodes[n])
        n = prev[n]
    return path[::-1]


def vln_walk(obs, rec, ep, instruction=None):
    """Phase 1: DualVLN walks to the table on the navigation instruction, from the cameras alone."""
    PHASE["name"] = "walk"
    cur["obs"] = obs
    nav2_driver = nav.driver
    if nav.driver is not None:  # Nav2 drives the walk: active first, and the sim held to real time
        cur["obs"] = nav2_wait_ready(cur["obs"])
        if NAV2["ready"]:
            NAV2["link"].clear_costmaps()  # marks left from before the reset, and from the body settling at the start
            nav.driver.still, nav.driver.clears = None, 0
            NAV2["pace"], NAV2["wall0"], NAV2["sim0"] = True, time.time(), NAV2["t"]
        else:  # Nav2's controller not up (once its activation timed out after a restart: it rejected every path of
            nav.driver = None  # the walk): our own pure pursuit follows DualVLN's paths this time
        PHASE["name"] = "walk"
    out = os.path.join(args.vln_out, f"{time.strftime('%m%d-%H%M%S')}_seed{args.seed}_ep{ep}")
    WALK_AWAY["d0"] = None
    try:
        log = nav.run(instruction or args.vln, out, args.vln_max_time)
    finally:
        nav.driver = nav2_driver
    NAV2["pace"] = False
    log.pop("camvid")
    rec["stages"]["vln"] = {"reason": log["reason"], "end": log["end"], "table_edge_m": log["table_edge_m"],
                            "arrived": log["reason"] in ("STOP", "BLOCKED", "near") and log["table_edge_m"] <= vln_nav.AT_TABLE + 0.05,
                            "time_s": log["sim_time_s"], "path_m": log["path_len_m"], "requests": len(log["requests"]),
                            "follower": log.get("follower"), "stops": log.get("stops"), "mean_speed": log.get("mean_speed"),
                            "rms_acc": log.get("rms_acc"), "log": out,
                            "costmap_clears": getattr(nav.driver, "clears", None)}
    if (MULTI or SHELF or DOOR) and log.get("end") is not None and rec.get("target"):  # nearest the named one?
        rects = {k: door_rect(k) for k in F.DOORS} if DOOR else \
            {k: tuple(v["rect"]) for k, v in F.SHELVES.items()} if SHELF else {t_: F.table_rect3(t_) for t_ in F.COLORS3}
        near = min(rects, key=lambda k: vln_nav.table_edge_dist(log["end"][0], log["end"][1], rects[k]))
        rec["stages"]["vln"]["nearest"] = near
        rec["stages"]["vln"]["right_target"] = near == rec["target"]["door" if DOOR else "table"]
    say(f"ep {ep} vln {rec['stages']['vln']}")
    return cur["obs"]


def back_off(obs, rec):
    """Phase 2a: DualVLN stops close to the table at any angle. Move straight away from the
    nearest point of the table -- no turning: a turn there sweeps the body into it -- until
    BACKOFF_CLEAR from it, closed loop at TRIM_SPEED (timed reverse pulses lurched forward)."""
    PHASE["name"] = "backoff"
    p, _ = base_pose()
    rect = nearest_table_rect(float(p[0]), float(p[1]))  # DualVLN may have stopped at another table
    t, x0, x1, y0, y1 = 0.0, *rect
    d0 = vln_nav.table_edge_dist(float(p[0]), float(p[1]), rect)
    while t < 6.0:
        p, yaw = base_pose()
        x, y = float(p[0]), float(p[1])
        d = vln_nav.table_edge_dist(x, y, rect)
        if d >= BACKOFF_CLEAR or d < 1e-6:
            break
        ax, ay = (x - min(max(x, x0), x1)) / d, (y - min(max(y, y0), y1)) / d  # away from the nearest point
        ex, ey = math.cos(yaw) * ax + math.sin(yaw) * ay, -math.sin(yaw) * ax + math.cos(yaw) * ay
        vel.vel_command_b[:] = torch.tensor((TRIM_SPEED * ex, TRIM_SPEED * ey, 0.0), device=dev)
        obs = step(obs)
        t += dt
    vel.vel_command_b[:] = 0
    obs = hold(obs, 0.5)
    p, _ = base_pose()
    rec["stages"]["backoff"] = {"from_m": round(d0, 2), "to_m": round(vln_nav.table_edge_dist(float(p[0]), float(p[1]),
                                                                                              rect), 2),
                                "time_s": round(t, 1)}
    return obs


def near_table(within=0.06):
    """Robot bodies whose origin is within `within` m (xy) of the table and below its top."""
    out = {}
    for n, pw in zip(robot.body_names, robot.data.body_pos_w[0]):
        if float(pw[2]) < F.TABLE_HEIGHT + 0.02:
            d = vln_nav_edge(float(pw[0]), float(pw[1]))
            if d < within:
                out[n] = round(d * 100, 1)
    return out


def vln_nav_edge(x, y):
    x0, x1, y0, y1 = TABLE_RECT
    return math.hypot(max(x0 - x, 0.0, x - x1), max(y0 - y, 0.0, y - y1))


TURN_COST = 1.0  # m of driving that one radian of turning is worth (0.6 rad/s vs ~0.6 m/s)


def plan_dock_route(start, predock, psi, yaw=None):
    """Waypoints to the pre-dock point: round the table if it is in the way, and via an entry
    point ENTRY_BACK behind the pre-dock point unless the start is already behind it, so the
    robot arrives heading psi (see navigate). Returns (route, stops, length, cost) with
    cost = length + TURN_COST * the heading changes along the way (from `yaw`, if given, to
    psi at the end) -- what picks the side to park on."""
    sx_, sy_ = float(start[0]), float(start[1])
    along = (sx_ - float(predock[0])) * math.cos(psi) + (sy_ - float(predock[1])) * math.sin(psi)
    target = [float(predock[0]), float(predock[1])]
    stops = []
    if along > -ENTRY_BACK / 2:
        entry = [target[0] - ENTRY_BACK * math.cos(psi), target[1] - ENTRY_BACK * math.sin(psi)]
        stops = route_around_table([sx_, sy_], entry)  # ends at the entry point
    route = (stops or [[sx_, sy_]])[:-1] + ([stops[-1]] if stops else [])
    route = [list(w) for w in route] + [list(w) for w in route_around_table(stops[-1] if stops else [sx_, sy_], target)]
    pts = [[sx_, sy_]] + route
    length = sum(math.dist(a_, b_) for a_, b_ in zip(pts[:-1], pts[1:]))
    heads = [math.atan2(b_[1] - a_[1], b_[0] - a_[0]) for a_, b_ in zip(pts[:-1], pts[1:]) if math.dist(a_, b_) > 0.05]
    seq = ([yaw] if yaw is not None else []) + heads + [psi]
    turn = sum(abs(math.atan2(math.sin(b_ - a_), math.cos(b_ - a_))) for a_, b_ in zip(seq[:-1], seq[1:]))
    return route, stops, length, length + TURN_COST * turn


def entry_point(bp_w):
    """E for the current side: ENTRY_BACK behind the pre-dock point, heading psi (plan_dock_route)."""
    _, predock, psi = dock_pose(bp_w)
    return (float(predock[0]) - ENTRY_BACK * math.cos(psi), float(predock[1]) - ENTRY_BACK * math.sin(psi), psi)


def staging_point(bp_w):
    """S for the current side: the docking server's staging pose, STAGING_BACK behind the pick point D on
    its line (along the table), heading psi. 1.7 m or more from the table: room to turn there."""
    d_xy, _, psi = dock_pose(bp_w)
    return (float(d_xy[0]) - STAGING_BACK * math.cos(psi), float(d_xy[1]) - STAGING_BACK * math.sin(psi), psi)


def new_kick():
    return {"still": 0.0, "until": -1.0, "still_w": 0.0, "until_w": -1.0, "n": 0}


def shape_cmd(cmd, kick):
    """Nav2's command -> what the ETH controller will actually execute.
    Dead-zone inverse: the controller does nothing for sideways commands below ~0.12-0.15 m/s or turn
    commands below ~0.1 rad/s (measured: vy 0.07 -> 0, 0.25 -> 0.13; wz 0.05 -> 0, 0.2 -> 0.15), while
    forward follows even 0.06 m/s; MPPI assumed it did what it asked, crept, overshot the staging pose by
    0.9 m and shuffled back and forth -> add the dead zone to those two axes.
    Breakaway (last resort): commanded to move but following < 35 % of it for BREAK_S -> raise the
    command to the smallest speed it follows from rest, for BREAK_HOLD s."""
    if abs(cmd[1]) > 0.02:
        cmd[1] = math.copysign(abs(cmd[1]) + DZ_VY, cmd[1])
    if abs(cmd[2]) > 0.02:
        cmd[2] = math.copysign(abs(cmd[2]) + DZ_WZ, cmd[2])
    lin_n = math.hypot(cmd[0], cmd[1])
    meas = float(robot.data.root_lin_vel_b[0, :2].norm())
    meas_w = abs(float(robot.data.root_ang_vel_b[0, 2]))
    kick["still"] = kick["still"] + dt if (lin_n >= 0.03 and meas < 0.35 * lin_n) else 0.0
    kick["still_w"] = kick["still_w"] + dt if (abs(cmd[2]) >= 0.02 and meas_w < 0.35 * abs(cmd[2])) else 0.0
    if kick["still"] > BREAK_S:
        kick["until"], kick["still"], kick["n"] = NAV2["t"] + BREAK_HOLD, 0.0, kick["n"] + 1
    if kick["still_w"] > BREAK_S:
        kick["until_w"], kick["still_w"], kick["n"] = NAV2["t"] + BREAK_HOLD, 0.0, kick["n"] + 1
    if NAV2["t"] < kick["until"] and 1e-6 < lin_n < BREAK_V:
        cmd[0], cmd[1] = cmd[0] * BREAK_V / lin_n, cmd[1] * BREAK_V / lin_n
    if NAV2["t"] < kick["until_w"] and 1e-6 < abs(cmd[2]) < BREAK_W:
        cmd[2] = math.copysign(BREAK_W, cmd[2])
    return cmd


def nav2_wait_ready(obs):
    """Wait (stepping, so the simulator keeps publishing) until Nav2 is active; once per run."""
    if NAV2["ready"]:
        return obs
    holder = {"obs": obs}

    def idle():
        vel.vel_command_b[:] = 0
        holder["obs"] = step(holder["obs"])

    NAV2["ready"] = NAV2["link"].wait_ready(120.0, tick=idle)
    for _ in range(int(2.0 / dt)):  # costmaps fill from the first scans
        idle()
    say(f"nav2 {'active' if NAV2['ready'] else 'NOT ACTIVE after 120 s'}")
    return holder["obs"]


def nav2_relay(holder, link, timeout, ntrace, kick, t0):
    """Relay Nav2's /cmd_vel to the base controller until the running action ends (or timeout s of sim
    time), shaped by shape_cmd. Returns (status or None on timeout, driven m)."""
    status, path, last = None, 0.0, robot.data.root_pos_w[0, :2].clone()
    t_start = NAV2["t"]
    while status is None and NAV2["t"] - t_start < timeout:
        raw = list(link.command())
        cmd = shape_cmd(list(raw), kick)
        vel.vel_command_b[:] = torch.tensor(cmd, device=dev)
        holder["obs"] = step(holder["obs"])
        pq, yq = base_pose()
        lv = robot.data.root_lin_vel_b[0]
        ntrace.append([NAV2["t"] - t0, float(pq[0]), float(pq[1]), yq, *raw, *cmd, float(lv[0]), float(lv[1]),
                       float(robot.data.root_ang_vel_b[0, 2])])
        xy = robot.data.root_pos_w[0, :2]
        if float((xy - last).norm()) > 0.02:
            path, last = path + float((xy - last).norm()), xy.clone()
        status = link.done()
    if status is None:
        link.cancel()
    vel.vel_command_b[:] = 0
    return status, path


def nav_pose_err(goal):
    p, yaw = base_pose()
    return (round(math.hypot(float(p[0]) - goal[0], float(p[1]) - goal[1]) * 100, 1),
            round(math.degrees(math.atan2(math.sin(yaw - goal[2]), math.cos(yaw - goal[2]))), 1))


STATUS = {4: "succeeded", 5: "canceled", 6: "aborted", None: "timeout"}


def nav2_drive(obs, rec, ep, bp_w, stages=None):
    """--nav2: plan on both sides with Nav2's planner (ComputePathToPose), take the shorter path and let
    Nav2 drive there (NavigateToPose -> /cmd_vel -> the base controller). --nav2_goal dock: to the
    staging pose S, then Nav2's docking server drives S -> D, the pick point, in its heading
    (DockRobot). --nav2_goal entry: only to E (the script then drives E -> P -> D). Returns (obs, ok);
    on failure the scripted side choice and route take over from wherever the robot is. Logged in `stages`
    (default rec["stages"]; the drive to the table the box goes on logs under rec["stages"]["place_drive"])."""
    stages = rec["stages"] if stages is None else stages
    link = NAV2["link"]
    PHASE["name"] = "nav2"
    NAV2["pace"], NAV2["wall0"], NAV2["sim0"] = True, time.time(), NAV2["t"]
    holder = {"obs": obs}

    def idle():
        vel.vel_command_b[:] = 0
        holder["obs"] = step(holder["obs"])

    holder["obs"] = nav2_wait_ready(holder["obs"])
    if not NAV2["ready"]:
        NAV2["pace"] = False
        stages["nav2"] = {"status": "not active"}
        return holder["obs"], False
    p0, yaw0 = base_pose()
    first = staging_point if args.nav2_goal == "dock" else entry_point
    plans = {}
    for name in sides_allowed():
        set_side(name)
        plans[name] = link.plan_length((float(p0[0]), float(p0[1]), yaw0), first(bp_w), tick=idle)[0]
    st = {"plan_m": {k: (round(v, 2) if math.isfinite(v) else None) for k, v in plans.items()}}
    stages["nav2"] = st
    if not any(math.isfinite(v) for v in plans.values()):
        st["status"] = "no plan"
        PROG.event("nav2_done", **st)
        NAV2["pace"] = False
        say(f"ep {ep} nav2 {st}: scripted route instead")
        return holder["obs"], False
    side = min(plans, key=plans.get)
    set_side(side)
    goal = first(bp_w)
    d_xy, _, psi_d = dock_pose(bp_w)
    dock = (float(d_xy[0]), float(d_xy[1]), psi_d)
    stages["side"] = {"chosen": side, "cost_m": st["plan_m"], "by": "nav2"}
    say(f"ep {ep} nav2: table on the {side} (paths {st['plan_m']}), driving to "
        f"{'the staging pose S' if args.nav2_goal == 'dock' else 'E'} ({goal[0]:.2f},{goal[1]:.2f})")
    PROG.event("nav2_plan", paths=st["plan_m"], side=side, goal=[round(goal[0], 2), round(goal[1], 2), round(math.degrees(goal[2]), 1)],
               stage="to staging pose" if args.nav2_goal == "dock" else "to entry point")
    NAV2["goal"] = goal
    link.clear_plan()
    t0, ntrace = NAV2["t"], []
    kick = new_kick()
    status, path = (nav2_relay(holder, link, args.nav2_timeout, ntrace, kick, t0) if link.navigate(goal, tick=idle) else (6, 0.0))
    e_cm, y_deg = nav_pose_err(goal)
    st.update(status=STATUS.get(status, str(status)), goal="staging" if args.nav2_goal == "dock" else "entry",
              time_s=round(NAV2["t"] - t0, 1), driven_m=round(path, 2), err_cm=e_cm, yaw_err_deg=y_deg)
    say(f"ep {ep} nav2 {st}")
    PROG.event("nav2_done", **st)
    ok = status == 4
    if ok and args.nav2_goal == "dock":  # Nav2's docking server: S -> the pick point D
        PHASE["name"] = "nav2_dock"
        NAV2["goal"] = dock
        PROG.event("nav2_plan", paths=st["plan_m"], side=side, goal=[round(dock[0], 2), round(dock[1], 2), round(math.degrees(dock[2]), 1)],
                   stage="docking at the pick point")
        t1 = NAV2["t"]
        sd, pd = (nav2_relay(holder, link, args.nav2_timeout, ntrace, kick, t0) if link.dock(dock, tick=idle) else (6, 0.0))
        res = link.last_result()
        e_cm, y_deg = nav_pose_err(dock)
        sdock = {"status": STATUS.get(sd, str(sd)), "success": bool(res.success) if res is not None else False,
                 "error_code": int(res.error_code) if res is not None else None, "time_s": round(NAV2["t"] - t1, 1),
                 "driven_m": round(pd, 2), "err_cm": e_cm, "yaw_err_deg": y_deg}
        stages["nav2_dock"] = sdock
        say(f"ep {ep} nav2 dock {sdock}")
        PROG.event("nav2_done", stage="docking", **sdock)
        ok = sd == 4 and sdock["success"]
        path += pd
    st["stall_kicks"] = kick["n"]
    NAV2["goal"] = None
    NAV2["pace"] = False
    if ntrace:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        np.savez(os.path.join(os.path.dirname(args.out), f"nav2_trace_ep{ep}{'' if stages is rec['stages'] else '_place'}.npz"), trace=np.array(ntrace, dtype=np.float32),
                 goal=np.array(dock), entry=np.array(goal))
    return holder["obs"], ok


def navigate(obs, rec, ep, bp_w=None, stages=None):
    """Three stages. Arriving diagonally and turning in place at the table swung the
    robot's front-right hip into the table (30-100 N; the arm then missed), so:
      A  drive to a pre-dock point PREDOCK m in front of the standoff point
      B  turn in place there to face the table (+x)
      C  approach straight, correcting sideways and heading on the way.
    model_2600 ignores small commands (dead zone), so corrections are sent at least
    at MIN_SPEED / MIN_TURN. bp_w: where the box is (default) or goes (the place); logged in `stages`."""
    PHASE["name"] = "dock"
    stages = rec["stages"] if stages is None else stages
    bp_w = box.data.root_pos_w[0] if bp_w is None else bp_w
    p0, yaw0 = base_pose()
    nav2_ok = False
    if args.nav2 and args.approach == "side":
        t_nav = NAV2["t"]
        obs, nav2_ok = nav2_drive(obs, rec, ep, bp_w, stages)
        if nav2_ok and args.nav2_goal == "dock":  # Nav2 parked at the pick point: no scripted approach
            goal, _, psi = dock_pose(bp_w)
            vel.vel_command_b[:] = 0
            obs = hold(obs, 0.5)
            p, yaw = base_pose()
            e = goal - p[:2]
            stages["dock_err_base_cm"] = [round(100 * (math.cos(yaw) * float(e[0]) + math.sin(yaw) * float(e[1])), 1),
                                                 round(100 * (-math.sin(yaw) * float(e[0]) + math.cos(yaw) * float(e[1])), 1)]
            yerr = math.degrees(math.atan2(math.sin(psi - yaw), math.cos(psi - yaw)))
            n2 = stages["nav2"]
            stages["locomotion"] = {"time_s": round(NAV2["t"] - t_nav, 2), "arrived": True, "by": "nav2",
                                           "pos_err_m": round(float(e.norm()), 3), "yaw_err_deg": round(yerr, 1),
                                           "path_m": n2.get("driven_m"), "touching_after": touching(), "stall_kicks": n2.get("stall_kicks")}
            say(f"ep {ep} locomotion {stages['locomotion']}")
            return obs
    if args.approach == "side" and not nav2_ok:  # the cheaper side to park on, from where the robot is now
        costs = {}
        for name in sides_allowed():
            set_side(name)
            _, pd, ps = dock_pose(bp_w)
            costs[name] = plan_dock_route([float(p0[0]), float(p0[1])], pd, ps, yaw0)[3]
        set_side(min(costs, key=costs.get))
        stages["side"] = {"chosen": SIDE["name"], "cost_m": {k: round(v, 2) for k, v in costs.items()}}
        say(f"ep {ep} side pick: table on the {SIDE['name']} (cost {stages['side']['cost_m']})")
    goal, predock, psi = dock_pose(bp_w)
    if stages is rec["stages"]:
        rec["box_start_w"] = [round(float(v), 3) for v in bp_w]
    start_xy = robot.data.root_pos_w[0, :2].clone()
    trace = {"path": 0.0, "last": start_xy.clone(), "tilt": 0.0, "t": 0.0, "still": 0.0, "kick_until": -1.0, "kicks": 0}
    limit = NAV_TIMEOUT  # extended below by the route length: DualVLN may stop far off (10 m once) or beyond the table

    def push(v, lo, hi, need):
        v = max(-hi, min(hi, v))
        return math.copysign(max(abs(v), lo), v) if need else 0.0

    def lin(vx, vy, vmax):
        """Linear command at least MIN_SPEED in norm (the policy's dead zone), at most vmax."""
        n = math.hypot(vx, vy)
        if n < 1e-6:
            return 0.0, 0.0
        k = min(max(n, MIN_SPEED), vmax) / n
        return vx * k, vy * k

    def go(obs, cmd, stage, dist):
        # Stall kick: model_2600, once slowed down, can ignore a small sideways or diagonal command
        # -- (0.15, -0.22) m/s held it still for 22 s with nothing touching it, a (0, -0.3) trim
        # crept 2 cm in 4 s (probe_cmd_response.py: decelerating from (0.3, -0.05) into
        # (0.15, -0.22) in the open, the sideways part was dropped). Commanded to move but
        # following < 35 % of it for STALL_S: send KICK x the command for KICK_S.
        lin_n = math.hypot(cmd[0], cmd[1])
        if trace["t"] < trace["kick_until"] and lin_n > 1e-6:
            k = min(KICK, KICK_MAX / lin_n)
            cmd = (cmd[0] * k, cmd[1] * k, cmd[2])
        vel.vel_command_b[:] = torch.tensor(cmd, device=dev)
        if int(round(trace["t"] / dt)) % int(round(2.0 / dt)) == 0:
            p, yaw = base_pose()
            say(f"  nav {stage} t={trace['t']:5.1f}s pos ({float(p[0]):+.2f},{float(p[1]):+.2f}) yaw {math.degrees(yaw):+6.1f} "
                f"dist {dist:.2f} cmd ({cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f}) touch {touching()} near {near_table()}")
        obs = step(obs)
        xy = robot.data.root_pos_w[0, :2].clone()
        if float((xy - trace["last"]).norm()) > 0.02:  # count real travel, not standing jitter
            trace["path"] += float((xy - trace["last"]).norm())
            trace["last"] = xy
        trace["tilt"] = max(trace["tilt"], tilt_deg())
        # "still" = following less than a third of the commanded speed: it also creeps (1 cm/s for 4 s)
        moving = float(robot.data.root_lin_vel_b[0, :2].norm()) > max(0.03, 0.35 * lin_n)
        trace["still"] = 0.0 if (moving or lin_n < 0.2) else trace["still"] + dt
        if trace["still"] > STALL_S and touching():  # held by a real contact: pushing harder could knock the table
            trace["still"] = 0.0
        if trace["still"] > STALL_S:
            trace["kick_until"], trace["still"] = trace["t"] + KICK_S, 0.0
            trace["kicks"] += 1
            say(f"  nav {stage} stalled under ({cmd[0]:+.2f},{cmd[1]:+.2f}): kick")
        trace["t"] += dt
        return obs

    def herr_of(yaw):
        return math.atan2(math.sin(psi - yaw), math.cos(psi - yaw))

    def face_table(obs, stage):
        ok, t_d = 0, trace["t"]
        while trace["t"] - t_d < 6.0 and ok < int(0.3 / dt):
            _, yaw = base_pose()
            herr = herr_of(yaw)
            ok = ok + 1 if abs(herr) < math.radians(3.0) else 0
            obs = go(obs, (0.0, 0.0, push(1.5 * herr, 0.55, 0.7, abs(herr) > math.radians(2.0))), stage, abs(herr))
        vel.vel_command_b[:] = 0
        return hold(obs, 1.0)

    # A0: round the table first if it is in the way (DualVLN may have stopped beyond it), and
    # arrive at the pre-dock point already heading psi: from the wrong side (DualVLN stopping
    # north of the table) it got there facing -73 deg, the 160 deg turn in place crept it 25 cm
    # toward the table and stage C drove the front hip into the table's corner (200 N). So
    # unless the start is already behind the pre-dock point, go via ENTRY_BACK m before it.
    sx_, sy_ = float(start_xy[0]), float(start_xy[1])
    route, stops, route_len, _ = plan_dock_route([sx_, sy_], predock, psi)
    if nav2_ok:  # Nav2 brought it to E heading psi: only the straight E -> P (stage A) is left
        route, stops, route_len = [list(map(float, predock))], [], float((predock - start_xy).norm())
    stages["route"] = [[round(c, 2) for c in wp] for wp in route[:-1]]
    limit = NAV_TIMEOUT + 6.0 * max(0.0, route_len - 2.0)
    for i_wp, wp in enumerate(route[:-1]):
        is_entry = bool(stops) and i_wp == len(route) - 2
        wp = torch.tensor(wp, device=dev)
        while trace["t"] < limit:
            p, yaw = base_pose()
            e = wp - p[:2]
            dist = float(e.norm())
            if dist < (0.2 if is_entry else 0.3):  # pass a corner, do not stop on it
                break
            herr = math.atan2(float(e[1]), float(e[0])) - yaw
            herr = math.atan2(math.sin(herr), math.cos(herr))
            obs = go(obs, (min(0.8, 0.9 * dist) * max(0.0, math.cos(herr)) ** 2, 0.0, max(-1.0, min(1.0, 1.8 * herr))),
                     "A0", dist)
        if is_entry:  # turn here, away from the table, to face the pre-dock point: the turn's creep is harmless
            ok = 0
            while trace["t"] < limit and ok < int(0.3 / dt):
                p, yaw = base_pose()
                e = predock - p[:2]
                herr = math.atan2(float(e[1]), float(e[0])) - yaw
                herr = math.atan2(math.sin(herr), math.cos(herr))
                ok = ok + 1 if abs(herr) < math.radians(10.0) else 0
                obs = go(obs, (0.0, 0.0, push(1.5 * herr, MIN_TURN, 0.8, abs(herr) > math.radians(8.0))), "A0-turn", abs(herr))
    # A: drive to the pre-dock point. Far: face it and drive. Near: move holonomically
    # (forward + sideways, no turning) -- steering toward a nearby point orbited it.
    while trace["t"] < limit:
        p, yaw = base_pose()
        e = predock - p[:2]
        dist = float(e.norm())
        if dist < 0.08 + STOP_LEAD:
            break
        if dist > 0.5:
            herr = math.atan2(float(e[1]), float(e[0])) - yaw
            herr = math.atan2(math.sin(herr), math.cos(herr))
            vx = min(1.0, 0.9 * dist) * max(0.0, math.cos(herr)) ** 2
            cmd = (vx, 0.0, max(-1.0, min(1.0, 1.8 * herr)))
        else:
            ex = math.cos(yaw) * float(e[0]) + math.sin(yaw) * float(e[1])
            ey = -math.sin(yaw) * float(e[0]) + math.cos(yaw) * float(e[1])
            cmd = lin(1.2 * ex, 1.2 * ey, 0.5) + (0.0,)
        obs = go(obs, cmd, "A", dist)
    # B: turn in place to the docking heading psi
    ok = 0
    while trace["t"] < limit and ok < int(0.3 / dt):
        _, yaw = base_pose()
        herr = herr_of(yaw)
        ok = ok + 1 if abs(herr) < math.radians(3.0) else 0
        obs = go(obs, (0.0, 0.0, push(1.5 * herr, MIN_TURN, 0.8, abs(herr) > math.radians(2.0))), "B", abs(herr))
    # C: straight approach
    while trace["t"] < limit:
        p, yaw = base_pose()
        e = goal - p[:2]
        ex = math.cos(yaw) * float(e[0]) + math.sin(yaw) * float(e[1])
        ey = -math.sin(yaw) * float(e[0]) + math.cos(yaw) * float(e[1])
        herr = herr_of(yaw)
        if ex < STOP_LEAD:
            break
        obs = go(obs, lin(1.0 * ex, 1.5 * ey, 0.3) + (push(1.5 * herr, MIN_TURN, 0.5, abs(herr) > math.radians(3.0)),),
                 "C", float(e.norm()))
    vel.vel_command_b[:] = 0
    obs = hold(obs, 1.0)
    # D: the stop turns the robot 8-28 deg (always counter-clockwise); face the table once.
    obs = face_table(obs, "D")
    # E: then trim the distance LAST, with short timed pulses at the smallest speed the policy
    # follows, holding the heading while moving. Turning in place creeps the robot forward
    # (22 cm in one episode), so no heading correction may come after this: a box too close
    # puts the arm's path through the robot's own head and lidar mast. The arm absorbs the
    # remaining yaw (the grasp is computed in the base frame).
    # CLOSED-LOOP, straight (no turning): drive at TRIM_SPEED until the target distance is
    # crossed. Timed pulses did not work: under 0.1-0.35 s reverse commands the policy
    # lurched FORWARD (-4 -> -15 cm over four "backward" pulses), and a turn mixed in made
    # it creep forward too. It coasts on in the direction of travel, so going forward it
    # stops STOP_LEAD early; going back it ends slightly behind the goal (the safe side).
    trims = []
    p, yaw = base_pose()
    e = goal - p[:2]
    ex = math.cos(yaw) * float(e[0]) + math.sin(yaw) * float(e[1])
    if abs(ex) >= 0.04:
        direction, t_e = math.copysign(1.0, ex), trace["t"]
        while trace["t"] - t_e < 4.0:
            p, yaw = base_pose()
            e = goal - p[:2]
            ex_now = math.cos(yaw) * float(e[0]) + math.sin(yaw) * float(e[1])
            if (direction > 0 and ex_now < STOP_LEAD / 2) or (direction < 0 and ex_now > 0.0):
                break
            obs = go(obs, (direction * TRIM_SPEED, 0.0, 0.0), "E", abs(ex_now))
        vel.vel_command_b[:] = 0
        obs = hold(obs, 1.0)
        p2, yaw2 = base_pose()
        e2 = goal - p2[:2]
        trims.append((round(ex * 100, 1), round(trace["t"] - t_e, 2),
                      round(100 * (math.cos(yaw2) * float(e2[0]) + math.sin(yaw2) * float(e2[1])), 1), round(math.degrees(yaw2), 1)))
    # sideways too (side docking: this is the gap to the table), same closed-loop method
    p, yaw = base_pose()
    e = goal - p[:2]
    ey = -math.sin(yaw) * float(e[0]) + math.cos(yaw) * float(e[1])
    if abs(ey) >= 0.03:
        direction, t_e = math.copysign(1.0, ey), trace["t"]
        while trace["t"] - t_e < 4.0:
            p, yaw = base_pose()
            e = goal - p[:2]
            ey_now = -math.sin(yaw) * float(e[0]) + math.cos(yaw) * float(e[1])
            if direction * ey_now < STOP_LEAD / 4:
                break
            obs = go(obs, (0.0, direction * TRIM_SPEED, 0.0), "E-y", abs(ey_now))
        vel.vel_command_b[:] = 0
        obs = hold(obs, 1.0)
        p2, yaw2 = base_pose()
        e2 = goal - p2[:2]
        trims.append(("y", round(ey * 100, 1), round(trace["t"] - t_e, 2),
                      round(100 * (-math.sin(yaw2) * float(e2[0]) + math.cos(yaw2) * float(e2[1])), 1)))
    stages["trims_cm_s_cm_deg"] = trims
    p, yaw = base_pose()
    e = goal - p[:2]
    stages["dock_err_base_cm"] = [round(100 * (math.cos(yaw) * float(e[0]) + math.sin(yaw) * float(e[1])), 1),
                                         round(100 * (-math.sin(yaw) * float(e[0]) + math.cos(yaw) * float(e[1])), 1)]
    t = trace["t"]
    straight = float((goal - start_xy).norm())
    stages["locomotion"] = {
        "time_s": round(t, 2), "arrived": t < limit, "pos_err_m": round(float((goal - p[:2]).norm()), 3),
        "yaw_err_deg": round(math.degrees(herr_of(yaw)), 1), "path_m": round(trace["path"], 2), "straight_m": round(straight, 2),
        "path_ratio": round(trace["path"] / max(straight, 1e-6), 2), "max_tilt_deg": round(trace["tilt"], 1),
        "touching_after": touching(), "stall_kicks": trace["kicks"],
    }
    say(f"ep {ep} locomotion {stages['locomotion']}")
    return obs


def deliver(obs, rec, ep):
    """Back away from the table, then drive to DROP_ZONE holding the box in the carry pose."""
    PHASE["name"] = "deliver"
    t, path, last = 0.0, 0.0, robot.data.root_pos_w[0, :2].clone()
    goal = torch.tensor(DROP_ZONE, device=dev)
    max_tilt = 0.0
    # move 0.5 m away from the table first: turning at it would swing the body into it
    # (front: reverse; side: strafe left, the table being on the right)
    away = (-0.4, 0.0, 0.0) if args.approach == "front" else (0.0, 0.4 if SIDE["name"] == "right" else -0.4, 0.0)
    p0 = robot.data.root_pos_w[0, :2].clone()
    while t < 6.0 and float((robot.data.root_pos_w[0, :2] - p0).norm()) < 0.5:
        vel.vel_command_b[:] = torch.tensor(away, device=dev)
        obs = step(obs)
        t += dt
    nav2_used = False
    if args.nav2 and NAV2["ready"]:
        # Nav2 to the drop zone, round the other tables (the scripted drive below goes straight there)
        PHASE["name"] = "deliver"
        NAV2["pace"], NAV2["wall0"], NAV2["sim0"] = True, time.time(), NAV2["t"]
        p, yaw = base_pose()
        gpose = (float(goal[0]), float(goal[1]), math.atan2(float(goal[1] - p[1]), float(goal[0] - p[0])))
        holder = {"obs": obs}
        NAV2["link"].clear_plan()
        t_nav = NAV2["t"]
        sd, pd = (nav2_relay(holder, NAV2["link"], NAV_TIMEOUT, [], new_kick(), t_nav)
                  if NAV2["link"].navigate(gpose, tick=lambda: holder.__setitem__("obs", step(holder["obs"]))) else (6, 0.0))
        NAV2["pace"] = False
        obs = holder["obs"]
        t += NAV2["t"] - t_nav
        path += pd
        nav2_used = sd == 4
        rec["stages"]["deliver_nav2"] = {"status": STATUS.get(sd, str(sd)), "time_s": round(NAV2["t"] - t_nav, 1)}
    while t < NAV_TIMEOUT and not nav2_used:
        p, yaw = base_pose()
        e = goal - p[:2]
        dist = float(e.norm())
        if dist < 0.15:
            break
        herr = math.atan2(float(e[1]), float(e[0])) - yaw
        herr = math.atan2(math.sin(herr), math.cos(herr))
        if dist > 0.5:
            cmd = (min(0.8, 0.8 * dist) * max(0.0, math.cos(herr)) ** 2, 0.0, max(-1.0, min(1.0, 1.8 * herr)))
        else:  # holonomic near the goal (steering toward a close point orbits it)
            ex = math.cos(yaw) * float(e[0]) + math.sin(yaw) * float(e[1])
            ey = -math.sin(yaw) * float(e[0]) + math.cos(yaw) * float(e[1])
            k = max(MIN_SPEED, min(0.4, 1.2 * dist)) / max(dist, 1e-6)
            cmd = (k * ex, k * ey, 0.0)
        vel.vel_command_b[:] = torch.tensor(cmd, device=dev)
        obs = step(obs)
        xy = robot.data.root_pos_w[0, :2].clone()
        if float((xy - last).norm()) > 0.02:
            path += float((xy - last).norm())
            last = xy
        max_tilt = max(max_tilt, tilt_deg())
        t += dt
    vel.vel_command_b[:] = 0
    obs = hold(obs, 1.5)
    held = float((box_in_base()[0] - ee_pose()[0]).norm())
    rec["stages"]["deliver"] = {"time_s": round(t, 1), "arrived": t < NAV_TIMEOUT or nav2_used, "path_m": round(path, 2),
                                "by": "nav2" if nav2_used else "script",
                                "max_tilt_deg": round(max_tilt, 1), "box_to_grasp_cm": round(held * 100, 1),
                                "box_height_m": round(float(box.data.root_pos_w[0, 2]), 3)}
    rec["delivered"] = bool((t < NAV_TIMEOUT or nav2_used) and held <= 0.06 and float(box.data.root_pos_w[0, 2]) > 0.3)  # in hand, off the floor
    say(f"ep {ep} deliver {rec['stages']['deliver']} -> delivered {rec['delivered']}")
    return obs


# ------------------------------------------------------------------ putting the box down (shelf scene)
PLACE_DROP = 0.008   # m: let go of the box this far above the table (it settles)
SHELF_LIP = 0.03     # m: the shelf trays' front lip (hospital MedShelf: 2.9 cm)
PLACE_ABOVE = 0.08   # m: MoveIt brings the box this far above its spot, then straight down
PLACE_BACK = 0.12    # m: after letting go, straight back out along the approach (the grasp's pitch)...
PLACE_UP = 0.04      # ...and up: level, the wrist RealSense (rolled under) ended over the robot's back
PLACE_TRIES = [(0, 0.0), (15, 0.0), (30, 0.0), (0, 0.04), (15, 0.04), (30, 0.04), (-45, 0.0), (45, 0.0)]
# (hand turned about the vertical, deg -- towards the robot's back --, m deeper onto the table): the place poses
# tried in turn. The box stays upright, only its heading changes; held nearly level (a high tray), the straight-in
# pose at its depth is on the edge of the arm's reach and turned 15-30 deg it is well inside (moveit/README.md)
GRASPED = {"pitch": None, "flip": False, "carry": None, "place_depth": None}  # the grasp's pitch and roll: the
# place keeps them (the box arrives upright); the carry pose it went into; how far onto the table it can go down


def place_spot(P):
    """Where the box goes on table P: its spot, or -- a box grasped nearly level (a high tray) -- deeper onto the table,
    where the arm can still hold the hand at that pitch (moveit/README.md)."""
    sx, sy = P["spot"]
    if GRASPED["place_depth"]:
        sx = P["near_x"] - P["face"] * GRASPED["place_depth"]
    return sx, sy


def box_pose_b():
    """Box centre and orientation in the robot's base frame."""
    p, q = mu.subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, box.data.root_pos_w,
                                        box.data.root_quat_w)
    return p[0], q[0]


def box_tilt_deg():
    up = mu.quat_apply(box.data.root_quat_w, torch.tensor([[0.0, 0.0, 1.0]], device=dev))[0]
    return math.degrees(math.acos(max(-1.0, min(1.0, float(up[2])))))


def place_moveit(obs, rec):
    """Put the held box on the table at the scene file's spot. MoveIt plans, the box attached to the gripper, to
    PLACE_ABOVE over the spot (around the things on the table and the robot's body), then straight down; open;
    straight back out; the arm folds to its stow pose. The box's pose in the jaws is measured first (it may have
    slipped on the way), and the gripper is turned so the BOX lands upright on the spot."""
    PHASE["name"] = "place"
    P = F.TABLES[PLACE_T["name"]]
    st = rec["stages"].setdefault("place", {})
    st["moveit"] = {}
    speed = args.moveit_speed
    BASE_HOLD["xy"], BASE_HOLD["active"], BASE_HOLD["engaged_steps"] = robot.data.root_pos_w[0, :2].clone(), False, 0
    BASE_HOLD["yaw"] = base_pose()[1]

    def done(ok, stage=None):
        if stage:
            st["moveit"]["failed_at"] = stage
        BASE_HOLD["xy"] = None
        vel.vel_command_b[:] = 0
        bw = box.data.root_pos_w[0]
        sx, sy = spot["xy"]
        x0, x1, y0, y1 = P["rect"]
        st["box_w"] = [round(float(v), 3) for v in bw]
        st["off_spot_cm"] = round(100 * math.hypot(float(bw[0]) - sx, float(bw[1]) - sy), 1)
        st["height_err_cm"] = round(100 * (float(bw[2]) - (P["top"] + F.BOX_SIZE[2] / 2)), 1)
        st["tilt_deg"] = round(box_tilt_deg(), 1)
        st["released_cm"] = round(float((box_in_base()[0] - ee_pose()[0]).norm()) * 100, 1)
        on_table = (x0 + 0.01 <= float(bw[0]) <= x1 - 0.01 and y0 + 0.01 <= float(bw[1]) <= y1 - 0.01
                    and abs(st["height_err_cm"]) < 2.0 and st["tilt_deg"] < 15.0)
        rec["placed"] = bool(ok and on_table and st["released_cm"] > 5.0)
        return obs

    spot = {"xy": place_spot(P)}
    # the box in the jaws (grasp-point frame), and the gripper pose that puts it upright on the spot
    gp_b, gq_b = ee_pose()
    bp_b, bq_b = box_pose_b()
    p_gb, q_gb = mu.subtract_frame_transforms(gp_b[None], gq_b[None], bp_b[None], bq_b[None])
    pitch = GRASPED["pitch"] if GRASPED["pitch"] is not None else args.pitch
    q_g = grasp_quat(ARM_YAW, pitch, GRASPED["flip"])  # straight into the table, the grasp's pitch and roll (against
    # the body: levelled against the world, a lean away from the table put the place out of reach; the box settles)
    q_box = mu.quat_mul(q_g[None], q_gb)
    tilt0 = math.degrees(math.acos(max(-1.0, min(1.0, float(mu.quat_apply(q_box, torch.tensor([[0.0, 0.0, 1.0]], device=dev))[0, 2])))))
    st["box_in_jaws"] = {"pos_cm": [round(100 * float(v), 1) for v in p_gb[0]], "tilt_deg": round(tilt0, 1)}
    if tilt0 < 30.0:  # it slipped a little: turn the gripper so the box is upright (keep its heading)
        yaw_b = float(mu.euler_xyz_from_quat(q_box)[2][0])
        q_up = torch.tensor([[math.cos(yaw_b / 2), 0.0, 0.0, math.sin(yaw_b / 2)]], device=dev)
        q_g = mu.quat_mul(q_up, mu.quat_inv(q_gb))[0]
    up = torch.tensor([0.0, 0.0, PLACE_ABOVE], device=dev)

    def place_pose(dyaw, deeper):  # gripper pose (base frame) that puts the box there, turned dyaw about the vertical
        sx, sy = spot["xy"][0] - P["face"] * deeper, spot["xy"][1]
        h = math.radians(dyaw) * (1.0 if ARM_YAW >= 0 else -1.0) / 2  # towards the back on either side
        q = mu.quat_mul(torch.tensor([[math.cos(h), 0.0, 0.0, math.sin(h)]], device=dev), q_g[None])[0]
        spot_w = torch.tensor([[sx, sy, P["top"] + F.BOX_SIZE[2] / 2 + PLACE_DROP]], device=dev)
        spot_b, _ = mu.subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, spot_w)
        return spot_b[0] - mu.quat_apply(q[None], p_gb)[0], q, (sx, sy)

    moveit_scene(with_target=False, attach_target=True)
    cq = GRASPED["carry"] or CARRY_Q
    if abs(float(q_arm()[0]) - (cq[0] + ARM_YAW)) > 0.5:  # the table on the other side: first swing the carry pose
        TRACE["sub"] = "swing"  # round on joint1 alone, the box upright (a free plan straight to the table had the
        swing = [cq[0] + ARM_YAW] + list(q_arm()[1:].tolist())  # jaws turn over the front: the box slipped out once)
        r = moveit_call(st, "swing", MOVEIT.plan_joints, q_arm().tolist(), swing, gripper=float(state["grip"]),
                        time=5.0, attempts=5, vel=speed * 0.7, acc=speed * 0.5)
        if r is not None:
            obs, _ = run_traj(obs, r, hold_s=0.3)
    TRACE["sub"] = "pre_place"  # free motion with the box attached, to a posture from which the straight lowering
    ap = None                   # works (a pose goal alone may dead-end), for the first place pose that has one
    for k, (dyaw, deeper) in enumerate(PLACE_TRIES):
        p_t, q_t, xy = place_pose(dyaw, deeper)
        r = MOVEIT.approach(q_arm().tolist(), base_to_l00(p_t + up, q_t), base_to_l00(p_t, q_t),
                            gripper=float(state["grip"]), tries=8 if k else 16, enough=1)
        if r.get("ok"):
            ap, (p_g, q_g), spot["xy"] = r, (p_t, q_t), xy
            st["moveit"]["place_ik"] = {**{n: r.get(n) for n in ("ok", "tried", "server_s")}, "turned_deg": dyaw,
                                        "deeper_m": deeper, "poses_tried": k + 1}
            break
    else:
        st["moveit"]["place_ik"] = {"ok": False, "poses_tried": len(PLACE_TRIES), "error": r.get("error")}
        print(f"FETCH| MoveIt place_ik failed for all {len(PLACE_TRIES)} place poses", flush=True)
        p_g, q_g, _ = place_pose(0, 0.0)
    if ap is not None:
        r = moveit_call(st, "pre_place", MOVEIT.plan_joints, q_arm().tolist(), ap["joints"], gripper=float(state["grip"]),
                        time=5.0, attempts=5, vel=speed * 0.7, acc=speed * 0.5)
    else:
        r = moveit_call(st, "pre_place", MOVEIT.plan_pose, q_arm().tolist(), base_to_l00(p_g + up, q_g),
                        gripper=float(state["grip"]), pos_tol=0.004, rot_tol=0.04, time=5.0, attempts=5, vel=speed * 0.7,
                        acc=speed * 0.5)
    if r is None:
        return done(False, "pre_place")
    obs, _ = run_traj(obs, r, hold_s=0.4)
    TRACE["sub"] = "lower"  # straight down onto the table
    st["request_l00"] = {"final": [round(v, 4) for v in base_to_l00(p_g, q_g)], "q_start": [round(float(v), 3) for v in q_arm()],
                         "base_z": round(float(robot.data.root_pos_w[0, 2]), 3),
                         "stance": round(eth_driver.crouch, 2) if args.loco == "eth" else None}
    r = moveit_call(st, "lower", MOVEIT.cartesian, q_arm().tolist(), [base_to_l00(p_g, q_g)], gripper=float(state["grip"]),
                    max_step=0.003, avoid_collisions=True, vel=speed / 2, acc=speed / 2)
    if r is None:
        r = moveit_call(st, "lower_unchecked", MOVEIT.cartesian, q_arm().tolist(), [base_to_l00(p_g, q_g)],
                        gripper=float(state["grip"]), max_step=0.003, avoid_collisions=False, vel=speed / 2, acc=speed / 2)
    if r is None:  # no straight line down from here (a box held nearly level): a free, collision-checked motion to the
        r = moveit_call(st, "lower_free", MOVEIT.plan_pose, q_arm().tolist(), base_to_l00(p_g, q_g),  # same end pose
                        gripper=float(state["grip"]), pos_tol=0.004, rot_tol=0.04, time=5.0, attempts=5, vel=speed / 2,
                        acc=speed / 2)
    if r is None:
        return done(False, "lower")
    obs, _ = run_traj(obs, r, hold_s=0.4)
    bw = box.data.root_pos_w[0]
    st["before_release"] = {"box_above_table_cm": round(100 * (float(bw[2]) - F.BOX_SIZE[2] / 2 - P["top"]), 1),
                            "off_spot_cm": round(100 * math.hypot(float(bw[0]) - spot["xy"][0], float(bw[1]) - spot["xy"][1]), 1)}
    TRACE["sub"] = "release"
    obs = grip_move(obs, F.GRIPPER_OPEN, 0.8, 0.6)
    TRACE["sub"] = "withdraw"  # straight back out and up; the box is part of the world again
    moveit_scene(with_target=True)
    p_now, q_now = ee_pose()
    a = mu.quat_apply(q_now[None], torch.tensor([[1.0, 0.0, 0.0]], device=dev))[0]
    back = p_now - PLACE_BACK * a / a.norm() + torch.tensor([0.0, 0.0, PLACE_UP], device=dev)
    r = MOVEIT.cartesian(q_arm().tolist(), [base_to_l00(back, q_now)], gripper=float(state["grip"]), max_step=0.005,
                         avoid_collisions=False, vel=speed / 2, acc=speed / 2)
    st["moveit"]["withdraw"] = {k: r.get(k) for k in ("ok", "code", "fraction", "server_s", "error") if k in r}
    if r.get("points") and r.get("fraction", 0.0) >= 0.25:  # as far back as the line goes...
        obs, _ = run_traj(obs, r, hold_s=0.2)
    if not (r.get("points") and r.get("fraction", 0.0) >= 0.9):  # ...and, short of the end, straight up, the jaws
        p_up, q_up = ee_pose()                                    # open: they slide up the box's sides, clear of it
        r = moveit_call(st, "withdraw_up", MOVEIT.cartesian, q_arm().tolist(),
                        [base_to_l00(p_up + torch.tensor([0.0, 0.0, 0.10], device=dev), q_up)], gripper=float(state["grip"]),
                        max_step=0.005, avoid_collisions=False, vel=speed / 2, acc=speed / 2)
        if r is None:
            return done(False, "withdraw")
        obs, _ = run_traj(obs, r, hold_s=0.2)
    st["hand_off_box_cm"] = round(100 * float((box_in_base()[0] - ee_pose()[0]).norm()), 1)
    if st["hand_off_box_cm"] > 8.0:  # close the jaws only clear of the box (closed beside it, they took it along once)
        obs = grip_move(obs, F.GRIPPER_CLOSED, 0.6, 0.2)
    TRACE["sub"] = "stow"  # fold the arm for driving
    moveit_scene(with_target=True)
    stow = [F.ARM_STOW[f"joint{k}"] for k in range(1, 7)]
    r = moveit_call(st, "stow", MOVEIT.plan_joints, q_arm().tolist(), stow, gripper=float(state["grip"]), time=5.0,
                    attempts=5, vel=speed, acc=speed)
    if r is None:  # the hand still low over the robot's back: lift it first, then fold
        p_now, q_now = ee_pose()
        r = moveit_call(st, "stow_lift", MOVEIT.cartesian, q_arm().tolist(),
                        [base_to_l00(p_now + torch.tensor([0.0, 0.0, 0.08], device=dev), q_now)],
                        gripper=float(state["grip"]), max_step=0.005, avoid_collisions=False, vel=speed / 2, acc=speed / 2)
        if r is not None:
            obs, _ = run_traj(obs, r, hold_s=0.2)
            r = moveit_call(st, "stow", MOVEIT.plan_joints, q_arm().tolist(), stow, gripper=float(state["grip"]),
                            time=5.0, attempts=5, vel=speed, acc=speed)
    if r is not None:
        obs, _ = run_traj(obs, r, hold_s=0.3)
    obs = hold(obs, 1.0)  # the box settles
    return done(True)


def deliver_place(obs, rec, ep):
    """Shelf scene: step away from the shelf, drive to the table of the scene file (Nav2: staging pose, then docking
    beside the spot; else the scripted route) and put the box down there (place_moveit)."""
    PHASE["name"] = "deliver"
    t0 = u.episode_length_buf[0].item() * dt
    # strafe away (turning at the shelf sweeps the body into it), then back up into the open: the shelf stands
    # against a wall, and Nav2's controller would not turn the robot round in front of it ("collision ahead")
    for cmd, dist in (((0.0, 0.4 if SIDE["name"] == "right" else -0.4, 0.0), 0.5), ((-0.3, 0.0, 0.0), 0.6)):
        p0, t = robot.data.root_pos_w[0, :2].clone(), 0.0
        while t < 6.0 and float((robot.data.root_pos_w[0, :2] - p0).norm()) < dist:
            vel.vel_command_b[:] = torch.tensor(cmd, device=dev)
            obs = step(obs)
            t += dt
    vel.vel_command_b[:] = 0
    P = F.TABLES[PLACE_T["name"]]
    use_surface(P)
    sx, sy = place_spot(P)
    drive = rec["stages"].setdefault("place_drive", {})
    obs = navigate(obs, rec, ep, bp_w=torch.tensor([sx, sy, P["top"]], device=dev), stages=drive)
    obs = place_moveit(obs, rec)
    rec["stages"]["place"]["time_s"] = round(u.episode_length_buf[0].item() * dt - t0, 1)
    say(f"ep {ep} place {rec['stages']['place']} -> placed {rec.get('placed')}")
    use_surface(shelf_surface(TGT["table"], TGT["box"]))
    return obs


def begin(obs):
    state["grip"] = F.GRIPPER_CLOSED
    vel.vel_command_b[:] = 0
    state["arm"] = q_arm()   # hold the stow fold
    return hold(obs, 1.0)


def episode(obs, ep):
    rec = {"episode": ep, "stages": {}, "task_instruction": args.task_instruction,
           "deliver_task": bool(args.deliver) and not SHELF, "place_task": SHELF}
    if MULTI or SHELF:
        rec["target"] = {"table": TGT["table"], "box": TGT["box"]}
        if SHELF:
            rec["target"]["place"] = PLACE_T["name"]
    t0 = u.episode_length_buf[0].item() * dt
    p, yaw = base_pose()
    rec["start"] = [round(float(p[0]), 3), round(float(p[1]), 3), round(math.degrees(yaw), 1)]
    TRACE["sub"] = ""
    PROG.event("episode", ep=ep, n=args.episodes, start=rec["start"], task=args.task_instruction, vln=args.vln,
               target=rec.get("target"))
    record_begin(rec, args.vln)
    obs = begin(obs)
    if args.vln:
        rec["nav_instruction"] = args.vln
        obs = vln_walk(obs, rec, ep)
        PROG.event("vln_done", **{k: rec["stages"]["vln"].get(k) for k in ("reason", "arrived", "table_edge_m", "time_s", "path_m")})
        obs = back_off(obs, rec)
    obs = navigate(obs, rec, ep)
    PROG.event("dock_done", dock_err_cm=rec["stages"].get("dock_err_base_cm"),
               time_s=rec["stages"].get("locomotion", {}).get("time_s"), side=rec["stages"].get("side", {}).get("chosen"))
    obs = pick(obs, rec, args.pitch)
    PROG.event("pick_done", success=rec.get("success"), carry=rec["stages"].get("carry"))
    if SHELF:  # out of the shelf and onto the table
        rec["placed"] = False
        if rec["success"]:
            obs = deliver_place(obs, rec, ep)
        PROG.event("place_done", placed=rec["placed"], place=rec["stages"].get("place"))
    elif args.deliver:
        obs = deliver(obs, rec, ep)
        PROG.event("deliver_done", **{k: rec["stages"].get("deliver", {}).get(k) for k in ("arrived", "time_s", "box_to_grasp_cm")})
    rec["fell"] = tilt_deg() > 30
    record_end(rec)
    rec["time_s"] = round(u.episode_length_buf[0].item() * dt - t0, 1)
    for k, v in rec["stages"].items():
        if k != "locomotion":
            say(f"ep {ep} {k:9s} {v}")
    say(f"ep {ep} RESULT {'SUCCESS' if rec['success'] else 'FAIL'}" + (f" placed {rec['placed']}" if SHELF else "")
        + f"  total {rec['time_s']} s, fell {rec['fell']}")
    PROG.event("result", ep=ep, success=rec["success"], time_s=rec["time_s"], fell=rec["fell"],
               delivered=rec["stages"].get("deliver", {}).get("arrived") if args.deliver else None)
    return obs, rec


# ------------------------------------------------------------------ doors: open the blue one
# scenes/hospital_doors.yaml: three hinged doors with lever handles (fetch_env_cfg ...Door...). One is painted the
# target colour each episode; the robot walks to it (DualVLN), Nav2 parks it in front of the handle, MoveIt grasps
# the lever with the jaws closing vertically round its bar and turns it down along the lever's arc; the latch lets
# go (virtual: the hinge's drive stiff while latched, free once the lever is past latch_release); holding the lever
# down, the hand then pushes the door round its hinge -- a moving target on the door's arc, damped-least-squares IK
# every step, the base following a point behind the hand -- turns the lever back up, eases the arm off it, lets go
# and drives into the room.
DD = F.DOOR if DOOR else {}
DOOR_T = {"name": None, "colors": {}}  # this episode's door and every door's colour
DOOR_J = {k: (d.find_joints("hinge")[0][0], d.find_joints("handle_joint")[0][0]) for k, d in DOOR_ARTS.items()} if DOOR else {}
LATCH = {k: True for k in DOOR_J}
DOOR_GRIP_UP = 0.006  # m: the grasp point above the bar's centre (the control point is placed for 5 cm boxes:
#                       the fixed jaw's face then sits 1 cm under the 3.8 cm bar)
DOOR_PRE = 0.12       # m: pre-grasp this far back from the bar along the approach
DOOR_PITCHES = [30.0, 15.0, 45.0, 60.0]  # deg below level, tried in turn: the jaws close round the bar in the
#                       approach's vertical plane; level is out of the Z1's reach 0.2 m above its shoulder, from above
#                       too (moveit/door_reach.py: 15-60 deg found from 0.55-0.75 m)
DOOR_GRASP = {"pitch": 30.0}
DOOR_OPEN_DEG = 65.0  # the door pushed this far open holding the lever, then let go
DOOR_RATE = math.radians(8.0)   # rad/s: how fast the commanded door angle grows...
DOOR_TRACK = 0.04     # m: ...while the hand is within this of its target on the arc
DOOR_REACH = 0.62     # m: the base follows a point this far behind the hand (along x, the door's normal)
DOOR_ENTER = 1.0      # m: drive in until the base centre is this far past the room's wall
DOOR_UNTURN_TRACK = 0.02  # m: the lever is turned back up (--door_unturn_s) only while the hand is this close to its
#                           target on the lever's arc; at most 2.5x the time, then let go wherever it got to


DOOR_JAWS_S = 1.0     # s: the jaws open this slowly letting go of the lever


def grip_watch(obs, target, seconds, hold_s, k=None):
    """grip_move, the arm's targets fixed, measuring meanwhile how far the grasp point is knocked (base frame), the arm
    joints' peak speeds and (door k) the lever's."""
    w = {"hand_swing_cm": 0.0, "arm_peak_rad_s": [0.0] * len(ARM)}
    if k is not None:
        w["lever_peak_deg_s"], lv = 0.0, [door_angles(k)[1]] * 2
    p0, g0, n = ee_pose()[0].clone(), state["grip"], max(1, int(seconds / dt))
    for i in range(n + int(hold_s / dt)):
        state["grip"] = g0 + min(1.0, (i + 1) / n) * (target - g0)
        obs = step(obs)
        w["hand_swing_cm"] = max(w["hand_swing_cm"], round(100 * float((ee_pose()[0] - p0).norm()), 1))
        w["arm_peak_rad_s"] = [max(a, round(abs(float(v)), 2)) for a, v in zip(w["arm_peak_rad_s"], robot.data.joint_vel[0, ARM])]
        if k is not None:
            w["lever_peak_deg_s"] = max(w["lever_peak_deg_s"],
                                        round(abs(math.degrees(float(DOOR_ARTS[k].data.joint_vel[0, DOOR_J[k][1]]))), 1))
            lv = [min(lv[0], door_angles(k)[1]), max(lv[1], door_angles(k)[1])]
    if k is not None:
        w["lever_range_deg"] = [round(math.degrees(v), 1) for v in lv]
    return obs, w


def door_origin(k):
    return [float(v) for v in F.DOORS[k]["origin"]]


def door_rect(k):
    """Floor footprint of the shut door with its levers (x0, x1, y0, y1)."""
    o = door_origin(k)
    x0, x1, y0, y1 = DD["leaf"]
    return (o[0] + x0 - 0.11, o[0] + x1 + 0.11, o[1] + y0, o[1] + y1)


def door_angles(k):
    """(hinge, lever) angles in rad: the door open (+) and the lever pressed (-)."""
    d, (h, lv) = DOOR_ARTS[k], DOOR_J[k]
    return float(d.data.joint_pos[0, h]), float(d.data.joint_pos[0, lv])


def door_latch_update():
    """The virtual latch, every step: unlatch once the lever is pressed past latch_release; latch again once the door
    is back shut and the lever let go."""
    for k, d in DOOR_ARTS.items():
        hinge, lever = door_angles(k)
        if LATCH[k] and lever < -math.radians(DD["latch_release"]):
            LATCH[k] = False
            d.write_joint_stiffness_to_sim(torch.zeros(1, 1, device=dev), joint_ids=[DOOR_J[k][0]])
            PROG.event("note", text=f"door {k}: unlatched")
        elif not LATCH[k] and hinge < math.radians(1.0) and lever > -math.radians(DD["latch_release"] - 5.0):
            LATCH[k] = True
            d.write_joint_stiffness_to_sim(torch.full((1, 1), float(DD["latch_stiffness"]), device=dev),
                                           joint_ids=[DOOR_J[k][0]])


def paint_doors(target, rng):
    """The target door in the target colour, the others in the remaining colours (shuffled)."""
    import omni.usd
    from pxr import Gf
    stage = omni.usd.get_context().get_stage()
    tc = DD["target_color"]
    others = [c for c in DD["colors"] if c != tc]
    others = [others[i] for i in rng.permutation(len(others))]
    DOOR_T["colors"] = {}
    for k in DOOR_ARTS:
        c = tc if k == target else others.pop(0) if others else tc
        rgb = DD["colors"][c]
        stage.GetPrimAtPath(f"/World/envs/env_0/Door_{k}/Looks/Paint/MI_Door_01").GetAttribute(
            "inputs:ColorAlbedo").Set(Gf.Vec4f(*map(float, rgb), 1.0))
        DOOR_T["colors"][k] = c
    DOOR_T["name"] = target
    if CAMS:
        vln_nav.TABLE_RECT = door_rect(target)  # DualVLN's distance-to-goal is to this door


def door_dock_pose(k):
    o = door_origin(k)
    return (o[0] - F.SCENE_DOORS["dock"]["back"], o[1] + F.SCENE_DOORS["dock"]["side"], 0.0)


def _rx(a):
    return torch.tensor([[1.0, 0.0, 0.0], [0.0, math.cos(a), -math.sin(a)], [0.0, math.sin(a), math.cos(a)]], device=dev)


def _rz(a):
    return torch.tensor([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]], device=dev)


def door_grasp_w(k, phi, theta, back=0.0, pitch=None):
    """World pose of the grasp point holding door k's corridor-side lever, the door open phi and the lever turned
    theta (both rad), `back` m out along the approach. The approach (gripper x) heads into the door (+x) pitched
    `pitch` deg below level (DOOR_GRASP), the closing axis (gripper z) perpendicular to it in the vertical plane, up;
    then the lever's turn about the spindle (x) and the door's about the hinge (z) carry the grasp along."""
    pr = math.radians(DOOR_GRASP["pitch"] if pitch is None else pitch)
    a_ = torch.tensor([math.cos(pr), 0.0, -math.sin(pr)], device=dev)
    c_ = torch.tensor([math.sin(pr), 0.0, math.cos(pr)], device=dev)
    R0 = torch.stack([a_, torch.cross(c_, a_, dim=0), c_], dim=1)
    o = torch.tensor(door_origin(k), device=dev)
    g = torch.tensor(DD["grip"], device=dev) + DOOR_GRIP_UP * c_ - back * a_
    s = torch.tensor(DD["spindle"], device=dev)
    h = torch.tensor([DD["hinge"][0], DD["hinge"][1], 0.0], device=dev)
    R = _rz(phi) @ _rx(theta) @ R0
    p = h + _rz(phi) @ (s + _rx(theta) @ (g - s) - h)
    return o + p, mu.quat_from_matrix(R)


def world_to_base(p_w, q_w):
    p, q = mu.subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, p_w.reshape(1, 3), q_w.reshape(1, 4))
    return p[0], q[0]


def door_scene(k, phi=0.0, skip=()):
    """MoveIt's planning scene for door k: its leaf (open phi, without the levers: the jaws close on one), the wall on
    both sides of the opening, the other doors shut, the robot's body and the floor (link00 frame)."""
    yaw0 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev)
    objs = []
    x0, x1, y0, y1 = DD["leaf"]
    for j in DOOR_ARTS:
        o = door_origin(j)
        a = phi if j == k else 0.0
        h = (o[0] + DD["hinge"][0], o[1] + DD["hinge"][1])
        cx, cy = (x0 + x1) / 2 - DD["hinge"][0], (y0 + y1) / 2 - DD["hinge"][1]  # leaf centre from the hinge
        c = torch.tensor([h[0] + math.cos(a) * cx - math.sin(a) * cy, h[1] + math.sin(a) * cx + math.cos(a) * cy,
                          o[2] + 1.12], device=dev)
        q = torch.tensor([math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)], device=dev)
        objs.append((f"leaf_{j}", world_to_l00(c, q), (x1 - x0, y1 - y0, 2.12)))
    o = door_origin(k)
    for name, ya, yb in (("wall_s", o[1] + y0 - 2.5, o[1] + y0 - 0.005), ("wall_n", o[1] + y1 + 0.005, o[1] + y1 + 2.5)):
        c = torch.tensor([o[0] + 0.04, (ya + yb) / 2, 1.5], device=dev)
        objs.append((name, world_to_l00(c, yaw0), (0.10, yb - ya, 3.0)))
    for name, c, size in moveit_link.BODY_BOXES:
        if name not in skip:
            objs.append((name, base_to_l00(torch.tensor(c, device=dev), yaw0), size))
    floor = torch.tensor([float(robot.data.root_pos_w[0, 0]), float(robot.data.root_pos_w[0, 1]), -0.05], device=dev)
    objs.append(("floor", world_to_l00(floor, yaw0), (6.0, 6.0, 0.1)))
    r = MOVEIT.scene(objs, [])
    if not r.get("ok"):
        raise RuntimeError(f"MoveIt planning scene: {r.get('error')}")


def door_drive(obs, rec, ep, k):
    """Nav2 to door k: its staging pose (nav2_staging before the parking pose, heading into the door), then the docking
    server straight on to the parking pose. Returns (obs, ok)."""
    link, st = NAV2["link"], rec["stages"].setdefault("nav2", {})
    PHASE["name"] = "nav2"
    NAV2["pace"], NAV2["wall0"], NAV2["sim0"] = True, time.time(), NAV2["t"]
    holder = {"obs": obs}

    def idle():
        vel.vel_command_b[:] = 0
        holder["obs"] = step(holder["obs"])

    holder["obs"] = nav2_wait_ready(holder["obs"])
    if not NAV2["ready"]:
        NAV2["pace"] = False
        st["status"] = "not active"
        return holder["obs"], False
    dock = door_dock_pose(k)
    goal = (dock[0] - STAGING_BACK, dock[1], dock[2])
    NAV2["goal"] = goal
    link.clear_plan()
    t0, ntrace, kick = NAV2["t"], [], new_kick()
    PROG.event("nav2_plan", goal=[round(goal[0], 2), round(goal[1], 2), 0.0], stage="to the door's staging pose")
    status, path = nav2_relay(holder, link, args.nav2_timeout, ntrace, kick, t0) if link.navigate(goal, tick=idle) else (6, 0.0)
    if status != 4:  # once more, the costmaps emptied (marks from the walk, which may have ended against a wall)
        link.clear_costmaps()
        for _ in range(int(1.0 / dt)):
            idle()
        st["first_try"] = STATUS.get(status, str(status))
        status, p2 = nav2_relay(holder, link, args.nav2_timeout, ntrace, kick, t0) if link.navigate(goal, tick=idle) else (6, 0.0)
        path += p2
    e_cm, y_deg = nav_pose_err(goal)
    st.update(status=STATUS.get(status, str(status)), goal="staging", time_s=round(NAV2["t"] - t0, 1),
              driven_m=round(path, 2), err_cm=e_cm, yaw_err_deg=y_deg)
    say(f"ep {ep} nav2 {st}")
    PROG.event("nav2_done", **st)
    ok = status == 4
    if ok:
        PHASE["name"] = "nav2_dock"
        NAV2["goal"] = dock
        PROG.event("nav2_plan", goal=[round(dock[0], 2), round(dock[1], 2), 0.0], stage="docking in front of the door")
        t1 = NAV2["t"]
        sd, pd = nav2_relay(holder, link, args.nav2_timeout, ntrace, kick, t0) if link.dock(dock, tick=idle) else (6, 0.0)
        res = link.last_result()
        e_cm, y_deg = nav_pose_err(dock)
        sdock = {"status": STATUS.get(sd, str(sd)), "success": bool(res.success) if res is not None else False,
                 "time_s": round(NAV2["t"] - t1, 1), "driven_m": round(pd, 2), "err_cm": e_cm, "yaw_err_deg": y_deg}
        rec["stages"]["nav2_dock"] = sdock
        say(f"ep {ep} nav2 dock {sdock}")
        PROG.event("nav2_done", stage="docking", **sdock)
        ok = sd == 4 and sdock["success"]
    NAV2["goal"], NAV2["pace"] = None, False
    return holder["obs"], ok


def door_settle_at(obs, pose, seconds=6.0, tol=0.03):
    """Without Nav2 (or after it fell short): hold the base on a pose with the proportional hold until within tol."""
    BASE_HOLD["xy"] = torch.tensor(pose[:2], device=dev)
    BASE_HOLD["yaw"] = pose[2]
    t = 0.0
    while t < seconds:
        obs = step(obs)
        t += dt
        p, _ = base_pose()
        if math.hypot(float(p[0]) - pose[0], float(p[1]) - pose[1]) < tol and t > 1.0:
            break
    return obs


def door_open(obs, rec, k):
    """Grasp door k's lever, turn it down, push the door open along its arc holding it, let go. rec["stages"]["door"]."""
    st = rec["stages"].setdefault("door", {"moveit": {}})
    speed = args.moveit_speed
    PHASE["name"] = "door"
    p, yaw = base_pose()
    BASE_HOLD["xy"], BASE_HOLD["yaw"], BASE_HOLD["engaged_steps"] = p[:2].clone(), yaw, 0
    MI["int"] = None
    d0 = door_dock_pose(k)
    st["dock_err_cm"] = [round(100 * (float(p[0]) - d0[0]), 1), round(100 * (float(p[1]) - d0[1]), 1)]
    theta = -math.radians(DD["handle_press"])

    def fail(stage):
        st["failed_at"] = stage
        st["hinge_deg"], st["lever_deg"] = (round(math.degrees(v), 1) for v in door_angles(k))
        BASE_HOLD["xy"] = None
        vel.vel_command_b[:] = 0
        say(f"door {k}: FAILED at {stage}")
        return obs

    # 1 pre-grasp: a posture from which the straight approach to the bar works (the jaws open)
    TRACE["sub"] = "pre_grasp"
    door_scene(k)
    for pitch in DOOR_PITCHES:
        DOOR_GRASP["pitch"] = pitch
        pre_w, q_w = door_grasp_w(k, 0.0, 0.0, back=DOOR_PRE)
        fin_w, _ = door_grasp_w(k, 0.0, 0.0)
        pre_b, qb = world_to_base(pre_w, q_w)
        fin_b, _ = world_to_base(fin_w, q_w)
        ap = moveit_call(st, "approach_ik", MOVEIT.approach, q_arm().tolist(), base_to_l00(pre_b, qb),
                         base_to_l00(fin_b, qb), gripper=float(state["grip"]), approach_gripper=F.GRIPPER_OPEN, tries=32)
        if ap is not None:
            break
    if ap is None:
        return fail("approach_ik")
    st["pitch"] = DOOR_GRASP["pitch"]
    r = moveit_call(st, "pre_grasp", MOVEIT.plan_joints, q_arm().tolist(), ap["joints"], gripper=float(state["grip"]),
                    time=5.0, attempts=5, vel=speed, acc=speed)
    if r is None:
        return fail("pre_grasp")
    obs, _ = run_traj(obs, r, hold_s=0.4)
    TRACE["sub"] = "open"
    obs, st["jaws_in_air"] = grip_watch(obs, F.GRIPPER_OPEN, 0.6, 0.3)  # (for comparison: opening, holding nothing)
    # 2 straight in round the bar, close
    TRACE["sub"] = "approach"
    fin_b, qb = world_to_base(*door_grasp_w(k, 0.0, 0.0))
    r = moveit_call(st, "approach", MOVEIT.cartesian, q_arm().tolist(), [base_to_l00(fin_b, qb)], gripper=float(state["grip"]),
                    max_step=0.005, avoid_collisions=True, vel=speed / 2, acc=speed / 2)
    if r is None:
        r = moveit_call(st, "approach_unchecked", MOVEIT.cartesian, q_arm().tolist(), [base_to_l00(fin_b, qb)],
                        gripper=float(state["grip"]), max_step=0.005, avoid_collisions=False, vel=speed / 2, acc=speed / 2)
    if r is None:
        return fail("approach")
    obs, _ = run_traj(obs, r, hold_s=0.4)
    st["grasp_err_cm"] = round(100 * pose_err(fin_b, qb)[0], 2)
    TRACE["sub"] = "close"
    obs = grip_move(obs, F.GRIPPER_CLOSED, 1.0, 0.4)
    st["grip_rad"] = round(float(robot.data.joint_pos[0, GRIP]), 3)
    # 3 turn the lever down: the hand on the lever's arc round the spindle
    TRACE["sub"] = "turn"
    wps = []
    for i in range(1, 9):
        pw, qw = door_grasp_w(k, 0.0, theta * i / 8)
        pb, qb = world_to_base(pw, qw)
        wps.append(base_to_l00(pb, qb))
    r = moveit_call(st, "turn", MOVEIT.cartesian, q_arm().tolist(), wps, gripper=float(state["grip"]), max_step=0.004,
                    avoid_collisions=False, vel=speed / 3, acc=speed / 3)
    if r is None:
        return fail("turn")
    obs, _ = run_traj(obs, r, hold_s=0.5)
    hinge, lever = door_angles(k)
    st["lever_turned_deg"] = round(math.degrees(lever), 1)
    st["unlatched"] = not LATCH[k]
    say(f"door {k}: lever {math.degrees(lever):.1f} deg, {'unlatched' if not LATCH[k] else 'STILL LATCHED'}")
    if LATCH[k]:
        return fail("turn")
    # 4 push the door open round its hinge, holding the lever down: the hand's target is the grasp on the door's arc
    #   at the commanded angle (IK every step), the base follows a point DOOR_REACH behind the hand
    TRACE["sub"] = "push"
    PROG.event("note", text=f"door {k}: pushing it open")
    phi, t, stuck = 0.0, 0.0, 0.0
    lever_min = lever
    o_ = door_origin(k)
    hx, hy = o_[0] + DD["hinge"][0], o_[1] + DD["hinge"][1]
    g0, _ = door_grasp_w(k, 0.0, theta)
    ang0 = math.atan2(float(g0[1]) - hy, float(g0[0]) - hx)

    def door_from_hand():
        """The door's angle as the hand holding its lever sees it: the hand's bearing round the hinge (no door state)."""
        pb_, qb_ = ee_pose()
        pw_, _ = mu.combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, pb_[None], qb_[None])
        a_ = math.atan2(float(pw_[0, 1]) - hy, float(pw_[0, 0]) - hx) - ang0
        return math.atan2(math.sin(a_), math.cos(a_))

    while phi < math.radians(DOOR_OPEN_DEG) and t < 30.0 and stuck < 4.0:
        phi = max(phi, door_from_hand())  # ahead of the command (the body nudged it): follow, do not fight it
        pw, qw = door_grasp_w(k, phi, theta)
        pb, qb = world_to_base(pw, qw)
        state["arm"] = ik_target(pb, qb)
        BASE_HOLD["xy"] = torch.tensor([float(pw[0]) - DOOR_REACH, d0[1]], device=dev)
        BASE_HOLD["yaw"] = 0.0
        obs = step(obs)
        t += dt
        e = pose_err(*world_to_base(pw, qw))[0]
        if e < DOOR_TRACK:
            phi += DOOR_RATE * dt
            stuck = 0.0
        else:
            stuck += dt
        lever_min = min(lever_min, door_angles(k)[1])
    hinge, lever = door_angles(k)
    st["push"] = {"commanded_deg": round(math.degrees(phi), 1), "door_deg": round(math.degrees(hinge), 1),
                  "lever_deg": round(math.degrees(lever), 1), "time_s": round(t, 1), "stuck_s": round(stuck, 1),
                  "hand_err_cm": round(100 * e, 1)}
    say(f"door {k}: pushed {st['push']}")
    # 5 let go. First the lever back up under the hand -- the turn of step 3 in reverse, about the spindle where the
    #   open door has carried it (the door's angle as the hand sees it), the base held. Let go pressed, the lever's
    #   return spring snapped it up into the opening jaws, knocked the hand and the arm swung. Then the jaws open, the
    #   hand straight back out of the lever, the arm folded.
    phi_r, th = door_from_hand(), theta
    if lever > -math.radians(5.0):  # the lever already back up: the jaws lost it on the way (it was pressed to its
        th = lever                  # -45 deg stop once and slipped out); nothing to turn back, the hand is not on it
        st["unturn"] = {"skipped": "lever not in hand", "lever_deg": round(math.degrees(lever), 1)}
    elif args.door_unturn_s > 0:
        TRACE["sub"] = "unturn"
        PROG.event("note", text=f"door {k}: turning the lever back up")
        s, t_u, held, e = 0.0, 0.0, 0.0, 0.0
        while held < 0.5 and t_u < 2.5 * args.door_unturn_s:
            m = s * s * s * (10 - 15 * s + 6 * s * s)  # min-jerk: starts and ends gently
            th = theta * (1.0 - m)
            pw, qw = door_grasp_w(k, phi_r, th)
            state["arm"] = ik_target(*world_to_base(pw, qw))
            obs = step(obs)
            t_u += dt
            e = pose_err(*world_to_base(pw, qw))[0]
            if s >= 1.0:
                held += dt
            elif e < DOOR_UNTURN_TRACK:
                s = min(1.0, s + dt / args.door_unturn_s)
        hinge, lever = door_angles(k)
        st["unturn"] = {"lever_deg": round(math.degrees(lever), 1), "commanded_deg": round(math.degrees(th), 1),
                        "door_deg": round(math.degrees(hinge), 1), "time_s": round(t_u, 1), "hand_err_cm": round(100 * e, 1)}
        say(f"door {k}: lever back up {st['unturn']}")
    TRACE["sub"] = "release"
    # Then the arm eased off the lever, the jaws still closed. The integral IK leaves the joint targets ahead of the
    # joints wherever the grip stops the hand (up to IK_WINDUP: 25 N*m a joint at stiffness 100): a wound spring
    # pressing on the lever (25-32 N*m measured), let loose the instant the jaws open: the hand knocked 11-24 cm.
    # The targets ramp to where the arm holds only its own weight: the joints plus gravity / stiffness, taken at the
    # start of the ramp (taken every step, the targets follow the joints and the arm has no stiffness left: it
    # drifted, the lever pulled down 13-21 deg); twice, the hand giving a little as the pressing goes. The arm stays
    # stiff round them while the jaws open. (Tried: the IK holding the hand on the lever through the ramp, its targets
    # clamped ever closer round the free ones -- the lever sank as much, 5-7 deg; kept on that weak hold while the
    # jaws opened, the arm drifted off, the hand 6-18 cm.)
    def arm_free():
        g_ = robot.root_physx_view.get_gravity_compensation_forces()[0].to(dev)
        g_ = g_[JAC_COLS] if g_.shape[0] == robot.num_joints + 6 else g_[ARM]  # floating base: 6 base columns first
        k_ = robot.data.joint_stiffness[0, ARM]
        return q_arm() + g_ / k_, k_ * (state["arm"] - q_arm()) - g_  # free targets; the torque pressing on the lever

    rel = {"lever_deg": round(math.degrees(lever), 1), "door_deg": round(math.degrees(hinge), 1),
           "pressing_nm": round(float(arm_free()[1].abs().max()), 1)}
    if args.door_ease_s > 0:
        for secs in (args.door_ease_s, 0.3):  # twice: 2-5 N*m were left after one ramp
            a0, a1, n = state["arm"].clone(), arm_free()[0], max(1, int(secs / dt))
            for i in range(n + int(0.3 / dt)):
                m = min(1.0, (i + 1) / n)
                state["arm"] = (1 - m) * a0 + m * a1
                obs = step(obs)
        hinge, lever = door_angles(k)
        rel["eased"] = {"pressing_nm": round(float(arm_free()[1].abs().max()), 1), "lever_deg": round(math.degrees(lever), 1),
                        "door_deg": round(math.degrees(hinge), 1)}
    # the jaws open, slowly: how far the hand is knocked, and how fast the lever and the arm move
    obs, w = grip_watch(obs, F.GRIPPER_OPEN, DOOR_JAWS_S, 0.6, k)
    rel.update(w)
    st["release"] = rel
    say(f"door {k}: let go {rel}")
    pw, qw = door_grasp_w(k, phi_r, th, back=0.10)
    obs, _, _ = ik_move(obs, lambda: world_to_base(pw, qw), 1.2, hold_s=0.2)
    obs = grip_move(obs, F.GRIPPER_CLOSED, 0.4, 0.1)
    TRACE["sub"] = "stow"
    door_scene(k, hinge)
    stow = [F.ARM_STOW[f"joint{i}"] for i in range(1, 7)]
    r = MOVEIT.plan_joints(q_arm().tolist(), stow, gripper=float(state["grip"]), time=5.0, attempts=5, vel=speed, acc=speed)
    st["moveit"]["stow"] = {kk: r.get(kk) for kk in ("ok", "code", "start_contacts") if kk in r}
    if not r.get("ok") and r.get("start_contacts") and all("lidar_cage" in c for c in r["start_contacts"]):
        # the arm ends the push reaching out over the robot's head, inside MoveIt's (generous) box round the lidar
        # cage: plan the fold without that box (the joint path then lifts the arm off it first)
        door_scene(k, hinge, skip=("lidar_cage",))
        r = moveit_call(st, "stow_no_cage", MOVEIT.plan_joints, q_arm().tolist(), stow, gripper=float(state["grip"]),
                        time=5.0, attempts=5, vel=speed, acc=speed)
    elif not r.get("ok"):
        r = None
    if r is not None and r.get("ok", True) and r.get("points"):
        obs, _ = run_traj(obs, r, hold_s=0.3)
    else:
        obs = joint_move(obs, stow, 2.5)
    BASE_HOLD["xy"] = None
    st["hinge_deg"], st["lever_deg"] = (round(math.degrees(v), 1) for v in door_angles(k))
    return obs


def door_enter(obs, rec, k):
    """Drive straight in through the opening (the body pushes the door further if it is in the way)."""
    PHASE["name"] = "enter"
    o, room = door_origin(k), F.DOORS[k]["room"]
    y_line = o[1] + DD["leaf"][2] / 2 - 0.13  # a little to the handle side of the opening's middle
    x_goal = room[0] + DOOR_ENTER
    t, p0 = 0.0, base_pose()[0][:2].clone()
    while t < 20.0:
        p, yaw = base_pose()
        if float(p[0]) >= x_goal:
            break
        ey = y_line - float(p[1])
        vel.vel_command_b[:] = torch.tensor([0.35, max(-0.2, min(0.2, 1.5 * ey)), max(-0.4, min(0.4, -2.0 * yaw))], device=dev)
        obs = step(obs)
        t += dt
    vel.vel_command_b[:] = 0
    obs = hold(obs, 1.0)
    p, _ = base_pose()
    inside = room[0] + 0.6 <= float(p[0]) <= room[1] and room[2] <= float(p[1]) <= room[3]
    rec["stages"]["enter"] = {"base": [round(float(p[0]), 2), round(float(p[1]), 2)], "inside": inside,
                              "time_s": round(t, 1), "door_deg": round(math.degrees(door_angles(k)[0]), 1)}
    return obs, inside


def episode_door(obs, ep, k):
    rec = {"episode": ep, "stages": {}, "task_instruction": args.task_instruction, "door_task": True,
           "target": {"door": k, "color": DOOR_T["colors"].get(k), "colors": dict(DOOR_T["colors"])}}
    t0 = u.episode_length_buf[0].item() * dt
    p, yaw = base_pose()
    rec["start"] = [round(float(p[0]), 3), round(float(p[1]), 3), round(math.degrees(yaw), 1)]
    TRACE["sub"] = ""
    PROG.event("episode", ep=ep, n=args.episodes, start=rec["start"], task=args.task_instruction, vln=args.vln,
               target=rec["target"])
    record_begin(rec, args.vln)
    obs = begin(obs)
    if args.vln:
        rec["nav_instruction"] = args.vln
        obs = vln_walk(obs, rec, ep)
        PROG.event("vln_done", **{kk: rec["stages"]["vln"].get(kk) for kk in ("reason", "arrived", "table_edge_m", "time_s", "path_m")})
        if args.walk_only:
            v = rec["stages"]["vln"]
            rec.update(success=bool(v.get("arrived")), fell=tilt_deg() > 30, door_opened=False, entered=False,
                       time_s=round(u.episode_length_buf[0].item() * dt - t0, 1))
            say(f"ep {ep} WALK {v['reason']} at {v['end']}, {v['table_edge_m']} m from door {k}, nearest {v.get('nearest')}")
            return obs, rec
    ok = False
    if NAV2["link"] is not None:
        obs, ok = door_drive(obs, rec, ep, k)
    if not ok:  # no Nav2, or it fell short: the proportional hold onto the parking pose
        obs = door_settle_at(obs, door_dock_pose(k), seconds=12.0)
        BASE_HOLD["xy"] = None
    obs = door_open(obs, rec, k)
    st = rec["stages"]["door"]
    rec["door_opened"] = st.get("hinge_deg", 0.0) >= 45.0
    rec["entered"] = False
    if rec["door_opened"]:
        obs, rec["entered"] = door_enter(obs, rec, k)
    rec["success"] = bool(rec["entered"])
    rec["fell"] = tilt_deg() > 30
    record_end(rec)
    rec["time_s"] = round(u.episode_length_buf[0].item() * dt - t0, 1)
    for kk, v in rec["stages"].items():
        say(f"ep {ep} {kk:9s} {v}")
    say(f"ep {ep} RESULT {'SUCCESS' if rec['success'] else 'FAIL'} door {k} ({rec['target']['color']}) opened "
        f"{rec['door_opened']} entered {rec['entered']}  total {rec['time_s']} s, fell {rec['fell']}")
    PROG.event("result", ep=ep, success=rec["success"], time_s=rec["time_s"], fell=rec["fell"])
    return obs, rec


def door_probe(obs, k):
    """--probe in the door scene: the robot set down on door k's parking pose (no walk, no Nav2), then the opening."""
    dock = door_dock_pose(k)
    with torch.inference_mode():
        st_ = robot.data.default_root_state.clone()
        st_[:, 0], st_[:, 1] = dock[0], dock[1]
        st_[:, 3:7] = torch.tensor([math.cos(dock[2] / 2), 0.0, 0.0, math.sin(dock[2] / 2)], device=dev)
        robot.write_root_state_to_sim(st_)
    obs = begin(obs)
    obs = door_settle_at(obs, dock, seconds=3.0)
    rec = {"episode": 0, "stages": {}, "target": {"door": k, "colors": dict(DOOR_T["colors"])}}
    obs = door_open(obs, rec, k)
    st = rec["stages"]["door"]
    opened = st.get("hinge_deg", 0.0) >= 45.0
    entered = False
    if opened:
        obs, entered = door_enter(obs, rec, k)
    say(f"door probe {k}: {rec['stages']}")
    say(f"door probe RESULT opened {opened} entered {entered}")
    return obs, rec


# ------------------------------------------------------------------ probe
def probe(obs):
    """No driving. The box is placed straight ahead at the pick distance by moving the
    ROBOT's start next to the table; then: ready pose for each roll sign, and IK
    errors for the pick waypoints."""
    global READY_ROLL
    if SHELF and args.target != "auto":  # shelf/colour/table
        ts, tc, tt = args.target.split("/")
        set_target(ts, tc, tt)
    for off, dyaw in [(float(o), math.radians(float(y))) for o in args.probe_lateral.split(",") for y in args.probe_yaw.split(",")]:
      for pitch in [args.pitch] * args.probe_n:
            with torch.inference_mode():
                obs, _ = env.reset()
                # teleport the robot to the standoff point in front of the box, facing +x
                root = robot.data.default_root_state.clone()
                set_side("left" if args.side == "left" else "right")
                goal, _, psi = dock_pose(box.data.root_pos_w[0])
                root[:, 0], root[:, 1] = float(goal[0]) + off, float(goal[1])  # +off: toward the table
                root[:, 3:7] = torch.tensor([math.cos((psi + dyaw) / 2), 0.0, 0.0, math.sin((psi + dyaw) / 2)], device=dev)
                robot.write_root_state_to_sim(root)
            if args.video:
                video_start()
            obs = begin(obs)
            rec = {"stages": {}}
            obs = pick(obs, rec, pitch)
            if SHELF and rec["success"]:  # and onto the table (scripted drive, or Nav2 with --nav2)
                rec["placed"] = False
                obs = deliver_place(obs, rec, 0)
            say(f"probe offset {off * 100:+.0f} cm yaw {math.degrees(dyaw):+.0f} deg pitch {pitch:.0f}: " + " | ".join(f"{k} {v}" for k, v in rec["stages"].items())
                + f" | success {rec['success']} tilt {tilt_deg():.1f}")
            video_end(f"{args.video_dir}/probe_seed{args.seed}_off{off * 100:+.0f}{'_' + TGT['box'] if SHELF else ''}_{args.cam}_"
                      f"{'SUCCESS' if rec['success'] else 'FAIL'}.mp4")
            TRACE["ep"] += 1
    if TRACE["path"] and TRACE["rows"]:
        np.savez(TRACE["path"], sub=np.array([r[1] for r in TRACE["rows"]]),
                 data=np.array([[r[0]] + list(r[2:]) for r in TRACE["rows"]], dtype=np.float32), dt=dt)
    return obs


TASK_TEXTS = ["Pick up the box on the wooden table.", "Go to the wooden table and pick up the box.",
              "Grab the small box from the wooden table.", "Lift the cardboard box off the table.",
              "Walk to the table and take the box."]
TASK_TEXTS_DELIVER = ["Pick up the box on the wooden table and bring it back.",
                      "Bring me the box from the wooden table.",
                      "Fetch the cardboard box from the table and carry it to the drop zone.",
                      "Go to the wooden table, grab the box and bring it back."]
PICK_VERB = r"\b(pick|grab|take|lift|fetch|bring)\b"  # not "get": "get to the table"
DELIVER = r"\b(bring|deliver|carry)\b|\bback\b|\bto me\b|drop zone"


def parse_task(text):
    """'go to the wooden table and pick the box' -> (navigation sentence for DualVLN, pick?, deliver?).
    The walking part is what comes before 'and pick ...'; DualVLN, a navigation model, gets only
    that, with 'and stop in front of it' so it knows where to end. A bare 'pick up the box on the
    wooden table' walks on the default navigation sentence. The pick is the scripted one: it
    picks THE box, whatever the words say about it."""
    low = text.lower()
    wants_pick = re.search(PICK_VERB, low) is not None
    wants_deliver = re.search(DELIVER, low) is not None
    split = re.search(r"\b(and then|and|then)\b\s+" + PICK_VERB, low)
    nav_part = text[:split.start()].strip(" ,.") if split else ("" if wants_pick else text.strip(" ,."))
    if not nav_part:
        nav_part = "Walk to the wooden table and stop in front of it"
    elif "stop" not in nav_part.lower():
        nav_part += " and stop in front of it"
    return vln_nav.normalise_instruction(nav_part), wants_pick, wants_deliver


def live():
    """--interactive: wait for typed tasks (terminal or --watch file) and do them."""
    global obs
    vln_nav.start_command_readers(commands, args.watch, say)
    holding, n = False, 0
    if args.starts:  # the first start pose is home (the task resets the robot to its default root state)
        sx, sy, syaw = (float(v) for v in args.starts.split(";")[0].split(","))
        with torch.inference_mode():
            st = robot.data.default_root_state
            st[:, 0], st[:, 1] = sx, sy
            st[:, 3:7] = torch.tensor([math.cos(math.radians(syaw) / 2), 0.0, 0.0, math.sin(math.radians(syaw) / 2)],
                                      device=dev)
            obs, _ = env.reset()
    obs = begin(obs)
    door_rng = np.random.default_rng(args.seed)
    if DOOR:
        paint_doors(args.target if args.target in F.DOORS else str(door_rng.choice(sorted(F.DOORS))), door_rng)
        _cols = ", ".join(f"{F.DOORS[k]['label']} {c}" for k, c in DOOR_T["colors"].items())
        say(f"READY. Type a task, e.g.  open the {DD['target_color']} door   ({_cols})")
        say(f"      the robot {'walks there with DualVLN, ' if USE_VLN else ''}parks in front of it, turns the handle, "
            "pushes it open and goes into the room")
        say("      home = reset (doors shut, colours shuffled)   q = quit")
    elif SHELF:
        _sh, _tb = next(iter(F.SHELVES.values()))["label"], next(iter(F.TABLES.values()))["label"]
        say(f"READY. Type a task, e.g.  take the red box from the {_sh} and put it on the {_tb}")
        say(f"      shelves: {', '.join(v['label'] for v in F.SHELVES.values())}; tables: "
            f"{', '.join(v['label'] for v in F.TABLES.values())}; boxes: {', '.join(F.SHELF_COLORS)}")
        if any("levels" in v for v in F.SHELVES.values()):
            say("      or by its tray: 'pick the box on the top shelf' (bottom / middle); 'pick up the green box' only "
                "picks it -- then 'put it on the table'")
        say(f"      home = reset robot and boxes   q = quit" + ("   (DualVLN walks to the shelf first)" if USE_VLN else ""))
    else:
        say("READY. Type a task, e.g.  go to the wooden table and pick the box")
        say("      'walk to the table' only walks; '... and bring it back' also delivers it to the drop zone")
        say("      stop = halt   home = reset robot and box   q = quit")
    while app.is_running():
        if commands.empty():
            vel.vel_command_b[:] = 0
            obs = step(obs)  # stand still, waiting
            continue
        cmd = commands.get()
        if cmd[0] == "quit":
            break
        if cmd[0] == "stop":
            say("stopped, waiting")
            continue
        if cmd[0] == "home":
            with torch.inference_mode():
                obs, _ = env.reset()
            obs, holding = begin(obs), False
            if DOOR:
                for _k in LATCH:
                    LATCH[_k] = True
                paint_doors(args.target if args.target in F.DOORS else str(door_rng.choice(sorted(F.DOORS))), door_rng)
                say("robot back at the start, doors shut: " + ", ".join(f"{F.DOORS[k]['label']} {c}"
                                                                       for k, c in DOOR_T["colors"].items()))
                continue
            say("robot and box back at the start")
            continue
        if DOOR:  # "open the blue door": the door painted that colour (any of the door colours)
            low = cmd[1].lower()

            def tell(text):
                say(text)
                PROG.event("note", text=text)

            m = re.search(rf"\b({'|'.join(map(re.escape, DD['colors']))})\b", low)
            colour = m.group(1) if m else (DD["target_color"] if re.search(r"\bdoor\b", low) else None)
            if colour is None:
                tell(f"which door? {', '.join(DD['colors'])} -- e.g. 'open the {DD['target_color']} door'")
                continue
            k = next((kk for kk, c in DOOR_T["colors"].items() if c == colour), None)
            if k is None:
                tell(f"no {colour} door here ({', '.join(DOOR_T['colors'].values())})")
                continue
            DOOR_T["name"] = k
            if CAMS:
                vln_nav.TABLE_RECT = door_rect(k)
            nav_text = cmd[1].strip() if args.walk_only else f"Walk to the {colour} door and stop in front of it."
            tell(f"the {colour} door ({F.DOORS[k]['label']}): {'walking there, ' if USE_VLN else ''}then opening it")
            rec = {"episode": n, "stages": {}, "task_instruction": cmd[1], "nav_instruction": nav_text if USE_VLN else "",
                   "target": {"door": k, "color": colour}, "door_task": True}
            p, yaw = base_pose()
            rec["start"] = [round(float(p[0]), 3), round(float(p[1]), 3), round(math.degrees(yaw), 1)]
            record_begin(rec, nav_text if USE_VLN else "")
            PROG.event("episode", ep=n, n=None, start=rec["start"], task=cmd[1], vln=nav_text if USE_VLN else "")
            if USE_VLN:
                obs = vln_walk(obs, rec, n, nav_text)
                if rec["stages"]["vln"]["reason"] == "interrupted":
                    tell("interrupted, waiting")
                    continue
                if args.walk_only:  # (comparing walk sentences: the typed sentence is the instruction)
                    v = rec["stages"]["vln"]
                    tell(f"WALK '{nav_text}': {v['reason']} at {v['end']}, {v['table_edge_m']} m from the {colour} door "
                         f"({k}), nearest {v.get('nearest')}; waiting")
                    n += 1
                    continue
            ok = False
            if NAV2["link"] is not None:
                obs, ok = door_drive(obs, rec, n, k)
            if not ok:
                obs = door_settle_at(obs, door_dock_pose(k), seconds=12.0)
                BASE_HOLD["xy"] = None
            obs = door_open(obs, rec, k)
            opened = rec["stages"]["door"].get("hinge_deg", 0.0) >= 45.0
            entered = False
            if opened:
                obs, entered = door_enter(obs, rec, k)
            rec["success"], rec["fell"] = entered, tilt_deg() > 30
            record_end(rec)
            PROG.event("result", ep=n, success=entered, time_s=None, fell=rec["fell"])
            n += 1
            tell(("in the room" if entered else "opened it, but did not get in" if opened else "could not open it")
                 + "; waiting for the next task ('home' to start over)")
            continue
        if SHELF:  # "take the green box from the shelf and put it on the table": (DualVLN,) Nav2, pick, Nav2, place;
            # "pick up the green box": pick and hold; "put it on the table": the held box onto the table
            low = cmd[1].lower()
            ts, tc, tt = parse_shelf_task(cmd[1])
            wants_place = re.search(r"\b(put|place|bring|move|deliver|carry|set|leave)\b|\btable\b", low) is not None

            def tell(text):  # in the terminal and in ./project/process.sh
                say(text)
                PROG.event("note", text=text)

            if holding:  # a box in the gripper: only putting it down makes sense
                if not wants_place:
                    tell(f"I am holding the {TGT['box']} box: say 'put it on the table' (or 'home' to start over)")
                    continue
                PLACE_T["name"] = tt or PLACE_T["name"] or next(iter(F.TABLES))
                tell(f"putting the {TGT['box']} box on the {F.TABLES[PLACE_T['name']]['label']}")
                rec = {"episode": n, "stages": {}, "task_instruction": cmd[1], "nav_instruction": "",
                       "target": {"table": TGT["table"], "box": TGT["box"], "place": PLACE_T["name"]}, "place_task": True,
                       "placed": False, "success": True}
                PROG.event("episode", ep=n, n=None, start=[round(float(v), 2) for v in base_pose()[0][:2]] + [0.0],
                           task=cmd[1], vln="")
                obs = deliver_place(obs, rec, n)
                PROG.event("place_done", placed=rec["placed"], place=rec["stages"].get("place"))
                holding = not rec["placed"]
                PROG.event("result", ep=n, success=bool(rec["placed"]), time_s=None, fell=tilt_deg() > 30)
                n += 1
                tell("placed -- waiting for the next task" if rec["placed"] else "could not put it down")
                continue
            boxes_here = {s: [b[0] for b in F.shelf_boxes(s)] for s in F.SHELVES}
            if tc is None:
                tell(f"which box? {', '.join(sorted({c for v in boxes_here.values() for c in v}))} -- e.g. 'take the "
                     f"green box from the {next(iter(F.SHELVES.values()))['label']} and put it on the "
                     f"{next(iter(F.TABLES.values()))['label']}'")
                continue
            if ts is None:  # not named: the shelf that holds a box of that colour, if only one does
                having = [s for s, cs in boxes_here.items() if tc in cs]
                if len(having) != 1:
                    tell(f"which shelf? {', '.join(F.SHELVES[s]['label'] for s in having or F.SHELVES)}")
                    continue
                ts = having[0]
            if tc not in boxes_here[ts]:
                tell(f"there is no {tc} box on the {F.SHELVES[ts]['label']} ({', '.join(boxes_here[ts])})")
                continue
            if tt is None and wants_place:
                if len(F.TABLES) > 1:
                    tell(f"which table? {', '.join(v['label'] for v in F.TABLES.values())}")
                    continue
                tt = next(iter(F.TABLES))
            set_target(ts, tc, tt)
            nav_text = shelf_texts(ts, tc, PLACE_T["name"])[0]
            tell(f"the {tc} box on the {F.SHELVES[ts]['label']}" + (f", then onto the {F.TABLES[PLACE_T['name']]['label']}"
                                                                     if wants_place else " -- I will hold it"))
            rec = {"episode": n, "stages": {}, "task_instruction": cmd[1], "nav_instruction": nav_text if USE_VLN else "",
                   "target": {"table": ts, "box": tc, "place": PLACE_T["name"] if wants_place else None},
                   "place_task": wants_place, "placed": False}
            p, yaw = base_pose()
            rec["start"] = [round(float(p[0]), 3), round(float(p[1]), 3), round(math.degrees(yaw), 1)]
            record_begin(rec, nav_text if USE_VLN else "")
            TRACE["sub"] = ""
            PROG.event("episode", ep=n, n=None, start=rec["start"], task=cmd[1], vln=nav_text if USE_VLN else "")
            if USE_VLN:  # DualVLN walks towards the shelf from the camera; then the robot backs off and parks
                obs = vln_walk(obs, rec, n, nav_text)
                PROG.event("vln_done", **{k: rec["stages"]["vln"].get(k) for k in ("reason", "arrived", "table_edge_m", "time_s", "path_m")})
                if rec["stages"]["vln"]["reason"] == "interrupted":
                    tell("interrupted, waiting")
                    continue
                obs = back_off(obs, rec)
            obs = navigate(obs, rec, n)
            obs = pick(obs, rec, args.pitch)
            PROG.event("pick_done", success=rec.get("success"), carry=rec["stages"].get("carry"))
            if rec.get("success") and wants_place:
                obs = deliver_place(obs, rec, n)
                PROG.event("place_done", placed=rec["placed"], place=rec["stages"].get("place"))
            holding = bool(rec.get("success")) and not rec.get("placed")
            rec["fell"] = tilt_deg() > 30
            record_end(rec)
            PROG.event("result", ep=n, success=bool(rec.get("placed") if wants_place else rec.get("success")), time_s=None,
                       fell=rec["fell"])
            n += 1
            tell(("picked the " + tc + " box" if rec.get("success") else "the pick FAILED") +
                 (", placed it" if rec.get("placed") else (" -- holding it: say 'put it on the table'" if holding else ""))
                 + "; waiting for the next task")
            continue
        nav_text, wants_pick, wants_deliver = parse_task(cmd[1])
        if MULTI:  # "... the green table ... the red box": the target; colours not said keep the last target
            low = cmd[1].lower()
            tables_re = "|".join(map(re.escape, F.COLORS3))      # the table names of the scene file
            colors_re = "|".join(map(re.escape, F.BOX_COLORS))   # its box colours
            mt = re.search(rf"\b({tables_re})\s+table\b", low)
            mb = re.search(rf"\b({colors_re})\s+(box|cube|block)\b", low)
            tt = mt.group(1) if mt else (TGT["table"] or ("yellow" if "yellow" in F.COLORS3 else F.COLORS3[0]))
            tc = mb.group(1) if mb else (TGT["box"] or ("red" if "red" in F.BOX_COLORS else F.BOX_COLORS[0]))
            set_target(tt, tc)
            if mt or not re.search(r"\btable\b", nav_text.lower()):
                nav_text = f"Walk to the {tt} table and stop in front of it."
            say(f"target: the {tc} box on the {tt} table")
        rec = {"episode": n, "stages": {}, "task_instruction": cmd[1], "nav_instruction": nav_text}
        if MULTI:
            rec["target"] = {"table": TGT["table"], "box": TGT["box"]}
        if holding:
            if wants_deliver and not wants_pick or re.search(r"\b(deliver|carry)\b|\bback\b|drop zone", cmd[1].lower()):
                say("already holding the box: delivering it")
                obs = deliver(obs, rec, n)
            else:
                say("already holding the box -- say 'bring it back', or 'home' to reset")
            continue
        say(f"task {cmd[1]!r} -> walk: {nav_text!r}" + (", then pick the box" if wants_pick else "")
            + (" and bring it to the drop zone" if wants_deliver and wants_pick else ""))
        p, yaw = base_pose()
        rec["start"] = [round(float(p[0]), 3), round(float(p[1]), 3), round(math.degrees(yaw), 1)]
        record_begin(rec, nav_text)  # with --record, every typed task is an episode of the dataset
        TRACE["sub"] = ""
        PROG.event("episode", ep=n, n=None, start=rec["start"], task=cmd[1], vln=nav_text)
        obs = vln_walk(obs, rec, n, nav_text)
        PROG.event("vln_done", **{k: rec["stages"]["vln"].get(k) for k in ("reason", "arrived", "table_edge_m", "time_s", "path_m")})
        if wants_pick and rec["stages"]["vln"]["reason"] != "interrupted":
            obs = back_off(obs, rec)
            obs = navigate(obs, rec, n)
            PROG.event("dock_done", dock_err_cm=rec["stages"].get("dock_err_base_cm"),
                       time_s=rec["stages"].get("locomotion", {}).get("time_s"), side=rec["stages"].get("side", {}).get("chosen"))
            obs = pick(obs, rec, args.pitch)
            holding = bool(rec.get("success"))
            PROG.event("pick_done", success=rec.get("success"), carry=rec["stages"].get("carry"))
            say(f"pick {'SUCCESS' if holding else 'FAILED'}")
            if holding and wants_deliver:
                obs = deliver(obs, rec, n)
                PROG.event("deliver_done", **{k: rec["stages"].get("deliver", {}).get(k) for k in ("arrived", "time_s", "box_to_grasp_cm")})
        rec["fell"] = tilt_deg() > 30
        record_end(rec)
        PROG.event("result", ep=n, success=bool(rec.get("success")) if wants_pick else bool(rec["stages"]["vln"].get("arrived")),
                   time_s=None, fell=rec["fell"])
        n += 1
        say("waiting for the next task")
        PROG.event("phase", phase="waiting for the next task")


results = []
obs, _ = env.reset(seed=args.seed)
if args.probe_stance:
    obs = begin(obs)
    for v in [float(s) for s in args.probe_stance.split(",")]:
        obs = set_crouch(obs, v, 2.0)
        zs, tilts = [], []
        for _ in range(int(4.0 / dt)):
            obs = step(obs)
            zs.append(float(robot.data.root_pos_w[0, 2]))
            tilts.append(tilt_deg())
        say(f"stance {v:+.2f}: base z {np.mean(zs[-50:]):.3f} m (min {min(zs):.3f}, max {max(zs):.3f}), "
            f"max tilt {max(tilts):.1f} deg, drift {float(robot.data.root_pos_w[0, :2].norm()):.2f}")
elif args.probe and DOOR:  # the robot set down in front of the door (--target: which one; default the middle one)
    _k = args.target if args.target in F.DOORS else sorted(F.DOORS)[len(F.DOORS) // 2]
    paint_doors(_k, np.random.default_rng(args.seed))
    if args.video:
        video_start()
    obs, _rec = door_probe(obs, _k)
    if frames["writer"] is not None:
        video_end(f"{args.video_dir}/door_probe_seed{args.seed}_{_k}_{args.cam}_"
                  f"{'SUCCESS' if _rec['stages'].get('enter', {}).get('inside') else 'FAIL'}.mp4")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump([_rec], f, indent=2)
elif args.probe:
    probe(obs)
elif args.interactive:
    live()
else:
    starts = [tuple(float(v) for v in st.split(",")) for st in args.starts.split(";") if st.strip()]
    rng = np.random.default_rng(args.seed + 7919)
    if args.random_start:
        _rs = [float(v) for v in args.random_start.split(",")]
        rx0, rx1, ry0, ry1 = _rs[:4]
        ryaw = tuple(_rs[4:6]) if len(_rs) >= 6 else (-180.0, 180.0)  # optional heading range (deg)
    vln_given = bool(args.vln)
    rng_deliver = np.random.default_rng(args.seed + 104729)  # its own stream: targets and starts stay as without it
    for ep in range(args.episodes):
        if args.deliver_frac is not None:
            args.deliver = bool(rng_deliver.random() < args.deliver_frac)
        if args.task_variants:
            args.task_instruction = str(rng.choice(TASK_TEXTS_DELIVER if args.deliver else TASK_TEXTS))
        if DOOR:  # which door is painted the target colour: given (a, b, c; a comma list in turn), or random
            doors_ = sorted(F.DOORS)
            if args.target == "auto":
                tk = str(rng.choice(doors_))
            else:
                tk = args.target.split(",")[ep % len(args.target.split(","))]
            DOOR_T["name"] = tk
            args.task_instruction = f"Go to the {DD['target_color']} door, open it and go into the room."
            if vln_given:  # the sentence given (run.py passes the scene's walk), else the scene's
                args.vln = args.vln or DD["walk"]
        elif SHELF:  # which box: given (shelf/colour/table, or just a colour), or a random one per episode
            shelves, tables = list(F.SHELVES), list(F.TABLES)
            if args.target == "auto":  # (a one-shelf, one-table scene draws only the colour, as it always did)
                ts = str(rng.choice(shelves)) if len(shelves) > 1 else shelves[0]
                tc = str(rng.choice([b[0] for b in F.shelf_boxes(ts)]))
                tt = str(rng.choice(tables)) if len(tables) > 1 else tables[0]
            else:  # (a comma list: one after the other, by episode)
                parts = args.target.split(",")[ep % len(args.target.split(","))].split("/")
                tc = parts[1] if len(parts) == 3 else parts[-1]
                ts = parts[0] if len(parts) == 3 else shelves[0]
                tt = parts[2] if len(parts) == 3 else tables[0]
            set_target(ts, tc, tt)
            nav_text, args.task_instruction = shelf_texts(ts, tc, tt, rng)
            if vln_given:
                args.vln = nav_text
        elif MULTI:  # which box: given, or a random one per episode
            if args.target == "auto":
                tt, tc = str(rng.choice(TABLE_COLORS)), str(rng.choice(F.BOX_COLORS))
            else:
                tt, tc = args.target.split("/")
            set_target(tt, tc)
            nav_text, args.task_instruction = target_texts(tt, tc, rng)
            if vln_given:
                args.vln = nav_text
        if args.random_start:
            while True:  # at least 1 m from every table
                sx, sy = rng.uniform(rx0, rx1), rng.uniform(ry0, ry1)
                if all(math.hypot(max(r[0] - sx, 0, sx - r[1]), max(r[2] - sy, 0, sy - r[3])) >= 1.0 for r in TABLE_RECTS):
                    break
            starts = [(sx, sy, rng.uniform(*ryaw))]
        if starts:  # the task resets the robot to its default root state: move that
            sx, sy, syaw = starts[ep % len(starts)]
            with torch.inference_mode():
                st = robot.data.default_root_state
                st[:, 0], st[:, 1] = sx, sy
                st[:, 3:7] = torch.tensor([math.cos(math.radians(syaw) / 2), 0.0, 0.0, math.sin(math.radians(syaw) / 2)],
                                          device=dev)
        if ep > 0 or starts:
            with torch.inference_mode():
                obs, _ = env.reset()
        if DOOR:  # shut and latched again by the reset event; painted for this episode
            for _k in LATCH:
                LATCH[_k] = True
            paint_doors(DOOR_T["name"], rng)
        if ep in VIDEO_EPS:
            video_start()
        if frames["on"] and args.stills:
            render_plate(f"{args.video_dir}/stills_seed{args.seed}_ep{TRACE['ep']}_{args.cam}")
        obs, rec = episode_door(obs, ep, DOOR_T["name"]) if DOOR else episode(obs, ep)
        if frames["writer"] is not None:
            tgt_s = f"_{TGT['box']}-on-{TGT['table']}" if MULTI else \
                (f"_{TGT['box']}-from-{TGT['table']}-to-{PLACE_T['name']}" if SHELF else
                 f"_door-{DOOR_T['name']}" if DOOR else "")
            video_end(f"{args.video_dir}/fetch_seed{args.seed}_ep{ep}{tgt_s}_{args.cam}_"
                      f"{'SUCCESS' if rec['success'] else 'FAIL'}.mp4")
        results.append(rec)
        TRACE["ep"] += 1
        if TRACE["path"] and TRACE["rows"]:
            np.savez(TRACE["path"], sub=np.array([r[1] for r in TRACE["rows"]]),
                     data=np.array([[r[0]] + list(r[2:]) for r in TRACE["rows"]], dtype=np.float32), dt=dt)
    say(f"SUMMARY success {sum(r['success'] for r in results)}/{len(results)}, falls {sum(r['fell'] for r in results)}"
        + (f", placed {sum(bool(r.get('placed')) for r in results)}/{len(results)}" if SHELF else ""))
    PROG.event("summary", success=sum(r["success"] for r in results), n=len(results), falls=sum(r["fell"] for r in results))
    if args.vln:
        say(f"SUMMARY vln arrived {sum(r['stages']['vln']['arrived'] for r in results)}/{len(results)}"
            + (f", delivered {sum(bool(r.get('delivered')) for r in results)}/{sum(r.get('deliver_task', False) for r in results)}"
               if any(r.get("deliver_task") for r in results) else ""))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    say(f"wrote {args.out}")
PROG.event("bye")
if NAV2["link"] is not None:
    NAV2["link"].close()
env.close()
app.close()
