"""Drive the Unitree B2W around a stage on its wheels, from the keyboard.

Rung 0b of the Phase 4 ladder: no RL, no gait. The twelve leg joints are held
in a fixed stance by position drives and the four wheels are given velocity
targets from a differential-drive mapping. The point is to prove the USD loads,
the physics is stable, the collisions work and the command path is wired -- not
to make the robot walk.

Run through run_teleop.sh, which pins the GPU and mounts the asset tree:

    ./run_teleop.sh                     # keyboard, over the livestream
    ./run_teleop.sh --gui               # keyboard, X11 window on this machine
    ./run_teleop.sh --selftest          # headless, scripted, prints a report

Keys:  W/S forward/back   A/D turn   SPACE stop   R respawn   Q quit

WHY THE GAINS ARE SET HERE AS WELL AS IN THE ASSET
==================================================
build_b2w_rl_asset.py authors corrected drive gains into the USD, but has to
express them per DEGREE because that is UsdPhysics' convention for angular
drives. Isaac Sim's articulation API takes them per RADIAN, with no ambiguity,
so this script sets them again at runtime. If the two ever disagree, this is
the one that took effect.
"""

import argparse
import math
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--stage", default="/workspace/robots_usd/b2w_in_warehouse.usd")
p.add_argument("--gui", action="store_true", help="open an X11 window")
p.add_argument("--livestream", action="store_true", help="serve the WebRTC livestream")
p.add_argument("--selftest", action="store_true",
               help="headless: drive a fixed pattern, print a report, exit")
p.add_argument("--stance", type=float, nargs=3, default=[0.0, 0.8, -1.5],
               metavar=("HIP", "THIGH", "CALF"),
               help="leg stance in radians (default 0 0.8 -1.5)")
p.add_argument("--max-speed", type=float, default=1.0, help="m/s at full stick")
p.add_argument("--max-yaw", type=float, default=1.5, help="rad/s at full stick")
p.add_argument("--wheel-sign", type=float, default=1.0,
               help="flip to -1 if W drives the robot backwards")
args = p.parse_args()


def say(*a):
    print("TELEOP|", *a, flush=True)


# --- geometry, measured off the asset ---------------------------------------
# Wheel radius from the FL_foot collision mesh bounding box (x/z half-extent).
# Track width from the joint frames: hip y 0.072 + thigh y 0.11973 per side.
WHEEL_R = 0.1129   # m
TRACK = 0.3845     # m

# Holding a static stance is a harder job for the position drives than tracking
# an RL policy's targets: nothing else carries the weight. With the wheel 0.23 m
# horizontally out from the calf joint, each leg's share of the 82 kg body puts
# ~46 N*m on that joint, so kp=150 N*m/rad sags a third of a radian and the
# robot ends up sitting on the floor. kp=800 keeps the sag under 0.06 rad and
# still leaves headroom under the calf's 320 N*m effort limit.
LEG_KP, LEG_KD = 800.0, 20.0   # N*m/rad, N*m*s/rad
WHEEL_KD = 20.0                # N*m*s/rad, no stiffness

headless = not (args.gui or args.livestream)
if args.selftest:
    headless = True

t0 = time.time()
from isaacsim import SimulationApp  # noqa: E402

sim_app = SimulationApp({"headless": headless, "renderer": "RaytracedLighting"})
say(f"SimulationApp up in {time.time() - t0:.1f}s (headless={headless})")

if args.livestream:
    import omni.kit.app
    mgr = omni.kit.app.get_app().get_extension_manager()
    mgr.set_extension_enabled_immediate("omni.kit.livestream.webrtc", True)
    say("livestream extension enabled")

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402

from isaacsim.core.api import SimulationContext  # noqa: E402
from isaacsim.core.utils.stage import open_stage  # noqa: E402

try:
    from isaacsim.core.prims import Articulation as ArticulationView, RigidPrim
except ImportError:  # older layout
    from omni.isaac.core.articulations import ArticulationView  # type: ignore
    from omni.isaac.core.prims import RigidPrimView as RigidPrim  # type: ignore

# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------
say("opening", args.stage)
if not open_stage(args.stage):
    sim_app.close()
    sys.exit("TELEOP| FATAL open_stage failed")

sim = SimulationContext(stage_units_in_meters=1.0)

# The warehouse asset ships no ground collider at the origin in every variant,
# and the offline stage authors its own plane, so only add one if nothing is
# there. Cheap insurance against the robot falling through the world.
stage = omni.usd.get_context().get_stage()
if not any(stage.GetPrimAtPath(q).IsValid()
           for q in ("/World/warehouse", "/World/ground")):
    from isaacsim.core.api.objects import GroundPlane
    GroundPlane(prim_path="/World/_fallback_ground", z_position=0.0)
    say("added a fallback ground plane")

# Every view has to exist BEFORE the first reset. Creating one afterwards makes
# PhysX re-parse the collision shapes, which deletes and rebuilds prims that the
# already-live tensor view is holding -- it invalidates itself and every
# subsequent read throws "Simulation view object is invalidated".
robot = ArticulationView("/World/b2w")
base_view = RigidPrim("/World/b2w/base_link")
feet_view = RigidPrim("/World/b2w/.*_foot")

sim.reset()
robot.initialize()
say(f"link views: base_link + {len(feet_view.prim_paths)} wheels "
    f"{[q.rsplit('/', 1)[-1] for q in feet_view.prim_paths]}")

names = list(robot.dof_names)
say(f"DOF count = {len(names)}")
for i, n in enumerate(names):
    say(f"  [{i:2d}] {n}")

# ---------------------------------------------------------------------------
# Actuator groups -- BY NAME, never by index order
# ---------------------------------------------------------------------------
# Unitree calls the wheel joints "*_foot_joint", inherited from the non-wheeled
# B2. Matching on "foot" therefore selects WHEELS, not feet. Getting this
# backwards is the single most likely way to break everything downstream, so it
# is asserted rather than assumed.
wheel_idx = np.array([i for i, n in enumerate(names) if n.endswith("_foot_joint")])
leg_idx = np.array([i for i, n in enumerate(names)
                    if n.endswith(("_hip_joint", "_thigh_joint", "_calf_joint"))])

assert len(wheel_idx) == 4, f"expected 4 wheel joints, found {len(wheel_idx)}"
assert len(leg_idx) == 12, f"expected 12 leg joints, found {len(leg_idx)}"
say(f"wheels {[names[i] for i in wheel_idx]}")
say(f"legs   {len(leg_idx)} joints")

# Which wheel is on which side, so the differential drive does not rely on
# the DOF ordering the importer happened to produce.
assert sum(names[i].startswith(("FL", "RL")) for i in wheel_idx) == 2
assert sum(names[i].startswith(("FR", "RR")) for i in wheel_idx) == 2

# ---------------------------------------------------------------------------
# Gains: legs are position-controlled, wheels are velocity-controlled
# ---------------------------------------------------------------------------
kps = np.zeros(len(names), dtype=np.float32)
kds = np.zeros(len(names), dtype=np.float32)
kps[leg_idx] = LEG_KP
kds[leg_idx] = LEG_KD
kps[wheel_idx] = 0.0
kds[wheel_idx] = WHEEL_KD
robot.set_gains(kps=kps[None, :], kds=kds[None, :])
say(f"gains set: legs kp={LEG_KP} kd={LEG_KD}, wheels kp=0 kd={WHEEL_KD}")

hip, thigh, calf = args.stance
stance = np.zeros(len(names), dtype=np.float32)
for i, n in enumerate(names):
    if n.endswith("_hip_joint"):
        stance[i] = hip
    elif n.endswith("_thigh_joint"):
        stance[i] = thigh
    elif n.endswith("_calf_joint"):
        # Front and rear calves mirror on Unitree quadrupeds only in sign
        # convention for the hip; the calf sign is the same on all four legs.
        stance[i] = calf
say(f"stance: hip={hip} thigh={thigh} calf={calf} rad")

spawn_pos, spawn_rot = robot.get_world_poses()
ZEROS = np.zeros((1, len(names)), dtype=np.float32)
ZEROS6 = np.zeros((1, 6), dtype=np.float32)   # root linear + angular velocity

CAL_POSE = spawn_pos.copy()
CAL_POSE[0][2] = 2.0   # free air, clear of any floor

# ---------------------------------------------------------------------------
# Command state
# ---------------------------------------------------------------------------
cmd = {"v": 0.0, "w": 0.0}
quit_flag = {"q": False}


# Isaac orders the DOFs by joint type, not per leg: all four hips, then all
# four thighs, then calves, then wheels. Both target arrays below are therefore
# built in *joint_indices order*, never by slicing a 16-long vector -- and when
# joint_indices is passed, the value array must be as long as the index list.
LEFT_MASK = np.array([names[i].startswith(("FL", "RL")) for i in wheel_idx])
LEG_TARGET = stance[leg_idx][None, :].astype(np.float32)


def wheel_targets(v, w):
    """Differential drive: body twist -> angular velocity for each wheel."""
    vl = (v - w * TRACK / 2.0) / WHEEL_R
    vr = (v + w * TRACK / 2.0) / WHEEL_R
    return (args.wheel_sign * np.where(LEFT_MASK, vl, vr)).astype(np.float32)[None, :]


# ---------------------------------------------------------------------------
# Keyboard
# ---------------------------------------------------------------------------
if not args.selftest:
    import carb.input
    import omni.appwindow

    KEY = carb.input.KeyboardInput
    held = set()

    def on_key(event, *_):
        et = event.type
        if et == carb.input.KeyboardEventType.KEY_PRESS:
            held.add(event.input)
            if event.input == KEY.Q:
                quit_flag["q"] = True
            elif event.input == KEY.R:
                respawn()
            elif event.input == KEY.SPACE:
                held.clear()
        elif et == carb.input.KeyboardEventType.KEY_RELEASE:
            held.discard(event.input)
        return True

    app_window = omni.appwindow.get_default_app_window()
    kb_sub = carb.input.acquire_input_interface().subscribe_to_keyboard_events(
        app_window.get_keyboard(), on_key
    )
    say("keyboard ready -- W/S drive, A/D turn, SPACE stop, R respawn, Q quit")

    def poll():
        v = args.max_speed * ((KEY.W in held) - (KEY.S in held))
        w = args.max_yaw * ((KEY.A in held) - (KEY.D in held))
        cmd["v"], cmd["w"] = v, w

# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------
dt = sim.get_physics_dt()
say(f"physics dt = {dt}")


def rpy():
    """Roll and pitch of the base, degrees. Isaac quaternions are (w, x, y, z)."""
    _, quat = robot.get_world_poses()
    w, x, y, z = (float(v) for v in quat[0])
    roll = math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))))
    return roll, pitch

def hold_targets():
    robot.set_joint_position_targets(LEG_TARGET, joint_indices=leg_idx)


def link_z():
    """World z of base_link and of the four wheel centres, straight from PhysX.

    Read through a RigidPrim view rather than off the USD transforms: with
    render=False nothing forces a USD write-back, so the stage transforms can
    be stale while PhysX has long since moved on.
    """
    bz = base_view.get_world_poses()[0][0][2]
    fz = feet_view.get_world_poses()[0][:, 2]
    return float(bz), fz


def calibrate_stance():
    """Measure how far the wheels hang below the body in the commanded stance.

    The leg extension for a given (thigh, calf) pair is trigonometry, but the
    sign conventions that come out of a URDF conversion are not worth deriving
    on paper -- an error there is exactly what put the robot on its belly the
    first time. So: pin the base in the air, let the position drives converge,
    and measure it.
    """
    # Pin it well clear of the floor. Pinning at the spawn pose is not good
    # enough: sim.reset() steps physics, so by the time the pose is read the
    # robot has already dropped some variable distance and may be resting on
    # the ground. Converging the legs against the floor stores energy in them,
    # and releasing that pitched the robot onto its rear wheels.
    for _ in range(int(2.0 / dt)):
        robot.set_world_poses(CAL_POSE, spawn_rot)
        # Pinning the pose without also pinning the velocity is not pinning.
        # PhysX keeps integrating gravity into the root every step, so after two
        # seconds the base carries ~20 m/s downward even though it never appears
        # to move -- and the moment it is released it drives itself into the
        # floor hard enough to flip the robot onto its back.
        robot.set_velocities(ZEROS6)
        robot.set_joint_velocities(ZEROS)
        hold_targets()
        sim.step(render=False)

    bz, fz = link_z()
    drop = bz - float(np.mean(fz))
    say(f"stance calibration (base pinned in free air at z={bz:.3f}):")
    say(f"  wheel spread {float(np.max(fz) - np.min(fz)):.4f} m "
        f"(should be ~0 -- a symmetric stance hangs level)")
    say(f"  wheel centres at z = {np.round(fz, 4).tolist()}")
    say(f"  base sits {drop:+.4f} m above the wheel centres")
    if drop <= 0:
        say("  WARNING: legs extend UPWARD -- the stance signs are inverted")
    return drop


# Calibrate in every mode: the stage's authored spawn height is a guess, and
# dropping the robot from the wrong one is how it ends up on its belly.
drop = calibrate_stance()
RIDE_Z = drop + WHEEL_R
say(f"ride height = {drop:.4f} + wheel r {WHEEL_R} = {RIDE_Z:.4f} m")

SPAWN_AT = spawn_pos.copy()
SPAWN_AT[0][2] = RIDE_Z + 0.005   # 5 mm, so contact is made by falling


def respawn():
    robot.set_world_poses(SPAWN_AT, spawn_rot)
    robot.set_velocities(ZEROS6)
    robot.set_joint_positions(stance[None, :])
    robot.set_joint_velocities(ZEROS)


respawn()

if args.selftest:
    ride_height = RIDE_Z
    SETTLE, DRIVE = 3.0, 3.0
    steps_settle, steps_drive = int(SETTLE / dt), int(DRIVE / dt)

    for _ in range(steps_settle):
        robot.set_joint_position_targets(LEG_TARGET, joint_indices=leg_idx)
        robot.set_joint_velocity_targets(
            wheel_targets(0.0, 0.0), joint_indices=wheel_idx)
        sim.step(render=False)

    pos0, _ = robot.get_world_poses()
    jp = robot.get_joint_positions()[0]
    say(f"settled base height = {pos0[0][2]:.4f} m (expected ride height {ride_height:.4f})")
    say(f"  wheel centres now at z = {np.round(link_z()[1], 4).tolist()} (wheel r {WHEEL_R})")
    r, pi_ = rpy()
    say(f"settled attitude: roll {r:+.1f} deg  pitch {pi_:+.1f} deg "
        f"({'level' if abs(r) < 5 and abs(pi_) < 5 else 'TILTED'})")
    say("settled leg angles:")
    for i in leg_idx:
        say(f"  {names[i]:16s} target {stance[i]:+.3f}  actual {jp[i]:+.3f} rad")

    for _ in range(steps_drive):
        robot.set_joint_position_targets(LEG_TARGET, joint_indices=leg_idx)
        robot.set_joint_velocity_targets(
            wheel_targets(args.max_speed, 0.0), joint_indices=wheel_idx)
        sim.step(render=False)

    pos1, _ = robot.get_world_poses()
    d = pos1[0] - pos0[0]
    wv = robot.get_joint_velocities()[0][wheel_idx]
    say(f"after {DRIVE}s at v={args.max_speed} m/s command:")
    say(f"  displacement  dx={d[0]:+.3f}  dy={d[1]:+.3f}  dz={d[2]:+.3f} m")
    say(f"  expected |dx| ~ {args.max_speed * DRIVE:.2f} m")
    say(f"  wheel speeds  {np.round(wv, 2).tolist()} rad/s "
        f"(target {args.wheel_sign * args.max_speed / WHEEL_R:.2f})")
    v_now = float(robot.get_velocities()[0][0])
    r1, p1 = rpy()
    say(f"  final height  {pos1[0][2]:.4f} m   roll {r1:+.1f} pitch {p1:+.1f} deg")
    say(f"  steady speed  {v_now:+.3f} m/s against a {args.max_speed} m/s command "
        f"({100 * v_now / args.max_speed:.0f}%)")
    say(f"  mean speed over the run {d[0] / DRIVE:+.3f} m/s "
        f"(includes accelerating from rest)")
    if d[0] < -0.05:
        say("  NOTE: moved in -X. Re-run with --wheel-sign -1 to make W go forward.")
    say("selftest done")
else:
    while sim_app.is_running() and not quit_flag["q"]:
        poll()
        robot.set_joint_position_targets(LEG_TARGET, joint_indices=leg_idx)
        robot.set_joint_velocity_targets(
            wheel_targets(cmd["v"], cmd["w"]), joint_indices=wheel_idx)
        sim.step(render=True)

sim_app.close()
say("closed")
