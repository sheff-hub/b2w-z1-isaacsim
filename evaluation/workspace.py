"""Where can the hand actually go? Measure the Z1's workspace, then add the body.

The question this answers: when we sample a world-frame hand target for training,
what range is HONEST? Up to now the ranges (x 0.25-0.60, z 0.15-0.95) were chosen
by hand from a rough reach estimate. If they are too big the policy spends its
time chasing points it physically cannot touch, and the hand error statistic
becomes meaningless; too small and we never find out what the robot can do.

The method has two stages, matching how the robot actually gains reach:

  1. ARM ALONE. The base is pinned level in free air and the six arm joints are
     sampled uniformly inside their real USD limits. The hand (link06) position
     is recorded in the BASE frame. That point cloud is the arm's workspace as
     built, including whatever the mount placement did to it -- not a datasheet
     number.

  2. ARM + BODY. The body can raise and lower that whole cloud. crouch_test.py
     measured the range the trained policy actually uses (0.325 - 0.616 m of base
     height, a 29 cm span), so the reachable world volume is the arm cloud swept
     over that span: world z = base height + (hand z in base frame).

     This is the "offset of how much the arm's base can move" -- taken from a
     measurement of the robot, not from the leg kinematics' theoretical limit,
     because a pose the policy never adopts is not reach we have.

Also reported: how much of the workspace is INSIDE the robot's own body. Those
points are reachable on paper and useless in practice, and they are where the arm
currently passes through the lidar mount. The body boxes are read from the
collision geometry in the USD.

    /isaac-sim/python.sh /workspace/evaluation/workspace.py [--samples 20000] [--out ...]

Approximation worth knowing about: the self-collision check is geometric (arm
link origins against inflated body collider boxes), not a physics query. It is
used to SIZE the safe workspace; training enforces the real thing with contact
sensors.
"""

import argparse
import json
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--stage", default="/workspace/robots_usd/b2w_z1_rl_on_ground.usd")
p.add_argument("--robot", default="/World/b2w_z1")
p.add_argument("--samples", type=int, default=20000)
p.add_argument("--batch", type=int, default=200, help="poses evaluated per physics step")
p.add_argument("--base-height-range", type=float, nargs=2, default=[0.325, 0.616],
               metavar=("LOW", "HIGH"),
               help="base heights the trained policy actually reaches (crouch_test.py, wb9)")
p.add_argument("--margin", type=float, default=0.05,
               help="safety inflation of the body keep-out boxes, m")
p.add_argument("--out", default=None)
args = p.parse_args()


def say(*a):
    print("WS|", *a, flush=True)


t0 = time.time()
from isaacsim import SimulationApp  # noqa: E402

sim_app = SimulationApp({"headless": True})
say(f"SimulationApp up in {time.time() - t0:.1f}s")

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

from isaacsim.core.api import SimulationContext  # noqa: E402
from isaacsim.core.prims import Articulation as ArticulationView, RigidPrim  # noqa: E402
from isaacsim.core.utils.stage import open_stage  # noqa: E402

if not open_stage(args.stage):
    sim_app.close()
    sys.exit("WS| FATAL open_stage failed")

sim = SimulationContext(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()

# views before reset -- creating one afterwards invalidates the tensor view
robot = ArticulationView(args.robot)
base_view = RigidPrim(f"{args.robot}/base_link")
hand_view = RigidPrim(f"{args.robot}/z1/link06")
# link02/link03/link04 carry the colliders that would strike the body
elbow_view = RigidPrim(f"{args.robot}/z1/link0[234]")

sim.reset()
robot.initialize()

names = list(robot.dof_names)
arm_idx = np.array([i for i, n in enumerate(names) if n.startswith("joint")])
leg_idx = np.array([i for i, n in enumerate(names)
                    if n.endswith(("_hip_joint", "_thigh_joint", "_calf_joint"))])
say(f"{len(names)} DOF; arm joints {[names[i] for i in arm_idx]}")
assert len(arm_idx) == 6, f"expected 6 arm joints, found {len(arm_idx)}"

limits = robot.get_dof_limits()
lim = np.asarray(limits[0] if limits.ndim == 3 else limits, dtype=np.float64)
lo, hi = lim[arm_idx, 0], lim[arm_idx, 1]
for k, i in enumerate(arm_idx):
    say(f"  {names[i]:8s} [{lo[k]:+.3f}, {hi[k]:+.3f}] rad "
        f"= [{np.rad2deg(lo[k]):+7.1f}, {np.rad2deg(hi[k]):+7.1f}] deg")

# ---------------------------------------------------------------------------
# Body keep-out boxes, straight from the collision geometry
# ---------------------------------------------------------------------------
xc = UsdGeom.XformCache()
bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"],
                         useExtentsHint=False)
base_prim = stage.GetPrimAtPath(f"{args.robot}/base_link")
T_base_inv = xc.GetLocalToWorldTransform(base_prim).GetInverse()

body_boxes = {}
for link in ("base_link", "lidar_link", "head_Link", "tail_link"):
    prim = stage.GetPrimAtPath(f"{args.robot}/{link}")
    if not prim.IsValid():
        continue
    mins, maxs = [], []
    # The robot's payload composes as INSTANCE PROXIES, and a plain PrimRange
    # stops at the instance boundary -- which is why an earlier version of this
    # script reported "no collision geometry" for a robot that visibly collides.
    for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)):
        if not child.HasAPI(UsdPhysics.CollisionAPI):
            continue
        r = bbox.ComputeWorldBound(child).ComputeAlignedRange()
        if r.IsEmpty():
            continue
        for corner in ((r.GetMin()[0], r.GetMin()[1], r.GetMin()[2]),
                       (r.GetMax()[0], r.GetMax()[1], r.GetMax()[2])):
            q = T_base_inv.Transform(corner)
            mins.append([q[0], q[1], q[2]])
            maxs.append([q[0], q[1], q[2]])
    if mins:
        allp = np.array(mins + maxs)
        body_boxes[link] = (allp.min(axis=0), allp.max(axis=0))

say("body collision boxes in the base frame (x forward, z up):")
for k, (a, b) in body_boxes.items():
    say(f"  {k:11s} x [{a[0]:+.3f},{b[0]:+.3f}]  y [{a[1]:+.3f},{b[1]:+.3f}]  z [{a[2]:+.3f},{b[2]:+.3f}]")
if not body_boxes:
    say("  WARNING: no collision geometry found; the keep-out check is disabled")

# ---------------------------------------------------------------------------
# Sample the arm's workspace with the base pinned level in free air
# ---------------------------------------------------------------------------
spawn_pos, spawn_rot = robot.get_world_poses()
PIN_POS = spawn_pos.copy()
PIN_POS[0][2] = 2.0                       # clear of the floor: legs are irrelevant here
PIN_ROT = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
ZERO_V = np.zeros((1, 6), dtype=np.float32)

# legs held at the nominal stance so nothing swings into the samples
stance = {"_hip_joint": 0.0, "_thigh_joint": 0.8, "_calf_joint": -1.5}
leg_targets = np.array([[next(v for k, v in stance.items() if names[i].endswith(k))
                         for i in leg_idx]], dtype=np.float32)

rng = np.random.default_rng(0)
q_all = rng.uniform(lo, hi, size=(args.samples, 6)).astype(np.float32)
hand_b = np.zeros((args.samples, 3), dtype=np.float64)
elbow_b = np.zeros((args.samples, 3, 3), dtype=np.float64)

say(f"sampling {args.samples} arm poses inside the joint limits...")
t1 = time.time()
for s in range(args.samples):
    robot.set_joint_positions(q_all[s][None, :], joint_indices=arm_idx)
    robot.set_joint_velocities(np.zeros((1, 6), dtype=np.float32), joint_indices=arm_idx)
    robot.set_joint_positions(leg_targets, joint_indices=leg_idx)
    robot.set_world_poses(PIN_POS, PIN_ROT)
    robot.set_velocities(ZERO_V)
    sim.step(render=False)
    bp, _ = base_view.get_world_poses()
    hp, _ = hand_view.get_world_poses()
    ep, _ = elbow_view.get_world_poses()
    hand_b[s] = np.asarray(hp[0]) - np.asarray(bp[0])
    elbow_b[s] = np.asarray(ep) - np.asarray(bp[0])
    if (s + 1) % 2000 == 0:
        say(f"  {s + 1}/{args.samples}  ({time.time() - t1:.0f}s)")

say(f"sampling done in {time.time() - t1:.0f}s")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
SHOULDER = np.array([0.166, 0.0, 0.191])       # arm mount in the base frame
radius = np.linalg.norm(hand_b - SHOULDER, axis=1)
say("")
say("=== 1. ARM ALONE, hand position in the BASE frame ===")
say(f"  x (forward)  [{hand_b[:, 0].min():+.3f}, {hand_b[:, 0].max():+.3f}] m")
say(f"  y (left)     [{hand_b[:, 1].min():+.3f}, {hand_b[:, 1].max():+.3f}] m")
say(f"  z (up)       [{hand_b[:, 2].min():+.3f}, {hand_b[:, 2].max():+.3f}] m")
say(f"  distance from the mount: max {radius.max():.3f} m, mean {radius.mean():.3f} m")


def inside_body(pts, margin):
    """True where a point lies inside any inflated body collider box."""
    bad = np.zeros(len(pts), dtype=bool)
    for a, b in body_boxes.values():
        bad |= np.all((pts >= (a - margin)) & (pts <= (b + margin)), axis=1)
    return bad


clash = inside_body(hand_b, args.margin)
for j in range(3):
    clash |= inside_body(elbow_b[:, j, :], args.margin)
say(f"  poses whose hand or arm links sit inside the body (+{args.margin * 100:.0f} cm): "
    f"{clash.sum()} / {len(clash)} = {100 * clash.mean():.1f}%")

free = hand_b[~clash]
say("")
say("=== 2. COLLISION-FREE hand positions, base frame ===")
say(f"  x [{free[:, 0].min():+.3f}, {free[:, 0].max():+.3f}]  "
    f"y [{free[:, 1].min():+.3f}, {free[:, 1].max():+.3f}]  "
    f"z [{free[:, 2].min():+.3f}, {free[:, 2].max():+.3f}]")

# The targets we sample are IN FRONT of the robot, which is the useful sector and
# the one the command already uses. Restrict to it before quoting ranges.
front = free[(free[:, 0] > 0.20) & (np.abs(free[:, 1]) < 0.35)]
say(f"  front sector (x>0.20, |y|<0.35): {len(front)} poses, "
    f"z [{front[:, 2].min():+.3f}, {front[:, 2].max():+.3f}], x max {front[:, 0].max():.3f}")

# A target is only worth sampling if a decent share of arm poses can serve it,
# otherwise it sits on the workspace boundary where the arm is fully extended
# and singular. Quote the 2nd/98th percentile as the practical envelope.
q_lo, q_hi = np.percentile(front, [2, 98], axis=0)
say(f"  practical envelope (2-98%): x [{q_lo[0]:.3f}, {q_hi[0]:.3f}]  "
    f"y [{q_lo[1]:+.3f}, {q_hi[1]:+.3f}]  z [{q_lo[2]:+.3f}, {q_hi[2]:+.3f}]")

LOW, HIGH = args.base_height_range
say("")
say("=== 3. ARM + BODY: the achievable WORLD volume ===")
say(f"  base height range used by the policy: {LOW:.3f} - {HIGH:.3f} m "
    f"({(HIGH - LOW) * 100:.0f} cm of travel)")
say(f"  arm-alone z span in the base frame: {q_lo[2]:+.3f} .. {q_hi[2]:+.3f} m (2-98%)")
say(f"  so world z spans roughly {LOW + q_lo[2]:+.3f} .. {HIGH + q_hi[2]:+.3f} m")

# ---------------------------------------------------------------------------
# The box has to be JOINTLY feasible, not feasible axis by axis
# ---------------------------------------------------------------------------
# Quoting each axis' 2-98% range separately describes a BOX, but the workspace
# is a shell: the hand reaches 0.75 m forward, and it reaches 1.43 m up, and it
# does NEITHER at the same time. Sampling that box independently -- which is
# exactly what UniformPoseCommand does -- would hand the policy corner targets
# it cannot touch at any body height, and an unreachable target teaches nothing
# except to ignore the arm. That is the failure wb10 already demonstrated.
#
# So: build the real reachable WORLD cloud (every collision-free arm pose, at
# every body height in the measured range) and then ask of each candidate box
# what fraction of it is actually reachable.
VOX = 0.04
n_h = 12
heights = np.linspace(LOW, HIGH, n_h)
world_cloud = np.concatenate([free + np.array([0.0, 0.0, h]) for h in heights], axis=0)
keys = set(map(tuple, np.floor(world_cloud / VOX).astype(np.int64)))
say(f"  reachable world cloud: {len(world_cloud)} points in {len(keys)} "
    f"{VOX * 100:.0f} cm voxels")


def coverage(bx, by, bz, n=40000):
    """Fraction of a uniformly sampled box that is actually reachable."""
    pts = np.column_stack([rng.uniform(*bx, n), rng.uniform(*by, n), rng.uniform(*bz, n)])
    vox = np.floor(pts / VOX).astype(np.int64)
    return float(np.mean([tuple(v) in keys for v in vox]))


# Percentiles of the reachable world cloud, restricted to the front sector the
# command samples from -- the natural distribution, before any box is imposed.
wf = world_cloud[(world_cloud[:, 0] > 0.20) & (np.abs(world_cloud[:, 1]) < 0.30)]
w_lo, w_hi = np.percentile(wf, [2, 98], axis=0)
say(f"  front-sector world cloud z: 2-98% = {w_lo[2]:+.3f} .. {w_hi[2]:+.3f} m, "
    f"full {wf[:, 2].min():+.3f} .. {wf[:, 2].max():+.3f} m")

candidates = [
    ("wb9, hand-chosen", (0.25, 0.60), (-0.30, 0.30), (0.15, 0.95)),
    ("per-axis 2-98% (naive)", (round(q_lo[0], 2), round(q_hi[0], 2)),
     (-0.30, 0.30), (round(LOW + q_lo[2], 2), round(HIGH + q_hi[2], 2))),
    ("world-cloud 2-98%", (0.25, 0.60), (-0.30, 0.30), (round(w_lo[2], 2), round(w_hi[2], 2))),
    ("wider reach, same z", (0.25, 0.70), (-0.30, 0.30), (round(w_lo[2], 2), round(w_hi[2], 2))),
    ("full assisted span", (0.25, 0.65), (-0.30, 0.30), (0.10, 1.20)),
]
say("")
say("  candidate target boxes, by how much of them the robot can actually reach:")
say(f"  {'box':24s} {'x':>14s} {'z':>14s}   coverage")
scored = []
for name, bx, by, bz in candidates:
    c = coverage(bx, by, bz)
    scored.append((c, name, bx, by, bz))
    say(f"  {name:24s} {f'{bx[0]:.2f}-{bx[1]:.2f}':>14s} {f'{bz[0]:.2f}-{bz[1]:.2f}':>14s}   {c * 100:5.1f}%")

# Pick the box that keeps the widest vertical span while staying mostly
# reachable: a low ceiling is safe but wastes the body's contribution, and the
# whole point of this task is the vertical range the crouch buys.
MIN_COVER = 0.75
ok = [s for s in scored if s[0] >= MIN_COVER]
best = max(ok, key=lambda s: s[4][1] - s[4][0]) if ok else max(scored)
say("")
say(f"=== RECOMMENDED WorldPoseCommandCfg ranges ({best[1]}, {best[0] * 100:.0f}% reachable) ===")
say(f"  pos_x=({best[2][0]:.2f}, {best[2][1]:.2f}),")
say(f"  pos_y=({best[3][0]:.2f}, {best[3][1]:.2f}),")
say(f"  pos_z=({best[4][0]:.2f}, {best[4][1]:.2f}),")
say(f"  vs wb9's hand-chosen pos_x=(0.25,0.60) pos_y=(-0.30,0.30) pos_z=(0.15,0.95)")
rec_x, rec_y, rec_z = best[2], best[3], best[4]

if args.out:
    def fl(v):
        return [float(x) for x in v]

    with open(args.out, "w") as f:
        json.dump({
            "samples": int(args.samples),
            "joint_limits": {names[i]: [float(lo[k]), float(hi[k])] for k, i in enumerate(arm_idx)},
            "body_boxes": {k: [fl(a), fl(b)] for k, (a, b) in body_boxes.items()},
            "arm_only_base_frame": {
                "x": [float(hand_b[:, 0].min()), float(hand_b[:, 0].max())],
                "y": [float(hand_b[:, 1].min()), float(hand_b[:, 1].max())],
                "z": [float(hand_b[:, 2].min()), float(hand_b[:, 2].max())],
                "max_radius_from_mount": float(radius.max()),
            },
            "self_collision_fraction": float(clash.mean()),
            "front_envelope_2_98": {"x": [float(q_lo[0]), float(q_hi[0])],
                                    "y": [float(q_lo[1]), float(q_hi[1])],
                                    "z": [float(q_lo[2]), float(q_hi[2])]},
            "base_height_range": [float(LOW), float(HIGH)],
            "world_cloud_front_z_2_98": [float(w_lo[2]), float(w_hi[2])],
            "candidates": [{"name": n, "x": fl(bx), "y": fl(by), "z": fl(bz), "coverage": float(c)}
                           for c, n, bx, by, bz in scored],
            "recommended": {"name": best[1], "coverage": float(best[0]),
                            "pos_x": fl(rec_x), "pos_y": fl(rec_y), "pos_z": fl(rec_z)},
        }, f, indent=2)
    say(f"wrote {args.out}")

say("done")
sim_app.close()
