#!/usr/bin/env python
"""Build a physics-correct B2W + Z1 from the hand-placed b2w_z1.usd.

Run in any venv that has pxr (the 3DGRUT one does):

    repos/3dgrut/.venv/bin/python tools/build_b2w_z1_rl_asset.py

Produces, next to the input:
    robots_usd/b2w_z1_rl.usd             the corrected robot
    robots_usd/b2w_z1_rl_on_ground.usd   ground plane + robot, for physics checks

robots_usd/b2w_z1.usd is read, never written. Every fix below is authored as an
override in b2w_z1_rl.usd itself, so Unitree_B2W*/ and Unitree_Z1/ are untouched
too.

WHAT IS WRONG WITH b2w_z1.usd, AND THE FIX FOR EACH
===================================================
1. THE ARM WAS MOVED VISUALLY, NOT PHYSICALLY.
   On 19 Sep 2026 the z1 prim was translated to sit ~0.17 m further forward on
   the body, but the FixedJoint that bolts the arm to base_link still carries
   its original localPos0 (0, 0, 0.0903). In simulation the joint wins: the arm
   is dragged back to the old spot, or jolts trying to. Fix: set localPos0 to
   wherever the z1 prim actually sits relative to base_link, read from the
   stage rather than hard-coded, so moving the arm and re-running this tool
   keeps the two in agreement.

2. TWO ARTICULATION ROOTS.
   Both base_link and z1/world carry PhysicsArticulationRootAPI (the latter
   inherited from the standalone Z1, where it was the fixed base). Isaac Lab
   needs exactly one per robot. Fix: remove it from z1/world.

3. THE ARM MOTORS ARE NEARLY LIMP.
   Z1 drive stiffness is 0.9-47 with damping ~1e-3: URDF-importer defaults,
   the same defect the B2W legs had. Fix: explicit PD gains (below).

4. THE WHEELS STILL HAVE THE ORIGINAL TORSION-SPRING DRIVE.
   b2w_z1.usd payloads ./Unitree_B2W/b2w_description.usd -- the untouched
   original, whose wheel joints carry stiffness 125 and cannot roll. Fix:
   payload ./Unitree_B2W_rl/ instead (built by build_b2w_rl_asset.py).

5. A MASSLESS RIGID BODY.
   z1/world is a rigid body with mass 0 and no colliders, so PhysX has nothing
   to compute a mass from and substitutes a default. Fix: give it an explicit
   negligible mass and inertia.

UNITS
=====
As in build_b2w_rl_asset.py: UsdPhysics angular drive gains are per DEGREE, so
the per-radian constants below are converted on the way out. Anything that sets
gains at runtime (an Isaac Lab ActuatorCfg, set_gains()) supersedes these; they
are what you get opening the stage in the GUI.
"""

import math
import shutil
import sys
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

ROOT = Path(__file__).resolve().parent.parent / "robots_usd"
SRC = ROOT / "b2w_z1.usd"
DST = ROOT / "b2w_z1_rl.usd"
GROUND = ROOT / "b2w_z1_rl_on_ground.usd"

BASE = "/Unitree_B2W/base_link"
Z1 = "/Unitree_B2W/z1"
Z1_WORLD = "/Unitree_B2W/z1/world"
MOUNT_JOINT = "/Unitree_B2W/z1/world/FixedJoint"
PATCHED_B2W = "./Unitree_B2W_rl/b2w_description.usd"

# Arm PD gains, per radian. Enough to hold the arm against gravity in a stowed
# or moderately extended pose (the distal links total ~3.3 kg, roughly 10 N*m
# about the shoulder when horizontal) while staying well inside the Z1's 30-60
# N*m effort limits, which are left as authored.
ARM_KP = 100.0   # N*m/rad
ARM_KD = 4.0     # N*m*s/rad
ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]
DEG = math.pi / 180.0

TINY_MASS = 1e-3               # kg, for the virtual z1/world link
TINY_INERTIA = (1e-6,) * 3     # kg*m^2


def say(*a):
    print("BUILD|", *a, flush=True)


def relative_translation(stage, child, parent):
    """Translation of `child` expressed in `parent`'s frame."""
    xc = UsdGeom.XformCache()
    rel = xc.GetLocalToWorldTransform(stage.GetPrimAtPath(child)) * \
        xc.GetLocalToWorldTransform(stage.GetPrimAtPath(parent)).GetInverse()
    return Gf.Vec3f(rel.ExtractTranslation())


def main():
    if not SRC.exists():
        sys.exit(f"BUILD| FATAL {SRC} not found")
    if not (ROOT / PATCHED_B2W).exists():
        sys.exit(f"BUILD| FATAL {PATCHED_B2W} missing -- run build_b2w_rl_asset.py first")

    shutil.copyfile(SRC, DST)
    say(f"copied {SRC.name} -> {DST.name}")

    stage = Usd.Stage.Open(str(DST))
    layer = stage.GetRootLayer()

    # 4. Wheels: point the body at the patched B2W. Done on the Sdf spec so the
    #    list op is replaced cleanly rather than appended to.
    spec = layer.GetPrimAtPath("/Unitree_B2W")
    old = list(spec.payloadList.prependedItems)
    spec.payloadList.prependedItems = [Sdf.Payload(PATCHED_B2W)]
    say(f"[4] body payload {[p.assetPath for p in old]} -> {PATCHED_B2W}")

    # Re-open so the new payload is composed before anything is measured.
    layer.Save()
    stage = Usd.Stage.Open(str(DST))

    # 1. Mount joint follows the arm's actual placement.
    mount = relative_translation(stage, Z1_WORLD, BASE)
    joint = UsdPhysics.Joint(stage.GetPrimAtPath(MOUNT_JOINT))
    before = joint.GetLocalPos0Attr().Get()
    joint.GetLocalPos0Attr().Set(mount)
    joint.GetLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
    say(f"[1] mount localPos0 {tuple(round(v, 4) for v in before)} -> "
        f"{tuple(round(v, 4) for v in mount)} (where the arm actually sits)")

    # 2. One articulation root.
    world = stage.GetPrimAtPath(Z1_WORLD)
    world.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    say("[2] removed ArticulationRootAPI from z1/world")

    # 5. Explicit negligible mass for the virtual link.
    mass = UsdPhysics.MassAPI(world)
    mass.GetMassAttr().Set(TINY_MASS)
    mass.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*TINY_INERTIA))
    say(f"[5] z1/world mass 0 -> {TINY_MASS} kg")

    # 3. Arm PD gains.
    for name in ARM_JOINTS:
        prim = stage.GetPrimAtPath(f"{Z1}/joints/{name}")
        drive = UsdPhysics.DriveAPI(prim, "angular")
        drive.GetStiffnessAttr().Set(ARM_KP * DEG)
        drive.GetDampingAttr().Set(ARM_KD * DEG)
    say(f"[3] arm drives kp {ARM_KP} N*m/rad, kd {ARM_KD} N*m*s/rad on {len(ARM_JOINTS)} joints")

    stage.GetRootLayer().Save()
    verify(DST)
    build_ground_stage()
    say("done")


def verify(path):
    stage = Usd.Stage.Open(str(path))
    roots = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    moving = [p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
    wheel = stage.GetPrimAtPath("/Unitree_B2W/joints/FL_foot_joint")
    mount = UsdPhysics.Joint(stage.GetPrimAtPath(MOUNT_JOINT)).GetLocalPos0Attr().Get()
    arm_now = relative_translation(stage, Z1_WORLD, BASE)

    say(f"verify: articulation roots {roots}")
    say(f"verify: revolute joints {len(moving)} (expect 22: 12 legs + 4 wheels + 6 arm)")
    say(f"verify: wheel stiffness "
        f"{UsdPhysics.DriveAPI(wheel, 'angular').GetStiffnessAttr().Get()} (expect 0)")
    say(f"verify: mount {tuple(round(v, 4) for v in mount)} vs arm placement "
        f"{tuple(round(v, 4) for v in arm_now)}")

    assert roots == [BASE], "expected base_link as the only articulation root"
    assert len(moving) == 22, "expected 22 revolute joints"
    assert UsdPhysics.DriveAPI(wheel, "angular").GetStiffnessAttr().Get() == 0.0, "wheels not patched"
    assert (Gf.Vec3f(mount) - arm_now).GetLength() < 1e-4, "mount joint does not match placement"


def build_ground_stage():
    """Ground plane + robot, spawned high enough to settle onto its wheels."""
    if GROUND.exists():
        GROUND.unlink()
    stage = Usd.Stage.CreateNew(str(GROUND))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    # UsdGeomPlane, not a flat Mesh: a coplanar-quad convex hull has zero
    # thickness and the wheels sink through it (found during Rung 0b).
    plane = UsdGeom.Plane.Define(stage, "/World/ground")
    plane.CreateAxisAttr(UsdGeom.Tokens.z)
    plane.CreateWidthAttr(100.0)
    plane.CreateLengthAttr(100.0)
    UsdPhysics.CollisionAPI.Apply(plane.GetPrim())

    robot = stage.DefinePrim("/World/b2w_z1", "Xform")
    robot.GetPayloads().AddPayload(f"./{DST.name}")
    xf = UsdGeom.Xformable(robot)
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(0, 0, 0.65))
            break
    else:
        xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.65))

    light = stage.DefinePrim("/Environment/defaultLight", "DistantLight")
    light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(1000.0)
    stage.GetRootLayer().Save()
    say(f"wrote {GROUND.name}")


if __name__ == "__main__":
    main()
