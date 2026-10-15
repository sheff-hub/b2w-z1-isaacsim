#!/usr/bin/env python
"""Build a control-ready copy of the Unitree B2W USD, leaving the original alone.

Run in any venv that has pxr (the 3DGRUT one does):

    repos/3dgrut/.venv/bin/python tools/build_b2w_rl_asset.py

Produces:
    robots_usd/Unitree_B2W_rl/     copy of Unitree_B2W with the drives corrected
    robots_usd/b2w_in_warehouse.usd  stage: NVIDIA warehouse + the corrected B2W

robots_usd/Unitree_B2W/ is never written to.

WHY THIS EXISTS
===============
The converted asset came out of the URDF importer with drive parameters that
cannot work for a wheeled-legged robot. Two defects, both in
Unitree_B2W/configuration/b2w_description_physics.usd:

1. THE WHEELS HAVE POSITION DRIVES.
   FL/FR/RL/RR_foot_joint are the wheels -- Unitree names the wheel joints
   "foot", inherited from the non-wheeled B2. They are continuous joints
   (limits +/-inf) but carry stiffness=125.1 with targetPosition=0. A position
   drive on a continuous joint is a torsion spring: it resists rotation and
   pulls back toward angle zero. The wheels cannot roll.

   Correct form for a velocity-controlled wheel is stiffness=0 with damping
   only, so the drive torque is proportional to velocity error alone.

2. THE LEG GAINS ARE IMPORTER DEFAULTS, NOT TUNED PD GAINS.
   stiffness 1180-1613 against damping 0.47-0.65 -- a damping ratio around
   4e-4, i.e. essentially undamped, and stiff enough to saturate maxForce on a
   fraction of a degree of error. These are what the importer's natural-
   frequency heuristic emits, not values anyone chose.

UNITS -- READ BEFORE CHANGING THE NUMBERS
=========================================
UsdPhysics expresses angular joint quantities in DEGREES: the limits this asset
carries (calf -161.57..-24.64) are degrees, and targetPosition is degrees. Drive
stiffness and damping follow the same convention, so they are torque per degree
and torque per (degree/second) respectively.

The constants below are therefore written in the units a person actually thinks
in -- N*m/rad -- and converted on the way out.

This unit convention is the one thing here worth distrusting. b2w_teleop.py does
not depend on it: it calls set_gains() at runtime, where Isaac Sim's articulation
API takes radians unambiguously, and the values it uses are authoritative. What
is authored here is what you get when you open the stage in the GUI and press
play without running the script.
"""

import argparse
import math
import shutil
import sys
from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics, Sdf

# --- gains, in the units a person thinks in ---------------------------------
LEG_KP = 800.0    # N*m/rad      -- enough to hold the stance statically
LEG_KD = 20.0     # N*m*s/rad
WHEEL_KD = 20.0   # N*m*s/rad    -- 1 rad/s of error saturates the 20 N*m limit

DEG = math.pi / 180.0  # multiply a per-radian gain by this to get per-degree

WHEEL_JOINTS = ("FL_foot_joint", "FR_foot_joint", "RL_foot_joint", "RR_foot_joint")
LEG_SUFFIXES = ("_hip_joint", "_thigh_joint", "_calf_joint")

WAREHOUSE_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
    "/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd"
)

ROOT = Path(__file__).resolve().parent.parent / "robots_usd"


def say(*a):
    print("BUILD|", *a, flush=True)


def copy_asset(src: Path, dst: Path, force: bool) -> None:
    if dst.exists():
        if not force:
            sys.exit(f"BUILD| {dst} already exists -- pass --force to rebuild it")
        say(f"removing existing {dst.name}/")
        shutil.rmtree(dst)
    say(f"copying {src.name}/ -> {dst.name}/")
    shutil.copytree(src, dst)


def patch_drives(physics_usd: Path) -> None:
    """Rewrite the drive gains in place, in the copy."""
    stage = Usd.Stage.Open(str(physics_usd))
    if stage is None:
        sys.exit(f"BUILD| FATAL could not open {physics_usd}")

    n_wheel = n_leg = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        name = prim.GetName()
        drive = UsdPhysics.DriveAPI(prim, "angular")
        if not drive:
            continue

        if name in WHEEL_JOINTS:
            # Velocity drive: no spring at all, damping only.
            drive.GetStiffnessAttr().Set(0.0)
            drive.GetDampingAttr().Set(WHEEL_KD * DEG)
            drive.GetTargetPositionAttr().Set(0.0)
            drive.GetTargetVelocityAttr().Set(0.0)
            n_wheel += 1
        elif name.endswith(LEG_SUFFIXES):
            drive.GetStiffnessAttr().Set(LEG_KP * DEG)
            drive.GetDampingAttr().Set(LEG_KD * DEG)
            n_leg += 1

    if n_wheel != 4:
        sys.exit(f"BUILD| FATAL patched {n_wheel} wheel joints, expected 4")
    if n_leg != 12:
        sys.exit(f"BUILD| FATAL patched {n_leg} leg joints, expected 12")

    stage.GetRootLayer().Save()
    say(f"patched {n_wheel} wheel drives (stiffness 0, damping {WHEEL_KD} N*m*s/rad)")
    say(f"patched {n_leg} leg drives (kp {LEG_KP} N*m/rad, kd {LEG_KD} N*m*s/rad)")


def build_stage(out: Path, robot_usd: Path, spawn, warehouse: bool) -> None:
    """Author the warehouse + robot stage.

    Mirrors the composition of the existing b2w_z1_in_warehouse.usd: a /World
    Xform holding the environment and the robot, each pulled in as a payload so
    the stage file itself stays tiny and the robot can be reloaded independently.
    """
    if out.exists():
        out.unlink()
    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    if warehouse:
        env = stage.DefinePrim("/World/warehouse")
        env.GetPayloads().AddPayload(WAREHOUSE_URL)
        say("warehouse payload:", WAREHOUSE_URL.rsplit("/", 1)[-1], "(remote)")
    else:
        # Flat ground so the stage opens with no network access.
        #
        # UsdGeomPlane, not a Mesh. A flat quad mesh with CollisionAPI gets
        # approximated as a convex hull, and the hull of four coplanar points
        # has zero thickness: contacts against it are unreliable and the wheels
        # sink straight through. UsdGeomPlane maps to a PhysX plane shape, which
        # is a true infinite half-space.
        plane = UsdGeom.Plane.Define(stage, "/World/ground")
        plane.CreateAxisAttr(UsdGeom.Tokens.z)
        plane.CreateWidthAttr(100.0)
        plane.CreateLengthAttr(100.0)
        UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
        say("ground plane instead of the warehouse (offline-safe)")

    robot = stage.DefinePrim("/World/b2w", "Xform")
    rel = f"./{robot_usd.parent.name}/{robot_usd.name}"
    robot.GetPayloads().AddPayload(rel)
    # The payload arrives carrying its own xformOpOrder (translate, orient,
    # scale), so AddTranslateOp() would collide with the op that is already
    # there. Reuse it when present.
    xf = UsdGeom.Xformable(robot)
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(*spawn))
            break
    else:
        xf.AddTranslateOp().Set(Gf.Vec3d(*spawn))
    say(f"robot payload: {rel} at {tuple(spawn)}")

    light = stage.DefinePrim("/Environment/defaultLight", "DistantLight")
    light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(1000.0)
    light.CreateAttribute("inputs:angle", Sdf.ValueTypeNames.Float).Set(0.53)

    stage.GetRootLayer().Save()
    say("wrote", out)


def verify(robot_usd: Path) -> None:
    """Re-open the patched asset and assert the things Rung 0 depends on."""
    stage = Usd.Stage.Open(str(robot_usd))
    roots = [p.GetPath() for p in stage.Traverse()
             if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
    revolute = [p for p in joints if p.IsA(UsdPhysics.RevoluteJoint)]

    say(f"verify: articulation roots = {len(roots)} {[str(r) for r in roots]}")
    say(f"verify: revolute joints = {len(revolute)} (expect 16)")
    for prim in revolute:
        d = UsdPhysics.DriveAPI(prim, "angular")
        k = d.GetStiffnessAttr().Get()
        c = d.GetDampingAttr().Get()
        kind = "wheel" if prim.GetName() in WHEEL_JOINTS else "leg"
        say(f"  {prim.GetName():16s} {kind:5s} k={k:8.4f} d={c:8.4f} "
            f"maxF={d.GetMaxForceAttr().Get()}")

    assert len(roots) == 1, "expected exactly one articulation root"
    assert len(revolute) == 16, "expected 16 revolute joints"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="rebuild Unitree_B2W_rl/ if it already exists")
    ap.add_argument("--spawn", type=float, nargs=3, default=[0.0, 0.0, 0.65],
                    metavar=("X", "Y", "Z"),
                    help="where to drop the robot (default 0 0 0.65)")
    ap.add_argument("--no-warehouse", action="store_true",
                    help="ground plane instead of the remote warehouse payload")
    args = ap.parse_args()

    src = ROOT / "Unitree_B2W"
    dst = ROOT / "Unitree_B2W_rl"
    if not src.is_dir():
        sys.exit(f"BUILD| FATAL {src} not found")

    copy_asset(src, dst, args.force)
    patch_drives(dst / "configuration" / "b2w_description_physics.usd")
    # Frame-only links' `visuals` reference prims the importer never wrote
    # (red "unresolved prim path" in the Stage tree); drop them in the copy.
    sys.path.insert(0, str(Path(__file__).resolve().parent))  # usd_fixes.py sits next to this file
    from usd_fixes import drop_dangling_refs
    say(f"dropped dangling references: {drop_dangling_refs(dst / 'configuration' / 'b2w_description_base.usd')}")
    verify(dst / "b2w_description.usd")

    name = "b2w_in_warehouse.usd" if not args.no_warehouse else "b2w_on_ground.usd"
    build_stage(ROOT / name, dst / "b2w_description.usd", args.spawn,
                warehouse=not args.no_warehouse)
    say("done")


if __name__ == "__main__":
    main()
