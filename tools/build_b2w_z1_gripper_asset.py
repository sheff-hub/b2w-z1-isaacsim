#!/usr/bin/env python
"""Build the B2W + Z1-with-gripper robot from the hand-placed b2w_z1.usd.

Run in any venv that has pxr (the 3DGRUT one does):

    repos/3dgrut/.venv/bin/python tools/build_b2w_z1_gripper_asset.py

Produces, next to the input:
    robots_usd/b2w_z1_gripper.usd             the robot, 23 DOFs, three cameras
    robots_usd/b2w_z1_gripper_on_ground.usd   ground plane + robot, for physics checks

Prerequisite: robots_usd/Unitree_Z1_gripper/z1_gripper.usd, made by
tools/make_z1_gripper_urdf.py + tools/convert_z1_gripper.py.

robots_usd/b2w_z1.usd is read, never written. Its arm placement, front/rear
cameras, lidar and ROS 2 publisher graphs carry over unchanged; everything below
is authored as overrides in b2w_z1_gripper.usd.

WHAT CHANGES RELATIVE TO b2w_z1.usd
===================================
A. The arm payload is the gripper arm. Kinematics and link masses of the
   gripper arm's joint1..link06 are identical to Unitree_Z1/z1.usd (checked
   27 Sep 2026); it adds gripperStator (0.526 kg), gripperMover (0.276 kg) and
   jointGripper (0 = closed, -pi/2 = open).

B. Its root_joint is deactivated. The URDF importer (fix_base) wrote a fixed
   joint from the world to z1/world and put the ArticulationRootAPI on it. Left
   active on the dog it bolts the arm, and so the robot, to a point in space.

C. Arm joint limits, drive type and max joint velocity are copied from
   Unitree_Z1/z1.usd. The gripper URDF is a newer Unitree release with wider
   limits (joint3 down to -274 deg, which folds link03 through link02), and the
   newer importer writes ACCELERATION drives where the original has FORCE
   drives. An acceleration drive scales its gains by the joint's inertia, so the
   same kp barely holds the light wrist: joint4 sagged 50.7 deg under the
   gripper's weight (0.6 deg with force drives). The gripper drive is made a
   force drive with the same max velocity as the arm.

D. The end-effector camera is moved. It sat 4.6 cm along link06, which is now
   inside the gripper housing (the stator starts at 4.4 cm). It is mounted BESIDE
   the fingers (+y, 4.4 cm clear of the housing), turned ~30 deg inward: the
   fingertip gap is ~20 deg off the image centre and the left edge of the image
   looks straight ahead. Not above:
   the moving jaw opens upward and would swing across the view. Not below
   (tried): the fixed jaw then fills the frame and hides the gap. The prim path
   is unchanged, so the ROS 2 camera graph still finds it.

E. The arm is moved 0.11 m towards the rear. In b2w_z1.usd's placement the
   stowed gripper (all arm joints 0) reaches x = 0.354 m in the body frame and
   hangs down to z = 0.211, while the lidar cage's roof starts at x = 0.270 and
   rises to z = 0.243: the gripper sat ~3 cm deep in the cage roof. Measured
   gripper-to-cage clearance by rearward shift (5 mm voxels, 27 Sep 2026):
   0.09 m touching, 0.10 m 1.5 cm, 0.11 m 2.5 cm. Every other arm link stays
   >= 6 cm from the body. The body top is flat (z 0.085) under the new arm base,
   so its height is unchanged. The mount joint follows via fix 1 below.

F. Composition errors cleaned up (they show as red prims in the Stage tree):
   - b2w_z1.usd carries a COPY of the B2W's Physics/Sensor/Robot variant sets
     in its own top layer, with payload paths (./b2w_description/configuration/
     ...) that only resolve from inside the B2W folder. The real variant sets
     come with the body payload; the copies are deleted (selections are kept).
   - The URDF importer gave frame-only links (head_Link, lidar_link, camera
     links, imu_link, z1/world, ...) a `visuals` reference to a prim it never
     wrote. That is fixed where it is authored (a stronger layer cannot delete
     a reference inside a payload), by tools/usd_fixes.py, called from
     build_b2w_rl_asset.py and tools/convert_z1_gripper.py. The verify step
     below fails the build if any composition error is left.

G. Self-collision ON, so the arm cannot pass through the robot (it was off:
   the arm went through the body "like a ghost"). With it on, PhysX only skips
   pairs joined BY a joint, and measured (probe_self_collision.py):
   - head_Link <-> lidar_link overlap by design: 150 kN from the first step
     -> filtered (both bolted to the body; they still collide with the arm)
   - link00 <-> base_link: the arm base is bolted to the body (via z1/world, so
     two joints apart) -> filtered; it can never move relative to it
   The body's own colliders cover much less than its shell: a box x +-0.25,
   top z 0.075, plus 2-4 cm plates at head and tail -- nothing between x 0.25
   and 0.39 at either end, while the shell's top is at z 0.085. A top-cover box
   on base_link (x +-0.40, y +-0.14, z 0.055-0.085, invisible) closes that from
   above, where the arm comes from. It overlaps only links jointed to base_link
   (lidar, head, tail, hips), which PhysX already skips.
   Isaac Lab's ArticulationCfg (enabled_self_collisions) overrides this flag
   at spawn; this is what the GUI and plain Isaac Sim scripts get.

H. The lidar mast's collider is replaced by a box matching the visible cage.
   The importer's capsule (r 0.076) reached z = 0.315 in the body frame, 7 cm
   above the cage's roof (0.243): with self-collision on, the arm reaching over
   the mast to a table in front of the robot leaned on that invisible cap
   (100-160 N) and was held up to 3.8 cm short of the box. New collider: x
   0.266-0.418, y +-0.076 (the capsule's footprint), z 0.085-0.245 (body top to
   the cage's roof; lower, its corners hit the front hip motors).

I. (--wrist d405) An Intel RealSense D405 on a 3D-printed ring clamp round the middle of link06, with an
   arm out to the gripper's -y side and a cradle round the camera, at the pose the user set by hand
   (b2w_z1_gripper_reference.usd); the end-effector camera moves onto the D405's left imager and gets its
   87 deg field of view (16:9). Design, mass update and offline checks: wrist_d405.py.
   The default since 13 Oct 2026 (--wrist none: the bare camera of section D).

J. (--body_cams d455) The front and rear head cameras get a RealSense D455's colour camera (OmniVision
   OV9782): the lens of Isaac Sim's own rsd455.usd (robots_usd/RealSense/, downloaded from the Isaac 5.1
   assets), focal 1.93 mm on 3.896 x 2.453 mm = 90 x 65 deg, 16:10. Positions are unchanged. (The fetch
   pipeline renders its head camera from vln_nav.add_cameras with the same lens.)

The physics fixes of build_b2w_z1_rl_asset.py, for the same reasons:
1. mount FixedJoint localPos0 follows where the arm actually sits
2. no ArticulationRootAPI on z1/world (one root: base_link)
3. arm PD gains, plus gripper gains
4. body payload repointed to the patched Unitree_B2W_rl (rolling wheels)
5. massless z1/world rigid body given 1 g
"""

import argparse
import math
import shutil
import sys
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_b2w_z1_rl_asset import (  # noqa: E402  (same fixes, same constants)
    ARM_JOINTS, ARM_KD, ARM_KP, BASE, DEG, MOUNT_JOINT, PATCHED_B2W, TINY_INERTIA,
    TINY_MASS, Z1, Z1_WORLD, relative_translation,
)

ROOT = Path(__file__).resolve().parent.parent / "robots_usd"
SRC = ROOT / "b2w_z1.usd"
DST = ROOT / "b2w_z1_gripper.usd"
GROUND = ROOT / "b2w_z1_gripper_on_ground.usd"
GRIPPER_ARM = "./Unitree_Z1_gripper/z1_gripper.usd"
ORIGINAL_ARM = ROOT / "Unitree_Z1" / "z1.usd"

ROOT_JOINT = f"{Z1}/root_joint"
GRIPPER_JOINT = "jointGripper"
EE_CAMERA = f"{Z1}/link06/end_effector_camera"
BODY_CAMERAS = ("/Unitree_B2W/f_oc_link/front_camera", "/Unitree_B2W/r_oc_ink/rear_camera")
D455 = ROOT / "RealSense" / "rsd455.usd"
D455_COLOR = "/Root/RSD455/Camera_OmniVision_OV9782_Color"

# Gripper PD gains, per radian. Stiff enough that closing on an object
# saturates at the URDF effort limit (30 N*m, left as authored) within ~0.3 rad
# of contact, so the grip force does not depend much on the object's width.
GRIP_KP = 100.0  # N*m/rad
GRIP_KD = 2.0    # N*m*s/rad

# End-effector camera, in the link06 frame (x = along the wrist, z = the side
# the moving jaw opens towards). The stator is fixed 0.051 m along link06, is
# 0.041 m half-wide in y, and its fingertips are at x = 0.16-0.20. Camera FOV
# (unchanged from b2w_z1.usd) is 60 x 45 deg.
EE_CAM_POS = Gf.Vec3d(0.10, 0.085, 0.0)
EE_CAM_LOOK_AT = Gf.Vec3d(0.25, 0.0, 0.0)  # fingertip gap ~20 deg off centre, left edge straight ahead

# E. Arm placement: shift towards the rear (body -x), and the clearance the
# build must keep between the stowed gripper and the lidar cage (front face at
# body x = 0.270 m, measured from the base_link mesh above z = 0.12).
ARM_SHIFT_BACK = 0.11  # m
LIDAR_CAGE_FRONT_X = 0.266  # m, body frame (= LIDAR_BOX_MIN.x, the collider)
MIN_CAGE_CLEARANCE = 0.02  # m

# G. Self-collision.
SELF_COLLISION_ATTR = "physxArticulation:enabledSelfCollisions"
FILTERED_PAIRS = [  # (prim, other): overlapping by design, both fixed to the body
    ("/Unitree_B2W/head_Link", "/Unitree_B2W/lidar_link"),
    (f"{Z1}/link00", BASE),
]
TOP_COVER = f"{BASE}/top_cover_collider"
TOP_COVER_MIN = Gf.Vec3d(-0.40, -0.14, 0.055)  # body frame, m
TOP_COVER_MAX = Gf.Vec3d(0.40, 0.14, 0.085)    # = the visible top surface

# H. Lidar mast collider: a box with the visible cage's extent (body frame, m).
LIDAR = "/Unitree_B2W/lidar_link"
# z starts at the body's top (0.085), not the capsule's 0.003: a box's square
# bottom corners reached into the front hip motors (the capsule's round bottom
# curved away), and FR_hip hit it at 20-80 N whenever the robot shifted its
# weight. Below 0.085 the mast is inside the body, covered by base_link.
LIDAR_BOX_MIN = Gf.Vec3d(0.266, -0.076, 0.085)
LIDAR_BOX_MAX = Gf.Vec3d(0.418, 0.076, 0.245)

MAX_VEL = "physxJoint:maxJointVelocity"  # deg/s
EXPECTED_REVOLUTE = 23  # 12 legs + 4 wheels + 6 arm + gripper


def say(*a):
    print("BUILD|", *a, flush=True)


def set_payload(layer, path, asset):
    spec = layer.GetPrimAtPath(path)
    pl = spec.payloadList
    old = list(pl.GetAddedOrExplicitItems())
    if pl.isExplicit:
        pl.explicitItems = [Sdf.Payload(asset)]
    else:
        pl.ClearEdits()
        pl.prependedItems = [Sdf.Payload(asset)]
    say(f"payload {path}: {[p.assetPath for p in old]} -> {asset}")


def look_at_orient(eye, target, up=Gf.Vec3d(0, 0, 1)):
    """Quaternion for a USD camera (looks down -Z, +Y up) at `eye` facing `target`."""
    z = (eye - target).GetNormalized()
    x = Gf.Cross(up, z).GetNormalized()
    y = Gf.Cross(z, x)
    m = Gf.Matrix3d(x[0], x[1], x[2], y[0], y[1], y[2], z[0], z[1], z[2])  # rows = camera axes
    return Gf.Quatf(m.ExtractRotation().GetQuat())


def set_local_pose(prim, pos, orient):
    """Author translate + orient on a prim that already has a translate/orient/scale stack."""
    xf = UsdGeom.Xformable(prim)
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(pos))
        elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            op.Set(orient if op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat else Gf.Quatd(orient))


def main():
    global DST, GROUND
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrist", choices=("none", "d405"), default="d405",
                    help="d405: RealSense D405 + bracket on the gripper, camera with its intrinsics (section I)")
    ap.add_argument("--body_cams", choices=("keep", "d455"), default="d455",
                    help="d455: front/rear head cameras get the RealSense D455 colour lens (section J)")
    ap.add_argument("--dst", default=DST.name, help="output file name in robots_usd/")
    args = ap.parse_args()
    DST = ROOT / args.dst
    GROUND = ROOT / (DST.stem + "_on_ground.usd")
    for f in (SRC, ROOT / PATCHED_B2W, ROOT / GRIPPER_ARM, ORIGINAL_ARM):
        if not f.exists():
            sys.exit(f"BUILD| FATAL {f} not found")

    shutil.copyfile(SRC, DST)
    say(f"copied {SRC.name} -> {DST.name}")
    stage = Usd.Stage.Open(str(DST))
    layer = stage.GetRootLayer()

    # A + 4. Payloads: gripper arm, patched body. Re-open so both are composed
    # before anything is measured.
    set_payload(layer, "/Unitree_B2W", PATCHED_B2W)
    set_payload(layer, Z1, GRIPPER_ARM)
    # F. The copied variant sets with unresolvable payload paths.
    spec = layer.GetPrimAtPath("/Unitree_B2W")
    copied = list(spec.variantSets.keys())
    for name in copied:
        spec.variantSetNameList.RemoveItemEdits(name)
        del spec.variantSets[name]
    say(f"[F] removed copied variant sets {copied} (selections kept: {dict(spec.variantSelections)})")
    layer.Save()
    stage = Usd.Stage.Open(str(DST))

    # E. Arm towards the rear, along the body's -x (converted into the frame the
    #    z1 prim's translate lives in). Fix 1 then moves the mount joint to match.
    z1 = stage.GetPrimAtPath(Z1)
    xc = UsdGeom.XformCache()
    to_parent = xc.GetLocalToWorldTransform(stage.GetPrimAtPath(BASE)) * \
        xc.GetLocalToWorldTransform(z1.GetParent()).GetInverse()
    shift = to_parent.TransformDir(Gf.Vec3d(-ARM_SHIFT_BACK, 0, 0))
    before = relative_translation(stage, Z1_WORLD, BASE)
    for op in UsdGeom.Xformable(z1).GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(op.Get() + type(op.Get())(*shift))
            break
    after = relative_translation(stage, Z1_WORLD, BASE)
    say(f"[E] arm base (body frame) {tuple(round(v, 4) for v in before)} -> {tuple(round(v, 4) for v in after)}")

    # G. Self-collision on, design overlaps filtered, top of the body covered.
    base = stage.GetPrimAtPath(BASE)
    base.GetAttribute(SELF_COLLISION_ATTR).Set(True)
    for a, b in FILTERED_PAIRS:
        UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(a)).CreateFilteredPairsRel().AddTarget(b)
    cover = UsdGeom.Cube.Define(stage, TOP_COVER)
    cover.CreateSizeAttr(1.0)
    cover.CreatePurposeAttr(UsdGeom.Tokens.guide)  # collider only, not rendered
    xf = UsdGeom.Xformable(cover)
    xf.AddTranslateOp().Set((TOP_COVER_MIN + TOP_COVER_MAX) / 2)
    xf.AddScaleOp().Set(Gf.Vec3f(*(TOP_COVER_MAX - TOP_COVER_MIN)))
    UsdPhysics.CollisionAPI.Apply(cover.GetPrim())
    say(f"[G] self-collision on; filtered {[(a.split('/')[-1], b.split('/')[-1]) for a, b in FILTERED_PAIRS]}; "
        f"top cover {tuple(TOP_COVER_MIN)}..{tuple(TOP_COVER_MAX)}")

    # H. Lidar cage collider: drop the too-tall capsule, add a box in its place.
    stage.GetPrimAtPath(f"{LIDAR}/collisions").SetActive(False)
    xc = UsdGeom.XformCache()
    base_to_lidar = xc.GetLocalToWorldTransform(stage.GetPrimAtPath(BASE)) * \
        xc.GetLocalToWorldTransform(stage.GetPrimAtPath(LIDAR)).GetInverse()
    lidar_rot = base_to_lidar.ExtractRotationMatrix()
    cage = UsdGeom.Cube.Define(stage, f"{LIDAR}/cage_collider")
    cage.CreateSizeAttr(1.0)
    cage.CreatePurposeAttr(UsdGeom.Tokens.guide)
    cxf = UsdGeom.Xformable(cage)
    cxf.AddTransformOp().Set(Gf.Matrix4d().SetScale(Gf.Vec3d(*(LIDAR_BOX_MAX - LIDAR_BOX_MIN)))
                             * Gf.Matrix4d().SetTranslate((LIDAR_BOX_MIN + LIDAR_BOX_MAX) / 2) * base_to_lidar)
    UsdPhysics.CollisionAPI.Apply(cage.GetPrim())
    say(f"[H] lidar capsule off; cage box {tuple(LIDAR_BOX_MIN)}..{tuple(LIDAR_BOX_MAX)} (body frame)")

    # B. The importer's world anchor.
    stage.GetPrimAtPath(ROOT_JOINT).SetActive(False)
    say(f"[B] deactivated {ROOT_JOINT} (world anchor + second articulation root)")

    # 1. Mount joint follows the arm's actual placement.
    mount = relative_translation(stage, Z1_WORLD, BASE)
    joint = UsdPhysics.Joint(stage.GetPrimAtPath(MOUNT_JOINT))
    joint.GetLocalPos0Attr().Set(mount)
    joint.GetLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
    say(f"[1] mount localPos0 -> {tuple(round(v, 4) for v in mount)}")

    # 2 + 5. z1/world: no articulation root, explicit tiny mass.
    world = stage.GetPrimAtPath(Z1_WORLD)
    world.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    mass = UsdPhysics.MassAPI.Apply(world)
    mass.CreateMassAttr().Set(TINY_MASS)
    mass.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*TINY_INERTIA))
    say(f"[2,5] z1/world: no articulation root, mass {TINY_MASS} kg")

    # C + 3. Arm limits from the original Z1, arm and gripper gains.
    orig = Usd.Stage.Open(str(ORIGINAL_ARM))
    for name in ARM_JOINTS:
        prim = stage.GetPrimAtPath(f"{Z1}/joints/{name}")
        src = UsdPhysics.RevoluteJoint(orig.GetPrimAtPath(f"/z1_description/joints/{name}"))
        rj = UsdPhysics.RevoluteJoint(prim)
        before = (rj.GetLowerLimitAttr().Get(), rj.GetUpperLimitAttr().Get())
        rj.GetLowerLimitAttr().Set(src.GetLowerLimitAttr().Get())
        rj.GetUpperLimitAttr().Set(src.GetUpperLimitAttr().Get())
        drive = UsdPhysics.DriveAPI(prim, "angular")
        drive.GetTypeAttr().Set(UsdPhysics.DriveAPI(src.GetPrim(), "angular").GetTypeAttr().Get())
        drive.GetStiffnessAttr().Set(ARM_KP * DEG)
        drive.GetDampingAttr().Set(ARM_KD * DEG)
        vmax = src.GetPrim().GetAttribute(MAX_VEL).Get()
        prim.GetAttribute(MAX_VEL).Set(vmax)
        say(f"[C] {name} limits ({before[0]:.1f}, {before[1]:.1f}) -> "
            f"({src.GetLowerLimitAttr().Get():.1f}, {src.GetUpperLimitAttr().Get():.1f}) deg, "
            f"drive {drive.GetTypeAttr().Get()}, max vel {vmax:.0f} deg/s")
    say(f"[3] arm drives kp {ARM_KP} N*m/rad, kd {ARM_KD} N*m*s/rad")
    gprim = stage.GetPrimAtPath(f"{Z1}/joints/{GRIPPER_JOINT}")
    grip = UsdPhysics.DriveAPI(gprim, "angular")
    grip.GetTypeAttr().Set("force")
    grip.GetStiffnessAttr().Set(GRIP_KP * DEG)
    grip.GetDampingAttr().Set(GRIP_KD * DEG)
    gprim.GetAttribute(MAX_VEL).Set(vmax)
    say(f"[3] gripper drive force, kp {GRIP_KP} N*m/rad, kd {GRIP_KD} N*m*s/rad, "
        f"max {grip.GetMaxForceAttr().Get()} N*m, max vel {vmax:.0f} deg/s")

    # D. End-effector camera under the fixed jaw.
    cam = stage.GetPrimAtPath(EE_CAMERA)
    set_local_pose(cam, EE_CAM_POS, look_at_orient(EE_CAM_POS, EE_CAM_LOOK_AT))
    yaw = math.degrees(math.atan2(EE_CAM_POS[1] - EE_CAM_LOOK_AT[1], EE_CAM_LOOK_AT[0] - EE_CAM_POS[0]))
    say(f"[D] {EE_CAMERA} -> link06 {tuple(EE_CAM_POS)}, turned {yaw:.1f} deg inward")
    if args.wrist == "d405":  # I. replaces D's camera pose
        import wrist_d405
        wrist_d405.author(stage, say)
    if args.body_cams == "d455":  # J.
        d455 = Usd.Stage.Open(str(D455))  # kept alive: a prim of a collected stage is invalid
        src = UsdGeom.Camera(d455.GetPrimAtPath(D455_COLOR))
        for path in BODY_CAMERAS:
            cam = UsdGeom.Camera(stage.GetPrimAtPath(path))
            for a in ("FocalLength", "HorizontalAperture", "VerticalAperture", "ClippingRange"):
                getattr(cam, f"Get{a}Attr")().Set(getattr(src, f"Get{a}Attr")().Get())
        f, ha, va = (getattr(src, f"Get{a}Attr")().Get() for a in ("FocalLength", "HorizontalAperture", "VerticalAperture"))
        say(f"[J] {[p.split('/')[-1] for p in BODY_CAMERAS]}: RealSense D455 colour lens, focal {f:.2f} mm, "
            f"{2 * math.degrees(math.atan(ha / 2 / f)):.1f} x {2 * math.degrees(math.atan(va / 2 / f)):.1f} deg")

    stage.GetRootLayer().Save()
    verify(DST)
    if args.wrist == "d405":
        import wrist_d405
        worst, _ = wrist_d405.check(Usd.Stage.Open(str(DST)), say)
        assert worst > 0.002, "the moving jaw comes within 2 mm of the camera holder"
    build_ground_stage()
    say("done")


def verify(path):
    stage = Usd.Stage.Open(str(path))
    roots = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    revolute = [p.GetName() for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
    wheel = stage.GetPrimAtPath("/Unitree_B2W/joints/FL_foot_joint")
    wheel_kp = UsdPhysics.DriveAPI(wheel, "angular").GetStiffnessAttr().Get()
    mount = UsdPhysics.Joint(stage.GetPrimAtPath(MOUNT_JOINT)).GetLocalPos0Attr().Get()
    arm_now = relative_translation(stage, Z1_WORLD, BASE)
    cameras = [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdGeom.Camera)]
    through = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    anchor = stage.GetPrimAtPath(ROOT_JOINT)
    drive_types = {n: UsdPhysics.DriveAPI(stage.GetPrimAtPath(f"{Z1}/joints/{n}"), "angular").GetTypeAttr().Get()
                   for n in ARM_JOINTS + [GRIPPER_JOINT]}

    say(f"verify: articulation roots {roots}")
    say(f"verify: revolute joints {len(revolute)} (expect {EXPECTED_REVOLUTE}); gripper: {GRIPPER_JOINT in revolute}")
    say(f"verify: wheel stiffness {wheel_kp} (expect 0)")
    say(f"verify: mount {tuple(round(v, 4) for v in mount)} vs arm placement {tuple(round(v, 4) for v in arm_now)}")
    say(f"verify: root_joint active={anchor.IsActive()}")
    say(f"verify: cameras {cameras}")
    errors = [str(e) for e in stage.GetCompositionErrors()]
    errors = [e for e in errors if "https:" not in e]
    say(f"verify: composition errors {len(errors)} (the S3 lidar reference only fails offline)")
    say(f"verify: arm/gripper drive types {set(drive_types.values())}")
    self_coll = stage.GetPrimAtPath(BASE).GetAttribute(SELF_COLLISION_ATTR).Get()
    filtered = {a: [str(t) for t in UsdPhysics.FilteredPairsAPI(stage.GetPrimAtPath(a)).GetFilteredPairsRel().GetTargets()]
                for a, _ in FILTERED_PAIRS}
    say(f"verify: self-collision {self_coll}; filtered pairs {filtered}")
    xc = UsdGeom.XformCache()
    m = xc.GetLocalToWorldTransform(stage.GetPrimAtPath(f"{LIDAR}/cage_collider")) * \
        xc.GetLocalToWorldTransform(stage.GetPrimAtPath(BASE)).GetInverse()
    corners = [m.Transform(Gf.Vec3d(x, y, z)) for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)]
    lo = Gf.Vec3d(*(min(c[i] for c in corners) for i in range(3)))
    hi = Gf.Vec3d(*(max(c[i] for c in corners) for i in range(3)))
    say(f"verify: lidar cage collider (body frame) {tuple(round(v, 3) for v in lo)}..{tuple(round(v, 3) for v in hi)}; "
        f"capsule active: {stage.GetPrimAtPath(f'{LIDAR}/collisions').IsActive()}")
    grip_x = gripper_front_x(stage)
    clearance = LIDAR_CAGE_FRONT_X - grip_x
    say(f"verify: stowed gripper front x {grip_x:.3f} m, lidar cage front {LIDAR_CAGE_FRONT_X:.3f} m "
        f"-> clearance {clearance * 100:.1f} cm (min {MIN_CAGE_CLEARANCE * 100:.0f})")

    assert roots == [BASE], "expected base_link as the only articulation root"
    assert len(revolute) == EXPECTED_REVOLUTE and GRIPPER_JOINT in revolute, "joint count"
    assert wheel_kp == 0.0, "wheels not patched"
    assert (Gf.Vec3f(mount) - arm_now).GetLength() < 1e-4, "mount joint does not match placement"
    assert not anchor.IsActive(), "root_joint still active"
    assert set(drive_types.values()) == {"force"}, f"non-force drives: {drive_types}"
    assert clearance >= MIN_CAGE_CLEARANCE, "stowed gripper too close to the lidar cage"
    assert not errors, errors
    assert self_coll is True, "self-collision off"
    assert (lo - LIDAR_BOX_MIN).GetLength() < 1e-3 and (hi - LIDAR_BOX_MAX).GetLength() < 1e-3, "lidar cage box misplaced"
    assert all(b in filtered[a] for a, b in FILTERED_PAIRS), "filtered pair missing"
    def count_meshes(st):
        return sum(1 for q in Usd.PrimRange(st.GetPseudoRoot(), through) if q.IsA(UsdGeom.Mesh) and q.IsActive())
    meshes = count_meshes(stage)
    expected = sum(count_meshes(Usd.Stage.Open(str(ROOT / src))) for src in (PATCHED_B2W, GRIPPER_ARM))
    if stage.GetPrimAtPath(f"{Z1}/link06/wrist_d405"):  # I. + Intel's D405 body + the holder's ring
        expected += count_meshes(Usd.Stage.Open(str(ROOT / "RealSense" / "d405.usd"))) + 1
    say(f"verify: {meshes} active meshes (body + arm sources: {expected})")
    assert meshes == expected, "meshes lost or duplicated"  # a mesh count, not a guess
    assert len(cameras) == 3 and EE_CAMERA in cameras, "expected front, rear and end-effector cameras"


def gripper_front_x(stage):
    """Largest body-frame x of any gripper vertex (visual or collider), at the authored (stowed) pose."""
    xc = UsdGeom.XformCache()
    base_inv = xc.GetLocalToWorldTransform(stage.GetPrimAtPath(BASE)).GetInverse()
    best = -1e9
    for link in ("gripperStator", "gripperMover"):  # gripperStator includes the wrist camera holder (I)
        for q in Usd.PrimRange(stage.GetPrimAtPath(f"{Z1}/{link}"), Usd.TraverseInstanceProxies()):
            if q.IsA(UsdGeom.Mesh):
                pts = q.GetAttribute("points").Get()
            elif q.IsA(UsdGeom.Cube):  # the holder (I): unit cubes scaled by their transform
                h = q.GetAttribute("size").Get() / 2
                pts = [(x, y, z) for x in (-h, h) for y in (-h, h) for z in (-h, h)]
            else:
                continue
            m = xc.GetLocalToWorldTransform(q) * base_inv
            best = max(best, max(m.Transform(Gf.Vec3d(*v))[0] for v in pts))
    return best


def build_ground_stage():
    """Ground plane + robot, spawned high enough to settle onto its wheels."""
    if GROUND.exists():
        GROUND.unlink()
    stage = Usd.Stage.CreateNew(str(GROUND))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    plane = UsdGeom.Plane.Define(stage, "/World/ground")
    plane.CreateAxisAttr(UsdGeom.Tokens.z)
    plane.CreateWidthAttr(100.0)
    plane.CreateLengthAttr(100.0)
    UsdPhysics.CollisionAPI.Apply(plane.GetPrim())

    robot = stage.DefinePrim("/World/b2w_z1", "Xform")
    robot.GetPayloads().AddPayload(f"./{DST.name}")
    xf = UsdGeom.Xformable(robot)  # the payload brings its own op stack
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
