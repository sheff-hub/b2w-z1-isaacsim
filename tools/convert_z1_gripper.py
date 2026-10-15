"""Convert the Z1-with-gripper URDF to USD, headless, and check the meshes landed.

The URDF Unitree ships references its meshes as package://z1_description/...,
which the importer cannot resolve outside a ROS workspace: it then writes the
link/joint tree with no geometry (an invisible robot, ~4 KB base layer). This
reads z1_gripper_local.urdf, made by tools/make_z1_gripper_urdf.py: paths
rewritten to ../meshes/, and the jaw meshes added as the jaws' colliders.

    python3 tools/make_z1_gripper_urdf.py && ./run_script.sh convert_z1_gripper.py
"""

import argparse

p = argparse.ArgumentParser()
p.add_argument("--urdf", default="/workspace/robots_usd/z1_gripper_description/urdf/z1_gripper_local.urdf")
p.add_argument("--out", default="/workspace/robots_usd/Unitree_Z1_gripper/z1_gripper.usd")
args = p.parse_args()

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import os

import omni.kit.commands
from isaacsim.asset.importer.urdf import _urdf
from pxr import Usd, UsdGeom, UsdPhysics

status, cfg = omni.kit.commands.execute("URDFCreateImportConfig")
cfg.merge_fixed_joints = False  # keep gripperStator as its own link: it is the grasp frame
cfg.fix_base = True
cfg.make_default_prim = True
cfg.create_physics_scene = False
cfg.import_inertia_tensor = True
cfg.distance_scale = 1.0
cfg.collision_from_visuals = False  # the URDF has its own primitive colliders
# Only the two jaw colliders are meshes; decompose them so the collider follows
# the fingers instead of one hull that fills the gap between them.
cfg.convex_decomp = True
cfg.self_collision = False
cfg.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION

os.makedirs(os.path.dirname(args.out), exist_ok=True)
ok, prim_path = omni.kit.commands.execute(
    "URDFParseAndImportFile", urdf_path=args.urdf, import_config=cfg, dest_path=args.out
)
print(f"[convert] import ok={ok} prim={prim_path} -> {args.out}")

# The importer gives frame-only links (z1/world) a `visuals` reference to a prim
# it never writes: a red "unresolved prim path" in the Stage tree.
from usd_fixes import drop_dangling_refs  # noqa: E402

base_layer = os.path.join(os.path.dirname(args.out), "configuration",
                          os.path.basename(args.out).replace(".usd", "_base.usd"))
print(f"[convert] dropped dangling references: {drop_dangling_refs(base_layer)}")

stage = Usd.Stage.Open(args.out)
meshes = [p for p in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()) if p.IsA(UsdGeom.Mesh)]
for link in ("gripperStator", "gripperMover"):
    lp = stage.GetPrimAtPath(f"{prim_path}/{link}")
    for q in Usd.PrimRange(lp, Usd.TraverseInstanceProxies()):
        if q.HasAPI(UsdPhysics.CollisionAPI):
            approx = q.GetAttribute("physics:approximation").Get() if q.HasAPI(UsdPhysics.MeshCollisionAPI) else "-"
            print(f"[convert] collider {link}: {q.GetTypeName()} approximation={approx} {q.GetPath()}")
joints = [p.GetName() for p in stage.Traverse() if p.GetTypeName().endswith("Joint")]
bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(
    stage.GetDefaultPrim()
).ComputeAlignedRange()
print(f"[convert] meshes={len(meshes)} joints={joints}")
print(f"[convert] bbox min={tuple(round(v, 3) for v in bbox.GetMin())} max={tuple(round(v, 3) for v in bbox.GetMax())}")
for root, _, files in os.walk(os.path.dirname(args.out)):
    for f in files:
        fp = os.path.join(root, f)
        print(f"[convert] {os.path.getsize(fp):>10} {fp}")

app.close()
