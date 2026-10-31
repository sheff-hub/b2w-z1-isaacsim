"""Render a splat + collider stage from given camera poses, overlay the collider, and drop test spheres.

Run inside the Isaac Sim 5.1 container (see run_render_views.sh):
    ./python.sh /workspace/Isaac/render_stage_views.py --stage <usda> --views <views.json> --out <dir>

views.json: {"width": W, "height": H, "views": [{"name", "c2w" (4x4, OpenCV camera, world, metres), "fx"}],
             "drops": [[x, y, z], ...]}
Per view writes <name>_splat.png (collider hidden), <name>_collider.png (collider shown orange, matte off) and,
after the drop test, <name>_drop.png. Prints RENDERVIEWS| lines, including where each sphere came to rest.
"""
import argparse, json, os
TAG = "RENDERVIEWS|"
say = lambda *a: print(TAG, *a, flush=True)
ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True); ap.add_argument("--views", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--samples", type=int, default=32)
ap.add_argument("--splat-prim", default="/World/room/splat"); ap.add_argument("--collider-prim", default="/World/room/collision")
args = ap.parse_args(); cfg = json.load(open(args.views)); W, H = cfg["width"], cfg["height"]

from isaacsim import SimulationApp  # noqa: E402
sim = SimulationApp({"headless": True, "renderer": "RaytracedLighting", "width": W, "height": H})
import numpy as np  # noqa: E402
import omni.kit.app, omni.usd  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Sdf, UsdGeom  # noqa: E402

omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("omni.usd.schema.omni_nurec_types", True)
os.makedirs(args.out, exist_ok=True)
ctx = omni.usd.get_context(); ctx.open_stage(args.stage); stage = ctx.get_stage()
for _ in range(20): sim.update()
col = UsdGeom.Mesh(stage.GetPrimAtPath(args.collider_prim))
has_splat = stage.GetPrimAtPath(args.splat_prim).IsValid()      # a textured-mesh stage has no splat: mesh = look + collider
say("stage open; collider prim valid:", col.GetPrim().IsValid(), "splat prim valid:", has_splat)

cam = UsdGeom.Camera.Define(stage, "/World/CheckCam"); h_ap = 20.955
cam.GetHorizontalApertureAttr().Set(h_ap); cam.GetVerticalApertureAttr().Set(h_ap * H / W)
cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 1000.0))
cx = cam.GetPrim().CreateAttribute("xformOp:transform", Sdf.ValueTypeNames.Matrix4d)
cam.GetPrim().CreateAttribute("xformOpOrder", Sdf.ValueTypeNames.TokenArray).Set(["xformOp:transform"])
rp = rep.create.render_product("/World/CheckCam", (W, H)); annot = rep.AnnotatorRegistry.get_annotator("rgb"); annot.attach([rp])

def shoot(v, suffix):
    M = np.array(v["c2w"], float) @ np.diag([1, -1, -1, 1])           # OpenCV camera -> USD camera (-Z forward, +Y up)
    cx.Set(Gf.Matrix4d(M.T.tolist()))                                  # USD: row vectors, translation in the last row
    cam.GetFocalLengthAttr().Set(float(v["fx"]) * h_ap / W)
    for _ in range(3): rep.orchestrator.step(rt_subframes=args.samples)
    a = np.asarray(annot.get_data())[..., :3].astype(np.uint8)
    Image.fromarray(a).save(os.path.join(args.out, f"{v['name']}_{suffix}.png")); return a

for _ in range(10): rep.orchestrator.step(rt_subframes=args.samples)
if has_splat: col.MakeInvisible()
for v in cfg["views"]: shoot(v, "splat")
say("splat views done")
matte = col.GetPrim().GetAttribute("primvars:isMatteObject")
if has_splat:
    splat = UsdGeom.Imageable(stage.GetPrimAtPath(args.splat_prim))
    splat.MakeInvisible(); col.MakeVisible()
    if matte.IsValid(): matte.Set(False)
dc = UsdGeom.PrimvarsAPI(col).GetPrimvar("displayColor"); dc.SetInterpolation(UsdGeom.Tokens.constant)
col.GetDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.45, 0.1)])
from pxr import UsdShade  # noqa: E402
binding = UsdShade.MaterialBindingAPI(col.GetPrim()); bound = binding.GetDirectBinding().GetMaterialPath()
if bound: binding.UnbindDirectBinding()                                # show the bare geometry in orange
from pxr import UsdLux  # noqa: E402
tmp_light = None
if not any(pr.HasAPI(UsdLux.LightAPI) for pr in stage.Traverse()):   # unlit stage (e.g. Stani's): light the mesh for this pass
    tmp_light = UsdLux.DomeLight.Define(stage, "/World/_check_dome_light"); tmp_light.CreateIntensityAttr(1000.0)
    say("stage has no light; temporary dome light added for the collider pass")
for v in cfg["views"]: shoot(v, "collider")                            # the collider alone, splat hidden
if tmp_light: stage.RemovePrim(tmp_light.GetPath())
say("collider views done")
if bound: binding.Bind(UsdShade.Material(stage.GetPrimAtPath(bound)))
if has_splat:
    splat.MakeVisible(); col.MakeInvisible()
    if matte.IsValid(): matte.Set(True)

# drop test with plain USD physics prims, read back from PhysX directly
import omni.timeline  # noqa: E402
from omni.physx import get_physx_interface  # noqa: E402
from pxr import UsdPhysics  # noqa: E402
if not any(p.IsA(UsdPhysics.Scene) for p in stage.Traverse()):
    ps = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    ps.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1)); ps.CreateGravityMagnitudeAttr(9.81)
names = []
for k, p in enumerate(cfg.get("drops", [])):
    sp = UsdGeom.Sphere.Define(stage, f"/World/drop_{k}"); sp.CreateRadiusAttr(0.1)
    sp.CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.6, 1.0)]); sp.AddTranslateOp().Set(Gf.Vec3d(*p))
    UsdPhysics.RigidBodyAPI.Apply(sp.GetPrim()); UsdPhysics.CollisionAPI.Apply(sp.GetPrim())
    UsdPhysics.MassAPI.Apply(sp.GetPrim()).CreateMassAttr(1.0); names.append(f"/World/drop_{k}")
    # no rolling: the ball should test the floor where it lands (rolling 0.5 m took one into a doorway never seen)
    from pxr import PhysxSchema  # noqa: E402
    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(sp.GetPrim()); rb.CreateAngularDampingAttr(50.0); rb.CreateLinearDampingAttr(1.0)
tl = omni.timeline.get_timeline_interface(); tl.play()
for _ in range(240): sim.update()
px = get_physx_interface()
for k, n in enumerate(names):
    pos = np.array(px.get_rigidbody_transformation(n)["position"])
    say(f"sphere {k}: dropped from {np.round(cfg['drops'][k], 2).tolist()} -> rests at {np.round(pos, 3).tolist()} "
        f"(centre {pos[2]:.3f} m above the floor plane, radius 0.10)")
tl.pause()
for v in cfg["views"]: shoot(v, "drop")
say("DONE")
sim.close()
