"""Open a NuRec USDZ stage headlessly and report anything that went wrong.

Run inside the Isaac Sim 5.1.0 container via run_stage_check.sh, which pins the
GPU and mounts the NuRec assets (config paths.nurec) at /workspace/scenes/nurec.

    ./run_stage_check.sh [<stage path inside container>]

Everything it prints is prefixed with CHECK| so the report can be separated from
Kit's own very chatty log.
"""

import argparse
import os
import sys
import time

DEFAULT_STAGE = (
    "/workspace/scenes/nurec/nvidia/hand_hold-endeavor-livingroom/stage_volume.usdz"
)

parser = argparse.ArgumentParser()
parser.add_argument("stage", nargs="?", default=DEFAULT_STAGE)
parser.add_argument("--frames", type=int, default=60)
parser.add_argument("--out", default="/workspace/scenes/nurec/mine/_stage_check")
args = parser.parse_args()


def say(*a):
    print("CHECK|", *a, flush=True)


say("stage:", args.stage)
say("exists:", os.path.exists(args.stage))
if not os.path.exists(args.stage):
    sys.exit("CHECK| FATAL stage file not found inside the container")

t0 = time.time()
from isaacsim import SimulationApp  # noqa: E402

sim = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})
say(f"SimulationApp up in {time.time() - t0:.1f}s")

import carb  # noqa: E402
import omni.kit.app  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils  # noqa: E402

# ---------------------------------------------------------------------------
# The NuRec USD schema is not shipped in the container image; Kit pulls it from
# the extension registry the first time a NuRec prim is encountered. Enabling it
# up front turns a silent "unknown prim type" into a clear failure here.
# ---------------------------------------------------------------------------
mgr = omni.kit.app.get_app().get_extension_manager()
for ext in ("omni.usd.schema.omni_nurec_types",):
    try:
        mgr.set_extension_enabled_immediate(ext, True)
        say(f"extension {ext}: enabled={mgr.is_extension_enabled(ext)}")
    except Exception as exc:  # noqa: BLE001
        say(f"extension {ext}: FAILED to enable -- {exc}")

# ---------------------------------------------------------------------------
# Open
# ---------------------------------------------------------------------------
ctx = omni.usd.get_context()
t0 = time.time()
result = ctx.open_stage(args.stage)
say(f"open_stage returned {result!r} in {time.time() - t0:.1f}s")

stage = ctx.get_stage()
if stage is None:
    sim.close()
    sys.exit("CHECK| FATAL open_stage produced no stage")

say("root layer:", stage.GetRootLayer().identifier)
say("upAxis:", UsdGeom.GetStageUpAxis(stage))
say("metersPerUnit:", UsdGeom.GetStageMetersPerUnit(stage))
say("defaultPrim:", stage.GetDefaultPrim().GetName())
say(f"timeCode range: {stage.GetStartTimeCode()} .. {stage.GetEndTimeCode()}")

# ---------------------------------------------------------------------------
# Composition: did every sublayer / reference actually resolve?
# ---------------------------------------------------------------------------
# The arity of ComputeAllDependencies changed across USD releases (3 vs 4
# return values), so unpack defensively rather than assume one of them.
try:
    dep = UsdUtils.ComputeAllDependencies(stage.GetRootLayer().identifier)
    used = dep[0]
    unresolved = dep[-1] if len(dep) >= 4 else []
    say(f"dependencies resolved: {len(used)}  unresolved: {len(unresolved)}")
    for u in used:
        say("  layer:", u.identifier if hasattr(u, "identifier") else u)
    for u in unresolved:
        say("  UNRESOLVED:", u)
except Exception as exc:  # noqa: BLE001
    say("dependency scan failed:", exc)

# Independently, check that every asset-valued attribute resolves to a real file.
for prim in stage.Traverse():
    for attr in prim.GetAttributes():
        if attr.GetTypeName() == "asset":
            val = attr.Get()
            if val and not val.resolvedPath:
                say(f"  UNRESOLVED ASSET: {attr.GetPath()} -> {val.path}")

# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------
types = {}
cameras, colliders, nurec, meshes = [], [], [], []
typeless = []
for prim in stage.Traverse():
    tn = prim.GetTypeName() or "<untyped>"
    types[tn] = types.get(tn, 0) + 1
    if prim.IsA(UsdGeom.Camera):
        cameras.append(prim.GetPath().pathString)
    if prim.HasAPI(UsdPhysics.CollisionAPI):
        colliders.append(prim.GetPath().pathString)
    if prim.IsA(UsdGeom.Mesh):
        meshes.append(prim)
    if "NuRec" in str(tn) or "nurec" in prim.GetName().lower():
        nurec.append(prim)
    # A prim whose type the running Kit has no schema for is the classic
    # symptom of a missing schema extension: it loads, but does nothing.
    if tn != "<untyped>" and not Usd.SchemaRegistry().FindConcretePrimDefinition(tn):
        typeless.append((prim.GetPath().pathString, str(tn)))

say("prim types:", dict(sorted(types.items(), key=lambda kv: -kv[1])))
say(f"cameras: {len(cameras)}")
for c in cameras:
    say("  ", c)
say(f"prims with UsdPhysics.CollisionAPI: {len(colliders)}")
for c in colliders:
    say("  ", c)

if typeless:
    say(f"prims with an UNRECOGNISED type ({len(typeless)}) -- schema missing:")
    for p, tn in typeless[:20]:
        say(f"   {p}  type={tn}")

# Mesh detail
for m in meshes:
    mesh = UsdGeom.Mesh(m)
    pts = mesh.GetPointsAttr().Get()
    fvc = mesh.GetFaceVertexCountsAttr().Get()
    say(
        f"mesh {m.GetPath()}: points={len(pts) if pts else 0} "
        f"faces={len(fvc) if fvc else 0} visibility={mesh.GetVisibilityAttr().Get()}"
    )
    if m.HasAPI(UsdPhysics.CollisionAPI):
        capi = UsdPhysics.CollisionAPI(m)
        say(f"   collisionEnabled={capi.GetCollisionEnabledAttr().Get()}")
        mcapi = UsdPhysics.MeshCollisionAPI(m)
        approx = mcapi.GetApproximationAttr().Get() if mcapi else None
        say(f"   approximation={approx}")

# NuRec volume detail
for p in nurec:
    say(f"nurec prim {p.GetPath()} type={p.GetTypeName()}")
    for attr in p.GetAttributes():
        n = attr.GetName()
        if "nurec" in n or n in ("filePath", "fieldName", "fieldRole"):
            say(f"   {n} = {attr.Get()}")

# ---------------------------------------------------------------------------
# Physics: is there a scene and a ground to stand on?
# ---------------------------------------------------------------------------
scenes = [p.GetPath().pathString for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
say("PhysicsScene prims:", scenes or "NONE")
ground = [
    p.GetPath().pathString
    for p in stage.Traverse()
    if p.GetTypeName() == "Plane" or "groundplane" in p.GetName().lower().replace("_", "")
]
say("ground plane prims:", ground or "NONE")

# ---------------------------------------------------------------------------
# Render: force the RTX/NuRec path to actually initialise
# ---------------------------------------------------------------------------
os.makedirs(args.out, exist_ok=True)
render_ok = False
try:
    import omni.replicator.core as rep

    cam = cameras[0] if cameras else None
    if cam is None:
        say("no camera in stage -- creating one for the render test")
        cam = str(rep.create.camera(position=(3, 0, 1.5), look_at=(8, 0, 1.4)))
    rp = rep.create.render_product(cam, (640, 400))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([rp])
    say(f"render product on {cam}; stepping {args.frames} frames...")
    t0 = time.time()
    for i in range(args.frames):
        sim.update()
    say(f"{args.frames} frames in {time.time() - t0:.1f}s")

    # sim.update() alone advances the app but does not drive Replicator's
    # capture graph, so the annotator stays empty. An orchestrator step is what
    # actually renders into the render product.
    rep.orchestrator.step(rt_subframes=16)
    data = rgb.get_data()
    if data is not None and getattr(data, "size", 0):
        import numpy as np

        arr = np.asarray(data)
        say(f"rgb annotator: shape={arr.shape} dtype={arr.dtype} mean={arr.mean():.2f}")
        if arr.mean() < 1.0:
            say("WARNING rendered image is essentially black -- nothing was drawn")
        else:
            render_ok = True
        try:
            from PIL import Image

            out = os.path.join(args.out, "render.png")
            Image.fromarray(arr[..., :3].astype("uint8")).save(out)
            say("wrote", out)
        except Exception as exc:  # noqa: BLE001
            say("could not save png:", exc)
    else:
        say("WARNING rgb annotator returned no data")
except Exception:  # noqa: BLE001
    import traceback

    say("render test raised:")
    traceback.print_exc()

# ---------------------------------------------------------------------------
# Collision: raycast straight down onto the mesh from above the room
# ---------------------------------------------------------------------------
try:
    from isaacsim.core.api import World
    from omni.physx import get_physx_scene_query_interface

    world = World(stage_units_in_meters=1.0)
    world.reset()
    for _ in range(5):
        world.step(render=False)
    hits = 0
    for x, y in ((3.0, 0.0), (6.0, -1.0), (9.0, 1.0), (11.0, -2.0)):
        hit = get_physx_scene_query_interface().raycast_closest(
            [x, y, 3.0], [0.0, 0.0, -1.0], 10.0
        )
        if hit["hit"]:
            hits += 1
            say(f"raycast ({x},{y},3)->-Z hit {hit['rigidBody']} at d={hit['distance']:.3f}")
        else:
            say(f"raycast ({x},{y},3)->-Z MISS")
    say(f"collision raycasts hit {hits}/4")
except Exception:  # noqa: BLE001
    import traceback

    say("collision test raised:")
    traceback.print_exc()

say("render_ok =", render_ok)
say("DONE")
sim.close()
