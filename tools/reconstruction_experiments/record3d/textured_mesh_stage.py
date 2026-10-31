#!/usr/bin/env python
"""Isaac Sim stage from a textured OBJ (e.g. nerfstudio `ns-export poisson`): the mesh is both what you see and
what collides.

Run in the 3DGRUT venv (pxr + trimesh):

    python textured_mesh_stage.py --obj mesh.obj --out stage.usda [--transform T.npy] [--no-collision]

--transform: 4x4 applied to the vertices (row-major, column vectors), e.g. nerfstudio frame -> metres, Z-up,
floor at z = 0. The texture is copied next to the stage and referenced relatively, so keep the folder together.
Layout: /World/room/mesh (UsdPreviewSurface with the texture; static triangle-mesh collider, approximation none),
/World/dome_light (intensity 1000, invisible to the camera) -- the same light as our NuRec stages.
"""
import argparse, shutil
from pathlib import Path
import numpy as np, trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt

ap = argparse.ArgumentParser(); ap.add_argument("--obj", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--transform"); ap.add_argument("--no-collision", action="store_true")
a = ap.parse_args(); out = Path(a.out).resolve(); out.parent.mkdir(parents=True, exist_ok=True)
m = trimesh.load(a.obj, process=False, force="mesh")
V = np.asarray(m.vertices, float)
if a.transform: T = np.load(a.transform); V = V @ T[:3, :3].T + T[:3, 3]
F = np.asarray(m.faces, np.int32)
uv = getattr(m.visual, "uv", None)
tex = None
if uv is not None and getattr(m.visual, "material", None) is not None:
    img = getattr(m.visual.material, "image", None) or getattr(m.visual.material, "baseColorTexture", None)
    if img is not None:
        tex = out.parent / f"{out.stem}_texture.png"; img.save(tex)
st = Usd.Stage.CreateNew(str(out)); UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(st, 1.0)
world = UsdGeom.Xform.Define(st, "/World"); st.SetDefaultPrim(world.GetPrim()); UsdGeom.Xform.Define(st, "/World/room")
mesh = UsdGeom.Mesh.Define(st, "/World/room/mesh")
mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(V.astype(np.float32)))
mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(F), 3, np.int32)))
mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(F.ravel()))
mesh.CreateExtentAttr([Gf.Vec3f(*V.min(0).tolist()), Gf.Vec3f(*V.max(0).tolist())])
mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none); mesh.CreateDoubleSidedAttr(True)
mat = UsdShade.Material.Define(st, "/World/room/material")
sh = UsdShade.Shader.Define(st, "/World/room/material/surface"); sh.CreateIdAttr("UsdPreviewSurface")
sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0); sh.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0, 0, 0))
sh.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
if tex is not None and uv is not None:
    pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    pv.Set(Vt.Vec2fArray.FromNumpy(np.asarray(uv, np.float32)))
    rd = UsdShade.Shader.Define(st, "/World/room/material/st_reader"); rd.CreateIdAttr("UsdPrimvarReader_float2")
    rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tx = UsdShade.Shader.Define(st, "/World/room/material/texture"); tx.CreateIdAttr("UsdUVTexture")
    tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"./{tex.name}")
    tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(), "result")
    tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), "rgb")
    # unlit look: the texture already holds the lighting, so emit it as well
    sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), "rgb")
else:
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.6, 0.6, 0.6))
mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
if not a.no_collision:
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
dome = UsdLux.DomeLight.Define(st, "/World/dome_light"); dome.CreateIntensityAttr(1000.0)
dome.GetPrim().CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(False)
st.GetRootLayer().Save()
print(f"wrote {out}: {len(V):,} vertices, {len(F):,} triangles, texture {tex.name if tex else 'none'}, "
      f"collision {'off' if a.no_collision else 'on'}; bounds {V.min(0).round(2)} .. {V.max(0).round(2)}")
