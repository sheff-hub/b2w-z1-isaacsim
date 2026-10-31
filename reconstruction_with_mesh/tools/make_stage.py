#!/usr/bin/env python
"""Build an Isaac Sim stage: the NuRec splat (by reference) + an invisible collision mesh.

Run in the tools environment (.venv: usd-core, trimesh) or the 3DGRUT venv:

    python make_collision_stage.py --splat export_fixed.usdz --mesh tsdf_500k_colmap.ply \
        --out room05_collision.usda [--scale 1.0]

Frame. 3DGRUT's NuRec serializer authors the volume's xformOp:transform as
inv(N) @ F flattened row-major: N is its gravity-normalising transform, F the
3DGRUT->USD axis flip. USD reads matrices as row vectors with translation in
row 3, so the rotation part acts as p @ M[:3,:3] (= F R p, gravity-aligned and
Z-up) and N's translation lands in column 3, which USD consumers (RTX, PhysX)
do not use. The mesh vertices are baked with exactly that rotation, read from
the referenced file, so mesh and splat share one frame under standard USD
semantics. Verify in Isaac before trusting contacts to the centimetre: if a
renderer ever honoured the column-3 values, the splat would sit ~0.16 units
from the mesh.

Scale. The asset is in COLMAP units, not metres. --scale goes on the parent
Xform, so splat and mesh scale together; measure one known distance in the
room to set it (room05 estimate: ~0.8 m/unit, from the bed height).
"""

import argparse
import os
from pathlib import Path

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, Vt


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splat", required=True, help="NuRec .usdz from export_nurec_fixed.py")
    p.add_argument("--mesh", required=True, help="Collision mesh in the COLMAP/3DGRUT world frame (.ply/.obj)")
    p.add_argument("--out", required=True, help="Output .usda")
    p.add_argument("--volume", default="/World/gauss/gauss")
    p.add_argument("--scale", type=float, default=1.0, help="Metres per scene unit, applied to splat and mesh together.")
    p.add_argument("--visible", action="store_true", help="Leave the collision mesh visible (for checking alignment).")
    p.add_argument("--nvidia-layout", action="store_true",
                   help="Wire the collider the way NVIDIA's NuRec scenes do: the volume's `proxy` relationship targets it, "
                        "it is a matte object (primvars:isMatteObject), and a dome light (intensity 1000, as in NVIDIA's scenes) is added for inserted assets.")
    p.add_argument("--level", default=None,
                   help="JSON with a 'level_frame' {e1, e2, up, origin} in the splat's data frame. Adds a rotation + "
                        "translation on /World/room so `up` becomes +Z and `origin` (a floor point) goes to 0. Needed "
                        "for nerfstudio splats, whose +Z is the mean phone up, not gravity.")
    p.add_argument("--room-matrix", default=None,
                   help="4x4 .npy (column-vector convention) put on /World/room instead of --level, mapping the splat's "
                        "data frame (after the volume rotation) to metres, Z-up, floor at 0 -- e.g. a nerfstudio "
                        "dataparser frame. May include scale.")
    a = p.parse_args()

    splat = Path(a.splat).resolve()
    src = Usd.Stage.Open(str(splat))
    vol = src.GetPrimAtPath(a.volume)
    M = np.array(vol.GetAttribute("xformOp:transform").Get())
    parent = np.array(UsdGeom.Xformable(vol.GetParent()).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    assert np.allclose(parent, np.eye(4)), "volume parent is not at identity; extend the bake to include it"
    rot = M[:3, :3]
    assert abs(np.linalg.det(rot) - 1) < 1e-3, "volume transform is not a pure rotation"
    print(f"volume rotation (row-vector, as USD applies it):\n{rot.round(4)}\nignored column 3: {M[:3, 3].round(4)}")

    mesh = trimesh.load(a.mesh, process=False)
    pts = np.asarray(mesh.vertices) @ rot  # row-vector, same as USD applies to the volume
    faces = np.asarray(mesh.faces)
    print(f"mesh: {len(pts):,} verts, {len(faces):,} tris | baked bbox {pts.min(0).round(2)} .. {pts.max(0).round(2)}")

    out = Path(a.out).resolve()
    st = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    world = UsdGeom.Xform.Define(st, "/World")
    st.SetDefaultPrim(world.GetPrim())

    room = UsdGeom.Xform.Define(st, "/World/room")
    room.AddScaleOp().Set(Gf.Vec3f(a.scale, a.scale, a.scale))
    room.GetPrim().SetDocumentation(f"Scale op: scene units -> metres (1.0 = data already in metres, e.g. ARKit; else COLMAP units). Splat: {splat.name}")
    if a.level:
        import json
        lf = json.loads(Path(a.level).read_text())["level_frame"]
        B = np.c_[lf["e1"], lf["e2"], lf["up"]]  # columns: new X, Y, Z in the data frame
        assert np.allclose(B.T @ B, np.eye(3), atol=1e-6) and np.linalg.det(B) > 0, "level frame is not a rotation"
        L = np.eye(4)
        L[:3, :3] = B                                   # row-vector: p @ B = (e1.p, e2.p, up.p)
        L[3, :3] = -np.asarray(lf["origin"]) @ B
        # xformOpOrder [scale, transform]: the transform acts first, then the scale (USD lists outermost first)
        room.AddTransformOp().Set(Gf.Matrix4d(L.tolist()))
        print(f"levelling: data up {np.round(lf['up'], 4)} -> +Z, floor point {np.round(lf['origin'], 4)} -> 0")

    if a.room_matrix:
        T = np.load(a.room_matrix)
        room.AddTransformOp().Set(Gf.Matrix4d(T.T.tolist()))   # USD row vectors: p_row @ T^T
        print(f"room matrix from {a.room_matrix}:\n{T.round(4)}")

    ref = st.DefinePrim("/World/room/splat")
    ref.GetReferences().AddReference(os.path.relpath(splat, out.parent))

    col = UsdGeom.Mesh.Define(st, "/World/room/collision")
    col.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
    col.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32)))
    col.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.astype(np.int32).ravel()))
    col.CreateExtentAttr([Gf.Vec3f(*pts.min(0)), Gf.Vec3f(*pts.max(0))])
    col.CreateDoubleSidedAttr(True)
    col.CreateDisplayColorAttr([Gf.Vec3f(0.95, 0.45, 0.15)])
    UsdPhysics.CollisionAPI.Apply(col.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(col.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.none)  # exact triangles, static
    if not a.visible:
        col.MakeInvisible()
    if a.nvidia_layout:
        UsdGeom.PrimvarsAPI(col).CreatePrimvar("isMatteObject", Sdf.ValueTypeNames.Bool).Set(True)
        vol_in_stage = Sdf.Path("/World/room/splat").AppendPath(Sdf.Path(a.volume).MakeRelativePath("/World"))
        st.OverridePrim(vol_in_stage).CreateRelationship("proxy", custom=True).SetTargets([col.GetPath()])
        dome = UsdLux.DomeLight.Define(st, "/World/dome_light")
        dome.CreateIntensityAttr(1000.0)
        dome.GetPrim().CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(False)
        print(f"NVIDIA layout: {vol_in_stage}.proxy -> {col.GetPath()}, matte collider, dome light 1000")
    st.GetRootLayer().Save()
    print(f"wrote {out}  (collision {'visible' if a.visible else 'invisible'}, scale {a.scale})")


if __name__ == "__main__":
    main()
