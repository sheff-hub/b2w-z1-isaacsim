#!/usr/bin/env python
"""Intel RealSense D405 as a USD asset, from Intel's own mesh (realsense-ros, Apache-2.0).

    repos/3dgrut/.venv/bin/python tools/make_d405_usd.py
    in:  robots_usd/RealSense/d405.stl, _d405.urdf.xacro  (realsense2_description, ros2-master)
    out: robots_usd/RealSense/d405.usd

Intel's official CAD for the D400 cameras is SolidWorks only (D405_Solid.SLDPRT, dev.realsenseai.com
download 41950); the STL in Intel's ROS description is the open version of the same body: 42.09 x 42 x
23 mm (datasheet: 42 x 42 x 23).

Frames. The STL is in mm with its origin on the front plate, midway between the two imagers, +z forward.
The xacro places it in the camera's depth frame (x forward, y left, z up) with rpy (pi/2, 0, pi/2), i.e.
mesh (x, y, z) -> (z, x, y), the front plate 3.8 mm ahead of the depth origin (zero_depth_to_glass 3.7 mm +
glass_to_front 0.1 mm) and the depth origin 9 mm left of the mesh centre (d405_cam_depth_py). The depth
origin is the LEFT imager, which also gives the D405's colour image. The asset's origin is that point,
x forward, so a camera placed at the asset's origin, looking along +x, is the D405's colour/depth camera.
"""

import struct
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

DIR = Path(__file__).resolve().parent.parent / "robots_usd" / "RealSense"
GLASS_AHEAD = 0.0037 + 0.0001  # m, front plate ahead of the depth origin
DEPTH_PY = 0.009               # m, depth origin left of the mesh centre
MASS = 0.060                   # kg (datasheet)
BODY_RGB = (0.10, 0.10, 0.11)  # anodised dark grey


def read_stl(path):
    b = path.read_bytes()
    n = struct.unpack("<I", b[80:84])[0]
    assert 84 + n * 50 == len(b), "not a binary STL"
    dt = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    return np.frombuffer(b[84:84 + n * 50], dtype=dt)["v"].astype(np.float64)  # (n, 3, 3) mm


def main():
    tris = read_stl(DIR / "d405.stl") / 1000.0
    P = tris.reshape(-1, 3)
    P = np.stack([P[:, 2], P[:, 0], P[:, 1]], axis=1) + np.array([GLASS_AHEAD, -DEPTH_PY, 0.0])
    tris = P.reshape(-1, 3, 3)
    nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    uniq, idx = np.unique(np.round(P, 7), axis=0, return_inverse=True)

    out = DIR / "d405.usd"
    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/D405")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().SetCustomDataByKey("source", "realsense-ros realsense2_description/meshes/d405.stl (Apache-2.0)")
    root.GetPrim().SetCustomDataByKey("frame", "origin = left imager (colour + depth), x forward, y left, z up")
    root.GetPrim().SetCustomDataByKey("mass_kg", MASS)
    mesh = UsdGeom.Mesh.Define(stage, "/D405/body")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(uniq.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(tris)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(idx.astype(np.int32).ravel()))
    nv = mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.repeat(nrm, 3, axis=0).astype(np.float32)))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)  # flat CAD facets, not smoothed
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    lo, hi = uniq.min(0), uniq.max(0)
    mesh.CreateExtentAttr([Gf.Vec3f(*lo), Gf.Vec3f(*hi)])
    mat = UsdShade.Material.Define(stage, "/D405/Looks/body")
    sh = UsdShade.Shader.Define(stage, "/D405/Looks/body/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*BODY_RGB))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.6)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*BODY_RGB)])
    stage.GetRootLayer().Save()
    print(f"wrote {out}: {len(tris)} triangles, {len(uniq)} vertices; body x {lo[0]:.4f}..{hi[0]:.4f}, "
          f"y {lo[1]:.4f}..{hi[1]:.4f}, z {lo[2]:.4f}..{hi[2]:.4f} m (camera frame)")


if __name__ == "__main__":
    main()
