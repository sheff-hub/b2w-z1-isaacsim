#!/usr/bin/env python3
"""A hinged door with a lever handle, built from the hospital's own door (SM_Door_01): an articulation for Isaac Lab.

    python3 tools/build_door_asset.py                      # downloads the prop, writes scenes/assets/door_hinged.usda
    python3 tools/build_door_asset.py --prop SM_Door_01.usd --out other.usda

The prop is one mesh in centimetres: the leaf, a lever handle on each side and a backplate behind each lever. Split
into its connected parts (shared corners welded), they become three links:

    frame   the articulation's root; Isaac Lab fixes it to the world where the door is spawned (fix_root_link)
    leaf    the leaf and both backplates, a box collider; on `hinge`, the vertical axis at the leaf's hinge edge on
            its +x face, so that opening (+angle) swings it towards +x, away from the -x side
    handle  both levers on one spindle; on `handle`, the axis through the door (x) at the levers' neck; pressed
            down is a negative angle

Frame: the prop instance's (hospital.usd SM_Door_01_48: x through the door, the leaf on its -y side of the hinge
edge at y = 0, z up, floor at 0) with the instance's scale (1, 1.0301, 1.0489) applied, in metres. The material is
the prop's own (MI_Door_01, by reference, so the textures load from NVIDIA's server); its ColorAlbedo input paints
most of the door and is what the task sets per door (dark teal as shipped).

Gains, limits of drive and the latch are the task's (fetch_env_cfg.py, fetch_box.py); here only the joint ranges.
"""
import argparse
import os
import tempfile
import urllib.request

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

HOSPITAL = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/Hospital"
PROP_URL = f"{HOSPITAL}/Props/SM_Door_01.usd"
SCALE = np.array([1.0, 1.0301350355148315, 1.0489460229873657]) * 0.01  # instance scale x the prop's cm -> m
HINGE_RANGE = (0.0, 100.0)    # deg: shut .. wide open, into the room
HANDLE_RANGE = (-45.0, 0.0)   # deg: pressed down .. at rest
LEAF_MASS, HANDLE_MASS = 15.0, 0.2  # kg
COLLIDER_INSET = 0.015  # m: the leaf's box collider is this much smaller than the leaf at each edge (not its faces)
COLLIDER_SILL = 0.05    # m: ...and starts this high: the hospital's doorways have a ~3 cm threshold (Geo_M_DoorFloor)


def components(points, counts, indices):
    """Connected parts of a mesh: faces sharing a vertex, and vertices at the same position, belong together."""
    par = list(range(len(points)))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    faces, k = [], 0
    for c in counts:
        f = list(indices[k:k + c])
        k += c
        faces.append(f)
        for v in f[1:]:
            par[find(v)] = find(f[0])
    seen = {}
    for i, q in enumerate(np.round(points, 3)):
        t = tuple(q)
        if t in seen:
            par[find(i)] = find(seen[t])
        else:
            seen[t] = i
    groups = {}
    for fi, f in enumerate(faces):
        groups.setdefault(find(f[0]), []).append(fi)
    return [sorted(fs) for fs in groups.values()], faces


def write_mesh(stage, path, faces, P, N, ST, collider=False):
    """A mesh of the given faces (indices into P, N, ST), re-indexed."""
    used = sorted({v for f in faces for v in f})
    remap = {v: i for i, v in enumerate(used)}
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, P[v])) for v in used]))
    m.CreateFaceVertexCountsAttr(Vt.IntArray([len(f) for f in faces]))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray([remap[v] for f in faces for v in f]))
    m.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, N[v])) for v in used]))
    m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    st = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set(Vt.Vec2fArray([Gf.Vec2f(*map(float, ST[v])) for v in used]))
    pts = P[used]
    m.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, pts.min(0))), Gf.Vec3f(*map(float, pts.max(0)))]))
    if collider:
        UsdPhysics.CollisionAPI.Apply(m.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr("convexDecomposition")
    return m, pts


def body(stage, path, mass, inertia=None):
    x = UsdGeom.Xform.Define(stage, path)
    UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
    ma = UsdPhysics.MassAPI.Apply(x.GetPrim())
    ma.CreateMassAttr(mass)
    if inertia is not None:  # a link without colliders needs its inertia given
        ma.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
        ma.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))
    return x


def revolute(stage, path, b0, b1, axis, anchor, limits):
    j = UsdPhysics.RevoluteJoint.Define(stage, path)
    j.CreateBody0Rel().SetTargets([b0])
    j.CreateBody1Rel().SetTargets([b1])
    j.CreateAxisAttr(axis)
    a = Gf.Vec3f(*map(float, anchor))  # all links sit at the door's origin: the anchor is the same in both
    j.CreateLocalPos0Attr(a)
    j.CreateLocalPos1Attr(a)
    j.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
    j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    j.CreateLowerLimitAttr(limits[0])
    j.CreateUpperLimitAttr(limits[1])
    UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")  # gains set by the task (Isaac Lab actuators)
    return j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prop", default="", help="local SM_Door_01.usd (default: downloaded)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "scenes", "assets", "door_hinged.usda"))
    a = ap.parse_args()
    prop = a.prop
    if not prop:
        prop = os.path.join(tempfile.mkdtemp(), "SM_Door_01.usd")
        urllib.request.urlretrieve(PROP_URL, prop)
    src = Usd.Stage.Open(prop)
    sm = UsdGeom.Mesh(src.GetPrimAtPath("/Root/SM_Door_01"))
    P = np.array(sm.GetPointsAttr().Get(), dtype=float) * SCALE
    N = np.array(sm.GetNormalsAttr().Get(), dtype=float)
    ST = np.array(UsdGeom.PrimvarsAPI(sm).GetPrimvar("st").Get(), dtype=float)
    counts, indices = list(sm.GetFaceVertexCountsAttr().Get()), list(sm.GetFaceVertexIndicesAttr().Get())
    parts, faces = components(P, counts, indices)

    def bounds(fs):
        v = sorted({i for fi in fs for i in faces[fi]})
        return P[v].min(0), P[v].max(0)

    # the leaf spans the whole height; levers stick out ~9 cm and are ~4 cm tall; backplates are the thin rest
    leaf = max(parts, key=lambda fs: bounds(fs)[1][2] - bounds(fs)[0][2])
    rest = [fs for fs in parts if fs is not leaf]
    lmin, lmax = bounds(leaf)
    levers = [fs for fs in rest if bounds(fs)[1][0] - bounds(fs)[0][0] > 0.05]
    plates = [fs for fs in rest if fs not in levers]
    assert len(levers) == 2 and len(plates) == 2, (len(levers), len(plates))

    # the spindle: through the levers' necks, where each lever meets its backplate (the lever's vertices nearest
    # the leaf), along x
    necks = []
    for fs in levers:
        v = np.array(sorted({i for fi in fs for i in faces[fi]}))
        d = np.minimum(np.abs(P[v, 0] - lmin[0]), np.abs(P[v, 0] - lmax[0]))
        necks.append(P[v[d < d.min() + 0.015]])
    neck = np.vstack(necks)
    pivot = np.array([(lmin[0] + lmax[0]) / 2, neck[:, 1].mean(), neck[:, 2].mean()])
    hinge = np.array([lmax[0], lmax[1], 0.0])  # +x face, hinge edge (y max: the leaf lies on -y of it)

    stage = Usd.Stage.CreateNew(a.out) if not os.path.exists(a.out) else Usd.Stage.Open(a.out)
    stage.GetRootLayer().Clear()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Door")
    stage.SetDefaultPrim(root.GetPrim())
    stage.GetRootLayer().documentation = (
        "Hinged door with a lever handle from hospital.usd's SM_Door_01 (tools/build_door_asset.py). Links frame / leaf "
        f"/ handle; joints hinge (z at {np.round(hinge, 4).tolist()}, {HINGE_RANGE} deg) and handle (x at "
        f"{np.round(pivot, 4).tolist()}, {HANDLE_RANGE} deg). Leaf {np.round(lmax - lmin, 3).tolist()} m.")

    looks = UsdGeom.Scope.Define(stage, "/Door/Looks")
    paint = stage.DefinePrim("/Door/Looks/Paint", "Material")
    paint.GetReferences().AddReference(PROP_URL, "/Root/Looks/MI_Door_01")
    mat = UsdShade.Material(paint)

    body(stage, "/Door/frame", 1.0, inertia=(0.1, 0.1, 0.1))
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath("/Door/frame"))  # Isaac Lab's fix_root_link wants it here

    body(stage, "/Door/leaf", LEAF_MASS)
    m, _ = write_mesh(stage, "/Door/leaf/visual", [faces[fi] for fi in leaf], P, N, ST)
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
    for i, fs in enumerate(plates):
        m, _ = write_mesh(stage, f"/Door/leaf/backplate{i}", [faces[fi] for fi in fs], P, N, ST)
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
    box = UsdGeom.Cube.Define(stage, "/Door/leaf/collider")  # inset from the edges: flush with the jambs and the
    box.CreateSizeAttr(1.0)                                    # frame, the leaf was held by their contacts
    lo = lmin + np.array([0.0, COLLIDER_INSET, COLLIDER_SILL])  # and clear of the doorway's raised threshold (3 cm)
    hi = lmax - np.array([0.0, COLLIDER_INSET, COLLIDER_INSET])
    box.AddTranslateOp().Set(Gf.Vec3d(*map(float, (lo + hi) / 2)))
    box.AddScaleOp().Set(Gf.Vec3f(*map(float, hi - lo)))
    box.CreatePurposeAttr(UsdGeom.Tokens.guide)
    UsdPhysics.CollisionAPI.Apply(box.GetPrim())

    body(stage, "/Door/handle", HANDLE_MASS)
    for i, fs in enumerate(levers):
        m, _ = write_mesh(stage, f"/Door/handle/lever{i}", [faces[fi] for fi in fs], P, N, ST, collider=True)
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)

    revolute(stage, "/Door/hinge", "/Door/frame", "/Door/leaf", "Z", hinge, HINGE_RANGE)
    revolute(stage, "/Door/handle_joint", "/Door/leaf", "/Door/handle", "X", pivot, HANDLE_RANGE)
    stage.GetRootLayer().Save()
    lv = [bounds(fs) for fs in levers]
    print(f"wrote {a.out}")
    print(f"  leaf {np.round(lmin, 3)} .. {np.round(lmax, 3)}")
    for i, (lo, hi) in enumerate(lv):
        print(f"  lever{i} {np.round(lo, 3)} .. {np.round(hi, 3)}")
    print(f"  hinge axis z at {np.round(hinge, 4)}; handle axis x at {np.round(pivot, 4)}")


if __name__ == "__main__":
    main()
