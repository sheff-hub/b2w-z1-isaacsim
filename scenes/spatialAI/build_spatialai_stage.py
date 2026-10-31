#!/usr/bin/env python
"""Merge the SpatialAI (Echo-3) splat and mesh of one room into ONE Isaac Sim stage: <scene>/spatialAI_<scene>.usdz.

One folder per scene next to this script (bedroom/, ...). Run in the 3DGRUT venv (pxr, torch, threedgrut):

    <paths.threedgrut>/.venv/bin/python scenes/spatialAI/build_spatialai_stage.py <scene folder>
    (paths.threedgrut in the repository's config.yaml, default repos/3dgrut; THREEDGRUT=<dir> overrides it)

Inputs, in the scene folder: the one *.ply there (3DGS PLY straight from SpatialAI) and the one *.usdc there (the
SpatialAI mesh, converted to USD in Blender). Outputs, also in it (<scene> = the folder name):
  spatialAI_<scene>.usdz            the stage: NuRec splat (what you see) + the mesh as an invisible collider
  spatialAI_<scene>_check.usda      references the .usdz with the mesh shown in its vertex colours (alignment check)
  spatialAI_<scene>_transform.json  4x4s from the two SpatialAI frames to the stage frame, and the fit numbers
  check/views.json                  cameras + drop points for isaac/run_render_views.sh

Frames, as measured on the bedroom capture (re-measured on every run; the run stops if they no longer hold):
- The splat is Y-up (glTF convention), in metres. Its origin is ~1.3 m above the floor: the phone at the start.
- The mesh is the splat turned 180 deg about X (y, z negated): in Blender's Z-up stage it is upside down, with
  height along -Y. If the mesh prim has an object transform (new_office: stood upright in Blender, +90 deg about
  X), it is baked into the points first, and "the mesh frame" below means the file's world frame. The rotation
  is found by trying all 24 axis rotations; ICP on top of it finds nothing left (scale 1.0004, 0.02 deg, 1 mm), so
  no ICP correction is applied.
- Which way is up was checked by rendering the splat both ways (+Y up shows the bed, door and pictures upright).
  A heuristic re-checks it: furniture tops face the camera, so the lower half of the room holds more horizontal
  area (bed, shelves) than the upper half. Every run also writes check/up_preview.jpg, a point render from the
  capture start, to look at. Where the heuristic is wrong (new_office: a lowered ceiling section outweighs its few
  desks), --up-checked overrides it.
Stage frame: Z-up, metres. Splat Y-up -> Z-up, levelled on the floor plane fitted to the mesh, turned about Z so the
walls run along X/Y (long side along X), floor at z = 0, and (x, y) = 0 at the most open floor spot (a spawn point).

Layout copies NVIDIA's NuRec stage.usdz (<paths.nurec>/nvidia_v0.0/*/stage.usdz):
  /World/volume/volume  NuRec Volume (3DGRUT export of the transformed gaussians), proxy -> /World/volume/mesh
  /World/volume/mesh    the SpatialAI mesh: invisible, matte, triangle-mesh collider, vertex colours kept
  /World/volume/floor   invisible box collider, top at z = 0, under the whole room
  /World/dome_light     intensity 1000, hidden from camera rays, lights robots/objects added to the scene
Floor: the floor is glossy, so mesh and splat both have reflection "bubbles" 10-30 cm under it. They are harmless
where the real floor surface sits above them, but where the floor was barely seen (a dark closet corner here) only
the bubble is left: a 20-40 cm pit. The floor box covers it; the mesh itself is left as SpatialAI made it.

Splat fixes: opacity exactly 1 is stored as logit +inf, and NuRec stores float16 pre-activation values, so it is
clamped to logit(0.9999). Opacity exactly 0 (logit -inf; 40 in new_office) adds nothing and is dropped. The PLY has
SH degree 0 only (no view dependence); higher bands are written as zeros. The build reads the packaged .nurec back
and fails if any value is not finite.
"""

import argparse
import gzip
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, Vt

REPO = Path(__file__).resolve().parents[2]  # scenes/spatialAI/<this> -> the repository


def _threedgrut() -> Path:
    """The 3DGRUT checkout: $THREEDGRUT, else paths.threedgrut of the repository's config.yaml (relative to it)."""
    if os.environ.get("THREEDGRUT"):
        return Path(os.environ["THREEDGRUT"]).expanduser()
    v = "repos/3dgrut"
    for name in ("config.yaml", "config.example.yaml"):
        cfg = REPO / name
        if cfg.exists():
            for line in cfg.read_text().splitlines():  # (no yaml in every venv: the one key, by hand)
                s = line.split("#")[0].strip()
                if s.startswith("threedgrut:"):
                    v = s.split(":", 1)[1].strip().strip("'\"") or v
            break
    q = Path(os.path.expanduser(v))
    return q if q.is_absolute() else REPO / q


GRUT = _threedgrut()
SPLAT_YUP_TO_ZUP = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])  # (x, y, z) -> (x, -z, y)
OPACITY_LOGIT_MAX = float(np.log(0.9999 / 0.0001))
GRID = 0.05  # m, floor/free-space maps
SAMPLES_PER_M2 = 2000  # mesh surface samples for the fit checks: ~2 cm apart, so the spacing does not inflate the gap


def only_file(folder, pattern):
    found = sorted(folder.glob(pattern))
    assert len(found) == 1, f"expected one {pattern} in {folder}, found {[f.name for f in found]}; or pass it"
    return found[0]


def read_ply(path):
    with open(path, "rb") as f:
        header = b""
        while not header.endswith(b"end_header\n"):
            header += f.readline()
        lines = header.decode().splitlines()
        assert "format binary_little_endian 1.0" in lines, "expected a binary little-endian PLY"
        props = [l.split()[-1] for l in lines if l.startswith("property")]
        assert all(l.split()[1] == "float" for l in lines if l.startswith("property")), "expected float32 properties"
        n = int(next(l for l in lines if l.startswith("element vertex")).split()[-1])
        data = np.fromfile(f, dtype=np.float32, count=n * len(props)).reshape(n, len(props))
    return {k: data[:, i] for i, k in enumerate(props)}, n


def read_mesh(path):
    st = Usd.Stage.Open(str(path))
    meshes = [p for p in st.Traverse() if p.IsA(UsdGeom.Mesh)]
    assert len(meshes) == 1, f"expected one mesh in {path}, found {[str(m.GetPath()) for m in meshes]}"
    prim = meshes[0]
    # an object transform set in Blender (e.g. the mesh stood upright) is baked in: V is in the file's world frame
    xf = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))  # row vectors
    assert np.allclose(xf[:, 3], [0, 0, 0, 1]), "projective mesh transform"
    m = UsdGeom.Mesh(prim)
    counts = np.array(m.GetFaceVertexCountsAttr().Get())
    assert (counts == 3).all(), "expected a triangle mesh"
    pv = UsdGeom.PrimvarsAPI(prim)
    col = pv.GetPrimvar("Col")
    normals = m.GetNormalsAttr().Get()
    return dict(
        V=np.array(m.GetPointsAttr().Get(), dtype=np.float64) @ xf[:3, :3] + xf[3, :3],
        F=np.array(m.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3),
        normals=None if normals is None else np.array(normals, dtype=np.float64) @ np.linalg.inv(xf[:3, :3]).T,
        xf=xf.T,  # column-vector convention, as in the transform JSON
        normals_interp=m.GetNormalsInterpolation(),
        col=None if not col else np.array(col.Get())[:, :3],
        col_interp=None if not col else col.GetInterpolation(),
        path=str(prim.GetPath()),
    )


def triangles(V, F):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(n, axis=1)
    return (a + b + c) / 3, n / np.maximum(2 * area[:, None], 1e-12), area


def sample_surface(V, F, k, rng):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    i = rng.choice(len(F), k, p=area / area.sum())
    u, v = rng.random((2, k))
    s = np.sqrt(u)
    return (1 - s)[:, None] * a[i] + (s * (1 - v))[:, None] * b[i] + (s * v)[:, None] * c[i]


def mesh_to_splat_rotation(mesh_pts, splat_pts):
    """Best of the 24 axis rotations R with mesh = R @ splat, by median nearest-neighbour distance."""
    tree = cKDTree(mesh_pts)
    best = []
    for perm in ([0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]):
        for signs in np.array(np.meshgrid([1, -1], [1, -1], [1, -1])).T.reshape(-1, 3):
            R = np.zeros((3, 3))
            R[range(3), perm] = signs
            if np.linalg.det(R) > 0:
                best.append((np.median(tree.query(splat_pts @ R.T)[0]), R))
    best.sort(key=lambda r: r[0])
    return best


def icp_residual(src, tree, dst):
    """Scaled ICP (trimmed 70 %) of src onto dst; returns how far it moved things (scale, deg, metres)."""
    s, R, t = 1.0, np.eye(3), np.zeros(3)
    for _ in range(30):
        d, i = tree.query(s * src @ R.T + t)
        k = d < max(np.percentile(d, 70), 0.01)
        A, B = src[k], dst[i[k]]
        ma, mb = A.mean(0), B.mean(0)
        U, S, Vt = np.linalg.svd((B - mb).T @ (A - ma) / len(A))
        D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
        R = U @ D @ Vt
        s = (S * np.diag(D)).sum() / ((A - ma) ** 2).sum(1).mean()
        t = mb - s * R @ ma
    return s, np.degrees(Rotation.from_matrix(R).magnitude()), np.linalg.norm(t)


def fit_floor(cen, nrm, area, rng):
    """Floor = biggest horizontal plane in the lower half; RANSAC then area-weighted SVD. Returns (normal, height)."""
    hz = np.abs(nrm[:, 2]) > 0.95
    z = cen[:, 2]
    lower = hz & (z < np.median(z))
    hist, edges = np.histogram(z[lower], bins=np.arange(z.min(), np.median(z), 0.02), weights=area[lower])
    z0 = edges[np.argmax(hist)] + 0.01
    band = hz & (np.abs(z - z0) < 0.15)
    P, w = cen[band], area[band]
    best_w, best_in = 0.0, None
    for _ in range(3000):
        p = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n = n / np.linalg.norm(n) * np.sign(n[2])
        if n[2] < 0.98:
            continue
        inl = np.abs((P - p[0]) @ n) < 0.015
        if w[inl].sum() > best_w:
            best_w, best_in = w[inl].sum(), inl
    Q, wq = P[best_in], w[best_in]
    mu = (Q * wq[:, None]).sum(0) / wq.sum()
    n = np.linalg.svd((Q - mu) * np.sqrt(wq[:, None]))[2][2]
    n *= np.sign(n[2])
    rms = np.sqrt((wq * ((Q - mu) @ n) ** 2).sum() / wq.sum())
    return n, float(n @ mu), float(wq.sum()), float(rms)


def wall_yaw(cen, nrm, area):
    """Dominant wall direction (deg, mod 90) from vertical faces, area-weighted, circularly smoothed."""
    v = np.abs(nrm[:, 2]) < 0.2
    ang = np.degrees(np.arctan2(nrm[v, 1], nrm[v, 0])) % 90
    hist, edges = np.histogram(ang, bins=360, range=(0, 90), weights=area[v])
    sm = ndimage.gaussian_filter1d(hist, 4, mode="wrap")  # sigma 1 deg
    i = int(np.argmax(sm))
    share = sm[[(i + k) % 360 for k in range(-8, 9)]].sum() / sm.sum()
    return float(edges[i] + 0.125), float(share)


def rot_z(deg):
    return Rotation.from_euler("z", deg, degrees=True).as_matrix()


def grid_maps(P, lo, shape):
    ix = ((P[:, 0] - lo[0]) / GRID).astype(int)
    iy = ((P[:, 1] - lo[1]) / GRID).astype(int)
    return ix.clip(0, shape[1] - 1), iy.clip(0, shape[0] - 1)


def up_preview(P, rgb, path, W=640, H=480):
    """Point render (90 deg, horizontal, 4 headings) of Z-up points from the origin (the capture start): the human
    check of which way is up. A ceiling on top and furniture standing on the floor means the frame is right."""
    views = []
    for yaw in np.radians([0, 90, 180, 270]):
        fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        right = np.cross(fwd, [0, 0, 1.0])
        zc = P @ fwd
        ok = zc > 0.2
        u = (W / 2 * (P[ok] @ right) / zc[ok] + W / 2).astype(int)
        v = (W / 2 * -P[ok, 2] / zc[ok] + H / 2).astype(int)
        inb = (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
        o = np.argsort(-zc[ok][inb])  # far first, so near points overwrite
        u, v, c = u[inb][o], v[inb][o], rgb[ok][inb][o]
        img = np.zeros((H, W, 3))
        for du, dv in ((0, 0), (0, 1), (1, 0), (1, 1)):  # 2x2 dots close most holes
            img[v + dv, u + du] = c
        views.append(img)
    from PIL import Image
    path.parent.mkdir(exist_ok=True)
    Image.fromarray((np.concatenate(views, 1) * 255).astype(np.uint8)).resize((2 * W, H // 2)).save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", type=Path, help="Scene folder (e.g. bedroom): inputs are read from it, outputs go in it.")
    ap.add_argument("--splat", type=Path, help="Splat PLY (default: the one *.ply in the scene folder).")
    ap.add_argument("--mesh", type=Path, help="Mesh USD (default: the one *.usdc in the scene folder).")
    ap.add_argument("--out", type=Path, help="Stage (default: <scene>/spatialAI_<scene>.usdz).")
    ap.add_argument("--no-yaw", action="store_true", help="Keep SpatialAI's heading instead of squaring the walls.")
    ap.add_argument("--keep-origin", action="store_true",
                    help="Keep SpatialAI's (x, y) origin (the capture start) instead of the most open floor spot.")
    ap.add_argument("--up-checked", action="store_true",
                    help="Build even if the furniture-top heuristic says upside down: check/up_preview.jpg was "
                         "looked at and is upright (new_office: a lowered ceiling section beats its few desks).")
    a = ap.parse_args()
    scene = a.scene.resolve()
    assert scene.is_dir(), f"no scene folder {scene}"
    a.splat = a.splat or only_file(scene, "*.ply")
    a.mesh = a.mesh or only_file(scene, "*.usdc")
    a.out = a.out or scene / f"spatialAI_{scene.name}.usdz"
    rng = np.random.default_rng(0)
    report = {"inputs": {"splat": a.splat.name, "mesh": a.mesh.name}}

    # ---- load -----------------------------------------------------------------------------------------------
    g, n = read_ply(a.splat)
    xyz = np.c_[g["x"], g["y"], g["z"]].astype(np.float64)
    opac = g["opacity"].astype(np.float64)
    others = ("scale_0", "scale_1", "scale_2", "f_dc_0", "f_dc_1", "f_dc_2", "rot_0", "rot_1", "rot_2", "rot_3")
    # dropped: NaN/inf outside opacity, and opacity logit -inf (alpha exactly 0, so they add nothing)
    dropped = ~np.isfinite(np.column_stack([xyz] + [g[k] for k in others])).all(1) | np.isnan(opac) | np.isneginf(opac)
    n_inf = int(np.isposinf(opac).sum())
    print(f"splat: {n:,} gaussians, {int(dropped.sum())} dropped (NaN/inf outside opacity, or opacity logit -inf), "
          f"{n_inf:,} with opacity logit +inf (clamped to {OPACITY_LOGIT_MAX:.2f})")
    alpha = 1 / (1 + np.exp(-np.clip(opac, -30, 30)))
    m = read_mesh(a.mesh)
    baked = not np.allclose(m["xf"], np.eye(4))
    print(f"mesh: {m['path']}  {len(m['V']):,} verts, {len(m['F']):,} tris"
          f"{'; its Blender object transform is baked in' if baked else ''}")
    if baked:
        report["mesh_object_transform_baked"] = m["xf"].round(6).tolist()

    # ---- mesh <-> splat rotation ------------------------------------------------------------------------------
    mesh_area = float(triangles(m["V"], m["F"])[2].sum())
    mesh_samples = sample_surface(m["V"], m["F"], max(200_000, int(mesh_area * SAMPLES_PER_M2)), rng)
    probe = xyz[rng.choice(np.where((alpha > 0.5) & ~dropped)[0], 100_000, replace=False)]
    ranked = mesh_to_splat_rotation(mesh_samples, probe)
    gap, R_ms = ranked[0]
    print(f"mesh = R @ splat with R = {R_ms.astype(int).tolist()}: median gap {gap * 100:.2f} cm "
          f"(next best {ranked[1][0] * 100:.1f} cm)")
    assert gap < 0.03 and ranked[1][0] > 2 * gap, "mesh and splat do not match under any axis rotation"
    s_icp, deg_icp, t_icp = icp_residual(probe @ R_ms.T, cKDTree(mesh_samples), mesh_samples)
    print(f"ICP on top of it: scale {s_icp:.4f}, {deg_icp:.3f} deg, {t_icp * 1000:.1f} mm (not applied)")
    assert abs(s_icp - 1) < 0.005 and deg_icp < 0.5 and t_icp < 0.02, "residual misalignment; add an ICP step"
    report["mesh_to_splat"] = {"rotation": R_ms.T.astype(int).tolist(), "median_gap_cm": round(gap * 100, 2),
                               "icp_residual": {"scale": round(s_icp, 5), "deg": round(deg_icp, 3),
                                                "mm": round(t_icp * 1000, 2)}}

    # ---- up, level, yaw, floor at 0 ----------------------------------------------------------------------------
    R = SPLAT_YUP_TO_ZUP @ R_ms.T  # mesh -> rough Z-up
    cen, nrm, area = triangles(m["V"] @ R.T, m["F"])
    z, zmid, hz = cen[:, 2], np.median(cen[:, 2]), np.abs(nrm[:, 2]) > 0.95
    h, e = np.histogram(z[hz], bins=np.arange(z.min(), z.max() + 0.02, 0.02), weights=area[hz])
    zc = (e[:-1] + e[1:]) / 2
    z_floor, z_ceil = zc[np.argmax(np.where(zc < zmid, h, 0))], zc[np.argmax(np.where(zc > zmid, h, 0))]
    lo_area = area[hz & (z > z_floor + 0.15) & (z < zmid)].sum()
    hi_area = area[hz & (z > zmid) & (z < z_ceil - 0.15)].sum()
    vis = np.where((alpha > 0.5) & ~dropped)[0]
    up_preview(xyz[vis] @ SPLAT_YUP_TO_ZUP.T, np.clip(0.5 + 0.28209479 * np.c_[g["f_dc_0"], g["f_dc_1"],
                                                                                g["f_dc_2"]][vis], 0, 1),
               a.out.parent / "check/up_preview.jpg")
    print(f"up check: horizontal area off floor/ceiling, lower half {lo_area:.1f} m^2 vs upper {hi_area:.1f} m^2; "
          f"look at check/up_preview.jpg")
    assert lo_area > hi_area or a.up_checked, (
        "furniture-top heuristic says the splat is upside down. Look at check/up_preview.jpg: if the ceiling is on "
        "top, rerun with --up-checked; if it is upside down, check SPLAT_YUP_TO_ZUP")
    report["up_check"] = {"lower_half_m2": round(float(lo_area), 1), "upper_half_m2": round(float(hi_area), 1),
                          "heuristic_overridden_after_preview": bool(lo_area <= hi_area)}

    fn, _, f_area, f_rms = fit_floor(cen, nrm, area, rng)
    tilt = float(np.degrees(np.arccos(fn[2])))
    R = Rotation.align_vectors([[0, 0, 1]], [fn])[0].as_matrix() @ R
    cen, nrm, area = triangles(m["V"] @ R.T, m["F"])
    yaw, share = wall_yaw(cen, nrm, area)
    if a.no_yaw:
        yaw = 0.0
    R = rot_z(-yaw) @ R
    cen, nrm, area = triangles(m["V"] @ R.T, m["F"])
    fn2, fz, _, _ = fit_floor(cen, nrm, area, rng)
    floorish = np.abs(cen[:, 2] - fz) < 0.05
    ext = np.ptp(cen[floorish, :2], axis=0)
    if ext[1] > ext[0]:  # long side along X
        R = rot_z(90) @ R
        yaw -= 90  # total turn is -yaw
    print(f"floor: tilt {tilt:.2f} deg, {f_area:.1f} m^2 of inliers, rms {f_rms * 100:.2f} cm; walls turned by "
          f"{-yaw:.2f} deg (peak holds {share:.0%} of wall area within +-2 deg)")

    Vs = m["V"] @ R.T
    Vs[:, 2] -= fz
    t = np.array([0.0, 0.0, -fz])

    # ---- maps on the floor: room footprint, obstacles, pits, most open spot -----------------------------------
    P = sample_surface(Vs, m["F"], max(3_000_000, int(mesh_area * 2 * SAMPLES_PER_M2)), rng)
    lo = P[:, :2].min(0) - 0.2
    shape = tuple((np.ceil((P[:, :2].max(0) + 0.2 - lo) / GRID).astype(int))[::-1])
    ix, iy = grid_maps(P, lo, shape)
    floor = np.zeros(shape, bool)
    floor[iy[np.abs(P[:, 2]) < 0.05], ix[np.abs(P[:, 2]) < 0.05]] = True
    obst = np.zeros(shape, bool)
    clear = np.full(shape, 1.9)  # lowest thing above the floor: the headroom over a cell
    k = (P[:, 2] > 0.06) & (P[:, 2] < 1.9)
    obst[iy[k], ix[k]] = True
    np.minimum.at(clear, (iy[k], ix[k]), P[k, 2])
    room = ndimage.binary_fill_holes(ndimage.binary_closing(floor | obst, iterations=2))  # inside the walls
    top = np.full(shape, -9.0)  # highest surface below 12 cm: where a wheel or foot would rest
    k = P[:, 2] < 0.12
    np.maximum.at(top, (iy[k], ix[k]), P[k, 2])
    pits = room & (top > -9) & (top < -0.10)
    lab, npits = ndimage.label(pits)
    sizes = ndimage.sum(pits, lab, range(1, npits + 1)) * GRID ** 2 if npits else np.array([])
    # spawn/drop candidates: floor that was reconstructed. Hole filling also fills rooms nobody walked into (new_office:
    # a 78 m^2 block inside a corridor loop), which has no floor but the box, so it must not count as open floor.
    seen = ndimage.binary_closing(floor, iterations=2) & room
    free = seen & ~obst
    dist = ndimage.distance_transform_edt(free) * GRID
    oy, ox = np.unravel_index(np.argmax(dist), shape)
    open_xy = lo + (np.array([ox, oy]) + 0.5) * GRID
    if a.keep_origin:
        open_xy_shift = np.zeros(2)
    else:
        open_xy_shift = open_xy
    t[:2] -= open_xy_shift
    Vs[:, :2] -= open_xy_shift
    P[:, :2] -= open_xy_shift
    lo = lo - open_xy_shift
    room_xy = np.argwhere(room)[:, ::-1] * GRID + lo
    room_lo, room_hi = room_xy.min(0), room_xy.max(0) + GRID
    ceiling = float(np.percentile(P[:, 2], 99.5))
    print(f"room footprint {room.sum() * GRID ** 2:.1f} m^2 ({seen.sum() * GRID ** 2:.1f} m^2 of it floor seen), "
          f"box {room_lo.round(2)} .. {room_hi.round(2)}, "
          f"ceiling ~{ceiling:.2f} m; most open spot {open_xy.round(2)} with {dist.max():.2f} m clearance"
          f"{' -> origin' if not a.keep_origin else ''}")
    big = np.argsort(sizes)[::-1][:3]
    pit_info = []
    for i in big:
        yy, xx = np.nonzero(lab == i + 1)
        c = lo + (np.array([xx.mean(), yy.mean()]) + 0.5) * GRID
        j = np.argmax(clear[yy, xx])  # drop point: the pit cell with the most headroom
        drop = lo + (np.array([xx[j], yy[j]]) + 0.5) * GRID
        pit_info.append({"centre_xy": c.round(3).tolist(), "area_m2": round(float(sizes[i]), 2),
                         "max_depth_m": round(float(-top[yy, xx].min()), 2),
                         "drop_xyz": [*drop.round(3).tolist(), round(float(np.clip(clear[yy[j], xx[j]] - 0.15,
                                                                                    0.3, 1.0)), 2)]})
    print(f"floor pits (top surface >10 cm below the floor): {len(sizes)}, {sizes.sum():.2f} m^2 total; "
          f"largest: {pit_info}")

    T_splat = np.eye(4)
    T_splat[:3, :3], T_splat[:3, 3] = R @ R_ms, t  # R was built from the mesh frame; splat -> mesh is R_ms
    T_mesh = np.eye(4)
    T_mesh[:3, :3], T_mesh[:3, 3] = R, t
    assert np.allclose(m["V"] @ T_mesh[:3, :3].T + T_mesh[:3, 3], Vs)

    # ---- gaussians into the stage frame ----------------------------------------------------------------------
    keep = ~dropped
    pos = xyz[keep] @ T_splat[:3, :3].T + T_splat[:3, 3]
    q = np.c_[g["rot_0"], g["rot_1"], g["rot_2"], g["rot_3"]][keep].astype(np.float64)  # w, x, y, z
    q = (Rotation.from_matrix(T_splat[:3, :3]) * Rotation.from_quat(q[:, [1, 2, 3, 0]])).as_quat()[:, [3, 0, 1, 2]]
    scales = np.c_[g["scale_0"], g["scale_1"], g["scale_2"]][keep]
    dens = np.minimum(opac[keep], OPACITY_LOGIT_MAX)
    albedo = np.c_[g["f_dc_0"], g["f_dc_1"], g["f_dc_2"]][keep]
    d_chk, _ = cKDTree(P).query(pos[rng.choice(np.where(alpha[keep] > 0.5)[0], 100_000, replace=False)])
    print(f"splat vs mesh in the stage frame: median {np.median(d_chk) * 100:.2f} cm, p90 "
          f"{np.percentile(d_chk, 90) * 100:.2f} cm")
    report["splat_vs_mesh_cm"] = {"median": round(float(np.median(d_chk) * 100), 2),
                                  "p90": round(float(np.percentile(d_chk, 90) * 100), 2)}

    # 3DGUT JIT-compiles its kernels with slangc from the venv's bin/, which is not on PATH unless the venv is activated
    os.environ["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"
    import torch
    from hydra import compose, initialize_config_dir
    from threedgrut.export import NuRecExporter
    from threedgrut.model.model import MixtureOfGaussians

    with initialize_config_dir(version_base=None, config_dir=str(GRUT / "configs")):
        conf = compose(config_name="apps/colmap_3dgut.yaml")
    model = MixtureOfGaussians(conf)
    dev = model.device
    as_param = lambda x: torch.nn.Parameter(torch.tensor(np.asarray(x), dtype=torch.float32, device=dev))
    model.positions, model.rotation, model.scale = as_param(pos), as_param(q), as_param(scales)
    model.density, model.features_albedo = as_param(dens[:, None]), as_param(albedo)
    model.features_specular = as_param(np.zeros((len(pos), 3 * ((model.max_n_features + 1) ** 2 - 1))))
    model.n_active_features = model.max_n_features

    out = a.out.resolve()
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".build_") as tmp:
        tmp = Path(tmp)
        NuRecExporter(export_cameras=False).export(model, tmp / "volume.usdz", dataset=None, conf=conf)
        del model
        torch.cuda.empty_cache()
        with zipfile.ZipFile(tmp / "volume.usdz") as z:
            names = z.namelist()
            gauss_name = next(x for x in names if x.endswith(".usda") and x != "default.usda")
            nurec_name = next(x for x in names if x.endswith(".nurec"))
            z.extract(gauss_name, tmp)
            # 3DGRUT writes the gzip stream with compresslevel=0; the half of it that is zero SH bands shrinks
            # to nothing at level 6 (NVIDIA's own .nurec files are compressed too)
            (tmp / nurec_name).write_bytes(gzip.compress(gzip.decompress(z.read(nurec_name)), 6, mtime=0))
        (tmp / gauss_name).rename(tmp / "volume.usda")
        vol_layer = Sdf.Layer.FindOrOpen(str(tmp / "volume.usda"))
        vol_src = next(iter(vol_layer.GetPrimAtPath("/World").nameChildren))
        render_settings = dict(vol_layer.customLayerData)

        # mesh layer: geometry in the stage frame, vertex colours, collider, invisible, matte
        ms = Usd.Stage.CreateNew(str(tmp / "mesh.usdc"))
        UsdGeom.SetStageUpAxis(ms, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(ms, 1.0)
        mesh = UsdGeom.Mesh.Define(ms, "/mesh")
        ms.SetDefaultPrim(mesh.GetPrim())
        mesh.GetPrim().SetDocumentation(f"SpatialAI mesh ({a.mesh.name}, {m['path']}), rigidly moved into the stage "
                                        f"frame; see {out.stem}_transform.json")
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(Vs.astype(np.float32)))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(m["F"]), 3, dtype=np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(m["F"].astype(np.int32).ravel()))
        mesh.CreateExtentAttr([Gf.Vec3f(*Vs.min(0)), Gf.Vec3f(*Vs.max(0))])
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDoubleSidedAttr(True)
        if m["normals"] is not None:
            mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy((m["normals"] @ R.T).astype(np.float32)))
            mesh.SetNormalsInterpolation(m["normals_interp"])
        if m["col"] is not None:  # linear rec709, as Blender wrote it
            mesh.CreateDisplayColorPrimvar(m["col_interp"]).Set(Vt.Vec3fArray.FromNumpy(m["col"].astype(np.float32)))
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.none)
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("isMatteObject", Sdf.ValueTypeNames.Bool).Set(True)
        mesh.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        ms.GetRootLayer().Save()

        # root layer
        st = Usd.Stage.CreateNew(str(tmp / "default.usda"))
        st.GetRootLayer().customLayerData = render_settings
        st.GetRootLayer().documentation = (f"SpatialAI (Echo-3) {scene.name}: NuRec splat + SpatialAI mesh collider. "
                                           f"Built by {Path(__file__).name}.")
        UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(st, 1.0)
        world = UsdGeom.Xform.Define(st, "/World")
        st.SetDefaultPrim(world.GetPrim())
        UsdGeom.Xform.Define(st, "/World/volume")
        vol = st.DefinePrim("/World/volume/volume")
        vol.GetReferences().AddReference("./volume.usda", vol_src.path)
        vol.CreateRelationship("proxy", custom=True).SetTargets([Sdf.Path("/World/volume/mesh")])
        st.DefinePrim("/World/volume/mesh").GetReferences().AddReference("./mesh.usdc")
        fl = UsdGeom.Cube.Define(st, "/World/volume/floor")
        fl.CreateSizeAttr(1.0)
        c = (room_lo + room_hi) / 2
        fl.AddTranslateOp().Set(Gf.Vec3d(float(c[0]), float(c[1]), -0.05))
        fl.AddScaleOp().Set(Gf.Vec3f(*(room_hi - room_lo + 0.4).tolist(), 0.1))
        fl.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        UsdPhysics.CollisionAPI.Apply(fl.GetPrim())
        fl.GetPrim().SetDocumentation("Flat floor at z = 0 under the whole room: covers the pits where the glossy "
                                      "floor was reconstructed as a reflection below the real floor.")
        dome = UsdLux.DomeLight.Define(st, "/World/dome_light")
        dome.CreateIntensityAttr(1000.0)
        dome.GetPrim().CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool, custom=True).Set(False)
        st.GetRootLayer().Save()

        out.unlink(missing_ok=True)
        w = Sdf.ZipFileWriter.CreateNew(str(out))
        for f in ("default.usda", "volume.usda", nurec_name, "mesh.usdc"):  # root layer first
            w.AddFile(str(tmp / f), f)
        w.Save()
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")

    # ---- check layer: mesh shown in its vertex colours, not a proxy, not matte ---------------------------------
    chk = out.with_name(out.stem + "_check.usda")
    chk.unlink(missing_ok=True)
    cs = Usd.Stage.CreateNew(str(chk))
    cs.GetRootLayer().customLayerData = render_settings
    cs.GetRootLayer().documentation = ("Alignment check: the mesh is visible in its vertex colours. Toggle "
                                       "/World/volume/volume (the splat) to compare.")
    UsdGeom.SetStageUpAxis(cs, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(cs, 1.0)
    w = cs.DefinePrim("/World")
    w.GetReferences().AddReference(f"./{out.name}")
    cs.SetDefaultPrim(w)
    cs.OverridePrim("/World/volume/volume").CreateRelationship("proxy", custom=True).SetTargets([])
    cm = cs.OverridePrim("/World/volume/mesh")
    cm.CreateAttribute("visibility", Sdf.ValueTypeNames.Token).Set(UsdGeom.Tokens.inherited)
    cm.CreateAttribute("primvars:isMatteObject", Sdf.ValueTypeNames.Bool).Set(False)
    cs.GetRootLayer().Save()
    print(f"wrote {chk}")

    # ---- verify the package ----------------------------------------------------------------------------------
    vs = Usd.Stage.Open(str(out))
    errs = vs.GetCompositionErrors() if hasattr(vs, "GetCompositionErrors") else []
    vp = vs.GetPrimAtPath("/World/volume/volume")
    mp = vs.GetPrimAtPath("/World/volume/mesh")
    fp = vp.GetChildren()[0].GetAttribute("filePath").Get()
    assert vp.GetTypeName() == "Volume" and mp.GetTypeName() == "Mesh", (vp.GetTypeName(), mp.GetTypeName())
    assert not errs, errs
    assert fp.resolvedPath, f"NuRec payload does not resolve: {fp}"
    assert [str(p) for p in vp.GetRelationship("proxy").GetTargets()] == ["/World/volume/mesh"]
    vxf = np.array(UsdGeom.Xformable(vp).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    assert np.allclose(vxf, np.eye(4)), f"volume transform is not identity:\n{vxf}"
    import msgpack
    with zipfile.ZipFile(out) as z:
        sd = msgpack.unpackb(gzip.decompress(z.read(nurec_name)))["nre_data"]["state_dict"]
    for k in ("positions", "rotations", "scales", "densities", "features_albedo"):  # float16, as NuRec stores them
        assert np.isfinite(np.frombuffer(sd[f".gaussians_nodes.gaussians.{k}"], np.float16)).all(), \
            f"non-finite {k} in the packaged .nurec"
    del sd
    print(f"verified: {len(list(vs.Traverse()))} prims, no composition errors, payload resolves to "
          f"{fp.resolvedPath}, proxy -> /World/volume/mesh, volume transform identity, every splat value in the "
          f"packaged .nurec finite")

    # ---- report + Isaac check views ------------------------------------------------------------------------
    report.update({
        "stage_frame": "Z-up, metres, floor at z=0, walls along X/Y, origin at the most open floor spot",
        "splat_to_stage": T_splat.round(6).tolist(),
        "mesh_to_stage": T_mesh.round(6).tolist(),
        "matrix_convention": "column vectors: p_stage = M @ [p, 1]",
        "floor": {"tilt_deg": round(tilt, 3), "fit_rms_cm": round(f_rms * 100, 2), "inlier_area_m2": round(f_area, 1)},
        "walls_turned_by_deg": round(-yaw, 2),
        "origin": {"most_open_spot_clearance_m": round(float(dist.max()), 2),
                   "spatialai_origin_in_stage": T_splat[:3, 3].round(3).tolist()},
        "room": {"footprint_m2": round(float(room.sum() * GRID ** 2), 1),
                 "floor_seen_m2": round(float(seen.sum() * GRID ** 2), 1),
                 "box_xy": [room_lo.round(2).tolist(), room_hi.round(2).tolist()], "ceiling_m": round(ceiling, 2)},
        "floor_pits": pit_info,
        "splat": {"gaussians": int(keep.sum()), "dropped": int(dropped.sum()), "opacity_clamped": n_inf},
        "outputs": {"stage": out.name, "check": chk.name},
    })
    out.with_name(out.stem + "_transform.json").write_text(json.dumps(report, indent=2))

    # views for isaac/run_render_views.sh: from the open spot and from the room corners, 1.3 m high
    def c2w(eye, look):
        f = np.asarray(look, float) - eye
        f /= np.linalg.norm(f)
        r = np.cross(f, [0, 0, 1.0])
        r /= np.linalg.norm(r)
        M = np.eye(4)
        M[:3, :3] = np.c_[r, np.cross(f, r), f]  # OpenCV: x right, y down, z forward
        M[:3, 3] = eye
        return M.round(6).tolist()

    W_, H_, fov = 960, 720, 90.0
    fx = W_ / 2 / np.tan(np.radians(fov / 2))
    eye0 = np.array([0.0, 0.0, 1.3]) if not a.keep_origin else np.r_[open_xy, 1.3]
    views = [{"name": f"open_yaw{d:03d}", "c2w": c2w(eye0, eye0 + [np.cos(np.radians(d)), np.sin(np.radians(d)),
                                                                     -0.35]), "fx": fx} for d in (0, 90, 180, 270)]
    free_xy = np.argwhere(dist > 0.35)[:, ::-1] * GRID + lo + GRID / 2
    for j, corner in enumerate(([room_lo[0], room_lo[1]], [room_hi[0], room_hi[1]])):
        e = free_xy[np.argmin(np.linalg.norm(free_xy - corner, axis=1))]
        views.append({"name": f"corner{j}", "c2w": c2w(np.r_[e, 1.6], [eye0[0], eye0[1], 0.4]), "fx": fx})
    pit_drops = [p["drop_xyz"] for p in pit_info[:2]]
    spread = [eye0[:2]]  # the rest of the 6 drops: farthest-point samples of the free floor, so big rooms are covered
    for _ in range(5 - len(pit_drops)):
        spread.append(free_xy[np.argmax(np.min([np.linalg.norm(free_xy - s, axis=1) for s in spread], axis=0))])
    drops = [[*eye0[:2], 1.0]] + pit_drops + [[*np.round(s, 3).tolist(), 1.0] for s in spread[1:]]
    (out.parent / "check").mkdir(exist_ok=True)
    (out.parent / "check/views.json").write_text(json.dumps({"width": W_, "height": H_, "views": views, "drops": drops}))
    print(f"wrote {out.with_name(out.stem + '_transform.json').name}, check/views.json ({len(views)} views, "
          f"{len(drops)} drops: the open spot, {len(pit_drops)} into floor pits, the rest spread over the free floor)")


if __name__ == "__main__":
    main()
