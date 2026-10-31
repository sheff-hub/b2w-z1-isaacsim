"""Score model_city collision meshes of the IMG_9206 corridor, and pack them for the viewer.

    python corridor_eval.py <repo> <viewer_out> name=path/to/mesh.ply [name=...]

Frame: nerfstudio's model frame (what ns-export writes splat.ply in): transforms.json poses
and sparse_pc.ply mapped through dataparser_transforms.json. Up = path axis x wall normal.
Floor = lowest SfM height layer holding >=15% of the busiest layer's points.
Metres: MPU_EST, an estimate from three cues (camera height, door-handle layer, ceiling) that
agree to +-10%. Measure one real distance to replace it.
"""
import base64
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from plyfile import PlyData
from scipy.spatial import cKDTree

MPU_EST = 5.4  # m per splat unit, ESTIMATE: camera 1.6 m -> 4.8, door-handle layer 1.05 m -> 5.8, ceiling 2.6 m -> 5.5
VIEW_TRIS = 200_000

repo, vout = Path(sys.argv[1]), Path(sys.argv[2])
meshes = dict(a.split("=", 1) for a in sys.argv[3:])
vout.mkdir(parents=True, exist_ok=True)
proc = repo / "data/processed"
dp = json.loads(next((proc / "training_output").rglob("dataparser_transforms.json")).read_text())
tj = json.loads((proc / "transforms.json").read_text())
# dataparser_transforms.json maps the ORIGINAL COLMAP frame (its transform already includes
# transforms.json's applied_transform), so undo applied_transform first.
A = np.r_[np.array(tj["applied_transform"]), [[0, 0, 0, 1]]]
T = (np.r_[np.array(dp["transform"]), [[0, 0, 0, 1]]] @ np.linalg.inv(A))[:3]
s = dp["scale"]
to_model = lambda p: (p @ T[:, :3].T + T[:, 3]) * s

c2w = np.array([f["transform_matrix"] for f in tj["frames"]])[:, :3]
cams = to_model(c2w[:, :, 3])
R = np.einsum("ij,njk->nik", T[:, :3], c2w[:, :, :3])
up_cam = R[:, :, 1].mean(0); up_cam /= np.linalg.norm(up_cam)
sv = PlyData.read(str(proc / "sparse_pc.ply"))["vertex"]
sfm = to_model(np.c_[sv["x"], sv["y"], sv["z"]])

# Up from the scene, not the cameras: the phone was pitched down the whole walk, so the mean
# image-up (= nerfstudio's +Z) leans ~15 deg forward. Corridor walls are vertical and parallel
# to the path, so up = path axis x dominant wall normal, signed like the cameras' up.
o3d.utility.random.seed(0)
c = cams.mean(0)
axis = np.linalg.svd(cams - c)[2][0]
pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sfm)); n_wall, best = None, 0
for _ in range(6):
    if len(pcd.points) < 300: break
    (a_, b_, c_, _d), inl = pcd.segment_plane(0.004, 3, 4000)
    n = np.array([a_, b_, c_])
    if abs(n @ axis) < 0.1 and abs(n @ up_cam) < 0.5 and len(inl) > best: n_wall, best = n, len(inl)
    pcd = pcd.select_by_index(inl, invert=True)
up = np.cross(axis, n_wall); up /= np.linalg.norm(up); up *= np.sign(up @ up_cam)
print(f"up from walls ({best} SfM pts on the main wall): {np.degrees(np.arccos(up @ up_cam)):.1f} deg from the mean camera up "
      f"(= nerfstudio +Z, {np.degrees(np.arccos(abs(up[2]))):.1f} deg)")
e1 = axis - (axis @ up) * up; e1 /= np.linalg.norm(e1); e2 = np.cross(up, e1)
Pxy = lambda p: np.c_[(p - c) @ e1, (p - c) @ e2]
H = lambda p: (p - c) @ up
cp = Pxy(cams)
# drop the few mis-registered cameras for the footprint
med = np.median(cp, 0); dev = np.linalg.norm(cp - med, axis=1)
good = dev < np.percentile(dev, 99) * 1.0 + 1e-9
good &= np.abs(H(cams) - np.median(H(cams))) < 3 * np.std(H(cams))
lo, hi = cp[good].min(0), cp[good].max(0)
span = np.ptp(cp[good], 0).max()
margin = 0.25 * span
inb = np.all((Pxy(sfm) > lo - margin) & (Pxy(sfm) < hi + margin), 1)
h = H(sfm[inb])
hist, ed = np.histogram(h, bins=np.linspace(np.percentile(h, 0.5), np.percentile(h, 99.5), 120))
floor = ed[np.argmax(hist >= 0.15 * hist.max())]
cam_h = np.median(H(cams[good])) - floor
mpu = MPU_EST
u = lambda m: m / mpu
print(f"floor {floor:+.3f} u; at {mpu} m/unit the camera is {cam_h * mpu:.2f} m up (median)")
# corridor width cross-check: SfM points 0.5-1.5 m above the floor, |across| distribution
band = (h - floor > u(0.5)) & (h - floor < u(1.5))
across = Pxy(sfm[inb])[band][:, 1]
wl, wr = np.percentile(across, [5, 95])
print(f"wall-to-wall (5-95% of SfM across the path, 0.5-1.5 m up): {(wr - wl) * mpu:.2f} m")

sfm_r = sfm[inb]
base = cams[good] - np.outer(H(cams[good]) - floor, up)
ang = np.linspace(0, 2 * np.pi, 180, endpoint=False)
dirs = np.outer(np.cos(ang), e1) + np.outer(np.sin(ang), e2)
tree = cKDTree(sfm_r)
report = {"m_per_unit_estimate": mpu, "cameras": int(good.sum()), "sfm_points": len(sfm_r), "meshes": {},
          "level_frame": {"note": "splat/mesh frame -> level: p' = [e1 e2 up]^T (p - origin); floor at 0",
                          "e1": e1.tolist(), "e2": e2.tolist(), "up": up.tolist(), "origin": (c + floor * up).tolist(),
                          "tilt_of_nerfstudio_z_deg": float(np.degrees(np.arccos(abs(up[2]))))}}

def to_view(p):  # Z-up, floor at 0, metres, centred on the path
    q = np.c_[Pxy(p) - (lo + hi) / 2, H(p) - floor]
    return q * mpu
box_lo = np.r_[(lo - (lo + hi) / 2 - margin) * mpu, -1.0]
box_hi = np.r_[(hi - (lo + hi) / 2 + margin) * mpu, 3.5]
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
quant = lambda p: np.clip(np.round((p - box_lo) / (box_hi - box_lo) * 65535), 0, 65535).astype("<u2")

fig, axes = plt.subplots(len(meshes), 1, figsize=(16, 4.2 * len(meshes)), squeeze=False)
for j, (name, path) in enumerate(meshes.items()):
    mesh = o3d.io.read_triangle_mesh(path)
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    sc = o3d.t.geometry.RaycastingScene(); sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    d_sfm = sc.compute_distance(o3d.core.Tensor(sfm_r.astype(np.float32))).numpy()
    d_v = tree.query(V)[0]
    leaks = {}
    for z in (0.3, 1.0):
        O = np.repeat(base + u(z) * up, len(dirs), 0); D = np.tile(dirs, (len(base), 1))
        t = sc.cast_rays(o3d.core.Tensor(np.c_[O, D].astype(np.float32)))["t_hit"].numpy()
        leaks[z] = float(np.mean(t > 3 * span))
    org = base + u(1.0) * up
    t = sc.cast_rays(o3d.core.Tensor(np.c_[org, np.tile(-up, (len(org), 1))].astype(np.float32)))["t_hit"].numpy()
    dz = (u(1.0) - t) * mpu
    # robot clearance: up-ray from above the floor at each path point; mesh within the
    # next 0.95 m = something in the way of a 1 m tall robot driving the photographer's route
    # (start at 0.2 m: the reconstructed floor itself is bumpy by up to ~0.2 m, reported separately)
    org_up = base + u(0.20) * up
    tu = sc.cast_rays(o3d.core.Tensor(np.c_[org_up, np.tile(up, (len(org_up), 1))].astype(np.float32)))["t_hit"].numpy()
    blocked = float(np.mean(tu < u(0.80)))
    org_lo = base + u(0.05) * up
    tl = sc.cast_rays(o3d.core.Tensor(np.c_[org_lo, np.tile(up, (len(org_lo), 1))].astype(np.float32)))["t_hit"].numpy()
    bumpy = float(np.mean(tl < u(0.15)))
    labels, counts, _ = mesh.cluster_connected_triangles()
    r = {"triangles": len(F), "components": int(len(counts)),
         "sfm_within_5cm": float(np.mean(d_sfm < u(.05))), "sfm_within_10cm": float(np.mean(d_sfm < u(.10))),
         "sfm_median_cm": float(np.median(d_sfm) * mpu * 100),
         "verts_far_from_sfm_30cm": float(np.mean(d_v > u(.30))),
         "leak_0.3m": leaks[0.3], "leak_1.0m": leaks[1.0],
         "floor_hit_8cm": float(np.mean(np.abs(dz) < .08)), "hit_higher": float(np.mean(dz >= .08)),
         "hit_below_floor": float(np.mean(np.isfinite(t) & (dz <= -.08))), "no_hit": float(np.mean(~np.isfinite(t))),
         "watertight": bool(mesh.is_watertight()), "path_blocked_0.2_to_1m": blocked, "floor_bump_5_to_20cm": bumpy}
    report["meshes"][name] = r
    print(f"\n== {name}: {len(F):,} tris, {len(counts)} components")
    print(f"   SfM->mesh median {r['sfm_median_cm']:.1f} cm; within 5 cm {r['sfm_within_5cm']:.0%}, 10 cm {r['sfm_within_10cm']:.0%}")
    print(f"   lidar leak 0.3 m {leaks[0.3]:.1%}, 1.0 m {leaks[1.0]:.1%}")
    print(f"   path blocked between 0.2 and 1 m above the floor: {blocked:.0%}; floor bump 5-20 cm under {bumpy:.0%} of it")
    print(f"   down-ray from 1 m: floor {r['floor_hit_8cm']:.0%} | higher {r['hit_higher']:.0%} | below {r['hit_below_floor']:.0%} | none {r['no_hit']:.0%}")

    # slice 1.0 m above the floor, plan view
    ax = axes[j, 0]; Vh, Vp = H(V), Pxy(V); z = floor + u(1.0)
    tv = Vh[F]; cr = (tv.min(1) < z) & (tv.max(1) > z)
    segs = []
    for tri in F[cr]:
        hs, ps = Vh[tri], Vp[tri]; pts = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            if (hs[a] - z) * (hs[b] - z) < 0:
                pts.append(ps[a] + (z - hs[a]) / (hs[b] - hs[a]) * (ps[b] - ps[a]))
        if len(pts) == 2: segs.append(pts)
    from matplotlib.collections import LineCollection
    ax.add_collection(LineCollection(np.array(segs) * mpu, colors="k", linewidths=0.4))
    bnd = np.abs(H(sfm_r) - z) < u(0.1)
    ax.scatter(*(Pxy(sfm_r)[bnd] * mpu).T, s=0.5, c="tab:green", alpha=.6)
    ax.plot(*(cp[good] * mpu).T, "r-", lw=0.7)
    ax.set_title(f"{name}: slice 1.0 m above floor (black = mesh, green = SfM, red = camera path; ~metres)")
    ax.set_aspect("equal"); ax.set_xlim((lo[0] - margin) * mpu, (hi[0] + margin) * mpu); ax.set_ylim((lo[1] - margin) * mpu, (hi[1] + margin) * mpu)

    # viewer pack
    Vv = to_view(V)
    keep = np.all((Vv[F] >= box_lo) & (Vv[F] <= box_hi), 2).all(1)
    m2 = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(Vv), o3d.utility.Vector3iVector(F[keep]))
    if mesh.has_vertex_colors(): m2.vertex_colors = mesh.vertex_colors
    m2.remove_unreferenced_vertices()
    n_crop = len(m2.triangles)
    if n_crop > VIEW_TRIS: m2 = m2.simplify_quadric_decimation(VIEW_TRIS)
    m2.remove_unreferenced_vertices(); m2.remove_degenerate_triangles()
    V2, F2 = np.asarray(m2.vertices), np.asarray(m2.triangles).astype("<u4")
    (vout / f"{name}.json").write_text(json.dumps({"nv": len(V2), "nt": len(F2), "f32": False, "v": b64(quant(V2)), "f": b64(F2)}))
    r["viewer_tris"] = len(F2); r["tris_in_box"] = n_crop
    print(f"   viewer: {n_crop:,} tris in box -> {len(F2):,} shown")
fig.tight_layout(); fig.savefig(vout.parent / "corridor_slices.png", dpi=80)

# splat centres for the viewer (opaque ones, as a point layer) + SfM + cameras
sp = PlyData.read(str(repo / "outputs/splat/splat.ply"))["vertex"]
G = np.c_[sp["x"], sp["y"], sp["z"]]; op = 1 / (1 + np.exp(-sp["opacity"]))
col = np.clip(0.5 + 0.28209479 * np.c_[sp["f_dc_0"], sp["f_dc_1"], sp["f_dc_2"]], 0, 1)
Gv = to_view(G); ok = (op > 0.5) & np.all((Gv >= box_lo) & (Gv <= box_hi), 1)
rng = np.random.default_rng(0); idx = rng.choice(np.flatnonzero(ok), min(150_000, ok.sum()), replace=False)
Sv = to_view(sfm_r); Sv = Sv[np.all((Sv >= box_lo) & (Sv <= box_hi), 1)]
(vout / "scene.json").write_text(json.dumps({
    "box_lo": box_lo.tolist(), "box_hi": box_hi.tolist(), "m_per_unit": mpu,
    "sfm": b64(quant(Sv)), "n_sfm": len(Sv), "cams": np.round(to_view(cams[good]), 3).tolist(),
    "splat": b64(quant(Gv[idx])), "splat_rgb": b64((col[idx] * 255).astype(np.uint8)), "n_splat": len(idx)}))
report["splat"] = {"gaussians": len(G), "opaque_gt_0.5": int((op > 0.5).sum()), "shown": len(idx)}
(vout.parent / "corridor_report.json").write_text(json.dumps(report, indent=1))
print(f"\nsplat: {len(G):,} gaussians, {int((op > .5).sum()):,} with opacity > 0.5; viewer shows {len(idx):,}")
