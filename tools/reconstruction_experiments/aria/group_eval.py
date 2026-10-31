"""Score corridor meshes in the COLMAP frame of one consistent camera group (same tests as corridor_eval.py).

    python group_eval.py <group sparse txt dir> <m_per_colmap_unit> <out_dir> name=mesh.ply [name=...]

Meshes must be in that model's COLMAP frame. Up = path axis x wall normal (RANSAC on the group's SfM
points), floor = lowest SfM height layer holding >=15% of the busiest layer's points, metres from
the given scale. Writes report.json, xsec_<name>.png (across-corridor sections) and plan.png.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path.home() / "Firdavs/repos/Depth-Anything-3/src"))
from depth_anything_3.utils.read_write_model import read_model  # noqa: E402

sparse, mpu, out = sys.argv[1], float(sys.argv[2]), Path(sys.argv[3])
meshes = dict(a.split("=", 1) for a in sys.argv[4:])
out.mkdir(parents=True, exist_ok=True)
u = lambda m: m / mpu
_, ims, pts = read_model(sparse, ".txt")
Rw = [Rotation.from_quat(np.r_[im.qvec[1:], im.qvec[0]]).as_matrix() for im in ims.values()]
cams = np.array([-R.T @ im.tvec for R, im in zip(Rw, ims.values())])
up_cam = -np.mean([R[1] for R in Rw], 0); up_cam /= np.linalg.norm(up_cam)
sfm = np.array([p.xyz for p in pts.values()])

o3d.utility.random.seed(0)
c = cams.mean(0)
axis = np.linalg.svd(cams - c)[2][0]
pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sfm)); n_wall, best = None, 0
for _ in range(6):
    if len(pcd.points) < 300:
        break
    (a_, b_, c_, _d), inl = pcd.segment_plane(u(0.02), 3, 4000)
    n = np.array([a_, b_, c_])
    if abs(n @ axis) < 0.1 and abs(n @ up_cam) < 0.5 and len(inl) > best:
        n_wall, best = n, len(inl)
    pcd = pcd.select_by_index(inl, invert=True)
up = np.cross(axis, n_wall); up /= np.linalg.norm(up); up *= np.sign(up @ up_cam)
e1 = axis - (axis @ up) * up; e1 /= np.linalg.norm(e1); e2 = np.cross(up, e1)
Pxy = lambda p: np.c_[(p - c) @ e1, (p - c) @ e2]
H = lambda p: (p - c) @ up
cp = Pxy(cams)
lo, hi = cp.min(0), cp.max(0)
span = np.ptp(cp, 0).max(); margin = 0.25 * span
inb = np.all((Pxy(sfm) > lo - margin) & (Pxy(sfm) < hi + margin), 1)
h = H(sfm[inb])
hist, ed = np.histogram(h, bins=np.linspace(np.percentile(h, 0.5), np.percentile(h, 99.5), 120))
floor = ed[np.argmax(hist >= 0.15 * hist.max())]
sfm_r = sfm[inb]
hh = H(sfm_r) - floor
band = (hh > u(0.5)) & (hh < u(1.5))
wl, wr = np.percentile(Pxy(sfm_r)[band][:, 1], [5, 95])
geo = {"m_per_unit": mpu, "cameras": len(cams), "sfm_points": len(sfm_r),
       "up_vs_mean_camera_up_deg": float(np.degrees(np.arccos(up @ up_cam))),
       "camera_height_m_median": float((np.median(H(cams)) - floor) * mpu),
       "wall_to_wall_m_5_95": float((wr - wl) * mpu),
       "ceiling_m_p98": float(np.percentile(hh, 98) * mpu)}
print("frame:", json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in geo.items()}))

base = cams - np.outer(H(cams) - floor, up)
ang = np.linspace(0, 2 * np.pi, 180, endpoint=False)
dirs = np.outer(np.cos(ang), e1) + np.outer(np.sin(ang), e2)
report = {"frame": geo, "meshes": {}}
xs = np.linspace(lo[0] + 0.05 * (hi[0] - lo[0]), hi[0] - 0.05 * (hi[0] - lo[0]), 6)
Lv = lambda p: np.c_[Pxy(p), H(p) - floor] * mpu  # level frame, metres, floor at 0
Sl, Cl = Lv(sfm_r), Lv(cams)
figp, axp = plt.subplots(len(meshes), 1, figsize=(16, 3.6 * len(meshes)), squeeze=False)
for j, (name, path) in enumerate(meshes.items()):
    mesh = o3d.io.read_triangle_mesh(path)
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    sc = o3d.t.geometry.RaycastingScene(); sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    cast = lambda O, D: sc.cast_rays(o3d.core.Tensor(np.c_[O, D].astype(np.float32)))["t_hit"].numpy()
    d_sfm = sc.compute_distance(o3d.core.Tensor(sfm_r.astype(np.float32))).numpy()
    leaks = {}
    for z in (0.3, 1.0):
        t = cast(np.repeat(base + u(z) * up, len(dirs), 0), np.tile(dirs, (len(base), 1)))
        leaks[z] = float(np.mean(t > 3 * span))
    t = cast(base + u(1.0) * up, np.tile(-up, (len(base), 1))); dz = (u(1.0) - t) * mpu
    tu = cast(base + u(0.20) * up, np.tile(up, (len(base), 1)))
    tl = cast(base + u(0.05) * up, np.tile(up, (len(base), 1)))
    _, counts, _ = mesh.cluster_connected_triangles()
    r = {"triangles": len(F), "components": int(len(counts)),
         "sfm_within_5cm": float(np.mean(d_sfm < u(.05))), "sfm_within_10cm": float(np.mean(d_sfm < u(.10))),
         "sfm_median_cm": float(np.median(d_sfm) * mpu * 100),
         "leak_0.3m": leaks[0.3], "leak_1.0m": leaks[1.0],
         "path_blocked_0.2_to_1m": float(np.mean(tu < u(0.80))), "floor_bump_5_to_20cm": float(np.mean(tl < u(0.15))),
         "floor_hit_8cm": float(np.mean(np.abs(dz) < .08)), "hit_higher": float(np.mean(dz >= .08)),
         "no_hit": float(np.mean(~np.isfinite(t)))}
    report["meshes"][name] = r
    print(f"== {name}: {len(F):,} tris, {len(counts)} components | SfM within 5 cm {r['sfm_within_5cm']:.0%} "
          f"(median {r['sfm_median_cm']:.1f} cm) | leak 0.3/1.0 m {leaks[0.3]:.1%}/{leaks[1.0]:.1%} | "
          f"route blocked {r['path_blocked_0.2_to_1m']:.0%} | floor bump {r['floor_bump_5_to_20cm']:.0%} | "
          f"down-ray floor {r['floor_hit_8cm']:.0%} higher {r['hit_higher']:.0%} none {r['no_hit']:.0%}")

    VL = Lv(V)
    fig, axes = plt.subplots(1, len(xs), figsize=(3.2 * len(xs), 3.6))
    for ax, x0 in zip(axes, xs * mpu):
        d = VL[:, 0] - x0; tv = d[F]; cr = (tv.min(1) < 0) & (tv.max(1) > 0); segs = []
        for tri in F[cr]:
            q = []
            for a, b in ((0, 1), (1, 2), (2, 0)):
                if d[tri[a]] * d[tri[b]] < 0:
                    w = d[tri[a]] / (d[tri[a]] - d[tri[b]]); q.append(VL[tri[a], 1:] + w * (VL[tri[b], 1:] - VL[tri[a], 1:]))
            if len(q) == 2:
                segs.append(q)
        ax.add_collection(LineCollection(np.array(segs).reshape(-1, 2, 2), colors="k", linewidths=0.6))
        s = np.abs(Sl[:, 0] - x0) < 0.25; ax.scatter(Sl[s, 1], Sl[s, 2], s=2, c="tab:green")
        s = np.abs(Cl[:, 0] - x0) < 0.5; ax.scatter(Cl[s, 1], Cl[s, 2], s=12, c="r", marker="x")
        ax.axhline(0, c="tab:blue", lw=0.5); ax.set_xlim(-2, 2); ax.set_ylim(-0.5, 3.2); ax.set_aspect("equal")
        ax.set_title(f"x = {x0:.1f} m", fontsize=9)
    fig.suptitle(f"{name}: across-corridor sections (black = mesh, green = COLMAP points, red = cameras, blue = floor)", fontsize=10)
    fig.tight_layout(); fig.savefig(out / f"xsec_{name}.png", dpi=80); plt.close(fig)

    ax = axp[j, 0]; z = u(1.0) + floor; Vh = H(V); Vp = Pxy(V); tv = Vh[F]; cr = (tv.min(1) < z) & (tv.max(1) > z); segs = []
    for tri in F[cr]:
        q = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            if (Vh[tri[a]] - z) * (Vh[tri[b]] - z) < 0:
                q.append(Vp[tri[a]] + (z - Vh[tri[a]]) / (Vh[tri[b]] - Vh[tri[a]]) * (Vp[tri[b]] - Vp[tri[a]]))
        if len(q) == 2:
            segs.append(q)
    ax.add_collection(LineCollection(np.array(segs).reshape(-1, 2, 2) * mpu, colors="k", linewidths=0.4))
    ax.plot(*(cp * mpu).T, "r-", lw=0.7)
    ax.set_aspect("equal"); ax.set_xlim((lo[0] - margin) * mpu, (hi[0] + margin) * mpu); ax.set_ylim((lo[1] - margin) * mpu, (hi[1] + margin) * mpu)
    ax.set_title(f"{name}: plan slice 1.0 m above the floor (black = mesh, red = camera path; metres)")
figp.tight_layout(); figp.savefig(out / "plan.png", dpi=80)
report["level_frame"] = {"e1": e1.tolist(), "e2": e2.tolist(), "up": up.tolist(), "origin": (c + floor * up).tolist()}
(out / "report.json").write_text(json.dumps(report, indent=1))
