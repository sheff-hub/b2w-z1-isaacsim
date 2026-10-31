"""Score meshes in a metric, Z-up frame with the floor at z = 0.

    python eval_mesh.py <scene dir> <out dir> name=mesh.ply [name=mesh2.ply@T.npy ...]

A mesh given as path@T.npy is first moved by the 4x4 matrix in T.npy (into the scene's frame).

Cameras: <scene>/record3d/poses_refined.npz (every 10th frame). Points: <scene>/sparse/0 (COLMAP, BA-refined).
  leak            share of a horizontal 180-ray fan at 0.3 / 1.0 m from each camera that escapes (> 3x the
                  path's span); counts real openings (doors, glass) too
  route blocked   ray up from 0.2 m at each camera position hits mesh below 1.0 m
  floor bump      ray up from 0.05 m hits below 0.20 m
  floor hit       ray down from 1.0 m lands within 8 cm of z = 0
  accuracy        COLMAP points within 5 / 10 cm of the mesh
Writes report.json, plan.png (slices at 0.3 and 1.0 m) and xsec_<name>.png (sections across the path).
"""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np, open3d as o3d, pycolmap

scene, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
meshes = dict(a.split("=", 1) for a in sys.argv[3:])
z = np.load(scene / "record3d" / "poses_refined.npz"); cams = z["c2w"][::10, :3, 3]
rec = pycolmap.Reconstruction(str(scene / "sparse" / "0")); sfm = np.array([p.xyz for p in rec.points3D.values()])
span = np.ptp(cams[:, :2], 0).max()
base = np.c_[cams[:, :2], np.zeros(len(cams))]
ang = np.linspace(0, 2 * np.pi, 180, endpoint=False); dirs = np.c_[np.cos(ang), np.sin(ang), np.zeros_like(ang)]
up = np.array([0, 0, 1.0])
# section sites along the path: every ~len/7, plane normal = local path direction
seg = np.linalg.norm(np.diff(cams[:, :2], axis=0), axis=1); s_ = np.r_[0, np.cumsum(seg)]
sites = [int(np.searchsorted(s_, f * s_[-1])) for f in np.linspace(0.06, 0.94, 7)]

def slice_h(V, F, zc):
    tv = V[F, 2]; cr = (tv.min(1) < zc) & (tv.max(1) > zc); segs = []
    for tri in F[cr]:
        q = [V[tri[a], :2] + (zc - V[tri[a], 2]) / (V[tri[b], 2] - V[tri[a], 2]) * (V[tri[b], :2] - V[tri[a], :2])
             for a, b in ((0, 1), (1, 2), (2, 0)) if (V[tri[a], 2] - zc) * (V[tri[b], 2] - zc) < 0]
        if len(q) == 2: segs.append(q)
    return np.array(segs).reshape(-1, 2, 2)

report = {"frame": {"cameras_used": len(cams), "sfm_points": len(sfm), "path_length_m": float(s_[-1]),
                    "camera_height_m_median": float(np.median(cams[:, 2]))}, "meshes": {}}
figp, axp = plt.subplots(len(meshes), 2, figsize=(20, 9 * len(meshes)), squeeze=False)
for j, (name, path) in enumerate(meshes.items()):
    path, _, tf = path.partition("@")
    mesh = o3d.io.read_triangle_mesh(path)
    if tf: mesh.transform(np.load(tf))
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    sc = o3d.t.geometry.RaycastingScene(); sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    cast = lambda O, D: sc.cast_rays(o3d.core.Tensor(np.c_[O, D].astype(np.float32)))["t_hit"].numpy()
    d_sfm = sc.compute_distance(o3d.core.Tensor(sfm.astype(np.float32))).numpy()
    leaks = {h: float(np.mean(cast(np.repeat(base + h * up, len(dirs), 0), np.tile(dirs, (len(base), 1))) > 3 * span)) for h in (0.3, 1.0)}
    t = cast(base + 1.0 * up, np.tile(-up, (len(base), 1))); dz = 1.0 - t
    tu = cast(base + 0.20 * up, np.tile(up, (len(base), 1))); tl = cast(base + 0.05 * up, np.tile(up, (len(base), 1)))
    tc = cast(base + 1.0 * up, np.tile(up, (len(base), 1)))
    _, counts, _ = mesh.cluster_connected_triangles()
    r = {"triangles": len(F), "components": int(len(counts)), "sfm_within_5cm": float(np.mean(d_sfm < .05)),
         "sfm_within_10cm": float(np.mean(d_sfm < .10)), "sfm_median_cm": float(np.median(d_sfm) * 100),
         "leak_0.3m": leaks[0.3], "leak_1.0m": leaks[1.0], "path_blocked_0.2_to_1m": float(np.mean(tu < 0.80)),
         "floor_bump_5_to_20cm": float(np.mean(tl < 0.15)), "floor_hit_8cm": float(np.mean(np.abs(dz) < .08)),
         "floor_z_median_cm": float(np.nanmedian(np.where(np.isfinite(dz), dz, np.nan)) * 100),
         "hit_higher": float(np.mean(dz >= .08)), "no_floor": float(np.mean(~np.isfinite(t))),
         "ceiling_m_median": float(np.median((1.0 + tc)[np.isfinite(tc)])) if np.isfinite(tc).any() else None}
    report["meshes"][name] = r
    print(f"== {name}: {len(F):,} tris, {len(counts)} components | COLMAP pts within 5 cm {r['sfm_within_5cm']:.0%} "
          f"(median {r['sfm_median_cm']:.1f} cm) | leak 0.3/1.0 m {leaks[0.3]:.1%}/{leaks[1.0]:.1%} | route blocked "
          f"{r['path_blocked_0.2_to_1m']:.1%} | floor bump {r['floor_bump_5_to_20cm']:.1%} | floor within 8 cm {r['floor_hit_8cm']:.0%} "
          f"(median {r['floor_z_median_cm']:+.1f} cm), none {r['no_floor']:.0%} | ceiling "
          f"{'none' if r['ceiling_m_median'] is None else format(r['ceiling_m_median'], '.2f') + ' m'}")
    for k, h in enumerate((0.3, 1.0)):
        ax = axp[j, k]; ax.add_collection(LineCollection(slice_h(V, F, h), colors="k", linewidths=0.35))
        ax.plot(cams[:, 0], cams[:, 1], "r-", lw=0.6); ax.set_aspect("equal")
        ax.set_xlim(cams[:, 0].min() - 3, cams[:, 0].max() + 3); ax.set_ylim(cams[:, 1].min() - 3, cams[:, 1].max() + 3)
        ax.set_title(f"{name}: plan slice {h} m above the floor (black = mesh, red = path; metres)")
    fig, axes = plt.subplots(1, len(sites), figsize=(3.3 * len(sites), 3.8))
    for ax, si in zip(axes, sites):
        dvec = cams[min(si + 3, len(cams) - 1), :2] - cams[max(si - 3, 0), :2]; dvec /= np.linalg.norm(dvec)
        side = np.array([-dvec[1], dvec[0]]); o = cams[si, :2]
        d = (V[:, :2] - o) @ dvec; lat = (V[:, :2] - o) @ side; tv = d[F]; cr = (tv.min(1) < 0) & (tv.max(1) > 0); segs = []
        for tri in F[cr]:
            q = [np.r_[lat[tri[a]], V[tri[a], 2]] + d[tri[a]] / (d[tri[a]] - d[tri[b]]) * (np.r_[lat[tri[b]], V[tri[b], 2]] - np.r_[lat[tri[a]], V[tri[a], 2]])
                 for a, b in ((0, 1), (1, 2), (2, 0)) if d[tri[a]] * d[tri[b]] < 0]
            if len(q) == 2: segs.append(q)
        ax.add_collection(LineCollection(np.array(segs).reshape(-1, 2, 2), colors="k", linewidths=0.6))
        ps = np.abs((sfm[:, :2] - o) @ dvec) < 0.15; ax.scatter((sfm[ps, :2] - o) @ side, sfm[ps, 2], s=2, c="tab:green")
        ax.scatter([0], [cams[si, 2]], c="r", marker="x"); ax.axhline(0, c="tab:blue", lw=0.5)
        ax.set_xlim(-2.5, 2.5); ax.set_ylim(-0.5, 3.3); ax.set_aspect("equal"); ax.set_title(f"{s_[si]:.0f} m along the path", fontsize=9)
    fig.suptitle(f"{name}: sections across the path (black = mesh, green = COLMAP points, red = camera)", fontsize=10)
    fig.tight_layout(); fig.savefig(out / f"xsec_{name}.png", dpi=80); plt.close(fig)
figp.tight_layout(); figp.savefig(out / "plan.png", dpi=70)
(out / "report.json").write_text(json.dumps(report, indent=1))
