"""Score DA3 per-frame NPZ depth against the COLMAP points each photo observed (same metric as tools/tsdf/check_depth.py).

usage: check_da3_depth.py <npz dir> <sparse text dir> [every] [conf_thr]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "Firdavs/repos/Depth-Anything-3/src"))
from depth_anything_3.utils.read_write_model import read_model  # noqa: E402

npz_dir, sparse = Path(sys.argv[1]), sys.argv[2]
every = int(sys.argv[3]) if len(sys.argv) > 3 else 1
conf_thr = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
cams, ims, pts = read_model(sparse, ".txt")
W, H = next(iter(cams.values())).width, next(iter(cams.values())).height
by_name = {im.name: im for im in ims.values()}
rels, reps, kept = [], [], []
for f in sorted(npz_dir.glob("*.npz"))[::every]:
    z = np.load(f)
    im = by_name[f.stem + ".jpg"]
    d, c, K, E = z["depth"], z["conf"], z["intrinsics"], z["extrinsics"]
    ok = im.point3D_ids >= 0
    if not ok.any():
        continue
    P = np.array([pts[i].xyz for i in im.point3D_ids[ok]])
    pc = P @ E[:3, :3].T + E[:3, 3]
    sx, sy = d.shape[1] / W, d.shape[0] / H
    uo, vo = im.xys[ok, 0] * sx, im.xys[ok, 1] * sy
    up, vp = K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2], K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]
    reps.append(np.median(np.hypot(up - uo, vp - vo)))
    u, v = np.floor(uo).astype(int), np.floor(vo).astype(int)
    inb = (u >= 0) & (u < d.shape[1]) & (v >= 0) & (v < d.shape[0])
    zr, zs, cc = d[v[inb], u[inb]], pc[inb, 2], c[v[inb], u[inb]]
    g = (zr > 0) & (cc >= conf_thr)
    kept.append(g.mean())
    rels.append((zr[g] - zs[g]) / zs[g])
r = np.concatenate(rels)
out = {"frames": len(rels), "points": int(len(r)), "conf_thr": conf_thr, "kept_share": float(np.mean(kept)),
       "reproj_px_median": float(np.median(reps)), "rel_err_median": float(np.median(r)),
       "abs_rel_err_median": float(np.median(np.abs(r))), "within_2pct": float(np.mean(np.abs(r) < 0.02)),
       "within_5pct": float(np.mean(np.abs(r) < 0.05)), "rel_err_p10": float(np.percentile(r, 10)),
       "rel_err_p90": float(np.percentile(r, 90))}
print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}))
