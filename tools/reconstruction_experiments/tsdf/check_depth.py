#!/usr/bin/env python
"""Score rendered depth against the COLMAP points each photo actually observed.

Run in the 3DGRUT venv (uses its COLMAP reader):

    python check_depth.py --depth-dir <render_depth --out> --sparse <scene>/sparse/0 [--every 15]

For every Nth frame it takes the 3D points COLMAP triangulated *from that
image* (so they are visible, no occlusion guessing), and reports
  reproj  - pixel distance between our projection (fitted intrinsics + pose)
            and COLMAP's observation; ~0.5 px at 1/4 resolution means the
            cameras are right
  rel err - (rendered z - COLMAP z) / COLMAP z; positive = rendered surface
            sits behind the real one
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "Firdavs/repos/3dgrut"))
from threedgrut.datasets.utils import read_colmap_extrinsics_binary  # noqa: E402


def read_points(path):
    xyz = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            pid, x, y, z, _r, _g, _b, _e = struct.unpack("<QdddBBBd", f.read(43))
            (tl,) = struct.unpack("<Q", f.read(8))
            f.seek(8 * tl, 1)
            xyz[pid] = (x, y, z)
    return xyz


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--depth-dir", required=True)
    p.add_argument("--sparse", required=True)
    p.add_argument("--every", type=int, default=15)
    a = p.parse_args()
    dd, sp = Path(a.depth_dir), Path(a.sparse)
    meta = json.loads((dd / "meta.json").read_text())
    K = meta["intrinsics"]
    # pixel scale from COLMAP's full-res observations to the fused depth image
    ims = {im.name: im for im in read_colmap_extrinsics_binary(sp / "images.bin")}
    xyz = read_points(sp / "points3D.bin")
    with open(sp / "cameras.bin", "rb") as f:  # first camera's width (all photos share one camera)
        f.read(8 + 4 + 4)
        (colmap_w,) = struct.unpack("<Q", f.read(8))
    rels, reps = [], []
    for fr in meta["frames"][:: a.every]:
        d = np.load(dd / "frames" / f"{fr['id']}.npz")
        im = ims[fr["image"]]
        ok = im.point3D_ids >= 0
        P = np.array([xyz[i] for i in im.point3D_ids[ok]])
        w2c = np.linalg.inv(d["c2w"])
        pc = P @ w2c[:3, :3].T + w2c[:3, 3]
        # COLMAP pixel coords are full resolution, origin at the pixel corner
        s = d["depth"].shape[1] / colmap_w
        uo, vo = im.xys[ok, 0] * s - 0.5, im.xys[ok, 1] * s - 0.5
        up, vp = K["fx"] * pc[:, 0] / pc[:, 2] + K["cx"], K["fy"] * pc[:, 1] / pc[:, 2] + K["cy"]
        reps.append(np.median(np.hypot(up - uo, vp - vo)))
        u, v = np.round(uo).astype(int), np.round(vo).astype(int)
        inb = (u >= 0) & (u < K["width"]) & (v >= 0) & (v < K["height"])
        zr, zs = d["depth"].astype(np.float32)[v[inb], u[inb]], pc[inb, 2]
        g = zr > 0
        rels.append((zr[g] - zs[g]) / zs[g])
    r = np.concatenate(rels)
    out = {"frames": len(rels), "points": int(len(r)), "reproj_px_median": float(np.median(reps)),
           "rel_err_median": float(np.median(r)), "abs_rel_err_median": float(np.median(np.abs(r))),
           "within_2pct": float(np.mean(np.abs(r) < 0.02)), "within_5pct": float(np.mean(np.abs(r) < 0.05)),
           "rel_err_p10": float(np.percentile(r, 10)), "rel_err_p90": float(np.percentile(r, 90))}
    print(json.dumps(out, indent=1))
    (dd / "depth_check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
