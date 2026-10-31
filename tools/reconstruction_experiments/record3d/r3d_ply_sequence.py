"""Stand-in for Record3D's "Zipped sequence of PLY point clouds" export, built from the EXR + JPG export.

    python r3d_ply_sequence.py <EXR_RGBD dir> <out dir> [--every 1] [--prevoxel 0.05]

The nerfstudio docs (custom_dataset.html#record3d, "Adding a Point Cloud") feed that export to
`ns-process-data record3d --ply-dir`, which merges the files with no transform, so the points must be in ARKit's
world frame (Y-up, metres), the same frame as the poses. One PLY per frame: LiDAR depth (0.1-5 m) lifted with the
frame's intrinsics and raw ARKit pose, coloured from the RGB frame. Each file is pre-thinned to --prevoxel
(nerfstudio voxel-downsamples every file again, at 0.8 m by default, so this does not change its result).
"""
import argparse, json, os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
from pathlib import Path
import cv2, numpy as np
from scipy.spatial.transform import Rotation

ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("out")
ap.add_argument("--every", type=int, default=1); ap.add_argument("--prevoxel", type=float, default=0.05)
a = ap.parse_args(); src, out = Path(a.src), Path(a.out); out.mkdir(parents=True, exist_ok=True)
m = json.load(open(src / "metadata.json")); P = np.array(m["poses"]); KK = np.array(m["perFrameIntrinsicCoeffs"])
W, dw, dh = m["w"], m["dw"], m["dh"]; s = dw / W
v, u = np.mgrid[0:dh, 0:dw] + 0.5
n_pts = 0
for i in range(0, len(P), a.every):
    d = cv2.imread(str(src / "depth" / f"{i}.exr"), cv2.IMREAD_UNCHANGED)[..., 2]
    rgb = cv2.resize(cv2.imread(str(src / "rgb" / f"{i}.jpg"))[..., ::-1], (dw, dh), interpolation=cv2.INTER_AREA)
    fx, fy, cx, cy = KK[i] * s; ok = (d > 0.1) & (d < 5.0)
    Xc = np.c_[(u[ok] - cx) / fx * d[ok], -(v[ok] - cy) / fy * d[ok], -d[ok]]          # OpenGL camera
    Xw = Xc @ Rotation.from_quat(P[i, :4]).as_matrix().T + P[i, 4:]
    col = rgb[ok]
    key = np.floor(Xw / a.prevoxel).astype(np.int64); _, first = np.unique(key, axis=0, return_index=True)
    Xw, col = Xw[first].astype(np.float32), col[first]
    rec = np.empty(len(Xw), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    rec["x"], rec["y"], rec["z"] = Xw.T; rec["red"], rec["green"], rec["blue"] = col.T
    with open(out / f"{i}.ply", "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(rec)}\nproperty float x\nproperty float y\n"
                 "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        f.write(rec.tobytes())
    n_pts += len(rec)
print(f"wrote {len(range(0, len(P), a.every))} PLY files, {n_pts:,} points, to {out}")
