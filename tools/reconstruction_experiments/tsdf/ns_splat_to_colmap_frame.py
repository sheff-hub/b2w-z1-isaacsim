#!/usr/bin/env python
"""Move a nerfstudio splat (ns-export gaussian-splat) back into its COLMAP world frame.

    python ns_splat_to_colmap_frame.py splat.ply dataparser_transforms.json splat_colmap.ply

nerfstudio trains in its dataparser frame: p_model = s * (R p_colmap + t), with R, t, s from
dataparser_transforms.json. That transform already includes transforms.json's
`applied_transform`, so it maps ORIGINAL COLMAP coordinates. The inverse is applied to positions,
rotations (quaternions, rot_0 = w) and log-scales, so 3DGRUT's COLMAP loader and render_depth.py
see the splat where the COLMAP cameras are.

Spherical-harmonic colour (f_rest_*) is NOT rotated, so view-dependent colour is slightly wrong
in the new frame. Geometry and opacity are exact, which is all depth rendering needs.
"""

import json
import sys

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation


def main():
    src, dpt, dst = sys.argv[1:4]
    dp = json.load(open(dpt))
    T, s = np.array(dp["transform"]), float(dp["scale"])
    R, t = T[:, :3], T[:, 3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-5), "dataparser transform is not a rotation"

    ply = PlyData.read(src)
    v = ply["vertex"].data.copy()
    p = np.c_[v["x"], v["y"], v["z"]].astype(np.float64)
    pc = (p / s - t) @ R  # = R^T (p/s - t) for row vectors
    v["x"], v["y"], v["z"] = pc.T.astype(np.float32)

    q = np.c_[v["rot_1"], v["rot_2"], v["rot_3"], v["rot_0"]].astype(np.float64)  # scipy: x, y, z, w
    q /= np.linalg.norm(q, axis=1, keepdims=True).clip(1e-12)
    qc = (Rotation.from_matrix(R.T) * Rotation.from_quat(q)).as_quat()
    v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"] = qc[:, 3], qc[:, 0], qc[:, 1], qc[:, 2]

    for k in ("scale_0", "scale_1", "scale_2"):
        v[k] = (v[k] - np.log(s)).astype(np.float32)

    PlyData([PlyElement.describe(v, "vertex")], byte_order=ply.byte_order).write(dst)
    print(f"{len(v):,} gaussians -> COLMAP frame (1/s = {1 / s:.4f}); position range {pc.min(0).round(2)} .. {pc.max(0).round(2)}")


if __name__ == "__main__":
    main()
