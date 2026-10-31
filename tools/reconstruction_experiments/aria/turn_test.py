"""DA3 pose estimate (no input poses) over a frame range; prints heading change relative to the first frame."""
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from depth_anything_3.api import DepthAnything3
I, a, b, step = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
ids = list(range(a, b + 1, step))
model = DepthAnything3.from_pretrained(sys.argv[5] if len(sys.argv) > 5 else "depth-anything/DA3-LARGE-1.1").to("cuda")
pred = model.inference([f"{I}/frame_{i:05d}.jpg" for i in ids], ref_view_strategy="first")
E = pred.extrinsics  # w2c, (N,3,4)
R0 = E[0, :, :3]
c = np.array([-e[:, :3].T @ e[:, 3] for e in E])
# heading: angle of the optical axis (camera +z) projected on the plane orthogonal to frame-0 image-up (-y)
up0 = -R0[1]; f0 = R0[2]
for i, e in zip(ids, E):
    fz = e[2, :3]  # camera z axis in world (row 2 of w2c rotation)
    fz_h = fz - (fz @ up0) * up0
    ang = np.degrees(np.arctan2(np.cross(f0, fz_h) @ up0, f0 @ fz_h))
    print(f"frame {i:3d}: heading {ang:+7.1f} deg, pos {(c[ids.index(i)] - c[0]).round(2).tolist()}")
