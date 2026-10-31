"""TSDF-fuse Record3D LiDAR depth with the refined poses (VDBFusion, as in Stani's fuse_da3_vdb.py).

    python fuse_lidar.py <scene dir> <out.ply> [--voxel 0.03] [--trunc 0.09] [--max-depth 4.5] [--every 1]

Run in the tools environment (.venv). Uses <scene>/record3d/poses_refined.npz and the LiDAR frames prep chose
(depth.npy + depth_frames.npy: one every 1.5 cm / 1 deg of motion; older scenes: depth_every<k>.npy).
Drops flying pixels at depth edges (>4% jump to a neighbour). Output is in the scene frame (metres, Z-up, floor 0).
"""
import argparse, os
import numpy as np, open3d as o3d, vdbfusion

ap = argparse.ArgumentParser(); ap.add_argument("scene"); ap.add_argument("out")
ap.add_argument("--voxel", type=float, default=0.03); ap.add_argument("--trunc", type=float, default=0.09)
ap.add_argument("--min-depth", type=float, default=0.3); ap.add_argument("--max-depth", type=float, default=4.5)
ap.add_argument("--edge", type=float, default=0.04); ap.add_argument("--every", type=int, default=1)
ap.add_argument("--poses", default="poses_refined.npz")
a = ap.parse_args()
z = np.load(f"{a.scene}/record3d/{a.poses}"); c2w, K = z["c2w"], z["K"]; s = int(z["dw"]) / int(z["w"])
if os.path.exists(f"{a.scene}/record3d/depth_frames.npy"):
    D = np.load(f"{a.scene}/record3d/depth.npy", mmap_mode="r"); frames = np.load(f"{a.scene}/record3d/depth_frames.npy")
else:                                     # scenes prepared before 27 Oct 2026: every k-th frame
    de = int(z["depth_every"]) if "depth_every" in z else 2
    D = np.load(f"{a.scene}/record3d/depth_every{de}.npy", mmap_mode="r"); frames = de * np.arange(len(D))
v, u = np.mgrid[0:int(z["dh"]), 0:int(z["dw"])] + 0.5
vol = vdbfusion.VDBVolume(a.voxel, a.trunc, space_carving=True)
n_pts = 0
for k in range(0, len(D), a.every):
    i = int(frames[k]); d = D[k].astype(np.float32)
    p = np.pad(d, 1, mode="edge")
    jump = np.max(np.abs(np.stack([p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]]) - d), axis=0)
    ok = (d > a.min_depth) & (d < a.max_depth) & (jump < a.edge * d)
    fx, fy, cx, cy = K[i] * s
    pc = np.c_[(u[ok] - cx) / fx * d[ok], (v[ok] - cy) / fy * d[ok], d[ok]]
    pw = pc @ c2w[i][:3, :3].T + c2w[i][:3, 3]
    vol.integrate(pw.astype(np.float64), c2w[i][:3, 3].astype(np.float64)); n_pts += len(pw)
    if k % 500 == 0: print(f"frame {i}: {len(pw):,} points", flush=True)
V, F = vol.extract_triangle_mesh()
m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))
m.remove_degenerate_triangles(); m.remove_unreferenced_vertices()
# drop small disconnected bits (< 200 triangles)
cl, n_tri, _ = m.cluster_connected_triangles(); cl = np.asarray(cl); n_tri = np.asarray(n_tri)
m.remove_triangles_by_mask(n_tri[cl] < 200); m.remove_unreferenced_vertices()
m.compute_vertex_normals(); o3d.io.write_triangle_mesh(a.out, m)
print(f"integrated {n_pts:,} points from {len(range(0, len(D), a.every))} frames -> {len(m.triangles):,} triangles, "
      f"bbox {np.round(np.asarray(m.vertices).min(0), 2)} .. {np.round(np.asarray(m.vertices).max(0), 2)}")
