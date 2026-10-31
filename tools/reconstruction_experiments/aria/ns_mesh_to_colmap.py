"""Mesh in nerfstudio model frame -> original COLMAP frame (inverse of dataparser_transforms.json: p = s (R q + t))."""
import json, sys
import numpy as np, open3d as o3d
src, dpt, dst = sys.argv[1:4]
dp = json.load(open(dpt)); T, s = np.array(dp["transform"]), float(dp["scale"])
m = o3d.io.read_triangle_mesh(src)
m.vertices = o3d.utility.Vector3dVector((np.asarray(m.vertices) / s - T[:, 3]) @ T[:, :3])
m.compute_vertex_normals(); o3d.io.write_triangle_mesh(dst, m)
print(dst, len(m.triangles))
