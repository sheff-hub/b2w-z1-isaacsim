"""Mesh in COLMAP frame -> nerfstudio model frame (p_model = s (R p + t), dataparser_transforms.json)."""
import json, sys
import numpy as np, open3d as o3d
src, dpt, dst = sys.argv[1:4]
dp = json.load(open(dpt)); T, s = np.array(dp["transform"]), float(dp["scale"])
m = o3d.io.read_triangle_mesh(src)
m.vertices = o3d.utility.Vector3dVector((np.asarray(m.vertices) @ T[:, :3].T + T[:, 3]) * s)
m.compute_vertex_normals(); o3d.io.write_triangle_mesh(dst, m)
print(dst, len(m.triangles))
