"""Quadric decimation of the fused mesh to a triangle budget (PhysX cooks a 1.2M-triangle collider in seconds).

    python decimate.py <in.ply> <out.ply> [--triangles 1200000]
"""
import argparse
import open3d as o3d

ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("out"); ap.add_argument("--triangles", type=int, default=1_200_000)
a = ap.parse_args()
m = o3d.io.read_triangle_mesh(a.src); n0 = len(m.triangles)
d = m.simplify_quadric_decimation(a.triangles) if n0 > a.triangles else m
d.remove_unreferenced_vertices(); d.compute_vertex_normals()
o3d.io.write_triangle_mesh(a.out, d)
print(f"{n0:,} -> {len(d.triangles):,} triangles -> {a.out}")
