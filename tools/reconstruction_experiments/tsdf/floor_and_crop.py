#!/usr/bin/env python
"""Turn a raw fused room mesh into a collision mesh: real floor plane + everything above it.

    python floor_and_crop.py --mesh tsdf_full_colmap.ply --sparse <scene>/sparse/0 \
        --depth-dir <render_depth --out> --out collision_colmap.ply

Why: 3DGS geometry on textureless floor is unconstrained. On room05 the splat's
floor sags to ~1.3 m below the real one in the middle of the room, while
COLMAP's triangulated floor points (at the textured edges) sit on a flat plane.
So the floor is taken from COLMAP, not from the splat: a RANSAC plane through
the lowest well-supported layer of SfM points, perpendicular-ish to gravity
(mean camera image-up). Mesh below plane + --clearance is cut away (the sag),
and a floor quad covering the camera footprint + --floor-margin is added.

Everything stays in the COLMAP/3DGRUT world frame; make_collision_stage.py
places it next to the splat.
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import open3d as o3d


def read_points(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        out = np.empty((n, 3))
        for i in range(n):
            out[i] = struct.unpack("<QdddBBBd", f.read(43))[1:4]
            (tl,) = struct.unpack("<Q", f.read(8))
            f.seek(8 * tl, 1)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mesh", required=True)
    p.add_argument("--sparse", required=True, help="COLMAP model dir (points3D.bin)")
    p.add_argument("--depth-dir", required=True, help="render_depth.py output (camera poses)")
    p.add_argument("--out", required=True)
    p.add_argument("--clearance", type=float, default=0.03, help="Cut mesh below floor + this (scene units).")
    p.add_argument("--floor-margin", type=float, default=1.0, help="Floor quad = camera footprint grown by this.")
    p.add_argument("--cluster", type=float, default=0.0,
                   help="Vertex-clustering cell before decimation (scene units); 0 = off. It tears thin wall "
                        "shells into dashes on room05, so fuse at the voxel size you want instead.")
    p.add_argument("--target-tris", type=int, default=500_000)
    p.add_argument("--min-component-tris", type=int, default=2000)
    p.add_argument("--up", choices=("camera", "walls"), default="camera",
                   help="camera: mean camera image-up (fine when the phone was held level). walls: corridor axis "
                        "(camera path) x the dominant wall normal from SfM, for walks with the phone pitched "
                        "down (IMG_9206: mean camera up was 16.5 deg off).")
    a = p.parse_args()

    meta = json.loads((Path(a.depth_dir) / "meta.json").read_text())
    c2w = np.array([np.load(Path(a.depth_dir) / "frames" / f"{fr['id']}.npz")["c2w"] for fr in meta["frames"]])
    cams = c2w[:, :3, 3]
    up = -c2w[:, :3, 1].mean(0)
    up /= np.linalg.norm(up)
    sfm = read_points(Path(a.sparse) / "points3D.bin")
    if a.up == "walls":
        o3d.utility.random.seed(0)
        axis = np.linalg.svd(cams - cams.mean(0))[2][0]
        rest, n_wall, best = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sfm)), None, 0
        scale = np.ptp(cams, 0).max()
        for _ in range(6):
            (pa, pb, pc_, _d), inl = rest.segment_plane(0.002 * scale, 3, 4000)
            nn = np.array([pa, pb, pc_])
            if abs(nn @ axis) < 0.1 and abs(nn @ up) < 0.5 and len(inl) > best:
                n_wall, best = nn, len(inl)
            rest = rest.select_by_index(inl, invert=True)
        assert n_wall is not None, "no wall plane parallel to the camera path"
        new = np.cross(axis, n_wall); new /= np.linalg.norm(new); new *= np.sign(new @ up)
        print(f"up from walls ({best} SfM points on the main wall): {np.degrees(np.arccos(new @ up)):.1f} deg from camera up")
        up = new

    # floor: lowest height layer holding >= 15% of the busiest layer's points, then RANSAC
    c = cams.mean(0)
    e1 = np.cross(up, [1.0, 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(up, e1)
    plane_xy = lambda q: np.c_[(q - c) @ e1, (q - c) @ e2]
    lo, hi = plane_xy(cams).min(0) - 1.0, plane_xy(cams).max(0) + 1.0
    h = (sfm - c) @ up
    room = np.all((plane_xy(sfm) > lo) & (plane_xy(sfm) < hi), 1) & (np.abs(h) < 4)
    hist, ed = np.histogram(h[room], bins=np.arange(-4, 3, 0.05))
    floor_h = ed[np.argmax(hist >= 0.15 * hist.max())]
    cand = sfm[room & (np.abs(h - floor_h - 0.025) < 0.1)]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cand))
    (A, B, C, D), inl = pc.segment_plane(0.015, 3, 5000)
    n = np.array([A, B, C]); s = np.sign(n @ up); n, D = n * s, D * s  # normal points up
    tilt = np.degrees(np.arccos(np.clip(n @ up, -1, 1)))
    print(f"floor plane: {len(inl):,}/{len(cand):,} SfM inliers, {tilt:.1f} deg from camera-up, "
          f"{n @ c + D:.3f} units below the mean camera")
    assert tilt < 10, "floor plane is not horizontal; check the SfM points"

    mesh = o3d.io.read_triangle_mesh(a.mesh)
    n0 = len(mesh.triangles)
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    above = (V @ n + D) > a.clearance
    mesh.remove_triangles_by_mask(~above[F].any(1))
    mesh.remove_unreferenced_vertices()
    n_cut = len(mesh.triangles)
    if a.cluster > 0:
        mesh = mesh.simplify_vertex_clustering(a.cluster)
    labels, counts, _ = mesh.cluster_connected_triangles()
    labels, counts = np.asarray(labels), np.asarray(counts)
    mesh.remove_triangles_by_mask(counts[labels] < a.min_component_tris)
    mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) > a.target_tris:
        mesh = mesh.simplify_quadric_decimation(a.target_tris)
    print(f"mesh: {n0:,} tris -> {n_cut:,} above floor -> {len(mesh.triangles):,} after clustering/decimation")

    # floor quad on the plane, over the camera footprint + margin
    fl = lo - a.floor_margin + 1.0, hi + a.floor_margin - 1.0
    corners2 = np.array([[fl[0][0], fl[0][1]], [fl[1][0], fl[0][1]], [fl[1][0], fl[1][1]], [fl[0][0], fl[1][1]]])
    corners = c + corners2[:, :1] * e1 + corners2[:, 1:] * e2
    corners -= np.outer(corners @ n + D, n)  # project onto the plane
    floor = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(corners), o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]]))
    floor.vertex_colors = o3d.utility.Vector3dVector(np.full((4, 3), 0.6))
    out = mesh + floor
    out.compute_vertex_normals()
    o3d.io.write_triangle_mesh(a.out, out)
    rep = {"floor_plane": {"normal": n.tolist(), "d": float(D), "inliers": len(inl), "tilt_deg": float(tilt)},
           "tris_in": n0, "tris_above_floor": n_cut, "tris_out": len(out.triangles), "settings": vars(a)}
    Path(a.out).with_suffix(".json").write_text(json.dumps(rep, indent=1))
    print(f"wrote {a.out} ({len(out.triangles):,} tris incl. 2-triangle floor)")


if __name__ == "__main__":
    main()
