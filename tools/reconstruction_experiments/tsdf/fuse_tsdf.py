#!/usr/bin/env python
"""Fuse depth rendered by render_depth.py into a triangle mesh (TSDF + marching cubes).

Stage 2 of 2. Needs open3d and scikit-image, deliberately NOT in the 3DGRUT
venv so its pinned packages stay untouched:

    python fuse_tsdf.py --depth-dir <render_depth --out> --out-dir <dir> [--voxel 0.0125]

Why fusion rather than Poisson on gaussian centres: every pixel's ray says
"empty up to here, surface here". Averaging those over all views carves free
space and puts surfaces where the rendered appearance puts them, which is what
the photos constrain, instead of where gaussian centres happen to sit.

Two Open3D 0.20 (pip) traps, both met on room05 on 20 Oct 2026:
- the legacy ScalableTSDFVolume integrates nothing from these frames, silently;
- the tensor VoxelBlockGrid integrates fine, but its CPU extract_triangle_mesh
  segfaults at this size (447k blocks). So Open3D only integrates here; the
  observed voxels are copied into a dense grid and meshed with scikit-image,
  in parallel slabs (one process takes >15 min on room05).
Fusing on CUDA needs ~5 GB free (447k blocks at 1 cm); on a shared card use CPU,
where integrating 428 frames takes ~15 s.

Outputs (all in --out-dir, COLMAP/3DGRUT world frame, scene units):
  tsdf_full_colmap.ply   full-resolution mesh with vertex colours
  tsdf_500k_colmap.ply   decimated to --target-tris
  report.json            sizes and settings
Use make_collision_stage.py to place the mesh next to the NuRec splat.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from skimage.measure import marching_cubes


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--depth-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--voxel", type=float, default=0.0125, help="Voxel edge in scene units (room05: ~1 cm).")
    p.add_argument("--trunc-voxels", type=float, default=5.0, help="SDF truncation, in voxels.")
    p.add_argument("--max-depth", type=float, default=4.0, help="Ignore depth beyond this (scene units).")
    p.add_argument("--crop-margin", type=float, default=1.5,
                   help="Mesh only within the camera positions' bounding box grown by this (scene units). "
                        "Depth seen through doorways otherwise stretches the grid ~5x.")
    p.add_argument("--weight-threshold", type=float, default=3.0, help="Only surface seen by at least this many views.")
    p.add_argument("--min-component-tris", type=int, default=2000, help="Drop disconnected bits smaller than this.")
    p.add_argument("--target-tris", type=int, default=500_000)
    p.add_argument("--device", default="CPU:0", help="Open3D device for integration (CPU:0 or CUDA:0).")
    p.add_argument("--block-count", type=int, default=600_000, help="Voxel blocks (8^3 voxels each) to preallocate.")
    p.add_argument("--workers", type=int, default=os.cpu_count(), help="Processes for marching cubes.")
    return p.parse_args()


def integrate(frames, intr, a):
    dev = o3d.core.Device(a.device)
    K = o3d.core.Tensor(np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]]), o3d.core.float64)
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"), attr_dtypes=(o3d.core.float32, o3d.core.float32, o3d.core.float32),
        attr_channels=((1), (1), (3)), voxel_size=a.voxel, block_resolution=8, block_count=a.block_count, device=dev)
    print(f"integrating {len(frames)} frames on {dev} | voxel {a.voxel} | trunc {a.voxel * a.trunc_voxels:.4f}")
    t0 = time.time()
    for i, f in enumerate(frames):
        d = np.load(f)
        depth = o3d.t.geometry.Image(o3d.core.Tensor(d["depth"].astype(np.float32))).to(dev)
        color = o3d.t.geometry.Image(o3d.core.Tensor(d["rgb"].astype(np.float32) / 255.0)).to(dev)
        E = o3d.core.Tensor(np.linalg.inv(d["c2w"]), o3d.core.float64)
        blocks = vbg.compute_unique_block_coordinates(depth, K, E, 1.0, a.max_depth, trunc_voxel_multiplier=a.trunc_voxels)
        vbg.integrate(blocks, depth, color, K, K, E, 1.0, a.max_depth, trunc_voxel_multiplier=a.trunc_voxels)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(frames)} | {time.time() - t0:.0f}s | blocks {vbg.hashmap().size():,}")
    return vbg, time.time() - t0


_GRID = {}  # dense arrays shared with forked workers (copy-on-write, read-only)


def _mc_slab(bounds):
    x0, x1 = bounds
    vol, seen = _GRID["vol"][x0:x1 + 1], _GRID["seen"][x0:x1 + 1]
    if seen.sum() == 0 or vol[seen].min() > 0 or vol[seen].max() < 0:
        return None
    try:
        v, f, _, _ = marching_cubes(vol, level=0.0, mask=seen, allow_degenerate=False)
    except (ValueError, RuntimeError):  # no surface in this slab
        return None
    v[:, 0] += x0
    return v.astype(np.float32), f.astype(np.int64)


def extract(vbg, voxel, weight_threshold, workers, box):
    """Marching cubes over the voxels seen by >= weight_threshold views, on a dense
    grid cropped to those voxels, in parallel slabs along x."""
    coords, flat = vbg.voxel_coordinates_and_flattened_indices()
    coords, flat = coords.cpu().numpy(), flat.cpu().numpy()
    weight = vbg.attribute("weight").cpu().numpy().reshape(-1)[flat]
    keep = (weight >= weight_threshold) & np.all((coords >= box[0]) & (coords <= box[1]), axis=1)
    coords, flat = coords[keep], flat[keep]
    del weight, keep
    tsdf = vbg.attribute("tsdf").cpu().numpy().reshape(-1)[flat]
    color = vbg.attribute("color").cpu().numpy().reshape(-1, 3)[flat]
    origin = coords.min(0)
    ijk = np.round((coords - origin) / voxel).astype(np.int64)
    shape = tuple(int(n) for n in ijk.max(0) + 1)
    print(f"dense grid {shape} = {np.prod(shape) / 1e6:.0f}M voxels, {len(flat) / 1e6:.0f}M observed")
    assert np.prod(shape) < 4e9, "grid too large for a dense pass; raise --voxel or lower --max-depth"
    vol = np.ones(shape, np.float32)  # unobserved = empty side of the surface
    seen = np.zeros(shape, bool)
    vol[tuple(ijk.T)] = tsdf
    seen[tuple(ijk.T)] = True
    _GRID.update(vol=vol, seen=seen)
    step = max(8, -(-shape[0] // (4 * workers)))
    slabs = [(x, min(x + step, shape[0] - 1)) for x in range(0, shape[0] - 1, step)]
    with mp.get_context("fork").Pool(workers) as pool:
        parts = [r for r in pool.map(_mc_slab, slabs) if r is not None]
    _GRID.clear()
    offs = np.cumsum([0] + [len(v) for v, _ in parts])
    verts = np.concatenate([v for v, _ in parts])
    faces = np.concatenate([f + o for (_, f), o in zip(parts, offs)])
    # nearest observed voxel's colour
    index = np.full(shape, -1, np.int32)
    index[tuple(ijk.T)] = np.arange(len(flat), dtype=np.int32)
    near = index[tuple(np.round(verts).astype(np.int64).clip(0, np.array(shape) - 1).T)]
    vcol = np.where(near[:, None] >= 0, color[near.clip(0)], 0.5).clip(0, 1)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(origin + verts.astype(np.float64) * voxel),
                                     o3d.utility.Vector3iVector(faces))
    mesh.vertex_colors = o3d.utility.Vector3dVector(vcol)
    mesh.merge_close_vertices(voxel * 1e-3)  # slab seams produce the same vertex twice
    mesh.remove_degenerate_triangles()
    return mesh


def main():
    a = parse_args()
    dd, out = Path(a.depth_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((dd / "meta.json").read_text())
    frames = sorted((dd / "frames").glob("*.npz"))
    report = {"settings": vars(a), "intrinsics": meta["intrinsics"], "n_frames": len(frames),
              "depth": {k: meta.get(k) for k in ("stop_transmittance", "min_opacity", "edge_rel", "stride")}}

    vbg, report["integrate_seconds"] = integrate(frames, meta["intrinsics"], a)
    report["blocks_used"] = int(vbg.hashmap().size())
    t0 = time.time()
    cams = np.array([np.load(f)["c2w"][:3, 3] for f in frames])
    box = (cams.min(0) - a.crop_margin, cams.max(0) + a.crop_margin)
    report["crop_box"] = [box[0].tolist(), box[1].tolist()]
    mesh = extract(vbg, a.voxel, a.weight_threshold, a.workers, box)
    del vbg
    report["extract_seconds"] = time.time() - t0

    n_raw = len(mesh.triangles)
    labels, counts, _ = mesh.cluster_connected_triangles()
    labels, counts = np.asarray(labels), np.asarray(counts)
    mesh.remove_triangles_by_mask(counts[labels] < a.min_component_tris)
    mesh.remove_unreferenced_vertices()
    report.update(raw_tris=n_raw, kept_tris=len(mesh.triangles),
                  components_kept=int((counts >= a.min_component_tris).sum()),
                  components_dropped=int((counts < a.min_component_tris).sum()))
    print(f"mesh: {n_raw:,} tris -> {len(mesh.triangles):,} after dropping {report['components_dropped']:,} small components")
    o3d.io.write_triangle_mesh(str(out / "tsdf_full_colmap.ply"), mesh)

    small = mesh.simplify_quadric_decimation(a.target_tris) if len(mesh.triangles) > a.target_tris else mesh
    small.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(out / "tsdf_500k_colmap.ply"), small)
    report["decimated_tris"] = len(small.triangles)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(f"wrote {out}: full {len(mesh.triangles):,} tris, decimated {len(small.triangles):,} | "
          f"integrate {report['integrate_seconds']:.0f}s, extract {report['extract_seconds']:.0f}s")


if __name__ == "__main__":
    main()
