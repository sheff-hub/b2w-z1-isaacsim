"""Stage 6 - the collider: LiDAR depth fused into a mesh with the refined poses, decimated, then checked.

TSDF fusion (VDBFusion, 3 cm voxels, space carving) of every second LiDAR frame, flying pixels at depth edges
dropped; decimated to 1.2M triangles; then small holes in the floor (shiny floor returns no LiDAR at some angles)
patched at floor height by fill_floor.py -- the M1 collider had 127, the largest 36 x 36 cm.
Measured against the alternatives on the 22 Oct walk: nvblox (NVIDIA's recipe), Depth Anything 3 + VDB
(Stani's), nerfstudio Poisson. This one blocked 0% of the walked route and put 100% of the floor within 8 cm of
z = 0; the Poisson mesh of nerfstudio's docs was a 50k-triangle blob.
The checks (eval_mesh.py) answer the questions a robot asks: is the floor where the floor is, is the walked path
free, do walls leak.
"""
from __future__ import annotations

import json
from pathlib import Path

from .common import Log, deep_get, tool


def run_stage(cfg: dict, scene: Path, log: Log) -> dict:
    log.rule("STAGE 6 - collider mesh from LiDAR (TSDF fusion)")
    m = deep_get(cfg, "mesh", {}) or {}
    (scene / "mesh").mkdir(exist_ok=True)
    fused, collider = scene / "mesh" / "lidar_fused.ply", scene / "mesh" / "collider.ply"
    r = tool(cfg, "fuse_lidar.py", [scene, fused, "--voxel", m.get("voxel_m", 0.03), "--trunc", m.get("trunc_m", 0.09),
                                    "--max-depth", m.get("max_depth_m", 4.5), "--edge", m.get("edge", 0.04)], log)
    log("fused         : " + r.stdout.strip().splitlines()[-1])
    dec = scene / "mesh" / "lidar_decimated.ply"
    r = tool(cfg, "decimate.py", [fused, dec, "--triangles", m.get("triangles", 1_200_000)], log)
    log("decimated     : " + r.stdout.strip().splitlines()[-1])
    (scene / "eval").mkdir(exist_ok=True)
    holes = {}
    if m.get("fill_floor_holes", True):
        r = tool(cfg, "fill_floor.py", [dec, collider, "--max-area", m.get("hole_max_area_m2", 2.0),
                                        "--report", scene / "eval" / "floor_holes.json"], log)
        log("floor holes   : " + r.stdout.strip().splitlines()[-2].replace("floor holes: ", ""))
        holes = json.loads((scene / "eval" / "floor_holes.json").read_text())
        dec.unlink()
    else:
        dec.replace(collider)
    log(f"collider      : {collider}")
    if not m.get("keep_fused", False):
        fused.unlink()
    tool(cfg, "eval_mesh.py", [scene, scene / "eval", f"collider={collider}"], log)
    rep = json.loads((scene / "eval" / "report.json").read_text())["meshes"]["collider"]
    log(f"floor         : {rep['floor_hit_8cm']:.0%} of camera positions have floor within 8 cm of z = 0 "
        f"(median {rep['floor_z_median_cm']:+.1f} cm); {rep['no_floor']:.0%} have none")
    log(f"walked path   : {rep['path_blocked_0.2_to_1m']:.1%} blocked between 0.2 and 1 m (should be ~0)")
    log(f"walls         : {rep['leak_1.0m']:.1%} of horizontal rays at 1 m escape (open doors and glass count too)")
    log(f"accuracy      : {rep['sfm_within_5cm']:.0%} of the BA points within 5 cm of the mesh")
    if rep["path_blocked_0.2_to_1m"] > 0.02:
        log("!! more than 2% of the walked path is blocked: something solid where the phone walked (check eval/plan.png)")
    if rep["floor_hit_8cm"] < 0.9:
        log("!! the floor is missing or off under more than 10% of the path (check eval/xsec_collider.png)")
    return {"collider": str(collider), "floor_holes_patched": len(holes.get("patched", [])),
            "floor_holes_patched_m2": holes.get("patched_area_m2", 0.0), **{k: rep[k] for k in ("triangles", "components", "floor_hit_8cm", "no_floor",
                                                              "floor_z_median_cm", "path_blocked_0.2_to_1m", "leak_0.3m",
                                                              "leak_1.0m", "sfm_within_5cm", "ceiling_m_median")}}
