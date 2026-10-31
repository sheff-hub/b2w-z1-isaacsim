"""Stages 3-5 - poses: COLMAP features and matches (and, with several passes, their alignment), loop closure, bundle
adjustment.

The phone's ARKit poses are the starting point, not COLMAP's own mapper: they are metric, gravity-aligned and
continuous. What is wrong with them on a long walk is drift and ARKit's relocalisation jumps (5 on the 407 s
two-lap walk of 22 Oct 2026, which also shifted the whole map 0.5 m vertically). So:
  3. COLMAP extracts SIFT features and matches only the pairs the poses suggest (time neighbours + spatial
     neighbours, including loop closures), not all pairs. Several passes (each its own ARKit world) are first put
     into one frame: a sparse set of cross-pass pairs is matched and align_passes.py turns the verified ones into the
     yaw-plane transform of each pass;
  4. loop_close.py removes ARKit's jumps and fits a pose graph to PnP loop edges (LiDAR depth, checked both ways);
     loop error 23 cm -> 0.4 cm on that walk. Edges between passes are loop edges like any other;
  5. refine_poses.py triangulates with those poses and bundle-adjusts: reprojection 1.16 -> 0.52 px (+1.7 dB in
     the splat on the first Record3D scene), then aligns back to the metric frame (scale x1.0015).
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np

from .common import Log, deep_get, docker_run, tool


def _colmap(cfg: dict, scene: Path, args: list[str], log: Log):
    return docker_run(cfg, deep_get(cfg, "colmap.docker_image", "colmap/colmap:latest"), ["colmap"] + args,
                      {str(scene): "/work"}, log)


def features_and_matches(cfg: dict, scene: Path, log: Log) -> dict:
    log.rule("STAGE 3 - COLMAP features + pose-guided matching")
    col = scene / "colmap"
    if col.exists():
        shutil.rmtree(col)            # a stale database would keep old images' features and matches
    col.mkdir()
    rec = scene / "record3d"
    for p in list(rec.glob("loop_edges_*.npz")) + [rec / "poses_aligned.npz", rec / "pass_align.json",
                                                   rec / "passes_aligned.png"]:
        p.unlink(missing_ok=True)     # loop edges are cached per database; the alignment is redone with it
    z = np.load(rec / "poses.npz")
    n_pass = len(z["srcs"]) if "srcs" in z else 1
    fx, fy, cx, cy = np.median(z["K"], axis=0)
    # one shared PINHOLE camera for extraction; refine_poses.py writes the per-focal-bin cameras before triangulating
    _colmap(cfg, scene, ["feature_extractor", "--database_path", "/work/colmap/database.db", "--image_path", "/work/images",
                         "--ImageReader.single_camera", "1", "--ImageReader.camera_model", "PINHOLE",
                         "--ImageReader.camera_params", f"{fx:.2f},{fy:.2f},{cx:.2f},{cy:.2f}",
                         "--FeatureExtraction.use_gpu", "1"], log)
    out = {}
    if n_pass > 1:
        al = deep_get(cfg, "align", {}) or {}
        r = tool(cfg, "align_passes.py", [scene, "pairs", "--per-pass", al.get("images_per_pass", 150),
                                          "--step-m", al.get("step_m", 0.75), "--step-deg", al.get("step_deg", 20)], log)
        log("alignment     : " + r.stdout.strip().splitlines()[-1])
        _match(cfg, scene, "pairs_align.txt", log)
        tool(cfg, "align_passes.py", [scene, "solve", "--min-inliers", al.get("min_inliers", 30),
                                      "--min-votes", al.get("min_votes", 5)], log)
        j = json.loads((rec / "pass_align.json").read_text())
        for p, t in j["transforms"].items():
            if p != "1":
                st = next((l for l in j["links"] if l["pair"] in j["used"] and int(p) in l["pair"]), {})
                log(f"pass {p} -> 1   : turned {t['yaw_deg']:+.1f} deg, shifted ({t['shift_m'][0]:+.2f}, "
                    f"{t['shift_m'][1]:+.2f}) m; {st.get('inliers', '?')} of {st.get('votes', '?')} votes agree")
        out["alignment"] = j
        out["align_pairs"] = sum(1 for _ in open(col / "pairs_align.txt"))
    else:
        tool(cfg, "align_passes.py", [scene, "solve"], log)      # one pass: poses_aligned.npz = poses.npz
    p = deep_get(cfg, "colmap.pairs", {}) or {}
    r = tool(cfg, "pairs.py", [scene, "--neighbours", p.get("neighbours", 8), "--radius", p.get("radius_m", 4.0),
                               "--angle", p.get("angle_deg", 75), "--near", p.get("near", 30), "--far", p.get("far", 30),
                               "--far-gap-s", p.get("far_gap_s", 10)], log)
    log("pairs         : " + r.stdout.strip().splitlines()[-1])
    _match(cfg, scene, "pairs.txt", log)
    out["pairs"] = sum(1 for _ in open(col / "pairs.txt")) + out.get("align_pairs", 0)
    return out


def _match(cfg: dict, scene: Path, pairs_file: str, log: Log):
    _colmap(cfg, scene, ["matches_importer", "--database_path", "/work/colmap/database.db",
                         "--match_list_path", f"/work/colmap/{pairs_file}", "--match_type", "pairs",
                         "--FeatureMatching.use_gpu", "1"], log)


def loop_close(cfg: dict, scene: Path, log: Log) -> dict:
    log.rule("STAGE 4 - loop closure (remove ARKit's jumps, pose graph on loop edges)")
    sp = scene / "sparse"
    if sp.exists():
        shutil.rmtree(sp)
    shutil.copytree(scene / "record3d" / "sparse_prep", sp / "0")    # the phone model as prep wrote it
    if (scene / "sparse_phone").exists():
        shutil.rmtree(scene / "sparse_phone")
    lc = deep_get(cfg, "loop_close", {}) or {}
    tool(cfg, "loop_close.py", [scene, "--min-gap-s", lc.get("min_gap_s", 10), "--min-inliers", lc.get("min_inliers", 40)], log)
    j = json.loads((scene / "record3d" / "loop_close.json").read_text())
    if j.get("pose_graph"):
        log(f"loop closure  : {j['loops_accepted']:,} loop edges; error {j['loop_err_before_cm']:.1f} cm -> "
            f"{j['loop_err_after_cm']:.1f} cm (p90 {j['loop_err_after_p90_cm']:.1f} cm); largest correction "
            f"{j['correction_max_cm']:.0f} cm")
    else:
        log(f"loop closure  : {j['loops_accepted']} loop edges: no loops to close (the walk does not revisit places, "
            "or too few verified matches); ARKit's trajectory with its jumps removed is used")
    return j


def refine(cfg: dict, scene: Path, log: Log) -> dict:
    log.rule("STAGE 5 - triangulate + bundle adjustment")
    if (scene / "sparse_phone").exists():
        shutil.rmtree(scene / "sparse_phone")   # refine_poses.py takes the phone model from sparse/0 when this is absent
    r = tool(cfg, "refine_poses.py", [scene, "poses_lc.npz"], log)
    out = {}
    for ln in r.stdout.splitlines():
        m = re.search(r"phone poses, fixed: ([\d,]+) points, reprojection mean ([\d.]+) px, median ([\d.]+) px", ln)
        if m:
            out["reproj_phone_px"] = float(m.group(3))
        m = re.search(r"written to sparse/0: ([\d,]+) points, reprojection mean ([\d.]+) px, median ([\d.]+) px", ln)
        if m:
            out.update(points=int(m.group(1).replace(",", "")), reproj_mean_px=float(m.group(2)),
                       reproj_median_px=float(m.group(3)))
        m = re.search(r"camera position change median ([\d.]+) cm, p95 ([\d.]+) cm, max ([\d.]+) cm", ln)
        if m:
            out.update(ba_shift_median_cm=float(m.group(1)), ba_shift_max_cm=float(m.group(3)))
        m = re.search(r"Sim3 back to the phone frame: scale ([\d.]+)", ln)
        if m:
            out["sim3_scale"] = float(m.group(1))
        m = re.search(r"(\d+) frames with < \d+ points keep their phone pose", ln)
        if m:
            out["frames_kept_phone_pose"] = int(m.group(1))
    if not out.get("points"):
        raise SystemExit("bundle adjustment wrote no points; see the refine_poses.py output above")
    log(f"BA            : reprojection {out.get('reproj_phone_px', '?')} px (phone poses) -> {out['reproj_median_px']} px "
        f"(median), {out['points']:,} points; cameras moved median {out.get('ba_shift_median_cm', '?')} cm")
    return out
