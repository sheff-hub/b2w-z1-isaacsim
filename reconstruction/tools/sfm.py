"""Stages 2-3 - COLMAP structure-from-motion, and verifying its output.

Follows NVIDIA's mono recipe, with three corrections learned the hard way:

  1. Two flags in the published recipe are HARD ERRORS on COLMAP 4.2:
        --SiftExtraction.max_image_size -> --FeatureExtraction.max_image_size
        --SiftMatching.use_gpu          -> --FeatureMatching.use_gpu
  2. `colmap mapper` can emit SEVERAL sub-models (sparse/0, sparse/1, ...).
     3DGRUT reads sparse/0 unconditionally, so the largest model must be
     promoted there. We once got a 2-image sparse/0 next to a 498-image
     sparse/1 — training would have silently used two photos.
  3. Intrinsics are seeded from EXIF when available.
"""
from __future__ import annotations

import re
import shutil
import struct
from pathlib import Path

from .common import Log, deep_get, docker_run


def _pick_max_image_size(cfg: dict, count: int, w: int, h: int, log: Log) -> int:
    v = deep_get(cfg, "colmap.max_image_size")
    if v:
        log(f"feature resolution: {v} (from config)")
        return int(v)
    below = deep_get(cfg, "colmap.auto_max_image_size.native_below_images", 500)
    if count < below:
        v = max(w, h)
        log(f"feature resolution: {v} (native; {count} < {below} images, so features are precious)")
    else:
        v = deep_get(cfg, "colmap.auto_max_image_size.default", 2000)
        log(f"feature resolution: {v} ({count} images; native would cost far more time)")
    return int(v)


def _seed_params(scan: dict, w: int, h: int, log: Log) -> str | None:
    f35 = [k for k, n in scan.get("focal35", {}).items() if k and n]
    if not f35:
        log("intrinsics: no EXIF focal length; letting COLMAP estimate")
        return None
    fpx = (float(f35[0]) / 36.0) * max(w, h)
    log(f"intrinsics: seeded fx=fy={fpx:.1f} from {f35[0]}mm-equivalent EXIF")
    return f"{fpx:.4f},{fpx:.4f},{w/2:.1f},{h/2:.1f}"


def run_colmap(cfg: dict, scene: Path, scan: dict, img: dict, log: Log) -> None:
    log.rule("STAGE 2 - COLMAP")
    image_root = scene.parent                     # mount the parent so symlinks resolve
    rel = scene.name
    cwork = f"/work/{rel}"
    mounts = {str(image_root): "/work"}
    image = deep_get(cfg, "colmap.docker_image", "colmap/colmap:latest")
    (scene / "sparse").mkdir(parents=True, exist_ok=True)

    mis = _pick_max_image_size(cfg, img["count"], img["width"], img["height"], log)
    seed = _seed_params(scan, img["width"], img["height"], log) \
        if deep_get(cfg, "colmap.seed_intrinsics", True) else None

    ex = ["colmap", "feature_extractor",
          "--database_path", f"{cwork}/database.db",
          "--image_path", f"{cwork}/images",
          "--ImageReader.single_camera", "1",
          "--ImageReader.camera_model", deep_get(cfg, "colmap.camera_model", "PINHOLE"),
          "--FeatureExtraction.max_image_size", str(mis)]
    if seed:
        ex += ["--ImageReader.camera_params", seed]
    if deep_get(cfg, "colmap.estimate_affine_shape", True):
        ex += ["--SiftExtraction.estimate_affine_shape", "1"]
    if deep_get(cfg, "colmap.domain_size_pooling", True):
        ex += ["--SiftExtraction.domain_size_pooling", "1"]
    log("[1/3] feature_extractor")
    docker_run(cfg, image, ex, mounts, log)

    matcher = deep_get(cfg, "colmap.matcher", "exhaustive")
    log(f"[2/3] {matcher}_matcher"
        + ("  (compares every pair; cost grows with the square of the count)"
           if matcher == "exhaustive" else ""))
    docker_run(cfg, image, ["colmap", f"{matcher}_matcher",
                            "--database_path", f"{cwork}/database.db",
                            "--FeatureMatching.use_gpu", "1"], mounts, log)

    log("[3/3] mapper  (incremental SfM — usually the long pole)")
    docker_run(cfg, image, ["colmap", "mapper",
                            "--database_path", f"{cwork}/database.db",
                            "--image_path", f"{cwork}/images",
                            "--output_path", f"{cwork}/sparse"], mounts, log)


def _analyze(cfg: dict, scene: Path, sub: str, log: Log) -> dict:
    out = docker_run(cfg, deep_get(cfg, "colmap.docker_image", "colmap/colmap:latest"),
                     ["colmap", "model_analyzer", "--path", f"/work/sparse/{sub}"],
                     {str(scene): "/work"}, log).stdout
    def grab(pat, cast=float):
        m = re.search(pat, out)
        return cast(m.group(1)) if m else 0
    return {"sub": sub,
            "images": grab(r"Registered images:\s*(\d+)", int),
            "points": grab(r"Points:\s*(\d+)", int),
            "track": grab(r"Mean track length:\s*([\d.]+)"),
            "reproj": grab(r"Mean reprojection error:\s*([\d.]+)")}


def verify_and_promote(cfg: dict, scene: Path, total_images: int, log: Log) -> dict:
    """THE step people skip. Check every sub-model, promote the best."""
    log.rule("STAGE 3 - verify SfM")
    sparse = scene / "sparse"
    subs = sorted([p.name for p in sparse.iterdir() if p.is_dir() and p.name.isdigit()])
    if not subs:
        raise SystemExit(f"COLMAP produced no sub-model in {sparse}. SfM failed.")

    stats = [_analyze(cfg, scene, s, log) for s in subs]
    for s in stats:
        log(f"  sparse/{s['sub']}: {s['images']:>5} images  {s['points']:>8,} points  "
            f"reproj {s['reproj']:.3f} px")
    best = max(stats, key=lambda s: s["images"])

    if len(stats) > 1:
        log(f"!! {len(stats)} sub-models. 3DGRUT reads sparse/0 blindly — promoting "
            f"sparse/{best['sub']} ({best['images']} images).")
    if best["sub"] != "0":
        junk = sparse / "0"
        if junk.exists():
            shutil.move(str(junk), str(sparse / f"fragment_was0_{best['sub']}"))
        shutil.move(str(sparse / best["sub"]), str(sparse / "0"))
        best["sub"] = "0"

    pct = 100.0 * best["images"] / max(total_images, 1)
    log(f"registered    : {best['images']} / {total_images}  ({pct:.1f}%)")
    log(f"3D points     : {best['points']:,}")
    log(f"reprojection  : {best['reproj']:.3f} px")
    if best["reproj"] > 1.5:
        log("!! reprojection above 1.5 px — geometry is loose; expect softness")
    if pct < 80:
        log("!! under 80% of frames registered. Usually means insufficient overlap; "
            "more photos will help more than any tuning downstream.")
    return best


def read_points_count(scene: Path) -> int:
    """Point count straight from the binary model (no docker round-trip)."""
    p = scene / "sparse" / "0" / "points3D.bin"
    if p.exists():
        with open(p, "rb") as f:
            return struct.unpack("<Q", f.read(8))[0]
    t = scene / "sparse" / "0" / "points3D.txt"
    if t.exists():
        return sum(1 for ln in open(t) if ln.strip() and not ln.startswith("#"))
    return 0
