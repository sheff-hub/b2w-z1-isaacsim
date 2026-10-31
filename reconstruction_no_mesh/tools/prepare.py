"""Stage 1 - stage the workspace and prepare images.

Converts RAW to JPEG (COLMAP cannot read DNG), links the frames into the scene,
and builds the downscaled set used for training.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps

from .common import Log, deep_get


def _convert_one(job):
    src, dst, long_side, quality = job
    try:
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im)      # bake orientation: COLMAP ignores the flag
            if im.mode != "RGB":
                im = im.convert("RGB")
            w, h = im.size
            s = long_side / max(w, h)
            if s < 1.0:
                im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
            im.save(dst, "JPEG", quality=quality, subsampling=0)
            return im.size, None
    except Exception as e:
        return None, f"{Path(src).name}: {type(e).__name__}: {e}"


def _batch(jobs, workers, log: Log, label: str):
    sizes, errs = set(), []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (sz, err) in enumerate(ex.map(_convert_one, jobs), 1):
            if err:
                errs.append(err)
            else:
                sizes.add(sz)
            if i % 100 == 0:
                log(f"  {label}: {i}/{len(jobs)}")
    if errs:
        raise SystemExit(f"{len(errs)} conversions failed, e.g. {errs[:3]}")
    if len(sizes) != 1:
        raise SystemExit(f"{label} produced mixed sizes {sizes}; single-camera SfM needs one size")
    return sizes.pop()


def run_stage(cfg: dict, scan: dict, src_dir: Path, scene: Path, log: Log) -> dict:
    log.rule("STAGE 1 - prepare")
    scene.mkdir(parents=True, exist_ok=True)
    images = scene / "images"
    workers = min(16, os.cpu_count() or 4)

    if scan["is_raw"]:
        # RAW: convert once into the scene. Not a link — the files are new.
        images.mkdir(exist_ok=True)
        long_side = deep_get(cfg, "prepare.raw_long_side", 2000)
        q = deep_get(cfg, "prepare.jpeg_quality", 95)
        todo = [(f, str(images / (Path(f).stem + ".jpg")), long_side, q)
                for f in scan["files"] if not (images / (Path(f).stem + ".jpg")).exists()]
        if todo:
            log(f"converting {len(todo)} RAW frames -> JPEG (long side {long_side})")
            size = _batch(todo, workers, log, "raw->jpeg")
            log(f"converted to {size}")
        else:
            log("RAW already converted; skipping")
    else:
        # Already JPEG/PNG: link rather than copy. A relative symlink keeps the
        # scene portable and resolves inside docker when the parent is mounted.
        if images.is_symlink() or images.exists():
            if images.is_symlink():
                images.unlink()
            elif images.is_dir() and not any(images.iterdir()):
                images.rmdir()
        if not images.exists():
            try:
                images.symlink_to(os.path.relpath(src_dir, scene))
                log(f"linked images -> {os.readlink(images)}")
            except OSError:
                import shutil
                shutil.copytree(src_dir, images)
                log("symlink unavailable; copied images")

    n = len([p for p in images.iterdir() if p.is_file()])
    with Image.open(next(p for p in sorted(images.iterdir()) if p.is_file())) as im:
        w, h = im.size
    log(f"scene images  : {n} at {w}x{h}")
    return {"images_dir": str(images), "count": n, "width": w, "height": h}


def choose_downsample(cfg: dict, count: int, w: int, h: int, log: Log) -> int:
    """Pick `dataset.downsample_factor` so training fits in VRAM.

    Cost is driven by pixels, not photo count. 932 photos at 12 MP would need
    roughly 63 GB for the image cache alone; at factor 2 it is comfortable.
    """
    forced = deep_get(cfg, "train.downsample_factor")
    if forced:
        log(f"downsample_factor : {forced} (from config)")
        return int(forced)
    from .common import gpu_free_gb
    free = gpu_free_gb(uuid=str(deep_get(cfg, "gpu.uuid", "all"))) or 40.0
    budget = max(4.0, free - deep_get(cfg, "gpu.reserve_gb", 8))
    for f in (1, 2, 3, 4):
        gb = count * (w / f) * (h / f) * 3 * 4 / 1e9      # float32 RGB
        if gb <= budget:
            log(f"downsample_factor : {f}  (~{gb:.1f} GB images, {budget:.0f} GB budget, "
                f"{round(w/f)}x{round(h/f)})")
            return f
    log("downsample_factor : 4 (largest reduction available)")
    return 4


def build_downsampled(scene: Path, factor: int, cfg: dict, log: Log) -> None:
    """3DGRUT reads `images_<factor>`; it does not downscale on the fly."""
    if factor == 1:
        return
    out = scene / f"images_{factor}"
    src = scene / "images"
    srcs = sorted(p for p in src.iterdir() if p.is_file())
    out.mkdir(exist_ok=True)
    # Keep the ORIGINAL filename, case included. COLMAP records names exactly as
    # they appear on disk, and 3DGRUT looks up images_<f>/<that name>. Writing
    # "IMG_1.jpg" for a source "IMG_1.JPG" yields FileNotFoundError at the first
    # training batch, ~3 minutes in.
    todo = [(str(p), str(out / p.name), 0, 0) for p in srcs
            if not (out / p.name).exists()]
    if not todo:
        log(f"images_{factor}: already present ({len(list(out.iterdir()))} files)")
        return
    q = deep_get(cfg, "prepare.jpeg_quality", 95)
    with Image.open(srcs[0]) as im:
        w, h = im.size
    jobs = [(s, d, max(w, h) // factor, q) for s, d, _, _ in todo]
    log(f"building images_{factor} ({len(jobs)} frames at {w//factor}x{h//factor})")
    _batch(jobs, min(16, os.cpu_count() or 4), log, f"images_{factor}")
