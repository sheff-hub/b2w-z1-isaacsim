"""Stage 0 - ingest and validate.

Cheap checks that catch the mistakes which otherwise surface an hour into
COLMAP: mixed lenses, mixed dimensions, stripped EXIF, motion blur.
"""
from __future__ import annotations

import collections
from pathlib import Path

from PIL import Image
from PIL.ExifTags import TAGS

from .common import Log, deep_get

IMG_EXT = {".jpg", ".jpeg", ".png", ".dng", ".tif", ".tiff"}


def _exif(im):
    ex = im.getexif()
    base = {TAGS.get(k, k): v for k, v in ex.items()}
    sub = {TAGS.get(k, k): v for k, v in ex.get_ifd(0x8769).items()}
    return base, sub


def scan(images_dir: Path, log: Log) -> dict:
    files = sorted(p for p in images_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in IMG_EXT)
    if not files:
        raise SystemExit(f"no images found in {images_dir}")

    dims, lens, iso, shutter, ori, f35 = (collections.Counter() for _ in range(6))
    unreadable = []
    for f in files:
        try:
            with Image.open(f) as im:
                dims[im.size] += 1
                b, s = _exif(im)
                ori[b.get("Orientation")] += 1
                lens[s.get("LensModel")] += 1
                iso[s.get("ISOSpeedRatings")] += 1
                f35[s.get("FocalLengthIn35mmFilm")] += 1
                et = s.get("ExposureTime")
                shutter[f"1/{round(1/float(et))}" if et else None] += 1
        except Exception as e:
            unreadable.append((f.name, type(e).__name__))

    return {
        "files": [str(f) for f in files], "count": len(files),
        "unreadable": unreadable, "dims": dict(dims), "orientation": dict(ori),
        "lens": dict(lens), "iso": dict(iso), "shutter": dict(shutter),
        "focal35": dict(f35),
        "is_raw": all(f.suffix.lower() == ".dng" for f in files),
    }


def photometrics(files, log: Log, sample_every: int = 1) -> dict:
    """Brightness and a Laplacian-variance sharpness proxy. Neither appears in
    metadata, and both cap achievable quality."""
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view
    K = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float)
    means, sharp = [], []
    for f in files[::sample_every]:
        try:
            im = Image.open(f)
            im.draft("L", (im.size[0] // 4, im.size[1] // 4))
            g = np.asarray(im.convert("L"), dtype=float)
            sharp.append(float((sliding_window_view(g, (3, 3)) * K).sum(axis=(2, 3)).var()))
            means.append(float(g.mean()))
        except Exception:
            pass
    if not means:
        return {}
    means, sharp = np.array(means), np.array(sharp)
    thr = float(np.median(sharp)) * 0.35
    return {
        "brightness_mean": float(means.mean()), "brightness_std": float(means.std()),
        "sharp_median": float(np.median(sharp)),
        "soft_count": int((sharp < thr).sum()), "dark_count": int((means < 60).sum()),
        "n": len(means),
    }


def run_stage(cfg: dict, images_dir: Path, log: Log) -> dict:
    log.rule("STAGE 0 - validate")
    info = scan(images_dir, log)
    log(f"images        : {info['count']}  ({images_dir})")
    log(f"dimensions    : {info['dims']}")
    log(f"orientation   : {info['orientation']}")
    log(f"lens          : {info['lens']}")
    log(f"ISO / shutter : {info['iso']} / {info['shutter']}")
    log(f"35mm equiv    : {info['focal35']}")

    if info["unreadable"]:
        log(f"!! unreadable : {len(info['unreadable'])} -> {info['unreadable'][:5]}")

    if deep_get(cfg, "validate.require_uniform_size", True) and len(info["dims"]) > 1:
        raise SystemExit(
            f"images have mixed dimensions {info['dims']}.\n"
            "COLMAP's single-camera model assumes one size. Fix the capture or "
            "set validate.require_uniform_size: false.")

    real_lenses = {k for k in info["lens"] if k}
    if deep_get(cfg, "validate.require_single_lens", True) and len(real_lenses) > 1:
        raise SystemExit(
            f"images use more than one lens: {real_lenses}\n"
            "This silently breaks --ImageReader.single_camera. Reshoot on one "
            "lens, or set validate.require_single_lens: false.")

    lo = deep_get(cfg, "validate.min_images", 100)
    if info["count"] < lo:
        raise SystemExit(f"only {info['count']} images; below validate.min_images ({lo}). "
                         "Coverage will be too thin to reconstruct.")

    pm = photometrics(info["files"], log)
    if pm:
        log(f"brightness    : mean {pm['brightness_mean']:.1f}  std {pm['brightness_std']:.1f}")
        log(f"soft frames   : {pm['soft_count']} / {pm['n']}   dark: {pm['dark_count']}")
        frac = pm["soft_count"] / max(pm["n"], 1)
        if frac > deep_get(cfg, "validate.max_soft_fraction", 0.25):
            log(f"!! {frac:.0%} of frames look soft — expect registration failures")
    info["photometrics"] = pm

    if not any(info["focal35"].values()):
        log("!! no EXIF focal length. COLMAP's fallback prior is 1.2*max(w,h), "
            "which can be badly wrong; intrinsics seeding will be skipped.")
    log(f"OK  {info['count']} images validated")
    return info
