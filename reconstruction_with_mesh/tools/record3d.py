"""Stages 1-2 - check the Record3D recordings, then prepare them: metric floor-pinned poses, images, LiDAR depth.

Record3D (iPhone/iPad with LiDAR) is exported in the app as "EXR + JPG": a folder EXR_RGBD/ with metadata.json
(ARKit poses, per-frame intrinsics, timestamps), rgb/<i>.jpg (720 x 960 at 60 fps on an iPhone 14 Pro) and
depth/<i>.exr (LiDAR, 192 x 256, metres).

A scene is one recording or several PASSES of the same place (e.g. the phone at three heights), plus optionally a
TEST clip that is never trained on and only scores the result (stage test_eval). They can be given as
  - one export folder (it may hold EXR_RGBD/ or be the EXR_RGBD folder itself): one pass;
  - a folder of export folders: every export inside is a pass, in name order; one whose folder name starts with
    "test" is the test clip;
  - a scene file listing them (recording.passes, recording.test).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from .common import Log, deep_get, expand, tool

MAX_FRAMES = 999_999     # images are named f000000.jpg ... (6 digits), over all passes


def is_export(p: Path) -> bool:
    return (p / "metadata.json").is_file() and (p / "rgb").is_dir()


def find_export(src: Path) -> Path:
    """The folder with metadata.json, rgb/ and depth/: src itself, src/EXR_RGBD, or the one subfolder that has them."""
    for c in [src, src / "EXR_RGBD"] + sorted(p for p in src.glob("*/") if p.is_dir()):
        if is_export(c):
            return c
    raise SystemExit(
        f"no Record3D export found in {src}\n"
        "Expected metadata.json + rgb/ + depth/ (Record3D: Library -> recording -> Export -> 'EXR + JPG'), either in\n"
        "this folder or in its EXR_RGBD/ subfolder.")


def find_exports(root: Path) -> tuple[list[Path], Path | None]:
    """(passes, test clip) in a folder: the folder itself if it is (or holds) one export, else every export up to
    three levels down, in name order; one under a folder named test* is the test clip."""
    for c in (root, root / "EXR_RGBD"):
        if is_export(c):
            return [c], None
    found = sorted({p.parent for p in root.glob("**/metadata.json")
                    if len(p.relative_to(root).parts) <= 4 and is_export(p.parent)})
    if not found:
        find_export(root)           # raises with the explanation
    is_test = lambda p: any(part.lower().startswith("test") for part in p.relative_to(root).parts)
    tests = [p for p in found if is_test(p)]
    if len(tests) > 1:
        raise SystemExit(f"more than one test clip in {root}: {[str(t.relative_to(root)) for t in tests]}")
    passes = [p for p in found if not is_test(p)]
    if not passes:
        raise SystemExit(f"{root} holds only a test clip, no recording to reconstruct from")
    return passes, (tests[0] if tests else None)


def resolve_recordings(cfg: dict, target: str | None) -> tuple[list[Path], Path | None]:
    """The passes and the test clip of this run, from recording.passes / recording.name / recording.test."""
    root = expand(deep_get(cfg, "paths.recordings_root", "."))
    at = lambda n: Path(n).expanduser() if Path(n).expanduser().is_absolute() else root / n
    names = deep_get(cfg, "recording.passes") or ([deep_get(cfg, "recording.name")] if deep_get(cfg, "recording.name") else [])
    if not names:
        raise SystemExit("no recording given. Pass a recording folder (under paths.recordings_root) or a scene file "
                         "(see --list-scenes).")
    passes, test = [], None
    for n in names:
        d = at(n)
        if not d.is_dir():
            raise SystemExit(f"recording folder not found: {d}")
        ps, t = find_exports(d)
        passes += ps
        test = test or t
    if deep_get(cfg, "recording.test"):
        d = at(deep_get(cfg, "recording.test"))
        if not d.is_dir():
            raise SystemExit(f"test clip folder not found: {d}")
        test = find_export(d)
    return passes, test


def _validate_one(cfg: dict, ex: Path, log: Log, tag: str) -> dict:
    m = json.loads((ex / "metadata.json").read_text())
    missing = [k for k in ("poses", "perFrameIntrinsicCoeffs", "frameTimestamps", "w", "h", "dw", "dh") if k not in m]
    if missing:
        raise SystemExit(f"{ex}: metadata.json lacks {missing}: not a Record3D EXR + JPG export")
    P = np.asarray(m["poses"], float)
    n = len(P)
    ts = np.asarray(m["frameTimestamps"], float)
    fps = float(m.get("fps") or (1.0 / np.median(np.diff(ts)) if n > 1 else 60.0))
    n_rgb = sum(1 for _ in (ex / "rgb").glob("*.jpg"))
    depth_dir = ex / "depth"
    n_exr = sum(1 for _ in depth_dir.glob("*.exr")) if depth_dir.is_dir() else 0
    n_other = sum(1 for p in depth_dir.glob("*") if p.suffix.lower() != ".exr") if depth_dir.is_dir() else 0
    log(f"{tag:<14}: {ex}")
    log(f"  frames      : {n:,} poses, {n_rgb:,} rgb/*.jpg, {n_exr:,} depth/*.exr")
    if n_exr == 0:
        raise SystemExit(f"{ex}: no depth/*.exr: export the recording as 'EXR + JPG' (LiDAR depth is needed for the "
                         "collider" + (f"; found {n_other} other depth files)" if n_other else ")"))
    if n_rgb < n or n_exr < n:
        raise SystemExit(f"{ex}: incomplete export: {n} poses but {n_rgb} images and {n_exr} depth maps")
    for i in (0, n // 2, n - 1):
        for p in (ex / "rgb" / f"{i}.jpg", ex / "depth" / f"{i}.exr"):
            if not p.is_file() or p.stat().st_size == 0:
                raise SystemExit(f"missing or empty frame file: {p}")
    dur = n / fps
    step = np.linalg.norm(np.diff(P[:, 4:], axis=0), axis=1)
    path = float(step.sum())
    jumps = int(np.sum(step > 0.04))
    K = np.median(np.asarray(m["perFrameIntrinsicCoeffs"], float), axis=0)
    log(f"  duration    : {dur:.0f} s at {fps:.0f} fps; walked {path:.0f} m (ARKit); {jumps} ARKit jumps "
        "(relocalisations; removed later, then the loops are closed)")
    log(f"  images      : {m['w']} x {m['h']}, depth {m['dw']} x {m['dh']}; fx {K[0]:.1f} cx {K[2]:.1f} cy {K[3]:.1f}")
    return {"export_dir": str(ex), "frames": n, "fps": fps, "duration_s": round(dur, 1), "path_m": round(path, 1),
            "arkit_jumps": jumps, "width": int(m["w"]), "height": int(m["h"]), "depth_w": int(m["dw"]),
            "depth_h": int(m["dh"]), "fx": round(float(K[0]), 2)}


def validate(cfg: dict, passes: list[Path], test: Path | None, log: Log) -> dict:
    log.rule("STAGE 1 - validate the recording" + ("s" if len(passes) > 1 or test else ""))
    ps = [_validate_one(cfg, ex, log, f"pass {k + 1}" if len(passes) > 1 else "recording") for k, ex in enumerate(passes)]
    min_s = float(deep_get(cfg, "validate.min_seconds", 20))
    for k, v in enumerate(ps):
        if v["duration_s"] < min_s:
            raise SystemExit(f"{v['export_dir']} is only {v['duration_s']:.0f} s (< validate.min_seconds = {min_s:.0f})")
        if v["path_m"] < float(deep_get(cfg, "validate.min_path_m", 3.0)):
            log(f"!! pass {k + 1}: only {v['path_m']:.1f} m walked: a scene needs parallax; expect a poor splat")
    sizes = {(v["width"], v["height"], v["depth_w"], v["depth_h"]) for v in ps}
    if len(sizes) > 1:
        raise SystemExit(f"the passes have different image sizes {sorted(sizes)}: record all with the same settings")
    total = sum(v["frames"] for v in ps)
    if total > MAX_FRAMES:
        raise SystemExit(f"{total:,} frames in all is more than {MAX_FRAMES:,} (~4.6 h at 60 fps)")
    t = None
    if test:
        t = _validate_one(cfg, test, log, "test clip")
        if (t["width"], t["height"]) != (ps[0]["width"], ps[0]["height"]):
            raise SystemExit(f"the test clip is {t['width']} x {t['height']}, the passes {ps[0]['width']} x "
                             f"{ps[0]['height']}: record it with the same settings")
    if len(ps) > 1:
        log(f"passes        : {len(ps)}, {total:,} frames, {sum(v['duration_s'] for v in ps) / 60:.1f} min, "
            f"{sum(v['path_m'] for v in ps):.0f} m (ARKit)")
    first = ps[0]
    return {"passes": ps, "test": t, "n_passes": len(ps), "export_dir": first["export_dir"], "fps": first["fps"],
            "frames": total, "duration_s": round(sum(v["duration_s"] for v in ps), 1),
            "path_m": round(sum(v["path_m"] for v in ps), 1), "arkit_jumps": sum(v["arkit_jumps"] for v in ps),
            "width": first["width"], "height": first["height"], "depth_w": first["depth_w"],
            "depth_h": first["depth_h"], "fx": first["fx"]}


def prep_args(cfg: dict, section: str = "prep") -> list:
    """Frame selection and floor settings for prep_record3d.py, from config section `prep` (or `test`)."""
    g = lambda k, d: deep_get(cfg, f"{section}.{k}", deep_get(cfg, f"prep.{k}", d))
    args = ["--image-step-m", g("image_step_m", 0.06), "--image-step-deg", g("image_step_deg", 3.0),
            "--min-sharpness", g("min_sharpness", 0.25), "--depth-step-m", g("depth_step_m", 0.015),
            "--depth-step-deg", g("depth_step_deg", 1.0), "--floor-factor", g("floor_factor", 0.6),
            "--floor-below", deep_get(cfg, f"{section}.floor_below", 0) or 0]
    if section == "prep" and deep_get(cfg, "prep.keep_every"):
        args += ["--keep-every", deep_get(cfg, "prep.keep_every")]
    return args


def prep(cfg: dict, info: dict, scene: Path, log: Log) -> dict:
    log.rule("STAGE 2 - prepare: floor-pinned poses, sharpest images, LiDAR depth")
    for d in ("images", "sparse", "sparse_phone", "record3d"):   # a fresh start: stale files would mix with new ones
        if (scene / d).exists():
            shutil.rmtree(scene / d)
    args = [scene]
    for v in info["passes"]:
        args += ["--pass", v["export_dir"]]
    tool(cfg, "prep_record3d.py", args + prep_args(cfg), log)
    # the phone model as prep wrote it: loop_close rewrites sparse/0 and refine replaces it, so a re-run starts here
    shutil.copytree(scene / "sparse" / "0", scene / "record3d" / "sparse_prep")
    pj = json.loads((scene / "record3d" / "prep.json").read_text())
    z = np.load(scene / "record3d" / "poses.npz")
    h = z["c2w"][:, 2, 3]
    for k, p in enumerate(pj["passes"]):
        log(f"{'pass ' + str(k + 1) if len(pj['passes']) > 1 else 'recording':<14}: {p['images']:,} images "
            f"({p['blurred_dropped']} blurred slots dropped), {p['depth_frames']:,} LiDAR frames; camera "
            f"{p['camera_height_m']:.2f} m above the floor (p5 {p['camera_height_p5']:.2f}, p95 {p['camera_height_p95']:.2f}); "
            f"floor found under {p['floor_found']:.0%} of the sampled frames")
    log(f"images        : {pj['images']:,} -> {scene / 'images'} ({pj['image_rule']})")
    return {"images": pj["images"], "image_rule": pj["image_rule"], "depth_frames": pj["depth_frames"],
            "snap_events": int(len(z["snaps"])), "camera_height_m": round(float(np.median(h)), 3),
            "passes": pj["passes"]}
