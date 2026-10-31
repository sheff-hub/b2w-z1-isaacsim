"""Stage 11 - score the splat on a TEST CLIP: a separate Record3D recording of the same place that is never trained on.

The PSNR that training reports is measured on every 8th image of the training walk itself: views right between
other training views. It cannot show how the scene looks from where the phone did not go (another height, across a
room), which is what a robot or a viewer in Isaac Sim sees. A test clip can, and the same clip scores any scene of
that place (e.g. the 22 Oct one-height walk against a three-height capture):

    python3 reconstruct.py <scene> --only test_eval --test-clip <Record3D folder>

Steps: prep the clip (prep_record3d.py --prefix t: floor-pinned ARKit poses, one image every 25 cm / 8 deg); COLMAP
features for the clip and the scene's images (the scene's database is copied when it still exists); place the clip
in the scene and localise every test image against the scene's bundle-adjusted cameras (localize_test.py); render
the splat there with 3DGRUT's own evaluation (render_test.py). Not fatal: a failure is reported, the scene is fine.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from .common import Log, deep_get, run, tool, venv_env, venv_python, PKG
from .record3d import _validate_one, prep_args
from .sfm import _colmap
from .train import wait_for_gpu

BANDS = [(k / 4, (k + 1) / 4, f"at {k / 4:.2f}-{(k + 1) / 4:.2f} m") for k in range(12)]   # camera height, 25 cm bins


def _extract(cfg, scene: Path, image_dir: str, K: np.ndarray, log: Log):
    fx, fy, cx, cy = np.median(K, axis=0)
    _colmap(cfg, scene, ["feature_extractor", "--database_path", "/work/test/colmap/database.db",
                         "--image_path", f"/work/{image_dir}", "--ImageReader.single_camera", "1",
                         "--ImageReader.camera_model", "PINHOLE",
                         "--ImageReader.camera_params", f"{fx:.2f},{fy:.2f},{cx:.2f},{cy:.2f}",
                         "--FeatureExtraction.use_gpu", "1"], log)


def _match(cfg, scene: Path, pairs: str, log: Log):
    _colmap(cfg, scene, ["matches_importer", "--database_path", "/work/test/colmap/database.db",
                         "--match_list_path", f"/work/test/colmap/{pairs}", "--match_type", "pairs",
                         "--FeatureMatching.use_gpu", "1"], log)


def run_stage(cfg: dict, scene: Path, state: dict, test_export: Path | None, log: Log) -> dict:
    log.rule("STAGE 11 - score the splat on the test clip (never trained on)")
    if test_export is None:
        log("no test clip given (recording.test, a test* folder next to the passes, or --test-clip): skipped")
        return {"skipped": True}
    best = (state["stages"].get("train", {}) or {}).get("best") or {}
    ckpt = Path(best.get("dir", "")) / "ckpt_last.pt"
    if not ckpt.exists():
        log(f"!! no trained splat found ({ckpt}); run the train stage first")
        return {"ok": False, "error": "no trained splat"}
    try:
        return _run(cfg, scene, test_export, ckpt, best, log)
    except SystemExit as e:
        log(f"!! the test clip could not be scored: {e}")
        log("   (the scene itself is fine; fix the cause and run --only test_eval)")
        return {"ok": False, "error": str(e)}


def _run(cfg, scene: Path, test_export: Path, ckpt: Path, best: dict, log: Log) -> dict:
    v = _validate_one(cfg, test_export, log, "test clip")
    test = scene / "test"
    if test.exists():
        shutil.rmtree(test)
    tc = deep_get(cfg, "test", {}) or {}
    tool(cfg, "prep_record3d.py", [test, "--pass", v["export_dir"], "--prefix", "t", "--no-depth",
                                   "--max-images", tc.get("max_images", 200)] + prep_args(cfg, "test"), log)
    pj = json.loads((test / "record3d" / "prep.json").read_text())["passes"][0]
    log(f"test images   : {pj['images']} (camera {pj['camera_height_m']:.2f} m above the floor, p5 "
        f"{pj['camera_height_p5']:.2f}, p95 {pj['camera_height_p95']:.2f})")
    (test / "colmap").mkdir()
    db = scene / "colmap" / "database.db"
    if db.exists():
        shutil.copy(db, test / "colmap" / "database.db")
        log("features      : the scene's COLMAP database copied; adding the test images")
    else:
        log("features      : the scene's COLMAP database is gone (cleaned up?): extracting the scene's images again")
        _extract(cfg, scene, "images", np.load(scene / "record3d" / "poses_refined.npz")["K"], log)
    _extract(cfg, scene, "test/images", np.load(test / "record3d" / "poses.npz")["K"], log)
    r = tool(cfg, "localize_test.py", [scene, "pairs-align"], log)
    log("placing       : " + r.stdout.strip().splitlines()[-1])
    _match(cfg, scene, "pairs_align.txt", log)
    r = tool(cfg, "localize_test.py", [scene, "align"], log)
    log("placed        : " + r.stdout.strip().splitlines()[-1])
    _match(cfg, scene, "pairs_loc.txt", log)
    r = tool(cfg, "localize_test.py", [scene, "localize"], log)
    log("localised     : " + r.stdout.strip().splitlines()[-1].replace("localised ", ""))
    loc = json.loads((test / "localize.json").read_text())
    if not tc.get("keep_database", False):           # a copy of the scene's (~2 GB): only needed to localise
        for f in (test / "colmap").glob("database.db*"):
            f.unlink()

    tg = Path(deep_get(cfg, "paths.threedgrut")).expanduser()
    py = venv_python(tg)
    wait_for_gpu(cfg, log, "rendering the test views")
    out = test / "render"
    r = run([str(py), str(PKG / "tools" / "render_test.py"), "--checkpoint", str(ckpt), "--path", str(test / "dataset"),
             "--out", str(out)], log, cwd=str(tg), env=venv_env(cfg, py), check=False)
    found = sorted(out.glob("**/metrics.json"))
    if r.returncode != 0 or not found:
        for t in r.stdout.splitlines()[-10:]:
            log("   |", t)
        raise SystemExit(f"rendering the test views failed (exit {r.returncode})")
    m = json.loads(found[-1].read_text())
    per = {p["name"]: p["psnr"] for p in json.loads((out / "per_image.json").read_text())}
    h = {f["name"]: f["height_m"] for f in loc["frames"] if f.get("ok")}
    pnp = [per[f["name"]] for f in loc["frames"] if f.get("how") == "pnp" and f["name"] in per]
    bands = []
    for lo, hi, lab in BANDS:
        x = [per[n] for n in per if n in h and lo <= h[n] < hi]
        if len(x) >= 5:
            bands.append({"band": lab, "images": len(x), "psnr_median": round(float(np.median(x)), 2)})
    sheet = next(iter(out.glob("sheet.jpg")), None)
    if sheet:
        shutil.copy(sheet, test / "sheet.jpg")
    n_scored = loc["localized"] + loc.get("interpolated", 0)
    log(f"test clip     : PSNR {m['mean_psnr']:.2f} dB (colour-corrected {m['mean_cc_psnr']:.2f}), SSIM "
        f"{m['mean_ssim']:.3f}, LPIPS {m['mean_lpips']:.3f} on {n_scored} of {loc['test_images']} test images "
        f"({loc['localized']} localised, {loc.get('interpolated', 0)} placed from their neighbours; localised only: "
        f"{np.mean(pnp):.2f} dB); training's held-out frames: {best.get('mean_psnr', float('nan')):.2f} dB")
    if loc["localized"] < 0.8 * loc["test_images"]:
        log(f"  only {loc['localized'] / loc['test_images']:.0%} of the clip matched the scene's images well enough to be "
            "localised: the views the scene covers badly")
    for b in bands:
        log(f"  camera {b['band']:<20}: {b['images']:4d} images, PSNR median {b['psnr_median']:.2f} dB")
    return {"ok": True, "export_dir": v["export_dir"], "duration_s": v["duration_s"], "images": pj["images"],
            "localized": loc["localized"], "interpolated": loc.get("interpolated", 0),
            "psnr_localized_only": round(float(np.mean(pnp)), 3) if pnp else None, "camera_height_m": pj["camera_height_m"],
            "loc_inliers_median": loc["inliers_median"], "loc_reproj_px": loc["reproj_px_median"],
            "psnr": m["mean_psnr"], "cc_psnr": m["mean_cc_psnr"], "ssim": m["mean_ssim"], "lpips": m["mean_lpips"],
            "bands": bands, "sheet": str(test / "sheet.jpg") if sheet else None}
