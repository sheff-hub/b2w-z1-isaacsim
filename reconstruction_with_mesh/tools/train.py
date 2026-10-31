"""Stage 7 - 3DGRUT training (3DGUT + MCMC) on the bundle-adjusted phone poses.

Record3D scenes are LARGE (a corridor loop, 40 x 30 m) and their images small (720 x 960). Two settings matter:
  - regularisation: with 3DGRUT's default lambda_opacity = lambda_scale = 0.01, MCMC collapsed on the 22 Oct walk
    (PSNR 17; relocated gaussians 8% -> 80%). 1e-4 / 0.005, the values of 3DGRUT's own cusfm_3dgut_mcmc.yaml for
    large scenes, gave 27.1 dB. They are train.overrides in config.yaml;
  - the budget: ~4x the bundle-adjustment points (2M for 476k points there; 1M was ~0.2 dB lower on the first
    Record3D scene), at most train.cap_max (2M: three heights of a place see no more of it than one; 4M fogged the
    corridors of the three-height office of 27 Oct). Training images are used at full size (downsample 1);
  - the length: 30,000 steps was ~11 views of each of the 2,669 training images of the 22 Oct walk. With more
    images (several passes) train.iterations "auto" keeps ~11 views per image (30k..90k steps), and every schedule
    of 3DGRUT that is written for 30k steps (MCMC growth/relocation/noise, learning-rate decay) is stretched with it.

Notes from the photo pipeline this module comes from (reconstruction_no_mesh):

The single most consequential setting is `max_n_gaussians`. NVIDIA's config
ships 1,000,000, which is right for a large scene and badly wrong for a small
one. What matters is the ratio of gaussians to COLMAP points:

    measured, 428-photo bedroom:  8.9x -> 24.67 dB (fogged)
                                  2.9x -> 25.73 dB   <- peak
    measured, 932-photo lounge:   3.0x -> 22.47 dB
                                  1.5x -> 22.62 dB   <- peak (flat curve)

Surplus gaussians are not harmless. In under-constrained space they settle at
near-zero opacity and fog the volume in front of good geometry, which is why
cutting the budget improves the well-covered regions too.
"""
from __future__ import annotations

import json
import re
import shutil
import struct
import time
from pathlib import Path

import numpy as np

from .common import Log, deep_get, gpu_free_gb, gpu_users, run, venv_env, venv_python


def resolve_caps(cfg: dict, points: int, log: Log) -> list[int]:
    caps = deep_get(cfg, "train.caps")
    if caps:
        caps = [int(c) for c in caps]
        log(f"gaussian caps : {caps} (explicit list -> will compare and pick the best)")
        return caps
    ratio = float(deep_get(cfg, "train.cap_ratio", 2.0))
    cap = int(points * ratio)
    cap = max(int(deep_get(cfg, "train.cap_min", 150000)),
              min(cap, int(deep_get(cfg, "train.cap_max", 2000000))))
    log(f"gaussian cap  : {cap:,} ({ratio}x the {points:,} COLMAP points)")
    return [cap]


def resolve_iterations(cfg: dict, images: int, log: Log) -> int:
    """train.iterations: a number, or "auto" = ~iterations_per_image views of every training image (7 of every 8
    images train; 3DGRUT holds every 8th out), rounded to 1000 and clipped to [iterations_min, iterations_max]."""
    it = deep_get(cfg, "train.iterations", 30000)
    if str(it).lower() != "auto":
        return int(it)
    per = float(deep_get(cfg, "train.iterations_per_image", 11))
    n_train = int(images * 7 / 8)
    it = int(np.clip(round(per * n_train / 1000) * 1000, int(deep_get(cfg, "train.iterations_min", 30000)),
                     int(deep_get(cfg, "train.iterations_max", 90000))))
    log(f"iterations    : {it:,} (auto: ~{per:g} views of each of the {n_train:,} training images)")
    return it


def schedule_overrides(iters: int) -> list[str]:
    """3DGRUT's schedules for its default 30k steps, stretched to `iters` (identical values at 30k)."""
    f = iters / 30000.0
    return [f"strategy.relocate.end_iteration={int(25000 * f)}", f"strategy.add.end_iteration={int(25000 * f)}",
            f"strategy.perturb.end_iteration={int(27500 * f)}", f"scheduler.positions.max_steps={iters}",
            f"scheduler.features.max_steps={iters}", f"checkpoint.iterations=[{iters}]"]


def wait_for_gpu(cfg: dict, log: Log, what: str) -> None:
    """The card is shared. Before a GPU-heavy step, wait (up to gpu.wait_min) until gpu.min_free_gb is free, and say
    who holds the rest -- the kitchen run of 14 Oct 2026 lost its end-of-training evaluation to another user's job that
    started halfway through, with only 0.8 GB left."""
    uuid = str(deep_get(cfg, "gpu.uuid", "all"))
    need = float(deep_get(cfg, "gpu.min_free_gb", 10))
    wait_s = float(deep_get(cfg, "gpu.wait_min", 30)) * 60
    t0 = time.time()
    free = gpu_free_gb(uuid=uuid)
    if free >= need or free == 0.0:  # 0.0: nvidia-smi not readable, nothing to judge by
        log(f"GPU free      : {free:.1f} GB (need ~{need:.0f} GB for {what})")
        return
    log(f"!! GPU has only {free:.1f} GB free, {what} needs ~{need:.0f} GB. On the card now:")
    for row in gpu_users(uuid):
        log("     " + row)
    log(f"   waiting up to {wait_s/60:.0f} min for memory to free up (gpu.wait_min in config.yaml) ...")
    while time.time() - t0 < wait_s:
        time.sleep(60)
        free = gpu_free_gb(uuid=uuid)
        if free >= need:
            log(f"GPU free      : {free:.1f} GB after {(time.time()-t0)/60:.0f} min, continuing")
            return
    log(f"!! still only {free:.1f} GB free; trying anyway (it may run out of memory)")


def finished_run(runs: Path, tag: str, iters: int, factor: int) -> Path | None:
    """A run of this cap that reached the last iteration at this factor (its checkpoint is complete even if what came
    after it -- 3DGRUT's own evaluation -- crashed)."""
    for d in sorted((runs / tag).glob("*/"), key=lambda p: p.stat().st_mtime, reverse=True):
        if not ((d / "ckpt_last.pt").exists() and (d / f"ours_{iters}" / f"ckpt_{iters}.pt").exists()):
            continue
        parsed = d / "parsed.yaml"
        f = next((ln.split(":", 1)[1].strip() for ln in (parsed.read_text().splitlines() if parsed.exists() else [])
                  if ln.strip().startswith("downsample_factor:")), None)
        if f is None or str(f) == str(factor):
            return d
    return None


def evaluate(cfg: dict, run_dir: Path, log: Log) -> bool:
    """Score a saved checkpoint on the test images with 3DGRUT's render.py (what train.py does at its end), and put
    metrics.json where train.py would have. Returns False if that fails too."""
    tg = Path(deep_get(cfg, "paths.threedgrut")).expanduser()
    py = venv_python(tg)
    wait_for_gpu(cfg, log, "evaluation")
    out = run_dir / "eval"
    cmd = [str(py), "render.py", "--checkpoint", str(run_dir / "ckpt_last.pt"), "--out-dir", str(out),
           "--save-gt"]  # (sic) this flag turns saving the ground-truth images OFF
    r = run(cmd, log, cwd=str(tg), env=venv_env(cfg, py), check=False)
    found = sorted(out.glob("**/metrics.json"), key=lambda p: p.stat().st_mtime)
    if r.returncode != 0 or not found:
        log(f"!! evaluation failed (exit {r.returncode}); the model is fine, only its scores are missing")
        for t in r.stdout.splitlines()[-8:]:
            log("   |", t)
        return False
    shutil.copy(found[-1], run_dir / "metrics.json")
    return True


def init_root(cfg: dict, scene: Path, cap: int, log: Log) -> Path:
    """The folder 3DGRUT trains from. It starts with one gaussian per bundle-adjustment point, and max_n_gaussians
    only stops it ADDING: a point cloud larger than the cap is kept whole (the dense three-height office of 28 Oct,
    2,783,017 points, trained as 2.78M gaussians under a 1.89M cap). Above train.init_share x cap points (0.5: MCMC
    then places the other half where the photos need them), this returns a folder that shares the scene's images and
    cameras but holds a random subsample of its points."""
    src = scene / "sparse" / "0" / "points3D.bin"
    buf = src.read_bytes()
    n = struct.unpack_from("<Q", buf, 0)[0]
    keep = int(cap * float(deep_get(cfg, "train.init_share", 0.5)))
    if n <= keep:
        return scene
    root = scene / f"train_init_{keep // 1000}k"
    if (root / "sparse" / "0" / "points3D.bin").exists():
        log(f"initial points: {keep:,} of the {n:,} bundle-adjustment points ({root.name}, made earlier)")
        return root
    pick = np.zeros(n, bool)
    pick[np.random.default_rng(0).choice(n, keep, replace=False)] = True
    out, o = [struct.pack("<Q", keep)], 8
    for i in range(n):
        tlen = struct.unpack_from("<Q", buf, o + 43)[0]
        if pick[i]:
            out.append(buf[o:o + 43] + struct.pack("<Q", 0))           # the track is not read by 3DGRUT
        o += 51 + 8 * tlen
    (root / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    (root / "sparse" / "0" / "points3D.bin").write_bytes(b"".join(out))
    for f in (scene / "sparse" / "0").iterdir():
        if not f.name.startswith("points3D"):
            (root / "sparse" / "0" / f.name).symlink_to(Path("../../..") / "sparse" / "0" / f.name)
    for d in scene.glob("images*"):
        (root / d.name).symlink_to(Path("..") / d.name)
    log(f"initial points: {keep:,} of the {n:,} bundle-adjustment points ({root.name}): 3DGRUT keeps every "
        f"starting point, so a cloud larger than the {cap:,} cap would overshoot it")
    return root


def train_one(cfg: dict, scene: Path, cap: int, factor: int, tag: str, iters: int, log: Log) -> Path:
    tg = Path(deep_get(cfg, "paths.threedgrut")).expanduser()
    py = venv_python(tg)
    runs = scene / "runs"
    runs.mkdir(exist_ok=True)

    d = finished_run(runs, tag, iters, factor)
    if d is not None and not (d / "metrics.json").exists():
        # an earlier run trained to the end and then died in its evaluation: keep the model, don't train 25 min again
        log(f"reusing       : {d.name} (trained to {iters} steps, only its evaluation is missing; "
            f"delete that folder to train again)")
    else:
        log(f"training cap={cap:,} factor={factor} -> {tag}")
        wait_for_gpu(cfg, log, "training")
        t0 = time.time()
        cmd = [str(py), "train.py",
               "--config-name", deep_get(cfg, "train.config_name", "apps/colmap_3dgut_mcmc.yaml"),
               f"path={init_root(cfg, scene, cap, log)}", f"out_dir={runs}", f"experiment_name={tag}",
               f"n_iterations={iters}",
               f"strategy.add.max_n_gaussians={cap}",
               "export_usd.enabled=false",
               f"dataset.downsample_factor={factor}"] + schedule_overrides(iters)
        cmd += [str(o) for o in (deep_get(cfg, "train.overrides") or [])]   # the config's own overrides win
        r = run(cmd, log, cwd=str(tg), env=venv_env(cfg, py), check=False)  # venv on PATH (slangc), GPU pinned
        log(f"  finished in {(time.time()-t0)/60:.1f} min")
        d = finished_run(runs, tag, iters, factor)
        if r.returncode != 0:
            log(f"!! train.py exited with code {r.returncode}:")
            for t in r.stdout.splitlines()[-25:]:
                log("   |", t)
            if d is None:
                raise SystemExit(f"training failed before step {iters} (exit {r.returncode}); see the lines above")
            log(f"   but the model reached step {iters} and is saved: only 3DGRUT's evaluation after training failed")
        if d is None:
            raise SystemExit(f"training produced no step-{iters} checkpoint under {runs/tag}")

    if not (d / "metrics.json").exists():
        evaluate(cfg, d, log)
    if not deep_get(cfg, "train.keep_intermediate", False):     # the 7k checkpoint: ~1.3 GB at 2M gaussians, never used
        for c in d.glob("ours_*/ckpt_*.pt"):
            if c.parent.name != f"ours_{iters}":
                c.unlink()
                log(f"removed intermediate checkpoint {c.relative_to(d)}")
    return d


def read_metrics(run_dir: Path) -> dict:
    """PSNR/SSIM/LPIPS of a run; empty if it could not be scored (the model is still usable)."""
    p = run_dir / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _f(v, fmt: str) -> str:
    return "n/a" if v is None else format(v, fmt)


def run_stage(cfg: dict, scene: Path, points: int, factor: int, log: Log, images: int = 0) -> dict:
    log.rule("STAGE 7 - train the splat (3DGRUT)")
    caps = resolve_caps(cfg, points, log)
    iters = resolve_iterations(cfg, images or len(list((scene / "images").glob("*.jpg"))), log)
    results = []
    for cap in caps:
        tag = f"gs_{cap//1000}k"
        d = train_one(cfg, scene, cap, factor, tag, iters, log)
        m = read_metrics(d)
        results.append({"cap": cap, "dir": str(d), "ratio": cap / max(points, 1), **m})
        log(f"  cap {cap:,} ({cap/max(points,1):.1f}x): PSNR {_f(m.get('mean_psnr'), '.3f')}  "
            f"SSIM {_f(m.get('mean_ssim'), '.4f')}  LPIPS {_f(m.get('mean_lpips'), '.4f')}")

    # unscored runs rank last; with one cap there is nothing to choose anyway
    best = max(results, key=lambda r: r.get("mean_psnr", float("-inf")))
    if len(results) > 1:
        log("")
        log(f"{'cap':>10} {'ratio':>7} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}")
        for r in sorted(results, key=lambda r: r["cap"]):
            mark = "  <-- best" if r["cap"] == best["cap"] else ""
            log(f"{r['cap']:>10,} {r['ratio']:>6.1f}x {_f(r.get('mean_psnr'), '8.3f')} "
                f"{_f(r.get('mean_ssim'), '8.4f')} {_f(r.get('mean_lpips'), '8.4f')}{mark}")
        scored = [r["mean_psnr"] for r in results if "mean_psnr" in r]
        spread = best.get("mean_psnr", 0.0) - min(scored) if scored else 0.0
        log(f"spread across caps: {spread:.3f} dB")
        if len(scored) > 1 and spread < 0.2:
            log("  (flat curve — the cap is not the limiting factor on this scene; "
                "coverage and texture are)")
    log(f"best cap      : {best['cap']:,} at PSNR {_f(best.get('mean_psnr'), '.3f')}")
    # the factor is part of the result: export and report must use the training images, whatever the GPU has free later
    return {"results": results, "best": best, "factor": factor, "iterations": iters}
