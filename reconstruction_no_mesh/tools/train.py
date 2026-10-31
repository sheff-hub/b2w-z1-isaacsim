"""Stage 4 - 3DGRUT training, with the gaussian budget chosen from evidence.

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
import time
from pathlib import Path

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


def train_one(cfg: dict, scene: Path, cap: int, factor: int, tag: str, log: Log) -> Path:
    tg = Path(deep_get(cfg, "paths.threedgrut")).expanduser()
    py = venv_python(tg)
    runs = scene / "runs"
    runs.mkdir(exist_ok=True)
    iters = int(deep_get(cfg, "train.iterations", 30000))

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
               f"path={scene}", f"out_dir={runs}", f"experiment_name={tag}",
               f"n_iterations={iters}",
               f"strategy.add.max_n_gaussians={cap}",
               "export_usd.enabled=false"]
        if factor > 1:
            cmd.append(f"dataset.downsample_factor={factor}")
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
    return d


def read_metrics(run_dir: Path) -> dict:
    """PSNR/SSIM/LPIPS of a run; empty if it could not be scored (the model is still usable)."""
    p = run_dir / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _f(v, fmt: str) -> str:
    return "n/a" if v is None else format(v, fmt)


def run_stage(cfg: dict, scene: Path, points: int, factor: int, log: Log) -> dict:
    log.rule("STAGE 4 - train")
    caps = resolve_caps(cfg, points, log)
    results = []
    for cap in caps:
        tag = f"gs_{cap//1000}k"
        d = train_one(cfg, scene, cap, factor, tag, log)
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
    return {"results": results, "best": best, "factor": factor}
