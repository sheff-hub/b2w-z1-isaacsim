"""Stage 7 - report: numbers, and images to look at.

The images matter as much as the metrics. Twice in this project the metrics
told the wrong story: LPIPS looked acceptable while renders were ruined by fog,
and SSIM scored HIGHER on a badly fogged model than on the good one. PSNR was
the only metric that caught it, and looking at the renders settled it.
"""
from __future__ import annotations

import json
from pathlib import Path

from .common import Log, deep_get


def comparison_sheet(scene: Path, run_dir: Path, factor: int, out_png: Path,
                     n: int, log: Log) -> bool:
    """Held-out renders beside their ground truth."""
    try:
        import numpy as np
        from PIL import Image
    except Exception as e:
        log(f"comparison sheet skipped: {e}")
        return False
    renders = sorted((run_dir / f"ours_{_last_iter(run_dir)}" / "renders").glob("*.png"))
    if not renders:
        log("no renders found; skipping comparison sheet")
        return False
    img_dir = scene / (f"images_{factor}" if factor > 1 else "images")
    gts = sorted(p for p in img_dir.iterdir() if p.is_file())
    # renders are indexed over the val split; sample a few spread across it
    idx = [0, len(renders) // 2, len(renders) - 1][:max(1, n)]
    rows = []
    for i in idx:
        try:
            rd = np.asarray(Image.open(renders[i]).convert("RGB"))
            # match by size; the val split order is not recoverable from disk alone
            gt = None
            for g in gts:
                a = np.asarray(Image.open(g).convert("RGB"))
                if a.shape == rd.shape:
                    gt = a
                    break
            if gt is None:
                continue
            rows.append(np.concatenate([gt, rd], axis=1))
        except Exception:
            continue
    if not rows:
        return False
    sheet = np.concatenate(rows, axis=0).astype(np.uint8)
    im = Image.fromarray(sheet)
    im.thumbnail((1600, 1600))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_png)
    log(f"comparison sheet: {out_png}  (left = ground truth, right = render)")
    return True


def _last_iter(run_dir: Path) -> int:
    its = [int(p.name.split("_")[1]) for p in run_dir.glob("ours_*") if p.name.split("_")[-1].isdigit()]
    return max(its) if its else 30000


def write_report(cfg: dict, scene: Path, state_data: dict, log: Log) -> Path:
    md = scene / "REPORT.md"
    g = state_data["stages"]
    val = g.get("validate", {})
    sfm = g.get("sfm", {})
    tr = g.get("train", {})
    ex = g.get("export", {})
    best = tr.get("best", {})
    lines = [
        f"# Reconstruction report — {scene.name}", "",
        "## Asset", "",
        "```", str(ex.get("usdz", "(not exported)")), "```", "",
        f"- gaussians: **{ex.get('gaussians', '?')}**",
        f"- size: {ex.get('size_mb', '?')} MB",
        f"- verified upright: {ex.get('reoriented', '?')} | extent finite: {ex.get('finite', '?')}",
        "",
        "Open in Isaac Sim: **File -> Import**, then **Create -> Physics -> Ground Plane**.",
        "The asset deliberately contains no floor and no physics; a robot falls through without one.",
        "",
        "## Capture", "",
        f"- images: **{val.get('count', '?')}**",
        f"- lens: {val.get('lens', '?')}",
        f"- dimensions: {val.get('dims', '?')}",
        "",
        "## Structure from motion", "",
        f"- registered: **{sfm.get('images', '?')} / {val.get('count', '?')}**",
        f"- 3D points: **{sfm.get('points', 0):,}**" if sfm.get("points") else "- 3D points: ?",
        f"- mean reprojection error: {sfm.get('reproj', '?')} px",
        "",
        "## Training", "",
    ]
    res = tr.get("results", [])
    if res:
        lines += ["| cap | ratio to points | PSNR | SSIM | LPIPS |", "|---|---|---|---|---|"]
        for r in sorted(res, key=lambda r: r["cap"]):
            mark = " **<- best**" if r["cap"] == best.get("cap") else ""
            m = {k: ("n/a" if r.get(k) is None else format(r[k], f)) for k, f in
                 (("mean_psnr", ".3f"), ("mean_ssim", ".4f"), ("mean_lpips", ".4f"))}
            lines.append(f"| {r['cap']:,} | {r['ratio']:.1f}x | {m['mean_psnr']} | "
                         f"{m['mean_ssim']} | {m['mean_lpips']}{mark} |")
        lines.append("")
    lines += [
        "## How to read these numbers", "",
        "- **PSNR is the metric to trust.** It is stable across evaluation resolution.",
        "- **LPIPS is inflated here.** It is scored at full image resolution, but the",
        "  metric is calibrated for ~256 px inputs. On identical images: 0.479 at",
        "  2000 px, 0.242 at 500 px, 0.130 at 256 px. Do not compare it to published",
        "  numbers; only compare runs against each other.",
        "- **SSIM can mislead.** A badly fogged model once scored *higher* SSIM than",
        "  the good one, because smooth fog against a smooth wall scores well.",
        "- **Look at the renders** in `comparison.png` before believing any of it.",
        "",
        "## Typical quality", "",
        "| scene | photos | PSNR |", "|---|---|---|",
        "| NVIDIA reference | 499 | 25.88 |",
        "| bedroom, 0.5x lens | 428 | 25.73 |",
        "| lounge, 1x lens | 932 | 22.62 |",
        "",
        "Below ~23 dB, the usual cause is large blank walls and empty floor, not the",
        "pipeline. The fix is capture: shoot *along* blank walls at grazing angles so",
        "skirting, sockets and door frames enter the frame, and include deliberate",
        "upward coverage for ceilings.",
    ]
    md.write_text("\n".join(lines) + "\n")
    log(f"report written: {md}")
    return md
