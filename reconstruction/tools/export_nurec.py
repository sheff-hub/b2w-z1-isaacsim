#!/usr/bin/env python
"""Export a 3DGRUT checkpoint to NuRec USDZ, correcting two defects in the
stock export path.

Run inside the 3DGRUT venv:

    cd <paths.threedgrut> && source .venv/bin/activate
    python tools/export_nurec_fixed.py \
        --checkpoint runs/.../ckpt_last.pt \
        --output     runs/.../export_fixed.usdz \
        --dataset    <paths.nurec>/mine/<scene>

Why this exists rather than `python -m threedgrut.export.scripts.export_usd`:

1. UP-AXIS / 90-DEGREE TILT.
   The NuRec serializer hardcodes the stage's upAxis to "Z"
   (export/usd/nurec/serializer.py:41) but only *reorients the data* when
   apply_coordinate_transform=True, which defaults to False
   (export/usd/nurec/exporter.py:321). COLMAP's world frame is Y-down /
   Z-forward, so the data stays Y-up while the stage claims Z-up and the scene
   loads rotated 90 degrees in Isaac Sim.

   The correction is get_3dgrut_to_usdz_coordinate_transform() (x->-X, y->-Z,
   z->-Y), which is exactly the rotation baked into NVIDIA's own reference
   asset for this scene. The stock CLI cannot enable it: --up-axis is
   documented in export_usd.py:547 as "stage upAxis metadata only (no
   reorientation)" and is ignored for --format nurec. Hence a direct call to
   NuRecExporter.export(..., apply_coordinate_transform=True).

2. FLOAT16 POSITION OVERFLOW.
   NuRec serialises positions as float16 (export/usd/nurec/templates.py:34),
   whose maximum finite value is 65504. 3DGS training strands a small number
   of near-transparent "floater" gaussians at enormous radii; those positions
   become +/-inf in the exported file and blow the authored extent up with
   them. On the zh_fourth_floor_iphone run: scene_extent 3.53, median gaussian
   radius 3.7, yet 2824 gaussians (1.02%) sat beyond 65504 -- out to 1.47e9 --
   producing an authored extent of +/-1.18e9 against NVIDIA's +/-5.7e4.

   Dropping them is the right fix rather than widening the dtype: they are
   junk, not signal (mean pre-activation density -0.30 against 2.08 for the
   scene as a whole), and float16 is what the NuRec format specifies.

Pruning happens on the checkpoint dict before the model is built, because
MixtureOfGaussians exposes no prune method. Every per-gaussian tensor must be
masked in step or the model loads with mismatched row counts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# torch, threedgrut and pxr are imported inside the functions that need them: this module is also imported by the
# pipeline driver (system python), which only uses run_stage() and runs the export itself in 3DGRUT's venv.

# Tensors in the checkpoint with one row per gaussian. Anything added to the
# checkpoint format in a future 3DGRUT version has to be added here too, so the
# script verifies the list against the actual row count instead of trusting it.
PER_GAUSSIAN_KEYS = (
    "positions",
    "rotation",
    "scale",
    "density",
    "features_albedo",
    "features_specular",
)

FLOAT16_MAX = 65504.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", "--checkpoint", required=True, help="3DGRUT checkpoint (.pt)")
    p.add_argument("-o", "--output", required=True, help="Output .usdz / .usda / .usd")
    p.add_argument("--dataset", default=None, help="Dataset path, for camera export")
    p.add_argument(
        "--max-radius-factor",
        type=float,
        default=50.0,
        help="Drop gaussians farther from the origin than this multiple of the "
        "checkpoint's scene_extent (default: 50). The floaters sit orders of "
        "magnitude beyond the scene, so the exact value is not delicate.",
    )
    p.add_argument(
        "--max-radius",
        type=float,
        default=None,
        help="Absolute radius cutoff in world units. Overrides --max-radius-factor.",
    )
    p.add_argument("--no-prune", action="store_true", help="Skip pruning (reproduces the overflow)")
    p.add_argument("--no-reorient", action="store_true", help="Skip the up-axis fix (reproduces the tilt)")
    p.add_argument("--no-cameras", action="store_true", help="Skip camera export")
    p.add_argument("--verify-json", default=None,
                   help="After exporting, check the asset (upright, finite bounds) and write the result to this JSON")
    return p.parse_args()


def prune_floaters(checkpoint: dict, max_radius: float) -> dict:
    """Drop gaussians beyond max_radius. Returns stats for reporting."""
    import torch
    positions = checkpoint["positions"].detach().float()
    n_before = positions.shape[0]

    radius = positions.norm(dim=1)
    keep = radius <= max_radius
    n_keep = int(keep.sum())

    if n_keep == n_before:
        return {"n_before": n_before, "n_after": n_before, "n_dropped": 0, "max_radius": max_radius}

    if n_keep == 0:
        raise SystemExit(f"ERROR: radius cutoff {max_radius:.4g} would drop every gaussian. Raise --max-radius.")

    # Density of what we are throwing away, as evidence it really is junk. This
    # is pre-activation; the model applies a sigmoid on top.
    density = checkpoint["density"].detach().float().ravel()
    dropped_density = float(density[~keep].mean())
    kept_density = float(density[keep].mean())
    max_dropped_radius = float(radius[~keep].max())
    n_overflow = int((radius > FLOAT16_MAX).sum())

    for key in PER_GAUSSIAN_KEYS:
        if key not in checkpoint:
            continue
        tensor = checkpoint[key]
        if not torch.is_tensor(tensor) or tensor.dim() == 0 or tensor.shape[0] != n_before:
            raise SystemExit(
                f"ERROR: checkpoint['{key}'] has shape {tuple(tensor.shape)}, expected {n_before} rows. "
                "The checkpoint format has changed; update PER_GAUSSIAN_KEYS."
            )
        # These are stored as nn.Parameter, and indexing one yields a plain
        # Tensor. MixtureOfGaussians assigns them straight onto nn.Module
        # attributes, which rejects anything that is not a Parameter, so the
        # wrapper has to be restored.
        masked = tensor[keep]
        if isinstance(tensor, torch.nn.Parameter):
            masked = torch.nn.Parameter(masked.detach(), requires_grad=tensor.requires_grad)
        checkpoint[key] = masked

    # The optimizer state is per-gaussian too, but export never touches it and
    # a stale one would silently mismatch if this checkpoint were ever resumed.
    # Drop it rather than leave a trap behind.
    checkpoint.pop("optimizer", None)
    checkpoint.pop("densify_grad_norm_accum", None)
    checkpoint.pop("densify_grad_norm_denom", None)

    return {
        "n_before": n_before,
        "n_after": n_keep,
        "n_dropped": n_before - n_keep,
        "n_overflow": n_overflow,
        "max_radius": max_radius,
        "max_dropped_radius": max_dropped_radius,
        "dropped_density": dropped_density,
        "kept_density": kept_density,
    }


def main():
    args = parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(f"ERROR: checkpoint not found: {ckpt_path}")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import torch

    print(f"Loading {ckpt_path}")
    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    scene_extent = float(checkpoint.get("scene_extent", 1.0))
    if args.max_radius is not None:
        max_radius = args.max_radius
    else:
        max_radius = args.max_radius_factor * scene_extent

    print(f"  scene_extent : {scene_extent:.4g}")
    print(f"  radius cutoff: {max_radius:.4g}")

    if args.no_prune:
        print("  pruning      : SKIPPED (--no-prune)")
        stats = None
    else:
        stats = prune_floaters(checkpoint, max_radius)
        if stats["n_dropped"] == 0:
            print("  pruning      : nothing beyond the cutoff")
        else:
            print(
                f"  pruned       : {stats['n_dropped']:,} of {stats['n_before']:,} "
                f"({100 * stats['n_dropped'] / stats['n_before']:.2f}%) -> {stats['n_after']:,} kept"
            )
            print(f"    farthest dropped : {stats['max_dropped_radius']:,.4g}")
            print(f"    float16-overflow : {stats['n_overflow']:,} (positions that became inf)")
            print(f"    density dropped  : {stats['dropped_density']:.3f} vs kept {stats['kept_density']:.3f}")

    # Build the model from the (possibly pruned) checkpoint. Mirrors
    # export_usd.load_model_from_checkpoint, minus the optimizer.
    from threedgrut.model.model import MixtureOfGaussians

    conf = checkpoint["config"]
    model = MixtureOfGaussians(conf, scene_extent=checkpoint.get("scene_extent"))
    model.init_from_checkpoint(checkpoint, setup_optimizer=False)
    model.eval()
    print(f"  model loaded : {model.get_positions().shape[0]:,} gaussians")

    dataset = None
    validation_dataset = None
    if not args.no_cameras:
        import threedgrut.datasets as datasets

        if args.dataset:
            conf.path = args.dataset
        dataset, validation_dataset = datasets.make(name=conf.dataset.type, config=conf, ray_jitter=None)
        print(f"  dataset      : {len(dataset)} train / {len(validation_dataset)} val frames")

    from threedgrut.export.usd.nurec.exporter import NuRecExporter

    exporter = NuRecExporter(export_cameras=not args.no_cameras)

    reorient = not args.no_reorient
    print(f"  reorient     : {'YES (3DGRUT -> USDZ, x->-X y->-Z z->-Y)' if reorient else 'NO (--no-reorient)'}")
    print(f"Exporting to {output_path}")

    exporter.export(
        model=model,
        output_path=output_path,
        dataset=dataset,
        conf=conf,
        background=model.background,
        validation_dataset=validation_dataset,
        apply_coordinate_transform=reorient,
    )
    print(f"Done: {output_path}  ({output_path.stat().st_size / 1e6:.1f} MB)")
    if args.verify_json:
        result = verify(output_path)
        Path(args.verify_json).write_text(json.dumps(result, indent=2))
    return 0


def verify(usdz) -> dict:
    """Confirm the asset is upright and finite BEFORE anyone opens it. Needs pxr (3DGRUT's venv has it)."""
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(usdz))
    prim = stage.GetPrimAtPath("/World/gauss/gauss")
    if not prim or not prim.IsValid():
        raise SystemExit(f"no NuRec volume prim in {usdz}")
    m = np.array(prim.GetAttribute("xformOp:transform").Get()).reshape(4, 4)
    ext = prim.GetAttribute("extent").Get()
    mn, mx = np.array(ext[0]), np.array(ext[1])
    rotated = not np.allclose(m[:3, :3], np.eye(3), atol=1e-3)
    finite = bool(np.isfinite(mn).all() and np.isfinite(mx).all())
    print(f"upAxis        : {UsdGeom.GetStageUpAxis(stage)}")
    print(f"reoriented    : {rotated}   (identity here means the 90-degree tilt bug)")
    print(f"extent finite : {finite}   {np.round(mn, 1)} .. {np.round(mx, 1)}")
    if not finite:
        raise SystemExit("extent contains inf/nan -- float16 overflow was not pruned")
    return {"reoriented": bool(rotated), "finite": finite, "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
            "extent_min": mn.tolist(), "extent_max": mx.tolist()}


if __name__ == "__main__":
    sys.exit(main())


# ---------------------------------------------------------------------------
# Pipeline stage wrapper (used by reconstruct.py; the CLI above still works)
# ---------------------------------------------------------------------------

def run_stage(cfg, scene, run_dir, log):
    """Export the trained checkpoint to a NuRec .usdz and check it, as a subprocess in 3DGRUT's venv.

    It has to run there, like training does: the checkpoint is a pickle written by the venv's numpy 2 / torch, which
    an older system numpy cannot read ("No module named 'numpy._core'"), the export compiles a kernel with the venv's
    `slangc`, and the check needs pxr. As a subprocess its output, errors included, goes into pipeline.log.
    Returns (usdz path, check result)."""
    from .common import deep_get, run, venv_env, venv_python
    log.rule("STAGE 5 - export NuRec USDZ")
    tg = Path(deep_get(cfg, "paths.threedgrut")).expanduser()
    py = venv_python(tg)
    run_dir = Path(run_dir)
    out = run_dir / "export_fixed.usdz"
    check = run_dir / "export_check.json"
    cmd = [str(py), str(Path(__file__).resolve()),
           "--checkpoint", str(run_dir / "ckpt_last.pt"),
           "--output", str(out),
           "--dataset", str(scene),
           "--max-radius-factor", str(deep_get(cfg, "export.prune_radius_factor", 50.0)),
           "--verify-json", str(check)]
    if not deep_get(cfg, "export.reorient", True):
        cmd.append("--no-reorient")
    run(cmd, log, cwd=str(tg), env=venv_env(cfg, py))
    result = json.loads(check.read_text())
    log(f"exported      : {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    log(f"upright       : {result['reoriented']}   (False = the 90-degree tilt bug)")
    log(f"finite bounds : {result['finite']}   {np.round(result['extent_min'], 1)} .. "
        f"{np.round(result['extent_max'], 1)}")
    if not result["reoriented"]:
        log("!! rotation is identity -- the asset will load tilted 90 degrees")
    return out, result
