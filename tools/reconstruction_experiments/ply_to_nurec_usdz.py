#!/usr/bin/env python
"""Convert a standard 3DGS PLY (e.g. nerfstudio `ns-export gaussian-splat`) to NuRec USDZ.

Run inside the 3DGRUT venv:

    cd ~/Firdavs/repos/3dgrut && source .venv/bin/activate
    python ~/Firdavs/tools/ply_to_nurec_usdz.py splat.ply splat.usdz

Same steps as `python -m threedgrut.export.scripts.ply_to_usd`, which fails in 3DGRUT a37ef72:
it builds `NuRecExporter()` with export_cameras=True (the default) but passes no dataset, and
the exporter refuses that ("export_cameras=True requires a dataset"). A bare PLY has no
cameras, so they are switched off here.

Two further points, from ~/Firdavs/tools/export_nurec_fixed.py:
- No reorientation. A nerfstudio PLY is already Z-up (its dataparser orients the mean camera
  up to +Z), which is what the NuRec stage declares. A 3DGRUT/COLMAP checkpoint is not; use
  export_nurec_fixed.py for those.
- NuRec stores positions as float16 (max 65504). Gaussians beyond --max-abs are dropped
  rather than written as inf.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir

from threedgrut.export import NuRecExporter
from threedgrut.model.model import MixtureOfGaussians

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("ply_to_nurec_usdz")
GRUT = Path(__file__).resolve().parents[1] / "repos/3dgrut"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--max-abs", type=float, default=60000.0, help="Drop gaussians with any |coordinate| above this.")
    a = p.parse_args()

    with initialize_config_dir(version_base=None, config_dir=str(GRUT / "configs")):
        conf = compose(config_name="apps/colmap_3dgut.yaml")
    model = MixtureOfGaussians(conf)
    model.init_from_ply(str(a.input), init_model=False)

    pos = model.positions.detach()
    bad = ~torch.isfinite(pos).all(1) | (pos.abs() > a.max_abs).any(1)
    if bad.any():
        keep = ~bad
        for name in ("positions", "rotation", "scale", "density", "features_albedo", "features_specular"):
            t = getattr(model, name)
            setattr(model, name, torch.nn.Parameter(t.detach()[keep], requires_grad=t.requires_grad))
    log.info(f"{len(pos):,} gaussians, dropped {int(bad.sum()):,} beyond |{a.max_abs:g}|; "
             f"position range {pos[~bad].amin(0).cpu().numpy().round(2)} .. {pos[~bad].amax(0).cpu().numpy().round(2)}")

    a.output.parent.mkdir(parents=True, exist_ok=True)
    NuRecExporter(export_cameras=False).export(model, a.output, dataset=None, conf=conf)
    log.info(f"wrote {a.output} ({a.output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
