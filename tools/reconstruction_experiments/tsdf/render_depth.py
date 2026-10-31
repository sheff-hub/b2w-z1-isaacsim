#!/usr/bin/env python
"""Render per-view depth from a trained 3DGRUT/3DGUT checkpoint, for TSDF fusion.

Stage 1 of 2 (stage 2 is fuse_tsdf.py). Run inside the 3DGRUT venv, on the GPU:

    cd ~/Firdavs/repos/3dgrut && source .venv/bin/activate
    python ~/Firdavs/tools/tsdf/render_depth.py \
        --checkpoint ~/Firdavs/scenes/nurec/mine/room05/runs/room05_mcmc_400k/room05-1308_233103/ckpt_last.pt \
        --out /path/to/depth_dir

What the renderer gives us: 3DGUT returns `pred_dist`, the alpha-weighted sum
of per-gaussian hit distances along each ray (sum w_i * t_i, NOT normalised),
and `pred_opacity` = sum w_i. Expected distance is their ratio. It is a
Euclidean distance from the camera centre, so it is converted to z-depth with
the ray's own direction before Open3D sees it.

Per frame this writes depth (z, float16, scene units, 0 = invalid), the
ground-truth photo (uint8, used to colour the mesh), the pinhole intrinsics
fitted from the dataset's own rays, and the camera-to-world pose. Frames are
rendered one at a time, so GPU memory stays around 1-2 GB.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path.home() / "Firdavs/repos/3dgrut"))
import threedgrut.datasets as datasets  # noqa: E402
from threedgrut.model.model import MixtureOfGaussians  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=None, help="3DGRUT checkpoint (.pt). Or use --ply + --dataset.")
    p.add_argument("--ply", default=None,
                   help="Standard 3DGS PLY already in the COLMAP frame of --dataset (for a nerfstudio splat, "
                        "convert it with ns_splat_to_colmap_frame.py first). Rendered with 3DGUT defaults.")
    p.add_argument("--dataset", default=None, help="COLMAP folder for --ply: sparse/0/*.bin + images/.")
    p.add_argument("--out", required=True)
    p.add_argument("--min-opacity", type=float, default=0.9,
                   help="Drop pixels whose accumulated opacity is below this (sky, holes, edges).")
    p.add_argument("--edge-rel", type=float, default=0.03,
                   help="Drop pixels whose depth jumps by more than this fraction to a 4-neighbour "
                        "(blended 'flying' depths between foreground and background).")
    p.add_argument("--stride", type=int, default=2,
                   help="Keep every Nth pixel in x and y for fusion (2: 2016x1512 -> 1008x756).")
    p.add_argument("--stop-transmittance", type=float, default=None,
                   help="End each ray once its transmittance drops below this (e.g. 0.5), so depth is the "
                        "mean over the FRONT of the opacity mass instead of all of it. MCMC leaves many "
                        "semi-transparent gaussians, and the full mean lands behind the surface. Changes a "
                        "compile-time define, so the 3DGUT plugin is rebuilt once (~2 min).")
    p.add_argument("--near-cull", type=float, default=0.0,
                   help="Per frame, hide gaussians whose centre is closer than this (scene units, along the "
                        "camera's viewing axis) or behind the camera. Forward-walking video trains a 'screen' "
                        "of floaters a few cm in front of each camera (IMG_9206: 10%% of COLMAP points rendered "
                        "at 3%% of their distance); this lets rays pass it. 3DGUT only (its build_acc is a no-op).")
    p.add_argument("--min-free-gb", type=float, default=2.0,
                   help="Refuse to start if the GPU has less free memory than this. The card is shared.")
    return p.parse_args()


def fit_pinhole(rays_dir: np.ndarray, stride: int):
    """Fit Open3D pinhole intrinsics (pixel centres at integer coords) to the rays
    of the strided image. Also returns the worst residual in pixels, which must be
    tiny for a PINHOLE camera."""
    d = rays_dir[::stride, ::stride]
    h, w = d.shape[:2]
    x, y = d[..., 0] / d[..., 2], d[..., 1] / d[..., 2]
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    fx, cx = np.polyfit(x.ravel(), u.ravel(), 1)
    fy, cy = np.polyfit(y.ravel(), v.ravel(), 1)
    res = max(np.abs(fx * x + cx - u).max(), np.abs(fy * y + cy - v).max())
    return dict(width=int(w), height=int(h), fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy)), float(res)


@torch.no_grad()
def main():
    args = parse_args()
    out = Path(args.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)

    free, total = torch.cuda.mem_get_info()
    print(f"GPU: {torch.cuda.get_device_name()} | free {free / 2**30:.1f} of {total / 2**30:.1f} GB")
    if free / 2**30 < args.min_free_gb:
        raise SystemExit(f"Only {free / 2**30:.1f} GB free (< {args.min_free_gb}); not starting on a shared card.")

    assert bool(args.checkpoint) != bool(args.ply), "give --checkpoint, or --ply with --dataset"
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, weights_only=False)
        conf = ckpt["config"]
    else:
        assert args.dataset, "--ply needs --dataset"
        from hydra import compose, initialize_config_dir
        with initialize_config_dir(version_base=None, config_dir=str(Path.home() / "Firdavs/repos/3dgrut/configs")):
            conf = compose(config_name="apps/colmap_3dgut.yaml", overrides=[f"path={Path(args.dataset).resolve()}"])
    assert not conf.dataset.get("normalize_world_space", False), "expected the raw COLMAP world frame"
    need_opacity = args.min_opacity
    if args.stop_transmittance is not None:
        conf.render.min_transmittance = args.stop_transmittance
        # a ray stopped at T < s has opacity just over 1 - s; require that it got there
        need_opacity = min(args.min_opacity, 1.0 - args.stop_transmittance - 1e-3)
    print(f"render.min_transmittance = {conf.render.min_transmittance} | pixel needs opacity >= {need_opacity:.3f}")
    model = MixtureOfGaussians(conf)
    if args.checkpoint:
        model.init_from_checkpoint(ckpt, setup_optimizer=False)
        del ckpt
    else:
        model.init_from_ply(args.ply, init_model=False)
    model.build_acc()
    model.eval()
    if args.near_cull > 0:
        assert conf.render.method == "3dgut", "--near-cull edits densities per frame; only safe without a BVH"
        density0 = model.density.detach().clone()

    train, val = datasets.make(name=conf.dataset.type, config=conf, ray_jitter=None)
    meta = {"checkpoint": str(args.checkpoint or args.ply), "units": "COLMAP scene units (not metric)",
            "min_opacity": need_opacity, "stop_transmittance": args.stop_transmittance, "near_cull": args.near_cull, "edge_rel": args.edge_rel, "stride": args.stride, "frames": []}
    intr = None
    t0 = time.time()
    n = 0
    for split, ds in (("train", train), ("val", val)):
        loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
        for batch in loader:
            gb = ds.get_gpu_batch_with_intrinsics(batch)
            if args.near_cull > 0:
                T = gb.T_to_world[0]
                zc = (model.positions.detach() - T[:3, 3]) @ T[:3, 2]  # depth along the viewing axis
                model.density.data.copy_(density0)
                model.density.data[zc < args.near_cull] = -30.0  # opacity ~1e-13
            o = model(gb)
            opac = o["pred_opacity"][0, ..., 0]
            dist = o["pred_dist"][0, ..., 0] / opac.clamp_min(1e-6)
            rd = gb.rays_dir[0]
            z = dist * rd[..., 2] / rd.norm(dim=-1)
            valid = (opac >= need_opacity) & torch.isfinite(z) & (z > 0)
            # flying-pixel filter: a big relative jump to any 4-neighbour
            zp = torch.nn.functional.pad(z[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
            jump = torch.stack([(zp[1:-1, 1:-1] - zp[:-2, 1:-1]).abs(), (zp[1:-1, 1:-1] - zp[2:, 1:-1]).abs(),
                                (zp[1:-1, 1:-1] - zp[1:-1, :-2]).abs(), (zp[1:-1, 1:-1] - zp[1:-1, 2:]).abs()]).amax(0)
            valid &= jump <= args.edge_rel * z
            z = torch.where(valid, z, torch.zeros_like(z))

            s = args.stride
            if intr is None:
                intr, res = fit_pinhole(rd.cpu().numpy(), s)
                print(f"intrinsics at stride {s}: {intr} | fit residual {res:.4f} px")
                assert res < 0.05, "rays are not pinhole; this script assumes a PINHOLE camera"
                meta["intrinsics"] = intr
            depth = z[::s, ::s].cpu().numpy().astype(np.float16)
            rgb = (gb.rgb_gt[0, ::s, ::s].clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
            c2w = gb.T_to_world[0].cpu().numpy().astype(np.float64)
            if c2w.shape == (3, 4):
                c2w = np.vstack([c2w, [0, 0, 0, 1]])
            name = f"{n:04d}"
            np.savez_compressed(out / "frames" / f"{name}.npz", depth=depth, rgb=rgb, c2w=c2w)
            image = Path(ds.image_paths[int(batch["frame_idx"][0])]).name
            meta["frames"].append({"id": name, "image": image, "split": split, "valid_frac": float(valid[::s, ::s].float().mean())})
            n += 1
            if n % 50 == 0 or n == 1:
                print(f"  {n} frames | {time.time() - t0:.0f}s | valid {meta['frames'][-1]['valid_frac']:.0%} | "
                      f"GPU peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    meta["seconds"] = time.time() - t0
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    vf = np.array([f["valid_frac"] for f in meta["frames"]])
    print(f"done: {n} frames in {meta['seconds']:.0f}s | valid pixels median {np.median(vf):.0%}, min {vf.min():.0%} | "
          f"GPU peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")


if __name__ == "__main__":
    main()
