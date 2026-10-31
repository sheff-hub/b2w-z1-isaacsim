"""Render a trained 3DGRUT splat at the localised test-clip cameras and score it (run in 3DGRUT's venv).

    python render_test.py --checkpoint <ckpt_last.pt> --path <scene>/test/dataset --out <scene>/test/render

3DGRUT's own evaluation (threedgrut.render.Renderer, the one train.py ends with), pointed at the test dataset and
told to use every image (its usual split keeps only every 8th as test images). PSNR / SSIM / LPIPS as there, plus the
colour-corrected PSNR (cc_psnr: per-image affine colour fit first, so a different exposure of the clip is not
counted as a modelling error). Also writes per_image.json (PSNR of every image, by name) and sheet.jpg
(photo | render for a few test images).
"""
import argparse, glob, json, os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from threedgrut.render import Renderer

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True); ap.add_argument("--path", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()

_orig = Renderer.create_test_dataloader


def every_image(self, conf):
    # every image a test image: interval 1 (every image index % 1 == 0). Not 0: 3DGRUT's "non-positive = all" path
    # keeps an integer index array and then np.where() on it drops image 0 (IndexError in __getitem__).
    conf.dataset.test_split_interval = 1
    conf.dataset.load_exif = False
    return _orig(self, conf)


Renderer.create_test_dataloader = every_image
r = Renderer.from_checkpoint(checkpoint_path=a.checkpoint, out_dir=a.out, path=a.path, save_gt=True,
                             computes_extra_metrics=True)
r.render_all()

names = []
for ln in open(os.path.join(a.path, "sparse", "0", "images.txt")):
    f = ln.split()
    if ln.startswith("#") or len(f) < 10:
        continue
    names.append(f[9])
names.sort()                                   # 3DGRUT orders the images by name
rd = sorted(glob.glob(os.path.join(a.out, "**", "ours_*", "renders", "*.png"), recursive=True))
gt = sorted(glob.glob(os.path.join(a.out, "**", "ours_*", "gt", "*.png"), recursive=True))
per = []
for k, (pr, pg) in enumerate(zip(rd, gt)):
    x = np.asarray(Image.open(pr).convert("RGB"), np.float32) / 255; y = np.asarray(Image.open(pg).convert("RGB"), np.float32) / 255
    mse = float(np.mean((x - y) ** 2))
    per.append({"name": names[k] if k < len(names) else str(k), "psnr": 10 * np.log10(1 / max(mse, 1e-10))})
json.dump(per, open(os.path.join(a.out, "per_image.json"), "w"), indent=1)
if rd:
    try:
        font = ImageFont.load_default(size=30)
    except TypeError:                          # Pillow < 10.1
        font = ImageFont.load_default()
    pick = np.linspace(0, len(rd) - 1, min(6, len(rd))).round().astype(int)
    tiles = []
    for k in pick:
        g, x = Image.open(gt[k]).convert("RGB"), Image.open(rd[k]).convert("RGB")
        w, h = g.size; t = Image.new("RGB", (2 * w, h)); t.paste(g, (0, 0)); t.paste(x, (w, 0))
        d = ImageDraw.Draw(t)
        for x0, txt in ((8, f"test photo {per[k]['name']}"), (w + 8, f"splat {per[k]['psnr']:.1f} dB")):
            d.text((x0, 8), txt, fill=(255, 255, 0), font=font, stroke_width=3, stroke_fill=(0, 0, 0))
        tiles.append(t.resize((w, h // 2)))
    sheet = Image.new("RGB", (tiles[0].width * 2, tiles[0].height * ((len(tiles) + 1) // 2)))
    for k, t in enumerate(tiles):
        sheet.paste(t, ((k % 2) * t.width, (k // 2) * t.height))
    sheet.save(os.path.join(a.out, "sheet.jpg"), quality=88)
print(f"RENDERTEST| {len(per)} images, PSNR median {np.median([p['psnr'] for p in per]):.2f} dB")
