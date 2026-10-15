#!/usr/bin/env python
"""Convert a folder of Apple ProRAW DNGs to JPEGs sized for the NuRec pipeline.

COLMAP cannot read DNG, and 3DGRUT would train on the full 48 MP frames, which
is far more resolution than the pipeline needs. This downsamples to a fixed long
side (default 2000 px, matching the reference scene's 2000x1498 at ~3 MP) and
writes high-quality JPEG.

    python dng_to_jpeg.py --src <dng dir> --dst <images dir> [--long-side 2000]

Notes:
  - EXIF is NOT carried over. Pillow exposes no raw EXIF blob for these DNGs
    ("has raw exif blob: False"), so there is nothing to copy. Intrinsics are
    instead seeded explicitly into COLMAP from FocalLengthIn35mmFilm; see
    run_colmap_docs_recipe.sh. That is the more reliable route regardless.
  - Orientation is applied via exif_transpose so every output is upright and
    consistent. COLMAP ignores the EXIF orientation flag and reads raw pixel
    dimensions, so baking it in avoids a mixed-orientation camera model.
  - Output names keep the source stem (IMG_7184.DNG -> IMG_7184.jpg) so a frame
    can always be traced back to its original.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from PIL import Image, ImageOps


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--long-side", type=int, default=2000)
    p.add_argument("--quality", type=int, default=95)
    p.add_argument("--ext", default=".DNG")
    p.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
    return p.parse_args()


def convert(job):
    src, dst, long_side, quality = job
    try:
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im)
            if im.mode != "RGB":
                im = im.convert("RGB")
            w, h = im.size
            scale = long_side / max(w, h)
            if scale < 1.0:
                im = im.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
            im.save(dst, "JPEG", quality=quality, subsampling=0)
            return (os.path.basename(src), im.size, None)
    except Exception as e:  # noqa: BLE001 - report and keep going
        return (os.path.basename(src), None, f"{type(e).__name__}: {e}")


def main():
    a = parse_args()
    os.makedirs(a.dst, exist_ok=True)
    srcs = sorted(f for f in os.listdir(a.src) if f.upper().endswith(a.ext.upper()))
    if not srcs:
        raise SystemExit(f"ERROR: no {a.ext} files in {a.src}")

    jobs = [
        (os.path.join(a.src, f), os.path.join(a.dst, os.path.splitext(f)[0] + ".jpg"), a.long_side, a.quality)
        for f in srcs
    ]
    print(f"converting {len(jobs)} files -> {a.dst}  (long side {a.long_side}, q{a.quality}, {a.workers} workers)", flush=True)

    sizes = {}
    failures = []
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for fut in as_completed([ex.submit(convert, j) for j in jobs]):
            name, size, err = fut.result()
            done += 1
            if err:
                failures.append((name, err))
            else:
                sizes[size] = sizes.get(size, 0) + 1
            if done % 25 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)}", flush=True)

    print(f"\nconverted : {len(jobs) - len(failures)}")
    print(f"sizes     : {sizes}")
    if failures:
        print(f"FAILURES  : {len(failures)}")
        for n, e in failures[:10]:
            print(f"  {n}: {e}")
        return 1
    if len(sizes) != 1:
        print("WARNING: output sizes are not uniform; COLMAP's single-camera model assumes they are.")
        return 1
    print("all outputs uniform in size")
    return 0


if __name__ == "__main__":
    sys.exit(main())
