"""Side-by-side grid of Isaac renders of the same recording frames across methods.

    python compare_grid.py <out.jpg> <EXR_RGBD dir> <frames comma list> label=check_renders_dir [label=dir ...]
A view in a dir is matched by frame number (nearest, e.g. cuSFM keyframes are a frame or two off).
"""
import sys, re, glob
from pathlib import Path
import cv2, numpy as np
out, src, frames = sys.argv[1], Path(sys.argv[2]), [int(f) for f in sys.argv[3].split(",")]
cols = [a.split("=", 1) for a in sys.argv[4:]]
def pick(d, f):
    c = [(abs(int(re.search(r"_f(\d+)_splat", p).group(1)) - f), p) for p in glob.glob(f"{d}/*_splat.png")]
    return cv2.imread(min(c)[1]) if c else None
s = 0.42; rows = []
for f in frames:
    ph = cv2.imread(str(src / "rgb" / f"{f}.jpg")); tiles = [("phone photo", ph)] + [(lab, pick(d, f)) for lab, d in cols]
    row = []
    for lab, t in tiles:
        t = cv2.resize(t if t is not None else np.zeros_like(ph), None, fx=s, fy=s)
        cv2.putText(t, lab, (5, 18), 0, 0.5, (255, 255, 255), 3); cv2.putText(t, lab, (5, 18), 0, 0.5, (0, 0, 0), 1); row.append(t)
    rows.append(np.hstack(row))
cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88]); print("wrote", out)
