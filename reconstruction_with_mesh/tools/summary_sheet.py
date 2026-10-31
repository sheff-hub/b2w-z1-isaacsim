"""Contact sheet of Isaac check renders: phone photo | render | collider alone | collider edges on render | after drop.

    python summary_sheet.py <check_renders dir> <views.json> <EXR_RGBD dir> <out.jpg> [--title "..."]

The phone photo of a view is views.json's "photo" (make_views.py writes it); older views.json files name the frame
only (<name>_f<frame>), which is then looked up in <EXR_RGBD dir>/rgb.
"""
import argparse, json
from pathlib import Path
import cv2, numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("views"); ap.add_argument("src"); ap.add_argument("out")
ap.add_argument("--title", default=""); ap.add_argument("--scale", type=float, default=0.4)
a = ap.parse_args(); d = Path(a.dir); V = json.load(open(a.views))["views"]
rows = []
for v in V:
    n = v["name"]
    ph = cv2.imread(v["photo"] if "photo" in v else str(Path(a.src) / "rgb" / f"{int(n.split('_f')[1])}.jpg"))
    sp, co, dr = (cv2.imread(str(d / f"{n}_{s}.png")) for s in ("splat", "collider", "drop"))
    if sp is None: continue
    co = co if co is not None else np.zeros_like(sp); dr = dr if dr is not None else np.zeros_like(sp)
    e = cv2.Canny(cv2.cvtColor(co, cv2.COLOR_BGR2GRAY), 40, 120); ov = sp.copy(); ov[cv2.dilate(e, np.ones((2, 2), np.uint8)) > 0] = (0, 140, 255)
    tiles = [ph, sp, co, ov, dr]
    tiles = [cv2.resize(t, None, fx=a.scale, fy=a.scale) for t in tiles]
    for t, lab in zip(tiles, ["phone photo", "render", "collider only", "collider edges on render", "after ball drop"]):
        cv2.putText(t, lab, (6, 20), 0, 0.55, (255, 255, 255), 3); cv2.putText(t, lab, (6, 20), 0, 0.55, (0, 0, 0), 1)
    lab = n + (f"  {v['height_m']:.2f} m" if "height_m" in v else "")
    cv2.putText(tiles[0], lab, (6, tiles[0].shape[0] - 10), 0, 0.55, (255, 255, 255), 3); cv2.putText(tiles[0], lab, (6, tiles[0].shape[0] - 10), 0, 0.55, (0, 0, 0), 1)
    rows.append(np.hstack(tiles))
sheet = np.vstack(rows)
if a.title:
    bar = np.full((40, sheet.shape[1], 3), 255, np.uint8); cv2.putText(bar, a.title, (8, 28), 0, 0.8, (0, 0, 0), 2); sheet = np.vstack([bar, sheet])
cv2.imwrite(a.out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 88]); print(f"wrote {a.out} ({len(rows)} views)")
