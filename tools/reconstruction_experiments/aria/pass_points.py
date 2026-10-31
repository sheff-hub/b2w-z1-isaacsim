"""Which pass sees which COLMAP points, and where those points sit in the levelled frame (m, estimate)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path.home() / "Firdavs/repos/Depth-Anything-3/src"))
from depth_anything_3.utils.read_write_model import read_model
repo = Path(sys.argv[1]); rep = json.loads(Path(sys.argv[2]).read_text()); sparse = sys.argv[3]
proc = repo / "data/processed"
dp = json.loads(next((proc / "training_output").rglob("dataparser_transforms.json")).read_text())
T, s = np.array(dp["transform"]), dp["scale"]          # maps ORIGINAL COLMAP coordinates
lf = rep["level_frame"]; e1, e2, up, c = (np.array(lf[k]) for k in ("e1", "e2", "up", "origin")); mpu = rep["m_per_unit_estimate"]
lev = lambda p: (lambda q: np.c_[(q - c) @ e1, (q - c) @ e2, (q - c) @ up] * mpu)((p @ T[:, :3].T + T[:, 3]) * s)
cams, ims, pts = read_model(sparse, ".txt")
fidx = {i: int(im.name.split("_")[1].split(".")[0]) for i, im in ims.items()}
passes = {"A (frames 1-83, walk out)": (1, 83), "B (frames 96-144, walk back)": (96, 144), "C (frames 145-341, rest)": (145, 341)}
lab = {}
for pid, p in pts.items():
    f = np.array([fidx[i] for i in p.image_ids])
    lab[pid] = tuple(int(np.any((f >= a) & (f <= b))) for a, b in passes.values())
from collections import Counter
cnt = Counter(lab.values())
print("points seen by passes (A,B,C):", dict(sorted(cnt.items(), key=lambda kv: -kv[1])))
for k, name in enumerate(passes):
    only = [pid for pid, l in lab.items() if l[k] and sum(l) == 1]
    P = lev(np.array([pts[i].xyz for i in only]))
    q = lambda a: np.percentile(a, [5, 50, 95]).round(2).tolist()
    print(f"{name}: {len(only)} points only it sees | across p5/50/95 {q(P[:,1])} | height {q(P[:,2])}")
