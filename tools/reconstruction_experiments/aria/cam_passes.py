"""Camera positions per frame in the levelled corridor frame (metres, estimate), to spot inconsistent passes."""
import json, sys
from pathlib import Path
import numpy as np
repo = Path(sys.argv[1]); rep = json.loads(Path(sys.argv[2]).read_text())
proc = repo / "data/processed"
dp = json.loads(next((proc / "training_output").rglob("dataparser_transforms.json")).read_text())
tj = json.loads((proc / "transforms.json").read_text())
A = np.r_[np.array(tj["applied_transform"]), [[0, 0, 0, 1]]]
T = (np.r_[np.array(dp["transform"]), [[0, 0, 0, 1]]] @ np.linalg.inv(A))[:3]; s = dp["scale"]
lf = rep["level_frame"]; e1, e2, up, c = (np.array(lf[k]) for k in ("e1", "e2", "up", "origin")); mpu = rep["m_per_unit_estimate"]
fr = sorted(tj["frames"], key=lambda f: f["file_path"])
idx = np.array([int(Path(f["file_path"]).stem.split("_")[1]) for f in fr])
P = np.array([f["transform_matrix"] for f in fr])[:, :3, 3]
P = (P @ T[:, :3].T + T[:, 3]) * s
L = np.c_[(P - c) @ e1, (P - c) @ e2, (P - c) @ up] * mpu
for i in range(0, len(idx), 6):
    print(f"frame {idx[i]:4d} (t={idx[i]/5:5.1f}s): along {L[i,0]:+6.2f}  across {L[i,1]:+6.2f}  height {L[i,2]:+5.2f}")
