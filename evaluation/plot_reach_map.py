"""Turn reach_map.py's JSON into a figure: which targets the robot reaches.

    python3 tools/plot_reach_map.py logs/eval/reach_map_wb9.json [more.json ...] -o out.png

Each input becomes a column of panels: hand error, base height, body pitch. Cells
the policy reached within tolerance are outlined, so the reachable region reads at
a glance on a slide.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("-o", "--out", default="reach_map.png")
ap.add_argument("--labels", default=None, help="comma-separated column titles")
args = ap.parse_args()

data = [json.load(open(f)) for f in args.files]
labels = (args.labels.split(",") if args.labels
          else [f.split("/")[-1].replace("reach_map_", "").replace(".json", "") for f in args.files])

fig, axes = plt.subplots(3, len(data), figsize=(5.2 * len(data), 11), squeeze=False)
for col, d in enumerate(data):
    xs, zs = d["xs"], d["zs"]
    err = np.array(d["err_m"]) * 100
    base = np.array(d["base_h"]).reshape(len(zs), len(xs))
    pitch = np.array(d["pitch_deg"]).reshape(len(zs), len(xs))
    tol = d["tol"] * 100
    ext = [xs[0] - 0.05, xs[-1] + 0.05, zs[0] - 0.05, zs[-1] + 0.05]

    for row, (grid, title, cmap, kw) in enumerate([
        (err, "hand error (cm)", "viridis_r", dict(vmin=0, vmax=20)),
        (base, "base height (m)", "coolwarm", dict(vmin=0.28, vmax=0.68)),
        (pitch, "body pitch (deg)", "PuOr", dict(vmin=-20, vmax=20)),
    ]):
        ax = axes[row][col]
        im = ax.imshow(grid, origin="lower", aspect="auto", extent=ext, cmap=cmap, **kw)
        # mark the cells that were actually reached
        for j, z in enumerate(zs):
            for i, x in enumerate(xs):
                if err[j][i] <= tol:
                    ax.plot(x, z, "o", ms=3, mfc="none", mec="white", mew=1.2)
        ax.set_title(f"{labels[col]} — {title}", fontsize=11)
        ax.set_xlabel("forward distance (m)")
        ax.set_ylabel("target height (m)")
        fig.colorbar(im, ax=ax, fraction=0.046)

    n_ok = int((err <= tol).sum())
    axes[0][col].text(0.02, 0.98, f"{n_ok}/{err.size} reached within {tol:.0f} cm",
                      transform=axes[0][col].transAxes, va="top", fontsize=10,
                      bbox=dict(fc="white", alpha=0.8, ec="none"))

fig.suptitle("Where the hand can be placed (white rings = reached)", fontsize=13)
fig.tight_layout()
fig.savefig(args.out, dpi=140)
print("wrote", args.out)
