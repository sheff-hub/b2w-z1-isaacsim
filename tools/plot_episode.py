"""One review image per recorded episode (raw format, project/recorder.py).

    python3 tools/plot_episode.py datasets/raw/<run> [...]

Writes <run>_review/<episode>.png (the raw episode folders belong to the container user):
  - the base path from above, coloured by phase, with the table(s), the start and the end, and
    (recorder format 2) the planned path at a few moments: DualVLN's / Nav2's / the docking line;
  - the base velocity command and the gripper target over time, with the phases shaded;
  - EE target height in the base frame;
  - 4 front-camera and 4 wrist-camera frames spread over the episode (format 2: with their depth).
And <run>_review/overview.png: every episode's path on one map, green = picked, red = failed.
"""

import glob
import json
import os
import sys

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TABLE = (4.0, 4.6, 0.5, 1.5)
TABLES3 = {"red": (4.0, 4.6, -5.0, -4.0), "yellow": (4.0, 4.6, 0.5, 1.5), "green": (4.0, 4.6, 6.0, 7.0)}
TABLE_RGB = {"red": "#9e2b24", "yellow": "#dbb338", "green": "#337a38"}
COLORS = {"idle": "#999999", "walk": "#1f77b4", "backoff": "#9467bd", "dock": "#ff7f0e", "pick": "#2ca02c",
          "carry": "#17becf", "deliver": "#d62728", "nav2": "#8c564b", "nav2_dock": "#e377c2"}


def table(ax, three=False):
    rects = TABLES3.items() if three else [("wood", TABLE)]
    for name, (x0, x1, y0, y1) in rects:
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, color=TABLE_RGB.get(name, "#c8a96e"), alpha=0.6))


def frames(path, n=4):
    r = imageio.get_reader(path)
    total = r.count_frames()
    out = [r.get_data(int(i)) for i in np.linspace(0, total - 1, n)]
    r.close()
    return out


def episode(ep, out_dir):
    meta = json.load(open(os.path.join(ep, "meta.json")))
    z = np.load(os.path.join(ep, "data.npz"))
    phases = meta["phases"]
    ph = z["phase"]
    t = np.arange(len(ph)) / meta["fps"]
    v2 = meta.get("format", 1) >= 2
    three = meta.get("scene") == "three_tables"
    fig = plt.figure(figsize=(16, 11 if v2 else 9))
    gs = fig.add_gridspec(4 if v2 else 3, 4)
    ax = fig.add_subplot(gs[0:2, 0:2])
    table(ax, three)
    w = z["world"]
    for i, name in enumerate(phases):
        m = ph == i
        if m.any():
            ax.scatter(w[m, 0], w[m, 1], s=4, c=COLORS.get(name, "k"), label=name)
    if v2:  # the planned path ahead at ~8 moments, back in the world frame
        plan = z["plan"]
        for i in np.linspace(0, len(ph) - 1, 24).astype(int):
            if np.isfinite(plan[i, 0, 0]):
                c, s_ = np.cos(w[i, 2]), np.sin(w[i, 2])
                px = w[i, 0] + c * plan[i, :, 0] - s_ * plan[i, :, 1]
                py = w[i, 1] + s_ * plan[i, :, 0] + c * plan[i, :, 1]
                ax.plot(px, py, "-", c="k", lw=0.6, alpha=0.5)
    ax.plot(w[0, 0], w[0, 1], "k^", ms=10, label="start")
    ax.plot(w[-1, 0], w[-1, 1], "ks", ms=8, label="end")
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="best")
    tgt = meta.get("target")
    ax.set_title(f"{os.path.basename(ep)}: {meta['task']}" + (f"  [target {tgt['box']} on {tgt['table']}]" if tgt else "")
                 + (f"\nDualVLN: {meta.get('nav_instruction')!r} right table {meta.get('vln_right_table')}" if meta.get("nav_instruction") else "")
                 + f"\nsuccess {meta.get('success')}  "
                 + (f"delivered {meta.get('delivered')}" if meta.get("deliver_task", True) else "pick only")
                 + f"  DualVLN {meta.get('vln_reason')} arrived {meta.get('vln_arrived')}",
                 fontsize=9)
    ax2 = fig.add_subplot(gs[0, 2:])
    a = z["action_base"]
    for k, lab in enumerate(["vx", "vy", "wz"]):
        ax2.plot(t, a[:, k], label=lab, lw=1)
    ax2.plot(t, -z["action_arm"][:, 6], label="gripper open (-target)", lw=1, ls="--")
    for i, name in enumerate(phases):
        m = ph == i
        if m.any():
            ax2.fill_between(t, -1.2, 1.2, where=m, color=COLORS.get(name, "k"), alpha=0.08, step="mid")
    ax2.set_ylim(-1.2, 1.5)
    ax2.legend(fontsize=7, ncol=4)
    ax2.set_title("base command (m/s, rad/s) and gripper", fontsize=9)
    ax3 = fig.add_subplot(gs[1, 2:])
    ee = z["action_ee"]
    ax3.plot(t, ee[:, 0], label="EE target x (base)")
    ax3.plot(t, ee[:, 1], label="y")
    ax3.plot(t, ee[:, 2], label="z")
    ax3.legend(fontsize=7, ncol=3)
    ax3.set_title("EE target pose in the base frame (m)", fontsize=9)
    idx = np.linspace(0, len(ph) - 1, 4).astype(int)
    fr_front, fr_wrist = frames(os.path.join(ep, "front.mp4")), frames(os.path.join(ep, "wrist.mp4"))
    for k, img in enumerate(fr_front):
        axi = fig.add_subplot(gs[2, k])
        axi.imshow(np.concatenate([img, fr_wrist[k]], axis=0))
        axi.set_title(f"t {idx[k] / meta['fps']:.0f} s  {phases[ph[idx[k]]]}", fontsize=8)
        axi.axis("off")
        if v2:
            axd = fig.add_subplot(gs[3, k])
            df, dw = z["depth_front"][idx[k]] / 1000.0, z["depth_wrist"][idx[k]] / 1000.0
            axd.imshow(np.concatenate([df, dw], axis=0), cmap="turbo", vmin=0, vmax=6)
            axd.set_title("depth front / wrist (0-6 m)", fontsize=8)
            axd.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, os.path.basename(ep) + ".png"), dpi=80)
    plt.close(fig)
    return meta, w


def run(root):
    eps = sorted(d for d in glob.glob(os.path.join(root, "ep_*")) if os.path.exists(os.path.join(d, "meta.json")))
    out_dir = root.rstrip("/") + "_review"
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    three = any(json.load(open(os.path.join(e, "meta.json"))).get("scene") == "three_tables" for e in eps)
    table(ax, three)
    for ep in eps:
        meta, w = episode(ep, out_dir)
        ax.plot(w[:, 0], w[:, 1], lw=0.8, c="g" if meta.get("success") else "r", alpha=0.7)
        ax.plot(w[0, 0], w[0, 1], "k.", ms=4)
    ax.set_aspect("equal")
    ok = sum(json.load(open(os.path.join(e, "meta.json"))).get("success", False) for e in eps)
    ax.set_title(f"{os.path.basename(root)}: {len(eps)} episodes, {ok} picked (green)")
    fig.savefig(os.path.join(out_dir, "overview.png"), dpi=90)
    plt.close(fig)
    print(f"{root}: {len(eps)} episodes reviewed -> {out_dir}/")


for r in sys.argv[1:]:
    run(r)
