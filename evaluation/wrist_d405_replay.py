"""Would the RealSense D405 holder have hit anything? Replays recorded picks (recorder format 2) with the
holder's geometry (tools/wrist_d405.py) attached to the recorded gripper pose, and measures, per episode,
its clearance to the table top and to the boxes. No simulator or GPU needed.

    repos/3dgrut/.venv/bin/python evaluation/wrist_d405_replay.py datasets/raw/demo_v2

The same is measured for a mirror-image holder on the gripper's +y side, the side rejected in the design.

Approximations, all conservative:
- the base's world pose is rebuilt from the recorded x, y, yaw, height and projected gravity;
- box yaw is not recorded: each box is the upright cylinder around it (radius = half its diagonal);
- the target box is static until the gripper has closed on it; only those frames are checked against it;
  the neighbours (slots 0.3 m either side) and the table top are checked over the whole pick and carry.
"""

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
import wrist_d405 as w  # noqa: E402

TABLE_HEIGHT, TABLE_SIZE = 0.60, (0.6, 1.0)
TABLES3 = {"red": (4.3, -4.5), "yellow": (4.3, 1.0), "green": (4.3, 6.5)}
BOX_H, BOX_R = 0.08, 0.05 / 2 * 2 ** 0.5
SLOT = 0.3


def quat_R(q):
    w_, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w_ * z), 2 * (x * z + w_ * y)],
                     [2 * (x * y + w_ * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w_ * x)],
                     [2 * (x * z - w_ * y), 2 * (y * z + w_ * x), 1 - 2 * (x * x + y * y)]])


def base_R(yaw, g_b):
    """Base -> world rotation with the recorded yaw whose projected gravity is g_b."""
    a = g_b / np.linalg.norm(g_b)
    b = np.array([0.0, 0.0, -1.0])
    v, c = np.cross(a, b), float(a @ b)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    tilt = np.eye(3) + K + K @ K / (1 + c)  # rotates a onto b (shortest arc)
    heading = np.arctan2(tilt[1, 0], tilt[0, 0])
    d = yaw - heading
    Rz = np.array([[np.cos(d), -np.sin(d), 0], [np.sin(d), np.cos(d), 0], [0, 0, 1]])
    return Rz @ tilt


def holder_points(mirror=False):
    P = w.holder_points() - np.array([w.STATOR_X, 0, 0])  # link06 -> gripperStator frame (= the recorded EE frame)
    if mirror:
        P[:, 1] *= -1
    return P


def episode(path, H):
    z = np.load(os.path.join(path, "data.npz"))
    m = json.load(open(os.path.join(path, "meta.json")))
    ph = z["phase"]
    names = m["phases"]
    pick = np.where(ph == names.index("pick"))[0]
    if len(pick) == 0:
        return None
    grip = z["action_arm"][:, 6]
    opened = np.where((np.arange(len(grip)) >= pick[0]) & (grip < -0.5))[0]
    closed_after = np.where((np.arange(len(grip)) > (opened[-1] if len(opened) else pick[0])) & (grip > -0.05))[0]
    t_closed = closed_after[0] if len(closed_after) else pick[-1]
    frames = np.where((ph == names.index("pick")) | (ph == names.index("carry")))[0]
    tx, ty = TABLES3[m["target"]["table"]] if m.get("target") else (4.3, 1.0)
    bx, by, bz = m["box_start_w"]
    out = {"table_mm": np.inf, "box_mm": np.inf, "neighbour_mm": np.inf}
    for i in frames:
        x, y, yaw = z["world"][i]
        Rb = base_R(float(yaw), z["base"][i, 6:9].astype(float))
        pb = np.array([x, y, z["base"][i, 9]])
        Re = quat_R(z["ee_pose"][i, 3:7].astype(float))
        pe = z["ee_pose"][i, :3]
        Pw = (H @ Re.T + pe) @ Rb.T + pb
        inside = (np.abs(Pw[:, 0] - tx) <= TABLE_SIZE[0] / 2) & (np.abs(Pw[:, 1] - ty) <= TABLE_SIZE[1] / 2)
        if inside.any():
            out["table_mm"] = min(out["table_mm"], float((Pw[inside, 2] - TABLE_HEIGHT).min()) * 1000)
        def cyl(cx, cy):  # distance (mm) from the holder to the solid upright cylinder around a box
            r = np.maximum(np.hypot(Pw[:, 0] - cx, Pw[:, 1] - cy) - BOX_R, 0.0)
            dz = np.maximum(0.0, np.maximum(TABLE_HEIGHT - Pw[:, 2], Pw[:, 2] - (TABLE_HEIGHT + BOX_H)))
            return float(np.hypot(r, dz).min()) * 1000
        if i <= t_closed:
            out["box_mm"] = min(out["box_mm"], cyl(bx, by))
        for s in (-SLOT, SLOT):
            if abs(by + s - ty) <= TABLE_SIZE[1] / 2:  # a neighbouring slot on the same table
                out["neighbour_mm"] = min(out["neighbour_mm"], cyl(bx, by + s))
    out["success"] = m.get("success")
    return out


def main():
    runs = sys.argv[1:] or [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets", "raw", "demo_v2")]
    eps = sorted(p for r in runs for p in glob.glob(os.path.join(r, "ep_*")))
    for label, mirror in (("D405 holder, -y side (the design)", False), ("mirror image on the +y side", True)):
        H = holder_points(mirror)
        rows = [r for r in (episode(p, H) for p in eps) if r]
        t = np.array([r["table_mm"] for r in rows]); b = np.array([r["box_mm"] for r in rows])
        nb = np.array([r["neighbour_mm"] for r in rows])
        print(f"{label}: {len(rows)} picks")
        print(f"  table top clearance  min {t.min():6.1f} mm, median {np.median(t):6.1f}; episodes < 10 mm: {(t < 10).sum()}")
        print(f"  target box (before it is gripped)  min {b.min():6.1f} mm, median {np.median(b):6.1f}; "
              f"< 10 mm: {(b < 10).sum()}")
        print(f"  neighbouring boxes   min {nb.min():6.1f} mm; < 10 mm: {(nb < 10).sum()}")


if __name__ == "__main__":
    main()
