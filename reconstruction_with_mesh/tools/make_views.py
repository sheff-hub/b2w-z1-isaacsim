"""views.json for render_stage_views.py (the Isaac check): camera poses of recording frames + ball drop points.

    python make_views.py <poses.npz> <out.json> --frames 900 4200 ... [--ahead 1.8] [--window-s 4]

poses.npz: c2w (N, 4, 4), OpenCV cameras, in the stage frame (metres, Z-up, floor at 0), K (N, 4), w, h (and, for
several passes, srcs + pass_start). Each requested frame is moved to the nearest frame of its pass (within
--window-s) that looks along the walk: the point the phone reached --ahead m later (along the same pass) is within
25 deg of the camera's heading. The ball is dropped there, 1 m above the floor: a spot the phone walked through, so it
is free space with floor under it, and in view. (Dropping straight ahead of the camera put balls behind walls
whenever the phone was turning in a doorway.) Frames with no such view nearby keep the requested frame and a drop
1.8 m straight ahead. Each view names its phone photo, for the contact sheet.
"""
import argparse, json
import numpy as np
import r3dlib as L

ap = argparse.ArgumentParser(); ap.add_argument("poses"); ap.add_argument("out")
ap.add_argument("--frames", type=int, nargs="+", required=True); ap.add_argument("--fps", type=float, default=60.0)
ap.add_argument("--ahead", type=float, default=1.8); ap.add_argument("--window-s", type=float, default=4.0)
ap.add_argument("--max-angle", type=float, default=25.0)
a = ap.parse_args()
z = np.load(a.poses); c2w = z["c2w"]; K = z["K"]; N = len(c2w); pl = L.Passes(z)
ts_local = z["ts_local"] if "ts_local" in z else z["ts"]
p = c2w[:, :2, 3]; s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
fwd = c2w[:, :2, 2] / np.maximum(np.linalg.norm(c2w[:, :2, 2], axis=1, keepdims=True), 1e-9)


def along(f):
    """(angle to the path point --ahead m later in the same pass, that point) or (inf, None) near the pass's end."""
    j = int(np.searchsorted(s, s[f] + a.ahead))
    if j >= N or pl.pass_of[j] != pl.pass_of[f]:
        return np.inf, None
    d = p[j] - p[f]
    return np.degrees(np.arccos(np.clip(d @ fwd[f] / max(np.linalg.norm(d), 1e-9), -1, 1))), p[j]


views, drops = [], []
win = int(a.window_s * a.fps)
for f0 in a.frames:
    best = None
    lo, hi = pl.start[pl.pass_of[f0]], pl.start[pl.pass_of[f0] + 1]
    for f in sorted(range(max(lo, f0 - win), min(hi, f0 + win + 1)), key=lambda f: abs(f - f0)):
        ang, q = along(f)
        if ang < a.max_angle:
            best = (f, q)
            break
    if best:
        f, q = best
        drop = [float(q[0]), float(q[1]), 1.0]
    else:
        f = f0
        drop = (c2w[f, :3, 3] + a.ahead * np.r_[fwd[f], 0] + np.array([0, 0, 1.0 - c2w[f, 2, 3]])).tolist()
    pp = int(pl.pass_of[f])
    tag = f"p{pp + 1}_" if pl.n > 1 else ""
    views.append({"name": f"{tag}t{ts_local[f] - ts_local[pl.start[pp]]:03.0f}s_f{f:06d}", "c2w": c2w[f].tolist(),
                  "fx": float(K[f][0]), "photo": str(pl.rgb_path(f)), "height_m": float(c2w[f, 2, 3])})
    drops.append(drop)
W, H = (int(z["w"]), int(z["h"])) if "w" in z else (720, 960)
json.dump({"width": W, "height": H, "views": views, "drops": drops}, open(a.out, "w"))
print(f"{len(views)} views -> {a.out}: " + ", ".join(v["name"] for v in views))
