"""Loop route for a robot through a reconstructed scene, centred in the corridors, from the collider mesh.

    .venv/bin/python tools/robot_route.py <scene>/mesh/collider.ply <scene>/robot/route.json \
        --waypoints "x,y; x,y; ..." [--closed] [--poses <scene>/record3d/poses_refined.npz]

Optional, after a reconstruction: waypoints are picked by eye from <scene>/eval/plan.png (metres, the stage frame).
The mesh must be in the stage frame (metres, Z-up, floor at z = 0), e.g. <scene>/mesh/collider.ply. Steps:
  1. rasterise the mesh into a 5 cm plan grid: floor = surfaces within 10 cm of z = 0 that face up,
     obstacle = anything between --zmin and --zmax (the robot's body height range);
  2. traversable = floor (gaps up to 15 cm closed) and not obstacle; clearance = distance to the nearest
     non-traversable cell;
  3. shortest paths between consecutive waypoints, on cells with clearance >= --clear, with step cost
     1 / min(clearance, 1.2 m)^2, which keeps the path on the corridor centre line;
  4. smoothed (Gaussian, --smooth m along the path) and resampled every 5 cm.
Writes out.json {"points": [[x, y], ...], "clearance" (m, per point), "closed", "length_m", "min_clearance_m", "start": [x, y, yaw]} and
out.png (plan: obstacles, clearance, route; --poses adds the recording's camera path in grey) and
out.grid.npz (clearance grids, [row = y, col = x]: clear = the robot's, clear_cam = to anything 0.12-2 m high, for
a chase camera; x0, y0, res). Example (the 22 Oct walk): --closed --clear 0.32 --smooth 0.3
--waypoints "3,-11; -4,-16.5; -8.8,-12; 0.7,0.3; 7.8,-7.3" gave a 46 m loop through a 0.35 m-clearance door.
"""
import argparse, json
from pathlib import Path
import numpy as np, trimesh
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

ap = argparse.ArgumentParser(); ap.add_argument("mesh"); ap.add_argument("out")
ap.add_argument("--waypoints", required=True); ap.add_argument("--closed", action="store_true")
ap.add_argument("--poses"); ap.add_argument("--res", type=float, default=0.05)
ap.add_argument("--zmin", type=float, default=0.12); ap.add_argument("--zmax", type=float, default=1.1)
ap.add_argument("--clear", type=float, default=0.40); ap.add_argument("--smooth", type=float, default=0.5)
a = ap.parse_args(); res = a.res

m = trimesh.load(a.mesh, process=False, force="mesh"); V = np.asarray(m.vertices, float); F = np.asarray(m.faces)
T = V[F]; nz = np.abs(m.face_normals[:, 2])
# barycentric samples, at most ~res/2 apart (decimated meshes have big flat triangles on walls and floor)
L = np.max(np.linalg.norm(T - np.roll(T, 1, axis=1), axis=2), axis=1); n = np.clip(np.ceil(L / (res / 2)), 1, 200).astype(int)
pts, up = [], []
for k in np.unique(n):
    sel = np.where(n == k)[0]; i, j = np.meshgrid(np.arange(k + 1), np.arange(k + 1)); ok = i + j <= k
    w = np.stack([i[ok], j[ok], k - i[ok] - j[ok]], 1) / k                    # (s, 3)
    P = np.einsum("sb,fbc->fsc", w, T[sel]).reshape(-1, 3); pts.append(P); up.append(np.repeat(nz[sel], len(w)))
P = np.concatenate(pts); up = np.concatenate(up)
x0, y0 = P[:, 0].min() - 1, P[:, 1].min() - 1; W = int((P[:, 0].max() + 1 - x0) / res) + 1; H = int((P[:, 1].max() + 1 - y0) / res) + 1
ij = lambda x, y: (np.floor((np.asarray(y) - y0) / res).astype(int), np.floor((np.asarray(x) - x0) / res).astype(int))
obst = np.zeros((H, W), bool); floor = np.zeros((H, W), bool)
r, c = ij(P[:, 0], P[:, 1]); z = P[:, 2]
s = (z > a.zmin) & (z < a.zmax); obst[r[s], c[s]] = True
s = (np.abs(z) < 0.10) & (up > 0.7); floor[r[s], c[s]] = True
floor = ndimage.binary_closing(floor, iterations=3)
trav = floor & ~obst
clear = ndimage.distance_transform_edt(trav) * res
print(f"grid {W}x{H} @ {res} m; floor {floor.mean():.1%}, obstacle {obst.mean():.1%}, traversable {trav.mean():.1%}")

# graph on cells with enough clearance
ok = clear >= a.clear; idx = -np.ones((H, W), int); idx[ok] = np.arange(ok.sum()); rr, cc = np.nonzero(ok)
cost_cell = 1.0 / np.minimum(clear[ok], 1.2) ** 2
rows, cols, wts = [], [], []
for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
    r2, c2 = rr + dr, cc + dc; v = (r2 >= 0) & (r2 < H) & (c2 >= 0) & (c2 < W); v[v] &= ok[r2[v], c2[v]]
    a_, b_ = idx[rr[v], cc[v]], idx[r2[v], c2[v]]
    w = res * np.hypot(dr, dc) * 0.5 * (cost_cell[a_] + cost_cell[b_])
    rows += [a_, b_]; cols += [b_, a_]; wts += [w, w]
G = coo_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))), shape=(ok.sum(),) * 2).tocsr()

wp = [tuple(map(float, p.split(","))) for p in a.waypoints.split(";") if p.strip()]
if a.closed: wp.append(wp[0])
def snap(x, y):                                       # the best-centred cell within 1 m
    r0, c0 = ij(x, y); R = int(1.0 / res)
    win = np.where(ok[max(r0 - R, 0):r0 + R, max(c0 - R, 0):c0 + R], clear[max(r0 - R, 0):r0 + R, max(c0 - R, 0):c0 + R], -1)
    rw, cw = np.unravel_index(np.argmax(win), win.shape); assert win[rw, cw] > 0, f"no free cell near {x, y}"
    return idx[max(r0 - R, 0) + rw, max(c0 - R, 0) + cw]
nodes = [snap(*p) for p in wp]
path = []
for s_, t_ in zip(nodes[:-1], nodes[1:]):
    _, pred = dijkstra(G, indices=s_, return_predecessors=True)
    assert pred[t_] >= 0 or s_ == t_, "waypoints not connected"
    seg = [t_]
    while seg[-1] != s_: seg.append(pred[seg[-1]])
    path += seg[::-1][(1 if path else 0):]
xy = np.stack([x0 + (cc[path] + 0.5) * res, y0 + (rr[path] + 0.5) * res], 1)
if a.closed: xy = xy[:-1]

def resample(p, step, closed):
    q = np.vstack([p, p[:1]]) if closed else p
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]; s = np.arange(0, d[-1] - (step / 2 if closed else 0), step)
    return np.stack([np.interp(s, d, q[:, 0]), np.interp(s, d, q[:, 1])], 1)
xy = resample(xy, res, a.closed)
xy = ndimage.gaussian_filter1d(xy, a.smooth / res, axis=0, mode="wrap" if a.closed else "nearest")
xy = resample(xy, res, a.closed)
r, c = ij(xy[:, 0], xy[:, 1]); cl = clear[r, c]
seglen = np.linalg.norm(np.diff(np.vstack([xy, xy[:1]]) if a.closed else xy, axis=0), axis=1)
yaw0 = float(np.arctan2(*(xy[3] - xy[0])[::-1]))
out = {"points": xy.round(4).tolist(), "clearance": cl.round(3).tolist(), "closed": a.closed, "length_m": float(seglen.sum()), "min_clearance_m": float(cl.min()),
       "start": [float(xy[0, 0]), float(xy[0, 1]), yaw0], "waypoints": wp, "mesh": str(Path(a.mesh).resolve()),
       "params": {k: getattr(a, k) for k in ("res", "zmin", "zmax", "clear", "smooth")}}
json.dump(out, open(a.out, "w"))
# for a camera riding along (Isaac/scene_tour.py chase view): distance to anything from 0.12 m up to 2 m
obst_cam = np.zeros((H, W), bool); r, c = ij(P[:, 0], P[:, 1]); s = (z > a.zmin) & (z < 2.0); obst_cam[r[s], c[s]] = True
clear_cam = ndimage.distance_transform_edt(~obst_cam) * res
np.savez_compressed(Path(a.out).with_suffix(".grid.npz"), clear=clear.astype(np.float32), clear_cam=clear_cam.astype(np.float32),
                    x0=x0, y0=y0, res=res)
print(f"route: {len(xy)} points, {out['length_m']:.1f} m, clearance min {cl.min():.2f} m / median {np.median(cl):.2f} m, "
      f"start ({xy[0, 0]:.2f}, {xy[0, 1]:.2f}) yaw {np.degrees(yaw0):.0f} deg -> {a.out}")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 10)); ext = [x0, x0 + W * res, y0, y0 + H * res]
ax.imshow(np.where(trav, np.minimum(clear, 1.5), np.nan), origin="lower", extent=ext, cmap="Greens", vmin=0, vmax=2.0)
ax.imshow(np.where(obst, 1.0, np.nan), origin="lower", extent=ext, cmap="gray_r", vmin=0, vmax=1)
if a.poses:
    cp = np.load(a.poses)["c2w"][:, :3, 3]; ax.plot(cp[:, 0], cp[:, 1], color="0.6", lw=0.5, label="phone path")
ax.plot(xy[:, 0], xy[:, 1], "r-", lw=1.5, label=f"robot route {out['length_m']:.0f} m")
ax.plot(*xy[0], "bo"); ax.annotate("start", xy[0], color="b")
ax.arrow(xy[0, 0], xy[0, 1], 0.8 * np.cos(yaw0), 0.8 * np.sin(yaw0), color="b", width=0.08)
ax.set_xlim(xy[:, 0].min() - 3, xy[:, 0].max() + 3); ax.set_ylim(xy[:, 1].min() - 3, xy[:, 1].max() + 3)
ax.set_aspect("equal"); ax.legend(loc="upper left"); ax.set_title(f"obstacles {a.zmin}-{a.zmax} m (black), clearance (green), route (red)")
fig.savefig(Path(a.out).with_suffix(".png"), dpi=110, bbox_inches="tight"); print("plot", Path(a.out).with_suffix(".png"))
