"""Patch small holes in the collider's floor: gaps the LiDAR left in a floor it saw all around them.

    python fill_floor.py <collider.ply> <out.ply> [--res 0.05] [--max-area 2.0] [--report holes.json] [--dry-run]

Shiny or dark floor (linoleum under ceiling lights) returns no LiDAR at some angles, and TSDF fusion leaves a hole:
on the 22 Oct walk a 35 x 35 cm one beside the path swallowed the Isaac check's ball, and would catch a wheel.
The floor is at z = 0 by construction (prep pins every frame's floor), so a hole can be closed with flat patches:
  1. floor cells (5 cm) = upward-facing mesh within 6 cm of z = 0;
  2. holes = empty regions completely enclosed by floor cells (scipy binary_fill_holes); a room the phone never
     entered is not enclosed by floor (its walls are not floor), so it stays open;
  3. holes larger than --max-area m^2 stay open (e.g. the block a ring corridor goes round: enclosed by floor, but
     no hole), and so do holes over --min-drop-area m^2 with geometry seen below -10 cm (stairs, a real drop);
     below-floor geometry under a smaller hole is fusion noise;
  4. every hole cell gets two triangles at the median height of the floor around that hole.
"""
import argparse, json
import numpy as np, trimesh
from scipy import ndimage

ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("out")
ap.add_argument("--res", type=float, default=0.05); ap.add_argument("--max-area", type=float, default=2.0)
ap.add_argument("--min-drop-area", type=float, default=0.25)
ap.add_argument("--report"); ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args(); res = a.res
m = trimesh.load(a.src, process=False, force="mesh"); V = np.asarray(m.vertices, float); F = np.asarray(m.faces)
T = V[F]; nz = m.face_normals[:, 2]
L = np.max(np.linalg.norm(T - np.roll(T, 1, axis=1), axis=2), axis=1); n = np.clip(np.ceil(L / (res / 2)), 1, 100).astype(int)
pts, up = [], []
for k in np.unique(n):                                    # barycentric samples, at most res/2 apart
    sel = np.where(n == k)[0]; i, j = np.meshgrid(np.arange(k + 1), np.arange(k + 1)); ok = i + j <= k
    w = np.stack([i[ok], j[ok], k - i[ok] - j[ok]], 1) / k
    pts.append(np.einsum("sb,fbc->fsc", w, T[sel]).reshape(-1, 3)); up.append(np.repeat(nz[sel], len(w)))
P = np.concatenate(pts); up = np.concatenate(up)
x0, y0 = P[:, 0].min() - 1, P[:, 1].min() - 1
W, H = int((P[:, 0].max() + 1 - x0) / res) + 1, int((P[:, 1].max() + 1 - y0) / res) + 1
r = np.floor((P[:, 1] - y0) / res).astype(int); c = np.floor((P[:, 0] - x0) / res).astype(int)
fl = (np.abs(P[:, 2]) < 0.06) & (up > 0.7)
floor = np.zeros((H, W), bool); floor[r[fl], c[fl]] = True
zsum = np.zeros((H, W)); zcnt = np.zeros((H, W)); np.add.at(zsum, (r[fl], c[fl]), P[fl, 2]); np.add.at(zcnt, (r[fl], c[fl]), 1)
below = np.zeros((H, W), bool); b = P[:, 2] < -0.10; below[r[b], c[b]] = True
holes = ndimage.binary_fill_holes(floor) & ~floor
lab, nlab = ndimage.label(holes)
area = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, nlab + 1)) * res * res
deep = ndimage.maximum(below, lab, index=np.arange(1, nlab + 1)) if nlab else np.zeros(0)
fill = [k + 1 for k in range(nlab) if area[k] <= a.max_area and not (deep[k] and area[k] > a.min_drop_area)]
quads, info = [], []
for k in fill:
    cells = np.argwhere(lab == k)
    ring = ndimage.binary_dilation(lab == k, iterations=2) & floor
    zf = float(np.median(zsum[ring] / zcnt[ring])) if ring.any() else 0.0
    for rr, cc in cells:
        xa, ya = x0 + cc * res, y0 + rr * res
        quads.append([[xa, ya, zf], [xa + res, ya, zf], [xa + res, ya + res, zf], [xa, ya + res, zf]])
    cy, cx = cells.mean(0)
    info.append({"x": round(float(x0 + (cx + 0.5) * res), 2), "y": round(float(y0 + (cy + 0.5) * res), 2),
                 "area_m2": round(float(area[k - 1]), 3), "z": round(zf, 3)})
kept = [{"area_m2": round(float(area[k]), 2), "reason": "too large" if area[k] > a.max_area else "below floor"}
        for k in range(nlab) if k + 1 not in fill]
big = sorted(info, key=lambda h: -h["area_m2"])
print(f"floor holes: {nlab} enclosed by floor; patched {len(fill)} ({sum(h['area_m2'] for h in info):.2f} m^2, largest "
      f"{big[0]['area_m2'] if big else 0:.2f} m^2 at ({big[0]['x'] if big else 0}, {big[0]['y'] if big else 0})); "
      f"left open {len(kept)} ({sum(1 for h in kept if h['reason'] == 'too large')} larger than {a.max_area} m^2, "
      f"{sum(1 for h in kept if h['reason'] == 'below floor')} with geometry below the floor)")
if a.report:
    json.dump({"patched": info, "left_open": kept, "patched_area_m2": round(sum(h["area_m2"] for h in info), 3)},
              open(a.report, "w"), indent=1)
if not a.dry_run:
    if quads:
        Q = np.asarray(quads).reshape(-1, 3); base = len(V) + 4 * np.arange(len(quads))[:, None]
        tri = np.vstack([np.c_[base, base + 1, base + 2], np.c_[base, base + 2, base + 3]])
        m = trimesh.Trimesh(np.vstack([V, Q]), np.vstack([F, tri]), process=False)
    m.export(a.out)
    print(f"wrote {a.out}: {len(m.faces):,} triangles")
