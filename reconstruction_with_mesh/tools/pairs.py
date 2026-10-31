"""COLMAP pair list from the phone poses: who should be matched with whom (instead of all pairs).

    python pairs.py <scene dir> [--neighbours 8] [--radius 4.0] [--angle 75] [--near 30] [--far 30] [--far-gap-s 10]

Each kept image is matched with its next --neighbours images in time (same pass), and with images whose camera is
within --radius m and looks within --angle deg of the same direction (record3d/poses_aligned.npz: all passes in one
frame). Where that is more than near + far partners, it keeps the --near closest and the --far closest that are more
than --far-gap-s apart in time or in another pass: those are the loop closures. 3,050 images of a two-lap walk gave
152k pairs (exhaustive would be 4.6M). Pairs already in colmap/pairs_align.txt (matched for the pass alignment) are
left out. Writes <scene>/colmap/pairs.txt.
"""
import argparse
from pathlib import Path
import numpy as np
import r3dlib as L

ap = argparse.ArgumentParser(); ap.add_argument("scene")
ap.add_argument("--neighbours", type=int, default=8); ap.add_argument("--radius", type=float, default=4.0)
ap.add_argument("--angle", type=float, default=75.0); ap.add_argument("--near", type=int, default=30)
ap.add_argument("--far", type=int, default=30); ap.add_argument("--far-gap-s", type=float, default=10.0)
a = ap.parse_args(); scene = Path(a.scene); rec = scene / "record3d"
z = np.load(rec / "poses_aligned.npz" if (rec / "poses_aligned.npz").exists() else rec / "poses.npz")
keep = np.load(rec / "kept_frames.npy"); pl = L.Passes(z)
names = {L.frame_of_name(p.name): p.name for p in (scene / "images").glob("*.jpg")}
C = z["c2w"][keep].astype(np.float32); c = C[:, :3, 3]; f = C[:, :3, 2]; t = z["ts"][keep]; pk = pl.pass_of[keep]
n = len(keep); pairs = set()
D = np.linalg.norm(c[:, None] - c[None], axis=2); A = f @ f.T
ii, jj = np.where(np.triu((D < a.radius) & (A > np.cos(np.radians(a.angle))), 1))
for i in range(n):
    for j in range(i + 1, min(n, i + a.neighbours + 1)):
        if pk[j] == pk[i]:
            pairs.add((i, j))
cand = {}
for i, j in zip(ii, jj):
    if j - i > a.neighbours or pk[j] != pk[i]:
        cand.setdefault(i, []).append(j)
far_of = lambda i, js: (np.abs(t[js] - t[i]) > a.far_gap_s) | (pk[js] != pk[i])
for i, js in cand.items():
    js = np.array(js)
    if len(js) > a.near + a.far:
        near = js[np.argsort(D[i, js])[:a.near]]
        far = js[far_of(i, js)]; far = far[np.argsort(D[i, far])][:a.far]
        js = np.unique(np.r_[near, far])
    for j in js:
        pairs.add((i, int(j)))
done = set()
pa = scene / "colmap" / "pairs_align.txt"
if pa.exists():
    done = {tuple(sorted(ln.split())) for ln in pa.read_text().splitlines() if ln.strip()}
(scene / "colmap").mkdir(exist_ok=True)
written = far = cross = 0
with open(scene / "colmap" / "pairs.txt", "w") as fh:
    for i, j in sorted(pairs):
        ni, nj = names[keep[i]], names[keep[j]]
        if tuple(sorted((ni, nj))) in done:
            continue
        fh.write(f"{ni} {nj}\n"); written += 1
        far += bool(far_of(i, np.array([j]))[0]); cross += pk[i] != pk[j]
print(f"{written} pairs for {n} images; {far} join frames > {a.far_gap_s:g} s apart or in different passes (loop "
      f"closure), {cross} of them across passes" + (f"; {len(pairs) - written} already matched for the alignment" if done else ""))
