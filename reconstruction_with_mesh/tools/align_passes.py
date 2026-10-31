"""Bring the passes of a scene into one frame, the first pass's: each Record3D recording has its own ARKit world.

    python align_passes.py <scene dir> pairs [--per-pass 150] [--step-m 0.75] [--step-deg 20]
    python align_passes.py <scene dir> solve [--min-inliers 30] [--min-votes 5]

Height and tilt are shared already (prep: gravity from ARKit, floor at z = 0 in every pass), so per pass what is
left is a yaw-plane transform: a turn about the vertical and a shift along the floor. Where the phone's ARKit world
starts is not where the recording starts (on 22 Oct the first frame was 11 m from ARKit's origin), so starting
every pass at the same spot does not align them; image matches do.
  pairs: from each pass, its kept images ~every --step-m or --step-deg (at most --per-pass), each paired with every
         such image of the other passes -> colmap/pairs_align.txt (COLMAP matches these before the pose-guided pairs);
  solve: for every verified cross-pass pair, the relative pose from LiDAR-lifted PnP both ways (as in loop_close.py)
         is a vote for the transform between the two passes (one way suffices here: the consensus filters); per pair of passes the consensus (votes that put the
         moved pass within 30 cm and 2 deg of each other), then from pass 1 outwards along the best-supported links.
Writes record3d/poses_aligned.npz (poses.npz with every pass moved into pass 1's frame; with one pass, a copy),
record3d/pass_align.json and record3d/passes_aligned.png.
"""
import argparse, itertools, json, shutil
from pathlib import Path
import numpy as np
import r3dlib as L

ap = argparse.ArgumentParser(); ap.add_argument("scene"); ap.add_argument("step", choices=["pairs", "solve"])
ap.add_argument("--per-pass", type=int, default=150); ap.add_argument("--step-m", type=float, default=0.75)
ap.add_argument("--step-deg", type=float, default=20.0); ap.add_argument("--min-inliers", type=int, default=30)
ap.add_argument("--min-votes", type=int, default=5)
a = ap.parse_args(); scene = Path(a.scene); rec = scene / "record3d"
z = dict(np.load(rec / "poses.npz")); c2w = z["c2w"]; pl = L.Passes(z)
keep = np.load(rec / "kept_frames.npy")
names = {L.frame_of_name(p.name): p.name for p in (scene / "images").glob("*.jpg")}
pass_of_kept = pl.pass_of[keep]

if a.step == "pairs":
    sub = []
    for p in range(pl.n):
        kp = keep[pass_of_kept == p]
        pick = [kp[lo] for lo, _ in L.motion_slots(c2w[kp], a.step_m, a.step_deg)]
        if len(pick) > a.per_pass:
            pick = [pick[int(k)] for k in np.linspace(0, len(pick) - 1, a.per_pass).round()]
        sub.append(pick)
        print(f"pass {p + 1}: {len(pick)} of its {len(kp)} images take part in the alignment", flush=True)
    n = 0
    with open(scene / "colmap" / "pairs_align.txt", "w") as fh:
        for p, q in itertools.combinations(range(pl.n), 2):
            for i in sub[p]:
                for j in sub[q]:
                    fh.write(f"{names[i]} {names[j]}\n"); n += 1
    print(f"{n} cross-pass pairs -> colmap/pairs_align.txt")
    raise SystemExit(0)

# ---------------- solve
if pl.n == 1:
    shutil.copy(rec / "poses.npz", rec / "poses_aligned.npz")
    json.dump({"passes": 1}, open(rec / "pass_align.json", "w"))
    print("one pass: nothing to align"); raise SystemExit(0)
mdb = L.MatchDB(scene / "colmap" / "database.db")
s = float(z["dw"]) / float(z["w"])
fr = lambda n: L.frame_of_name(n)
votes, why = L.alignment_votes(mdb, lambda n: int(pl.pass_of[fr(n)]), lambda n: c2w[fr(n)], lambda n: pl.depth(fr(n)),
                               lambda n: z["K"][fr(n)], s, a.min_inliers, log_every=500)
print(f"cross-pass pairs with >= {a.min_inliers} verified matches: {why['tried']}; votes {sum(len(v) for v in votes.values())} "
      f"(rejected: {why['depth']} too little LiDAR depth, {why['pnp']} PnP failed, {why['disagree']} the two ways disagree)")
links = {}
for (p, q), vs in sorted(votes.items()):
    G = np.stack([v[0] for v in vs])
    ref = c2w[keep[pass_of_kept == q], :3, 3].mean(0)          # the middle of the pass being moved
    T, inl, st = L.consensus_yaw_transform(G, ref)
    st.update(pair=[p + 1, q + 1])
    if T is not None:
        st.update(yaw_deg=float(np.degrees(np.arctan2(T[1, 0], T[0, 0]))), shift_m=[float(T[0, 3]), float(T[1, 3])])
        links[(p, q)] = (T, st)
    print(f"passes {p + 1} <- {q + 1}: {st['votes']} votes, {st.get('sane', 0)} level, {st.get('inliers', 0)} agree"
          + (f" (spread {st['spread_cm_median']:.1f} cm, {st['yaw_spread_deg']:.2f} deg): turn {st['yaw_deg']:+.1f} deg, "
             f"shift ({st['shift_m'][0]:+.2f}, {st['shift_m'][1]:+.2f}) m" if T is not None else ""), flush=True)
# from pass 1 outwards, always along the link with the most agreeing votes
T_of = {0: np.eye(4)}; used = []
while len(T_of) < pl.n:
    best = None
    for (p, q), (T, st) in links.items():
        if st["inliers"] < a.min_votes or (p in T_of) == (q in T_of):
            continue
        if best is None or st["inliers"] > best[2]["inliers"]:
            best = (p, q, st, T)
    if best is None:
        lost = [k + 1 for k in range(pl.n) if k not in T_of]
        raise SystemExit(f"pass(es) {lost} share no verified views with the others (need >= {a.min_votes} agreeing "
                         "votes). Were they recorded in the same place, with overlap? Check record3d/pass_align.json.")
    p, q, st, T = best
    if p in T_of:
        T_of[q] = T_of[p] @ T                       # T maps q's frame onto p's
    else:
        T_of[p] = T_of[q] @ np.linalg.inv(T)
    used.append(st["pair"])
out = c2w.copy()
for p, lo, hi in pl.ranges():
    out[lo:hi] = T_of[p] @ c2w[lo:hi]
z2 = dict(z); z2["c2w"] = out; z2["pass_T"] = np.stack([T_of[p] for p in range(pl.n)])
np.savez(rec / "poses_aligned.npz", **z2)
json.dump({"passes": pl.n, "tried": why, "links": [st for _, st in links.values()], "used": used,
           "transforms": {str(p + 1): {"yaw_deg": float(np.degrees(np.arctan2(T[1, 0], T[0, 0]))),
                                       "shift_m": [float(T[0, 3]), float(T[1, 3])]} for p, T in T_of.items()}},
          open(rec / "pass_align.json", "w"), indent=1)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    for k, (C, ttl) in enumerate([(c2w, "each pass in its own ARKit world"), (out, "aligned to pass 1")]):
        for p, lo, hi in pl.ranges():
            ax[k].plot(C[lo:hi:10, 0, 3], C[lo:hi:10, 1, 3], ".", ms=1, label=f"pass {p + 1}")
        ax[k].set_aspect("equal"); ax[k].set_title(ttl); ax[k].legend(markerscale=10)
    plt.tight_layout(); plt.savefig(rec / "passes_aligned.png", dpi=80)
except Exception as ex:
    print("plot failed:", ex)
print("wrote record3d/poses_aligned.npz")
