"""Close the loops of a long Record3D walk (one pass or several): remove ARKit's relocalisation jumps, then fit a pose
graph to image matches.

    python loop_close.py <scene dir> [--min-gap-s 10] [--min-inliers 40]

Needs record3d/poses_aligned.npz (prep_record3d.py, then align_passes.py: floor-pinned, horizontally raw, all passes
in one frame; older scenes: poses.npz) and COLMAP features + matches for <scene>/images in <scene>/colmap/database.db.

1. De-snap, per pass: ARKit's per-frame motion (x, y, yaw in the Z-up world) is integrated again with the frames of
   every snap event replaced by the motion of the frame before, so the trajectory is continuous (and drifts
   smoothly). Height stays the per-frame LiDAR floor pin; roll and pitch stay ARKit's (gravity is good).
2. Loop edges: for image pairs more than --min-gap-s apart in time, or in different passes, with >= --min-inliers
   verified matches, the metric relative pose comes from PnP: keypoints of one image lifted with its own LiDAR
   depth, matched into the other (done both ways; pairs whose two answers disagree are dropped).
3. Pose graph over the kept images (x, y, yaw per image; ARKit odometry between neighbours of the same pass, loop
   edges with a Cauchy loss whose scale is lowered in steps 30 -> 1 sigma, starting from ARKit's own snapped poses;
   pass 1's first image is held, the other passes' first images weakly kept where the alignment put them). The
   correction is interpolated to all frames of each pass.
4. Several passes: LiDAR ties. Image matches tie passes recorded at different heights only weakly (the 0.5 m pass of
   27 Oct had 286 edges to the 1.4 m pass, against 24,843 between 1.4 m and 1.9 m), and between them a pass drifts
   with ARKit: its walls stood up to 55 cm off the other passes' after the graph. So every stretch of a pass (about
   2 m of LiDAR frames) is ICP-fitted onto each other pass's LiDAR cloud around it, and the fit becomes an edge of
   the graph between the stretch's middle image and the other pass's nearest image, weighted by what the walls fix
   (a straight corridor fixes the sideways position, not the one along it). Solved again, in --lidar-rounds rounds.
Writes record3d/poses_lc.npz (all frames), sparse/0/images.txt (kept images, new poses) and record3d/loop_close.png.
"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation
import r3dlib as L
from r3dlib import rz, wrap, yaw_of

ap = argparse.ArgumentParser(); ap.add_argument("scene")
ap.add_argument("--min-gap-s", type=float, default=10.0); ap.add_argument("--min-inliers", type=int, default=40)
ap.add_argument("--sig-odo-t", type=float, default=0.005); ap.add_argument("--sig-odo-deg", type=float, default=0.02)
ap.add_argument("--sig-loop-t", type=float, default=0.03); ap.add_argument("--sig-loop-deg", type=float, default=0.5)
ap.add_argument("--sig-pass-t", type=float, default=2.0); ap.add_argument("--sig-pass-deg", type=float, default=20.0)
ap.add_argument("--lidar-rounds", type=int, default=3)
ap.add_argument("--sig-lidar-t", type=float, default=0.03); ap.add_argument("--sig-lidar-deg", type=float, default=0.5)
a = ap.parse_args(); scene = Path(a.scene); rec = scene / "record3d"
z = dict(np.load(rec / "poses_aligned.npz" if (rec / "poses_aligned.npz").exists() else rec / "poses.npz"))
c2w = z["c2w"].copy(); K = z["K"]; ts = z["ts"]; N = len(c2w); pl = L.Passes(z)
s = float(z["dw"]) / float(z["w"])
keep = np.load(rec / "kept_frames.npy")

# ---------------- 1. de-snap, each pass on its own
c2w_ds = c2w.copy(); n_bad = 0
for p, lo, hi in pl.ranges():
    C = c2w[lo:hi]; n = hi - lo
    if n < 3:
        continue
    pp = C[:, :2, 3].copy(); yaw = yaw_of(C[:, :3, :3])
    dp = np.diff(pp, axis=0); dyaw = wrap(np.diff(yaw))
    loc = np.einsum("nij,nj->ni", rz(-yaw[:-1])[:, :2, :2], dp)        # motion in the previous frame's yaw frame
    step = np.linalg.norm(dp, axis=1)
    bad = np.zeros(n - 1, bool)
    for sn in z["snaps"][(z["snaps"] > lo) & (z["snaps"] < hi)] - lo:
        k = sn - 1                                                     # step index of the first jump
        before = step[max(0, k - 120):max(0, k - 5)]
        lim = max(0.012, 2.5 * np.median(before)) if len(before) else 0.012
        e = k
        while e < n - 1 and step[e] > lim:
            e += 1
        bad[k:max(e, k + 1)] = True
    for k in np.where(bad)[0]:
        j = k - 1
        while j > 0 and bad[j]:
            j -= 1
        loc[k] = loc[max(j, 0)]; dyaw[k] = dyaw[max(j, 0)]
    yaw2 = np.r_[yaw[0], yaw[0] + np.cumsum(dyaw)]
    p2 = np.vstack([pp[0], pp[0] + np.cumsum(np.einsum("nij,nj->ni", rz(yaw2[:-1])[:, :2, :2], loc), axis=0)])
    c2w_ds[lo:hi, :3, :3] = rz(yaw2 - yaw) @ C[:, :3, :3]; c2w_ds[lo:hi, :2, 3] = p2
    n_bad += int(bad.sum())
print(f"de-snap: replaced {n_bad} frame steps at {len(z['snaps'])} snap events", flush=True)

# ---------------- 2. loop edges from verified matches + LiDAR depth
node_of = {int(f): k for k, f in enumerate(keep)}
loops, n_try, n_disagree = [], 0, 0
cache = rec / f"loop_edges_gap{a.min_gap_s:g}_in{a.min_inliers}.npz"
if cache.exists():
    c = np.load(cache); loops = [(int(i), int(j), M, int(q)) for i, j, M, q in zip(c["i"], c["j"], c["M"], c["n"])]
    print(f"loop edges from {cache.name}: {len(loops)}")
else:
    mdb = L.MatchDB(scene / "colmap" / "database.db")
    for i1, i2, m in mdb.verified(a.min_inliers):
        fa, fb = L.frame_of_name(mdb.name_of[i1]), L.frame_of_name(mdb.name_of[i2])
        if fa not in node_of or fb not in node_of:
            continue
        if pl.pass_of[fa] == pl.pass_of[fb] and abs(ts[fa] - ts[fb]) < a.min_gap_s:
            continue
        n_try += 1
        M, info = L.relative_pose_both_ways(pl.depth(fa), K[fa], pl.depth(fb), K[fb], s,
                                            mdb.keypoints(i1)[m[:, 0]], mdb.keypoints(i2)[m[:, 1]])
        if M is None:
            n_disagree += info == "disagree"
            continue
        loops.append((node_of[fa], node_of[fb], M, info))
    print(f"loop candidates (gap > {a.min_gap_s:.0f} s or another pass, >= {a.min_inliers} inliers): {n_try}; accepted "
          f"{len(loops)}; rejected for disagreeing both ways: {n_disagree}", flush=True)
    np.savez(cache, i=np.array([l[0] for l in loops], int), j=np.array([l[1] for l in loops], int),
             M=np.stack([l[2] for l in loops]) if loops else np.zeros((0, 4, 4)), n=np.array([l[3] for l in loops], int))
MIN_LOOPS = 10
if len(loops) < MIN_LOOPS:   # nothing to close (no revisits, or too few verified edges): keep the de-snapped walk
    print(f"only {len(loops)} loop edges (< {MIN_LOOPS}): no pose graph; the de-snapped ARKit trajectory is used as is", flush=True)

# ---------------- 3. pose graph (x, y, yaw corrections per kept image, rotation about the camera centre)
n = len(keep); P0 = c2w_ds[keep]; pk = P0[:, :2, 3]; yk = yaw_of(P0[:, :3, :3]); pass_k = pl.pass_of[keep]
oi = np.where(pass_k[1:] == pass_k[:-1])[0]; oj = oi + 1                     # odometry: neighbours in one pass
first = np.array([int(np.argmax(pass_k == p)) for p in range(pl.n) if (pass_k == p).any()])[1:]   # other passes
dpo = pk[oj] - pk[oi]
st, sr = a.sig_odo_t, np.radians(a.sig_odo_deg); lt, lr = a.sig_loop_t, np.radians(a.sig_loop_deg)
li = np.array([l[0] for l in loops], int); lj = np.array([l[1] for l in loops], int)
Mt = np.stack([l[2][:3, 3] for l in loops]) if loops else np.zeros((0, 3))
MR = np.stack([l[2][:3, :3] for l in loops]) if loops else np.zeros((0, 3, 3))
prior_sig = np.array([a.sig_pass_t, a.sig_pass_t, np.radians(a.sig_pass_deg)])
# LiDAR ties between passes (step 4): node pairs, measured relative pose, 2x2 position weight (world frame, / sigma)
ti = tj = np.zeros(0, int); TMt = np.zeros((0, 3)); TMR = np.zeros((0, 3, 3)); TW = np.zeros((0, 2, 2))


FS = None   # Cauchy scale (sigmas) for the loop edges; None = plain residuals


def cauchy(r, f):
    """r -> r~ with r~^2 = f^2 log(1 + (r/f)^2): Cauchy's loss on these residuals only, with a linear solver loss."""
    return np.sign(r) * f * np.sqrt(np.log1p((r / f) ** 2))


def residuals(x, fs="auto"):
    d = x.reshape(n, 3); dxy, th = d[:, :2], d[:, 2]
    ro_t = np.einsum("nij,nj->ni", rz(-th[oi])[:, :2, :2], dpo + dxy[oj] - dxy[oi]) - dpo
    ro_r = wrap(th[oj] - th[oi])               # wrapped: a whole turn is no rotation (else a loss-saturated edge lets
                                               # a stretch sit whole turns off, and the frames between spin)
    Ri = rz(th[li]) @ P0[li, :3, :3]
    pj_meas = (P0[li, :2, 3] + dxy[li]) + np.einsum("nij,nj->ni", Ri, Mt)[:, :2]
    rl_t = (pk[lj] + dxy[lj]) - pj_meas
    rl_r = wrap((yk[lj] + th[lj]) - yaw_of(Ri @ MR))
    rl = np.concatenate([(rl_t / lt).ravel(), rl_r / lr])
    Ti = rz(th[ti]) @ P0[ti, :3, :3]
    rt_t = np.einsum("nij,nj->ni", TW, (pk[tj] + dxy[tj]) - ((P0[ti, :2, 3] + dxy[ti]) + np.einsum("nij,nj->ni", Ti, TMt)[:, :2]))
    rt = np.concatenate([rt_t.ravel(), wrap((yk[tj] + th[tj]) - yaw_of(Ti @ TMR)) / np.radians(a.sig_lidar_deg)])
    fs = FS if fs == "auto" else fs
    if fs is not None:   # robust on the loop edges only: odometry within a pass is never broken, or a few look-alike
        rl = cauchy(rl, fs)  # matches (identical desks, doors) pull single images 20-50 cm off their own track
        rt = cauchy(rt, fs)
    return np.concatenate([(ro_t / st).ravel(), ro_r / sr, rl, x[:3] * 1e3, (d[first] / prior_sig).ravel(), rt])


def sparsity():
    no, nl = len(oi), len(loops)
    S = lil_matrix((3 * no + 3 * nl + 3 + 3 * len(first) + 3 * len(ti), 3 * n), dtype=int); r = 0
    for q, (i, j) in enumerate(zip(oi, oj)):
        for c in range(2): S[r + 2 * q + c, [3 * i, 3 * i + 1, 3 * i + 2, 3 * j, 3 * j + 1]] = 1
    r += 2 * no
    for q, (i, j) in enumerate(zip(oi, oj)): S[r + q, [3 * i + 2, 3 * j + 2]] = 1
    r += no
    for q, (i, j) in enumerate(zip(li, lj)):
        for c in range(2): S[r + 2 * q + c, [3 * i, 3 * i + 1, 3 * i + 2, 3 * j, 3 * j + 1]] = 1
    r += 2 * nl
    for q, (i, j) in enumerate(zip(li, lj)): S[r + q, [3 * i + 2, 3 * j + 2]] = 1
    r += nl; S[r, 0] = S[r + 1, 1] = S[r + 2, 2] = 1; r += 3
    for q, f in enumerate(first):
        for c in range(3): S[r + 3 * q + c, 3 * f + c] = 1
    r += 3 * len(first)
    for q, (i, j) in enumerate(zip(ti, tj)):
        for c in range(2): S[r + 2 * q + c, [3 * i, 3 * i + 1, 3 * i + 2, 3 * j, 3 * j + 1]] = 1
    r += 2 * len(ti)
    for q, (i, j) in enumerate(zip(ti, tj)): S[r + q, [3 * i + 2, 3 * j + 2]] = 1
    return S


def loop_err(x):
    r = residuals(x, None); o = 3 * len(oi); e = r[o:o + 2 * len(loops)].reshape(-1, 2) * lt
    return np.linalg.norm(e, axis=1), np.degrees(np.abs(r[o + 2 * len(loops):o + 3 * len(loops)]) * lr)


def solve(x, scales=(30.0, 10.0, 3.0, 1.0)):
    """Graduated: near-L2 first, so large true errors are not mistaken for outliers."""
    global FS
    S_ = sparsity()
    for fs in scales:
        FS = fs
        x = least_squares(residuals, x, jac_sparsity=S_, loss="linear", method="trf", x_scale="jac", max_nfev=60).x
        e_, _ = loop_err(x)
        print(f"  cauchy scale {fs:4.0f}: loop error median {np.median(e_) * 100:.1f} cm, p90 {np.percentile(e_, 90) * 100:.1f} cm, "
              f"{(e_ < 3 * lt).mean() * 100:.0f}% within {3 * lt * 100:.0f} cm", flush=True)
    return x


def all_frames(d):
    """The kept images' corrections (x, y, yaw) interpolated to every frame of each pass."""
    out = c2w_ds.copy()
    for p, lo, hi in pl.ranges():
        sel = pass_k == p
        if not sel.any():
            continue
        kp, dd = keep[sel], d[sel].copy()
        dd[:, 2] = np.unwrap(dd[:, 2])            # neighbours < 180 deg apart: frames between turn the short way
        fi = np.clip(np.arange(lo, hi), kp[0], kp[-1])
        dx, dy, dt = (np.interp(fi, kp, dd[:, c]) for c in range(3))
        out[lo:hi, :3, :3] = rz(dt) @ c2w_ds[lo:hi, :3, :3]; out[lo:hi, 0, 3] += dx; out[lo:hi, 1, 3] += dy
    return out


# start from ARKit's own (snapped) horizontal poses: its relocalisations are rough loop closures already
Pr = c2w[keep]; x0 = np.c_[Pr[:, :2, 3] - pk, wrap(yaw_of(Pr[:, :3, :3]) - yk)].ravel(); x0[:3] = 0
GRAPH = len(loops) >= MIN_LOOPS
if GRAPH:
    e0, a0 = loop_err(x0); ez, _ = loop_err(np.zeros(3 * n))
    print(f"loop edges before: position error median {np.median(e0) * 100:.1f} cm, p90 {np.percentile(e0, 90) * 100:.1f} cm; "
          f"yaw median {np.median(a0):.2f} deg (de-snapped without ARKit's jumps: median {np.median(ez) * 100:.1f} cm)")
    x = solve(x0)
    e1, a1 = loop_err(x)
    inl = e1 < 3 * lt
    print(f"loop edges after:  position error median {np.median(e1) * 100:.1f} cm, p90 {np.percentile(e1, 90) * 100:.1f} cm; "
          f"yaw median {np.median(a1):.2f} deg; {inl.mean() * 100:.0f}% within {3 * lt * 100:.0f} cm")
else:
    x = np.zeros(3 * n)

# ---------------- 4. LiDAR ties between passes
lidar = {}
if pl.n > 1 and a.lidar_rounds > 0 and (rec / "depth.npy").exists():
    import open3d as o3d
    dfr = np.load(rec / "depth_frames.npy"); D = np.load(rec / "depth.npy", mmap_mode="r")
    sel_d = np.arange(0, len(dfr), 2); fr_d = dfr[sel_d]; pass_d = pl.pass_of[fr_d]
    hh, ww = D.shape[1:]; gv, gu = np.mgrid[2:hh:6, 2:ww:6]
    cam_pts = []                                  # LiDAR points in each (used) depth frame's camera, 0.3-3.5 m
    for k, g in zip(sel_d, fr_d):
        dd = D[k][gv, gu].astype(np.float32); ok = np.isfinite(dd) & (dd > 0.3) & (dd < 3.5)
        fx, fy, cx, cy = K[g] * s
        cam_pts.append(np.c_[(gu[ok] + 0.5 - cx) / fx * dd[ok], (gv[ok] + 0.5 - cy) / fy * dd[ok], dd[ok]].astype(np.float32))
    CH = 60                                       # stretch: 60 used LiDAR frames (~1.8 m of walking, or a turn)
    kp_of_pass = {p: np.where(pass_k == p)[0] for p in range(pl.n)}
    reg = o3d.pipelines.registration

    def cloud(xyz, vox=0.04, normals=False):
        c = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz)).voxel_down_sample(vox)
        if normals:
            c.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.12, max_nn=30))
        return c

    def ties(x):
        """ICP every stretch of every pass onto every other pass; returns the tie edges and the offsets found."""
        C = all_frames(x.reshape(n, 3)); W = [P @ C[g, :3, :3].T.astype(np.float32) + C[g, :3, 3].astype(np.float32)
                                              for P, g in zip(cam_pts, fr_d)]
        full = {}
        for p in range(pl.n):
            c = cloud(np.concatenate([W[i] for i in np.where(pass_d == p)[0]]), normals=True)
            full[p] = (np.asarray(c.points), np.asarray(c.normals))
        E, offs = [], {p: [] for p in range(pl.n)}
        for p in range(pl.n):
            idx = np.where(pass_d == p)[0]
            for c0 in range(0, len(idx) - CH // 3, CH):
                ch = idx[c0:c0 + CH]
                src = cloud(np.concatenate([W[i] for i in ch])); sp = np.asarray(src.points)
                lo_, hi_ = sp.min(0) - 0.6, sp.max(0) + 0.6
                gmid = int(fr_d[ch[len(ch) // 2]])
                na = kp_of_pass[p][np.argmin(np.abs(keep[kp_of_pass[p]] - gmid))]
                for q in range(pl.n):
                    if q == p:
                        continue
                    tp, tn = full[q]; m = np.all((tp >= lo_) & (tp <= hi_), 1)
                    if m.sum() < 1500:
                        continue
                    tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tp[m])); tgt.normals = o3d.utility.Vector3dVector(tn[m])
                    T = np.eye(4)
                    for dist in (0.4, 0.15, 0.05):
                        rr = reg.registration_icp(src, tgt, dist, T, reg.TransformationEstimationPointToPlane(),
                                                  reg.ICPConvergenceCriteria(max_iteration=40))
                        T = rr.transformation
                    ang = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))
                    if rr.fitness < 0.5 or rr.inlier_rmse > 0.03 or np.linalg.norm(T[:3, 3]) > 3.0 or ang > 10:
                        continue
                    cs = np.asarray(rr.correspondence_set)
                    nrm = tn[m][cs[:, 1]]; H = (nrm[:, :2].T @ nrm[:, :2]) / len(nrm)   # what the walls fix, in x-y
                    ev, V = np.linalg.eigh(H)
                    if ev[-1] < 0.05:                  # floor only: no horizontal hold
                        continue
                    Wt = V @ np.diag(np.minimum(1.0, np.sqrt(np.maximum(ev, 0) / 0.15))) @ V.T / a.sig_lidar_t
                    Ca = C[keep[na]]; pa = T[:3, :3] @ Ca[:3, 3] + T[:3, 3]
                    Ca2 = Ca.copy(); Ca2[:3, :3] = rz(np.arctan2(T[1, 0], T[0, 0])) @ Ca[:3, :3]; Ca2[:2, 3] = pa[:2]
                    cand = kp_of_pass[q]; nb = cand[np.argmin(np.linalg.norm(C[keep[cand], :2, 3] - pa[:2], axis=1))]
                    M = np.linalg.inv(Ca2) @ C[keep[nb]]
                    E.append((na, nb, M, Wt))
                    shift = T[:3, :3] @ sp.mean(0) + T[:3, 3] - sp.mean(0)     # across the walls: sliding along a
                    offs[p].append(float(np.linalg.norm(Wt @ shift[:2])) * a.sig_lidar_t)   # corridor is no offset
        return E, offs

    def summary(offs):
        return {f"pass{p + 1}": {"stretches": len(v), "median_cm": round(float(np.median(v)) * 100, 1),
                                 "p90_cm": round(float(np.percentile(v, 90)) * 100, 1),
                                 "max_cm": round(float(np.max(v)) * 100, 1)} for p, v in offs.items() if v}

    for rnd in range(a.lidar_rounds + 1):        # measure, solve with the ties, measure again ...; the last only measures
        E, offs = ties(x)
        allo = np.concatenate([v for v in offs.values() if v]) if any(offs.values()) else np.zeros(0)
        lidar["before" if rnd == 0 else f"after_round{rnd}"] = summary(offs)
        print(f"LiDAR ties, {'before' if rnd == 0 else f'after round {rnd}'}: {len(E)} stretch fits; stretches off the "
              "other passes by " + ", ".join(f"pass {p + 1} median {np.median(v) * 100:.1f} cm (p90 "
              f"{np.percentile(v, 90) * 100:.1f}, max {np.max(v) * 100:.1f})" for p, v in offs.items() if v), flush=True)
        if not E or rnd == a.lidar_rounds or (np.median(allo) < 0.02 and np.percentile(allo, 90) < 0.04):
            break
        ti = np.array([e[0] for e in E]); tj = np.array([e[1] for e in E]); TMt = np.stack([e[2][:3, 3] for e in E])
        TMR = np.stack([e[2][:3, :3] for e in E]); TW = np.stack([e[3] for e in E])
        x = solve(x)                              # graduated again: a stretch 40 cm off is no outlier
    if len(ti):
        e1, a1 = loop_err(x)
        print(f"loop edges with the LiDAR ties: position error median {np.median(e1) * 100:.1f} cm, p90 "
              f"{np.percentile(e1, 90) * 100:.1f} cm", flush=True)
d = x.reshape(n, 3).copy(); d[:, 2] = wrap(d[:, 2])
print(f"correction applied to kept images: |dxy| median {np.median(np.linalg.norm(d[:, :2], axis=1)) * 100:.1f} cm, "
      f"max {np.linalg.norm(d[:, :2], axis=1).max() * 100:.1f} cm; |dyaw| max {np.degrees(np.abs(d[:, 2]).max()):.2f} deg")
cross = {}
for i, j in zip(li, lj):
    pa, pb = sorted((int(pass_k[i]) + 1, int(pass_k[j]) + 1))
    cross[f"{pa}-{pb}"] = cross.get(f"{pa}-{pb}", 0) + 1
if pl.n > 1:
    print("loop edges by pass pair: " + ", ".join(f"{k}: {v:,}" for k, v in sorted(cross.items())))
    for p in range(1, pl.n):
        if not any(int(k.split("-")[0]) != int(k.split("-")[1]) and str(p + 1) in k.split("-") for k in cross):
            print(f"!! pass {p + 1} has no loop edge to another pass: it stays where the alignment put it")

# ---------------- interpolate to all frames of each pass, write
out = all_frames(d)
z2 = dict(z); z2["c2w"] = out; z2["c2w_desnap"] = c2w_ds; np.savez(rec / "poses_lc.npz", **z2)
sp = scene / "sparse" / "0"; lines = (sp / "images.txt").read_text().splitlines(); new = []
for ln in lines:
    if ln.startswith("#") or not ln.strip():
        new.append(ln); continue
    f = ln.split()
    fr = L.frame_of_name(f[9]); w2c = np.linalg.inv(out[fr]); qx, qy, qz, qw = Rotation.from_matrix(w2c[:3, :3]).as_quat()
    new.append(f"{f[0]} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {w2c[0, 3]:.6f} {w2c[1, 3]:.6f} {w2c[2, 3]:.6f} {f[8]} {f[9]}")
(sp / "images.txt").write_text("\n".join(new) + "\n")
json.dump({"loops_accepted": len(loops), "snap_events": int(len(z["snaps"])), "pose_graph": GRAPH,
           "loop_err_before_cm": float(np.median(e0) * 100) if GRAPH else None,
           "loop_err_after_cm": float(np.median(e1) * 100) if GRAPH else None,
           "loop_err_after_p90_cm": float(np.percentile(e1, 90) * 100) if GRAPH else None,
           "correction_max_cm": float(np.linalg.norm(d[:, :2], axis=1).max() * 100),
           "passes": pl.n, "loops_by_pass_pair": cross, "lidar_ties": lidar},
          open(rec / "loop_close.json", "w"), indent=1)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(21, 8))
    colour = pl.pass_of if pl.n > 1 else ts
    for k, (T, ttl) in enumerate([(c2w, "ARKit (floor-pinned" + (", passes aligned)" if pl.n > 1 else ")")),
                                  (c2w_ds, "de-snapped"), (out, "loop-closed")]):
        sc = ax[k].scatter(T[::10, 0, 3], T[::10, 1, 3], c=colour[::10], s=1, cmap="jet"); ax[k].set_aspect("equal"); ax[k].set_title(ttl)
        if k == 2 and len(loops):
            for i, j in zip(li[::20], lj[::20]): ax[k].plot(*np.c_[out[keep[i], :2, 3], out[keep[j], :2, 3]], "k-", lw=0.2)
    plt.colorbar(sc, ax=ax[2], label="pass" if pl.n > 1 else "time (s)"); plt.tight_layout()
    plt.savefig(rec / "loop_close.png", dpi=80)
except Exception as ex:
    print("plot failed:", ex)
print("wrote record3d/poses_lc.npz and sparse/0/images.txt")
