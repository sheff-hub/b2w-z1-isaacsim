"""Put the frames of a test clip (a Record3D recording never trained on) into a finished scene, precisely enough to
score the splat there.

    python localize_test.py <scene dir> pairs-align [--test-images 60] [--scene-images 300]
    python localize_test.py <scene dir> align [--partners 20]
    python localize_test.py <scene dir> localize

Needs <scene>/test/ from prep_record3d.py --prefix t (the clip's floor-pinned ARKit poses, images t<g:06d>.jpg) and
<scene>/test/colmap/database.db holding the features of the scene's images and of the test images.
  pairs-align: ~every 0.75 m of the clip x ~every 1 m of the scene -> test/colmap/pairs_align.txt;
  align:       every verified test-scene pair is a vote (LiDAR-lifted PnP both ways, as align_passes.py) for the
               yaw-plane transform from the clip's ARKit world to the scene; the consensus places the whole clip.
               Then each test image is paired with the --partners nearest scene images that look the same way ->
               test/colmap/pairs_loc.txt;
  localize:    per test image, every keypoint matched in the scene's images gets a 3D point from the scene alone (so
               it owes nothing to the test image): triangulated from two or more scene images (their bundle-adjusted
               poses, sparse/0) where their rays meet at >= 1.5 deg; else lifted with a scene image's own LiDAR depth
               (looking down a corridor, neighbouring scene images see points at < 1 deg); else triangulated down
               to 0.5 deg. PnP (RANSAC + LM) on them gives its pose. The ARKit placement is only a sanity check (frames more than
               0.5 m / 10 deg from it are dropped). A frame that matches too little of the scene to be localised
               (often a view the scene covers badly, so leaving it out would flatter the score) takes its ARKit
               pose corrected by the corrections of the localised frames around it, when there is one within
               --interp-s on each side and no ARKit jump in between (ARKit is accurate over a second or two).
               Writes test/dataset/ (images + sparse/0 for 3DGRUT) and test/localize.json.
"""
import argparse, json, os
from pathlib import Path
import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation, Slerp
import cv2
import r3dlib as L

ap = argparse.ArgumentParser(); ap.add_argument("scene"); ap.add_argument("step", choices=["pairs-align", "align", "localize"])
ap.add_argument("--test-images", type=int, default=60); ap.add_argument("--scene-images", type=int, default=300)
ap.add_argument("--partners", type=int, default=20); ap.add_argument("--min-inliers", type=int, default=30)
ap.add_argument("--min-votes", type=int, default=5); ap.add_argument("--interp-s", type=float, default=2.0)
a = ap.parse_args(); scene = Path(a.scene); test = scene / "test"; col = test / "colmap"

zt = dict(np.load(test / "record3d" / "poses.npz")); plt_ = L.Passes(zt); keep_t = np.load(test / "record3d" / "kept_frames.npy")
zs = dict(np.load(scene / "record3d" / "poses_refined.npz")); pls = L.Passes(zs)
rec = pycolmap.Reconstruction(str(scene / "sparse" / "0"))
train = {}                                                  # name -> (c2w 4x4, K4, 3x4 projection)
for im in rec.images.values():
    w2c = np.r_[im.cam_from_world().matrix(), [[0, 0, 0, 1]]]
    fx, fy, cx, cy = rec.cameras[im.camera_id].params[:4]
    train[im.name] = (np.linalg.inv(w2c), np.array([fx, fy, cx, cy]), L.kmat([fx, fy, cx, cy]) @ w2c[:3])
tnames = {L.frame_of_name(p.name): p.name for p in (test / "images").glob("*.jpg")}
s = float(zt["dw"]) / float(zt["w"])


def subsample(frames, c2w, step_m, step_deg, cap):
    pick = [frames[lo] for lo, _ in L.motion_slots(c2w[frames], step_m, step_deg)]
    return pick if len(pick) <= cap else [pick[int(k)] for k in np.linspace(0, len(pick) - 1, cap).round()]


if a.step == "pairs-align":
    tsub = subsample(keep_t, zt["c2w"], 0.75, 25, a.test_images)
    order = sorted(train, key=L.frame_of_name)
    fr_s = np.array([L.frame_of_name(n) for n in order])
    C = np.stack([train[n][0] for n in order])
    ssub = [order[lo] for lo, _ in L.motion_slots(C, 1.0, 30)]
    if len(ssub) > a.scene_images:
        ssub = [ssub[int(k)] for k in np.linspace(0, len(ssub) - 1, a.scene_images).round()]
    with open(col / "pairs_align.txt", "w") as fh:
        for g in tsub:
            for n in ssub:
                fh.write(f"{tnames[g]} {n}\n")
    print(f"{len(tsub)} test images x {len(ssub)} scene images = {len(tsub) * len(ssub)} pairs -> pairs_align.txt")
    raise SystemExit(0)

mdb = L.MatchDB(col / "database.db")
is_test = lambda n: n.startswith("t")

if a.step == "align":
    votes, why = L.alignment_votes(
        mdb, lambda n: 1 if is_test(n) else (0 if n in train else None),
        lambda n: zt["c2w"][L.frame_of_name(n)] if is_test(n) else train[n][0],
        lambda n: plt_.depth(L.frame_of_name(n)) if is_test(n) else pls.depth(L.frame_of_name(n)),
        lambda n: zt["K"][L.frame_of_name(n)] if is_test(n) else train[n][1], s, a.min_inliers)
    G = np.stack([v[0] for v in votes.get((0, 1), [])]) if votes.get((0, 1)) else np.zeros((0, 4, 4))
    T, inl, st = L.consensus_yaw_transform(G, zt["c2w"][keep_t, :3, 3].mean(0))
    print(f"test-scene pairs tried {why['tried']}: {st['votes']} votes, {st.get('inliers', 0)} agree"
          + (f" (spread {st['spread_cm_median']:.1f} cm, {st['yaw_spread_deg']:.2f} deg)" if T is not None else ""))
    if T is None or st["inliers"] < a.min_votes:
        raise SystemExit("the test clip could not be placed in the scene: too few verified views in common "
                         f"({st.get('inliers', 0)} agreeing votes, need {a.min_votes}). Was it recorded in the same place?")
    ct = T @ zt["c2w"]
    np.save(test / "record3d" / "c2w_placed.npy", ct)
    # pose-guided partners for every test image
    order = sorted(train)
    C = np.stack([train[n][0] for n in order]); cs, fs = C[:, :3, 3], C[:, :3, 2]
    done = {tuple(sorted(l.split())) for l in (col / "pairs_align.txt").read_text().splitlines() if l.strip()}
    n_pairs = 0
    with open(col / "pairs_loc.txt", "w") as fh:
        for g in keep_t:
            d = np.linalg.norm(cs - ct[g, :3, 3], axis=1); ok = (d < 4.0) & (fs @ ct[g, :3, 2] > np.cos(np.radians(70)))
            for k in np.where(ok)[0][np.argsort(d[ok])][:a.partners]:
                pr = tuple(sorted((tnames[g], order[k])))
                if pr not in done:
                    fh.write(f"{pr[0]} {pr[1]}\n"); n_pairs += 1
    json.dump({**st, "yaw_deg": float(np.degrees(np.arctan2(T[1, 0], T[0, 0]))), "shift_m": [float(T[0, 3]), float(T[1, 3])],
               "tried": why}, open(test / "align.json", "w"), indent=1)
    print(f"placed: turned {np.degrees(np.arctan2(T[1, 0], T[0, 0])):+.1f} deg, shifted ({T[0, 3]:+.2f}, {T[1, 3]:+.2f}) m; "
          f"{n_pairs} pose-guided pairs -> pairs_loc.txt")
    raise SystemExit(0)

# ---------------- localize
ct = np.load(test / "record3d" / "c2w_placed.npy")
obs = {}                                                   # test image -> {test keypoint: [(scene name, uv), ...]}
for i1, i2, m in mdb.verified(15):
    n1, n2 = mdb.name_of[i1], mdb.name_of[i2]
    if is_test(n1) == is_test(n2):
        continue
    if is_test(n2):
        n1, n2, i1, i2, m = n2, n1, i2, i1, m[:, ::-1]
    if n2 not in train:
        continue
    uv2 = mdb.keypoints(i2)[m[:, 1]]
    d = obs.setdefault(n1, {})
    for q, uv in zip(m[:, 0], uv2):
        d.setdefault(int(q), []).append((n2, uv))


def triangulate(views, min_deg):
    """3D point from >= 2 scene views [(name, uv)], dropping the worst view while any is > 2 px off; None if the
    rays are too parallel (< min_deg) or it falls behind a camera."""
    views = list(views)
    while len(views) >= 2:
        Ps = np.stack([train[n][2] for n, _ in views]); uv = np.stack([u for _, u in views])
        A = np.concatenate([uv[:, :1] * Ps[:, 2] - Ps[:, 0], uv[:, 1:2] * Ps[:, 2] - Ps[:, 1]])
        X = np.linalg.svd(A)[2][-1]
        if abs(X[3]) < 1e-12:
            return None
        X = X[:3] / X[3]; xh = Ps @ np.r_[X, 1.0]
        if (xh[:, 2] <= 0).any():
            return None
        err = np.linalg.norm(xh[:, :2] / xh[:, 2:] - uv, axis=1)
        if err.max() <= 2.0:
            cs = np.stack([train[n][0][:3, 3] for n, _ in views]); r = X - cs; r /= np.linalg.norm(r, axis=1, keepdims=True)
            ang = np.degrees(np.arccos(np.clip(r @ r.T, -1, 1))).max()
            return X if ang >= min_deg else None
        views.pop(int(np.argmax(err)))
    return None


def lidar_point(views):
    """The point where a scene image's LiDAR puts the keypoint (world frame, its bundle-adjusted pose); with several
    views, their median if they agree within 5 cm."""
    P = []
    for n, uv in views[:4]:
        f = L.frame_of_name(n)
        X, ok = L.lift(pls.depth(f), zs["K"][f], s, uv[None])
        if ok[0]:
            P.append(train[n][0][:3, :3] @ X[0] + train[n][0][:3, 3])
    if not P:
        return None
    P = np.array(P); m = np.median(P, axis=0)
    return m if np.linalg.norm(P - m, axis=1).max() < 0.05 else None


ds = test / "dataset"; (ds / "sparse" / "0").mkdir(parents=True, exist_ok=True); (ds / "images").mkdir(exist_ok=True)
for p in (ds / "images").glob("*"):
    p.unlink()
res, rows = [], []
for g in keep_t:
    name = tnames[g]; K4 = zt["K"][g]
    kp = mdb.keypoints(mdb.id_of[name]) if name in mdb.id_of else None
    o = obs.get(name, {})
    pts, how = [], {"triangulated": 0, "lidar": 0, "triangulated_low_angle": 0}
    for q, v in o.items():
        X = triangulate(v, 1.5) if len(v) >= 2 else None
        if X is not None:
            pts.append((q, X)); how["triangulated"] += 1; continue
        X = lidar_point(v)
        if X is not None:
            pts.append((q, X)); how["lidar"] += 1; continue
        X = triangulate(v, 0.5) if len(v) >= 2 else None
        if X is not None:
            pts.append((q, X)); how["triangulated_low_angle"] += 1
    info = {"name": name, "frame": int(g), "partners": len({n for v in o.values() for n, _ in v}), "points": len(pts),
            **how}
    if kp is None or len(pts) < 30:
        res.append({**info, "ok": False, "why": "too few triangulated points"}); continue
    X = np.stack([X for _, X in pts]); uv = kp[[q for q, _ in pts]]
    r = cv2.solvePnPRansac(X, uv.astype(np.float64), L.kmat(K4), None, iterationsCount=1000, reprojectionError=3.0,
                           confidence=0.9999, flags=cv2.SOLVEPNP_EPNP)
    if not r[0] or r[3] is None or len(r[3]) < 30:
        res.append({**info, "ok": False, "why": "PnP failed"}); continue
    inl = r[3][:, 0]
    rv, tv = cv2.solvePnPRefineLM(X[inl], uv[inl].astype(np.float64), L.kmat(K4), None, r[1], r[2])
    w2c = np.eye(4); w2c[:3, :3] = cv2.Rodrigues(rv)[0]; w2c[:3, 3] = tv[:, 0]
    proj = (L.kmat(K4) @ (w2c[:3, :3] @ X[inl].T + w2c[:3, 3:])).T
    err = np.linalg.norm(proj[:, :2] / proj[:, 2:] - uv[inl], axis=1)
    c2w = np.linalg.inv(w2c)
    dpos = float(np.linalg.norm(c2w[:3, 3] - ct[g, :3, 3])); drot = L.rotation_angle_deg(c2w[:3, :3], ct[g, :3, :3])
    info.update(inliers=int(len(inl)), inlier_ratio=round(len(inl) / len(X), 3), reproj_px=round(float(np.median(err)), 2),
                arkit_diff_cm=round(dpos * 100, 1), arkit_diff_deg=round(drot, 2), height_m=round(float(c2w[2, 3]), 3))
    if len(inl) < 40 or len(inl) < 0.3 * len(X) or dpos > 0.5 or drot > 10:
        res.append({**info, "ok": False, "why": "weak or inconsistent pose"}); continue
    res.append({**info, "ok": True})
    rows.append((name, w2c, K4))
if not rows:
    raise SystemExit("no test image could be localised in the scene")
# frames that could not be localised: their ARKit pose, corrected as their localised neighbours were
loc = {r["frame"]: np.linalg.inv(w2c) for r, (_, w2c, _) in zip([r for r in res if r["ok"]], rows)}
lf = np.array(sorted(loc)); ts = zt["ts"]; snaps = np.asarray(zt["snaps"], int)
corr = {f: loc[f] @ np.linalg.inv(ct[f]) for f in lf}
for r in res:
    if r["ok"]:
        r["how"] = "pnp"; continue
    g = r["frame"]; i = int(np.searchsorted(lf, g))
    a_, b_ = (lf[i - 1] if i > 0 else None), (lf[i] if i < len(lf) else None)
    near = lambda f: f is not None and abs(ts[f] - ts[g]) <= a.interp_s and not ((snaps > min(f, g)) & (snaps <= max(f, g))).any()
    if near(a_) and near(b_):
        w = (ts[g] - ts[a_]) / max(ts[b_] - ts[a_], 1e-9)
        R = Slerp([0, 1], Rotation.from_matrix([corr[a_][:3, :3], corr[b_][:3, :3]]))([w]).as_matrix()[0]
        C = np.eye(4); C[:3, :3] = R; C[:3, 3] = (1 - w) * corr[a_][:3, 3] + w * corr[b_][:3, 3]
    else:
        continue
    c2w = C @ ct[g]
    r.update(ok=True, how="interpolated", height_m=round(float(c2w[2, 3]), 3))
    rows.append((r["name"], np.linalg.inv(c2w), zt["K"][g]))
fb = sorted({int(round(K4[0])) for _, _, K4 in rows}); cam_of = {f: k + 1 for k, f in enumerate(fb)}
W, H = int(zt["w"]), int(zt["h"])
with open(ds / "sparse" / "0" / "cameras.txt", "w") as f:
    f.write("# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n")
    for fv in fb:
        Ks = np.median([K4 for _, _, K4 in rows if int(round(K4[0])) == fv], axis=0)
        f.write(f"{cam_of[fv]} PINHOLE {W} {H} {Ks[0]:.4f} {Ks[1]:.4f} {Ks[2]:.4f} {Ks[3]:.4f}\n")
with open(ds / "sparse" / "0" / "images.txt", "w") as f:
    f.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n")
    for k, (name, w2c, K4) in enumerate(rows):
        qx, qy, qz, qw = Rotation.from_matrix(w2c[:3, :3]).as_quat()
        f.write(f"{k + 1} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {w2c[0, 3]:.6f} {w2c[1, 3]:.6f} {w2c[2, 3]:.6f} "
                f"{cam_of[int(round(K4[0]))]} {name}\n\n")
        os.symlink(os.path.relpath(test / "images" / name, ds / "images"), ds / "images" / name)
(ds / "sparse" / "0" / "points3D.txt").write_text("# POINT3D_ID X Y Z R G B ERROR TRACK[]\n")
ok = [r for r in res if r.get("how") == "pnp"]
summ = {"test_images": len(res), "localized": len(ok), "interpolated": sum(1 for r in res if r.get("how") == "interpolated"),
        "inliers_median": float(np.median([r["inliers"] for r in ok])),
        "reproj_px_median": float(np.median([r["reproj_px"] for r in ok])),
        "arkit_diff_cm_median": float(np.median([r["arkit_diff_cm"] for r in ok])),
        "height_m_median": float(np.median([r["height_m"] for r in ok])),
        "failed": {w: sum(1 for r in res if not r["ok"] and r["why"] == w) for w in {r.get("why") for r in res if not r["ok"]}}}
json.dump({**summ, "frames": res}, open(test / "localize.json", "w"), indent=1)
print(f"localised {len(ok)}/{len(res)} test images (+{summ['interpolated']} from their neighbours): median "
      f"{summ['inliers_median']:.0f} inliers, reprojection {summ['reproj_px_median']:.2f} px, "
      f"{summ['arkit_diff_cm_median']:.1f} cm from the ARKit placement; not localised: {summ['failed']}")
