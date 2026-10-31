"""Record3D "EXR + JPG" exports -> floor-pinned metric poses, the images to reconstruct from, the LiDAR frames for the
collider, and a COLMAP text model. One recording, or several passes of the same place (e.g. the phone at three heights).

    python prep_record3d.py <out dir> --pass <EXR_RGBD dir> [--pass <EXR_RGBD dir> ...]
        [--image-step-m 0.06] [--image-step-deg 3] [--min-sharpness 0.25] [--keep-every 0] [--max-images 0]
        [--depth-step-m 0.015] [--depth-step-deg 1.0] [--floor-factor 0.6] [--floor-below 0] [--prefix f] [--no-depth]

Run in the tools environment (.venv). Each pass is prepared in its own ARKit world (OpenGL camera, Y up, metres):
  1. vertical: ARKit drifts vertically and jumps when it relocalises. Under every 5th frame the floor is found in the
     LiDAR points (r3dlib.floor_drop: the densest layer more than 0.6 x the camera's local height below it) and moved
     to height 0; per stretch between ARKit's jumps, the floor is median-filtered over ~10 s and points more than
     15 cm off are rejected (a table top seen for a few seconds);
  2. horizontally nothing is changed here: loop_close.py removes the jumps and closes the loops from image matches;
  3. images: a new slot every --image-step-m of movement or --image-step-deg of turning, and from each slot the
     sharpest frame (Laplacian variance). A slot whose best frame is still below --min-sharpness x the local median
     sharpness is dropped (blurred all through: a fast turn). Standing still adds no images, so walking slowly costs
     nothing. 0.06 m / 3 deg gave 2,905 images on the 22 Oct walk (the first version's sharpest-of-8 gave 3,050);
     --keep-every N brings that first rule back;
  4. LiDAR frames for the collider: one every --depth-step-m / --depth-step-deg (9.5k on the 22 Oct walk).
The passes are NOT aligned to each other here; each keeps its own horizontal origin and heading. The colmap stage
does that from image matches (align_passes.py). Height is shared already: every pass has its floor at z = 0.

Output world: Z-up (x, -z, y of ARKit), floor at z = 0, metres; OpenCV cameras. Frames of all passes share one
index g. Writes into <out dir>: images/<prefix><g:06d>.jpg, sparse/0/{cameras,images,points3D}.txt (PINHOLE cameras
binned by 1 px of focal length; no points), record3d/poses.npz (every frame), record3d/kept_frames.npy,
record3d/depth.npy + depth_frames.npy (float16 metres), record3d/prep.json (per-pass numbers).
"""
import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation

import r3dlib as L

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--pass", dest="passes", action="append", required=True, help="a Record3D EXR_RGBD folder; repeat")
ap.add_argument("--image-step-m", type=float, default=0.06); ap.add_argument("--image-step-deg", type=float, default=3.0)
ap.add_argument("--min-sharpness", type=float, default=0.25); ap.add_argument("--keep-every", type=int, default=0)
ap.add_argument("--max-images", type=int, default=0, help="thin each pass's images evenly to at most this many")
ap.add_argument("--depth-step-m", type=float, default=0.015); ap.add_argument("--depth-step-deg", type=float, default=1.0)
ap.add_argument("--floor-factor", type=float, default=0.6)
ap.add_argument("--floor-below", type=float, default=0.0, help="fixed floor window (m below the camera); 0 = adaptive")
ap.add_argument("--prefix", default="f"); ap.add_argument("--no-depth", action="store_true")
a = ap.parse_args()
out = Path(a.out)
A = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)      # ARKit Y-up -> Z-up
pool = ThreadPoolExecutor(8)


def segment_floor(idx, fw, snaps, N, fps):
    """Per stretch between ARKit's jumps: a robust ~10 s median of the measured floor heights, interpolated."""
    floor = np.full(N, np.nan)
    size = max(3, int(10 * fps / 5)) | 1
    bounds = np.r_[0, snaps, N]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        sel = (idx >= lo + 6) & (idx < hi)          # skip the frames of the jump itself
        if sel.sum() < 3:
            sel = (idx >= lo) & (idx < hi)
        if not sel.any():
            continue
        f0 = fw[sel]
        med = median_filter(f0, size=min(size, sel.sum() | 1), mode="nearest")
        ok = np.abs(f0 - med) < 0.15
        sm = median_filter(f0[ok], size=min(size, ok.sum() | 1), mode="nearest") if ok.sum() > 2 else med
        floor[lo:hi] = np.interp(np.arange(lo, hi), idx[sel][ok] if ok.sum() > 2 else idx[sel], sm)
    if np.isnan(floor).all():
        raise SystemExit("no floor found under the camera")
    good = np.where(~np.isnan(floor))[0]           # a stretch with no measurement takes its neighbour's floor
    return np.interp(np.arange(N), good, floor[good])


def prep_pass(p: int, src: Path):
    m = L.load_meta(src)
    P = np.asarray(m["poses"], float); KK = np.asarray(m["perFrameIntrinsicCoeffs"], float)
    ts = np.asarray(m["frameTimestamps"], float); N = len(P); s = m["dw"] / m["w"]
    fps = float(m.get("fps") or 1.0 / np.median(np.diff(ts)))
    R_gl = Rotation.from_quat(P[:, :4]).as_matrix(); t = P[:, 4:].copy()
    step = np.linalg.norm(np.diff(t, axis=0), axis=1)
    big = np.where(step > 0.04)[0] + 1              # first frame after each large step: ARKit relocalised
    snaps = np.array([b for k, b in enumerate(big) if k == 0 or b - big[k - 1] > 30], int)
    print(f"[pass {p}] {src}: {N:,} frames, {N / fps:.0f} s, {step.sum():.0f} m (ARKit); {len(snaps)} jumps", flush=True)

    # 1. the floor under every 5th frame
    idx = np.arange(0, N, 5)
    H = np.stack(list(pool.map(lambda i: L.drop_histogram(L.read_depth(src, i), KK[i], s, R_gl[i]), idx)))
    if a.floor_below > 0:
        thr = np.full(len(idx), a.floor_below)
    else:
        strong = np.array([L.strongest_layer(h) for h in H])
        if not np.isfinite(strong).any():
            raise SystemExit(f"pass {p}: no LiDAR points below the camera at all ({src})")
        win = max(1, int(30 * fps / 5))             # the camera's height over the last/next 30 s
        loc = np.array([np.nanmedian(w) if np.isfinite(w).any() else np.nan
                        for w in (strong[max(0, k - win):k + win + 1] for k in range(len(idx)))])
        loc = np.where(np.isfinite(loc), loc, np.nanmedian(strong))
        thr = np.clip(a.floor_factor * loc, 0.2, 0.9)
    hcam = np.array([L.floor_drop(h, th) for h, th in zip(H, thr)])
    ok = np.isfinite(hcam)
    if ok.sum() < 3:
        raise SystemExit(f"pass {p}: the floor was found under only {ok.sum()} frames; the phone must see the floor")
    fw = t[idx[ok], 1] - hcam[ok]                   # the measured floor height, ARKit y
    floor = segment_floor(idx[ok], fw, snaps, N, fps)
    t[:, 1] -= floor
    print(f"[pass {p}] floor found under {ok.mean():.0%} of the sampled frames; window {np.median(thr):.2f} m below "
          f"the camera; floor per stretch: " + ", ".join(f"{floor[lo]:+.2f}" for lo in np.r_[0, snaps]), flush=True)

    # 2. Z-up world, OpenCV cameras
    c2w = np.tile(np.eye(4), (N, 1, 1))
    c2w[:, :3, :3] = A @ R_gl @ np.diag([1, -1, -1]); c2w[:, :3, 3] = t @ A.T

    # 3. images
    sharp = np.array(list(pool.map(lambda i: L.sharpness(src / "rgb" / f"{i}.jpg"), range(N))))
    if a.keep_every > 0:
        keep = [b + int(np.argmax(sharp[b:b + a.keep_every])) for b in range(0, N, a.keep_every)]
        dropped, rule = 0, f"sharpest of every {a.keep_every} frames"
    else:
        slots = L.motion_slots(c2w, a.image_step_m, a.image_step_deg, min_frames=3)
        keep, dropped = L.pick_sharpest(slots, sharp, a.min_sharpness, window=int(10 * fps))
        rule = (f"sharpest frame per {a.image_step_m * 100:g} cm / {a.image_step_deg:g} deg of motion"
                + (f", blurred slots (< {a.min_sharpness:g} x local median) dropped" if a.min_sharpness > 0 else ""))
    if a.max_images and len(keep) > a.max_images:
        keep = [keep[int(k)] for k in np.linspace(0, len(keep) - 1, a.max_images).round()]
    # 4. LiDAR frames for the collider
    dfr = [lo for lo, _ in L.motion_slots(c2w, a.depth_step_m, a.depth_step_deg, 1)]
    h = c2w[:, 2, 3]
    st = {"src": str(src), "frames": N, "fps": fps, "duration_s": round(N / fps, 1), "path_m": round(float(step.sum()), 1),
          "snaps": int(len(snaps)), "images": len(keep), "blurred_dropped": int(dropped), "depth_frames": len(dfr),
          "floor_found": round(float(ok.mean()), 3), "floor_window_m": round(float(np.median(thr)), 2),
          "camera_height_m": round(float(np.median(h)), 3), "camera_height_p5": round(float(np.percentile(h, 5)), 3),
          "camera_height_p95": round(float(np.percentile(h, 95)), 3),
          "kept_sharpness_p5": round(float(np.percentile(sharp[keep], 5)), 1)}
    print(f"[pass {p}] images {len(keep):,} ({rule}; {dropped} blurred slots dropped), LiDAR frames {len(dfr):,}; "
          f"camera {st['camera_height_m']:.2f} m above the floor (p5 {st['camera_height_p5']:.2f}, p95 "
          f"{st['camera_height_p95']:.2f})", flush=True)
    return dict(c2w=c2w, K=KK, ts=ts, floor=floor, snaps=snaps, floor_idx=idx[ok], floor_meas=fw,
                keep=np.asarray(keep, int), dfr=np.asarray(dfr, int), stats=st, rule=rule,
                size=(int(m["w"]), int(m["h"]), int(m["dw"]), int(m["dh"])))


passes = [Path(x) for x in a.passes]
res = [prep_pass(p, src) for p, src in enumerate(passes)]
sizes = {r["size"] for r in res}
if len(sizes) > 1:
    raise SystemExit(f"the passes have different image sizes {sorted(sizes)}: record all with the same Record3D settings")
W, H, dw, dh = res[0]["size"]
start = np.r_[0, np.cumsum([len(r["c2w"]) for r in res])]
cat = lambda k, off=False: np.concatenate([r[k] + (start[p] if off else 0) for p, r in enumerate(res)])
c2w, K, keep, dfr = cat("c2w"), cat("K"), cat("keep", True), cat("dfr", True)
ts_local = cat("ts")
ts = np.concatenate([r["ts"] - r["ts"][0] + p * L.PASS_GAP_S for p, r in enumerate(res)])
pass_of = np.repeat(np.arange(len(res)), np.diff(start))
if len(c2w) > 999_999:
    raise SystemExit(f"{len(c2w):,} frames in all is more than 999,999 (images are numbered with 6 digits)")

rec = out / "record3d"; rec.mkdir(parents=True, exist_ok=True)
np.savez(rec / "poses.npz", c2w=c2w, K=K, ts=ts, ts_local=ts_local, floor=cat("floor"), snaps=cat("snaps", True),
         floor_idx=cat("floor_idx", True), floor_meas=cat("floor_meas"), pass_of=pass_of, pass_start=start,
         srcs=np.array([str(s.resolve()) for s in passes]), src=str(passes[0].resolve()), w=W, h=H, dw=dw, dh=dh,
         multi_snap=True, image_rule=res[0]["rule"])
np.save(rec / "kept_frames.npy", keep)
print(f"camera height above the floor, all passes: median {np.median(c2w[:, 2, 3]):.2f} m", flush=True)

pl = L.Passes({"srcs": [str(s) for s in passes], "pass_start": start, "c2w": c2w})
if not a.no_depth:
    D = open_memmap(rec / "depth.npy", mode="w+", dtype=np.float16, shape=(len(dfr), dh, dw))
    for k0 in range(0, len(dfr), 2000):
        chunk = dfr[k0:k0 + 2000]
        D[k0:k0 + len(chunk)] = np.stack(list(pool.map(lambda g: L.read_depth(*pl.locate(int(g))), chunk))).astype(np.float16)
    D.flush(); del D
    np.save(rec / "depth_frames.npy", dfr)
    print(f"depth: {len(dfr):,} LiDAR frames -> record3d/depth.npy", flush=True)

img_dir = out / "images"; img_dir.mkdir(parents=True, exist_ok=True)
names = [f"{a.prefix}{g:06d}.jpg" for g in keep]
list(pool.map(lambda gn: shutil.copy2(pl.rgb_path(int(gn[0])), img_dir / gn[1]), zip(keep, names)))
sp = out / "sparse" / "0"; sp.mkdir(parents=True, exist_ok=True)
fbin = np.round(K[keep, 0]).astype(int); cams = {f: j + 1 for j, f in enumerate(sorted(set(fbin)))}
with open(sp / "cameras.txt", "w") as f:
    f.write("# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n")
    for fb, cid in cams.items():
        fx, fy, cx, cy = np.median(K[keep[fbin == fb]], axis=0)
        f.write(f"{cid} PINHOLE {W} {H} {fx:.4f} {fy:.4f} {cx:.4f} {cy:.4f}\n")
with open(sp / "images.txt", "w") as f:
    f.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n# POINTS2D[] (empty)\n")
    for j, (g, name) in enumerate(zip(keep, names)):
        w2c = np.linalg.inv(c2w[g]); qx, qy, qz, qw = Rotation.from_matrix(w2c[:3, :3]).as_quat()
        f.write(f"{j + 1} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {w2c[0, 3]:.6f} {w2c[1, 3]:.6f} {w2c[2, 3]:.6f} "
                f"{cams[fbin[j]]} {name}\n\n")
(sp / "points3D.txt").write_text("# POINT3D_ID X Y Z R G B ERROR TRACK[]\n")
(rec / "prep.json").write_text(json.dumps({"passes": [r["stats"] for r in res], "images": len(keep),
                                           "depth_frames": len(dfr), "image_rule": res[0]["rule"],
                                           "focal_bins": len(cams)}, indent=1))
print(f"images: {len(keep):,} in {len(res)} pass(es) -> {img_dir}; cameras: {len(cams)} focal bins", flush=True)
