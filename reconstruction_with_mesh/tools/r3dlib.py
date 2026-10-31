"""Shared helpers for the tools that run in the tools environment (.venv): Record3D passes, frame selection by motion,
the LiDAR floor, LiDAR-lifted PnP between two images, COLMAP database access, and the yaw-plane maths of a Z-up world.

Frames of all passes share one global index g (images are named f<g:06d>.jpg); record3d/poses.npz says which pass a
frame belongs to (pass_of, pass_start) and which Record3D export holds it (srcs). Scenes made before passes existed
have one `src` and no pass arrays: Passes() reads those too.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

PASS_GAP_S = 100_000.0     # pass k's timestamps are shifted by k x this: "far apart in time" holds across passes


# ------------------------------------------------------------------ names
def frame_of_name(name: str) -> int:
    """f000123.jpg -> 123 (also the 5-digit names of older scenes)."""
    return int(Path(name).stem[1:])


def image_name(g: int) -> str:
    return f"f{g:06d}.jpg"


# ------------------------------------------------------------------ Record3D
def load_meta(src: Path) -> dict:
    return json.loads((Path(src) / "metadata.json").read_text())


def read_depth(src: Path, i: int) -> np.ndarray:
    """LiDAR depth of frame i of a Record3D export (metres; Record3D puts it in every channel, we take the last)."""
    d = cv2.imread(str(Path(src) / "depth" / f"{i}.exr"), cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(Path(src) / "depth" / f"{i}.exr")
    return d[..., 2] if d.ndim == 3 else d


def sharpness(path) -> float:
    return float(cv2.Laplacian(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), cv2.CV_64F).var())


class Passes:
    """Global frame index -> (Record3D export, frame index in it), for poses.npz of any age."""

    def __init__(self, z):
        if "srcs" in z:
            self.srcs = [Path(str(s)) for s in z["srcs"]]
            self.start = np.asarray(z["pass_start"], int)
        else:                                   # one recording, before passes existed
            self.srcs = [Path(str(z["src"]))]
            self.start = np.array([0, len(z["c2w"])])
        self.n = len(self.srcs)
        self.pass_of = np.repeat(np.arange(self.n), np.diff(self.start))
        self._dcache: dict[int, np.ndarray] = {}

    def locate(self, g: int) -> tuple[Path, int]:
        p = int(self.pass_of[g])
        return self.srcs[p], int(g - self.start[p])

    def rgb_path(self, g: int) -> Path:
        src, i = self.locate(g)
        return src / "rgb" / f"{i}.jpg"

    def depth(self, g: int) -> np.ndarray:
        if g not in self._dcache:
            if len(self._dcache) > 4000:
                self._dcache.clear()
            self._dcache[g] = read_depth(*self.locate(g))
        return self._dcache[g]

    def ranges(self):
        """(pass, first frame, end frame) of each pass, global indices."""
        return [(p, int(self.start[p]), int(self.start[p + 1])) for p in range(self.n)]


# ------------------------------------------------------------------ frame selection
def rotation_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def motion_slots(c2w: np.ndarray, step_m: float, step_deg: float, min_frames: int = 1) -> list[tuple[int, int]]:
    """Cut a frame sequence into slots: a new slot starts once the camera has moved step_m or turned step_deg (any
    axis) since the current slot began, and the slot is at least min_frames long. Standing still adds nothing; a turn
    on the spot adds a slot every step_deg. Returns [(first, end), ...] covering all frames."""
    n = len(c2w)
    slots, s0 = [], 0
    for i in range(1, n):
        if i - s0 < min_frames:
            continue
        if (np.linalg.norm(c2w[i, :3, 3] - c2w[s0, :3, 3]) >= step_m
                or rotation_angle_deg(c2w[s0, :3, :3], c2w[i, :3, :3]) >= step_deg):
            slots.append((s0, i))
            s0 = i
    slots.append((s0, n))
    return slots


def pick_sharpest(slots, sharp: np.ndarray, min_ratio: float = 0.0, window: int = 600) -> tuple[list[int], int]:
    """The sharpest frame of every slot; a slot whose best frame is below min_ratio x the local median sharpness
    (+-window frames) is dropped (motion blur everywhere in it). Returns (kept frames, slots dropped)."""
    keep, dropped = [], 0
    if min_ratio > 0:
        sub = sharp[::10]
        local = np.array([np.median(sub[max(0, k - window // 10):k + window // 10 + 1]) for k in range(len(sub))])
    for a, b in slots:
        f = a + int(np.argmax(sharp[a:b]))
        if min_ratio > 0 and sharp[f] < min_ratio * local[min(f // 10, len(local) - 1)]:
            dropped += 1
            continue
        keep.append(f)
    return keep, dropped


# ------------------------------------------------------------------ LiDAR floor
# Heights of LiDAR points relative to the camera, binned: 0.5 cm bins from 3.5 m below the camera to 0.2 m below it.
DROP_EDGES = np.arange(-3.5, -0.2 + 1e-9, 0.005)
DROP_CENTRES = 0.5 * (DROP_EDGES[:-1] + DROP_EDGES[1:])


def drop_histogram(d: np.ndarray, K4: np.ndarray, s: float, R_gl: np.ndarray) -> np.ndarray:
    """Histogram (DROP_EDGES) of how far below the camera the points of one LiDAR frame lie, gravity-aligned
    (R_gl: ARKit camera-to-world rotation, world Y up)."""
    dh, dw = d.shape
    v, u = np.mgrid[0:dh, 0:dw] + 0.5
    fx, fy, cx, cy = K4 * s
    ok = (d > 0.3) & (d < 3.5)
    y = (R_gl @ np.c_[(u[ok] - cx) / fx * d[ok], -(v[ok] - cy) / fy * d[ok], -d[ok]].T)[1]
    return np.histogram(y, bins=DROP_EDGES)[0].astype(np.int32)


def _smooth(h: np.ndarray) -> np.ndarray:
    return np.convolve(h, np.ones(4), "same")            # 2 cm layers


def strongest_layer(h: np.ndarray, min_points: int = 300) -> float:
    """Drop (m, > 0) of the most populated horizontal layer below the camera, NaN if too few points."""
    if h.sum() < min_points:
        return np.nan
    return float(-DROP_CENTRES[int(np.argmax(_smooth(h)))])


def floor_drop(h: np.ndarray, floor_below: float, min_points: int = 300) -> float:
    """Drop (m, > 0) of the floor below the camera: the most populated layer more than floor_below under the camera,
    refined to the median of the points within 3 cm of it. NaN if too few points.

    floor_below is what keeps table tops out: prep sets it from the camera's own height (0.6 x the local median of
    strongest_layer), so a phone at 0.5 m and one at 1.6 m both find their floor. A fixed 0.7 m (the first version)
    found no floor at all for a phone held lower than that."""
    h = np.where(DROP_CENTRES < -floor_below, h, 0)
    if h.sum() < min_points:
        return np.nan
    k = int(np.argmax(_smooth(h)))
    if DROP_CENTRES[k] > -floor_below - 0.03:
        return np.nan                                     # densest right at the window's edge: a wall, no floor in view
    lo, hi = max(0, k - 6), k + 7
    w = h[lo:hi]
    if w.sum() == 0:
        return np.nan
    c = np.cumsum(w)
    return float(-DROP_CENTRES[lo + int(np.searchsorted(c, c[-1] / 2))])


# ------------------------------------------------------------------ yaw-plane maths (Z-up world, OpenCV cameras)
def yaw_of(R):
    f = R[..., :, 2]
    return np.arctan2(f[..., 1], f[..., 0])


def rz(t):
    c, s_ = np.cos(t), np.sin(t)
    M = np.zeros(np.shape(t) + (3, 3))
    M[..., 0, 0] = c; M[..., 0, 1] = -s_; M[..., 1, 0] = s_; M[..., 1, 1] = c; M[..., 2, 2] = 1
    return M


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def yaw_transform(yaw: float, tx: float, ty: float, tz: float = 0.0) -> np.ndarray:
    G = np.eye(4)
    G[:3, :3] = rz(yaw)
    G[:3, 3] = (tx, ty, tz)
    return G


# ------------------------------------------------------------------ COLMAP database
class MatchDB:
    """Read-only access to a COLMAP database: image names, keypoints, verified matches."""

    def __init__(self, path: Path):
        self.db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        self.name_of = {i: n for i, n in self.db.execute("SELECT image_id, name FROM images")}
        self.id_of = {n: i for i, n in self.name_of.items()}
        self._kp: dict[int, np.ndarray] = {}

    def keypoints(self, iid: int) -> np.ndarray:
        if iid not in self._kp:
            if len(self._kp) > 3000:                 # before adding: clearing after it dropped the entry returned below
                self._kp.clear()
            r, c, d = self.db.execute("SELECT rows, cols, data FROM keypoints WHERE image_id=?", (iid,)).fetchone()
            self._kp[iid] = np.frombuffer(d, np.float32).reshape(r, c)[:, :2]
        return self._kp[iid]

    def verified(self, min_inliers: int = 1):
        """(image_id_1, image_id_2, matches (k, 2) uint32) for every geometrically verified pair."""
        for pid, rows, cols, data in self.db.execute(
                "SELECT pair_id, rows, cols, data FROM two_view_geometries WHERE rows >= ?", (min_inliers,)):
            i2 = pid % 2147483647
            i1 = (pid - i2) // 2147483647
            yield i1, i2, np.frombuffer(data, np.uint32).reshape(rows, cols)


# ------------------------------------------------------------------ LiDAR-lifted PnP
def lift(d: np.ndarray, K4: np.ndarray, s: float, uv: np.ndarray, max_depth: float = 5.0):
    """Keypoints (full-resolution pixels) -> 3D points in that camera (OpenCV), with LiDAR depth; ok = usable depth
    (in range, not on a depth edge)."""
    h, w = d.shape
    x = np.clip((uv[:, 0] * s).astype(int), 1, w - 2)
    y = np.clip((uv[:, 1] * s).astype(int), 1, h - 2)
    zc = d[y, x]
    nb = np.stack([d[y + dy, x + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
    ok = (zc > 0.3) & (zc < max_depth) & (nb.max(0) - nb.min(0) < 0.04 * zc)
    fx, fy, cx, cy = K4
    return np.c_[(uv[:, 0] - cx) / fx * zc, (uv[:, 1] - cy) / fy * zc, zc], ok


def kmat(K4) -> np.ndarray:
    fx, fy, cx, cy = K4
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


def pnp(X: np.ndarray, uv: np.ndarray, K4, min_inl: int = 25, reproj: float = 3.0):
    """Pose of the camera that sees uv, in the frame of the 3D points X (4x4 cam->X-frame), and the inlier count."""
    if len(X) < min_inl:
        return None, 0
    r = cv2.solvePnPRansac(X.astype(np.float64), uv.astype(np.float64), kmat(K4), None, iterationsCount=300,
                           reprojectionError=reproj, confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    if not r[0] or r[3] is None or len(r[3]) < min_inl or len(r[3]) < 0.35 * len(X):
        return None, 0
    inl = r[3][:, 0]
    rv, tv = cv2.solvePnPRefineLM(X[inl].astype(np.float64), uv[inl].astype(np.float64), kmat(K4), None, r[1], r[2])
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rv)[0]
    T[:3, 3] = tv[:, 0]                       # X-frame -> camera
    return np.linalg.inv(T), len(inl)


def relative_pose_both_ways(da, Ka, db_, Kb, s, uva, uvb, tol_m=0.06, tol_deg=1.5, one_way_ok=False):
    """Pose of camera b in camera a's frame from LiDAR-lifted PnP, computed both ways (a's depth into b, b's depth
    into a) and accepted only when the two agree. With one_way_ok, a pair where only one way works (the shared
    features of the other image lie beyond the LiDAR's 5 m) returns that one answer: fine where a consensus of many
    votes follows (aligning passes), not for loop edges. Returns (4x4, inliers) or (None, reason)."""
    Xa, oka = lift(da, Ka, s, uva)
    Xb, okb = lift(db_, Kb, s, uvb)
    Tab, na = pnp(Xa[oka], uvb[oka], Kb) if oka.sum() >= 25 else (None, 0)      # b-cam in a-cam frame
    Tba, nb = pnp(Xb[okb], uva[okb], Ka) if okb.sum() >= 25 else (None, 0)      # a-cam in b-cam frame
    if Tab is None or Tba is None:
        if one_way_ok and (Tab is not None or Tba is not None):
            return (Tab, na) if Tab is not None else (np.linalg.inv(Tba), nb)
        return None, ("depth" if max(oka.sum(), okb.sum()) < 25 else "pnp")
    E = Tab @ Tba
    if np.linalg.norm(E[:3, 3]) > tol_m or np.degrees(Rotation.from_matrix(E[:3, :3]).magnitude()) > tol_deg:
        return None, "disagree"
    Tba_inv = np.linalg.inv(Tba)
    M = np.eye(4)
    M[:3, :3] = Rotation.from_matrix(np.stack([Tab[:3, :3], Tba_inv[:3, :3]])).mean().as_matrix()
    M[:3, 3] = (Tab[:3, 3] + Tba_inv[:3, 3]) / 2
    return M, min(na, nb)


# ------------------------------------------------------------------ consensus of frame-to-frame alignments
def consensus_yaw_transform(G: np.ndarray, ref_point: np.ndarray, tol_m: float = 0.3, tol_deg: float = 2.0,
                            max_tilt_deg: float = 3.0, max_dz: float = 0.2):
    """Votes G (k, 4, 4) for one rigid transform between two Z-up, floor-at-0 frames -> the consensus as a yaw-plane
    transform (yaw about z, shift in x and y; height and tilt are already shared), its inlier mask and stats.
    Votes are compared where they matter: by where they put ref_point (the middle of the moved pass) and by yaw."""
    k = len(G)
    if k == 0:
        return None, np.zeros(0, bool), {"votes": 0}
    tilt = np.degrees(np.arccos(np.clip(G[:, 2, 2], -1, 1)))
    sane = (tilt < max_tilt_deg) & (np.abs(G[:, 2, 3]) < max_dz)
    yaw = np.arctan2(G[:, 1, 0], G[:, 0, 0])
    q = np.einsum("kij,j->ki", G[:, :3, :3], ref_point) + G[:, :3, 3]
    best = np.zeros(k, bool)
    for v in np.where(sane)[0]:
        inl = sane & (np.linalg.norm(q[:, :2] - q[v, :2], axis=1) < tol_m) & (np.abs(wrap(yaw - yaw[v])) < np.radians(tol_deg))
        if inl.sum() > best.sum():
            best = inl
    if not best.any():
        return None, best, {"votes": k, "sane": int(sane.sum()), "inliers": 0}
    y = float(np.arctan2(np.sin(yaw[best]).mean(), np.cos(yaw[best]).mean()))
    # translation: the shift that puts ref_point where the inliers put it, on average
    t = q[best, :2].mean(0) - (rz(y)[:2, :2] @ ref_point[:2])
    out = yaw_transform(y, float(t[0]), float(t[1]))
    spread = np.linalg.norm(q[best, :2] - q[best, :2].mean(0), axis=1)
    return out, best, {"votes": k, "sane": int(sane.sum()), "inliers": int(best.sum()),
                       "spread_cm_median": float(np.median(spread) * 100),
                       "yaw_spread_deg": float(np.degrees(np.std(wrap(yaw[best] - y))))}


def alignment_votes(mdb: MatchDB, label_of, c2w_of, depth_of, K_of, s: float, min_inliers: int = 30, log_every: int = 0):
    """Votes for the transforms between groups of images that live in different frames (passes, or a test clip and
    the scene). For every verified pair whose images carry different labels la < lb (label_of(name) -> int, or None to
    skip the image), the pose of b in a's camera from LiDAR-lifted PnP both ways gives
        G = c2w_of(a) @ (b in a) @ inv(c2w_of(b)),
    which maps lb's frame onto la's. Returns {(la, lb): [(G, name_a, name_b, inliers), ...]} and rejection counts."""
    votes: dict[tuple[int, int], list] = {}
    why = {"tried": 0, "depth": 0, "pnp": 0, "disagree": 0}
    for i1, i2, m in mdb.verified(min_inliers):
        n1, n2 = mdb.name_of[i1], mdb.name_of[i2]
        l1, l2 = label_of(n1), label_of(n2)
        if l1 is None or l2 is None or l1 == l2:
            continue
        if l1 > l2:
            n1, n2, l1, l2, m = n2, n1, l2, l1, m[:, ::-1]
            i1, i2 = mdb.id_of[n1], mdb.id_of[n2]
        why["tried"] += 1
        uva, uvb = mdb.keypoints(i1)[m[:, 0]], mdb.keypoints(i2)[m[:, 1]]
        M, info = relative_pose_both_ways(depth_of(n1), K_of(n1), depth_of(n2), K_of(n2), s, uva, uvb, one_way_ok=True)
        if M is None:
            why[info] += 1
            continue
        votes.setdefault((l1, l2), []).append((c2w_of(n1) @ M @ np.linalg.inv(c2w_of(n2)), n1, n2, info))
        if log_every and why["tried"] % log_every == 0:
            print(f"  {why['tried']} cross pairs tried, {sum(len(v) for v in votes.values())} votes", flush=True)
    return votes, why
