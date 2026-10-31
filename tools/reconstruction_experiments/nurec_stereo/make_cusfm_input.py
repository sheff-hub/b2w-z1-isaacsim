"""Record3D "EXR + JPG" export -> cuSFM input (NVIDIA NuRec stereo workflow, step "ROS bag -> cuSFM format").

    python make_cusfm_input.py <EXR_RGBD dir> <workspace> [--every 1]

NVIDIA's workflow takes a stereo rosbag and converts it with `rosbag_to_mapping_data`, using cuVSLAM's poses as the
initial ego-motion. A phone has no stereo pair and no cuVSLAM; the stand-ins are:
  - images: every --every-th phone frame, hard-linked to <workspace>/images/iphone/fNNNNN.jpg (one PINHOLE camera,
    median ARKit intrinsics; cuSFM wants a fixed calibrated camera);
  - initial poses (EGO_MOTION): ARKit, made continuous: ARKit's relocalisation jumps are removed by replacing the
    motion of the jump frames with the motion of the frame before (an odometry like cuVSLAM's odom pose, which is
    what cusfm_cli uses by default); roll and pitch stay ARKit's (gravity).
    No LiDAR floor pinning and no image-based correction here -- that is cuSFM's job.
  - world frame: Z-up (x, -z, y of ARKit), metres; camera_to_world in the optical (OpenCV) convention, as in
    pyCuSFM's r2b example.
Writes <workspace>/images/frames_meta.json.
"""
import argparse, json, os
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("ws"); ap.add_argument("--every", type=int, default=1)
a = ap.parse_args(); src, ws = Path(a.src), Path(a.ws)
m = json.load(open(src / "metadata.json")); P = np.array(m["poses"]); KK = np.array(m["perFrameIntrinsicCoeffs"])
ts = np.array(m["frameTimestamps"]); N = len(P); W, H = m["w"], m["h"]
R = Rotation.from_quat(P[:, :4]).as_matrix(); t = P[:, 4:]
# ---- de-snap, gravity kept: x, y, z and heading are re-integrated from ARKit's per-frame motion with the jump
#      frames replaced by the frame before; roll and pitch stay ARKit's absolute values (a VIO keeps gravity, and
#      integrating full 6-DoF steps lets a 1-2 deg tilt error at a jump turn into metres of height drift)
T = np.tile(np.eye(4), (N, 1, 1)); T[:, :3, :3] = R; T[:, :3, 3] = t
fwd = -R[:, :, 2]; yaw = np.arctan2(fwd[:, 0], fwd[:, 2])           # heading about ARKit's +Y (up)
def roty(a):
    c, s_ = np.cos(a), np.sin(a); M = np.zeros(np.shape(a) + (3, 3)); M[..., 0, 0] = c; M[..., 0, 2] = s_
    M[..., 2, 0] = -s_; M[..., 2, 2] = c; M[..., 1, 1] = 1; return M
dt_ = np.diff(t, axis=0); loc = np.einsum("nji,nj->ni", roty(yaw[:-1]), dt_)   # step in the previous heading frame
dyaw = (np.diff(yaw) + np.pi) % (2 * np.pi) - np.pi
step = np.linalg.norm(dt_, axis=1)
big = np.where(step > 0.04)[0]; events = [b for k, b in enumerate(big) if k == 0 or b - big[k - 1] > 30]
bad = np.zeros(N - 1, bool)
for k in events:
    hi = k
    while hi < N - 1 and step[hi] > max(0.012, 2.5 * np.median(step[max(0, k - 120):max(1, k - 5)])): hi += 1
    bad[k:max(hi, k + 1)] = True
for k in np.where(bad)[0]:
    j = k - 1
    while j > 0 and bad[j]: j -= 1
    loc[k] = loc[j]; dyaw[k] = dyaw[j]
yaw2 = np.r_[yaw[0], yaw[0] + np.cumsum(dyaw)]
t2 = np.vstack([t[0], t[0] + np.cumsum(np.einsum("nij,nj->ni", roty(yaw2[:-1]), loc), axis=0)])
T2 = T.copy(); T2[:, :3, :3] = roty(yaw2 - yaw) @ R; T2[:, :3, 3] = t2
print(f"de-snapped {bad.sum()} frame steps at {len(events)} jumps ({np.round(ts[np.array(events) + 1], 1)} s); "
      f"end moved {np.linalg.norm(T2[-1, :3, 3] - T[-1, :3, 3]):.2f} m; camera height range {np.ptp(t2[:, 1]):.2f} m")
A = np.eye(4); A[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
C = A @ T2 @ np.diag([1, -1, -1, 1])                     # Z-up world, OpenCV camera
# ---- images + frames_meta.json
img = ws / "images" / "iphone"; img.mkdir(parents=True, exist_ok=True)
fx, fy, cx, cy = np.median(KK, axis=0)
frames = []
for i in range(0, N, a.every):
    dst = img / f"f{i:05d}.jpg"
    if not dst.exists(): os.link(src / "rgb" / f"{i}.jpg", dst)
    rv = Rotation.from_matrix(C[i, :3, :3]).as_rotvec(); ang = np.linalg.norm(rv); ax = rv / ang if ang > 1e-12 else np.array([0, 0, 1.0])
    frames.append({"id": str(i), "camera_params_id": "0", "timestamp_microseconds": str(int(round(ts[i] * 1e6)) + 1_000_000),
                   "image_name": f"iphone/f{i:05d}.jpg",
                   "camera_to_world": {"axis_angle": {"x": ax[0], "y": ax[1], "z": ax[2], "angle_degrees": float(np.degrees(ang))},
                                       "translation": {"x": C[i, 0, 3], "y": C[i, 1, 3], "z": C[i, 2, 3]}},
                   "synced_sample_id": str(i)})
meta = {"keyframes_metadata": frames, "initial_pose_type": "EGO_MOTION", "camera_params_id_to_session_name": {"0": "0"},
        "camera_params_id_to_camera_params": {"0": {
            "sensor_meta_data": {"sensor_id": 0, "sensor_type": "CAMERA", "sensor_name": "iphone", "frequency": round(m["fps"] / a.every),
                                 "sensor_to_vehicle_transform": {"axis_angle": {"x": 0, "y": 0, "z": 1, "angle_degrees": 0},
                                                                 "translation": {"x": 0, "y": 0, "z": 0}}},
            "calibration_parameters": {"image_width": W, "image_height": H,
                                       "camera_matrix": {"data": [fx, 0, cx, 0, fy, cy, 0, 0, 1]},
                                       "distortion_coefficients": {"data": [0, 0, 0, 0, 0]},
                                       "projection_matrix": {"data": [fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0], "row_count": 3, "column_count": 4}},
            "camera_projection_model_type": "PINHOLE"}}}
json.dump(meta, open(ws / "images" / "frames_meta.json", "w"), indent=1)
np.savez(ws / "arkit_desnapped_poses.npz", c2w=C, K=KK, ts=ts, every=a.every)
print(f"{len(frames)} frames -> {ws / 'images'} (camera fx {fx:.1f} cx {cx:.1f} cy {cy:.1f})")
