"""Convert raw fetch recordings (project/recorder.py) to a LeRobot v2.1 dataset.

Runs on the host, in the small venv that has pyarrow (the Isaac image does not):

    python3 tools/raw_to_lerobot.py \\
        --raw datasets/raw/<run> [--raw <run2> ...] \\
        --out datasets/lerobot/<name> [--only_success] [--only_right_table] [--no_depth]

Layout (LeRobot v2.1, codebase_version "v2.1" -- what openpi's pi0 / pi0.5 fine-tuning reads):
  meta/info.json, meta/tasks.jsonl, meta/episodes.jsonl, meta/episodes_stats.jsonl
  data/chunk-000/episode_XXXXXX.parquet
  videos/chunk-000/observation.images.{front,wrist}/episode_XXXXXX.mp4   (copied, h264)

Features of a recorder format-2 episode (the agreed VLA spec: vision, language, proprioception, action, labels):
  VISION
    observation.images.front / .wrist      video, RGB: front 240 x 320; wrist 180 x 320 (RealSense D405, 16:9)
                                           or 240 x 320 (older bare camera) -- meta.json["cameras"]
    observation.depth.front / .wrist       uint16 at half the image size, millimetres (0 = no return / < 7 cm on the D405)
  LANGUAGE
    task (task_index -> meta/tasks.jsonl)  the task sentence; episodes.jsonl also has nav_instruction
  PROPRIOCEPTION
    observation.state       63: joint pos (23), joint vel (23), base lin vel (3), base ang vel (3),
                            projected gravity (3), base height (1), EE pose in the base frame (7)
    observation.joint_pos / .joint_vel   23 each (the same numbers, by group)
    observation.prev_action 11: the previous frame's action (zeros at frame 0)
  ACTION
    action                  11: the WHOLE-BODY COMMAND -- base velocity [vx, vy, wz], EE target pose in
                            the base frame (pos 3, quat wxyz 4; smoothed, --smooth_ee), gripper target
    action.arm_joints       7: arm joint targets + gripper (an alternative arm action)
  LABELS (not model inputs)
    label.phase             int: recorder.PHASES
    label.plan              20: the planned path ahead, 10 (x, y) points every 0.2 m in the base frame (NaN = none)
    observation.base_pose_world   3: x, y, yaw -- evaluation only
  Longer histories ("last states") and future chunks ("future states / actions") are neighbouring frames:
  LeRobotDataset(..., delta_timestamps={"observation.state": [-0.1, 0], "action": [0, 0.1, ..., 4.9]}).
"""

import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

p = argparse.ArgumentParser()
p.add_argument("--raw", action="append", required=True, help="raw run directory (repeatable)")
p.add_argument("--out", required=True)
p.add_argument("--only_success", action="store_true", help="keep only episodes whose pick succeeded")
p.add_argument("--only_right_table", action="store_true",
               help="three-table scene: keep only episodes where DualVLN headed for the named table")
p.add_argument("--no_depth", action="store_true", help="leave the depth images out (smaller)")
p.add_argument("--robot_type", default="unitree_b2w_z1_gripper")
p.add_argument("--smooth_ee", type=int, default=5,
               help="centered moving average (frames, odd; 1 = off) on the EE target: it is the MEASURED next "
                    "EE pose, which shakes ~3 cm during the pick (pilot review plots)")
args = p.parse_args()

eps = []
for r in args.raw:
    for d in sorted(os.listdir(r)):
        full = os.path.join(r, d)
        if d.startswith("ep_") and os.path.exists(os.path.join(full, "meta.json")):
            meta = json.load(open(os.path.join(full, "meta.json")))
            if args.only_success and not meta.get("success"):
                continue
            if args.only_right_table and meta.get("vln_right_table") is False:
                continue
            eps.append((full, meta))
if not eps:
    raise SystemExit("no episodes found")
fmt = eps[0][1].get("format", 1)
assert all(m.get("format", 1) == fmt for _, m in eps), "mixed recorder formats: convert them separately"
CAMS = eps[0][1].get("cameras", {c: {"size_wh": [320, 240], "depth_wh": eps[0][1].get("depth", {}).get("size_wh", [160, 120])}
                                  for c in ("front", "wrist")})
assert all(m.get("cameras", CAMS) == CAMS for _, m in eps), \
    "different camera setups (e.g. bare wrist camera vs RealSense D405): convert them separately"
if os.path.exists(args.out):
    shutil.rmtree(args.out)
for sub in ("meta", "data/chunk-000", "videos/chunk-000/observation.images.front", "videos/chunk-000/observation.images.wrist"):
    os.makedirs(os.path.join(args.out, sub), exist_ok=True)

m0 = eps[0][1]
fps = m0["fps"]
an = m0["action_names"]
action_names = an["base"] + an["ee"] + ["gripper"]
if fmt >= 2:
    jn = m0["joint_names"]
    state_names = [f"{j}.pos" for j in jn] + [f"{j}.vel" for j in jn] + m0["base_names"] + m0["ee_names"]
else:
    state_names = m0["state_names"]


def smooth(ee, k):
    """Moving average of positions and (sign-aligned, renormalised) quaternions over k frames."""
    if k <= 1:
        return ee
    ee = ee.copy()
    q = ee[:, 3:7]
    for i in range(1, len(q)):  # q and -q are the same rotation: keep a continuous sign
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    pad = k // 2
    kern = np.ones(k) / k
    out = np.empty_like(ee)
    for c in range(7):
        x = np.pad(ee[:, c] if c < 3 else q[:, c - 3], pad, mode="edge")
        out[:, c] = np.convolve(x, kern, mode="valid")
    out[:, 3:7] /= np.linalg.norm(out[:, 3:7], axis=1, keepdims=True)
    return out


def stats(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    x = x.reshape(len(x), -1)
    return {"min": np.nanmin(x, 0).tolist(), "max": np.nanmax(x, 0).tolist(), "mean": np.nanmean(x, 0).tolist(),
            "std": np.nanstd(x, 0).tolist(), "count": [int(len(x))]}


def col(a):
    """A per-frame numpy array -> a pyarrow column (fixed-size lists for vectors / images)."""
    a = np.ascontiguousarray(a)
    if a.ndim == 1:
        return pa.array(a)
    flat = pa.array(a.reshape(-1))
    for size in reversed(a.shape[1:]):
        flat = pa.FixedSizeListArray.from_arrays(flat, size)
    return flat


tasks, task_rows, ep_rows, stat_rows = {}, [], [], []
index = 0
EP_KEYS = ("nav_instruction", "deliver_task", "success", "delivered", "vln_arrived", "vln_reason", "vln_right_table", "nav2_docked",
           "fell", "start", "box_start_w", "loco", "seed", "side", "target", "scene", "nav2", "vln_follower", "time_s")
for ei, (path, meta) in enumerate(eps):
    z = np.load(os.path.join(path, "data.npz"))
    n = len(z["t"])
    task = meta["task"]
    if task not in tasks:
        tasks[task] = len(tasks)
        task_rows.append({"task_index": tasks[task], "task": task})
    ee = smooth(z["action_ee"], args.smooth_ee)
    action = np.concatenate([z["action_base"], ee, z["action_arm"][:, 6:7]], axis=1).astype(np.float32)
    prev = np.vstack([np.zeros((1, action.shape[1]), np.float32), action[:-1]])
    cols = {}
    if fmt >= 2:
        state = np.concatenate([z["joint_pos"], z["joint_vel"], z["base"], z["ee_pose"]], axis=1).astype(np.float32)
        cols.update({"observation.state": state, "observation.joint_pos": z["joint_pos"], "observation.joint_vel": z["joint_vel"],
                     "observation.prev_action": prev})
        if not args.no_depth:
            cols["observation.depth.front"] = z["depth_front"]
            cols["observation.depth.wrist"] = z["depth_wrist"]
        cols["label.plan"] = z["plan"].reshape(n, -1).astype(np.float32)
    else:
        state = z["state"].astype(np.float32)
        cols.update({"observation.state": state, "observation.prev_action": prev})
    cols.update({"action": action, "action.arm_joints": z["action_arm"].astype(np.float32),
                 "label.phase": z["phase"].astype(np.int64), "observation.base_pose_world": z["world"].astype(np.float32),
                 "timestamp": (np.arange(n) / fps).astype(np.float32), "frame_index": np.arange(n, dtype=np.int64),
                 "episode_index": np.full(n, ei, dtype=np.int64), "index": np.arange(index, index + n, dtype=np.int64),
                 "task_index": np.full(n, tasks[task], dtype=np.int64)})
    table = pa.table({k: col(v) for k, v in cols.items()})
    pq.write_table(table, os.path.join(args.out, "data/chunk-000", f"episode_{ei:06d}.parquet"), compression="zstd")
    for cam in ("front", "wrist"):
        shutil.copyfile(os.path.join(path, f"{cam}.mp4"),
                        os.path.join(args.out, f"videos/chunk-000/observation.images.{cam}", f"episode_{ei:06d}.mp4"))
    ep_rows.append({"episode_index": ei, "tasks": [task], "length": n, "source": os.path.relpath(path, os.path.dirname(args.raw[0])),
                    **{k: meta.get(k) for k in EP_KEYS}})
    stat_rows.append({"episode_index": ei, "stats": {k: stats(v) for k, v in cols.items()
                                                      if not k.startswith("observation.depth")}})
    index += n

def video(cam):
    w, h = CAMS[cam]["size_wh"]
    return {"dtype": "video", "shape": [h, w, 3], "names": ["height", "width", "channels"],
            "info": {"video.height": h, "video.width": w, "video.codec": "h264", "video.pix_fmt": "yuv420p",
                     "video.is_depth_map": False, "video.fps": fps, "video.channels": 3, "has_audio": False,
                     **{f"camera.{k}": v for k, v in CAMS[cam].items() if k not in ("size_wh", "depth_wh")}}}



f32 = lambda names: {"dtype": "float32", "shape": [len(names)], "names": names}  # noqa: E731
features = {"observation.images.front": video("front"), "observation.images.wrist": video("wrist"),
            "observation.state": f32(state_names), "observation.prev_action": f32(action_names),
            "action": f32(action_names), "action.arm_joints": f32(an["arm"]),
            "label.phase": {"dtype": "int64", "shape": [1], "names": None},
            "observation.base_pose_world": f32(["x", "y", "yaw"])}
if fmt >= 2:
    jn = m0["joint_names"]
    features.update({"observation.joint_pos": f32(jn), "observation.joint_vel": f32(jn),
                     "label.plan": f32([f"plan_{a}{i}" for i in range(m0["plan"]["points"]) for a in "xy"])})
    if not args.no_depth:
        for cam in ("front", "wrist"):
            w, h = CAMS[cam]["depth_wh"]
            features[f"observation.depth.{cam}"] = {"dtype": "uint16", "shape": [h, w], "names": ["height", "width"],
                                                    "info": {"unit": "mm", "max_m": m0["depth"]["max_m"],
                                                             "min_m": CAMS[cam].get("min_depth_m", 0.0)}}
features.update({k: {"dtype": "int64" if k != "timestamp" else "float32", "shape": [1], "names": None}
                 for k in ("timestamp", "frame_index", "episode_index", "index", "task_index")})
info = {
    "codebase_version": "v2.1", "robot_type": args.robot_type, "total_episodes": len(eps), "total_frames": index,
    "total_tasks": len(tasks), "total_videos": 2 * len(eps), "total_chunks": 1, "chunks_size": 1000, "fps": fps,
    "splits": {"train": f"0:{len(eps)}"},
    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    "features": features,
    "phases": m0["phases"],
}
with open(os.path.join(args.out, "meta/info.json"), "w") as f:
    json.dump(info, f, indent=2)
for name, rows in (("tasks", task_rows), ("episodes", ep_rows), ("episodes_stats", stat_rows)):
    with open(os.path.join(args.out, f"meta/{name}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
size = sum(os.path.getsize(os.path.join(dp, fn)) for dp, _, fns in os.walk(args.out) for fn in fns)
print(f"wrote {args.out}: {len(eps)} episodes, {index} frames ({index / fps / 60:.1f} min), {len(tasks)} tasks, "
      f"{size / 2**20:.1f} MiB")
