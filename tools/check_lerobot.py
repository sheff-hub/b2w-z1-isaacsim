"""Load a converted dataset with the official LeRobotDataset (lerobot 0.3.3) and print one sample: proves the
format is what VLA trainers read. Needs the separate venv (lerobot pulls torch; the converter venv stays small):

    HF_HUB_OFFLINE=1 python3 tools/check_lerobot.py \\
        datasets/lerobot/<name> [--frame 400]
"""

import argparse

from lerobot.datasets.lerobot_dataset import LeRobotDataset

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--frame", type=int, default=None, help="global frame index (default: the middle)")
args = ap.parse_args()
# 0.2 s of state history and a 1 s action chunk, the way a VLA reads "last states" and "future actions"
ds = LeRobotDataset("local/" + args.root.rstrip("/").split("/")[-1], root=args.root,
                    delta_timestamps={"observation.state": [-0.2, -0.1, 0.0], "action": [i / 10 for i in range(10)]})
print(f"episodes {ds.num_episodes}, frames {ds.num_frames}, fps {ds.fps}, tasks {len(ds.meta.tasks)}")
s = ds[args.frame if args.frame is not None else ds.num_frames // 2]
for k, v in s.items():
    print(f"  {k:32s} {tuple(v.shape) if hasattr(v, 'shape') else v!r} {getattr(v, 'dtype', '')}")
print("  phase:", ds.meta.info["phases"][int(s["label.phase"])])
