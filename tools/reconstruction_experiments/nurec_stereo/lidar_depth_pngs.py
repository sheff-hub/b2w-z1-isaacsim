"""Depth images for nvblox's fuse_cusfm (stand-in for the FoundationStereo step of NVIDIA's NuRec stereo workflow).

    python lidar_depth_pngs.py <EXR_RGBD dir> <workspace> [--frames-meta cusfm/kpmap/keyframes/frames_meta.json]

NVIDIA runs FoundationStereo on the stereo pair of every cuSFM keyframe and writes depth/<camera>/<name>.png at the
colour resolution. A phone has no stereo pair, so the phone's own LiDAR depth (192x256, metres) is used instead:
pixels on depth edges (neighbour jump > 4% of depth; LiDAR "flying pixels") and outside 0.2-5 m are set to 0,
then the map is upsampled to the colour resolution with nearest neighbour (no invented in-between depths) and saved
as uint16 millimetres (nvblox's default depth scale 1/1000).
"""
import argparse, json, os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
from pathlib import Path
import cv2, numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("ws")
ap.add_argument("--frames-meta", default="cusfm/kpmap/keyframes/frames_meta.json"); ap.add_argument("--edge", type=float, default=0.04)
a = ap.parse_args(); src, ws = Path(a.src), Path(a.ws)
meta = json.load(open(ws / a.frames_meta)); m = json.load(open(src / "metadata.json")); W, H = m["w"], m["h"]
n = 0
for kf in meta["keyframes_metadata"]:
    name = kf["image_name"]; i = int(Path(name).stem[1:])
    d = cv2.imread(str(src / "depth" / f"{i}.exr"), cv2.IMREAD_UNCHANGED)[..., 2].astype(np.float32)
    k = np.ones((3, 3), np.uint8); jump = cv2.dilate(d, k) - cv2.erode(d, k)
    d[(jump > a.edge * d) | (d < 0.2) | (d > 5.0) | ~np.isfinite(d)] = 0
    up = cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)
    out = ws / "depth" / Path(name).with_suffix(".png"); out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.round(up * 1000).astype(np.uint16)); n += 1
print(f"wrote {n} depth PNGs (uint16 mm, {W}x{H}) to {ws / 'depth'}")
