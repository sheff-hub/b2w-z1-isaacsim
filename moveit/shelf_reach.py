#!/usr/bin/env python3
"""Can the arm get a box out of a scene's shelf? Asks MoveIt (no simulator) before you run a single episode.

    python3 run.py moveit start
    python3 moveit/shelf_reach.py hospital_reception            # the scene file's shelf, its box positions
    python3 moveit/shelf_reach.py hospital_hall --level green      # a shelf with a box per tray
    python3 moveit/shelf_reach.py hospital_reception --pitches 30,40,45,50 --gaps 0.05,0.09,0.13

For each docking gap (robot's hips to the shelf face) and approach pitch, and box yaws from -24 to +24 deg, it builds
the planning scene the pick would (the scene file's shelf boxes, the robot's body, the floor) and asks the bridge's
/approach for a pre-grasp posture from which the straight approach to the grasp works -- the step the pick fails at
when the shelf is out of reach. Y = found. The robot stands where fetch_box.py would dock it (the shelf on its
right, the box `abeam` ahead), at ETH's standing height; the gripper as the scene sets it (jaw_flip).
"""
import argparse
import json
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "project"))
from moveit_link import BODY_BOXES  # noqa: E402  (plain module: requests only)

L00 = np.array([0.0563, 0.0, 0.0871])  # link00 in the B2W base frame (robots_usd/b2w_z1_gripper.usd)
ROOT_Z = 0.762                         # base height standing (ETH controller, measured in the simulator)
HIP_HALF_WIDTH, BOX_H, GRASP_RAISE, PRE_DIST = 0.287, 0.08, 0.015, 0.16  # as fetch_box.py


def post(url, path, body):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())


def quat(R):
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return [w, x, y, z]


def grasp_rot(yaw, pitch_deg, flip):
    """fetch_box.grasp_quat as a matrix: approach pitched down, jaws closing horizontally (flip: rolled 180 deg)."""
    ph = math.radians(pitch_deg)
    a, c = np.array([math.cos(ph), 0.0, -math.sin(ph)]), np.array([0.0, -1.0 if flip else 1.0, 0.0])
    cz, sz = math.cos(yaw), math.sin(yaw)
    return np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.stack([a, np.cross(c, a), c], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", help="scenes/<name>.yaml of layout shelf")
    ap.add_argument("--shelf", default="", help="which shelf (default: the first)")
    ap.add_argument("--pitches", default="", help="comma-separated (default: the scene's)")
    ap.add_argument("--gaps", default="", help="comma-separated, m (default: the scene's gap and +4 cm, as Nav2 docks)")
    ap.add_argument("--level", default="", help="a shelf with levels: the box (colour) whose tray and settings to check")
    ap.add_argument("--top", type=float, default=None, help="the tray the box stands on (default: the scene's)")
    ap.add_argument("--crouch", type=float, default=0.0, help="m the body is lowered (ETH's crouch: 0.06)")
    ap.add_argument("--flip", choices=("scene", "yes", "no"), default="scene", help="gripper rolled 180 deg")
    ap.add_argument("--abeam", type=float, default=None, help="m the box is ahead of the robot's centre (default: the scene's)")
    ap.add_argument("--url", default="http://127.0.0.1:5810")
    a = ap.parse_args()
    s = yaml.safe_load((ROOT / "scenes" / f"{a.scene}.yaml").read_text())
    if "shelves" in s:
        name = a.shelf or next(iter(s["shelves"]))
        sh = {**s.get("shelf_defaults", {}), **s["shelves"][name]}
    else:
        sh = s["shelf"]
    if a.level:  # that tray's settings; its stance as a height (ETH: 1 = 5.8 cm lower, -0.75 = 1.8 cm higher)
        lv = sh["levels"][a.level]
        sh = {**sh, **lv}
        st_ = lv.get("stance", 0.0)
        a.crouch = a.crouch or (0.058 * st_ if st_ >= 0 else 0.024 * st_)
    face = sh.get("face", -1)
    yaw = -face * math.pi / 2  # the shelf on the robot's right: facing +y from its -x side, -y from its +x side
    pitches = [float(v) for v in a.pitches.split(",")] if a.pitches else \
        (sh["pitch"] if isinstance(sh.get("pitch"), list) else [sh.get("pitch", 45)])
    gap0 = sh.get("gap", 0.05)
    gaps = [float(v) for v in a.gaps.split(",")] if a.gaps else [gap0, gap0 + 0.04]
    flip, abeam = bool(sh.get("jaw_flip", False)) if a.flip == "scene" else a.flip == "yes", sh.get("abeam", 0.08)
    abeam = abeam if a.abeam is None else a.abeam
    top = a.top if a.top is not None else sh["top"]
    root_z = ROOT_Z - a.crouch
    yaws = list(range(-24, 25, 6))
    print(f"{a.scene}: box at x {sh['box_x']}, tray {top}, crouch {a.crouch}; abeam {abeam}, jaw_flip {flip}. Y = straight approach found")
    print(f"{'gap':>5} {'pitch':>5}  " + " ".join(f"{d:+4d}" for d in yaws) + "   (box yaw, deg)")
    for gap in gaps:
        for pitch in pitches:
            box_y = sh["y"] if "y" in sh else sh["slots_y"][len(sh["slots_y"]) // 2]
            cy, sy = math.cos(yaw), math.sin(yaw)
            rx, ry = sh["near_x"] + face * (HIP_HALF_WIDTH + gap), box_y - sy * abeam
            l00w = np.array([rx + cy * L00[0], ry + sy * L00[0], root_z + L00[2]])

            def w2l(p):  # world -> link00 (the base frame turned by yaw, z up)
                d = np.asarray(p) - l00w
                return np.array([cy * d[0] + sy * d[1], -sy * d[0] + cy * d[1], d[2]])

            objs = []
            for n, (x0, y0, z0, x1, y1, z1) in sh["collision"].items():
                c = w2l([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
                objs.append({"id": n, "pose": list(c) + [1, 0, 0, 0], "size": [y1 - y0, x1 - x0, z1 - z0]})
            for n, c, size in BODY_BOXES:
                objs.append({"id": n, "pose": list(np.array(c) - L00) + [1, 0, 0, 0], "size": list(size)})
            objs.append({"id": "floor", "pose": [0, 0, -L00[2] - root_z - 0.05, 1, 0, 0, 0], "size": [6, 6, 0.1]})
            post(a.url, "/scene", {"objects": objs, "attached": []})
            fin = w2l([sh["box_x"], box_y, top + BOX_H / 2 + GRASP_RAISE])
            row = ""
            for d in yaws:
                R = grasp_rot(-math.pi / 2 + math.radians(d), pitch, flip)
                q = quat(R)
                r = post(a.url, "/approach", {"start": [0, 0.15, -0.15, 0, 0, 0], "pre": list(fin - PRE_DIST * R[:, 0]) + q,
                                              "final": list(fin) + q, "gripper": 0.0, "approach_gripper": -1.4,
                                              "tries": 16, "enough": 1})
                row += "    Y" if r.get("ok") else "    ."
            print(f"{gap:5.2f} {pitch:5.0f}  {row}")
    post(a.url, "/scene", {"objects": [], "attached": []})


if __name__ == "__main__":
    main()
