#!/usr/bin/env python3
"""Can the arm grasp a door's lever? Asks MoveIt (no simulator) before a single episode, like shelf_reach.py.

    python3 run.py moveit start
    python3 moveit/door_reach.py                          # scenes/hospital_doors.yaml, door b, the scene's parking pose
    python3 moveit/door_reach.py --backs 0.55,0.65,0.75 --sides -0.7,-0.8,-0.9 --pitches 0,20,40,60

The robot stands where fetch_box.py parks it (heading into the door, `dock` of the scene file: back from the leaf,
along it from the hinge), at ETH's standing height. The jaws close round the lever's bar in the vertical plane: the
approach pitched `pitch` deg below level (0: straight at the door, 90: from above), the closing axis perpendicular
to it in that plane (flip: rolled 180 deg). /approach looks for a pre-grasp from which the straight approach to the
bar works, with the leaf, the walls beside the opening, the robot's body and the floor in the scene. Y = found.
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
sys.path.insert(0, str(ROOT / "moveit"))
from moveit_link import BODY_BOXES  # noqa: E402
from shelf_reach import L00, ROOT_Z, quat  # noqa: E402

GRIP_UP, PRE = 0.006, 0.12  # as fetch_box.py (DOOR_GRIP_UP, DOOR_PRE)


def post(url, path, body):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())


def grip_rot(pitch_deg, yaw_deg, flip):
    """Columns: approach (stator x), stator y, closing axis (stator z) -- in the robot's frame (x ahead)."""
    p, y = math.radians(pitch_deg), math.radians(yaw_deg)
    a = np.array([math.cos(p), 0.0, -math.sin(p)])
    c = np.array([math.sin(p), 0.0, math.cos(p)]) * (-1 if flip else 1)
    R = np.stack([a, np.cross(c, a), c], axis=1)
    cz, sz = math.cos(y), math.sin(y)
    return np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="hospital_doors")
    ap.add_argument("--door", default="b")
    ap.add_argument("--backs", default="", help="comma-separated m (default: the scene's dock.back)")
    ap.add_argument("--sides", default="", help="comma-separated m (default: the scene's dock.side)")
    ap.add_argument("--pitches", default="0,15,30,45,60,75,90")
    ap.add_argument("--yaws", default="0", help="the approach turned about the vertical, deg (+: to the left)")
    ap.add_argument("--grip_y", default="", help="where along the bar, asset frame y (default: the scene's grip)")
    ap.add_argument("--url", default="http://127.0.0.1:5810")
    a = ap.parse_args()
    s = yaml.safe_load((ROOT / "scenes" / f"{a.scene}.yaml").read_text())
    D, o = s["door"], np.array(s["doors"][a.door]["origin"])
    backs = [float(v) for v in a.backs.split(",")] if a.backs else [s["dock"]["back"]]
    sides = [float(v) for v in a.sides.split(",")] if a.sides else [s["dock"]["side"]]
    pitches = [float(v) for v in a.pitches.split(",")]
    yaws = [float(v) for v in a.yaws.split(",")]
    gy = [float(v) for v in a.grip_y.split(",")] if a.grip_y else [D["grip"][1]]
    x0, x1, y0, y1 = D["leaf"]
    print(f"door {a.door} at {o.tolist()}; columns: pitch, flip (Y = straight approach found)")
    print(f"{'back':>5} {'side':>5} {'gripy':>6} {'yaw':>4}  " + " ".join(f"{p:>3.0f}{f}" for p in pitches for f in "+-"))
    for back in backs:
        for side in sides:
            rx, ry = o[0] - back, o[1] + side
            l00w = np.array([rx + L00[0], ry, ROOT_Z + L00[2]])

            def w2l(p):
                return np.asarray(p, dtype=float) - l00w  # heading +x: the link00 frame is the world's, shifted

            objs = [{"id": "leaf", "pose": list(w2l([o[0] + (x0 + x1) / 2, o[1] + (y0 + y1) / 2, o[2] + 1.12])) + [1, 0, 0, 0],
                     "size": [x1 - x0, y1 - y0, 2.12]}]
            for n, ya, yb in (("wall_s", o[1] + y0 - 2.5, o[1] + y0 - 0.005), ("wall_n", o[1] + y1 + 0.005, o[1] + y1 + 2.5)):
                objs.append({"id": n, "pose": list(w2l([o[0] + 0.04, (ya + yb) / 2, 1.5])) + [1, 0, 0, 0], "size": [0.10, yb - ya, 3.0]})
            for n, c, size in BODY_BOXES:
                objs.append({"id": n, "pose": list(np.array(c) - L00) + [1, 0, 0, 0], "size": list(size)})
            objs.append({"id": "floor", "pose": [0, 0, -L00[2] - ROOT_Z - 0.05, 1, 0, 0, 0], "size": [6, 6, 0.1]})
            post(a.url, "/scene", {"objects": objs, "attached": []})
            for g in gy:
                for yaw in yaws:
                    row = ""
                    for pitch in pitches:
                        for flip in (False, True):
                            R = grip_rot(pitch, yaw, flip)
                            q = quat(R)
                            fin = w2l(o + np.array([D["grip"][0], g, D["grip"][2]])) + R[:, 2] * GRIP_UP
                            r = post(a.url, "/approach", {"start": [0, 0.15, -0.15, 0, 0, 0],
                                                          "pre": list(fin - PRE * R[:, 0]) + q, "final": list(fin) + q,
                                                          "gripper": 0.0, "approach_gripper": -1.4, "tries": 24, "enough": 1})
                            row += "    Y" if r.get("ok") else "    ."
                    print(f"{back:5.2f} {side:5.2f} {g:6.3f} {yaw:4.0f}  {row}")
    post(a.url, "/scene", {"objects": [], "attached": []})


if __name__ == "__main__":
    main()
