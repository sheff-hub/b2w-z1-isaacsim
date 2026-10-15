"""Summarise fetch_box.py result files: success rate, efficiency, robustness, failure causes.

Runs on the host (plain Python, no Isaac):

    python3 evaluation/summarize_fetch.py renders/fetch/<run>/results.json [more.json ...]

Metrics (the Phase 5 plan's three):
  success     strict: box lifted >= 8 cm off the table and still held after carrying it;
              picked+held: off the table (>= 2 cm) and in the gripper (the plan's question:
              "did it pick up the object without dropping it?"). Wilson 95% intervals.
  efficiency  episode time, locomotion time, path length / straight-line distance
  robustness  falls (tilt > 30 deg), peak body tilt while driving
Failures are classified by the first stage that went wrong.
"""

import json
import math
import statistics as st
import sys


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def classify(x):
    s = x["stages"]
    if not s["locomotion"]["arrived"]:
        return "did not reach the table"
    if s["descend"]["pos_err_cm"] > 3:
        return "arm did not reach the grasp pose"
    if s["descend"].get("box_moved_cm", 0) > 2:
        return "box knocked while descending"
    if s["close"]["gripper_rad"] > -0.1:
        return "jaws closed on nothing"
    if s["carry"]["box_to_grasp_cm"] > 6:
        return "box slipped out while lifting"
    return "held, but lifted < 8 cm"


def row(name, v, unit=""):
    return f"  {name:28s} mean {st.mean(v):7.2f}{unit}  median {st.median(v):7.2f}  min {min(v):7.2f}  max {max(v):7.2f}"


runs = []
for f in sys.argv[1:]:
    runs += [dict(x, file=f) for x in json.load(open(f))]
n = len(runs)
ok = [x for x in runs if x["success"]]
lo, hi = wilson(len(ok), n)
loc = [x["stages"]["locomotion"] for x in runs]
print(f"{n} episodes from {len(sys.argv) - 1} file(s)")
print(f"SUCCESS     {len(ok)}/{n} = {100 * len(ok) / n:.0f}%  (95% CI {100 * lo:.0f}-{100 * hi:.0f}%)   strict: lifted >= 8 cm and held")
held = [x for x in runs if x["stages"]["carry"]["box_lift_cm"] >= 2 and x["stages"]["carry"]["box_to_grasp_cm"] <= 6]
hlo, hhi = wilson(len(held), n)
print(f"PICKED+HELD {len(held)}/{n} = {100 * len(held) / n:.0f}%  (95% CI {100 * hlo:.0f}-{100 * hhi:.0f}%)   off the table (>= 2 cm) and in the gripper")
dl = [x for x in runs if "delivered" in x]
if dl:
    dok = sum(x["delivered"] for x in dl)
    picked = sum(1 for x in dl if x["stages"]["carry"]["box_lift_cm"] >= 2 and x["stages"]["carry"]["box_to_grasp_cm"] <= 6)
    print(f"DELIVERED   {dok}/{len(dl)} carried to the drop zone and still held ({dok}/{picked} of the picked+held)")
print("EFFICIENCY")
print(row("episode time (s)", [x["time_s"] for x in runs]))
print(row("locomotion time (s)", [l["time_s"] for l in loc]))
def loc_row(label, key):  # Nav2-driven episodes do not record every scripted-route field
    vals = [l[key] for l in loc if key in l]
    print(row(label, vals) if vals else f"  {label:28s} n/a (not recorded by these runs)")


loc_row("path / straight line", "path_ratio")
print(f"  reached the table            {sum(l['arrived'] for l in loc)}/{n}")
print("ROBUSTNESS")
print(f"  falls (tilt > 30 deg)        {sum(x['fell'] for x in runs)}/{n}")
loc_row("peak tilt driving (deg)", "max_tilt_deg")
print("ACCURACY")
if all("dock_err_base_cm" in x["stages"] for x in runs):
    print(row("dock error forward (cm)", [x["stages"]["dock_err_base_cm"][0] for x in runs]))
    print(row("dock error sideways (cm)", [x["stages"]["dock_err_base_cm"][1] for x in runs]))
print(row("grasp-point error (cm)", [x["stages"]["descend"]["pos_err_cm"] for x in runs]))
if ok:
    print(row("lift, successes (cm)", [x["stages"]["carry"]["box_lift_cm"] for x in ok]))
fails = {}
for x in runs:
    if not x["success"]:
        fails.setdefault(classify(x), []).append(f"{x['file'].split('/')[-1]}:{x['episode']}")
print("FAILURES")
for k, v in sorted(fails.items(), key=lambda kv: -len(kv[1])):
    print(f"  {len(v):3d}  {k}   ({', '.join(v)})")
