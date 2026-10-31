"""Summarise fetch_box.py result files: success rate, efficiency, robustness, failure causes.

Runs on the host (plain Python, no Isaac):

    python3 evaluation/summarize_fetch.py renders/fetch/<run>/results.json [more.json ...]

Metrics (the Phase 5 plan's three):
  success     strict: box lifted >= 8 cm off the table and still held after carrying it;
              picked+held: off the table (>= 2 cm) and in the gripper (the plan's question:
              "did it pick up the object without dropping it?"). Wilson 95% intervals.
  efficiency  episode time, locomotion time, path length / straight-line distance
  robustness  falls (tilt > 30 deg), peak body tilt while driving
  placed      shelf scenes: the box carried to the table and left on it, upright, near its spot
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
    if s.get("moveit", {}).get("failed_at"):
        return f"MoveIt found no plan ({s['moveit']['failed_at']})"
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


def summarize_doors(runs):
    """The door task (scenes/hospital_doors.yaml): walked to the named door, lever turned (latch), door pushed open,
    robot in the room."""
    n = len(runs)

    def rate(name, k, note):
        lo, hi = wilson(k, n)
        print(f"{name:11s} {k}/{n} = {100 * k / n:.0f}%  (95% CI {100 * lo:.0f}-{100 * hi:.0f}%)   {note}")

    D = [x["stages"].get("door", {}) for x in runs]
    rate("SUCCESS", sum(bool(x.get("success")) for x in runs), "in the room behind the named door")
    rate("UNLATCHED", sum(bool(d.get("unlatched")) for d in D), "the lever turned past the latch")
    rate("OPENED", sum(bool(x.get("door_opened")) for x in runs), "pushed open >= 45 deg holding the lever")
    vl = [x["stages"]["vln"] for x in runs if "vln" in x["stages"]]
    if vl:
        print(f"DUALVLN     walk ended within 1 m of the named door {sum(bool(v.get('arrived')) for v in vl)}/{len(vl)}; "
              f"nearest the named one {sum(bool(v.get('right_target')) for v in vl)}/{len(vl)}")
    for name, vals, unit in (("docking error (cm)", [math.hypot(*d["dock_err_cm"]) for d in D if d.get("dock_err_cm")], ""),
                             ("grasp error (cm)", [d["grasp_err_cm"] for d in D if "grasp_err_cm" in d], ""),
                             ("lever turned (deg)", [d["lever_turned_deg"] for d in D if "lever_turned_deg" in d], ""),
                             ("door pushed to (deg)", [d["push"]["door_deg"] for d in D if "push" in d], ""),
                             ("push time (s)", [d["push"]["time_s"] for d in D if "push" in d], ""),
                             ("lever as the jaws open (deg)", [d["release"]["lever_deg"] for d in D if "release" in d], ""),
                             ("hand knocked at let-go (cm)", [d["release"]["hand_swing_cm"] for d in D if "release" in d], ""),
                             ("arm peak speed then (rad/s)", [max(d["release"]["arm_peak_rad_s"]) if isinstance(d["release"]["arm_peak_rad_s"], list)
                                                              else d["release"]["arm_peak_rad_s"] for d in D if "release" in d], ""),
                             ("door moved, let-go (deg)", [d["hinge_deg"] - d["release"]["door_deg"] for d in D
                                                                if "release" in d and "hinge_deg" in d], ""),
                             ("episode time (s)", [x["time_s"] for x in runs if "time_s" in x], "")):
        if vals:
            print(row(name, vals, unit))
    print(f"  falls (tilt > 30 deg)        {sum(bool(x.get('fell')) for x in runs)}/{n}")
    fails = [(x["file"].split("/")[-1], x["episode"], x["stages"].get("door", {}).get("failed_at") or
              ("not in the room" if x.get("door_opened") else "not opened")) for x in runs if not x.get("success")]
    if fails:
        print("FAILURES")
        for f in fails:
            print(f"  {f[0]}:{f[1]}  {f[2]}")


runs = []
for f in sys.argv[1:]:
    runs += [dict(x, file=f) for x in json.load(open(f))]
if runs and all(x.get("door_task") for x in runs):
    summarize_doors(runs)
    sys.exit(0)
n = len(runs)
ok = [x for x in runs if x["success"]]
lo, hi = wilson(len(ok), n)
loc = [x["stages"]["locomotion"] for x in runs]
print(f"{n} episodes from {len(sys.argv) - 1} file(s)")
print(f"SUCCESS     {len(ok)}/{n} = {100 * len(ok) / n:.0f}%  (95% CI {100 * lo:.0f}-{100 * hi:.0f}%)   strict: lifted >= 8 cm "
      f"(from a shelf: out of it) and held")
def picked_held(x):  # off the table and in the gripper; from a shelf: out of the shelf and in the gripper
    c = x["stages"]["carry"]
    up = c["box_out_cm"] > 3 if "box_out_cm" in c else c["box_lift_cm"] >= 2
    return up and c["box_to_grasp_cm"] <= 6


held = [x for x in runs if picked_held(x)]
hlo, hhi = wilson(len(held), n)
print(f"PICKED+HELD {len(held)}/{n} = {100 * len(held) / n:.0f}%  (95% CI {100 * hlo:.0f}-{100 * hhi:.0f}%)   off the table (>= 2 cm) and in the gripper")
dl = [x for x in runs if "delivered" in x]
if dl:
    dok = sum(x["delivered"] for x in dl)
    picked = sum(1 for x in dl if picked_held(x))
    print(f"DELIVERED   {dok}/{len(dl)} carried to the drop zone and still held ({dok}/{picked} of the picked+held)")
pl = [x for x in runs if x.get("place_task")]
if pl:
    pok = [x for x in pl if x.get("placed")]
    plo, phi = wilson(len(pok), len(pl))
    print(f"PLACED      {len(pok)}/{len(pl)} = {100 * len(pok) / len(pl):.0f}%  (95% CI {100 * plo:.0f}-{100 * phi:.0f}%)   "
          f"out of the shelf, onto the table, upright ({len(pok)}/{len(ok)} of the picked)")
    if pok:
        print(row("box off its spot (cm)", [x["stages"]["place"]["off_spot_cm"] for x in pok]))
        print(row("box tilt on table (deg)", [x["stages"]["place"]["tilt_deg"] for x in pok]))
    for x in pl:
        if x["success"] and not x.get("placed"):
            p_ = x["stages"].get("place", {})
            why = (f"MoveIt found no plan ({p_['moveit']['failed_at']})" if p_.get("moveit", {}).get("failed_at") else
                   "did not reach the table" if not x["stages"].get("place_drive", {}).get("locomotion", {}).get("arrived", True)
                   else f"box not on the table (off spot {p_.get('off_spot_cm')} cm, tilt {p_.get('tilt_deg')} deg, "
                        f"height {p_.get('height_err_cm')} cm)")
            print(f"  not placed: {x['file'].split('/')[-1]}:{x['episode']}  {why}")
vl = [x for x in runs if "vln" in x["stages"]]
if vl:
    arr = sum(bool(x["stages"]["vln"].get("arrived")) for x in vl)
    rt = [x for x in vl if "right_target" in x["stages"]["vln"]]
    print(f"DUALVLN     arrived (within 1 m) {arr}/{len(vl)}" + (f"; ended nearest the named one {sum(x['stages']['vln']['right_target'] for x in rt)}/{len(rt)}" if rt else ""))
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
