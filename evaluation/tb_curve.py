"""Pull a scalar out of an RSL-RL tensorboard run, so runs can be compared at
MATCHING iterations rather than by eye.

    python3 tools/tb_curve.py <run_dir> [more_dirs...] --tag Metrics/ee_pose/position_error
"""
import argparse
import glob
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ap = argparse.ArgumentParser()
ap.add_argument("dirs", nargs="+")
ap.add_argument("--tag", default="Metrics/ee_pose/position_error")
ap.add_argument("--at", default="500,1000,2000,3000,4000,5000,6000,7000,8000")
ap.add_argument("--csv", default=None)
args = ap.parse_args()

marks = [int(v) for v in args.at.split(",")]
rows = {}
for d in args.dirs:
    ev = sorted(glob.glob(os.path.join(d, "events.out.tfevents.*")))
    if not ev:
        print(f"{d}: no event file")
        continue
    acc = EventAccumulator(ev[-1], size_guidance={"scalars": 0})
    acc.Reload()
    if args.tag not in acc.Tags()["scalars"]:
        print(f"{d}: tag not found. available sample: "
              f"{[t for t in acc.Tags()['scalars'] if 'ee_pose' in t or 'position' in t][:5]}")
        continue
    pts = {e.step: e.value for e in acc.Scalars(args.tag)}
    name = os.path.basename(d.rstrip("/"))
    rows[name] = pts
    print(f"{name}: {len(pts)} points, steps {min(pts)}..{max(pts)}")

print(f"\n{args.tag}")
print(f"{'iteration':>10s}" + "".join(f"{n[-28:]:>30s}" for n in rows))
for m in marks:
    line = f"{m:>10d}"
    for n, pts in rows.items():
        near = [s for s in pts if abs(s - m) <= 25]
        line += f"{(f'{pts[min(near, key=lambda s: abs(s - m))]:.4f}' if near else '-'):>30s}"
    print(line)

if args.csv:
    import csv
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iteration"] + list(rows))
        allsteps = sorted(set().union(*[set(p) for p in rows.values()]))
        for s in allsteps:
            w.writerow([s] + [rows[n].get(s, "") for n in rows])
    print("wrote", args.csv)
