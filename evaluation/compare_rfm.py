"""Side-by-side table of two (or more) RFM evaluations.

    python3 tools/compare_rfm.py <tag> <tag> [...]

Reads logs/eval/{approach,approach_high,reach_map}_<tag>.json as written by
evaluation/eval_rfm.sh and prints one row per metric, one column per tag.
"""
import json
import statistics as st
import sys
from pathlib import Path

EVAL = Path(__file__).resolve().parent.parent / "repos/robot_lab/logs/eval"  # paths.robot_lab in config.yaml


def load(name):
    p = EVAL / name
    return json.loads(p.read_text()) if p.exists() else None


def med(rows, key, scale=1.0):
    v = [r[key] for r in rows if key in r]
    return st.median(v) * scale if v else float("nan")


def approach_rows(d, label):
    if d is None:
        return {}
    rows = d["cases"]
    held = [r for r in rows if r["closest_m"] < 0.03]
    out = {
        f"{label}: reached (<10 cm)": f"{sum(r['closest_m'] < 0.10 for r in rows)}/{len(rows)}",
        f"{label}: median closest, cm": f"{med(rows, 'closest_m', 100):.1f}",
        f"{label}: falls (tilt>75 deg)": f"{sum(r['fell'] for r in rows)}/{len(rows)}",
        f"{label}: max pitch (sit-back), median deg": f"{med(rows, 'max_pitch_deg'):+.0f}",
        f"{label}: lowest body, m": f"{min(r['min_height_m'] for r in rows):.2f}",
        f"{label}: target-to-shoulder at end, m": f"{med(rows, 'shoulder_dist_m'):.2f}",
        f"{label}: driving in reverse, median %": f"{med(rows, 'reverse_frac', 100):.0f}",
        # shaking only means something for cases that actually reached the target;
        # the rest never had a hand "on target" and read zero
        f"{label}: arm shaking on target, rad/s": f"{med(held, 'hold_arm_rms'):.2f}" if held else "-",
        f"{label}: hand speed on target, cm/s": f"{med(held, 'hold_hand_speed', 100):.1f}" if held else "-",
        f"{label}: orientation error, rad": f"{med(rows, 'ori_rad'):.2f}",
    }
    return out


def map_rows(d):
    if d is None:
        return {}
    import math

    errs = [e for row in d["err_m"] for e in row]
    ok = [e for e in errs if e <= d["tol"]]
    intr = d.get("intrusion_m", [])
    return {
        "reach map: reached (<5 cm)": f"{len(ok)}/{len(errs)}",
        "reach map: median error on reached, cm": f"{st.median(ok) * 100:.1f}" if ok else "-",
        "reach map: arm into robot volume": f"{sum(v > 0 for v in intr)}/{len(intr)}" if intr else "-",
    }


tags = sys.argv[1:]
cols = {}
for t in tags:
    r = {}
    r.update(approach_rows(load(f"approach_{t}.json"), "mid 0.45 m"))
    r.update(approach_rows(load(f"approach_high_{t}.json"), "high 1.0 m"))
    r.update(map_rows(load(f"reach_map_{t}.json")))
    cols[t] = r

keys = []
for r in cols.values():
    for k in r:
        if k not in keys:
            keys.append(k)
w = max(len(k) for k in keys) + 2
print(f"{'':{w}s}" + "".join(f"{t:>16s}" for t in tags))
for k in keys:
    print(f"{k:{w}s}" + "".join(f"{cols[t].get(k, '-'):>16s}" for t in tags))
