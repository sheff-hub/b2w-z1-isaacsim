"""Compare two fetch_box result files episode by episode (same seed and random starts), e.g. the
scripted route vs --nav2:

    python3 evaluation/compare_nav2.py renders/fetch_ik_fix/results_val24.json renders/fetch_nav2/results_val24.json
"""

import json
import sys

import numpy as np


def stats(path):
    R = json.load(open(path))
    out = []
    for r in R:
        st = r["stages"]
        loco = st.get("locomotion", {})
        nav2 = st.get("nav2", {})
        nav_t = nav2.get("time_s", 0.0) if nav2 else 0.0
        by_nav2 = loco.get("by") == "nav2"  # Nav2 docked: its locomotion entry already covers the Nav2 drive
        dockd = st.get("nav2_dock", {})
        out.append({
            "ok": bool(r.get("success")), "total_s": r.get("time_s", np.nan),
            "drive_s": loco.get("time_s", np.nan) + (0.0 if by_nav2 else nav_t),
            "driven_m": (nav2.get("driven_m", 0.0) + dockd.get("driven_m", 0.0)) if by_nav2 else
                        loco.get("path_m", np.nan) + (nav2.get("driven_m", 0.0) if nav2 else 0.0),
            "nav2_docked": bool(dockd.get("success")),
            "dock_cm": float(np.hypot(*st.get("dock_err_base_cm", [np.nan, np.nan]))),
            "side": st.get("side", {}).get("chosen"), "touch": bool(loco.get("touching_after")),
            "nav2": nav2.get("status") if nav2 else None, "fell": bool(r.get("fell")),
        })
    return out


A, B = stats(sys.argv[1]), stats(sys.argv[2])
names = [sys.argv[1].split("/")[-2], sys.argv[2].split("/")[-2]]
print(f"{'':22s} {names[0]:>18s} {names[1]:>18s}")
for key, fmt in (("ok", "{:.0f}/{n}"), ("nav2_docked", "{:.0f}/{n}"), ("fell", "{:.0f}"), ("touch", "{:.0f}")):
    a, b = sum(x[key] for x in A), sum(x[key] for x in B)
    print(f"{key:22s} {fmt.format(a, n=len(A)):>18s} {fmt.format(b, n=len(B)):>18s}")
for key in ("total_s", "drive_s", "driven_m", "dock_cm"):
    a, b = np.nanmean([x[key] for x in A]), np.nanmean([x[key] for x in B])
    print(f"{key + ' (mean)':22s} {a:18.2f} {b:18.2f}")
n = min(len(A), len(B))
same_side = sum(A[i]["side"] == B[i]["side"] for i in range(n))
print(f"{'same side chosen':22s} {same_side}/{n}")
print(f"{'nav2 status':22s} " + str({s: sum(x['nav2'] == s for x in B) for s in {x['nav2'] for x in B}}))
print("\nper episode: ok / drive s / driven m / dock cm / side")
for i in range(n):
    a, b = A[i], B[i]
    print(f"  ep {i:2d}  {int(a['ok'])} {a['drive_s']:5.1f} {a['driven_m']:5.2f} {a['dock_cm']:4.1f} {a['side']:5s} | "
          f"{int(b['ok'])} {b['drive_s']:5.1f} {b['driven_m']:5.2f} {b['dock_cm']:4.1f} {b['side']:5s} {b['nav2']}")
