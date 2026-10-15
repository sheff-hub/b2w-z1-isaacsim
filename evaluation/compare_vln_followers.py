"""Compare the DualVLN walk with different trajectory followers (fetch_box.py --vln_follower pp | follow |
goal), episode by episode on the same starts, and make side-by-side videos.

    python3 evaluation/compare_vln_followers.py [--videos]
Reads renders/vln_follow/<follower>/results.json (+ each episode's DualVLN log.json); writes
renders/vln_follow_compare/comparison.md and, with --videos, side_by_side_ep<k>.mp4 there (pp | follow | goal).
"""

import argparse
import glob
import json
import os
import subprocess

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "renders", "vln_follow")
OUT = ROOT + "_compare"  # the runs' folder belongs to the container's user (uid 1234): write next to it
FOLLOWERS = [("pp", "pure pursuit (ours)"), ("follow", "Nav2 FollowPath (A)"), ("goal", "Nav2 NavigateToPose (B)")]


def vln_log(rec):
    """The DualVLN log of an episode (container path -> host path)."""
    p = rec["stages"]["vln"].get("log", "").replace("/workspace", os.path.join(ROOT, "..", ".."))  # container -> repo
    try:
        return json.load(open(os.path.join(p, "log.json")))
    except (OSError, ValueError):
        return {}


def stats(follower):
    path = os.path.join(ROOT, follower, "results.json")
    if not os.path.exists(path):
        return None
    out = []
    for r in json.load(open(path)):
        v, lg = r["stages"]["vln"], vln_log(r)
        mot = np.array(lg.get("motion", []), dtype=float).reshape(-1, 6)
        clear = mot[:, 5] if len(mot) else np.array([9.0])
        out.append({
            "arrived": bool(v.get("arrived")), "reason": v.get("reason"), "table_m": v.get("table_edge_m"),
            "walk_s": v.get("time_s"), "walk_m": v.get("path_m"), "requests": v.get("requests"),
            "mean_speed": lg.get("mean_speed"), "stops": lg.get("stops"), "rms_acc": lg.get("rms_acc"),
            "rms_ang_acc": lg.get("rms_ang_acc"),
            "bumper_close_s": round(float((clear < 0.8).sum()) * 0.02, 1),  # time with something < 0.8 m ahead
            "min_clear_m": round(float(clear.min()), 2),
            "docked": bool(r["stages"].get("nav2_dock", {}).get("success")), "success": bool(r.get("success")),
            "total_s": r.get("time_s"),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", action="store_true")
    args = ap.parse_args()
    S = {f: stats(f) for f, _ in FOLLOWERS}
    lines = ["# DualVLN walk: who follows the trajectory", "",
             "Same 6 starts and instruction for each follower; Nav2 parks and the arm picks in all of them.", ""]
    hdr = "| | " + " | ".join(name for f, name in FOLLOWERS if S[f]) + " |"
    lines += [hdr, "|---" * (1 + sum(1 for f, _ in FOLLOWERS if S[f])) + "|"]

    def row(label, fn):
        cells = []
        for f, _ in FOLLOWERS:
            if S[f]:
                cells.append(fn(S[f]))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    n = lambda e: len(e)  # noqa: E731
    row("DualVLN arrived (< 1 m from the table)", lambda e: f"{sum(x['arrived'] for x in e)}/{n(e)}")
    row("ended by STOP / bumper / timeout", lambda e: "/".join(str(sum(x['reason'] == k for x in e)) for k in ("STOP", "BLOCKED", "timeout")))
    row("distance to the table at the end (m, mean)", lambda e: f"{np.mean([x['table_m'] for x in e]):.2f}")
    row("walk time (s sim, mean)", lambda e: f"{np.mean([x['walk_s'] for x in e]):.1f}")
    row("walk distance (m, mean)", lambda e: f"{np.mean([x['walk_m'] for x in e]):.2f}")
    row("mean speed (m/s)", lambda e: f"{np.mean([x['mean_speed'] or 0 for x in e]):.2f}")
    row("stops while walking (mean)", lambda e: f"{np.mean([x['stops'] or 0 for x in e]):.1f}")
    row("rms acceleration (m/s², smoothness)", lambda e: f"{np.mean([x['rms_acc'] or 0 for x in e]):.2f}")
    row("rms angular acceleration (rad/s²)", lambda e: f"{np.mean([x['rms_ang_acc'] or 0 for x in e]):.2f}")
    row("closest obstacle ahead (m, min)", lambda e: f"{min(x['min_clear_m'] for x in e):.2f}")
    row("Nav2 docked / picked", lambda e: f"{sum(x['docked'] for x in e)}/{n(e)} / {sum(x['success'] for x in e)}/{n(e)}")
    row("episode time (s, mean)", lambda e: f"{np.mean([x['total_s'] for x in e]):.1f}")
    lines += ["", "Per episode (arrived, end reason, m from the table, walk s):", ""]
    k_max = max(len(e) for e in S.values() if e)
    for k in range(k_max):
        cells = []
        for f, _ in FOLLOWERS:
            e = S[f]
            if e and k < len(e):
                x = e[k]
                cells.append(f"{'yes' if x['arrived'] else 'no'}, {x['reason']}, {x['table_m']}, {x['walk_s']}")
        lines.append(f"- ep {k}: " + " | ".join(cells))
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, "comparison.md")
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", out)
    if args.videos:
        font = subprocess.run(["fc-match", "-f", "%{file}", "DejaVu Sans:bold"], capture_output=True, text=True).stdout
        for k in range(k_max):
            vids = []
            for f, name in FOLLOWERS:
                m = sorted(glob.glob(os.path.join(ROOT, f, f"fetch_seed*_ep{k}_*.mp4")))
                if m:
                    vids.append((m[0], name))
            if len(vids) < 2:
                continue
            durs = [float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", v],
                                         capture_output=True, text=True).stdout) for v, _ in vids]
            D = max(durs)
            filt, labels = [], []
            for i, ((v, name), d) in enumerate(zip(vids, durs)):
                filt.append(f"[{i}:v]scale=640:360,tpad=stop_mode=clone:stop_duration={D - d + 0.1},"
                            f"drawtext=fontfile={font}:text='{name}':x=12:y=12:fontsize=24:fontcolor=white:box=1:"
                            f"boxcolor=black@0.55:boxborderw=6[v{i}]")
                labels.append(f"[v{i}]")
            filt.append("".join(labels) + f"hstack=inputs={len(vids)},trim=duration={D}[out]")
            dst = os.path.join(OUT, f"side_by_side_ep{k}.mp4")
            cmd = ["ffmpeg", "-v", "error", "-y"] + sum([["-i", v] for v, _ in vids], []) + \
                ["-filter_complex", ";".join(filt), "-map", "[out]", "-r", "30", "-c:v", "libx264", "-crf", "24",
                 "-pix_fmt", "yuv420p", dst]
            subprocess.run(cmd, check=True)
            print("wrote", dst)


if __name__ == "__main__":
    main()
