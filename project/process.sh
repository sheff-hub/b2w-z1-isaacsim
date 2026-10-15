#!/bin/bash
# Follow a running fetch (live.sh fetch / fetchdemo, fetch_box.py) from another terminal:
#   ./process.sh            live: new events as they happen (Ctrl-C to quit; the run goes on)
#   ./process.sh --all      the whole current run from its start, then live
#   ./process.sh --once     print what happened so far and exit
# Reads renders/progress/progress.jsonl, which fetch_box.py rewrites at every start (progress.py).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 - "$HERE/renders/progress/progress.jsonl" "$@" <<'EOF'
import json, os, sys, time

path, args = sys.argv[1], sys.argv[2:]
ALL, ONCE = "--all" in args or "--once" in args, "--once" in args
B, G, R, Y, C, D, X = ("\033[1m", "\033[32m", "\033[31m", "\033[33m", "\033[36m", "\033[2m", "\033[0m") \
    if sys.stdout.isatty() else ("",) * 7
PHASES = {"walk": "DualVLN is walking the robot to the table", "backoff": "backing off from the table",
          "nav2": "Nav2 is driving", "nav2_dock": "Nav2's docking server is driving to the pick point",
          "dock": "scripted final approach to the pick point",
          "pick": "picking up the box", "deliver": "carrying the box to the drop zone", "idle": "idle"}
STAGES = {"crouch": "crouching", "unfold": "unfolding the arm", "pre_grasp": "arm above the box",
          "open": "opening the gripper", "descend": "lowering onto the box", "close": "closing the gripper",
          "lift": "lifting", "standup": "standing up", "carry_move": "arm to the carry pose", "carry_hold": "checking the grip"}
state = {"n": None}


def fmt_cm(v):
    return "?" if v is None else f"{v:.0f} cm" if isinstance(v, (int, float)) else str(v)


def show(e):
    ev, clock = e["ev"], time.strftime("%H:%M:%S", time.localtime(e["t"]))
    pre = f"{D}{clock}  sim {e['sim']:7.1f}s{X}  "
    if ev == "sim_ready":
        state["n"] = e.get("episodes")
        print(pre + f"{B}simulation started{X}  (base controller {e['loco']}, Nav2 {'on' if e['nav2'] else 'off'}, "
              f"DualVLN {'on' if e['vln'] else 'off'}{', delivery' if e['deliver'] else ''}"
              f"{', interactive' if e.get('interactive') else ''})")
    elif ev == "episode":
        print(pre + f"{B}{C}{'task' if e.get('n') is None else 'episode'} {e['ep'] + 1}{'/' + str(e['n']) if e.get('n') else ''} started{X} at "
              f"({e['start'][0]:.1f}, {e['start'][1]:.1f}), heading {e['start'][2]:.0f} deg -- task: {e.get('task')}")
    elif ev == "phase":
        ph = e["phase"]
        name, _, sub = ph.partition(":")
        if sub:
            print(pre + f"   pick: {STAGES.get(sub, sub)}")
        else:
            print(pre + f"{B}> {PHASES.get(name, name)}{X}")
    elif ev == "beat":
        bits = [f"pos ({e['x']:.2f}, {e['y']:.2f})", f"{e['speed']:.2f} m/s", f"table {e['table_m']:.2f} m"]
        if "goal_m" in e:
            bits.insert(0, f"{Y}{e['goal_m']:.2f} m to the goal, heading off by {e['goal_yaw_err_deg']:+.0f} deg{X}")
        print(pre + f"{D}   ... " + ", ".join(bits) + X)
    elif ev == "vln_done":
        ok = e.get("arrived")
        print(pre + f"{G if ok else Y}DualVLN finished{X}: {e.get('reason')} at {e.get('table_edge_m')} m from the table "
              f"after {e.get('time_s')} s ({'arrived' if ok else 'stopped short of 1.0 m; parking continues'})")
    elif ev == "nav2_plan":
        g = e["goal"]
        if e.get("stage") == "docking at the pick point":
            print(pre + f"Nav2 docking at the pick point ({g[0]}, {g[1]}) heading {g[2]} deg")
        else:
            print(pre + f"Nav2 planned both sides: " + ", ".join(f"{k} {v} m" for k, v in e["paths"].items())
                  + f" -> {B}{e['side']}{X}; {e.get('stage', 'goal')} ({g[0]}, {g[1]}) heading {g[2]} deg")
    elif ev == "nav2_done":
        st = e.get("status")
        col = G if st == "succeeded" else R
        what = "docking" if e.get("stage") == "docking" else "Nav2"
        if e.get("stage") == "docking":
            col = G if e.get("success") else R
            st = "docked" if e.get("success") else f"docking failed ({e.get('status')}, code {e.get('error_code')})"
        print(pre + f"{col}{what} {st}{X}" + (f" in {e.get('time_s')} s, drove {e.get('driven_m')} m, "
              f"{fmt_cm(e.get('err_cm', e.get('err_E_cm')))} from the goal, heading off {e.get('yaw_err_deg')} deg"
              if "time_s" in e else ""))
    elif ev == "dock_done":
        print(pre + f"{G}at the pick point{X} (table on the {e.get('side')}), parking error {e.get('dock_err_cm')} cm")
    elif ev == "pick_done":
        c = e.get("carry") or {}
        ok = e.get("success")
        print(pre + f"{G if ok else R}pick {'succeeded' if ok else 'FAILED'}{X}: box {c.get('box_lift_cm')} cm above the "
              f"table, {c.get('box_to_grasp_cm')} cm from the grasp point")
    elif ev == "deliver_done":
        print(pre + f"{G if e.get('arrived') else R}delivered{X}" + ("" if e.get("arrived") else " (NOT arrived)"))
    elif ev == "result":
        ok = e["success"]
        print(pre + f"{B}{G if ok else R}episode {e['ep'] + 1}: {'SUCCESS' if ok else 'FAIL'}{X}"
              + (f" in {e['time_s']} s" if e.get("time_s") is not None else "")
              + (", FELL" if e.get("fell") else "") + "\n")
    elif ev == "summary":
        print(pre + f"{B}all done: {e['success']}/{e['n']} succeeded, {e['falls']} falls{X}")
    elif ev == "bye":
        print(pre + "simulation closed")
    sys.stdout.flush()


while not os.path.exists(path):
    if ONCE:
        sys.exit(f"no run yet ({path})")
    print(f"waiting for a run to start ({path}) ...", end="\r")
    time.sleep(1)
f = open(path)
if not ALL:
    lines = f.readlines()  # live: show only the latest status, then follow
    for line in lines[-3:]:
        try:
            show(json.loads(line))
        except Exception:
            pass
ino = os.stat(path).st_ino
while True:
    line = f.readline()
    if line:
        try:
            show(json.loads(line))
        except json.JSONDecodeError:
            pass
        continue
    if ONCE:
        break
    time.sleep(0.5)
    try:  # a new run truncates the file: start over
        if os.stat(path).st_size < f.tell() or os.stat(path).st_ino != ino:
            f.close()
            f, ino = open(path), os.stat(path).st_ino
            print(f"{B}--- a new run started ---{X}")
    except FileNotFoundError:
        pass
EOF
