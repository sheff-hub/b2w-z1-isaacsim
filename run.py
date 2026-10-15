#!/usr/bin/env python3
"""One entry point for the project. Plain python3 on the host; the heavy parts run in containers.

    python3 run.py fetch                         the final pipeline, batch: DualVLN + Nav2 + pick (+ delivery)
    python3 run.py fetch --scene warehouse_one_table --no-vln --episodes 5 --deliver
    python3 run.py fetch --record demo_v5 --episodes 30    ... and record a LeRobot-ready dataset
    python3 run.py live fetch                    interactive: type tasks into a running simulator (project/live.sh)
    python3 run.py live goto_mine                one 6D whole-body policy: type a hand target
    python3 run.py train --task <gym id>         RL training (isaac/run_train.sh)
    python3 run.py script <path> [args]          any script of this repository in the Isaac Lab container
    python3 run.py dataset <raw run> [...] --out <name>    raw recordings -> LeRobot v2.1 (tools/raw_to_lerobot.py)
    python3 run.py reconstruct <scene>           phone photos -> USD scene (reconstruction/)
    python3 run.py nav2 start|stop [--scene S]   the Nav2 container with the scene's map
    python3 run.py vln start|stop                the DualVLN server
    python3 run.py scenes                        the scene files
    python3 run.py check                         what is installed (./setup.sh --check)

Machine settings: config.yaml. Scenes: scenes/*.yaml. Outputs: renders/, datasets/.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "isaac"))


def sh(cmd, env=None, check=True):
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    e = dict(os.environ)
    e.update(env or {})
    r = subprocess.run([str(c) for c in cmd], env=e, cwd=ROOT)
    if check and r.returncode != 0:
        sys.exit(r.returncode)
    return r.returncode


def config() -> dict:
    out = subprocess.run([sys.executable, str(ROOT / "isaac/config.py"), "--shell"], capture_output=True, text=True,
                         check=True).stdout
    vals = {}
    for line in out.splitlines():
        k, _, v = line.partition("=")
        v = v.strip('"')
        if v.startswith("${") and ":-" in v:  # NAME="${NAME:-value}": the environment wins
            v = os.environ.get(k, v.split(":-", 1)[1].rstrip("}"))
        vals[k] = v
    return vals


def scene(name: str) -> dict:
    out = subprocess.run([sys.executable, str(ROOT / "isaac/scene_info.py"), name], capture_output=True, text=True)
    if out.returncode:
        sys.exit(out.stderr.strip())
    vals = dict(line.split("=", 1) for line in out.stdout.splitlines())
    import shlex
    vals = {k: shlex.split(v)[0] if v else "" for k, v in vals.items()}
    import yaml
    vals["data"] = yaml.safe_load(Path(vals["SCENE_FILE"]).read_text())
    return vals


def container_running(name: str) -> bool:
    out = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True).stdout
    return name in out.split()


# ------------------------------------------------------------------------------------------------ services
def nav2_up(scene_name: str, cfg: dict, restart=False):
    sc = scene(scene_name)
    params = ROOT / "nav2/b2w_nav2_params.yaml"
    same_map = params.exists() and f"map {sc['SCENE_NAV2_MAP']})" in params.read_text().splitlines()[0]
    if container_running(cfg["NAV2_CONTAINER"]) and same_map and not restart:
        print(f"nav2 already running with map {sc['SCENE_NAV2_MAP']}")
        return
    sh([ROOT / "nav2/run_nav2.sh"], env={"NAV2_MAP": sc["SCENE_NAV2_MAP"]})
    time.sleep(5)


def vln_server_up(cfg: dict, quant=None):
    url = f"http://127.0.0.1:{cfg['DUALVLN_PORT']}/health"

    def up():
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            return False

    if up():
        print("DualVLN server is up")
        return
    log = ROOT / "vln/out/server.log"
    log.parent.mkdir(exist_ok=True)
    print(f"starting the DualVLN server ({quant or cfg['DUALVLN_QUANT']}); log: {log}")
    subprocess.Popen([str(ROOT / "vln/run_dualvln_server.sh"), "--quant", quant or cfg["DUALVLN_QUANT"]],
                     stdout=open(log, "w"), stderr=subprocess.STDOUT, cwd=ROOT, start_new_session=True)
    for _ in range(80):
        if up():
            print("DualVLN server is up")
            return
        time.sleep(3)
    sys.exit(f"the DualVLN server did not come up; see {log}")


# ------------------------------------------------------------------------------------------------ commands
def cmd_fetch(a, extra):
    cfg = config()
    sc = scene(a.scene)
    tag = a.name or time.strftime("%m%d-%H%M%S")
    out_dir = f"/workspace/renders/fetch/{tag}"
    args = ["project/fetch_box.py", "--device", "cpu", "--loco", a.loco, "--task", sc["SCENE_TASK"],
            "--episodes", str(a.episodes), "--seed", str(a.seed), "--out", f"{out_dir}/results.json",
            "--video_dir", out_dir, "--vln_out", f"{out_dir}/vln"]
    if sc["SCENE_LAYOUT"] == "multi_table":
        args += ["--target", a.target, "--task_variants"]
    if not a.no_nav2:
        nav2_up(a.scene, cfg)
        args += ["--nav2"]
    if not a.no_vln:
        vln_server_up(cfg)
        # multi-table scenes: the sentence is made per episode from the target; one table: say its name
        first = next(iter(sc["data"]["tables"]))
        args += ["--vln", f"Walk to the {first} table and stop in front of it.", "--vln_server",
                 f"http://127.0.0.1:{cfg['DUALVLN_PORT']}"]
        if not a.no_nav2:
            args += ["--vln_follower", "follow"]
    if a.deliver:
        args += ["--deliver"]
    elif a.deliver_frac is not None:
        args += ["--deliver_frac", str(a.deliver_frac)]
    rs = sc["data"]["robot"].get("random_start")
    if rs and not a.fixed_start:
        args += ["--random_start=" + ",".join(f"{v:g}" for v in rs)]
    else:
        args += ["--starts", sc["SCENE_START"]]
    if a.record:
        args += ["--record", f"/workspace/datasets/raw/{a.record}"]
    if a.video:
        args += ["--video", "--video_episodes", a.video]
    env = {"FETCH_SCENE": sc["SCENE_FILE_CONTAINER"], "CONTAINER_NAME": a.container or f"isaac-fetch-{tag}"}
    print(f"results, videos and DualVLN logs: renders/fetch/{tag}/" + (f"; recording: datasets/raw/{a.record}/" if a.record else ""))
    sh([ROOT / "isaac/run_script.sh", *args, *extra], env=env)
    res = ROOT / "renders/fetch" / tag / "results.json"
    if res.exists():
        sh([sys.executable, ROOT / "evaluation/summarize_fetch.py", res], check=False)


def cmd_live(a, extra):
    env = {"SCENE": a.scene} if a.scene else {}
    sh([ROOT / "project/live.sh", a.mode, *extra], env=env)


def cmd_train(a, extra):
    sh([ROOT / "isaac/run_train.sh", *extra])


def cmd_script(a, extra):
    sh([ROOT / "isaac/run_script.sh", a.path, *extra])


def cmd_dataset(a, extra):
    raws = []
    for r in a.raw:
        raws += ["--raw", str((ROOT / "datasets/raw" / r) if not os.path.isabs(r) else r)]
    sh([sys.executable, ROOT / "tools/raw_to_lerobot.py", *raws, "--out", ROOT / "datasets/lerobot" / a.out, *extra])


def cmd_reconstruct(a, extra):
    sh([sys.executable, ROOT / "reconstruction/reconstruct.py", *([a.scene] if a.scene else []), *extra])


def cmd_nav2(a, extra):
    if a.action == "stop":
        sh([ROOT / "nav2/run_nav2.sh", "stop"])
    else:
        nav2_up(a.scene, config(), restart=True)


def cmd_vln(a, extra):
    cfg = config()
    if a.action == "stop":
        sh(["docker", "stop", cfg["DUALVLN_CONTAINER"]], check=False)
    else:
        vln_server_up(cfg, a.quant)


def cmd_scenes(a, extra):
    import yaml
    for p in sorted((ROOT / "scenes").glob("*.yaml")):
        s = yaml.safe_load(p.read_text())
        print(f"{p.stem:26s} {s.get('layout', ''):12s} tables: {', '.join(s.get('tables', {}))}   map: {s.get('nav2_map')}")
    print("reconstruction scenes (photos -> USD):  python3 reconstruction/reconstruct.py --list-scenes")


def cmd_check(a, extra):
    sh([ROOT / "setup.sh", "--check", *extra], check=False)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="batch fetch episodes (the report's pipeline by default)")
    f.add_argument("--scene", default="warehouse_three_tables", help="scenes/<name>.yaml (default: the three tables)")
    f.add_argument("--episodes", type=int, default=1)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--target", default="auto", help="multi-table scenes: table/box, e.g. green/red, or auto")
    f.add_argument("--no-vln", action="store_true", help="no DualVLN walk: Nav2 (or the script) drives from the start")
    f.add_argument("--no-nav2", action="store_true", help="the scripted route instead of Nav2")
    f.add_argument("--deliver", action="store_true", help="always carry the box to the drop zone")
    f.add_argument("--deliver_frac", type=float, default=0.5, help="else: deliver in this share of episodes (0.5)")
    f.add_argument("--fixed-start", action="store_true", help="start at the scene's live_start, not a random pose")
    f.add_argument("--loco", default="eth", choices=("eth", "ours"), help="walking policy (eth: ETH's + our fine-tune)")
    f.add_argument("--record", default="", help="record every episode to datasets/raw/<name>")
    f.add_argument("--video", default="", help="film these episodes, e.g. 0,1")
    f.add_argument("--name", default="", help="output folder name under renders/fetch/ (default: a time stamp)")
    f.add_argument("--container", default="", help="container name (default: isaac-fetch-<name>)")
    f.set_defaults(fn=cmd_fetch)
    lv = sub.add_parser("live", help="interactive demos in an Isaac Sim window (project/live.sh)")
    lv.add_argument("mode", help="vln | fetch | fetchdemo | pick | goto_mine | goto_paper | eth | say | stop")
    lv.add_argument("--scene", default="", help="scene file for vln/fetch/fetchdemo/pick (default: one table)")
    lv.set_defaults(fn=cmd_live)
    tr = sub.add_parser("train", help="RL training / play (isaac/run_train.sh; all options pass through)")
    tr.set_defaults(fn=cmd_train)
    sc = sub.add_parser("script", help="run a repository script in the Isaac Lab container")
    sc.add_argument("path")
    sc.set_defaults(fn=cmd_script)
    ds = sub.add_parser("dataset", help="raw recordings -> LeRobot v2.1 (needs numpy, pyarrow, pandas, imageio)")
    ds.add_argument("raw", nargs="+", help="raw runs under datasets/raw/ (or absolute paths)")
    ds.add_argument("--out", required=True, help="name under datasets/lerobot/")
    ds.set_defaults(fn=cmd_dataset)
    rc = sub.add_parser("reconstruct", help="phone photos -> USD scene (reconstruction/reconstruct.py)")
    rc.add_argument("scene", nargs="?")
    rc.set_defaults(fn=cmd_reconstruct)
    nv = sub.add_parser("nav2", help="start/stop Nav2 with a scene's map")
    nv.add_argument("action", choices=("start", "stop"))
    nv.add_argument("--scene", default="warehouse_three_tables")
    nv.set_defaults(fn=cmd_nav2)
    vl = sub.add_parser("vln", help="start/stop the DualVLN server")
    vl.add_argument("action", choices=("start", "stop"))
    vl.add_argument("--quant", default=None, choices=("none", "8bit"))
    vl.set_defaults(fn=cmd_vln)
    sub.add_parser("scenes", help="list the scene files").set_defaults(fn=cmd_scenes)
    sub.add_parser("check", help="what is installed (./setup.sh --check)").set_defaults(fn=cmd_check)
    a, extra = p.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    a.fn(a, extra)


if __name__ == "__main__":
    main()
