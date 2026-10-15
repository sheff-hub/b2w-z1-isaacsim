#!/usr/bin/env python3
"""What the launchers need to know about a scene file (scenes/<name>.yaml), as shell variable assignments.

    eval "$(python3 isaac/scene_info.py warehouse_three_tables)"     # or 1 / 3 (the two warehouse scenes), or a path

Sets SCENE_NAME, SCENE_FILE (host path), SCENE_FILE_CONTAINER (/workspace/scenes/...), SCENE_TASK, SCENE_LAYOUT,
SCENE_NAV2_MAP, SCENE_START ("x,y,yaw") and SCENE_DEMO_STARTS ("x,y,yaw; ...").
"""
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALIASES = {"1": "warehouse_one_table", "3": "warehouse_three_tables"}


def main():
    try:
        import yaml
    except ImportError:
        sys.exit("isaac/scene_info.py needs PyYAML on the host:  python3 -m pip install --user pyyaml")
    arg = sys.argv[1] if len(sys.argv) > 1 else "1"
    arg = ALIASES.get(arg, arg)
    path = Path(arg) if arg.endswith(".yaml") else ROOT / "scenes" / f"{arg}.yaml"
    path = path.resolve()
    if not path.exists():
        names = ", ".join(sorted(p.stem for p in (ROOT / "scenes").glob("*.yaml")))
        sys.exit(f"no scene file {path} (scenes: {names})")
    try:
        container = "/workspace/" + str(path.relative_to(ROOT))
    except ValueError:
        sys.exit(f"{path}: scene files must be inside the repository (scenes/), the container sees only that")
    s = yaml.safe_load(path.read_text())

    def pose(p):
        return ",".join(f"{v:g}" for v in p)

    out = {
        "SCENE_NAME": s.get("name", path.stem),
        "SCENE_FILE": str(path),
        "SCENE_FILE_CONTAINER": container,
        "SCENE_TASK": s["task"],
        "SCENE_LAYOUT": s.get("layout", "one_table"),
        "SCENE_NAV2_MAP": s.get("nav2_map", ""),
        "SCENE_START": pose(s["robot"]["live_start"]),
        "SCENE_DEMO_STARTS": "; ".join(pose(p) for p in s["robot"].get("demo_starts", [s["robot"]["live_start"]])),
    }
    for k, v in out.items():
        print(f"{k}={shlex.quote(str(v))}")


if __name__ == "__main__":
    main()
