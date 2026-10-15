#!/usr/bin/env python3
"""Read config.yaml (falling back to config.example.yaml) and print it as shell variable assignments.

    eval "$(python3 isaac/config.py --shell)"     # used by isaac/isaac_env.sh and the other launchers
    python3 isaac/config.py gpu.uuid               # one value

Paths are made absolute (relative to the repository root, ~ expanded). A variable already set in the environment
keeps its value, so ISAAC_GPU_UUID=... ./script.sh still overrides the file.
"""
import os
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# shell variable  <-  config key  (path: made absolute)
VARS = [
    ("ISAAC_GPU_UUID", "gpu.uuid", False),
    ("ISAACLAB_DIR", "paths.isaaclab", True),
    ("ROBOT_LAB_DIR", "paths.robot_lab", True),
    ("INTERNNAV_DIR", "paths.internnav", True),
    ("THREEDGRUT_DIR", "paths.threedgrut", True),
    ("CACHE_ROOT_5", "paths.isaac_cache", True),
    ("CACHE_ROOT_6", "paths.isaac_cache_6", True),
    ("NUREC_ROOT", "paths.nurec", True),
    ("HF_CACHE", "paths.hf_cache", True),
    ("ROS_DOMAIN_ID", "ros.domain_id", False),
    ("LIVESTREAM_ADDR", "livestream.address", False),
    ("LIVESTREAM_PORT", "livestream.port", False),
    ("IMAGE_ISAAC_SIM", "images.isaac_sim", False),
    ("IMAGE_ISAAC_SIM_6", "images.isaac_sim_6", False),
    ("IMAGE_ISAAC_LAB", "images.isaac_lab", False),
    ("IMAGE_NAV2", "images.nav2", False),
    ("IMAGE_DUALVLN", "images.dualvln", False),
    ("NAV2_CONTAINER", "containers.nav2", False),
    ("DUALVLN_CONTAINER", "containers.dualvln", False),
    ("LIVE_CONTAINER", "containers.live", False),
    ("DUALVLN_PORT", "dualvln.port", False),
    ("DUALVLN_QUANT", "dualvln.quant", False),
]


def load() -> dict:
    try:
        import yaml
    except ImportError:
        sys.exit("isaac/config.py needs PyYAML on the host:  python3 -m pip install --user pyyaml")
    for name in ("config.yaml", "config.example.yaml"):
        p = ROOT / name
        if p.exists():
            return yaml.safe_load(p.read_text()) or {}
    sys.exit(f"no config.yaml in {ROOT}: run ./setup.sh first")


def get(cfg: dict, key: str):
    cur = cfg
    for k in key.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def absolute(v: str) -> str:
    p = Path(os.path.expanduser(str(v)))
    return str(p if p.is_absolute() else (ROOT / p).resolve())


def main():
    cfg = load()
    if len(sys.argv) == 2 and sys.argv[1] != "--shell":
        v = get(cfg, sys.argv[1])
        print("" if v is None else v)
        return
    print(f"REPO_ROOT={shlex.quote(str(ROOT))}")
    for var, key, is_path in VARS:
        v = get(cfg, key)
        if v is None:
            continue
        v = absolute(v) if is_path else str(v)
        # keep a value the caller already exported
        print(f'{var}="${{{var}:-{v}}}"' if "'" not in v and '"' not in v and "$" not in v
              else f"{var}=${{{var}:-{shlex.quote(v)}}}")


if __name__ == "__main__":
    main()
