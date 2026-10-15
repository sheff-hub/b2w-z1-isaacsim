"""Shared helpers: config loading, state tracking, subprocess and docker."""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import yaml


def expand(p) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(p)))).resolve()


def deep_merge(base: dict, over: dict) -> dict:
    """Merge `over` into `base` in place: nested dicts merge, everything else is replaced."""
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base


REPO = Path(__file__).resolve().parents[2]  # reconstruction/tools/common.py -> the repository


def repo_machine_config() -> dict:
    """The machine settings this script shares with the rest of the project, from the repository's config.yaml
    (config.example.yaml if there is none yet): the NuRec folder, the 3DGRUT checkout and the GPU."""
    for name in ("config.yaml", "config.example.yaml"):
        p = REPO / name
        if p.exists():
            rc = yaml.safe_load(p.read_text()) or {}
            break
    else:
        return {}

    def absolute(v):
        q = Path(os.path.expanduser(str(v)))
        return str(q if q.is_absolute() else (REPO / q).resolve())

    nurec = absolute(deep_get(rc, "paths.nurec", "scenes/nurec"))
    out = {"paths": {"dataset_root": f"{nurec}/mine/dataset", "scenes_root": f"{nurec}/mine"}, "gpu": {}}
    if deep_get(rc, "paths.threedgrut"):
        out["paths"]["threedgrut"] = absolute(deep_get(rc, "paths.threedgrut"))
    if deep_get(rc, "gpu.uuid"):
        out["gpu"]["uuid"] = str(deep_get(rc, "gpu.uuid"))
    return out


def load_config(path, *overlays) -> dict:
    """The base config, then each overlay (a scene file) merged on top of it. Machine settings left null are
    filled from the repository's config.yaml."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for section, values in repo_machine_config().items():
        for k, v in values.items():
            if deep_get(cfg, f"{section}.{k}") is None:
                cfg.setdefault(section, {})[k] = v
    for o in overlays:
        if o:
            with open(o) as f:
                deep_merge(cfg, yaml.safe_load(f) or {})
    return cfg


def deep_get(cfg: dict, dotted: str, default=None):
    cur = cfg
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return default if cur is None else cur


class Log:
    """Console + file logging. Stages are noisy on purpose: a run is long and
    unattended, so the log is the only record of what happened."""

    def __init__(self, path: Path | None = None):
        self.path = path
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.fh = open(path, "a", buffering=1)
        else:
            self.fh = None

    def __call__(self, *a):
        msg = " ".join(str(x) for x in a)
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        if self.fh:
            self.fh.write(line + "\n")

    def rule(self, title):
        self(f"{'=' * 8} {title} {'=' * max(0, 60 - len(title))}")


class State:
    """Records which stages finished, so a re-run resumes instead of redoing
    hours of COLMAP. Terminals die; this is not optional."""

    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text()) if path.exists() else {"stages": {}}

    def done(self, stage: str) -> bool:
        return self.data["stages"].get(stage, {}).get("ok", False)

    def mark(self, stage: str, **info):
        self.data["stages"][stage] = {"ok": True, "at": time.strftime("%Y-%m-%d %H:%M:%S"), **info}
        self.save()

    def get(self, stage: str, key: str, default=None):
        return self.data["stages"].get(stage, {}).get(key, default)

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2))


def run(cmd, log: Log, cwd=None, env=None, check=True) -> subprocess.CompletedProcess:
    """Run a command, streaming output into the log."""
    if isinstance(cmd, str):
        cmd = shlex.split(cmd)
    log(f"$ {' '.join(shlex.quote(c) for c in cmd)}")
    e = dict(os.environ)
    if env:
        e.update({k: str(v) for k, v in env.items()})
    p = subprocess.Popen(cmd, cwd=cwd, env=e, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail = []
    for line in p.stdout:
        line = line.rstrip()
        tail.append(line)
        if len(tail) > 400:
            tail.pop(0)
        if log.fh:
            log.fh.write(line + "\n")
    p.wait()
    if check and p.returncode != 0:
        log("!! command failed with exit code", p.returncode)
        for t in tail[-25:]:
            log("   |", t)
        raise SystemExit(f"stage failed: {' '.join(cmd[:3])} ... (exit {p.returncode})")
    return subprocess.CompletedProcess(cmd, p.returncode, "\n".join(tail), "")


def docker_gpu_flag(cfg: dict) -> str:
    """--gpus value. Pinning by UUID is deliberate: `all` would hand the job a
    GPU that may belong to another user on a shared machine."""
    uuid = deep_get(cfg, "gpu.uuid", "all")
    return "all" if str(uuid).lower() == "all" else f"device={uuid}"


def docker_run(cfg: dict, image: str, args: list[str], mounts: dict[str, str],
               log: Log, workdir: str | None = None) -> subprocess.CompletedProcess:
    cmd = ["docker", "run", "--rm", "--gpus", docker_gpu_flag(cfg),
           "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp"]
    for host, cont in mounts.items():
        cmd += ["-v", f"{host}:{cont}"]
    if workdir:
        cmd += ["-w", workdir]
    cmd += [image] + args
    return run(cmd, log)


def _smi(args: list[str], uuid: str | None) -> list[str]:
    cmd = ["nvidia-smi"] + args + ["--format=csv,noheader,nounits"]
    if uuid and uuid.lower() != "all":
        cmd += ["-i", uuid]  # the pinned card, not whichever nvidia-smi lists first
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return [l.strip() for l in out.splitlines() if l.strip()]


def gpu_free_gb(log: Log | None = None, uuid: str | None = None) -> float:
    try:
        total, used = (float(x) for x in _smi(["--query-gpu=memory.total,memory.used"], uuid)[0].split(","))
        return (total - used) / 1024.0
    except Exception:
        return 0.0


def gpu_users(uuid: str | None = None) -> list[str]:
    """'pid user GB command' for every process on the card (the shared machine's other jobs)."""
    rows = []
    try:
        for line in _smi(["--query-compute-apps=pid,used_memory"], uuid):
            pid, mib = (x.strip() for x in line.split(","))
            ps = subprocess.run(["ps", "-o", "user:32=,args=", "-p", pid], capture_output=True, text=True).stdout.strip()
            user, _, args = ps.partition(" ")
            args = args.strip()
            rows.append(f"pid {pid:>8}  {user or '?':<16} {float(mib)/1024:5.1f} GB  {args.strip()[:90]}")
    except Exception:
        pass
    return rows


def venv_env(cfg: dict, py: Path) -> dict:
    """Environment for anything run with 3DGRUT's venv python (training, export, asset check).

    Activating the venv is NOT optional: 3DGRUT's JIT shells out to `slangc`, which lives in .venv/bin; calling
    .venv/bin/python directly leaves that off PATH and the kernel compile dies with
      FileNotFoundError: [Errno 2] No such file or directory: 'slangc'
    The GPU is pinned by the same UUID the COLMAP container uses."""
    env = {
        "CUDA_HOME": deep_get(cfg, "paths.cuda_home", "/usr"),
        "PYTHONUNBUFFERED": "1",
        "VIRTUAL_ENV": str(py.parent.parent),
        "PATH": str(py.parent) + os.pathsep + os.environ.get("PATH", ""),
        # less fragmentation when the card is shared and nearly full
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    uuid = str(deep_get(cfg, "gpu.uuid", "all"))
    if uuid.lower() != "all":
        env["CUDA_VISIBLE_DEVICES"] = uuid
    return env


def venv_python(threedgrut: Path) -> Path:
    p = threedgrut / ".venv" / "bin" / "python"
    if not p.exists():
        raise SystemExit(
            f"3DGRUT venv not found at {p}\n"
            "Install it first — see README section 'Install 3DGRUT'.")
    return p
