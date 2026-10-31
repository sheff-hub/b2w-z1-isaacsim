"""Stage 10 - check the stage in Isaac Sim itself (headless, Docker): does it render, does the collider sit on the
splat, does the floor hold.

Six recording frames spread over the walk (with several passes: spread over every pass, so the check sees each
camera height) become camera views. For each, render_stage_views.py renders the splat,
the collider alone (orange) and, after dropping a 10 cm ball from 1 m, 1.8 m in front of the camera, the scene
again. isaac/summary.jpg puts them beside the phone's photo: phone | render | collider | collider edges on the
render | after the drop. A ball resting with its centre ~0.10 m above z = 0 means the floor collider is there.
Not fatal: the stages are already written when this runs.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .common import PKG, Log, deep_get, isaac_docker, tool


def run_stage(cfg: dict, scene: Path, name: str, info: dict, log: Log) -> dict:
    log.rule("STAGE 10 - check in Isaac Sim (renders, collider overlay, ball drop)")
    isaac = scene / "isaac"
    z = np.load(scene / "record3d" / "poses_refined.npz")
    n = len(z["c2w"])
    fracs = deep_get(cfg, "isaac.view_fractions", [0.08, 0.25, 0.42, 0.58, 0.75, 0.92])
    if "pass_start" in z and len(z["pass_start"]) > 2:      # several passes: the same number of views, shared out
        st = z["pass_start"]
        k = max(2, int(round(len(fracs) / (len(st) - 1))))
        fr = [int(st[p] + f * (st[p + 1] - st[p] - 1)) for p in range(len(st) - 1) for f in np.linspace(0.15, 0.85, k)]
    else:
        fr = [int(f * (n - 1)) for f in fracs]
    tool(cfg, "make_views.py", [scene / "record3d" / "poses_refined.npz", isaac / "views.json", "--fps", info["fps"],
                                "--frames"] + fr, log)
    chk = isaac / "check"
    chk.mkdir(exist_ok=True)
    chk.chmod(0o777)                   # the container writes as its own user (uid 1234)
    cmd = (f"./python.sh /tools/render_stage_views.py --stage /scene/isaac/{name}.usda --views /scene/isaac/views.json "
           f"--out /scene/isaac/check 2>&1; rc=$?; chmod -R a+rwX /scene/isaac/check; exit $rc")
    r = isaac_docker(cfg, {str(scene): "/scene", str(PKG / "tools"): "/tools:ro"}, cmd, log)
    rest = []
    for ln in r.stdout.splitlines():
        m = re.search(r"RENDERVIEWS\| sphere (\d+): .* rests at \[([-\d.e, ]+)\] \(centre ([-\d.]+) m", ln)
        if m:
            rest.append(float(m.group(3)))
    done = any("RENDERVIEWS| DONE" in ln for ln in r.stdout.splitlines())
    if r.returncode != 0 or not done:
        log(f"!! the Isaac check did not finish (exit {r.returncode}); the stages are fine to open, but unchecked. "
            "Last lines:")
        for t in r.stdout.splitlines()[-12:]:
            log("   |", t)
        return {"ok": False, "exit": r.returncode}
    on_floor = [h for h in rest if abs(h - 0.10) < 0.05]
    log(f"ball drop     : {len(on_floor)}/{len(rest)} balls rest on the floor (centre heights "
        f"{', '.join(f'{h:.2f}' for h in rest)} m; 0.10 = on the floor, higher = on furniture, < 0 = fell through)")
    tool(cfg, "summary_sheet.py", [chk, isaac / "views.json", info["export_dir"], isaac / "summary.jpg",
                                   "--title", f"{name}: phone | render | collider | collider edges on render | after ball drop"],
         log)
    log(f"check sheet   : {isaac / 'summary.jpg'}")
    return {"ok": True, "ball_rest_heights_m": rest, "balls_on_floor": len(on_floor), "balls": len(rest),
            "fell_through": int(sum(h < -0.05 for h in rest)), "summary": str(isaac / "summary.jpg")}
