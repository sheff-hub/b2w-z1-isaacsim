#!/usr/bin/env python3
"""
Record3D recording (iPhone LiDAR) -> photoreal splat + LiDAR collider -> Isaac Sim stage a robot can drive on.

    python3 reconstruct.py 2026-10-22--16-54-08      # a recording folder in <paths.nurec>/mine/dataset_for_with_mesh
    python3 reconstruct.py flat_3heights              # a folder of passes: every Record3D export in it is one pass of
                                                      # the same place (a test* subfolder is the test clip)
    python3 reconstruct.py mid low high --name flat   # several recording folders = several passes
    python3 reconstruct.py my_flat                    # or a scene file: scenes/my_flat.yaml (recording, name, settings)
    python3 reconstruct.py my_flat --from train       # redo a stage; what was built from it is redone too
    python3 reconstruct.py my_flat --only test_eval --test-clip clip_0p8m   # score on a clip never trained on
    python3 reconstruct.py --list-scenes

Stages: validate -> prep -> colmap -> loop_close -> refine -> mesh -> train -> export -> stage -> isaac_check ->
test_eval -> report.
Settings: config.yaml (defaults; machine paths, GPU and Isaac Sim from the repository's config.yaml), then
scenes/<name>.yaml on top, then the command line.
Any python3 with PyYAML and numpy runs this driver; the heavy steps run in their own environments (COLMAP and Isaac
Sim in Docker, 3DGRUT in its venv, the rest in this package's .venv), each as a subprocess whose output goes into
<scene>/pipeline.log. Every finished stage is recorded in <scene>/pipeline_state.json: re-running the same command
after a failure or a dropped connection continues where it stopped.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from tools import export_nurec, isaac_check, mesh, record3d, report, sfm, stage, test_clip, train   # noqa: E402
from tools.common import Log, State, deep_get, expand, load_config                          # noqa: E402

STAGES = ["validate", "prep", "colmap", "loop_close", "refine", "mesh", "train", "export", "stage", "isaac_check",
          "test_eval", "report"]
SCENES_DIR = HERE / "scenes"
# what each stage is built from: a stage that runs again makes everything built from it (transitively) stale.
# The splat (train) needs only the poses, so a new collider (mesh) does not retrain it.
DEPENDS = {"prep": ["validate"], "colmap": ["prep"], "loop_close": ["colmap"], "refine": ["loop_close"],
           "mesh": ["refine"], "train": ["refine"], "export": ["train"], "stage": ["mesh", "export"],
           "isaac_check": ["stage"], "test_eval": ["train"], "report": []}


def built_from(stage: str) -> list[str]:
    """Every stage that depends on `stage`, directly or through others."""
    out, todo = [], [stage]
    while todo:
        s = todo.pop()
        for t, deps in DEPENDS.items():
            if s in deps and t not in out:
                out.append(t)
                todo.append(t)
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("target", nargs="*",
                   help="a scene file name (scenes/<name>.yaml), or a recording folder under paths.recordings_root "
                        "(or an absolute path): one Record3D export, or a folder of several = several passes; "
                        "several folders = several passes")
    p.add_argument("-c", "--config", default=str(HERE / "config.yaml"), help="base config (machine + defaults)")
    p.add_argument("--scene-config", help="a scene file by path (instead of scenes/<target>.yaml)")
    p.add_argument("--name", help="scene name for outputs (default: from the scene file, else the recording folder)")
    p.add_argument("--caps", help="comma-separated gaussian caps to compare, e.g. 1000000,2000000")
    p.add_argument("--cap-ratio", type=float, help="single cap as this multiple of the bundle-adjustment point count")
    p.add_argument("--keep-every", type=int, help="use the sharpest of every N frames (default: one image per 6 cm / "
                                                  "3 deg of motion)")
    p.add_argument("--test-clip", help="a Record3D folder never trained on, to score the splat (stage test_eval)")
    p.add_argument("--from", dest="start", choices=STAGES, help="run from this stage on (redoing it if it was done)")
    p.add_argument("--only", choices=STAGES, help="run just this stage (redoing it if it was done)")
    p.add_argument("--force", action="store_true", help="redo every stage, ignoring recorded state")
    p.add_argument("--no-isaac-check", action="store_true", help="skip the Isaac Sim check (no Isaac Sim image here)")
    p.add_argument("--list-stages", action="store_true")
    p.add_argument("--list-scenes", action="store_true")
    return p.parse_args()


def resolve_config(a) -> tuple[dict, Path | None]:
    scene_file = Path(a.scene_config) if a.scene_config else None
    if scene_file is None and len(a.target) == 1 and (SCENES_DIR / f"{a.target[0]}.yaml").exists():
        scene_file = SCENES_DIR / f"{a.target[0]}.yaml"
    cfg = load_config(a.config, scene_file)
    if a.target and scene_file is None:
        cfg["recording"]["name"] = a.target[0]
        cfg["recording"]["passes"] = list(a.target) if len(a.target) > 1 else None
    if a.test_clip:
        cfg["recording"]["test"] = a.test_clip
    if a.name:
        cfg["recording"]["scene_name"] = a.name
    if a.caps:
        cfg["train"]["caps"] = [int(x) for x in a.caps.split(",") if x.strip()]
    if a.cap_ratio:
        cfg["train"]["cap_ratio"] = a.cap_ratio
        cfg["train"]["caps"] = None
    if a.keep_every:
        cfg["prep"]["keep_every"] = a.keep_every
    if a.no_isaac_check:
        cfg["isaac"]["check"] = False
    return cfg, scene_file


def main() -> int:
    a = parse_args()
    if a.list_stages:
        print("stages:", " -> ".join(STAGES))
        return 0
    if a.list_scenes:
        for f in sorted(SCENES_DIR.glob("*.yaml")):
            print(f"{f.stem:24s} {f}")
        return 0
    cfg, scene_file = resolve_config(a)
    try:
        passes, test = record3d.resolve_recordings(cfg, a.target)
    except SystemExit as e:
        return _die(str(e))
    first = (deep_get(cfg, "recording.passes") or [deep_get(cfg, "recording.name")])[0]
    scene_name = deep_get(cfg, "recording.scene_name") or Path(str(first)).expanduser().name
    scene = expand(deep_get(cfg, "paths.scenes_root")) / scene_name
    scene.mkdir(parents=True, exist_ok=True)

    log = Log(scene / "pipeline.log")
    st = State(scene / "pipeline_state.json")
    todo = STAGES[STAGES.index(a.start):] if a.start else list(STAGES)
    if a.only:
        todo = [a.only] + (["report"] if a.only != "report" else [])     # the report shows what --only changed
    redo = set(STAGES) if a.force else ({a.only} if a.only else {a.start} if a.start else set())
    if not deep_get(cfg, "isaac.check", True) and "isaac_check" in todo:
        todo.remove("isaac_check")

    log.rule(f"RECONSTRUCTION WITH MESH  {scene_name}")
    log(f"scene file: {scene_file or '(none: base config only)'}")
    for k, ex in enumerate(passes):
        log(f"{'pass ' + str(k + 1) if len(passes) > 1 else 'recording':<10}: {ex}")
    if test:
        log(f"test clip : {test}")
    log(f"scene     : {scene}")
    log(f"stages    : {' -> '.join(todo)}")
    t_all = time.time()
    try:
        return run_stages(cfg, passes, test, scene, scene_name, todo, redo, log, st, t_all)
    except BaseException as exc:  # noqa: BLE001 -- the log is the only record of an unattended run
        if isinstance(exc, KeyboardInterrupt):
            log("!! interrupted")
            raise
        log("!! FAILED: " + (str(exc) if isinstance(exc, SystemExit) else f"{type(exc).__name__}: {exc}"))
        if not isinstance(exc, SystemExit):
            for line in traceback.format_exc().rstrip().splitlines():
                log("   | " + line)
        log("   (fix the cause, then re-run the same command: finished stages are skipped)")
        return 1


def run_stages(cfg, passes, test, scene, name, todo, redo, log, st, t_all) -> int:
    def need(s):
        if s == "test_eval" and test and st.get("test_eval", "skipped"):
            return s in todo                  # a test clip has been added since
        return s in todo and (s in redo or not st.done(s) or s == "report")

    before = ([p.get("export_dir") for p in st.get("validate", "passes")] if st.get("validate", "passes")
              else [st.get("validate", "export_dir")])           # scenes made before passes existed
    same = lambda x, y: [str(Path(q).resolve()) for q in x if q] == [str(Path(q).resolve()) for q in y]
    if st.done("validate") and not same(before, passes) and "validate" in todo:
        log("the recordings differ from the last run's: starting again from validate")
        redo = set(redo) | {"validate"}

    def finish(s, t0, **info):
        st.mark(s, minutes=round((time.time() - t0) / 60, 1), **info)

    for s in todo:
        if not need(s):
            log(f"{s}: already done, skipping (--from {s} redoes it)")
            continue
        st.drop(built_from(s))          # built from this stage's old output: stale now
        t0 = time.time()
        info = st.data["stages"].get("validate", {})
        if s == "validate":
            finish(s, t0, **record3d.validate(cfg, passes, test, log))
        elif s == "prep":
            finish(s, t0, **record3d.prep(cfg, info, scene, log))
        elif s == "colmap":
            finish(s, t0, **sfm.features_and_matches(cfg, scene, log))
        elif s == "loop_close":
            finish(s, t0, **sfm.loop_close(cfg, scene, log))
        elif s == "refine":
            finish(s, t0, **sfm.refine(cfg, scene, log))
        elif s == "mesh":
            finish(s, t0, **mesh.run_stage(cfg, scene, log))
        elif s == "train":
            points = st.get("refine", "points") or 0
            if not points:
                return _die("no bundle-adjustment points recorded; run the refine stage first", log)
            out = train.run_stage(cfg, scene, points, int(deep_get(cfg, "train.downsample_factor", 1)), log,
                                  images=st.get("prep", "images") or 0)
            finish(s, t0, **out)
        elif s == "export":
            best = st.get("train", "best")
            if not best:
                return _die("no trained model recorded; run the train stage first", log)
            usdz, v = export_nurec.run_stage(cfg, scene, best["dir"], log)
            finish(s, t0, usdz=str(usdz), size_mb=round(usdz.stat().st_size / 1e6, 1), **v)
        elif s == "stage":
            finish(s, t0, **stage.run_stage(cfg, scene, name, log, float(deep_get(cfg, "stage.friction", 1.0))))
        elif s == "isaac_check":
            out = isaac_check.run_stage(cfg, scene, name, info, log)
            finish(s, t0, **out)              # ok=False when it did not finish: the next run tries again
        elif s == "test_eval":
            finish(s, t0, **test_clip.run_stage(cfg, scene, st.data, test, log))
        elif s == "report":
            report.write_report(scene, name, st.data, log)

    log.rule("DONE")
    log(f"total {(time.time() - t_all) / 60:.1f} min")
    if st.get("stage", "scene_usda"):
        log(f"scene        : {st.get('stage', 'scene_usda')}")
        log(f"robot ground : {st.get('stage', 'robot_usda')}")
        log(f"report       : {scene / 'REPORT.md'}")
    return 0


def _die(msg: str, log: Log | None = None) -> int:
    if log:
        log(f"!! ERROR: {msg}")
    print(f"ERROR: {msg}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
