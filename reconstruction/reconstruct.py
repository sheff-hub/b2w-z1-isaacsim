#!/usr/bin/env python3
"""
Photos -> COLMAP -> 3DGRUT -> NuRec .usdz for Isaac Sim.

    python3 reconstruct.py kitchen                  # a scene file: scenes/kitchen.yaml (dataset, name, settings)
    python3 reconstruct.py my_photos                # or just a photo folder under paths.dataset_root
    python3 reconstruct.py kitchen --caps 300000,400000,500000
    python3 reconstruct.py kitchen --from train     # resume (skips COLMAP)
    python3 reconstruct.py kitchen --only export
    python3 reconstruct.py --list-scenes

Settings: config.yaml (this machine + defaults), then scenes/<name>.yaml on top, then the command line.
Any python3 runs this driver; the heavy steps run in their own environments (COLMAP in Docker, training and
export in 3DGRUT's venv), each as a subprocess whose output goes into <scene>/pipeline.log.

Every stage records completion in <scene>/pipeline_state.json, so re-running after a dropped connection resumes
instead of repeating hours of work.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from tools import export_nurec, prepare, report, sfm, train, validate          # noqa: E402
from tools.common import Log, State, deep_get, expand, load_config             # noqa: E402

STAGES = ["validate", "prepare", "sfm", "train", "export", "report"]
SCENES_DIR = HERE / "scenes"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("target", nargs="?",
                   help="a scene file name (scenes/<name>.yaml), or a photo folder under paths.dataset_root, "
                        "or an absolute path to one")
    p.add_argument("-c", "--config", default=str(HERE / "config.yaml"), help="base config (machine + defaults)")
    p.add_argument("--scene-config", help="a scene file by path (instead of scenes/<target>.yaml)")
    p.add_argument("--name", help="scene name for outputs (default: from the scene file, else the photo folder)")
    p.add_argument("--caps", help="comma-separated gaussian caps to compare, e.g. 300000,400000,500000")
    p.add_argument("--cap-ratio", type=float, help="single cap as this multiple of the COLMAP point count")
    p.add_argument("--downsample", type=int, help="training downsample factor (1=full res)")
    p.add_argument("--max-image-size", type=int, help="COLMAP feature resolution")
    p.add_argument("--from", dest="start", choices=STAGES, help="resume from this stage")
    p.add_argument("--only", choices=STAGES, help="run just this stage")
    p.add_argument("--force", action="store_true", help="ignore recorded state and redo everything")
    p.add_argument("--list-stages", action="store_true")
    p.add_argument("--list-scenes", action="store_true")
    return p.parse_args()


def resolve_config(a) -> tuple[dict, Path | None]:
    """Base config + scene file + command line. A positional target that names a file in scenes/ is a scene file;
    anything else is a photo folder (the old behaviour)."""
    scene_file = Path(a.scene_config) if a.scene_config else None
    if scene_file is None and a.target and (SCENES_DIR / f"{a.target}.yaml").exists():
        scene_file = SCENES_DIR / f"{a.target}.yaml"
    cfg = load_config(a.config, scene_file)
    if a.target and scene_file is None:
        cfg["dataset"]["name"] = a.target
    if a.name:
        cfg["dataset"]["scene_name"] = a.name
    if a.caps:
        cfg["train"]["caps"] = [int(x) for x in a.caps.split(",") if x.strip()]
    if a.cap_ratio:
        cfg["train"]["cap_ratio"] = a.cap_ratio
        cfg["train"]["caps"] = None
    if a.downsample:
        cfg["train"]["downsample_factor"] = a.downsample
    if a.max_image_size:
        cfg["colmap"]["max_image_size"] = a.max_image_size
    return cfg, scene_file


def trained_factor(st: State, log: Log) -> int | None:
    """The downsample factor the recorded model was trained at (state, else the best run's parsed.yaml)."""
    f = st.get("train", "factor")
    if f:
        return int(f)
    best = st.get("train", "best")
    if best:
        parsed = Path(best["dir"]) / "parsed.yaml"
        if parsed.exists():
            for line in parsed.read_text().splitlines():
                if line.strip().startswith("downsample_factor:"):
                    try:
                        return int(line.split(":", 1)[1])
                    except ValueError:
                        pass
    return None


def main() -> int:
    a = parse_args()
    if a.list_stages:
        print("stages:", " -> ".join(STAGES))
        return 0
    if a.list_scenes:
        for f in sorted(SCENES_DIR.glob("*.yaml")):
            print(f"{f.stem:12s} {f}")
        return 0

    cfg, scene_file = resolve_config(a)
    ds_name = deep_get(cfg, "dataset.name")
    if not ds_name:
        return _die("no dataset given. Pass a scene file name (see --list-scenes) or a photo folder.")
    src = Path(ds_name)
    if not src.is_absolute():
        src = expand(deep_get(cfg, "paths.dataset_root")) / ds_name
    if not src.is_dir():
        return _die(f"dataset folder not found: {src}")

    scene_name = deep_get(cfg, "dataset.scene_name") or src.name
    scene = expand(deep_get(cfg, "paths.scenes_root")) / scene_name
    scene.mkdir(parents=True, exist_ok=True)

    log = Log(scene / "pipeline.log")
    st = State(scene / "pipeline_state.json")
    if a.force:
        st.data = {"stages": {}}
        st.save()

    todo = STAGES[STAGES.index(a.start):] if a.start else STAGES
    if a.only:
        todo = [a.only]

    log.rule(f"RECONSTRUCTION  {scene_name}")
    log(f"scene file: {scene_file or '(none: base config only)'}")
    log(f"dataset   : {src}")
    log(f"scene     : {scene}")
    log(f"stages    : {' -> '.join(todo)}")
    t_all = time.time()
    try:
        return run_stages(cfg, a, src, scene, todo, log, st, t_all)
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


def run_stages(cfg, a, src, scene, todo, log, st, t_all) -> int:
    scan = img = None

    # ---- 0/1 validate + prepare -------------------------------------------
    if "validate" in todo or "prepare" in todo:
        scan = validate.run_stage(cfg, src, log)
        st.mark("validate", count=scan["count"], dims=str(scan["dims"]),
                lens=str(scan["lens"]), photometrics=scan.get("photometrics", {}))
        img = prepare.run_stage(cfg, scan, src, scene, log)
        st.mark("prepare", **img)
    else:
        img = {k: st.get("prepare", k) for k in ("images_dir", "count", "width", "height")}

    # The training resolution: chosen (and its images built) only when training runs; afterwards it is the one the
    # model was trained at -- choosing again from the GPU memory free NOW gave a different factor (and a stray
    # images_3 folder) on an export-only run.
    factor = None
    if "train" in todo and not (st.done("train") and not a.force and not a.caps):
        factor = prepare.choose_downsample(cfg, img["count"], img["width"], img["height"], log)
        prepare.build_downsampled(scene, factor, cfg, log)
    else:
        factor = trained_factor(st, log)

    # ---- 2/3 COLMAP --------------------------------------------------------
    if "sfm" in todo:
        if st.done("sfm") and not a.force:
            log("STAGE 2-3 - COLMAP: already done, skipping (use --force to redo)")
        else:
            if scan is None:
                scan = validate.scan(Path(img["images_dir"]), log)
            sfm.run_colmap(cfg, scene, scan, img, log)
            best = sfm.verify_and_promote(cfg, scene, img["count"], log)
            st.mark("sfm", **best)

    if "train" in todo or "sfm" in todo:
        points = sfm.read_points_count(scene)
        if points == 0:
            return _die("no COLMAP points found; cannot train. Check the sfm stage.", log)

    # ---- 4 train -----------------------------------------------------------
    if "train" in todo:
        if st.done("train") and not a.force and not a.caps:
            log("STAGE 4 - train: already done, skipping")
        else:
            out = train.run_stage(cfg, scene, points, factor, log)
            st.mark("train", **out)

    # ---- 5 export + check --------------------------------------------------
    if "export" in todo:
        best = st.get("train", "best")
        if not best:
            return _die("no trained model recorded; run the train stage first.", log)
        usdz, v = export_nurec.run_stage(cfg, scene, best["dir"], log)
        st.mark("export", usdz=str(usdz), size_mb=round(usdz.stat().st_size / 1e6, 1), **v)
        link = scene / "scene.usdz"  # the asset, easy to find: <scene>/scene.usdz
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(os.path.relpath(usdz, scene), link)
        log(f"linked        : {link} -> {os.path.relpath(usdz, scene)}")

    # ---- 6 report ----------------------------------------------------------
    if "report" in todo and deep_get(cfg, "report.write_markdown", True):
        best = st.get("train", "best")
        if best and factor:
            report.comparison_sheet(scene, Path(best["dir"]), factor,
                                    scene / "comparison.png",
                                    deep_get(cfg, "report.comparison_images", 3), log)
        report.write_report(cfg, scene, st.data, log)

    log.rule("DONE")
    log(f"total {(time.time()-t_all)/60:.1f} min")
    u = st.get("export", "usdz")
    if u:
        log(f"asset : {scene / 'scene.usdz'}  (= {u})")
        log("open  : Isaac Sim 6.0.1 -> File > Import, then Create > Physics > Ground Plane")
    return 0


def _die(msg: str, log: Log | None = None) -> int:
    if log:
        log(f"!! ERROR: {msg}")
    print(f"ERROR: {msg}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
