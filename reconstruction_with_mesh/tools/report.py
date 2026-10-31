"""Stage 11 - REPORT.md: what was built, how good it is, what to open."""
from __future__ import annotations

from pathlib import Path

from .common import Log


def _f(v, fmt=".2f", unit=""):
    return "?" if v is None else f"{format(v, fmt)}{unit}"


def _test_section(te: dict) -> list[str]:
    if not te or te.get("skipped"):
        return []
    if not te.get("ok"):
        return ["## Test clip", "", f"**Not scored**: {te.get('error', 'see pipeline.log')}", ""]
    L = ["## Test clip (never trained on)", "",
         f"- `{te['export_dir']}`: {te['duration_s']:.0f} s, {te['images']} images, camera median "
         f"{te['camera_height_m']:.2f} m; {te['localized']} ({te['localized'] / max(te['images'], 1):.0%}) localised in the "
         f"scene (median {te['loc_inliers_median']:.0f} inliers, {te['loc_reproj_px']:.2f} px), "
         f"{te.get('interpolated', 0)} more placed from their neighbours; the rest match too little of the scene",
         f"- **PSNR {te['psnr']:.2f} dB** (colour-corrected {te['cc_psnr']:.2f} dB), SSIM {te['ssim']:.3f}, LPIPS "
         f"{te['lpips']:.3f}: how the splat looks from where the training walk did not go"
         + (f" (localised frames only: {te['psnr_localized_only']:.2f} dB)" if te.get("psnr_localized_only") else "")]
    for b in te.get("bands", []):
        L.append(f"  - camera {b['band']}: {b['images']} images, PSNR median {b['psnr_median']:.2f} dB")
    if te.get("sheet"):
        L += ["", "![test](test/sheet.jpg)", "", "`test/sheet.jpg`: test photo | splat render"]
    return L + [""]


def write_report(scene: Path, name: str, state: dict, log: Log) -> Path:
    log.rule("STAGE 12 - report")
    g = state["stages"]
    val, prep, col, lc = g.get("validate", {}), g.get("prep", {}), g.get("colmap", {}), g.get("loop_close", {})
    ba, me, tr, ex = g.get("refine", {}), g.get("mesh", {}), g.get("train", {}), g.get("export", {})
    stg, chk, te = g.get("stage", {}), g.get("isaac_check", {}), g.get("test_eval", {})
    passes = prep.get("passes") or []
    heights = ", ".join(f"{p['camera_height_m']:.2f}" for p in passes)
    best = tr.get("best", {}) or {}
    rel = lambda p: str(Path(p).relative_to(scene)) if p and str(p).startswith(str(scene)) else str(p)
    L = [f"# {name}: Record3D -> splat + collider", "",
         "## Open in Isaac Sim (File > Open, RTX renderer)", "",
         "| file | what |", "|---|---|",
         f"| `{rel(stg.get('scene_usda'))}` | **the scene**: photoreal splat + invisible LiDAR collider (metres, Z-up, floor at z = 0) |",
         f"| `{rel(stg.get('robot_usda'))}` | the same + friction-1.0 material on the collider: **use as a robot's ground** (Isaac Lab `terrain_type=\"usd\"`) |",
         f"| `{rel(stg.get('visible_usda'))}` | collider drawn orange, to check alignment |",
         "", "The stages reference `export/scene.usdz` relatively: move the scene folder as a whole.",
         "To see the collider in the main stage, hide the splat (`/World/room/splat`); making the collider visible "
         "shows nothing, because the volume uses it as its proxy.", ""]
    if chk.get("ok"):
        L += ["## Isaac check", "",
              f"![check](isaac/summary.jpg)", "",
              f"`isaac/summary.jpg`: phone photo | render | collider | collider edges on the render | after a ball drop, at "
              f"{chk.get('balls', '?')} points of the walk.",
              f"- balls resting on the floor: **{chk.get('balls_on_floor')}/{chk.get('balls')}** (centre heights "
              f"{', '.join(_f(h) for h in chk.get('ball_rest_heights_m', []))} m; 0.10 = floor, higher = on furniture)",
              f"- fell through: {chk.get('fell_through', 0)}", ""]
    elif chk:
        L += ["## Isaac check", "", "**Did not finish**: see the end of `pipeline.log`. The stages can still be opened.", ""]
    if len(passes) > 1:
        rec_lines = [f"- {len(passes)} passes of the same place, {val.get('frames', 0):,} frames, "
                     f"{_f(val.get('duration_s', 0) / 60, '.1f')} min, {_f(val.get('path_m'), '.0f')} m walked (ARKit)", "",
                     "| pass | recording | min | m walked | camera height (p5-p95) | images | blurred dropped |",
                     "|---|---|---|---|---|---|---|"]
        for k, p in enumerate(passes):
            src = Path(p["src"])
            rec_lines.append(f"| {k + 1} | `{src.parent.name if src.name == 'EXR_RGBD' else src.name}` "
                             f"| {p['duration_s'] / 60:.1f} | {p['path_m']:.0f} | {p['camera_height_m']:.2f} m "
                             f"({p['camera_height_p5']:.2f}-{p['camera_height_p95']:.2f}) | {p['images']:,} | {p['blurred_dropped']} |")
        rec_lines += ["", f"- {prep.get('images', '?'):,} images used ({prep.get('image_rule', '?')}), "
                          f"{val.get('width')} x {val.get('height')}"]
    else:
        rec_lines = [f"- `{val.get('export_dir', '?')}`",
                     f"- {val.get('frames', 0):,} frames, {_f(val.get('duration_s'), '.0f')} s at {_f(val.get('fps'), '.0f')} fps; "
                     f"{_f(val.get('path_m'), '.0f')} m walked; camera median {_f(prep.get('camera_height_m'))} m above the floor",
                     f"- {prep.get('images', '?')} images used ("
                     + (prep.get("image_rule") or f"the sharpest of every {prep.get('keep_every', '?')} frames")
                     + f"), {val.get('width')} x {val.get('height')}"]
    align = []
    if col.get("alignment"):
        for p, t in sorted(col["alignment"]["transforms"].items()):
            if p != "1":
                align.append(f"- pass {p} brought into pass 1's frame from image matches: turned {t['yaw_deg']:+.1f} deg, "
                             f"shifted ({t['shift_m'][0]:+.2f}, {t['shift_m'][1]:+.2f}) m (`record3d/passes_aligned.png`)")
    if lc.get("loops_by_pass_pair") and len(passes) > 1:
        align.append("- loop edges by pass pair: " + ", ".join(f"{k}: {v:,}" for k, v in sorted(lc["loops_by_pass_pair"].items())))
    L += ["## Recording", ""] + rec_lines + ["",
          "## Poses", "",
          f"- ARKit relocalisation jumps: {prep.get('snap_events', '?')} (removed)"] + align + [
          (f"- loop closure: {lc.get('loops_accepted', 0):,} verified loop edges; loop error "
           f"{_f(lc.get('loop_err_before_cm'), '.1f')} cm -> **{_f(lc.get('loop_err_after_cm'), '.1f')} cm** "
           f"(p90 {_f(lc.get('loop_err_after_p90_cm'), '.1f')} cm); largest correction {_f(lc.get('correction_max_cm'), '.0f')} cm"
           if lc.get("pose_graph") else
           f"- loop closure: {lc.get('loops_accepted', 0)} loop edges, nothing to close (no revisits); ARKit's trajectory used"),
          f"- bundle adjustment: reprojection {_f(ba.get('reproj_phone_px'))} px with the phone poses -> "
          f"**{_f(ba.get('reproj_median_px'))} px** (median); {ba.get('points', 0):,} points; cameras moved median "
          f"{_f(ba.get('ba_shift_median_cm'), '.1f')} cm; scale x{_f(ba.get('sim3_scale'), '.4f')}",
          f"- COLMAP pairs matched: {col.get('pairs', 0):,} (chosen from the poses)",
          "",
          "## Collider (`mesh/collider.ply`)", "",
          f"- {me.get('triangles', 0):,} triangles, LiDAR TSDF (3 cm), in `eval/`: `plan.png` (slices at 0.3 and 1.0 m), "
          f"`xsec_collider.png` (sections across the path)",
          f"- floor holes patched: {me.get('floor_holes_patched', 0)} ({_f(me.get('floor_holes_patched_m2'), '.2f')} m^2), "
          f"list in `eval/floor_holes.json`",
          f"- floor within 8 cm of z = 0 under **{_f(me.get('floor_hit_8cm'), '.0%')}** of the camera positions "
          f"(none under {_f(me.get('no_floor'), '.0%')})",
          f"- walked path blocked (0.2-1 m): **{_f(me.get('path_blocked_0.2_to_1m'), '.1%')}**",
          f"- horizontal rays escaping at 1 m: {_f(me.get('leak_1.0m'), '.1%')} (open doors and glass count)",
          f"- BA points within 5 cm of the mesh: {_f(me.get('sfm_within_5cm'), '.0%')}",
          "",
          "## Splat (`export/scene.usdz`)", "",
          f"- {ex.get('gaussians', best.get('cap', 0)):,} gaussians (cap {best.get('cap', 0):,}), {ex.get('size_mb', '?')} MB",
          f"- {tr.get('iterations', 30000):,} training steps",
          f"- held-out frames (every 8th image of the walk itself): **PSNR {_f(best.get('mean_psnr'))} dB**, SSIM "
          f"{_f(best.get('mean_ssim'), '.3f')}, LPIPS {_f(best.get('mean_lpips'), '.3f')} (LPIPS is scored at full size "
          "and reads high; trust PSNR)",
          f"- upright: {ex.get('reoriented', '?')}, finite bounds: {ex.get('finite', '?')}",
          "- for reference, the 407 s two-lap walk of 22 Oct 2026 (M1): 27.1 dB, loop error 0.4 cm, 0% of the path "
          "blocked, 100% floor",
          "",
          ] + _test_section(te) + [
          "## Limits", "",
          "- Glass is not solid in the collider, and where the phone never looked there is no floor.",
          (f"- Viewpoints are the phone's heights ({heights} m): "
           "a camera far outside them sees blur in places." if len(passes) > 1 else
           f"- Viewpoints are the phone's height (~{_f(prep.get('camera_height_m'), '.1f')} m): a camera much lower or "
           "higher sees blur in places."),
          "- Close to door frames and furniture the splat blurs where the phone passed only once.",
          "",
          "## Time", "",
          "| stage | min |", "|---|---|"]
    for s, d in g.items():
        if "minutes" in d:
            L.append(f"| {s} | {d['minutes']:.1f} |")
    md = scene / "REPORT.md"
    md.write_text("\n".join(L) + "\n")
    log(f"report        : {md}")
    return md
