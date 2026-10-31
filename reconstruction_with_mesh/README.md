# reconstruction_with_mesh: iPhone LiDAR recording -> Isaac Sim scene with a collider

One iPhone LiDAR recording in (or several passes of the same place, e.g. the phone at three heights), an Isaac
Sim scene a robot can drive on out: a **photoreal splat** that you see, plus an **invisible LiDAR collision mesh**
that the robot touches, aligned to the centimetre, in metres, Z-up, with the floor at z = 0.

```
Record3D (EXR + JPG), 1 or more passes  ->  ARKit poses, floor-pinned  ->  COLMAP features
   ->  passes aligned from image matches  ->  pose-guided matching  ->  loop closure (pose graph)
   ->  bundle adjustment  ->  LiDAR TSDF collider  ->  3DGRUT splat  ->  NuRec .usdz  ->  Isaac stage
   ->  check in Isaac Sim  ->  score on a test clip never trained on  ->  REPORT.md
```

This automates the best of four methods compared on one recording on 22 Oct 2026 (called **M1** there; the others
were Stani's Depth Anything 3 + VDB, nerfstudio's documented Record3D route and NVIDIA's stereo NuRec recipe):
PSNR 27.1 dB, loop error 0.4 cm, 0% of the walked path blocked, floor under 100% of it. The B2W and the B2W + Z1
drove a loop in that scene.

Its sibling `../reconstruction_no_mesh` takes ordinary photos and gives a splat only (no collider).

---

## How to reconstruct a scene, step by step

1. **Record** the place with the Record3D app (free tier; an iPhone Pro or iPad Pro with LiDAR): one recording, or
   several **passes** of the same place, e.g. the phone at three heights (see "Recording several passes" below).
   Walk at a normal slow pace, **turn slowly**, come back through places you have been, keep some floor in view.
2. **Export** each recording in the app: Library -> the recording -> Export -> **"EXR + JPG"**, and copy the
   exported folders to the dataset folder `<paths.nurec>/mine/dataset_for_with_mesh/` (`paths.nurec` is set in the
   repository's `config.yaml`; default `scenes/nurec`). Each export holds `EXR_RGBD/` with `metadata.json`, `rgb/`
   and `depth/`.
   - one recording: `dataset_for_with_mesh/<recording>/EXR_RGBD/...`
   - several passes of one place: one folder for the place, one subfolder per pass (they are taken in name
     order, the first is the reference), and optionally a test clip in a subfolder whose name starts with `test`:
     ```
     dataset_for_with_mesh/my_flat/
     ├── 1_mid/EXR_RGBD/...       pass 1 (reference): chest height, two laps
     ├── 2_low/EXR_RGBD/...       pass 2
     ├── 3_high/EXR_RGBD/...      pass 3
     └── test_0p8m/EXR_RGBD/...   never trained on: the splat is scored on it
     ```
3. **Optional: a scene file** `scenes/<name>.yaml` with the output name and settings (`scenes/README.md`).
4. **Run** (in `tmux`, or `nohup ... &`: about 1.5 hours for a 7-minute walk, several hours for three passes):
   ```sh
   python3 run.py reconstruct-mesh <recording>   # from the repository root: one recording
   python3 run.py reconstruct-mesh my_flat       # a folder of passes (or a scene file of that name)
   # = python3 reconstruction_with_mesh/reconstruct.py ...
   ```
5. **Look at the result** in `<paths.nurec>/mine/withmesh/<name>/`:
   - `REPORT.md`: the numbers, `isaac/summary.jpg` (phone photo | render | collider | ball drop) and, with a test
     clip, `test/sheet.jpg` (test photo | render);
   - `isaac/<name>.usda`: **open it in Isaac Sim** (File -> Open, RTX): the photoreal scene, with an invisible
     collider in metres, Z-up, floor at z = 0;
   - `isaac/<name>_robot.usda`: the same as **a robot's ground** (Isaac Lab `terrain_type="usd"`).

If it stops, fix the cause shown at the end of `<scene>/pipeline.log` and run the same command again: finished
stages are skipped. First time on a machine: install the requirements below (`./setup_tools_env.sh` and 3DGRUT).

The scene of this project (`python3 reconstruct.py --list-scenes` lists the scene files):

```sh
python3 run.py reconstruct-mesh record3d_2laps    # the 407 s two-lap corridor walk of 22 Oct 2026 (89 min)
```

### Recording several passes

Each Record3D recording has its own ARKit world: its own origin and heading (the origin is where ARKit started,
not where the recording starts: 11 m away on 22 Oct). Height and tilt are shared anyway (gravity, and every pass
has its floor pinned at z = 0), so the `colmap` stage only has to find, per pass, a turn about the vertical and a
shift along the floor. It does that from image matches: ~150 images of each pass are matched against those of the
others, every verified pair votes for the transform (LiDAR-lifted PnP, checked both ways), and the consensus
places the pass (`record3d/passes_aligned.png`). After that the passes are one walk: the pose graph closes loops
within and between them, and bundle adjustment, collider and splat use all of them.

Image matches tie passes at different heights only weakly (a 0.5 m pass and a 1.4 m pass see the same wall at very
different angles: 286 loop edges between them on 27 Oct, against 24,843 between 1.4 m and 1.9 m), and between those
few ties a pass drifts with ARKit. So the pose graph also gets **LiDAR ties**: every ~2 m stretch of a pass is
ICP-fitted onto each other pass's LiDAR walls, weighted by what the walls fix, in up to three rounds (~25 min).

What makes passes merge well:
- **the same route in the same direction** in every pass: matches need the same surfaces seen from similar
  directions (a pass walked the other way round sees the other side of everything);
- **pass 1 is the reference**: record the chest-height pass first (two laps, as on 22 Oct); a low pass and a high
  pass each match it better than they match each other;
- **in each room a loop**: along the walls looking into the room, then through the middle looking out, then on;
  pause in doorways and look back;
- **turn slowly** (a full turn in 25 s or more): on 22 Oct, 37% of the frames taken while turning at 15-30 deg/s
  were blurred and 62% at 30-60 deg/s, against 11% below 15 deg/s; walking speed made no difference;
- **nothing changes between passes**: the same doors open, the same lights on, blinds as they were, no people
  in view; record the passes back to back (daylight moves);
- passes of ~5-10 minutes with a short break: the phone gets hot under LiDAR + 60 fps, and a bad pass is redone
  alone;
- a low pass is smoothest on a wheeled chair or trolley at ~0.55 m; a high pass (~1.6 m) tilted 20-30 deg down
  sees table tops and the floor from above (fewer LiDAR holes on shiny floors).

**A test clip** (optional, 1-2 minutes) is a separate recording of the same place that is never trained on, best
from where the passes did not go: another height (e.g. 0.8 m), across rooms diagonally. The training PSNR is
measured on every 8th image of the training walk itself, between other training views, and cannot show how the
scene looks from elsewhere; the test clip can. The same clip scores any scene of that place:

```sh
python3 reconstruct.py record3d_2laps --only test_eval --test-clip my_flat/test_0p8m   # the old one-height scene
```

### Resuming and redoing

Every stage that finishes is recorded in `<scene>/pipeline_state.json`. After a failure or a dropped connection,
fix the cause and **run the same command again**: finished stages are skipped.

```sh
python3 reconstruct.py my_flat --from train      # redo training, the export, stage and check
python3 reconstruct.py my_flat --only isaac_check
python3 reconstruct.py my_flat --force           # redo everything
python3 reconstruct.py --list-stages
```

A stage that runs again makes what was built from it stale, and that runs again too: `--from mesh` builds a new
collider, stage and check but keeps the trained splat (it needs only the poses); `--from refine` redoes both.

### Other overrides

```sh
python3 reconstruct.py my_flat --name flat_v2            # output folder name
python3 reconstruct.py mid low high --name flat          # several recording folders = several passes
python3 reconstruct.py my_flat --test-clip clip_0p8m     # score the splat on a recording never trained on
python3 reconstruct.py my_flat --keep-every 8            # the first version's frame rule (default: per 6 cm / 3 deg)
python3 reconstruct.py my_flat --caps 1000000,2000000    # train both, export the better one
python3 reconstruct.py my_flat --no-isaac-check          # a machine without the Isaac Sim image
```

---

## Requirements

### Hardware
- NVIDIA GPU with RT cores; training a 2M-gaussian scene needs ~14 GB free (`gpu.min_free_gb`)
- ~10 GB disk per scene (7-minute walk), 32 GB RAM

### Software
| requirement | notes |
|---|---|
| Linux, NVIDIA driver, CUDA 12 toolkit | `CUDA_HOME` may be `/usr` (`paths.cuda_home`) |
| Docker + NVIDIA Container Toolkit | COLMAP and Isaac Sim run in containers |
| `colmap/colmap:latest` | `docker pull colmap/colmap:latest` (4.2+, with CUDA; Ubuntu's apt colmap is CPU-only) |
| `nvcr.io/nvidia/isaac-sim:5.1.0` | only for the check stage (`isaac.check`), which can be switched off |
| 3DGRUT with its venv | as in `../reconstruction_no_mesh/README.md` ("3. 3DGRUT"); `paths.threedgrut` in the repository's `config.yaml` |
| Python 3.10 + `uv` (or pip) | for the tools environment below |

**No `sudo` is needed** anywhere.

### The tools environment

```sh
cd reconstruction_with_mesh && ./setup_tools_env.sh   # makes .venv: pycolmap 4.2.1, open3d, vdbfusion, opencv (EXR), scipy, trimesh, usd-core
```

The driver itself runs with any `python3` that has PyYAML and numpy (`requirements.txt`).

Machine settings come from the repository's `config.yaml`: `paths.nurec` (datasets and scenes), `paths.threedgrut`,
`paths.isaac_cache`, `gpu.uuid` and `images.isaac_sim`. This folder's `config.yaml` holds the method's defaults;
a value set there (not null) wins.

---

## What each stage does

| # | stage | what | output |
|---|---|---|---|
| 1 | `validate` | every export is complete (poses, images, EXR depth), long enough, how far you walked; passes and test clip the same image size | log |
| 2 | `prep` | per pass: ARKit poses to metres, Z-up; **each frame's LiDAR floor pinned to z = 0** (the densest layer more than 0.6 x the camera's own height below it, so a phone at 0.5 m or 1.6 m finds it too); **one image per 6 cm / 3 deg of motion** (the sharpest of its slot; blurred slots dropped); one LiDAR frame per 1.5 cm / 1 deg | `images/`, `record3d/poses.npz` |
| 3 | `colmap` | SIFT features; **several passes aligned** (a turn and a shift each, from verified image matches); then matching only the pairs the poses suggest (time + space neighbours, loop closures, across passes): 152k pairs instead of 4.6M for one walk | `colmap/database.db`, `record3d/passes_aligned.png` |
| 4 | `loop_close` | ARKit's jumps removed (per pass); a pose graph (x, y, yaw) fitted to PnP loop edges with LiDAR depth, each checked both ways, robust loss graduated 30 -> 1 sigma; edges between passes are loop edges like any other; several passes: LiDAR ties (each stretch ICP-fitted onto the other passes) | `record3d/poses_lc.npz`, `record3d/loop_close.png` |
| 5 | `refine` | triangulate with those poses, bundle-adjust twice, align back to metres | `sparse/0`, `record3d/poses_refined.npz` |
| 6 | `mesh` | TSDF fusion of the LiDAR frames (3 cm voxels), decimated to 1.2M triangles, **small holes in the floor patched**; checks: floor, walked path, leaks | `mesh/collider.ply`, `eval/` |
| 7 | `train` | 3DGRUT 3DGUT + MCMC, ~11 views of every training image (30k-90k steps, schedules stretched with it), ~4x the BA points as gaussians (at most 2M: three heights of a place are not a larger place), large-scene regularisation and MCMC noise | `runs/` |
| 8 | `export` | NuRec .usdz with the tilt and float16 fixes (see nomesh's README) | `export/scene.usdz` |
| 9 | `stage` | Isaac stage: splat + invisible collider wired as NVIDIA's scenes do (proxy, matte, dome light); a visible-collider copy; a robot-ground wrapper | `isaac/*.usda` |
| 10 | `isaac_check` | headless Isaac Sim: six views looking along the walk (shared out over the passes) rendered, collider overlaid, a ball dropped on the walked path 1.8 m ahead in each | `isaac/summary.jpg` |
| 11 | `test_eval` | only with a test clip: its images localised against the bundle-adjusted cameras (points triangulated from the scene's images alone), the splat rendered there and scored, also per camera height | `test/`, `test/sheet.jpg` |
| 12 | `report` | everything above in one page | `REPORT.md` |

The mesh is built before the splat on purpose: it needs only the poses, and a bad collider shows up an hour
earlier.

---

## Capture guide (Record3D)

| setting | do this | why |
|---|---|---|
| app | Record3D, **LiDAR** (back) camera, "Higher quality" depth if offered | the collider is built from the LiDAR |
| export | **EXR + JPG** | metric depth per frame; the PNG/video exports lose it |
| hold | the phone upright, at chest height (~1 m) for a single recording | the robot's head camera rides at 1.05 m |
| **turning** | **slowly: a full turn in 25 s or more**; turn along an arc while walking, not on the spot | turns blur frames (22 Oct: 11% blurred below 15 deg/s, 37% at 15-30, 62% at 30-60); a pure spin gives no depth |
| walking | a normal slow pace (0.3-0.5 m/s) | walking speed did not blur frames; standing still costs nothing (images are taken per distance moved) |
| route | **walk loops: come back through places you have been**, ideally twice; in each room a loop (walls looking in, middle looking out) | loop closure needs revisits; it took the loop error of a two-lap walk from 23 cm to 0.4 cm |
| several passes | same route, same direction, nothing moved in between; chest height first | see "Recording several passes" |
| floor | keep some floor in view most of the time | each frame's height is pinned on the floor it sees |
| corners and doors | pass through doorways straight, look into rooms you will drive into, pause and look back | the collider has no floor where the phone never looked; door frames blur when passed once |
| length | 1-10 minutes per recording | the phone heats up; longer works but a failure costs more |
| glass, mirrors | expect holes | LiDAR does not see glass: it is not solid in the collider |

---

## Interpreting the results

- **Loop error after closure** (`REPORT.md`, Poses): a few cm or less is right. If it stays high, the walk had
  too few revisits or too little texture where it came back.
- **Collider checks**: *floor within 8 cm* should be ~100%; *walked path blocked* should be ~0% (where you walked
  is free); *rays escaping* counts real openings too (doors, glass), so a few percent is normal.
- **PSNR**: 27.1 dB on the 22 Oct walk (held-out frames of the walk itself). LPIPS reads high here (scored at full
  size); compare runs to each other, and look at the renders.
- **Test clip** (`REPORT.md`, Test clip): the PSNR of views the training never saw; lower than the held-out PSNR
  by nature. The colour-corrected PSNR discounts a different exposure of the clip; the per-height lines show where
  the scene holds up. Compare scenes of one place on the same clip.
- **Several passes**: every pass should be placed with many agreeing votes (`pipeline.log`, "pass N -> 1") and have
  loop edges to the others (REPORT, Poses); `record3d/passes_aligned.png` and `loop_close.png` show the paths.
- **`isaac/summary.jpg`**: the collider's edges should sit on the splat's walls and furniture; balls should rest
  with their centre at ~0.10 m (on the floor) or on furniture, never below 0.

In Isaac Sim, making the collider visible in the main stage shows nothing (the splat uses it as its proxy): hide
the splat instead, or open `<name>_visible.usda`.

### Using it with a robot (Isaac Lab)

Use `isaac/<name>_robot.usda` as the terrain: `terrain_type="usd"`, `usd_path=<it>`. Isaac Lab binds its terrain
physics material only to terrains it builds, so the wrapper binds friction 1.0 to the collider itself. Turn off
the task's `terrain_out_of_bounds` termination and its random reset pose. `tools/robot_route.py` plans a
corridor-centred route from `mesh/collider.ply` through waypoints you pick on `eval/plan.png`:

```sh
.venv/bin/python tools/robot_route.py <scene>/mesh/collider.ply <scene>/robot/route.json --closed \
    --poses <scene>/record3d/poses_refined.npz --waypoints "3,-11; -4,-16.5; -8.8,-12; 0.7,0.3; 7.8,-7.3"
```

It writes the route (with the clearance of every point), a plan image and clearance grids, for a controller that
drives the robot round it (a pure-pursuit follower with the ETH policy did 46 m through that door with no contact).

---

## Timings

Measured on the RTX 6000 Ada (shared with other jobs, which made COLMAP's GPU steps slower than alone):

| stage | 407 s walk (24,397 frames, 3,050 images) | 70 s clip (700 images) | two passes, 346 s (2,352 images) |
|---|---|---|---|
| prep | 1.4 | 0.3 | 1.2 |
| colmap (features + alignment + matching) | 22 | 2.7 | 5.7 |
| loop_close | 1.4 | <0.1 | 0.9 |
| refine (BA) | 12 | 1.5 | 7.9 |
| mesh (fusion + holes + checks) | 35 | 6.3 | 24 |
| train | 15 | 12-19 (alone / beside another run) | 20 |
| export, stage | <1 | <1 | <1 |
| isaac_check | 2 | 3-4 | 6 |
| test_eval (60 s clip) | - | - | 1.7 |
| **total** | **89 min** | **25 min** | **67 min** |

Three passes of ~8 minutes (three times the 22 Oct walk's images) should take 4-5 hours, most of it matching,
bundle adjustment and training (up to 90k steps): run it overnight.

Disk: 7.6 GB for the 407 s walk, 8.6 GB for the two passes (with the test clip); the Record3D exports
themselves take ~280 MB per minute of recording.

---

## Tested (24 and 27 Oct 2026): one recording

**The 407 s two-lap walk** (`python3 reconstruct.py record3d_2laps`), against M1, the same method run by hand on 22 Oct:

| | M1 by hand | this script, 24 Oct | this script, 27 Oct (frames by motion, adaptive floor) |
|---|---|---|---|
| images | 3,050 (every 8th) | 3,050 (every 8th) | 2,905 (per 6 cm / 3 deg; 212 blurred dropped) |
| loop edges / loop error | 16,812 / 23.3 -> 0.4 cm | 16,765 / 22.9 -> 0.5 cm | 13,367 / 27.2 -> 0.5 cm |
| BA reprojection, points | 0.52 px, 476k | 0.51 px, 472k | 0.51 px, 502k |
| floor within 8 cm / path blocked / rays escaping | 100% / 0.0% / 3.6% | 100% / 0.0% / 3.6% | 100% / 0.0% / 3.8% |
| floor holes | 127 (not patched) | 109 patched | 99 patched |
| BA points within 5 cm of the collider | | 56% | 69% |
| PSNR (held-out frames) | 27.11 dB (2.0M gaussians) | 26.98 dB (1.89M) | **27.71 dB** (2.0M) |
| Isaac check balls on the floor | (views chosen by hand) | 6/6 | 6/6 |
| total time | | 89 min | 93 min |

The B2W + Z1 then drove the start of the loop (through the 0.35 m-clearance kitchenette door) on
`isaac/record3d_2laps_robot.usda` with no contact, on a route planned on `mesh/collider.ply`.

**A 70 s one-way clip** of the same walk (no revisits): loop closure correctly finds nothing to close, the BA,
collider and splat (32.4 dB) work, and the Isaac check finds 6/6 balls on the floor. A re-run after a failure
skipped the finished stages.

## Tested (27 Oct 2026): several passes, test clip

No three-height recording existed yet, so the 22 Oct walk was cut into **two passes and a test clip**, each given
its own ARKit world (lap 2 turned 73 deg about the vertical and shifted 8 m, as a separate recording would be):
lap 1 (229 s), the first 117 s of lap 2, and the last 60 s as a test clip never trained on.

| | two passes (lap 1 + lap 2 cut) |
|---|---|
| pass 2 placed by the alignment | -74.7 deg (true: -73 deg, plus lap 2's own drift and ARKit jumps) |
| loop edges (within 1, between, within 2) | 2,493 / **3,295** / 1,946; loop error 9.3 -> 0.6 cm (p90 1.7 cm) |
| BA | 1.09 -> 0.49 px, 438k points |
| collider | floor 100%, walked path blocked 0.0%, 117 floor holes patched |
| splat (held-out frames of the walk) | 27.83 dB, 1.75M gaussians, 30k steps, 4.2 GB of GPU memory |
| Isaac check | 5/6 balls on the floor, 1 on the kitchenette counter (a view facing a wall) |
| test clip (last 60 s, never trained on) | 121/170 images localised; **22.6 dB** |
| the same test clip on the 24 Oct scene (which trained on those views) | 170/170 localised; 27.7 dB |
| total time | 67 min |

The test clip separates the two cases by 5 dB: it measures what the held-out frames cannot. Its images that could
not be localised (29%) were the views the two-pass scene covers worst (where the splat breaks up in doorways
never seen from that side). Reusing the old scene for the test clip re-extracted its features (its COLMAP database
had been cleaned up) and needed nothing else.

---

## Output

```
<scenes_root>/<scene>/
├── isaac/
│   ├── <scene>.usda             <- the scene (splat + invisible collider)
│   ├── <scene>_robot.usda       <- as a robot's ground (friction 1.0 on the collider)
│   ├── <scene>_visible.usda     collider drawn orange
│   ├── summary.jpg              the Isaac check sheet
│   └── check/                   its renders
├── export/scene.usdz            the splat (NuRec); the stages reference it
├── mesh/collider.ply            the collider (metres, Z-up, floor at 0)
├── eval/                        collider checks: report.json, plan.png, xsec_collider.png
├── images/  sparse/0/  colmap/  the training images, bundle-adjusted model, features and matches
├── record3d/                    poses (phone, passes aligned, loop-closed, refined), LiDAR frames, loop_close.png
├── test/                        the test clip: images, its COLMAP database, localize.json, render/, sheet.jpg
├── runs/                        3DGRUT run(s): checkpoint, metrics.json
├── REPORT.md
├── pipeline.log                 everything every stage printed
└── pipeline_state.json          resume state
```

---

## Known issues this handles for you

Each of these failed **silently** at least once while M1 was built by hand:

1. **ARKit drifts vertically and jumps** when it relocalises (5 times and 0.5 m on one walk). Every frame is
   pinned on its own LiDAR floor; the jumps are removed and the loops closed from image matches.
2. **Starting the pose graph from the de-snapped poses** with a plain robust loss left 44% of the loop edges
   unclosed. It starts from ARKit's own poses, with the loss graduated from near-L2.
3. **Frames with almost no points drift in bundle adjustment** (blank wall, close-up): frames with < 50 points
   keep their phone pose.
4. **pycolmap 4.x database order**: cameras and rigs, then frames, then images, or the triangulation fails its
   `HasDataId` / `RigId` checks.
5. **3DGRUT MCMC collapses on large scenes** with its default regularisation (PSNR 17): large-scene values are set.
6. **The stock NuRec export tilts the scene 90 degrees and overflows float16**: fixed in `tools/export_nurec.py`.
7. **Isaac Lab gives USD terrain no physics material**: `<name>_robot.usda` binds one.
8. **The Isaac container writes as uid 1234**: the check's output folder is made world-writable and chmod-ed after.
9. **Shiny floor leaves holes in the LiDAR mesh** (127 in M1's collider, the largest 36 x 36 cm: enough for a wheel).
   Holes enclosed by floor are patched at floor height (`tools/fill_floor.py`, list in `eval/floor_holes.json`);
   rooms never entered, regions over 2 m^2, and larger holes with geometry below the floor (stairs) stay open.
10. **A check ball dropped straight ahead of the camera lands behind a wall** whenever the phone was turning in a
    doorway, and a rolling ball runs off into places never seen: views are picked looking along the walk, balls go
    on the walked path and do not roll.
11. **A re-run must not mix old and new files**: prep, matching and loop closure start from clean folders, and a
   stage that runs again invalidates the later ones.
12. **ARKit's origin is not where the recording starts** (11 m away on 22 Oct), so passes cannot be aligned by
    starting at the same spot: they are aligned from image matches.
13. **A fixed floor window ("more than 0.7 m below the camera") finds no floor for a phone held at 0.5 m**: the
    window follows the camera's own height. At 1 m it agrees with the old rule within 3 cm on 96% of frames.
14. **One image every N frames over-samples slow walking and standing still**, and keeps blurred frames from fast
    turns: images are taken per distance moved and angle turned, and slots blurred all through are dropped.
15. **More images need more training steps**: 30k steps are ~11 views per image on a 3,000-image walk; the steps
    and every 3DGRUT schedule written for 30k scale with the image count.
16. **A pass that image matches barely tie drifts off the others**, and its walls stand in the others' corridors:
    the three-height office of 27 Oct scored 22.0 dB, mid-pass views rendering a blurred wall right in front of
    the camera, because the 0.5 m pass sat up to 55 cm off (median 8.5 cm; one ICP over the whole pass said
    2.5 cm, the offsets pointing different ways along the walk). The mid pass alone scored 26.1 dB on the same
    views. LiDAR ties in the pose graph bring the low pass to a median of ~4 cm, without the double walls.
17. **A larger gaussian budget is not better**: 4M on that capture left 62% of the gaussians nearly transparent
    (median opacity 0.007, against 0.27 on 22 Oct); the cap is 2M again.
18. **The cap does not bound the starting point cloud**: 3DGRUT starts with one gaussian per bundle-adjustment
    point and the cap only stops it adding, so the dense rebuild of that office (2,783,017 points) trained as
    2.78M gaussians under a 1.89M cap. A cloud above `train.init_share` x cap is subsampled first.

## Troubleshooting

| symptom | cause |
|---|---|
| `no Record3D export found` | export as "EXR + JPG"; the folder needs `metadata.json`, `rgb/`, `depth/` |
| `tools environment not found` | run `./setup_tools_env.sh` |
| `3DGRUT venv not found` | `paths.threedgrut` is wrong, or its `install_env_uv.sh` did not finish |
| loop closure: "no loops to close" | the walk never came back to a place it had seen; fine for a one-way corridor, but drift stays |
| high *walked path blocked* | something solid where you walked: look at `eval/plan.png` (often a person or a door that moved) |
| balls fall through in the check | no floor there: the phone never looked at the floor at that spot (often the first metre of the walk) |
| `CUDA out of memory` in training | the card is shared: the driver waits up to `gpu.wait_min` for `gpu.min_free_gb`; lower `train.cap_max` if needed |
| the Isaac check failed | the stages are still written; read the end of `pipeline.log`, then `--only isaac_check` |
| a run ended early | read the end of `<scene>/pipeline.log` (every failure is logged with its traceback), fix, re-run |
| `pass(es) [N] share no verified views with the others` | that pass was not recorded in the same place, or along another route / the other way round; check `record3d/pass_align.json` |
| `the passes have different image sizes` | record every pass with the same Record3D settings |
| a pass has "no loop edge to another pass" | it is placed by the alignment only; walk the same route in the same direction next time |
| `the test clip could not be placed` | the clip shares too few views with the scene: record it in the same rooms, overlapping the passes |
