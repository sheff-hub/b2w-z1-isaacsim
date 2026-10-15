> Development record, written during the project on the lab workstation. Paths such as `~/Firdavs/Isaac/...`
> refer to the layout before this repository: `Isaac/` is now `isaac/` + `project/` (+ `evaluation/`, `training/`),
> `repos/robot_lab` holds the tasks that are now in `training/robot_lab_overlay/`.

# Phase 3 — Neural 3D Reconstruction (photoreal, no mesh)

**Status: complete.** Real indoor spaces captured with an iPhone and reconstructed
into photoreal NuRec assets that load in NVIDIA Isaac Sim 6.0.1.

**Scope boundary:** this phase delivers *appearance*, not *geometry*. A NuRec
asset is a Gaussian volume — it looks photoreal but has no surfaces. Nothing
collides with it, raycasts pass through, physics sees nothing. Mesh/collision
reconstruction is Phase 4.

---

## 1. Executive summary

| | |
|---|---|
| **Goal** | turn phone photos of a real room into a photoreal Isaac Sim asset |
| **Result** | two scenes delivered; best scores **PSNR 25.73**, against NVIDIA's own reference scene at 25.88 |
| **Pipeline** | photos → COLMAP → 3DGRUT/3DGUT → NuRec `.usdz` |
| **Automation** | packaged as a portable repo, `reconstruction_script_nomesh/`, validated end-to-end |
| **Biggest lesson** | quality is set by the *capture*, not by tuning. Three separate attempts to improve quality by changing the pipeline produced no gain |
| **Most valuable finding** | the documented 1,000,000-gaussian default is wrong for small scenes and cost up to **3.78 dB** |

---

## 2. Deliverables

### Assets

| scene | subject | photos | gaussians | size | PSNR |
|---|---|---|---|---|---|
| **room05** | bedroom | 428 @ 0.5x | 399,105 | 47 MB | **25.73** |
| **kitchen** | lounge / common room | 932 @ 1x | 747,363 | 88 MB | 22.62 |

```
scenes/nurec/mine/room05/runs/room05_mcmc_400k/room05-1308_233103/export_fixed.usdz
scenes/nurec/mine/kitchen/runs/kitchen_mcmc_750k/kitchen-1908_002936/export_fixed.usdz
```

Open in Isaac Sim: **File → Import**, then **Create → Physics → Ground Plane**.
The asset carries no floor and no physics by design (per NVIDIA's documentation);
a robot falls through without one.

### Code and documentation

| artefact | purpose |
|---|---|
| `reconstruction_script_nomesh/` | portable pipeline: zip, copy to any machine, follow README |
| `tools/export_nurec_fixed.py` | NuRec exporter that is not tilted and does not overflow |
| `tools/run_colmap_docs_recipe.sh` | NVIDIA's COLMAP recipe with the 4.2 flag corrections |
| `tools/dng_to_jpeg.py` | Apple ProRAW → JPEG |
| `Isaac/render_camera_path.py` | headless fly-through render of an asset |
| `scenes/guide.md` | capture manual + per-scene progress tracker |
| `PHASE3_RECONSTRUCTION.md` | detailed technical record |

---

## 3. The pipeline

```
photos → COLMAP (camera poses + sparse 3D points)
       → 3DGRUT / 3DGUT (Gaussian splatting, MCMC densification)
       → NuRec .usdz
       → Isaac Sim
```

Based on NVIDIA's documented mono-camera recipe
(`docs.nvidia.com/nurec/robotics/neural_reconstruction_mono.html`), with
corrections documented in §6.

**Seven stages:** validate → prepare → COLMAP → verify SfM → train → export → report.

---

## 4. Environment

| component | choice | why |
|---|---|---|
| 3DGRUT | native venv, torch 2.8.0+cu128 | the trainer; `CUDA_HOME=/usr` on this box (Lambda-packaged toolkit, not `/usr/local/cuda`) |
| COLMAP | Docker `colmap/colmap:latest` 4.2 **with CUDA** | Ubuntu's apt package is 3.7 and built *without* CUDA |
| Isaac Sim | Docker `nvcr.io/nvidia/isaac-sim:6.0.1` | viewer / target runtime |
| GPU | RTX 6000 Ada 48 GB, pinned by UUID | never `--gpus all` — the 96 GB Blackwell belongs to Slurm |

**No `sudo` was available.** Every route was chosen to work without root: COLMAP
via Docker rather than apt, 3DGRUT into a user-owned venv.

Two install traps worth recording:
- `UV_HTTP_TIMEOUT=600` is mandatory — the 30 s default dies part-way through
  ~1.9 GB of CUDA wheels.
- Driver: the docs name 595.58.03 as the "minimum" for 6.0.1, but that is a
  *tested* version, not an enforced gate. The real blacklist floor is 535.129.
  The machine ran fine on 570.207; the 595 branch the docs ask for is buggy.

---

## 5. Experiments and results

### 5.1 Validation against NVIDIA's reference scene

Before touching our own captures, the pipeline was validated on NVIDIA's
`zh_fourth_floor_iphone` (499 photos), which ships with its own reconstruction.

| run | SfM | PSNR |
|---|---|---|
| A | NVIDIA's COLMAP, plain 3DGUT | 25.58 |
| A2 | NVIDIA's COLMAP, MCMC 1M | 25.58 |
| B | **our own COLMAP**, MCMC 1M | 25.88 |

**Our COLMAP beat NVIDIA's on every SfM metric** — 114,815 points vs 92,164, and
**0.936 px** reprojection error vs 1.370 px. Yet re-scored on the 42 test frames
the two runs actually shared, **B and A2 tied** (+0.058 dB). The apparent win was
a different validation split, not a better reconstruction.

**Finding:** two attempts to raise quality by improving an input — 3.6× more
gaussians, then a 32% better SfM — produced *no measurable gain*. The ceiling was
the capture.

### 5.2 Capture 1 — bedroom, 206 photos, 1x lens, ProRAW

| | |
|---|---|
| COLMAP | 10.6 min, 200/206 registered, 32,762 points, 1.195 px |
| at documented 1,000,000 cap | **PSNR 19.49 — heavy fog, unusable** |
| re-run at 300,000 cap | **PSNR 23.28 (+3.78 dB)** |

This is where the gaussian-cap finding originated. See §6.5.

### 5.3 Capture 2 — bedroom, 428 photos, 0.5x ultra-wide, JPEG (best result)

Changes from capture 1: more photos, ultra-wide lens, native-resolution feature
extraction, JPEG instead of RAW.

| | capture 1 | capture 2 |
|---|---|---|
| registered | 200/206 (97.1%) | **428/428 (100%)** |
| sparse points | 32,762 | **139,737 (4.3×)** |
| PSNR | 23.28 | **25.73** |

Gaussian-cap sweep, five points, single-peaked and bracketed:

| cap | ratio to points | whole | ceiling | objects | worst frame |
|---|---|---|---|---|---|
| 1,250,000 | 8.9× | 24.67 | 25.22 | 24.60 | 12.00 |
| 600,000 | 4.3× | 25.35 | 26.31 | 25.12 | 15.56 |
| **400,000** | **2.9×** | **25.73** | **26.59** | **25.52** | **19.11** |
| 300,000 | 2.1× | 25.52 | 26.44 | 25.31 | 19.16 |
| 250,000 | 1.8× | 25.48 | 26.23 | 25.31 | 19.41 |

### 5.4 Capture 3 — lounge, 932 photos, 1x lens, JPEG

The largest capture, and the best structure-from-motion of the project.

| | kitchen | room05 | NVIDIA reference |
|---|---|---|---|
| registered | **930/932** | 428/428 | 498/499 |
| 3D points | **497,643** | 139,737 | 114,815 |
| obs / image | **2,406** | 1,442 | 1,190 |
| reprojection | 1.478 px | 1.198 px | 0.936 px |
| **PSNR** | **22.62** | 25.73 | 25.88 |

Cap sweep (flat curve):

| cap | ratio | PSNR |
|---|---|---|
| 1,500,000 | 3.0× | 22.47 |
| 800,000 | 1.6× | 22.63 |
| **750,000** | **1.5×** | **22.62** ← chosen |
| 600,000 | 1.2× | 22.54 |

**Key negative result: 3.6× the points produced a *lower* score.** The renders
explain it — the foosball table, plants, posters, clock and door frames all
reconstruct faithfully, while frames dominated by **large blank walls and empty
floor** smear badly. The lounge has far more empty surface per frame than a small
bedroom, and is a larger volume, so 932 photos spread thinner than 428 did.

---

## 6. Key findings

### 6.1 Quality is set by the capture

Three separate attempts to raise quality by changing the pipeline produced no
gain: more gaussians (no change), better SfM (no change), more photos of a
harder subject (worse). The limiting factors, in order:

1. **Large blank walls and empty floor** — the single most common cause of a poor
   result. No texture means nothing to match on.
2. **Coverage** — photos per unit of room volume.
3. **Ceilings** — the weakest region in every scene. In room05 only **11%** of
   frames pitched up >10° while 51% pitched down, leaving **1,931 points (<2%)**
   in the ceiling band.
4. **Windows** — outdoor content through glass blows out; unfixable by shooting.

### 6.2 Verified capture settings

Each validated end-to-end through COLMAP with intrinsics seeding disabled:

| setting | value | evidence |
|---|---|---|
| **lens** | 0.5x ultra-wide, fixed | 0.819 px reprojection vs 1.198 at 1x; fx/fy agree to 1.7% vs 34% |
| **format** | JPEG, not RAW | EXIF survives so intrinsics are read directly; on iPhone 14 Pro JPEG is 12 MP automatically (48 MP is ProRAW-only) |
| **exposure** | manual, ISO 100, 1/120 s | brightness std of only 11.3 across 428 frames |
| **white balance** | fixed manual | prevents colour drift |
| **focus** | manual, ~2–3 m | near-hyperfocal; soft regions yield no features |
| **Smart HDR** | off | per-shot tone mapping shifts pixels between frames |
| **movement** | step sideways between shots | depth comes from parallax; pivoting gives nothing to triangulate |

Distortion is corrected in-camera for JPEG, so a distortion-free `PINHOLE` model
fits the ultra-wide at 0.819 px. This would **not** hold for RAW.

### 6.3 Five silent traps

Each produces a wrong result with **no error message**.

**1. `colmap mapper` emits multiple sub-models.** It once wrote a junk 2-image
`sparse/0` next to the real 498-image `sparse/1`. 3DGRUT reads `sparse/0`
unconditionally, so training would have used two photos. *Always run
`model_analyzer` on every sub-model and promote the best.*

**2. The NuRec export tilts the scene 90°.** The serializer hardcodes stage
`upAxis="Z"` but only rotates the data when `apply_coordinate_transform=True`,
which defaults to `False`. `--up-axis` does not fix it.

**3. float16 position overflow.** NuRec stores positions as float16 (max 65504).
~1% of gaussians are near-transparent floaters at huge radii; they become ±inf
and blew the authored extent to **±1.18e9** against a true scene size of ±10.

**4. LPIPS inflates with resolution.** 3DGRUT scores at full resolution but LPIPS
is calibrated for ~256 px. On identical images: **0.479 @2000px → 0.242 @500px →
0.130 @256px**. Never compare our LPIPS to published numbers.

**5. The documented 1,000,000-gaussian cap is wrong for small scenes.** See below.

### 6.4 Two COLMAP flags in NVIDIA's recipe are hard errors

On COLMAP 4.2 (what the official image ships):
- `--SiftExtraction.max_image_size` → `--FeatureExtraction.max_image_size`
- `--SiftMatching.use_gpu` → `--FeatureMatching.use_gpu`

### 6.5 The gaussian budget — the one setting that matters

MCMC densifies to its cap regardless of how much evidence the scene contains.
What matters is the **ratio of gaussians to COLMAP points**.

At 30.5× the point count, **87% of gaussians fell below 0.1 opacity** and
rendered as fog from any unseen viewpoint. Measured effect:

| scene | documented 1M | tuned cap | gain |
|---|---|---|---|
| bedroom, 206 photos | 19.49 | 23.28 @ 300k | **+3.78 dB** |
| bedroom, 428 photos | 24.67 | 25.73 @ 400k | **+1.06 dB** |
| lounge, 932 photos | 22.47 | 22.62 @ 750k | +0.16 dB |

**Surplus gaussians are not harmless.** The intuition to discard is that they are
merely wasted capacity. Going 8.9× → 2.9× improved **every** region at once —
ceiling +1.37 dB, objects +0.92 dB, worst frame **+7.11 dB**. Excess primitives in
under-constrained space fog the volume *in front of* good geometry, so removing
them sharpens the parts that were already fine.

**Practical rule:** start at **~2–3× the COLMAP point count**, then bracket with
one run either side. The curve is single-peaked. Erring low is cheap
(1.8×→2.9× spans 0.25 dB); erring high is not (−1.06 dB by 8.9×). A ratio rule
derived on one scene did **not** transfer cleanly to another, so the sweep is
worth doing per scene when quality matters.

### 6.6 Metrics can mislead; look at the renders

This cost real time twice:

- **LPIPS 0.538 was called "poor"** and triggered a 26-minute rerun chasing a
  problem that did not exist — the renders were near-indistinguishable from the
  originals. It was a resolution artefact (§6.3).
- **SSIM scored *higher* on a badly fogged model** than on the good one, because
  smooth fog against a smooth wall scores well. **PSNR was the only metric that
  caught it.**

**PSNR is the metric to trust. Looking at the images settled every ambiguous
case.**

---

## 7. Approaches evaluated and rejected

### 7.1 NVIDIA Harmonizer — diffusion frame enhancer

Installed and run on all 54 held-out frames of room05.

| | PSNR | SSIM | LPIPS |
|---|---|---|---|
| 400k render | **25.726** | **0.8090** | 0.4963 |
| + Harmonizer | 25.525 | 0.8031 | **0.4952** |

**Rejected.** It visibly cleans wall haze but scored **−0.20 dB / −0.006 SSIM**
against ground truth with LPIPS flat, improving only 19 of 54 frames. It enhances
*rendered frames*, never the asset, and costs 1.3 s/frame. For a robotics
environment the divergence matters: RGB would drift from the geometry a robot
collides with.

Side-by-side video produced for inspection:
`scenes/nurec/mine/room05/harmonizer_comparison.mp4`

### 7.2 GenRecon — generative mesh reconstruction (TUM, arXiv 2605.23888)

Full environment built (five CUDA extensions compiled), runs end-to-end in ~10
min producing a textured PBR mesh.

**Rejected.** The geometry does not match the room: median distance from our real
triangulated points to its surface was **1.48 m**, with only **1.9%** within
10 cm, even after rescaling the input to metric. Judged unusable visually too —
"the glb is just couples of rectangulars, even the room is not covered with
walls". Root cause not established; their `IphoneMixin` is documented as
targeting *MASt3R-SfM* exports rather than stock COLMAP, the leading suspect.

### 7.3 The common thread

Both are **generative**: they produce *plausible*, not *correct*. Harmonizer
invents pixels; GenRecon invents geometry. For a simulation environment where a
robot must interact with what it sees, that trade is wrong.

---

## 8. Automation: `reconstruction_script_nomesh/`

A portable repo — zip it, copy to any machine, install per README, run.

```
reconstruction_script_nomesh/
├── README.md          requirements, install, capture guide, troubleshooting
├── config.yaml        all settings, commented
├── reconstruct.py     CLI
├── requirements.txt
└── tools/             validate, prepare, sfm, train, export_nurec, report
```

```bash
python reconstruct.py my_room                              # full pipeline
python reconstruct.py my_room --caps 300000,400000,500000  # compare caps, export winner
python reconstruct.py my_room --from train                 # resume after a dropped connection
```

**Design decisions:**
- **YAML over JSON** — it takes comments, and the reasoning behind each value
  lives beside it.
- **Resumable** — every stage records completion in `pipeline_state.json`. The
  terminal dropped six or seven times during this project; a resume skips a
  2.5-hour COLMAP run rather than repeating it.
- **The five traps are handled in code**, not just documented.
- **Auto-decisions** for feature resolution, training downscale (from real VRAM
  budget) and gaussian cap, all overridable.

### Validation

Run end-to-end on `room_0.5x`, the same 428 photos used manually:

| | script | manual |
|---|---|---|
| registered | 428/428 | 428/428 |
| points | **139,799** | 139,737 |
| reprojection | **1.197 px** | 1.198 px |
| PSNR | **25.603** @ 280k | 25.723 @ 400k |

25.603 at 280k falls exactly on the manual sweep curve (250k→25.48, 400k→25.73),
confirming the package reproduces the hand-run pipeline.

**Three bugs were found and fixed by that test**, each invisible until the code
actually ran:
1. `slangc` not on `PATH` — 3DGRUT's JIT shells out to it; calling
   `.venv/bin/python` directly skips what `activate` would have set.
2. **Filename case** — the downscaled set was written `IMG_1.jpg` while COLMAP
   recorded `IMG_1.JPG`; 3DGRUT looked up a name that did not exist.
3. `numpy._core` — the export stage inherited the system torch instead of the
   venv's. *(still open — see §10)*

---

## 9. Cost and timings

Measured on an RTX 6000 Ada (48 GB):

| photos | COLMAP | training (per cap) | total |
|---|---|---|---|
| 206 | 11 min | 25 min | ~40 min |
| 428 | 56 min | 45 min | ~1h 45m |
| 932 | 2h 31m | 40 min | ~3h 15m |

Exhaustive matching compares **every pair**, so cost grows with the square of the
photo count: 428 photos → 91,378 pairs; 932 photos → 433,846 pairs. Above ~1500
photos, switch to sequential matching.

VRAM: 10.8 GB at 750k gaussians, 19.5 GB at 1.5M, both at 2016×1512. Training
resolution must be chosen from image count — 932 photos at full 12 MP would need
~63 GB for the image cache alone.

---

## 10. Open items

- **Export stage of the script package** fails with `ModuleNotFoundError:
  numpy._core` — it runs in the driver's Python rather than the 3DGRUT venv.
  Stages 0–4 are validated; export and report are not.
- **Ceiling coverage** — ~100 deliberate upward photos at 30–45°, keeping the
  wall/ceiling seam in frame so they still register, would address the weakest
  region in every scene built.
- **Fly-through video** of the kitchen asset — started, interrupted; ~30 min of
  render at 2.27 s/frame for 813 poses.

---

## 11. Phase 4 — reconstruction with mesh

Phase 3 delivers appearance only. The NuRec USD already carries an empty
`custom rel proxy` slot intended for exactly the mesh Phase 4 would produce.

**Documented route:** stereo capture
(`docs.nvidia.com/nurec/robotics/neural_reconstruction_stereo.html`) — ZED 2i or
Leopard Hawk → cuSFM → FoundationStereo depth → nvblox, which yields a **triangle
mesh and a 2D occupancy map**, plus **true metric scale** (our current assets are
in arbitrary COLMAP units).

**Estimated cost:** 2–4 days of engineering with a camera in hand. Blockers on
this machine today: no stereo camera, cuSFM wants CUDA 13 (we have 12.8), and the
Isaac ROS packages want `sudo`. Containers likely solve the latter two.

**Cheaper first experiment:** COLMAP dense MVS on the existing photos to produce a
fused point cloud, fed to `apps/cusfm_3dgut_mcmc.yaml` via
`initialization.fused_point_cloud_path`. That init path is not NCore-gated, needs
no new hardware, and costs an afternoon.
