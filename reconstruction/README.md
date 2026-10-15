# reconstruction_script_nomesh

Photoreal 3D reconstruction of indoor scenes from phone photos, producing a
**NuRec `.usdz`** that opens in NVIDIA Isaac Sim.

```
photos  ->  COLMAP (poses + sparse points)  ->  3DGRUT / 3DGUT  ->  NuRec .usdz
```

**Appearance only — this produces no mesh.** The output is a Gaussian volume: it
looks photoreal but has no surfaces, so nothing collides with it and raycasts
pass through. Add a ground plane in Isaac Sim, or use the stereo/nvblox route if
you need real geometry.

---

## Requirements

### Hardware
- NVIDIA GPU with RT cores, **16 GB VRAM minimum**, 24 GB+ comfortable
  (measured: 10.8 GB at 750k gaussians, 19.5 GB at 1.5M, both at 2016x1512)
- ~50 GB free disk per scene
- 32 GB system RAM

### System software
| requirement | notes |
|---|---|
| Linux | tested on Ubuntu 22.04 |
| NVIDIA driver >= 535.129 | the enforced floor; the docs' "minimum" is a *tested* version, not a hard gate |
| CUDA toolkit 12.x | `nvcc --version`. Note `CUDA_HOME` may be `/usr`, not `/usr/local/cuda` |
| Docker + NVIDIA Container Toolkit | `docker info \| grep nvidia` must print something |
| Python 3.10+ | for this driver |
| gcc/g++ | to build 3DGRUT's CUDA extensions |

**No `sudo` is required** if Docker is already set up and you can reach a CUDA
toolkit. Every step here was built on a machine without root.

### 1. This driver
```sh
pip install -r requirements.txt          # PyYAML, Pillow, numpy
```
Any `python3` runs the driver. It does not need torch or 3DGRUT itself: COLMAP runs in Docker, and training and
export run in 3DGRUT's own venv, each as a subprocess whose output (errors included) goes into `pipeline.log`.

### 2. COLMAP (Docker — do not use apt)
```sh
docker pull colmap/colmap:latest
```
Ubuntu's `colmap` package is 3.7 and is built **without CUDA**, which makes
matching an order of magnitude slower. The official image is 4.2+ with CUDA.

Verify:
```sh
docker run --rm --gpus device=<gpu.uuid> colmap/colmap:latest colmap -h | grep -i cuda
# expect: "COLMAP ... with CUDA"
```

### 3. 3DGRUT
```sh
cd <repository>/repos            # paths.threedgrut defaults to repos/3dgrut
git clone --recursive https://github.com/nv-tlabs/3dgrut.git
cd 3dgrut
git checkout a37ef72 && git submodule update --init --recursive   # the version the results were made with
curl -LsSf https://astral.sh/uv/install.sh | sh          # installer requires uv
export CUDA_HOME=/usr           # or /usr/local/cuda — wherever nvcc lives
export UV_HTTP_TIMEOUT=600      # IMPORTANT: the 30 s default dies mid-download
./install_env_uv.sh 3dgrut
```
`UV_HTTP_TIMEOUT` is not optional. The install pulls ~1.9 GB of CUDA wheels and
the default timeout kills it part-way through. The venv is resumable, so a
re-run continues rather than starting over.

Verify:
```sh
cd 3dgrut && source .venv/bin/activate
python -c "import torch, kaolin, tinycudann; print(torch.cuda.is_available())"
```

Elsewhere? Point `paths.threedgrut` in the repository's `config.yaml` at the checkout.

---

## Usage

Put a folder of photos under `paths.dataset_root`, write a short scene file for it (`scenes/my_room.yaml`, see
`scenes/README.md`), then:

```sh
python3 reconstruct.py my_room
```

That runs everything: validate -> prepare -> COLMAP -> train -> export -> report.
Expect **1-4 hours** depending on photo count (see Timings). It runs unattended; to keep it alive when the terminal
closes, start it in `tmux` (or with `nohup ... &`) and follow `<scene>/pipeline.log`. The two rooms of this project:

```sh
python3 reconstruct.py home       # home (bedroom + kitchen corner), 428 photos, ~1h 45m
python3 reconstruct.py kitchen    # lounge, 932 photos, ~3h 15m
python3 reconstruct.py --list-scenes
```

Without a scene file, a bare photo folder name still works: `python3 reconstruct.py my_photos` (defaults only).

### Comparing gaussian counts

The single most consequential setting. To reproduce a 300k/400k/500k comparison:

```sh
python3 reconstruct.py my_room --caps 300000,400000,500000
```

Each cap is trained, scored on the same held-out frames, and the **best is
exported automatically**. The comparison table lands in `REPORT.md`. Or set it
in `config.yaml`:

```yaml
train:
  caps: [300000, 400000, 500000]
```

### Resuming

Connections drop; long runs should not have to restart.

```sh
python3 reconstruct.py my_room --from train    # skip COLMAP, it is already done
python3 reconstruct.py my_room --only export
python3 reconstruct.py my_room --force         # ignore state, redo everything
```

Progress lives in `<scene>/pipeline_state.json`.

### Other overrides
```sh
python3 reconstruct.py my_room --downsample 2       # training resolution
python3 reconstruct.py my_room --max-image-size 4032 # COLMAP feature resolution
python3 reconstruct.py my_room --name lounge        # output name
```

---

## Configuration

Settings are layered, later wins:

1. `config.yaml` (this folder): defaults. Where things live on this machine and the GPU come from the
   repository's `config.yaml` (one file for the whole project): photos in `<paths.nurec>/mine/dataset/<folder>`,
   scenes built in `<paths.nurec>/mine/<name>`, 3DGRUT at `paths.threedgrut`, the GPU from `gpu.uuid`.
2. `scenes/<name>.yaml`: one short file per room: its photo folder, output name and the settings it needs.
3. the command line, e.g. `--caps`, `--downsample`.

The values worth knowing:

| key | meaning |
|---|---|
| `gpu.uuid` (repository `config.yaml`) | **pin the GPU by UUID.** `nvidia-smi --query-gpu=uuid,name --format=csv`. Use `all` only on a machine you own |
| `train.caps` | list of gaussian caps to compare; `null` for a single run |
| `train.cap_ratio` | single-run cap as a multiple of the COLMAP point count |
| `colmap.max_image_size` | feature resolution; `null` auto-picks |
| `train.downsample_factor` | training resolution; `null` auto-picks from VRAM |

---

## Capture guide

Quality is set by the photos, far more than by any setting here. Two attempts to
raise quality by tuning the pipeline (3.6x more gaussians, then a 32% better
SfM) produced **no measurable gain**.

| setting | do this | why |
|---|---|---|
| **count** | **400-900 photos** | below ~200 coverage is too thin |
| **lens** | one lens, fixed | mixing lenses silently breaks single-camera SfM |
| **format** | **JPEG**, not RAW | keeps EXIF, so intrinsics are read directly; RAW needs conversion and loses it |
| **exposure** | **manual, locked** (e.g. ISO 100, 1/120 s) | auto-exposure drift is uncorrectable later |
| **white balance** | fixed manual | prevents colour drift between frames |
| **focus** | manual, ~2-3 m | soft regions yield no features |
| **HDR** | off | per-shot tone mapping shifts pixels between frames |
| **overlap** | ~60% between consecutive shots | |
| **movement** | **step sideways between shots** | depth comes from parallax; pivoting on the spot gives almost nothing to triangulate |
| **ceilings** | shoot 25-30% of frames tilted **up 30-45 deg** | ceilings are the weakest region in every scene we built |

Hard subjects: mirrors, windows, chrome, glossy worktops, and **large blank
walls** — the single most common cause of a poor result. Shoot *along* blank
walls at a grazing angle so skirting, sockets and door frames enter the frame.

---

## Interpreting the results

- **PSNR is the metric to trust.** Stable across evaluation resolution.
- **LPIPS is inflated ~3.7x** here. It is scored at full resolution but
  calibrated for ~256 px. Same images: 0.479 @2000px, 0.242 @500px, 0.130
  @256px. Compare runs to each other, never to published numbers.
- **SSIM can mislead.** A badly fogged model once scored *higher* SSIM than the
  good one — smooth fog against a smooth wall scores well.
- **Look at `comparison.png`.** Metrics misled us twice; the images did not.

Reference quality:

| scene | photos | PSNR |
|---|---|---|
| NVIDIA reference scene | 499 | 25.88 |
| bedroom, 0.5x lens | 428 | 25.73 |
| lounge, 1x lens | 932 | 22.62 |

Below ~23 dB, suspect the capture (blank walls, thin coverage), not the pipeline.

---

## Timings

Measured on an RTX 6000 Ada (48 GB):

| photos | COLMAP | training (per cap) | total |
|---|---|---|---|
| 206 | 11 min | 25 min | ~40 min |
| 428 | 56 min | 45 min | ~1h 45m |
| 932 | 2h 31m | 40 min | ~3h 15m |

Matching compares **every pair**, so cost grows with the square of the photo
count. Above ~1500 photos switch `colmap.matcher` to `sequential`.

---

## Output

```
<scenes_root>/<scene>/
├── images/                     source frames (or a link to them)
├── images_2/                   downscaled set used for training
├── sparse/0/                   COLMAP model (best sub-model, promoted here)
├── scene.usdz                  <- the asset (a link to the exported file below)
├── runs/gs_<cap>/<timestamp>/
│   ├── ckpt_last.pt
│   ├── export_fixed.usdz       the exported asset
│   └── export_check.json       upright? finite bounds?
├── comparison.png              held-out renders vs ground truth
├── REPORT.md                   metrics and how to read them
├── pipeline.log
└── pipeline_state.json         resume state
```

Open the `.usdz` in Isaac Sim: **File -> Import**, then
**Create -> Physics -> Ground Plane**. The asset has no floor by design; a robot
falls through without one.

---

## Known issues this handles for you

Each of these fails **silently** if unhandled:

1. **`colmap mapper` emits several sub-models.** 3DGRUT reads `sparse/0`
   blindly. We once got a junk 2-image `sparse/0` beside the real 498-image
   `sparse/1`. Every sub-model is checked and the best is promoted.
2. **The stock NuRec export tilts the scene 90 degrees** — the stage claims
   Z-up while the data stays Y-up. `--up-axis` does *not* fix it.
3. **float16 position overflow.** NuRec stores positions as float16 (max
   65504); stray floaters become +/-inf and destroy the asset bounds. They are
   pruned before export.
4. **The documented 1,000,000 gaussian cap is wrong for small scenes.** At 8.9x
   the COLMAP point count, 87% of gaussians fell below 0.1 opacity and rendered
   as fog. The cap is derived from the point count instead.
5. **Two COLMAP flags in NVIDIA's published recipe are hard errors** on COLMAP
   4.2 (`SiftExtraction.max_image_size`, `SiftMatching.use_gpu`). The current
   names are used.
6. **The export must run in 3DGRUT's venv**, not in the driver's python: the checkpoint is a pickle written with
   numpy 2 (an older numpy fails with `No module named 'numpy._core'`), the export compiles a kernel with the venv's
   `slangc`, and the asset check needs `pxr`. Until October 2026 the export ran inside the driver and its error went
   only to the terminal, so a run could end after training with no asset and nothing in the log.
7. **The training resolution is remembered.** An export- or report-only run uses the factor the model was trained
   at instead of choosing again from the GPU memory free at that moment.

## Troubleshooting

| symptom | cause |
|---|---|
| `3DGRUT venv not found` | `paths.threedgrut` is wrong, or `install_env_uv.sh` did not finish |
| install dies mid-download | `export UV_HTTP_TIMEOUT=600` and re-run; it resumes |
| CUDA OOM during training | lower `train.downsample_factor`, or lower the cap |
| few images registered | insufficient overlap — more photos, not more tuning |
| asset loads tilted | exported without `reorient`; re-run the export stage |
| `docker: permission denied` | add yourself to the `docker` group, then log out and back in |
| `FileNotFoundError: 'slangc'` | 3DGRUT's JIT needs `.venv/bin` on `PATH`; the driver sets this for training and export, but if you invoke `train.py` yourself, `source .venv/bin/activate` first |
| `No module named 'numpy._core'` | a 3DGRUT checkpoint opened with the system python; the driver runs the export in the venv, so this only appears when calling `tools/export_nurec.py` by hand: use `.venv/bin/python` |
| `CUDA out of memory` | the card is shared: someone else's job took the memory. Before training and evaluation the driver now waits up to `gpu.wait_min` for `gpu.min_free_gb` and logs who holds the card. A model that finished training is kept even if the evaluation after it runs out of memory; re-running scores it again without retraining |
| a run ended with no asset | read the end of `<scene>/pipeline.log` (every failure is logged there with its traceback), fix it, re-run the same command |
