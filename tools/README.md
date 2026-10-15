# tools/: robot assets and datasets

## Robot assets (`robots_usd/`)

The robot is built from Unitree's descriptions by scripts, in this order. They run on the host in any Python with
`pxr` (the 3DGRUT venv has it: `repos/3dgrut/.venv/bin/python`), except `convert_z1_gripper.py`, which needs Isaac Sim
(`./isaac/run_script.sh tools/convert_z1_gripper.py`).

| Script | Makes |
|---|---|
| `make_z1_gripper_urdf.py` | a Z1-with-gripper URDF that imports: local mesh paths (`package://` silently gave an invisible gripper) and real jaw colliders |
| `convert_z1_gripper.py` (+ `usd_fixes.py`) | that URDF as USD, checked for missing meshes |
| `build_b2w_rl_asset.py` | `Unitree_B2W_rl/`: the B2W with its drives corrected (the wheels shipped with position drives) |
| `build_b2w_z1_rl_asset.py` | `b2w_z1_rl.usd`: B2W + Z1 as one articulation from the hand-placed `b2w_z1.usd` (training robot) |
| `make_d405_usd.py` | `RealSense/d405.usd` from Intel's mesh |
| `build_b2w_z1_gripper_asset.py` (+ `wrist_d405.py`) | `b2w_z1_gripper.usd`: the fetch robot, 23 DoF, arm 11 cm back, gripper, D455 + D405 on the printed holder, OS0 lidar, self-collision filtered |

`b2w_z1.usd` (where the arm sits on the body) was placed by hand in Isaac Sim; the scripts above read it, never write it.

## Datasets (`datasets/`)

Host Python with numpy, pandas, pyarrow, imageio (+ matplotlib for plots); `pip install -r tools/requirements.txt`.

| Script | Does |
|---|---|
| `raw_to_lerobot.py` | raw recordings (`datasets/raw/<run>`, from `fetch_box.py --record`) -> LeRobot v2.1 (`run.py dataset` calls it) |
| `check_lerobot.py` | load a converted dataset with the official LeRobot loader (needs `lerobot`) |
| `plot_episode.py` | one review image per recorded episode |

## Other

`dng_to_jpeg.py`: iPhone RAW (DNG) photos to JPEG for reconstruction (JPEG reconstructed better, see `reconstruction/`).
