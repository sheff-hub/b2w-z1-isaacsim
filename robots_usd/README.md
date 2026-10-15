# robots_usd/: robot assets

Not in git (174 MB, single files up to 58 MB). `./setup.sh` downloads them from the repository's release
(`ASSETS_URL=<release tarball> ./setup.sh`) or copies them (`./setup.sh --assets-from <folder>`); `tools/pack_assets.sh`
makes the tarball for a release.

| Asset | What |
|---|---|
| `b2w_z1_gripper.usd` | **the fetch robot**: B2W + Z1 + gripper, 23 DoF, RealSense D455 (head) and D405 (wrist holder), Ouster OS0-128 |
| `b2w_z1_rl.usd` | B2W + Z1 (no gripper), 22 DoF: the whole-body training robot |
| `b2w_z1.usd` | the hand-placed arm on the body that both are built from |
| `Unitree_B2W/`, `Unitree_B2W_rl/` | Unitree's B2W, and a copy with corrected wheel drives |
| `Unitree_Z1/`, `Unitree_Z1_gripper/`, `z1_gripper_description/` | the Z1 arm, with and without gripper, and the gripper URDF |
| `RealSense/` | Intel's D405 mesh and Isaac Sim's D455 model (Apache-2.0, `LICENSE_realsense-ros`) |
| `*_on_ground.usd`, `*_in_warehouse.usd` | stages with a robot on a ground plane or in the warehouse, for checks |

How they are built: `tools/README.md`.
