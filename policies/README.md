# policies/: trained checkpoints

RSL-RL checkpoints (PPO actor-critic), kept at `<experiment>/<run>/model_<iteration>.pt` as robot_lab wrote them.
Inside the containers this folder is `/workspace/policies/`.

| Checkpoint | Task | Used by |
|---|---|---|
| `b2w_z1_rfm/2026-10-08_19-58-51_rfm19_delta_sphere/model_37600.pt` | `RobotLab-Isaac-RFM-Delta-Sphere-Unitree-B2W-Z1-v0` | **the final whole-body policy**: one 6D hand target, shoulder-sphere phase rule (`live.sh goto_mine`, report Fig. 7) |
| `b2w_z1_rfm/2026-10-08_20-28-32_rfm19_delta_paper/model_36700.pt` | `RobotLab-Isaac-RFM-Delta-Paper-Unitree-B2W-Z1-v0` | the same with the paper's phase rule (`live.sh goto_paper`) |
| `b2w_z1_wholebody_worldtarget/2026-09-23_13-25-50_wb9/model_7700.pt` | `RobotLab-Isaac-WholeBody-WorldTarget-Unitree-B2W-Z1-v0` | velocity command + world-fixed hand target (`show.sh`, `goto.py`) |
| `b2w_z1_wholebody_tidy/2026-09-24_06-32-21_wb15/model_9500.pt`, `..._wb12/model_6200.pt` | `RobotLab-Isaac-WholeBody-Tidy*-Unitree-B2W-Z1-v0` | the tidy-posture versions (`show.sh`) |
| `b2w_z1_reach/2026-09-22_16-02-34_v2/model_999.pt` | `RobotLab-Isaac-Reach-B2W-Z1-v0` | arm-only reaching (`loco_reach.py`) |
| `unitree_b2w_flat/reference_dda1510/model_2600.pt` | robot_lab's `RobotLab-Isaac-Velocity-Flat-Unitree-B2W-v0` | robot_lab walking policy, inherited from a colleague (`--loco ours`) |

The walking controller used by the fetch pipeline is ETH's, in `eth/`. Training them: `training/README.md`.
