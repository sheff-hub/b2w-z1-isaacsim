# training/: our Isaac Lab tasks

The tasks are an overlay on [robot_lab](https://github.com/fan-ziqi/robot_lab) 2.3.2: `robot_lab_overlay/` holds our
files at the paths they have inside robot_lab. `setup.sh` links them into `repos/robot_lab` (relative symlinks, so an
edit made "in robot_lab" lands here and is tracked by git), or copies them if `paths.robot_lab` is outside the
repository.

```
robot_lab_overlay/source/robot_lab/robot_lab/tasks/manager_based/
  locomotion/velocity/config/wheeled/unitree_b2w/
    __init__.py               every task id below (replaces robot_lab's file: it keeps robot_lab's own B2W tasks)
    agents/rsl_rl_ppo_cfg.py  PPO settings (replaces robot_lab's file, adds the ETH fine-tune runner)
    arm_cargo_env_cfg.py      the B2W + Z1 robot (b2w_z1_rl.usd) for the locomotion and whole-body tasks
    whole_body_env_cfg.py     whole-body tasks: drive + reach with one policy, world-fixed targets, RFM
    rfm.py                    the Reward Fusion Module (Jiang et al.) with our shoulder-sphere phase rule,
                              and incremental arm actions (DeltaJointPositionAction)
    world_pose_command.py     6D hand targets fixed in the world, sampled outside the robot's own body
    self_clearance.py         self-collision handling for the arm
    loco_reach_env_cfg.py     two-policy drive + reach (an earlier approach)
    eth_finetune_env_cfg.py   fine-tuning ETH's walking policy for a wider stance
    warehouse_env_cfg.py      the warehouse terrain
    fetch_env_cfg.py          the fetch scenes, built from scenes/*.yaml (not training tasks)
  manipulation/reach_b2w_z1/  arm-only reaching (the first reach policy)
```

## Tasks

| Task id | What | Final checkpoint (`policies/`) |
|---|---|---|
| `RobotLab-Isaac-RFM-Delta-Sphere-Unitree-B2W-Z1-v0` | **one 6D command**: drive, crouch and reach a hand pose 1–3 m away; shoulder-sphere phase rule, incremental arm actions | `b2w_z1_rfm/..._rfm19_delta_sphere/model_37600.pt` (`live.sh goto_mine`) |
| `RobotLab-Isaac-RFM-Delta-Paper-Unitree-B2W-Z1-v0` | the same with the paper's distance phase rule | `b2w_z1_rfm/..._rfm19_delta_paper/model_36700.pt` (`live.sh goto_paper`) |
| `RobotLab-Isaac-WholeBody-WorldTarget-Unitree-B2W-Z1-v0` | velocity command + hand target fixed in the world | `b2w_z1_wholebody_worldtarget/..._wb9/model_7700.pt` |
| `RobotLab-Isaac-WholeBody-Tidy-Unitree-B2W-Z1-v0` (and variants) | the same with a tidy posture | `b2w_z1_wholebody_tidy/..._wb15/model_9500.pt`, `..._wb12/model_6200.pt` |
| `RobotLab-Isaac-EthFinetuneV5-Flat-Unitree-B2W-Z1-v0` | ETH's walking policy, wider rear stance (`train_eth_ft.py`) | `eth/policy_stance_v5_600.pt` |
| `RobotLab-Isaac-Velocity-Flat-Unitree-B2W-v0` (robot_lab's) | walking, bare B2W; drives the robot with the arm too (`Velocity-Flat-Unitree-B2W-Z1-v0`, `--loco ours`) | `unitree_b2w_flat/reference_dda1510/model_2600.pt` (inherited from a colleague) |
| `RobotLab-Isaac-Reach-B2W-Z1-v0` | arm-only reaching | `b2w_z1_reach/..._v2/model_999.pt` |
| `RobotLab-Isaac-Fetch-Warehouse-...`, `...Fetch3-...` | the fetch scenes (driven by `project/fetch_box.py`, not trained) | — |

The other `WholeBody-*`, `RFM-*` and `EthFinetune*` ids are the experiments that led there (one change per run);
`docs/notes/phase4_control.md` records each and what it showed.

## Training

```bash
python3 run.py train --task RobotLab-Isaac-RFM-Delta-Sphere-Unitree-B2W-Z1-v0          # 4096 robots, headless
python3 run.py train --task <id> --num_envs 64 --max_iterations 2                       # a quick smoke test
FOLLOW=1 python3 run.py train --task <id>                                              # attached to the log
python3 run.py train --play --task <id> --checkpoint /workspace/policies/<...>.pt --num_envs 16 --stream
```

Checkpoints and TensorBoard logs go to `repos/robot_lab/logs/rsl_rl/<experiment>/<time>/`. About 4000 iterations
take two hours on an RTX 6000 Ada.

**How the final RFM policy was made.** It is the end of a chain, not one run: RFM training up to rfm14
(~34,000 iterations, `docs/notes/phase4_control.md`), then rfm14's network with the arm's output weights zeroed
(`calm_init.py --zero_arm_head`) and ~3,800 iterations on `RFM-Delta-Sphere` with incremental arm commands (rfm19).
To continue from the shipped checkpoint, copy it into a log folder and resume:

```bash
mkdir -p repos/robot_lab/logs/rsl_rl/b2w_z1_rfm/rfm19_sphere
cp policies/b2w_z1_rfm/2026-10-08_19-58-51_rfm19_delta_sphere/model_37600.pt repos/robot_lab/logs/rsl_rl/b2w_z1_rfm/rfm19_sphere/
chmod -R a+rwX repos/robot_lab/logs
python3 run.py train --task RobotLab-Isaac-RFM-Delta-Sphere-Unitree-B2W-Z1-v0 --resume --load_run rfm19_sphere --checkpoint model_37600.pt
```

## Helpers

| File | What it does |
|---|---|
| `train_eth_ft.py` | Fine-tune ETH's walking policy (started from `eth/policy_force_new_rslrl.npz`) |
| `calm_init.py` | Starting checkpoint with the arm's output head zeroed or its action std capped |
| `add_obs_columns.py` | Extend a trained actor-critic with new observation inputs, initialised to zero |
| `inspect_task.py` | Print what a policy sees and commands in a task (joint order of every term): use it after changing the robot |
| `PHASE5_ETH_STANCE_FINETUNE.md` | How the stance fine-tune was made and evaluated |
