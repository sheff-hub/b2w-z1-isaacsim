# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-Unitree-B2W-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeB2WFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2WFlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeB2WFlatTrainerCfg",
    },
)

# Local addition: flat-task policy, warehouse scene. Shares the flat PPO config
# so checkpoints from the flat task load directly.
gym.register(
    id="RobotLab-Isaac-Velocity-Warehouse-Unitree-B2W-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.warehouse_env_cfg:UnitreeB2WWarehouseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2WFlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeB2WFlatTrainerCfg",
    },
)

# Local addition: flat task, B2W with the Z1 arm held as cargo. Shares the flat
# PPO config so flat-task checkpoints load directly.
gym.register(
    id="RobotLab-Isaac-Velocity-Flat-Unitree-B2W-Z1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.arm_cargo_env_cfg:UnitreeB2WZ1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2WFlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeB2WFlatTrainerCfg",
    },
)

# Local addition: driving + reaching with the two existing policies side by side
# (project/loco_reach.py). Not a training task.
gym.register(
    id="RobotLab-Isaac-LocoReach-Flat-Unitree-B2W-Z1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loco_reach_env_cfg:UnitreeB2WZ1LocoReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2WFlatPPORunnerCfg",
    },
)

# Local addition, Phase 5: warehouse fetch -- drive to a table (model_2600 +
# scripted go-to-pose), pick a box (differential IK + gripper). Driven by
# project/fetch_box.py. Not a training task.
gym.register(
    id="RobotLab-Isaac-Fetch-Warehouse-Unitree-B2W-Z1-Gripper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fetch_env_cfg:UnitreeB2WZ1GripperFetchEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2WFlatPPORunnerCfg",
    },
)

# Local addition: ONE policy for all 22 joints -- drive and reach together.
gym.register(
    id="RobotLab-Isaac-WholeBody-Flat-Unitree-B2W-Z1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.whole_body_env_cfg:B2WZ1WholeBodyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.whole_body_env_cfg:B2WZ1WholeBodyPPORunnerCfg",
    },
)

# Local addition: wb2 -- whole-body with double forward-tracking weight.
gym.register(
    id="RobotLab-Isaac-WholeBody-Track-Unitree-B2W-Z1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.whole_body_env_cfg:B2WZ1WholeBodyTrackEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.whole_body_env_cfg:B2WZ1WholeBodyTrackPPORunnerCfg",
    },
)

# Local addition: wb3 -- whole-body with double wheel action scale.
gym.register(
    id="RobotLab-Isaac-WholeBody-WheelScale-Unitree-B2W-Z1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.whole_body_env_cfg:B2WZ1WholeBodyWheelScaleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.whole_body_env_cfg:B2WZ1WholeBodyWheelScalePPORunnerCfg",
    },
)

# Local addition: wb4 -- whole-body with a forward-dominant command distribution.
gym.register(
    id="RobotLab-Isaac-WholeBody-Forward-Unitree-B2W-Z1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.whole_body_env_cfg:B2WZ1WholeBodyForwardEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.whole_body_env_cfg:B2WZ1WholeBodyForwardPPORunnerCfg",
    },
)

# Local addition: wb5 -- whole-body with halved effort/smoothness penalties.
gym.register(
    id="RobotLab-Isaac-WholeBody-SoftPenalty-Unitree-B2W-Z1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.whole_body_env_cfg:B2WZ1WholeBodySoftPenaltyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.whole_body_env_cfg:B2WZ1WholeBodySoftPenaltyPPORunnerCfg",
    },
)

# Local addition: wb6 -- whole-body with wheel drive damping 1.0 -> 20.
gym.register(
    id="RobotLab-Isaac-WholeBody-StiffWheel-Unitree-B2W-Z1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.whole_body_env_cfg:B2WZ1WholeBodyStiffWheelEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.whole_body_env_cfg:B2WZ1WholeBodyStiffWheelPPORunnerCfg",
    },
)

# Local addition: wb7 -- soft penalties + stiff wheel drive (best-config candidate).
gym.register(
    id="RobotLab-Isaac-WholeBody-Best-Unitree-B2W-Z1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.whole_body_env_cfg:B2WZ1WholeBodyBestEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.whole_body_env_cfg:B2WZ1WholeBodyBestPPORunnerCfg",
    },
)

# Local addition: wb8 -- wb7 plus a sharper velocity-tracking reward kernel.
gym.register(
    id="RobotLab-Isaac-WholeBody-SharpTrack-Unitree-B2W-Z1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.whole_body_env_cfg:B2WZ1WholeBodySharpTrackEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.whole_body_env_cfg:B2WZ1WholeBodySharpTrackPPORunnerCfg",
    },
)

# Local addition: wb9 -- whole-body with WORLD-frame hand targets.
gym.register(
    id="RobotLab-Isaac-WholeBody-WorldTarget-Unitree-B2W-Z1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.whole_body_env_cfg:B2WZ1WholeBodyWorldTargetEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.whole_body_env_cfg:B2WZ1WholeBodyWorldTargetPPORunnerCfg",
    },
)

# Local addition: wb10 -- world-frame targets at full driving speed.
gym.register(
    id="RobotLab-Isaac-WholeBody-WorldDrive-Unitree-B2W-Z1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.whole_body_env_cfg:B2WZ1WholeBodyWorldDriveEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.whole_body_env_cfg:B2WZ1WholeBodyWorldDrivePPORunnerCfg",
    },
)

# Local addition: wb11 -- world targets at half speed (the merge attempt).
gym.register(
    id="RobotLab-Isaac-WholeBody-WorldDriveHalf-Unitree-B2W-Z1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.whole_body_env_cfg:B2WZ1WholeBodyWorldDriveHalfEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.whole_body_env_cfg:B2WZ1WholeBodyWorldDriveHalfPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Unitree-B2W-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeB2WRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2WRoughPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeB2WRoughTrainerCfg",
    },
)

# Local addition: wb12 -- posture rewards, real self-collisions, and a target box
# taken from the measured workspace (evaluation/workspace.py).
gym.register(
    id="RobotLab-Isaac-WholeBody-Tidy-Unitree-B2W-Z1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.whole_body_env_cfg:B2WZ1WholeBodyTidyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.whole_body_env_cfg:B2WZ1WholeBodyTidyPPORunnerCfg",
    },
)

# Local addition: wb13 -- wb12 with wb9's action-rate weight (contingency).
gym.register(
    id="RobotLab-Isaac-WholeBody-TidyFree-Unitree-B2W-Z1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.whole_body_env_cfg:B2WZ1WholeBodyTidyFreeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.whole_body_env_cfg:B2WZ1WholeBodyTidyFreePPORunnerCfg",
    },
)

# Local addition: wb14 -- wb12 with a stronger, earlier mast-clearance penalty.
gym.register(
    id="RobotLab-Isaac-WholeBody-TidyHard-Unitree-B2W-Z1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.whole_body_env_cfg:B2WZ1WholeBodyTidyHardEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.whole_body_env_cfg:B2WZ1WholeBodyTidyHardPPORunnerCfg",
    },
)

# Local addition: wb15 -- wb14's clearance weight with a margin that matches the
# target sampler, so no target sits inside the penalised shell.
gym.register(
    id="RobotLab-Isaac-WholeBody-TidyFirm-Unitree-B2W-Z1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.whole_body_env_cfg:B2WZ1WholeBodyTidyFirmEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.whole_body_env_cfg:B2WZ1WholeBodyTidyFirmPPORunnerCfg",
    },
)

# Local addition: wb16 -- the RFM method of Jiang et al. (arXiv:2412.03012) in
# full: 6D omni-directional targets, reward prioritisation, loco-mani fusion.
gym.register(
    id="RobotLab-Isaac-RFM-Unitree-B2W-Z1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.whole_body_env_cfg:B2WZ1RFMEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.whole_body_env_cfg:B2WZ1RFMPPORunnerCfg",
    },
)

# Local addition: the phase-criterion comparison -- identical tasks except for
# how locomotion vs manipulation is decided (see whole_body_env_cfg.py).
for _mode, _cls in (("Sphere", "B2WZ1RFMSphereEnvCfg"), ("Paper", "B2WZ1RFMPaperEnvCfg")):
    gym.register(
        id=f"RobotLab-Isaac-RFM-{_mode}-Unitree-B2W-Z1-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.whole_body_env_cfg:{_cls}",
            # same experiment as the base RFM task, so both can resume from it
            "rsl_rl_cfg_entry_point": f"{__name__}.whole_body_env_cfg:B2WZ1RFMPPORunnerCfg",
        },
    )

# Local addition, Phase 5: fine-tune ETH's SRU locomotion policy for a normal stance width
# (its rear wheels pull together). Weights are loaded by training/train_eth_ft.py.
gym.register(
    id="RobotLab-Isaac-EthFinetune-Flat-Unitree-B2W-Z1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.eth_finetune_env_cfg:UnitreeB2WZ1EthFinetuneEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2WZ1EthFinetunePPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-EthFinetuneV2-Flat-Unitree-B2W-Z1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.eth_finetune_env_cfg:UnitreeB2WZ1EthFinetuneV2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2WZ1EthFinetunePPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-EthFinetuneV3-Flat-Unitree-B2W-Z1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.eth_finetune_env_cfg:UnitreeB2WZ1EthFinetuneV3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2WZ1EthFinetunePPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-EthFinetuneV4-Flat-Unitree-B2W-Z1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.eth_finetune_env_cfg:UnitreeB2WZ1EthFinetuneV4EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2WZ1EthFinetunePPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-EthFinetuneV5-Flat-Unitree-B2W-Z1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.eth_finetune_env_cfg:UnitreeB2WZ1EthFinetuneV5EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2WZ1EthFinetunePPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-EthFinetuneV6-Flat-Unitree-B2W-Z1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.eth_finetune_env_cfg:UnitreeB2WZ1EthFinetuneV6EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2WZ1EthFinetunePPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-EthFinetuneV8-Flat-Unitree-B2W-Z1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.eth_finetune_env_cfg:UnitreeB2WZ1EthFinetuneV8EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2WZ1EthFinetunePPORunnerCfg",
    },
)

# Local addition: rfm15 -- the same two tasks with the arm's bang-bang hold
# trained out (arm action/torque rate, hand stillness, cleaner arm velocity obs).
# CalmFixed (rfm15c) also stops randomising the ARM's motor gains; CalmFilter
# (rfm15d) adds a low-pass filter to the arm's actuation.
for _variant, _mode in [(v, m) for v in ("Calm", "CalmFixed", "CalmFilter") for m in ("Sphere", "Paper")]:
    gym.register(
        id=f"RobotLab-Isaac-RFM-{_variant}-{_mode}-Unitree-B2W-Z1-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.whole_body_env_cfg:B2WZ1RFM{_variant}{_mode}EnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.whole_body_env_cfg:B2WZ1RFMCalmPPORunnerCfg",
        },
    )

# rfm16: CalmFilter from scratch, with its own entropy setting.
for _mode in ("Sphere", "Paper"):
    gym.register(
        id=f"RobotLab-Isaac-RFM-Scratch-{_mode}-Unitree-B2W-Z1-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.whole_body_env_cfg:B2WZ1RFMCalmFilter{_mode}EnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.whole_body_env_cfg:B2WZ1RFMScratchPPORunnerCfg",
        },
    )

for _mode in ("Sphere", "Paper"):
    gym.register(
        id=f"RobotLab-Isaac-RFM-ScratchB-{_mode}-Unitree-B2W-Z1-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.whole_body_env_cfg:B2WZ1RFMScratchB{_mode}EnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.whole_body_env_cfg:B2WZ1RFMScratchPPORunnerCfg",
        },
    )

# rfm17: CalmFilter + integral of the hand's pose error in the observations.
for _mode in ("Sphere", "Paper"):
    gym.register(
        id=f"RobotLab-Isaac-RFM-Integral-{_mode}-Unitree-B2W-Z1-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.whole_body_env_cfg:B2WZ1RFMIntegral{_mode}EnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.whole_body_env_cfg:B2WZ1RFMCalmPPORunnerCfg",
        },
    )

# rfm18 ablations (sphere only): A = fixed arm gains + arm filter, B = A + arm rate.
for _abl in ("AblA", "AblB", "AblF", "AblG"):
    gym.register(
        id=f"RobotLab-Isaac-RFM-{_abl}-Sphere-Unitree-B2W-Z1-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.whole_body_env_cfg:B2WZ1RFM{_abl}SphereEnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.whole_body_env_cfg:B2WZ1RFMCalmPPORunnerCfg",
        },
    )

# rfm19: incremental (delta) arm commands.
for _mode in ("Sphere", "Paper"):
    gym.register(
        id=f"RobotLab-Isaac-RFM-Delta-{_mode}-Unitree-B2W-Z1-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.whole_body_env_cfg:B2WZ1RFMDelta{_mode}EnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.whole_body_env_cfg:B2WZ1RFMCalmPPORunnerCfg",
        },
    )

# three coloured tables, a red, a yellow and a green box on each (fetch_box.py --target)
gym.register(
    id="RobotLab-Isaac-Fetch3-Warehouse-Unitree-B2W-Z1-Gripper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fetch_env_cfg:UnitreeB2WZ1GripperFetch3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2WFlatPPORunnerCfg",
    },
)
