# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
#
# Local addition (not upstream robot_lab): the flat B2W task with the Z1 arm
# mounted and held still as cargo. Phase 4, first arm step: does the locomotion
# policy still drive with 4.4 kg mounted forward on the body?
#
# Only the ROBOT changes. Observations, actions and rewards are those of
# UnitreeB2WFlatEnvCfg, so a flat-task checkpoint loads unmodified. The arm is
# not part of the action space -- its own PD drives hold it at the stow pose.

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from robot_lab.assets.unitree import UNITREE_B2W_CFG

from .flat_env_cfg import UnitreeB2WFlatEnvCfg

# Built by tools/build_b2w_z1_rl_asset.py from the hand-placed
# b2w_z1.usd (arm mount, single articulation root, arm gains, patched wheels).
# Path is as seen INSIDE the isaac-lab container, where robots_usd is
# mounted at /workspace/robots_usd.
B2W_Z1_USD = os.environ.get("B2W_Z1_USD", "/workspace/robots_usd/b2w_z1_rl.usd")

# Z1 folded. joint2 and joint3 are 0..170 deg and -165..0 deg, so "0" sits ON a
# hard stop. Isaac Lab also clamps reset positions to soft limits at 90% of each
# range (soft_joint_pos_limit_factor), which for these two is +/-0.148 rad.
# +/-0.15 is the fold, just inside both.
ARM_STOW = {"joint1": 0.0, "joint2": 0.15, "joint3": -0.15, "joint4": 0.0, "joint5": 0.0, "joint6": 0.0}

UNITREE_B2W_Z1_CFG = UNITREE_B2W_CFG.replace(
    spawn=sim_utils.UsdFileCfg(
        usd_path=B2W_Z1_USD,
        activate_contact_sensors=True,
        rigid_props=UNITREE_B2W_CFG.spawn.rigid_props,
        articulation_props=UNITREE_B2W_CFG.spawn.articulation_props,
    ),
    init_state=UNITREE_B2W_CFG.init_state.replace(
        joint_pos={**UNITREE_B2W_CFG.init_state.joint_pos, **ARM_STOW},
    ),
    actuators={
        **UNITREE_B2W_CFG.actuators,
        # Holds the arm at ARM_STOW. Effort limits are left to the USD
        # (30/60/30/30/30/30 N*m, the Z1's own ratings).
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-6]"],
            stiffness=100.0,
            damping=4.0,
        ),
    },
)


def hold_default_joint_pos(env, env_ids, asset_cfg: SceneEntityCfg):
    """Make joints with NO action term hold their default pose.

    Isaac Lab initialises every joint's position target to ZERO
    (isaaclab/assets/articulation/articulation.py:1641) and only action terms ever
    change it. A joint the policy does not drive is therefore pulled toward 0,
    not toward init_state -- for the B2W legs that is thigh 0 / calf 0, straight
    legs, which launches the body ~0.3 m up and topples it. Run on reset so the
    held joints' targets are their defaults from the first step.
    """
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = slice(None)
    target = asset.data.default_joint_pos[env_ids][:, asset_cfg.joint_ids]
    asset.set_joint_position_target(target, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)


# The order the CRITIC read joints in when model_2600.pt was trained: the plain
# B2W's articulation order (measured with training/inspect_task.py on
# RobotLab-Isaac-Velocity-Flat-Unitree-B2W-v0). The upstream critic reads "all
# joints", which was exactly this on a 16-joint robot but becomes 22 joints once
# the arm is on -- a 72-input critic that no longer loads the checkpoint.
B2W_TRAINING_JOINT_ORDER = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
    "FL_foot_joint", "FR_foot_joint", "RL_foot_joint", "RR_foot_joint",
]


def joint_pos_rel_zero_wheels(env, asset_cfg: SceneEntityCfg, wheel_asset_cfg: SceneEntityCfg):
    """Joint positions relative to default, with the wheel entries zeroed.

    Replaces robot_lab's mdp.joint_pos_rel_without_wheel, which selects
    asset_cfg's joints into a smaller tensor and then zeroes COLUMNS using the
    wheels' indices in the FULL articulation. On the 16-joint B2W those indices
    (12-15) coincide with the wheel columns, so it works by accident; with the
    arm's 6 extra joints they run past the end (CUDA "index out of bounds"), and
    had they landed in range they would have silently zeroed leg readings.
    Here the wheels are located by COLUMN within the selection instead.
    """
    asset = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    if isinstance(ids, slice):
        ids = list(range(asset.num_joints))[ids]
    rel = asset.data.joint_pos[:, ids] - asset.data.default_joint_pos[:, ids]
    wheels = set(wheel_asset_cfg.joint_ids)
    rel[:, [col for col, joint in enumerate(ids) if joint in wheels]] = 0
    return rel


@configclass
class UnitreeB2WZ1FlatEnvCfg(UnitreeB2WFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = UNITREE_B2W_Z1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Keep BOTH networks' inputs exactly as model_2600.pt was trained on --
        # the same joints, in the same order -- so the arm's joints never leak in.
        policy_joints = SceneEntityCfg("robot", joint_names=self.joint_names, preserve_order=True)
        critic_joints = SceneEntityCfg("robot", joint_names=B2W_TRAINING_JOINT_ORDER, preserve_order=True)
        for group, joints in ((self.observations.policy, policy_joints), (self.observations.critic, critic_joints)):
            group.joint_pos.func = joint_pos_rel_zero_wheels
            group.joint_pos.params["asset_cfg"] = joints
            group.joint_vel.params["asset_cfg"] = joints.replace()

        # The arm has no action term here, so without this its motors hold it at
        # joint target 0 -- on joint2/joint3's hard stops -- not at ARM_STOW.
        self.events.hold_arm_stow = EventTerm(
            func=hold_default_joint_pos, mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(ARM_STOW))},
        )

        # flat_env_cfg only prunes zero-weight rewards when the class name is
        # exactly "UnitreeB2WFlatEnvCfg"; a subclass would silently skip it.
        self.disable_zero_weight_rewards()
