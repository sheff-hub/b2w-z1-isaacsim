# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
#
# Local addition (not upstream robot_lab). Phase 4, step 3, option A: reach while
# driving by running the two EXISTING policies side by side -- no new training.
#
#   driving  (model_2600.pt): 57 inputs -> 12 legs + 4 wheels, from the flat task
#   reaching (reach v2)     : 25 inputs -> 6 arm joints, from the reach task
#
# One environment, two observation groups and three action terms. Each policy
# must receive EXACTLY the inputs it was trained on: the "policy" group is the
# cargo task's (itself identical to the flat task's), and the "arm" group is
# copied term for term from the reach task. The only change to either is the
# "last action" input -- see last_actions_of().
#
# Not a training task. Driven by project/loco_reach.py, which runs both policies.

import torch

from isaaclab.utils import configclass

from robot_lab.tasks.manager_based.manipulation.reach_b2w_z1.reach_env_cfg import B2WZ1ReachEnvCfg

from .arm_cargo_env_cfg import UnitreeB2WZ1FlatEnvCfg

DRIVE_ACTIONS = ("joint_pos", "joint_vel")   # the flat task's action terms, in order
ARM_ACTION = "arm_action"                    # the reach task's action term


def last_actions_of(env, names):
    """Last raw actions of the NAMED action terms only, concatenated in order.

    The stock mdp.last_action returns the WHOLE action vector. With both
    policies' terms in one environment that is 16 + 6 = 22 numbers, so the
    driving policy would see its 16 plus the arm's 6 and the arm policy its 6
    plus the legs' 16 -- neither matches training. For the drive terms this
    reproduces the flat task's value exactly (it too was the raw concatenation
    of joint_pos then joint_vel).
    """
    return torch.cat([env.action_manager.get_term(n).raw_actions for n in names], dim=-1)


@configclass
class UnitreeB2WZ1LocoReachEnvCfg(UnitreeB2WZ1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        reach = B2WZ1ReachEnvCfg()

        # the arm is now DRIVEN (by the reach policy), with the reach task's scaling
        self.actions.arm_action = reach.actions.arm_action
        # same hand-target generator as training: body-frame box, new target every 4 s
        self.commands.ee_pose = reach.commands.ee_pose

        # arm inputs: the reach task's group, term for term, in the same order
        self.observations.arm = reach.observations.policy

        # "last action" restricted to each policy's own terms. Only the function
        # and its argument change; each term keeps its original clip/scale/noise
        # (the drive task's is clip (-100, 100), scale 1.0).
        self.observations.arm.actions.func = last_actions_of
        self.observations.arm.actions.params = {"names": (ARM_ACTION,)}
        for group in (self.observations.policy, self.observations.critic):
            group.actions.func = last_actions_of
            group.actions.params = {"names": DRIVE_ACTIONS}

        # the arm has an action term now, so it no longer needs holding at stow
        self.events.hold_arm_stow = None

