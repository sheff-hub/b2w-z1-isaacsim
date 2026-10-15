# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
#
# Local addition (not upstream robot_lab). Phase 4, step 3, OPTION C: ONE policy
# for all 22 joints -- drive and reach at the same time, from scratch.
#
# Option A (project/loco_reach.py) already does both by running the two separate
# policies side by side, and does it well (hand 1.0-1.7 cm while driving). What
# it cannot do is let the body HELP the arm: the legs never lean or crouch toward
# a target, because the driving policy does not know a target exists. That
# coupling is the only reason to train this, and it is what the ETH whole-body
# MPC paper demonstrates (crouching for low targets, rising for high ones).
#
# Inputs (82; critic 85 with base_lin_vel): body angular velocity, gravity
# direction, drive command, all 22 joint positions and velocities, last 22
# actions, hand target pose.

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

import isaaclab_tasks.manager_based.manipulation.reach.mdp as reach_mdp

from robot_lab.tasks.manager_based.manipulation.reach_b2w_z1.reach_env_cfg import ARM, HAND, B2WZ1ReachEnvCfg

from .arm_cargo_env_cfg import UnitreeB2WZ1FlatEnvCfg, joint_pos_rel_zero_wheels

ALL_JOINTS = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
    "FL_foot_joint", "FR_foot_joint", "RL_foot_joint", "RR_foot_joint",
] + ARM

# Reach reward weights, scaled up from the standalone reach task's -0.2 / +0.1.
# There, reaching was the only job. Here it competes with velocity tracking at
# 3.0 (linear) and 1.5 (angular), so at the template's weights the arm would be
# a rounding error and the policy would simply ignore it. These put a perfect
# reach on a par with perfect velocity tracking.
REACH_L2_WEIGHT = -2.0      # coarse: pulls the hand in from far away
REACH_FINE_WEIGHT = 2.0     # fine: pays inside ~10 cm (tanh, std 0.1)


@configclass
class B2WZ1WholeBodyEnvCfg(UnitreeB2WZ1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        reach = B2WZ1ReachEnvCfg()

        # --- actions: legs + wheels (inherited) + arm = 22
        self.actions.arm_action = reach.actions.arm_action

        # --- commands: drive command (inherited) + hand target every 4 s
        self.commands.ee_pose = reach.commands.ee_pose

        # --- observations: everything, for one policy. joint_pos/joint_vel widen
        # from the 16 locomotion joints to all 22, and the hand target is added.
        every_joint = SceneEntityCfg("robot", joint_names=ALL_JOINTS, preserve_order=True)
        for group in (self.observations.policy, self.observations.critic):
            group.joint_pos.func = joint_pos_rel_zero_wheels
            group.joint_pos.params["asset_cfg"] = every_joint.replace()
            group.joint_vel.params["asset_cfg"] = every_joint.replace()
            group.pose_command = ObsTerm(func=reach_mdp.generated_commands, params={"command_name": "ee_pose"})

        # --- rewards: driving (inherited) + reaching
        hand = SceneEntityCfg("robot", body_names=[HAND])
        self.rewards.ee_position_tracking = RewTerm(
            func=reach_mdp.position_command_error,
            weight=REACH_L2_WEIGHT,
            params={"asset_cfg": hand.replace(), "command_name": "ee_pose"},
        )
        self.rewards.ee_position_tracking_fine = RewTerm(
            func=reach_mdp.position_command_error_tanh,
            weight=REACH_FINE_WEIGHT,
            params={"asset_cfg": hand.replace(), "std": 0.1, "command_name": "ee_pose"},
        )
        self.rewards.arm_joint_limits = RewTerm(
            func=reach_mdp.joint_pos_limits,
            weight=-1.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM, preserve_order=True)},
        )

        # These two forbid the very behaviour this task exists to learn. They
        # penalise the legs for leaving the default stance while the drive
        # command is ~zero -- i.e. exactly when the robot should be crouching or
        # leaning to reach. stand_still goes; joint_pos_penalty is kept at a
        # quarter weight so the legs still have a reason not to wander.
        self.rewards.stand_still = None
        self.rewards.joint_pos_penalty.weight = -0.25

        # --- resets: upright, random heading. The inherited B2W reset starts the
        # robot at ANY orientation to teach fall recovery; that is a second hard
        # skill and not what this run is for.
        self.events.randomize_reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.05),
                           "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")},
        }

        # the arm is policy-driven now, so it is not held at stow
        self.events.hold_arm_stow = None

        # No disable_zero_weight_rewards() here: the parent already ran it, and a
        # second call crashes on the terms that pass has set to None. The reward
        # manager skips zero-weight terms anyway (reward_manager.py:146).


@configclass
class B2WZ1WholeBodyPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Locomotion-sized network and PPO settings (the B2W flat task's), since this
    is a locomotion problem with an arm attached, not a table-top reach."""

    num_steps_per_env = 24
    max_iterations = 20000        # stopped by hand; checkpoints every 100
    save_interval = 100
    experiment_name = "b2w_z1_wholebody"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


# ---------------------------------------------------------------------------
# Variant wb2: is the 0.76 m/s forward ceiling a REWARD ceiling?
#
# Every policy measured so far -- the inherited model_2600, the two-policy
# option A, and wb1 -- reaches ~76% of a 1.0 m/s forward command, while wb1
# turns at 85-89% of its command. Nothing physical explains it: at 1.0 m/s the
# wheels turn 8.9 rad/s against a 50 rad/s limit, and four wheels at their 20
# N*m limit could push ~700 N on an 87 kg robot.
#
# The suspicion is the reward: track_lin_vel_xy_exp is exp(-err^2/0.25), so
# 0.76 m/s against a 1.0 m/s command still collects 80% of the tracking reward,
# while closing that last gap costs torque, joint power and action-rate
# penalties. Doubling the tracking weight changes that trade and nothing else --
# one variable, so the answer is unambiguous either way.
@configclass
class B2WZ1WholeBodyTrackEnvCfg(B2WZ1WholeBodyEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.track_lin_vel_xy_exp.weight = 6.0   # was 3.0


@configclass
class B2WZ1WholeBodyTrackPPORunnerCfg(B2WZ1WholeBodyPPORunnerCfg):
    experiment_name = "b2w_z1_wholebody_track"


# ---------------------------------------------------------------------------
# Variant wb3: is the 0.76 m/s ceiling an ACTION-SCALE ceiling?
#
# wb2 refuted the reward explanation: doubling the forward-tracking weight left
# forward speed at 0.75 m/s and merely degraded turning, strafing and the hand.
#
# The wheel action term scales the policy's output by 5.0 rad/s (rough_env_cfg
# sets actions.joint_vel.scale = 5.0). Commanding 1.0 m/s needs 8.86 rad/s at a
# 0.1129 m wheel, so the policy has to emit 1.77 -- deep in the tail of a
# roughly unit-scale Gaussian output, and taxed by the action-rate penalty all
# the way. 0.76 m/s corresponds to an output of ~1.35, which is a far more
# comfortable place for the policy to sit.
#
# Doubling the scale to 10.0 makes 1.0 m/s reachable at an output of 0.89. If
# the ceiling moves, the cause was the action mapping; if it does not, the cause
# is physical or elsewhere in the task. Single variable against wb1.
@configclass
class B2WZ1WholeBodyWheelScaleEnvCfg(B2WZ1WholeBodyEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_vel.scale = 10.0   # was 5.0


@configclass
class B2WZ1WholeBodyWheelScalePPORunnerCfg(B2WZ1WholeBodyPPORunnerCfg):
    experiment_name = "b2w_z1_wholebody_wheelscale"


# ---------------------------------------------------------------------------
# Variant wb4: is the 0.76 m/s ceiling a COMMAND-DISTRIBUTION artefact?
#
# wb2 (reward weight) and wb3 (action scale) both left forward speed at
# 0.73-0.76 m/s, while Rung 0b showed the wheels themselves reach 0.97 m/s under
# a direct velocity command. So the robot can do it and the policy does not.
#
# Training samples lin_vel_x and lin_vel_y independently in (-1, 1), so a
# command that is nearly pure forward AND near full speed (|vx| > 0.9,
# |vy| < 0.1) is about 1% of samples: the policy almost never practises the one
# thing the evaluation asks for, and it is rewarded for a policy that hedges
# across the whole square. Narrowing the sideways range to (-0.3, 0.3) makes
# forward-dominant commands the common case. Nothing else changes.
#
# This also matters beyond the ceiling: a waypoint follower will mostly command
# forward motion, so the command distribution should match the intended use.
@configclass
class B2WZ1WholeBodyForwardEnvCfg(B2WZ1WholeBodyEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)   # was (-1.0, 1.0)


@configclass
class B2WZ1WholeBodyForwardPPORunnerCfg(B2WZ1WholeBodyPPORunnerCfg):
    experiment_name = "b2w_z1_wholebody_forward"


# ---------------------------------------------------------------------------
# Variant wb5: is the ceiling the EFFORT/SMOOTHNESS penalties?
#
# Three explanations are now refuted -- reward weight (wb2), action scale (wb3)
# and command distribution (wb4) -- and forward speed sat at 0.72-0.76 m/s in
# all of them, against 0.97 m/s from direct wheel commands in Rung 0b.
#
# What is left pulling the other way is the family of penalties that all grow
# with speed: action rate, joint torques, joint power, joint accelerations.
# Going from 0.76 to 0.95 m/s buys about 0.2 of tracking reward (the exp kernel
# is already near its top) while every one of these costs more. Halving them
# all tests the family at once. It is several numbers, but one hypothesis.
@configclass
class B2WZ1WholeBodySoftPenaltyEnvCfg(B2WZ1WholeBodyEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        for term, weight in (
            (self.rewards.action_rate_l2, -0.005),       # was -0.01
            (self.rewards.joint_torques_l2, -5.0e-6),    # was -1e-5
            (self.rewards.joint_power, -5.0e-6),         # was -1e-5
            (self.rewards.joint_acc_l2, -5.0e-8),        # was -1e-7
        ):
            if term is not None:
                term.weight = weight


@configclass
class B2WZ1WholeBodySoftPenaltyPPORunnerCfg(B2WZ1WholeBodyPPORunnerCfg):
    experiment_name = "b2w_z1_wholebody_softpenalty"


# ---------------------------------------------------------------------------
# Variant wb6: is the ceiling the WHEEL DRIVE itself?
#
# Four policy-side explanations are refuted (wb2-wb5): forward speed sits at
# 0.72-0.76 m/s regardless of reward weight, action scale, command distribution
# or penalty strength.
#
# That points away from the policy and at the actuator. robot_lab gives the
# wheels a velocity drive with damping 1.0 N*m*s/rad, so the torque is
# 1.0 * (target - actual): holding any load needs a standing velocity ERROR.
# A few N*m of rolling resistance therefore costs several rad/s -- 0.1 m/s at a
# 0.113 m wheel per N*m. Rung 0b, which reached 0.97 m/s of a 1.0 m/s command,
# used damping 20 on the same wheels.
#
# wb6 raises the wheel damping to 20 and changes nothing else. If forward speed
# finally moves, the ceiling was never about learning at all.
@configclass
class B2WZ1WholeBodyStiffWheelEnvCfg(B2WZ1WholeBodyEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.actuators["wheel"] = self.scene.robot.actuators["wheel"].replace(damping=20.0)


@configclass
class B2WZ1WholeBodyStiffWheelPPORunnerCfg(B2WZ1WholeBodyPPORunnerCfg):
    experiment_name = "b2w_z1_wholebody_stiffwheel"


# ---------------------------------------------------------------------------
# Variant wb7: the two changes that each helped, together.
#
# wb5 (halved effort/smoothness penalties) gave the best turning and strafing;
# wb6 (wheel damping 20) gave the only improvement in forward speed, 0.76 ->
# 0.81, but turned left poorly. Neither is a ceiling explanation on its own.
# This is not a hypothesis test -- it is the best-config candidate.
@configclass
class B2WZ1WholeBodyBestEnvCfg(B2WZ1WholeBodySoftPenaltyEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.actuators["wheel"] = self.scene.robot.actuators["wheel"].replace(damping=20.0)


@configclass
class B2WZ1WholeBodyBestPPORunnerCfg(B2WZ1WholeBodyPPORunnerCfg):
    experiment_name = "b2w_z1_wholebody_best"


# ---------------------------------------------------------------------------
# Variant wb8: is the ceiling the reward's TOLERANCE (not its weight)?
#
# wb6/wb7 fixed the wheel drive and forward speed rose 0.76 -> 0.80, but the
# policy still settles well below the 0.97 m/s the wheels can deliver. Weight
# (wb2), action scale (wb3), command distribution (wb4) and penalties (wb5) are
# all ruled out. What is left is the SHAPE of the tracking reward:
#
#     exp(-err^2 / std^2),  std^2 = 0.25
#
# At a 0.2 m/s error that still pays 0.92, and the gradient for closing the last
# 0.2 m/s is nearly flat -- the reward says "close enough". Halving std to 0.5
# (std^2 = 0.25 -> 0.0625) makes the same error pay 0.45 instead. Same weight,
# same everything else; only the tolerance changes.
@configclass
class B2WZ1WholeBodySharpTrackEnvCfg(B2WZ1WholeBodyBestEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.25   # was sqrt(0.25) = 0.5
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.25


@configclass
class B2WZ1WholeBodySharpTrackPPORunnerCfg(B2WZ1WholeBodyPPORunnerCfg):
    experiment_name = "b2w_z1_wholebody_sharptrack"


# ---------------------------------------------------------------------------
# Variant wb9: WORLD-frame hand targets -- the one that can actually show
# whether the body helps the arm.
#
# Everything up to wb8 used a body-frame target, which rides along with the
# robot: crouching moves the target down too, so body motion cannot buy reach.
# wb9 fixes the target in the world (see world_pose_command.py) and samples
# heights from 0.15 to 0.95 m. With the shoulder near 0.80 m standing and 0.69 m
# of arm, the low end is only reachable by lowering the body -- so if the policy
# learns to crouch, it is because crouching pays.
#
# The drive command is narrowed to +/-0.3 m/s and +/-0.5 rad/s on purpose: a
# world-fixed target is left behind within a second at 0.8 m/s, and this task is
# about reaching with the body's help, not about driving fast. Driving speed is
# already settled by wb7/wb8.
from .world_pose_command import WorldPoseCommandCfg


@configclass
class B2WZ1WholeBodyWorldTargetEnvCfg(B2WZ1WholeBodyBestEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        old = self.commands.ee_pose
        self.commands.ee_pose = WorldPoseCommandCfg(
            asset_name="robot",
            body_name=HAND,
            resampling_time_range=(5.0, 5.0),
            debug_vis=True,
            ranges=WorldPoseCommandCfg.Ranges(
                pos_x=(0.25, 0.60),     # in front, in the heading frame at sampling time
                pos_y=(-0.30, 0.30),
                pos_z=(0.15, 0.95),     # ABSOLUTE height: low ones need a crouch
                roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0),
            ),
        )
        del old

        self.commands.base_velocity.ranges.lin_vel_x = (-0.3, 0.3)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)


@configclass
class B2WZ1WholeBodyWorldTargetPPORunnerCfg(B2WZ1WholeBodyPPORunnerCfg):
    experiment_name = "b2w_z1_wholebody_worldtarget"


# ---------------------------------------------------------------------------
# Variant wb10: world-frame targets AND full driving speed -- one policy for the
# whole Phase 4 job.
#
# wb9 proved the body helps the arm (29 cm of crouch, 0.5-2.1 cm hand error) but
# it is a poor driver: trained to +/-0.3 m/s, it manages 0.38 of a 1.0 m/s
# command and fell once in the standard script. wb8 drives well but its target
# is body-frame, so "put the hand on that point in the room" cannot even be
# expressed.
#
# The obstacle to combining them is that a world-fixed target is left behind
# within a second at 0.8 m/s, so reaching becomes impossible and the policy
# learns to ignore it. max_distance re-draws a target the robot has driven past,
# which keeps reaching live at any speed.
@configclass
class B2WZ1WholeBodyWorldDriveEnvCfg(B2WZ1WholeBodyWorldTargetEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.ee_pose.max_distance = 1.2
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


@configclass
class B2WZ1WholeBodyWorldDrivePPORunnerCfg(B2WZ1WholeBodyPPORunnerCfg):
    experiment_name = "b2w_z1_wholebody_worlddrive"


# ---------------------------------------------------------------------------
# Variant wb11: the same merge at HALF speed.
#
# wb10 (world targets at +/-1.0 m/s) learned to drive and ignored the arm: hand
# error stuck at 0.38 m where wb9 reached 0.07 m by the same iteration. At full
# speed the robot passes every world target within about a second, a replacement
# appears ahead, and it is passed too -- reaching almost never pays, so the
# policy drops it. max_distance keeps targets NEARBY but cannot make them
# CATCHABLE.
#
# wb11 changes one thing from wb10: the drive command range, +/-1.0 -> +/-0.6.
# That is the midpoint between wb9's +/-0.3 (reaching works, driving is weak) and
# wb10's +/-1.0 (driving works, reaching dies), and it tests whether a usable
# speed and world-frame reaching can coexist in one policy at all.
@configclass
class B2WZ1WholeBodyWorldDriveHalfEnvCfg(B2WZ1WholeBodyWorldDriveEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (-0.6, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.6, 0.6)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)


@configclass
class B2WZ1WholeBodyWorldDriveHalfPPORunnerCfg(B2WZ1WholeBodyPPORunnerCfg):
    experiment_name = "b2w_z1_wholebody_worlddrive_half"


# ---------------------------------------------------------------------------
# Variant wb12: tidy legs, a body that cannot pass through itself, and a target
# range taken from the MEASURED workspace instead of chosen by hand.
#
# wb9 answered the scientific question (the body helps the arm: 29 cm of crouch)
# but it is not presentable. Three faults, all of them mine:
#
#   1. THE LEGS WANDER. To let the robot crouch I removed stand_still entirely
#      and cut joint_pos_penalty to a quarter, and robot_lab already disables the
#      gait rewards for wheeled robots. Almost nothing was left constraining leg
#      posture, so the legs drift into whatever pose happens to pay.
#
#      The fix cannot be "put stand_still back": stand_still and joint_pos_penalty
#      both measure deviation from the DEFAULT STANCE, and crouching is a large
#      deviation from the default stance -- turning them up punishes precisely the
#      behaviour this task exists to produce. Posture is restored instead with
#      terms that do not reference the default pose:
#        * hip deviation (-0.5): the hips stay under the body. Crouching is done
#          with thigh and calf, so this costs the crouch nothing.
#        * diagonal symmetry (-0.05 -> -0.3): both diagonals do the same thing,
#          which is what "tidy" actually looks like on a quadruped.
#        * a level body (-0.5, recreated -- it was weight 0 and therefore None):
#          deliberately WEAK, because leaning is how the robot reaches its lowest
#          targets, and a strong term here would forbid that.
#        * action rate back to -0.01 from wb5's -0.005, against the jitter.
#
#   2. THE ARM PASSES THROUGH THE ROBOT. Measured from the collision geometry
#      (evaluation/workspace.py), the lidar mast is a capsule spanning z 0.00-0.34 at
#      x 0.27-0.42 in the base frame -- directly in front of the shoulder, so the
#      arm meets it constantly.
#
#      The obvious fix, enabled_self_collisions=True, was tried and REVERTED. It
#      does not work on this asset: PhysX filters only collider pairs joined by a
#      joint, and two pairs here overlap by construction while being two joints
#      apart, so they jam from the first step at 20-25 kN (probe_pairs.py):
#        head_Link <-> lidar_link  24.6 kN   both bolted to the base
#        link00    <-> base_link   19.2 kN   the arm's base is sunk into the body
#      The arm stopped moving at all: hand error stayed at 0.55 m where wb9 was at
#      0.04 m by the same iteration. Reassuringly, the pairs we actually want --
#      the arm's moving links against the mast -- read 0.0 N at rest, so the
#      geometry is right and only those two mounts are wrong. Cleaning them up
#      means editing collision geometry in the asset: worth doing, not tonight.
#
#      Instead the behaviour is trained directly: a penalty on the arm entering
#      the measured volume of the body and mast (self_clearance.py), and the
#      target sampler refuses to place a target inside that volume either.
#
#   3. THE TARGET BOX WAS A GUESS, AND A THIRD OF IT WAS UNREACHABLE. Sampling
#      25,000 arm poses inside the real joint limits and sweeping the body over
#      the 0.325-0.616 m height range it actually uses gives the reachable volume.
#      wb9's hand-chosen box is only 69.6% reachable -- the policy spent a third of
#      its training chasing points it could not touch at any body height.
#
#      The new box keeps the low end (0.15 m, which needs a lean as well as a
#      crouch: joint2/joint3 cannot swing back past the shoulder, so a level body
#      bottoms out at 0.24 m) and raises the ceiling 0.95 -> 1.25 m, which is the
#      mirror image of the same skill -- the robot must stand tall to reach high.
#      Forward reach goes 0.60 -> 0.65 m, just inside the 0.694 m arm.
#
# Three changes at once, which breaks the one-variable rule the rest of this
# ladder follows. They are done together because they are one request and each
# is separately verifiable after the fact: crouch_test.py for the crouch,
# the arm-contact reward term for the collisions, and the hand error for reach.
from isaaclab.managers import SceneEntityCfg as _SEC
from isaaclab.sensors import ContactSensorCfg

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as velocity_mdp

from .self_clearance import KEEPOUT_BOXES, arm_body_clearance

# Measured by evaluation/workspace.py; see logs/eval/workspace.json.
WS_POS_X = (0.25, 0.65)
WS_POS_Y = (-0.30, 0.30)
WS_POS_Z = (0.15, 1.25)


@configclass
class B2WZ1WholeBodyTidyEnvCfg(B2WZ1WholeBodyWorldTargetEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # --- 1. the measured target box
        self.commands.ee_pose.ranges = WorldPoseCommandCfg.Ranges(
            pos_x=WS_POS_X, pos_y=WS_POS_Y, pos_z=WS_POS_Z,
            roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0),
        )
        # ... and never inside the robot's own body or lidar mast.
        self.commands.ee_pose.keepout = KEEPOUT_BOXES

        # --- 2. the arm must stop passing through the robot
        #
        # enabled_self_collisions=True was tried FIRST and had to be abandoned:
        # PhysX only filters collider pairs joined by a joint, and this asset has
        # two pairs that overlap by construction two joints apart, so they jam at
        # 20-25 kN from the first step (probe_pairs.py):
        #     head_Link <-> lidar_link, and link00 <-> base_link.
        # The arm stopped moving entirely -- hand error 0.55 m where wb9 was at
        # 0.04 m by the same iteration. The pairs we care about (the arm's moving
        # links vs the lidar mast) correctly read 0 N at rest, so the fix is
        # collider surgery in the asset, which is follow-up work.
        #
        # Meanwhile the BEHAVIOUR is trained directly: a penalty on the arm
        # entering the measured volume of the body and the lidar mast. It touches
        # no physics, so it cannot destabilise anything.
        self.rewards.arm_body_clearance = RewTerm(
            func=arm_body_clearance,
            weight=-15.0,
            params={
                # link00/link01 live inside the mount by design; these are the
                # links that actually sweep through the mast.
                "asset_cfg": _SEC("robot", body_names=["link0[3-6]"]),
                "margin": 0.04,
            },
        )
        # Contact sensing over the arm, which the stock sensor never covered
        # (its prim_path "{ENV_REGEX_NS}/Robot/.*" matches one level, and the arm
        # is at Robot/z1/*). With self-collisions off this catches the arm
        # dragging on the GROUND, which is worth discouraging on its own.
        self.scene.arm_contacts = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/z1/.*", history_length=3, track_air_time=False,
        )
        self.rewards.arm_self_contacts = RewTerm(
            func=velocity_mdp.undesired_contacts,
            weight=-0.5,
            params={"sensor_cfg": _SEC("arm_contacts", body_names=".*"), "threshold": 1.0},
        )

        # --- 3. posture, without forbidding the crouch
        self.rewards.create_joint_deviation_l1_rewterm(
            "joint_deviation_hip_l1", -0.5, [".*_hip_joint"]
        )
        self.rewards.joint_mirror.weight = -0.3
        self.rewards.flat_orientation_l2 = RewTerm(
            func=velocity_mdp.flat_orientation_l2,
            weight=-0.5,
            params={"asset_cfg": _SEC("robot")},
        )
        self.rewards.action_rate_l2.weight = -0.01


@configclass
class B2WZ1WholeBodyTidyPPORunnerCfg(B2WZ1WholeBodyPPORunnerCfg):
    experiment_name = "b2w_z1_wholebody_tidy"


# ---------------------------------------------------------------------------
# Variant wb13: wb12 with the action-rate penalty back where wb9 had it.
#
# Contingency, held ready while wb12 trains. At iteration 500 wb12 was well
# behind wb9 at the same point -- hand error 0.33 m against 0.099 m, and the fine
# reach reward 0.086 against 0.573 -- and its mean hand distance was ~2.7x wb9's
# while its target box is only ~1.3x larger. So it is not merely a harder task.
#
# The clearance penalty is NOT the cause: it scores -0.026 against reach terms of
# -0.68, far too small to matter. The suspect is action_rate_l2, which this run
# doubled (-0.005 -> -0.01) as part of the posture package. That term taxes fast
# joint motion, and the arm has to move fast to chase a target that is resampled
# every 5 seconds -- wb9 paid -0.397 for it at iteration 500, wb12 pays -0.802.
#
# One variable against wb12. If reaching recovers, the tidiness package costs
# reach through this term specifically and the posture work should lean on the
# other four terms instead.
@configclass
class B2WZ1WholeBodyTidyFreeEnvCfg(B2WZ1WholeBodyTidyEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.action_rate_l2.weight = -0.005


@configclass
class B2WZ1WholeBodyTidyFreePPORunnerCfg(B2WZ1WholeBodyPPORunnerCfg):
    experiment_name = "b2w_z1_wholebody_tidyfree"


# ---------------------------------------------------------------------------
# Variant wb14: the same policy, told much more firmly to stay out of the mast.
#
# wb12 halved the problem but did not remove it. Measured on the reach map with
# the geometric intrusion metric (evaluation/reach_map.py):
#
#              cells where an arm link enters the robot's volume   mean depth
#   wb9                    36/84  (43%)                              4.4 cm
#   wb12 @6000             22/84  (26%)                              2.3 cm
#
# and the breakdown says ALL of what is left is the LIDAR MAST -- zero intrusion
# into the body box. That is exactly the part that was complained about, so it
# is worth one more run rather than a footnote.
#
# Two changes, both to the same term, aimed at the same thing:
#   weight -15 -> -50   reaching through the mast has to stop being worth it;
#   margin 0.04 -> 0.08 the penalty starts EARLIER. With a 4 cm margin the arm
#                       is already almost touching before anything is paid, so
#                       the gradient that pushes it aside acts far too late.
#
# This is engineering to a requirement, not a hypothesis test, so the two move
# together and the verdict is the intrusion metric, not attribution.
#
# Trained by RESUMING wb12 rather than from scratch: the reaching skill is
# already there and only the routing has to change, which takes ~1500 iterations
# instead of ~6000. Run it with:
#   --task ...-TidyHard-... agent.resume=true \
#     agent.load_run=2026-09-24_00-58-38_wb12 agent.load_checkpoint=model_6000.pt
@configclass
class B2WZ1WholeBodyTidyHardEnvCfg(B2WZ1WholeBodyTidyEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.arm_body_clearance.weight = -50.0
        self.rewards.arm_body_clearance.params["margin"] = 0.08


@configclass
class B2WZ1WholeBodyTidyHardPPORunnerCfg(B2WZ1WholeBodyPPORunnerCfg):
    # SAME experiment name as wb12 on purpose: rsl-rl resolves --load_run inside
    # the experiment directory, so resuming from wb12 requires sharing it.
    experiment_name = "b2w_z1_wholebody_tidy"


# ---------------------------------------------------------------------------
# Variant wb15: wb14's strength, without wb14's self-inflicted blind spot.
#
# wb14 (weight -50, margin 0.08) drove intrusion to ZERO -- 0/84 cells -- but lost
# the two nearest columns of the reach map: targets at x = 0.25 and 0.35 went from
# 1.5-9 cm error (wb12) to 10-25 cm. It simply refused to go there.
#
# That is a bug in how I set it up, not a trade-off. The keep-out shell the REWARD
# punishes was inflated to 8 cm, while the shell the TARGET SAMPLER avoids stayed
# at 4 cm. So targets were being placed 4-8 cm from the mast -- inside the
# penalised zone -- and the policy was asked to reach points it was simultaneously
# punished for approaching. Refusing them is the correct response to a
# contradictory reward.
#
# wb15 keeps the strength that worked (weight -50) and puts the margin back to
# 0.04, so the penalised shell and the sampler's exclusion shell are the SAME
# volume and no target can ever sit inside the penalty.
@configclass
class B2WZ1WholeBodyTidyFirmEnvCfg(B2WZ1WholeBodyTidyEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.arm_body_clearance.weight = -50.0
        self.rewards.arm_body_clearance.params["margin"] = 0.04   # == sampler's keepout_margin


@configclass
class B2WZ1WholeBodyTidyFirmPPORunnerCfg(B2WZ1WholeBodyPPORunnerCfg):
    experiment_name = "b2w_z1_wholebody_tidy"


# ---------------------------------------------------------------------------
# Variant wb16: the paper's method, in full.
#
# Jiang et al., RA-L 2024 (arXiv:2412.03012), "Learning Whole-Body
# Loco-Manipulation for Omni-Directional Task Space Pose Tracking with a
# Wheeled-Quadrupedal-Manipulator" -- same class of robot as ours (a wheeled
# quadruped with a 6-DoF arm), same task, and it fixes the two things our own
# policies get wrong. See rfm.py for the mechanism and why each piece is there.
#
# WHAT CHANGES FROM wb15
#
#   1. 6D TARGETS. Orientation is commanded and rewarded, not just position.
#      The command the policy sees becomes 12 numbers -- the target rotation as
#      a flattened matrix plus its position, both in the base frame, which is
#      the paper's representation (Table I) and better behaved for learning than
#      a quaternion.
#
#   2. OMNI-DIRECTIONAL, AND OFTEN FAR. Targets are drawn at a uniform bearing
#      0.3-3.0 m around the robot instead of in a box in front, so turning
#      around and driving are part of the task rather than something a script
#      does. This is the change our own failure demanded: at 2.5 m wb15 stretched
#      the arm and fell, because it had never seen a target it could not reach.
#
#   3. REWARDS FUSE INSTEAD OF SUMMING. Velocity tracking is gone -- there is no
#      base velocity command in this task at all, which is the paper's whole
#      point: ONE 6D end-effector command, and the policy works out whether that
#      means driving, reaching, or both. The locomotion and manipulation reward
#      groups are blended by the phase variable, and inside each group the terms
#      multiply so that stability gates tracking.
#
# The old additive terms are switched off, with two groups kept: r_basic from
# the paper (action rate, smoothness, collision, alive, power, torque) and the
# clearance/posture work from wb12-wb15, which the paper has no equivalent of
# but our lidar mast needs.
from .rfm import RewardFusion

ARM_SPAN = 0.694          # measured Z1 reach from the mount, evaluation/workspace.py


def ee_pose_command_12d(env, command_name: str):
    """The 6D target as the paper feeds it: Vec(R*) then P*, in the base frame."""
    import torch as _torch

    from isaaclab.utils.math import matrix_from_quat as _mat

    cmd = env.command_manager.get_command(command_name)
    rot = _mat(cmd[:, 3:7]).reshape(cmd.shape[0], 9)
    return _torch.cat([rot, cmd[:, :3]], dim=1)


@configclass
class B2WZ1RFMEnvCfg(B2WZ1WholeBodyTidyEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # --- 6D, omni-directional, often out of reach
        self.commands.ee_pose = WorldPoseCommandCfg(
            asset_name="robot",
            body_name=HAND,
            resampling_time_range=(10.0, 15.0),  # long enough to drive there AND reach:
            # a 3 m target at the 0.35 m/s reference speed needs ~9 s just to close
            debug_vis=True,
            radial_range=(0.30, 1.00),   # grown by the curriculum below
            approach_speed=0.35,
            phase_mu=2.0 * ARM_SPAN,
            phase_slope=2.0 * ARM_SPAN,
            keepout=KEEPOUT_BOXES,
            ranges=WorldPoseCommandCfg.Ranges(
                pos_x=(0.25, 0.65), pos_y=(-0.30, 0.30),   # unused: radial_range wins
                pos_z=(0.15, 1.25),
                roll=(-0.5, 0.5), pitch=(-0.5, 1.0), yaw=(-1.0, 1.0),
            ),
        )

        # Episodes must outlast a target, or the robot never sees the end of an
        # approach it started.
        self.episode_length_s = 20.0

        # --- the policy sees the 12-number command instead of position + quaternion
        for group in (self.observations.policy, self.observations.critic):
            group.pose_command = ObsTerm(
                func=ee_pose_command_12d, params={"command_name": "ee_pose"}
            )

        # --- no base velocity command: the EE target alone decides what to do
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        for name in ("track_lin_vel_xy_exp", "track_ang_vel_z_exp",
                     "ee_position_tracking", "ee_position_tracking_fine",
                     "stand_still", "joint_pos_penalty", "joint_mirror",
                     "feet_contact_without_cmd", "upward", "joint_deviation_hip_l1",
                     "flat_orientation_l2"):
            if getattr(self.rewards, name, None) is not None:
                setattr(self.rewards, name, None)

        # --- the fused reward (everything that is not r_basic)
        self.rewards.rfm = RewTerm(
            func=RewardFusion,
            weight=1.0,
            params={
                "asset_cfg": _SEC("robot"),
                "command_name": "ee_pose",
                "sigma_p": 0.25,
                "sigma_o": 0.5,
                "micro_m": 2.0,
                "w_static_arm": 0.15,
                "nominal_height": 0.55,
                "arm_joints": ARM,
                # phase from the arm's workspace: drive until the target is
                # inside half the arm's reach, then reach. Centred on the
                # SHOULDER (link02, the first pitch joint), which is where the
                # 0.694 m reach was measured from -- the mounting plate (link00)
                # sits 10 cm lower, and from there targets above ~1.10 m could
                # never count as inside at any body height.
                "mount_body": "link02",
                "arm_reach": ARM_SPAN,
                "reach_frac": 0.5,
                "phase_width": 0.25,
                "tilt_free": 0.26,
                "w_tilt": 20.0,
            },
        )

        # --- r_basic, at the paper's weights (Table II)
        self.rewards.action_rate_l2.weight = -0.003
        if getattr(self.rewards, "joint_torques_l2", None) is not None:
            self.rewards.joint_torques_l2.weight = -4.0e-5
        if getattr(self.rewards, "joint_power", None) is not None:
            self.rewards.joint_power.weight = -3.3e-4
        # Table II lists "Alive: 1 if alive else 0, weight 2.0" and the -5 belongs
        # to COLLISION, not termination -- I had that backwards on the first run
        # and the total reward drifted structurally negative, which is how an
        # agent learns that falling over early is the cheapest option.
        self.rewards.alive = RewTerm(func=velocity_mdp.is_alive, weight=2.0)

        # The lidar mast again. wb12's -15 let the arm graze it in 26% of poses;
        # wb15's -50 with a matched margin got that to 11%. Inheriting the weaker
        # one here was an oversight.
        self.rewards.arm_body_clearance.weight = -50.0
        self.rewards.arm_body_clearance.params["margin"] = 0.04

        # --- targets move further out only once the policy can hit the near ones
        from isaaclab.managers import CurriculumTermCfg as CurrTerm

        from .rfm import target_radius_curriculum

        self.curriculum.target_radius = CurrTerm(
            func=target_radius_curriculum,
            params={"command_name": "ee_pose", "start": 1.0, "end": 3.0,
                    "step": 0.2, "success_error": 0.20, "success_frac": 0.5},
        )

        # --- the paper's failure conditions (Section V-B): a run counts as failed
        # if the body tips past ~84 deg or the base drops below 0.2 m. Ours had no
        # height termination, so a collapsed robot kept collecting reward.
        self.terminations.base_too_low = DoneTerm(
            func=velocity_mdp.root_height_below_minimum,
            params={"minimum_height": 0.20, "asset_cfg": _SEC("robot")},
        )
        # ...and the other half of the paper's failure test, which I left out the
        # first time: "Z-component of the projected gravity is larger than -0.1".
        # Without it, rolling over is never terminated, so the policy never
        # learns that it is fatal -- measured as 4/12 flips at sideways targets,
        # several of them AFTER the hand had already arrived.
        self.terminations.base_tipped = DoneTerm(
            func=velocity_mdp.bad_orientation,
            # 1.0 rad (57 deg) let the policy park at 44-55 deg, just under the
            # limit. 0.6 rad (~34 deg) still covers the useful lean.
            params={"limit_angle": 0.6, "asset_cfg": _SEC("robot")},
        )


@configclass
class B2WZ1RFMPPORunnerCfg(B2WZ1WholeBodyPPORunnerCfg):
    experiment_name = "b2w_z1_rfm"


# ---------------------------------------------------------------------------
# The controlled comparison: OUR phase criterion vs THE PAPER'S.
#
# Two tasks identical in every respect -- rewards, speed profile, face-the-
# target, anti-shake, curriculum, terminations -- except how the phase variable
# D (locomotion vs manipulation) is computed:
#
#   RFM-Sphere  distance from the SHOULDER to the target position, measured from
#               the real state, blended from full arm reach (D ~ 0.95) down to
#               half reach (D ~ 0.05). The user's proposal.
#   RFM-Paper   the HAND's SE(3) distance on the retreating reference schedule,
#               midpoint = slope = 2 x arm span, exactly as Jiang et al. eqs.
#               (11)-(12) state it.
#
# Both resume from the same checkpoint and train for the same iterations, so a
# difference in behaviour is a difference in the phase criterion.
from .rfm import ActionSmoothness


@configclass
class B2WZ1RFMSphereEnvCfg(B2WZ1RFMEnvCfg):
    PHASE_MODE = "sphere"

    def __post_init__(self):
        super().__post_init__()
        self.rewards.rfm.params["phase_mode"] = self.PHASE_MODE
        self.rewards.rfm.params["w_arm_still"] = 0.05
        # the paper's second-difference smoothness term, Table II
        self.rewards.action_smoothness = RewTerm(func=ActionSmoothness, weight=-0.001)
        # speed profile: 1.0 m/s cruise, easing to 0.3 m/s at the workspace sphere
        self.commands.ee_pose.approach_speed_max = 1.0
        self.commands.ee_pose.approach_speed_min = 0.3
        self.commands.ee_pose.approach_gain = 1.0
        self.commands.ee_pose.approach_stop = 0.35


@configclass
class B2WZ1RFMPaperEnvCfg(B2WZ1RFMSphereEnvCfg):
    PHASE_MODE = "paper"


# ---------------------------------------------------------------------------
# rfm15: A CALM ARM. Same tasks as rfm14 (sphere / paper), resumed from its
# 33,800-iteration checkpoints, with the arm's bang-bang hold trained out.
#
# Measured on rfm14 (evaluation/shake_diag.py): on target the hand wobbles ~15 mm
# RMS at ~1 Hz while the arm's actions jump ~3.0 per 20 ms step. The fixes,
# all aimed at the ARM so locomotion keeps its freedom:
#
#   * arm action rate + second difference, -0.1   (was effectively -0.003)
#   * the paper's relative torque rate on the arm, -0.1 (Table II; was missing)
#   * hand speed scales the tracking bonus (a gated penalty failed, rfm.py)
#   * arm joint-velocity observation noise 1.5 -> 0.15 rad/s. +-1.5 is the
#     locomotion template's value; on the arm it buried every velocity below
#     ~1.5 rad/s, so the policy could not see -- let alone damp -- its own
#     wobble. Same observation size, so the checkpoint still loads.
#   * entropy 0.01 -> 0.002 and the arm's action std reset 2.0 -> 0.3 in the
#     starting checkpoint (training/calm_init.py): the paper DECAYS its entropy
#     coefficient; ours was constant for 33k iterations and the arm noise grew.
from isaaclab.utils.noise import NoiseCfg as _NoiseCfg

from .rfm import ArmActionRate, ArmTorqueRate


def per_joint_uniform_noise(data, cfg):
    """Additive uniform noise with its own half-width per column."""
    import torch as _torch

    n = _torch.tensor(cfg.half_width, device=data.device, dtype=data.dtype)
    return data + (2.0 * _torch.rand_like(data) - 1.0) * n


@configclass
class PerJointUniformNoiseCfg(_NoiseCfg):
    """Isaac Lab's UniformNoiseCfg takes a tensor for this, but Hydra cannot
    serialise a tensor inside a task config -- a plain tuple it can."""

    func = per_joint_uniform_noise
    half_width: tuple = ()


def _calm(cfg):

    cfg.rewards.arm_action_rate = RewTerm(
        func=ArmActionRate, weight=-0.1, params={"term_name": "arm_action", "w2": 0.5})
    cfg.rewards.arm_torque_rate = RewTerm(
        func=ArmTorqueRate, weight=-0.1, params={"asset_cfg": _SEC("robot"), "joints": ARM})
    # Hand stillness scales the tracking bonus inside RFM. (A separate penalty
    # gated on distance, hand_still_on_target, taught the robot to stay 25 cm
    # away from the target -- see rfm.py.) sigma 0.03: a hand moving 15 cm/s
    # keeps ~half the bonus, 30 cm/s keeps ~5%.
    cfg.rewards.rfm.params["sigma_hand_v"] = 0.03
    cfg.rewards.rfm.params["w_hand_ang"] = 0.05
    # per-joint noise, in ALL_JOINTS order: 16 locomotion joints keep 1.5, arm 0.15
    cfg.observations.policy.joint_vel.noise = PerJointUniformNoiseCfg(
        half_width=tuple([1.5] * 16 + [0.15] * 6))


@configclass
class B2WZ1RFMCalmSphereEnvCfg(B2WZ1RFMSphereEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _calm(self)


@configclass
class B2WZ1RFMCalmPaperEnvCfg(B2WZ1RFMPaperEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _calm(self)


# rfm15c: the same, with the ARM's motor gains no longer randomised.
#
# Why the policy drives the arm bang-bang in the first place: every reset scales
# every joint's stiffness AND damping by a random 0.5-2.0 (velocity_env_cfg's
# randomize_actuator_gains, joint_names ".*" -- the arm included). The arm's
# nominal kp is 100, so it is anywhere from 50 to 200, and a memoryless policy
# cannot tell which. A smooth position target then sags under gravity by an
# unknown amount -- several cm at the hand. Slamming the target past the torque
# limit in both directions gives an average torque that does NOT depend on kp:
# the bang-bang hold is the policy's answer to the randomisation. Penalising it
# alone fights that incentive head-on (rfm15b: tracking 0.15 -> 0.25 m, ori
# 0.34 -> 0.50 rad within 300 iterations).
# The legs and wheels keep their randomisation: locomotion's robustness needs it,
# and their torques are not what the user watches.
LOCO_JOINT_REGEX = ".*_(hip|thigh|calf|foot)_joint"


@configclass
class B2WZ1RFMCalmFixedSphereEnvCfg(B2WZ1RFMCalmSphereEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.events.randomize_actuator_gains.params["asset_cfg"] = _SEC("robot", joint_names=[LOCO_JOINT_REGEX])


@configclass
class B2WZ1RFMCalmFixedPaperEnvCfg(B2WZ1RFMCalmPaperEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.events.randomize_actuator_gains.params["asset_cfg"] = _SEC("robot", joint_names=[LOCO_JOINT_REGEX])


# rfm15d: CalmFixed + a low-pass filter in the arm's actuation (rfm.py,
# FilteredJointPositionAction) + a softer hand-stillness factor. rfm15b/c showed
# the same decline (orientation error 0.34 -> 0.6-0.9 rad): at sigma 0.03 a hand
# moving 30 cm/s keeps 5% of the tracking bonus, so while the arm was still
# flicking there was almost no reward left for tracking at all -- no signal to
# learn smooth tracking from. At 0.1: 15 cm/s keeps 80%, 30 cm/s 40%.
def _filtered(cfg, alpha: float = 0.25):
    from .rfm import FilteredJointPositionActionCfg

    old = cfg.actions.arm_action
    cfg.actions.arm_action = FilteredJointPositionActionCfg(
        asset_name=old.asset_name, joint_names=old.joint_names, scale=old.scale,
        use_default_offset=old.use_default_offset, preserve_order=old.preserve_order,
        clip=old.clip, alpha=alpha)
    cfg.rewards.rfm.params["sigma_hand_v"] = 0.1


@configclass
class B2WZ1RFMCalmFilterSphereEnvCfg(B2WZ1RFMCalmFixedSphereEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _filtered(self)


@configclass
class B2WZ1RFMCalmFilterPaperEnvCfg(B2WZ1RFMCalmFixedPaperEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _filtered(self)


@configclass
class B2WZ1RFMCalmPPORunnerCfg(B2WZ1RFMPPORunnerCfg):
    def __post_init__(self):
        parent = getattr(super(), "__post_init__", None)
        if parent is not None:
            parent()
        self.algorithm.entropy_coef = 0.002


# rfm16: the CalmFilter task trained FROM SCRATCH.
#
# Fine-tuning rfm14 (rfm15b/c/d) un-learns its bang-bang arm too slowly to be
# practical: the arm's commands still jumped 2.3 per step after 600 iterations.
# Two reasons. The hold is a RELAY controller -- full command one way, then the
# other -- and a relay with lag settles into a limit cycle, the ~1 Hz wobble;
# shake_diag.py splits it as 18-22 mm of hand motion relative to the body against
# 4-7 mm from the body itself, with legs and wheels smooth (0.06-0.10). And PPO's
# KL limit is measured in units of the action std, so capping the arm's std from
# ~2.0 to 0.3 also cut how far its mean may move per update by ~7x.
# A policy that never learns the relay does not have to un-learn it.
@configclass
class B2WZ1RFMScratchPPORunnerCfg(B2WZ1RFMPPORunnerCfg):
    def __post_init__(self):
        parent = getattr(super(), "__post_init__", None)
        if parent is not None:
            parent()
        # between rfm14's constant 0.01 (the arm noise grew to 2.0) and the
        # fine-tune's 0.002 (too little to learn a whole task from nothing)
        self.algorithm.entropy_coef = 0.005


# rfm16b: rfm16 with the arm rate charged on the FILTERED targets (see
# rfm.ArmActionRate, use_processed) -- rfm16's raw-output charge collapsed the
# arm's exploration noise.
@configclass
class B2WZ1RFMScratchBSphereEnvCfg(B2WZ1RFMCalmFilterSphereEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.arm_action_rate.params["use_processed"] = True


@configclass
class B2WZ1RFMScratchBPaperEnvCfg(B2WZ1RFMCalmFilterPaperEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.arm_action_rate.params["use_processed"] = True


# rfm17: CalmFilter + the hand's integral error as an observation (rfm.py,
# HandErrorIntegral), for actor and critic. Appended LAST in both groups, so a
# CalmFilter checkpoint extends by zero-initialised input columns
# (training/add_obs_columns.py) and starts out behaving exactly as before.
def _integral(cfg):
    from .rfm import HandErrorIntegral

    for group in (cfg.observations.policy, cfg.observations.critic):
        group.hand_err_int = ObsTerm(func=HandErrorIntegral, params={"command_name": "ee_pose"})


@configclass
class B2WZ1RFMIntegralSphereEnvCfg(B2WZ1RFMCalmFilterSphereEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _integral(self)


@configclass
class B2WZ1RFMIntegralPaperEnvCfg(B2WZ1RFMCalmFilterPaperEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _integral(self)


# ---------------------------------------------------------------------------
# rfm18 ABLATIONS: the smallest treatment that removes the shake.
#
# rfm15d/rfm17 did calm the arm (0-3 mm on target) but gave up orientation
# (shake_diag on-target ori 1.0-1.3 rad against rfm14's 0.11) and settled 4-6 cm
# short: with arm-rate and torque-rate penalties plus a stillness factor stacked
# on the reward, parking the arm motionless near the target became the optimum.
#   A  remove only the REASON for bang-bang (arm gains no longer randomised) and
#      the MEANS (low-pass filter in the arm's actuation), plus clean arm velocity
#      observations. No new reward terms.
#   B  A + the arm rate penalty on the filtered targets the motors receive.
def _ablation_base(cfg):
    from isaaclab.envs.mdp.actions import JointPositionActionCfg as _J  # noqa: F401

    from .rfm import FilteredJointPositionActionCfg

    cfg.events.randomize_actuator_gains.params["asset_cfg"] = _SEC("robot", joint_names=[LOCO_JOINT_REGEX])
    old = cfg.actions.arm_action
    cfg.actions.arm_action = FilteredJointPositionActionCfg(
        asset_name=old.asset_name, joint_names=old.joint_names, scale=old.scale,
        use_default_offset=old.use_default_offset, preserve_order=old.preserve_order,
        clip=old.clip, alpha=0.25)
    cfg.observations.policy.joint_vel.noise = PerJointUniformNoiseCfg(
        half_width=tuple([1.5] * 16 + [0.15] * 6))


@configclass
class B2WZ1RFMAblASphereEnvCfg(B2WZ1RFMSphereEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _ablation_base(self)


@configclass
class B2WZ1RFMAblBSphereEnvCfg(B2WZ1RFMAblASphereEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.arm_action_rate = RewTerm(
            func=ArmActionRate, weight=-0.1,
            params={"term_name": "arm_action", "w2": 0.5, "use_processed": True})


# rfm18 F/G: also stop randomising the ARM's link masses.
#
# Ablations A/B (fixed arm gains + filter) kept rfm14's precision (15/16 on
# target, ori 0.18-0.22 rad) but not its calm: the arm's raw commands still
# jumped 2.4-2.9 per step and the hand wobbled 12-16 mm. One hidden load was
# left: every link but the base has its mass scaled 0.7-1.3 at startup, arm
# links included, so the gravity sag of a smooth controller is still unknown to
# a memoryless policy, and the relay still pays. Fixed masses make the sag a
# fixed function of the pose, which a smooth policy can learn exactly.
#   F  B + arm masses fixed
#   G  F + the hand's integral error as an observation (payload robustness
#      comes back through the integral rather than through bang-bang)
def _fixed_arm_mass(cfg):
    cfg.events.randomize_rigid_body_mass_others.params["asset_cfg"] = _SEC(
        "robot", body_names=["^(?!.*base_link)(?!link0[0-9]).*"])


@configclass
class B2WZ1RFMAblFSphereEnvCfg(B2WZ1RFMAblBSphereEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _fixed_arm_mass(self)


@configclass
class B2WZ1RFMAblGSphereEnvCfg(B2WZ1RFMAblFSphereEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _integral(self)


# rfm19: incremental arm commands (rfm.py, DeltaJointPositionAction) on top of
# fixed arm gains and masses and clean arm velocity observations. No filter (the
# target is already an integral of the commands) and none of the rfm15 penalties.
@configclass
class B2WZ1RFMDeltaSphereEnvCfg(B2WZ1RFMSphereEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        from .rfm import DeltaJointPositionActionCfg, arm_action_l2

        self.events.randomize_actuator_gains.params["asset_cfg"] = _SEC("robot", joint_names=[LOCO_JOINT_REGEX])
        _fixed_arm_mass(self)
        self.observations.policy.joint_vel.noise = PerJointUniformNoiseCfg(
            half_width=tuple([1.5] * 16 + [0.15] * 6))
        old = self.actions.arm_action
        self.actions.arm_action = DeltaJointPositionActionCfg(
            asset_name=old.asset_name, joint_names=old.joint_names, scale=1.0,
            use_default_offset=True, preserve_order=old.preserve_order,
            delta_scale=0.05, window=0.4, action_clip=2.0)
        self.rewards.arm_action_l2 = RewTerm(func=arm_action_l2, weight=-0.01,
                                             params={"term_name": "arm_action"})
        self.rewards.arm_action_rate = RewTerm(
            func=ArmActionRate, weight=-0.02, params={"term_name": "arm_action", "w2": 0.5})


@configclass
class B2WZ1RFMDeltaPaperEnvCfg(B2WZ1RFMDeltaSphereEnvCfg):
    PHASE_MODE = "paper"
