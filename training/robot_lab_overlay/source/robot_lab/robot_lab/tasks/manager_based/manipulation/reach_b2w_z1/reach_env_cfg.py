# Local addition (not upstream robot_lab).
#
# Phase 4, step 2: the Z1 learns to put its hand on target points while the B2W
# stands still. The mirror image of the arm-as-cargo step: there the policy
# drove the legs and the arm was held; here the policy drives only the 6 arm
# joints and the legs and wheels are held by their motors.
#
# Built on Isaac Lab's tuned reach task (isaaclab_tasks .../manipulation/reach),
# changing only what a legged, floating-base robot needs.

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.unitree_b2w.arm_cargo_env_cfg import (
    UNITREE_B2W_Z1_CFG,
    hold_default_joint_pos,
)

ARM = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
LEGS_AND_WHEELS = [".*_hip_joint", ".*_thigh_joint", ".*_calf_joint", ".*_foot_joint"]
HAND = "link06"  # last Z1 link (the flange; there is no gripper in this model yet)

# With no policy on the legs, robot_lab's locomotion gains (stiffness 160) sag
# the body onto its belly -- found in Rung 0b, where 150 did exactly that and 800
# held the stance. Wheels get real damping so the body does not roll away when
# the arm swings (the locomotion value, 1.0, is nearly free-wheeling).
#
# The legs also move to PhysX's IMPLICIT drive. robot_lab models them as
# DCMotorCfg, an explicit actuator whose torque Isaac Lab computes in Python each
# step; that suits a policy re-targeting them every step, but it is not how the
# static stance was proven. Rung 0b held 800/20 through the implicit drive.
_act = UNITREE_B2W_Z1_CFG.actuators
UNITREE_B2W_Z1_STANDING_CFG = UNITREE_B2W_Z1_CFG.replace(
    actuators={
        **_act,
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            stiffness=800.0,
            damping=20.0,
            effort_limit_sim={".*_hip_joint": 200.0, ".*_thigh_joint": 200.0, ".*_calf_joint": 320.0},
        ),
        "wheel": _act["wheel"].replace(damping=20.0),
    },
)
for _k in ("hip", "thigh", "calf"):
    UNITREE_B2W_Z1_STANDING_CFG.actuators.pop(_k)


@configclass
class B2WZ1ReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # --- timing: the locomotion tasks' 200 Hz physics / 50 Hz policy, not the
        # reach template's 60/30 Hz. Needed for the wheeled-legged contact, and it
        # keeps the arm policy on the same clock as the locomotion policy for
        # when the two are combined in step 3.
        self.sim.dt = 0.005
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 12.0
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # --- scene: robot on the floor. The template puts the ground 1.05 m down
        # under a table for a fixed-base arm; neither applies here.
        self.scene.table = None
        self.scene.ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
        self.scene.robot = UNITREE_B2W_Z1_STANDING_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # --- command: hand target, in the robot's BODY frame, so it moves with
        # the robot. Box chosen from the measured Z1 geometry: shoulder at
        # (0.166, 0, 0.191), reach ~0.69 m, body top at z = 0.243. Every point is
        # above the body and <= ~0.62 m from the shoulder. Position only for now:
        # orientation fixed and not rewarded (the MPC paper's simplification).
        c = self.commands.ee_pose
        c.body_name = HAND
        c.ranges.pos_x = (0.30, 0.60)
        c.ranges.pos_y = (-0.25, 0.25)
        c.ranges.pos_z = (0.30, 0.55)
        c.ranges.roll = (0.0, 0.0)
        c.ranges.pitch = (0.0, 0.0)
        c.ranges.yaw = (0.0, 0.0)

        # --- actions: the 6 arm joints only. Legs and wheels are not in the
        # action space, so their actuators hold the default stance.
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=ARM, scale=0.5, use_default_offset=True, preserve_order=True
        )

        # --- observations: arm joints only. The template reads ALL joints, which
        # would hand the policy 16 leg/wheel readings it cannot act on.
        arm = SceneEntityCfg("robot", joint_names=ARM, preserve_order=True)
        self.observations.policy.joint_pos.params = {"asset_cfg": arm}
        self.observations.policy.joint_vel.params = {"asset_cfg": arm.replace()}

        # --- rewards: the template's coarse + fine position tracking, unchanged.
        for term in (self.rewards.end_effector_position_tracking,
                     self.rewards.end_effector_position_tracking_fine_grained):
            term.params["asset_cfg"].body_names = [HAND]
        self.rewards.end_effector_orientation_tracking = None  # position only, for now
        self.rewards.joint_vel.params["asset_cfg"] = arm.replace()
        # Not in the template (its arms have no delicate end stops to worry about):
        # the Z1's motors are small, so keep them off their joint limits.
        self.rewards.arm_joint_limits = RewTerm(
            func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": arm.replace()}
        )

        # --- resets. The template only perturbs joints of a fixed-base arm; a
        # floating robot also needs its body put back upright, and the legs back
        # in stance (the template's +/-50% "by scale" reset would fold the legs
        # into random shapes). The arm keeps the template's variation.
        still = (0.0, 0.0)
        self.events.reset_base = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {k: still for k in ("x", "y", "z", "roll", "pitch", "yaw")},
                "velocity_range": {k: still for k in ("x", "y", "z", "roll", "pitch", "yaw")},
            },
        )
        self.events.reset_body_joints = EventTerm(
            func=mdp.reset_joints_by_scale,
            mode="reset",
            params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0),
                    "asset_cfg": SceneEntityCfg("robot", joint_names=LEGS_AND_WHEELS)},
        )
        self.events.reset_robot_joints.params["asset_cfg"] = arm.replace()
        # Legs have no action term, so their motor targets would stay at Isaac
        # Lab's initial ZERO -- straight legs -- instead of the stance.
        self.events.hold_stance = EventTerm(
            func=hold_default_joint_pos, mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEGS_AND_WHEELS)},
        )

        # --- terminations. The template's only ending is the time-out, which is
        # fine for a bolted-down arm; here a swinging arm could tip the robot, and
        # it would just lie there unnoticed until the 12 s ran out. Ending the
        # episode past ~29 deg makes tipping costly (lost future reward) and makes
        # every occurrence visible in the log as Episode_Termination/base_tilted.
        self.terminations.base_tilted = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.5})


@configclass
class B2WZ1ReachEnvCfg_PLAY(B2WZ1ReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
