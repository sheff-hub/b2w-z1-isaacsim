# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
"""Fine-tune ETH's SRU B2W locomotion policy on our B2W + Z1 + gripper, for a normal stance.

ETH's deployment policy (leggedrobotics/sru-robot-deployment, policy_force_new.onnx) keeps
commanding both rear hips ~0.3 rad inward, and the rear wheel track shrinks from ~0.4 m to
0.1-0.2 m while driving (even below 0 in turns) -- on our robot, without the arm, and on
ETH's own b2w_rsl.usd alike (probe_track.py). This task reproduces ETH's controller
contract exactly, so its weights load unchanged (training/train_eth_ft.py):

  observations (60, unscaled)  base lin vel, base ang vel, projected gravity, [vx vy wz],
                               joint pos - default wrapped to [-2pi, 2pi), joint vel, last
                               action -- joints grouped hip/thigh/calf/foot x FL/FR/RL/RR
  actions (16)                 legs: default + 0.5 a; wheels: velocity 5 a; 50 Hz
  drives / stance              implicit PD 100 / 3.5 (calf effort 320), wheel damping 3.0;
                               thigh 0.4, calf -1.3, spawn 0.75 m

and adds what it is missing: a reward for the wheel track width (front and rear, in the
base frame) and a mild hip-deviation penalty, on top of robot_lab's velocity-tracking task.
The arm and gripper are held at the stow pose (no action term).
"""

import math

import torch

import isaaclab.envs.mdp as il_mdp
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

from .arm_cargo_env_cfg import ARM_STOW, UnitreeB2WZ1FlatEnvCfg, hold_default_joint_pos
from .fetch_env_cfg import GRIPPER_CLOSED, UNITREE_B2W_Z1_GRIPPER_CFG

ETH_LEGS = [f"{leg}_{part}_joint" for part in ("hip", "thigh", "calf") for leg in ("FL", "FR", "RL", "RR")]
ETH_WHEELS = [f"{leg}_foot_joint" for leg in ("FL", "FR", "RL", "RR")]
ETH_JOINTS = ETH_LEGS + ETH_WHEELS
ETH_STANCE = {"hip": 0.0, "thigh": 0.4, "calf": -1.3, "foot": 0.0}
TRACK_WIDTH = 0.42  # m, wheel-to-wheel across the body: what the front keeps on its own, and model_2600 both ends


def eth_joint_pos(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Joint positions minus default, wrapped to [-2pi, 2pi) like b2w_controllers.cpp."""
    asset = env.scene[asset_cfg.name]
    rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.remainder(rel + 2 * math.pi, 4 * math.pi) - 2 * math.pi


def wheel_tracks(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """(front, rear) wheel track widths in the base frame, m. asset_cfg bodies: FL, FR, RL, RR feet."""
    asset = env.scene[asset_cfg.name]
    rel = asset.data.body_pos_w[:, asset_cfg.body_ids] - asset.data.root_pos_w[:, None]
    q = asset.data.root_quat_w[:, None].expand(-1, rel.shape[1], -1)
    yb = quat_apply_inverse(q.reshape(-1, 4), rel.reshape(-1, 3)).reshape(rel.shape)[..., 1]
    return torch.stack([yb[:, 0] - yb[:, 1], yb[:, 2] - yb[:, 3]], dim=1)


def track_width_error(env, asset_cfg: SceneEntityCfg, target: float) -> torch.Tensor:
    """|front - target| + |rear - target| (penalty), only while upright."""
    err = (wheel_tracks(env, asset_cfg) - target).abs().sum(dim=1)
    return err * torch.clamp(-env.scene[asset_cfg.name].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7


def hip_deviation(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    d = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(d * d, dim=1)


def straight_gate(env, vy_scale: float = 0.1, wz_scale: float = 0.3) -> torch.Tensor:
    """1 when the command is straight driving or standing, -> 0 with sideways / turning commands."""
    c = env.command_manager.get_command("base_velocity")
    return torch.exp(-torch.square(c[:, 1] / vy_scale) - torch.square(c[:, 2] / wz_scale))


def track_width_error_straight(env, asset_cfg: SceneEntityCfg, target: float) -> torch.Tensor:
    """track_width_error, only while the command is straight driving / standing."""
    return track_width_error(env, asset_cfg, target) * straight_gate(env)


def hip_deviation_straight(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return hip_deviation(env, asset_cfg) * straight_gate(env)


class MixedVelocityCommand(il_mdp.UniformVelocityCommand):
    """Uniform velocity command, with a share of envs given straight driving (vy = wz = 0) or
    turning on the spot (vx = vy = 0) -- how the fetch pipeline drives; uniform sampling alone
    almost never produces a straight command (|vy| < 0.1 and |wz| < 0.3: ~6 %)."""

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        r = torch.rand(len(env_ids), device=self.device)
        straight, spin = env_ids[r < 0.35], env_ids[(r >= 0.35) & (r < 0.5)]
        self.vel_command_b[straight, 1:] = 0.0
        self.vel_command_b[spin, :2] = 0.0


@configclass
class MixedVelocityCommandCfg(il_mdp.UniformVelocityCommandCfg):
    class_type: type = MixedVelocityCommand


def hip_target_deviation(env, vy_scale: float = 0.1) -> torch.Tensor:
    """Squared hip TARGETS (0.5 * action, relative to the zero default), except while moving
    sideways. The measured hips are the wrong signal: a loaded wheel cannot slide sideways, so
    while driving straight friction holds the hips near zero whatever the target -- ETH's
    constant ~0.3 rad inward rear targets only take effect in a turn (the wheels slip), and the
    stance then stays narrow (v5 at 250: rear 0.38 m until the first turn, 0.19-0.21 after)."""
    a = env.action_manager.action[:, :4] * 0.5
    c = env.command_manager.get_command("base_velocity")
    return torch.sum(a * a, dim=1) * torch.exp(-torch.square(c[:, 1] / vy_scale))


def track_lin_vel_y_exp(env, std: float) -> torch.Tensor:
    """Sideways velocity tracking with a tight kernel (|err| 0.1 m/s -> 0.37 at std 0.1)."""
    asset = env.scene["robot"]
    c = env.command_manager.get_command("base_velocity")
    return torch.exp(-torch.square(c[:, 1] - asset.data.root_lin_vel_b[:, 1]) / std**2)


ETH_WEIGHTS = "/workspace/eth/policy_force_new_rslrl.npz"


def eth_imitation(env, weights: list[float], hip_turn_weight: float = 0.0) -> torch.Tensor:
    """Squared distance between the action taken and what the ORIGINAL ETH policy would do on
    the same input, per ETH action index weighted by `weights` (16). A frozen copy of ETH's
    network runs alongside, its LSTM state carried per env and zeroed when an episode starts.
    At reward time env.obs_buf still holds o_t, the input the action a_t was computed from."""
    ref = getattr(env, "_eth_reference", None)
    if ref is None:
        import numpy as np
        W = np.load(ETH_WEIGHTS)
        lstm = torch.nn.LSTM(60, 256, 1).to(env.device)
        lstm.load_state_dict({k.replace("memory_a.rnn.", ""): torch.tensor(v) for k, v in W.items()
                              if k.startswith("memory_a.rnn.")})
        dims, layers = [256, 256, 256, 128, 64, 16], []
        for i in range(5):
            layers += [torch.nn.Linear(dims[i], dims[i + 1])] + ([torch.nn.ELU()] if i < 4 else [])
        mlp = torch.nn.Sequential(*layers).to(env.device)
        mlp.load_state_dict({k.replace("actor.", ""): torch.tensor(v) for k, v in W.items() if k.startswith("actor.")})
        h = torch.zeros(1, env.num_envs, 256, device=env.device)
        ref = env._eth_reference = {"lstm": lstm.eval(), "mlp": mlp.eval(), "h": h, "c": h.clone(),
                                    "w": torch.tensor(weights, device=env.device)}
    obs = env.obs_buf["policy"]
    new = env.episode_length_buf <= 1
    with torch.no_grad():
        ref["h"][:, new] = 0.0
        ref["c"][:, new] = 0.0
        y, (ref["h"], ref["c"]) = ref["lstm"](obs[None], (ref["h"], ref["c"]))
        a_ref = ref["mlp"](y[0])
    err = torch.square(env.action_manager.action - a_ref)
    out = torch.sum(ref["w"] * err, dim=1)
    if hip_turn_weight:  # the hips imitate ETH while sideways / turning (its way of doing those)
        out = out + hip_turn_weight * (1.0 - straight_gate(env)) * err[:, :4].sum(dim=1)
    return out


def _eth_robot():
    r = UNITREE_B2W_Z1_GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    acts = dict(r.actuators)
    acts["hip"] = ImplicitActuatorCfg(joint_names_expr=[".*_hip_joint"], effort_limit_sim=200.0, velocity_limit_sim=23.0,
                                      stiffness=100.0, damping=3.5, friction=0.0)
    acts["thigh"] = ImplicitActuatorCfg(joint_names_expr=[".*_thigh_joint"], effort_limit_sim=200.0,
                                        velocity_limit_sim=23.0, stiffness=100.0, damping=3.5, friction=0.0)
    acts["calf"] = ImplicitActuatorCfg(joint_names_expr=[".*_calf_joint"], effort_limit_sim=320.0,
                                       velocity_limit_sim=14.0, stiffness=100.0, damping=3.5, friction=0.0)
    acts["wheel"] = ImplicitActuatorCfg(joint_names_expr=[".*_foot_joint"], effort_limit_sim=20.0,
                                        velocity_limit_sim=50.0, stiffness=0.0, damping=3.0, friction=0.0)
    jp = {k: v for k, v in r.init_state.joint_pos.items() if not any(k.endswith(f"_{p}_joint") for p in ETH_STANCE)}
    jp.update({f".*_{p}_joint": v for p, v in ETH_STANCE.items()})
    return r.replace(actuators=acts, soft_joint_pos_limit_factor=0.95,
                     init_state=r.init_state.replace(pos=(0.0, 0.0, 0.75), joint_pos=jp))


@configclass
class EthPolicyCfg(ObsGroup):
    """ETH's 60 inputs, in ETH's order, unscaled."""

    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
    projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos = ObsTerm(func=eth_joint_pos, params={"asset_cfg": SceneEntityCfg("robot", joint_names=ETH_JOINTS,
                                                                                 preserve_order=True)},
                        noise=Unoise(n_min=-0.01, n_max=0.01))
    joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=ETH_JOINTS,
                                                                                 preserve_order=True)},
                        noise=Unoise(n_min=-1.5, n_max=1.5))
    actions = ObsTerm(func=mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class EthCriticCfg(EthPolicyCfg):
    """The same, noise-free, plus what makes the stance reward predictable: the wheel tracks
    and the base height."""

    tracks = ObsTerm(func=wheel_tracks, params={"asset_cfg": SceneEntityCfg(
        "robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True)})
    base_height = ObsTerm(func=mdp.base_pos_z)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class UnitreeB2WZ1EthFinetuneEnvCfg(UnitreeB2WZ1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = _eth_robot()
        self.scene.num_envs = 2048

        # actions: ETH's mapping, in ETH's joint order (so last_action is ETH's "previous action")
        self.actions.joint_pos.joint_names = ETH_LEGS
        self.actions.joint_pos.preserve_order = True
        self.actions.joint_pos.scale = 0.5
        self.actions.joint_vel.joint_names = ETH_WHEELS
        self.actions.joint_vel.preserve_order = True
        self.actions.joint_vel.scale = 5.0

        self.observations.policy = EthPolicyCfg()
        self.observations.critic = EthCriticCfg()

        # commands: plain uniform velocity (robot_lab's threshold command zeroes |v| <= 0.2,
        # which ETH was never trained with); ETH's tested range
        cmd = self.commands.base_velocity
        cmd.class_type = il_mdp.UniformVelocityCommand
        cmd.heading_command = False
        cmd.rel_heading_envs = 0.0
        cmd.rel_standing_envs = 0.1
        cmd.ranges.lin_vel_x = (-1.0, 1.5)
        cmd.ranges.lin_vel_y = (-0.4, 0.4)
        cmd.ranges.ang_vel_z = (-1.0, 1.0)
        cmd.ranges.heading = None

        # arm and gripper held still (no action terms)
        self.events.hold_arm_stow = EventTerm(
            func=hold_default_joint_pos, mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(ARM_STOW) + ["jointGripper"])},
        )
        # start upright: ETH's policy is a driving policy, not a fall-recovery one
        self.events.randomize_reset_base.params["pose_range"].update({"roll": (-0.1, 0.1), "pitch": (-0.1, 0.1)})

        # the stance terms this fine-tune is for
        feet = SceneEntityCfg("robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True)
        self.rewards.track_width = RewTerm(func=track_width_error, weight=-6.0,
                                           params={"asset_cfg": feet, "target": TRACK_WIDTH})
        self.rewards.hip_deviation = RewTerm(func=hip_deviation, weight=-0.5,
                                             params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"])})
        self.rewards.joint_pos_penalty.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=ETH_LEGS)
        self.rewards.stand_still.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=ETH_LEGS)
        _ = GRIPPER_CLOSED  # (the parent already pruned its zero-weight rewards)


@configclass
class UnitreeB2WZ1EthFinetuneV2EnvCfg(UnitreeB2WZ1EthFinetuneEnvCfg):
    """v2: end the episode when the robot tips past ~57 deg (robot_lab's B2W task has no fall
    termination at all, so a fallen robot lay there for the rest of its 20 s), lighter stance
    weights and no command curriculum (v1, -6 with the curriculum: the stance was fixed by
    iteration 600 but the robot followed forward 0.3 m/s at 46 %, sideways 0-28 %, turning 48 %
    -- moving less was its way to keep the stance), plus a small flat-orientation penalty."""

    def __post_init__(self):
        super().__post_init__()
        from isaaclab.managers import TerminationTermCfg as DoneTerm
        self.terminations.bad_orientation = DoneTerm(func=il_mdp.bad_orientation, params={"limit_angle": 1.0})
        self.rewards.track_width.weight = -3.0
        self.rewards.hip_deviation.weight = -0.2
        # the full command range from the start: robot_lab's curriculum starts at 10 % and only
        # widens while tracking is good -- in v1 it stayed narrow and the policy forgot large
        # commands (forward 0.3 m/s followed at 46 % at iteration 600, ETH 106 %)
        for name in ("command_levels_lin_vel", "command_levels_ang_vel"):
            if hasattr(self.curriculum, name):
                setattr(self.curriculum, name, None)
        self.rewards.flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
        self.rewards.is_terminated = RewTerm(func=mdp.is_terminated, weight=-50.0)


@configclass
class UnitreeB2WZ1EthFinetuneV3EnvCfg(UnitreeB2WZ1EthFinetuneV2EnvCfg):
    """v3: stay ETH, change the stance. PPO fine-tuning with robot_lab's general rewards (v1,
    v2) changed the whole gait -- at v2 iteration 300: small forward commands reversed, the
    diagonal followed at 32 %, 0.67 m drift standing with the arm out (ETH 0.095). So: an
    imitation reward toward the frozen original ETH policy (thighs/calves 1, wheels 0.3, hips
    0.05 -- the hips are what should change), the stance reward, velocity tracking and the fall
    terms; robot_lab's posture shaping (default-pose pull, mirror, contact terms) removed."""

    def __post_init__(self):
        super().__post_init__()
        w = [0.05] * 4 + [1.0] * 8 + [0.3] * 4  # ETH order: hips, thighs, calves, wheels
        self.rewards.eth_imitation = RewTerm(func=eth_imitation, weight=-4.0, params={"weights": w})
        self.rewards.track_width.weight = -3.0
        self.rewards.hip_deviation = None
        for name in ("joint_pos_penalty", "stand_still", "joint_mirror", "feet_contact_without_cmd", "contact_forces",
                     "joint_power", "joint_acc_l2", "joint_acc_wheel_l2"):
            if getattr(self.rewards, name, None) is not None:
                setattr(self.rewards, name, None)


@configclass
class UnitreeB2WZ1EthFinetuneV4EnvCfg(UnitreeB2WZ1EthFinetuneV3EnvCfg):
    """v4, from v3 at iteration 250 (straight driving kept: forward 0.1-1.0 m/s followed at
    90-103 %; but the rear track still closed to 0.26 m after the first turn and stayed there,
    yaw followed at 73 %): the target track is the one the legs give with the hips at zero
    (0.40 m, measured with model_2600 standing; 0.42 made the front splay out to 0.45-0.52),
    a direct pull of every hip toward zero (the reward that cannot be satisfied by narrowing one
    end and widening the other), the hips and most of the wheels out of the imitation (turning
    with a wider rear needs different wheel speeds), and tighter velocity kernels so a 30 %
    yaw shortfall costs something (std 0.5 -> 0.3; yaw weight 1.5 -> 3)."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.track_width.params["target"] = 0.40
        self.rewards.hip_deviation = RewTerm(func=hip_deviation, weight=-2.0,
                                             params={"asset_cfg": SceneEntityCfg("robot", joint_names=ETH_LEGS[:4])})
        self.rewards.eth_imitation.params["weights"] = [0.0] * 4 + [1.0] * 8 + [0.1] * 4
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.3
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.3
        self.rewards.track_ang_vel_z_exp.weight = 3.0


@configclass
class UnitreeB2WZ1EthFinetuneV5EnvCfg(UnitreeB2WZ1EthFinetuneV3EnvCfg):
    """v5, from ETH. Measured in the fine-tune env with fixed commands (no randomisation):
    ETH moves sideways (0.2 m/s -> 0.175) and turns (0.5 rad/s -> 0.51) BY narrowing, rear
    track 0.18-0.22 m -- and keeps it narrow while driving straight too (0.24 at 0.1 m/s).
    v3 at 400 held 0.35 m everywhere and lost both: sideways 0.077, turning 0.284. So the
    stance terms apply only while the command is straight / standing (gate exp(-(vy/0.1)^2 -
    (wz/0.3)^2)), and while sideways / turning the hips imitate ETH instead; target 0.40 m (the
    track with the hips at zero); tighter velocity kernels (v4)."""

    def __post_init__(self):
        super().__post_init__()
        feet = self.rewards.track_width.params["asset_cfg"]
        self.rewards.track_width = RewTerm(func=track_width_error_straight, weight=-4.0,
                                           params={"asset_cfg": feet, "target": 0.40})
        self.rewards.hip_deviation = RewTerm(func=hip_deviation_straight, weight=-1.0,
                                             params={"asset_cfg": SceneEntityCfg("robot", joint_names=ETH_LEGS[:4])})
        self.rewards.eth_imitation.params = {"weights": [0.0] * 4 + [1.0] * 8 + [0.3] * 4, "hip_turn_weight": 1.0}
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.3
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.3
        self.rewards.track_ang_vel_z_exp.weight = 3.0
        cmd = self.commands.base_velocity
        self.commands.base_velocity = MixedVelocityCommandCfg(
            asset_name=cmd.asset_name, resampling_time_range=cmd.resampling_time_range,
            rel_standing_envs=cmd.rel_standing_envs, rel_heading_envs=0.0, heading_command=False,
            debug_vis=False, ranges=cmd.ranges)


@configclass
class UnitreeB2WZ1EthFinetuneV6EnvCfg(UnitreeB2WZ1EthFinetuneV5EnvCfg):
    """v6, from v5 at 300: penalise the hip targets themselves (hip_target_deviation) except
    while moving sideways, and a tight sideways-tracking term (v5 at 250 followed sideways
    0.2 m/s with 0.08-0.10 in the fine-tune env, 1-2 % from standing in the benchmark)."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.hip_target_deviation = RewTerm(func=hip_target_deviation, weight=-3.0)
        self.rewards.track_lin_vel_y_exp = RewTerm(func=track_lin_vel_y_exp, weight=2.0, params={"std": 0.1})


@configclass
class UnitreeB2WZ1EthFinetuneV8EnvCfg(UnitreeB2WZ1EthFinetuneV6EnvCfg):
    """v8, from v7 (v6 + desired_kl 0.02): sideways driving is lost in every fine-tune so far
    (v6 at 400: 0.2 m/s -> 0.086 in the fine-tune env, ETH 0.167) although nothing in the
    stance terms acts while moving sideways -- so the suspects are robot_lab's smoothness
    penalties, which ETH was not trained with (body roll/pitch rate, vertical velocity, flat
    orientation): removed, and the sideways term doubled."""

    def __post_init__(self):
        super().__post_init__()
        for name in ("ang_vel_xy_l2", "lin_vel_z_l2", "flat_orientation_l2"):
            setattr(self.rewards, name, None)
        self.rewards.track_lin_vel_y_exp.weight = 4.0
