# Local addition (not upstream robot_lab).
#
# Reward Fusion Module (RFM), after:
#   K. Jiang, Z. Fu, J. Guo, W. Zhang, H. Chen, "Learning Whole-Body
#   Loco-Manipulation for Omni-Directional Task Space Pose Tracking with a
#   Wheeled-Quadrupedal-Manipulator", IEEE RA-L, accepted Nov 2024
#   (arXiv:2412.03012).
#
# WHY WE ARE COPYING THIS
# =======================
# Our own policies fail in two specific ways that this paper's reward structure
# is designed to prevent:
#
#   1. Given a target metres away, our policy STRETCHES THE ARM AT IT and falls
#      over -- base height 0.55 -> 0.12 m and 68 deg of tilt within a second.
#      It never learned "too far to reach, drive first", because reaching and
#      driving were simply added together and reaching always paid something.
#
#   2. It trades balance for reach: nothing stops the policy risking a fall to
#      shave a centimetre off the hand error.
#
# RFM addresses both, and it does so by changing HOW rewards combine rather than
# by adding more of them. Three ideas, all from the paper:
#
#   REWARD PRIORITIZATION (RP).  Instead of a weighted sum, rewards MULTIPLY:
#
#       r_mani = r_reg + r_reg * (r_ep + r_ep * r_eo)
#
#   Read it as a chain of permissions. Orientation tracking (r_eo) only pays
#   once position (r_ep) is good, because it enters as a product with r_ep. And
#   BOTH only pay once the regularisation term r_reg is good -- so a policy that
#   is about to fall over (r_reg -> 0) earns nothing at all for chasing the
#   target. The paper's words: "the agent will not choose to take risks of
#   failure to get a higher tracking-related reward". That is fault 2, fixed
#   structurally rather than by tuning a penalty.
#
#   LOCO-MANI FUSION (LMF).  A phase variable blends two whole reward groups:
#
#       r = (1 - D) * r_mani + D * r_loco + r_basic,     D = sigmoid(eps_ref)
#
#   When the target is far, D -> 1 and only LOCOMOTION pays; as the robot closes
#   in, D -> 0 and MANIPULATION takes over. Continuous, so there is no mode
#   switch and no hand-tuned standoff distance -- which is exactly the scripted
#   state machine in project/goto.py, replaced by something learned. That is
#   fault 1. Ablation in the paper: removing LMF costs 99.0% -> 82.1% success
#   and produces "jittery motion".
#
#   MICRO-ENHANCEMENT.  r* = r + r^M (M > 1) sharpens the reward gradient only
#   where the error is already small, so precision improves without making the
#   tracking term dominate everywhere. Paper: EE error 0.036 -> 0.022 m.
#
# Plus a cumulative (integral, PID-like) error penalty to escape local minima.
#
# WHAT WE HAD TO DEFINE OURSELVES
# ===============================
# The paper deliberately leaves r_reg open: "The specific meaning of
# 'regularization' varies according to the requirements of users and tasks ...
# our customized definition of 'regularization' is omitted." It does say what
# the two versions must mean, and the asymmetry matters:
#
#   * LOCOMOTION regularisation = "stable base height and flat base orientation"
#   * MANIPULATION regularisation = NOT that, "because in manipulation task, the
#     base need to bend or stretch to cooperate with the manipulator"
#
# So r_reg_loco below rewards a level body at nominal height, while r_reg_mani
# rewards only SMOOTHNESS (low angular rate, low vertical velocity) and says
# nothing about posture -- otherwise it would forbid the 36 cm crouch and 10 deg
# lean our robot uses to reach low targets, which is the whole point of the task.

from __future__ import annotations

import math

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.managers import RewardTermCfg
from isaaclab.utils.math import matrix_from_quat


def se3_distance(
    pos_err: torch.Tensor,
    quat_cur: torch.Tensor,
    quat_des: torch.Tensor,
    a1: float = 0.1,
    a2: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The paper's SE(3) distance, eq. (9):

        eps = a1 * ||log(R*_ee R_ee^T)||_F + a2 * ||P*_ee - P_ee||_2

    Returns (eps, position error, rotation angle). For a rotation of angle t,
    ||log(R)||_F = sqrt(2) * t, so the angle is used directly and a1 absorbs the
    constant.

    The paper gives a1 and a2 as "weighting parameters" without values, and the
    choice matters more than it looks: eps drives the PHASE variable, whose
    midpoint is 1.4 m. At a1 = 1 a half-turn of orientation error adds 3.14 to
    eps all by itself, which pins the phase in "locomotion" forever and the
    manipulation rewards never switch on. Position therefore dominates
    (a1 = 0.1: a half-turn is worth 0.31 m).
    """
    d_p = torch.norm(pos_err, dim=-1)
    # relative rotation angle, via the trace of R_des R_cur^T
    r_cur = matrix_from_quat(quat_cur)
    r_des = matrix_from_quat(quat_des)
    rel = torch.bmm(r_des, r_cur.transpose(1, 2))
    trace = rel[:, 0, 0] + rel[:, 1, 1] + rel[:, 2, 2]
    d_theta = torch.acos(torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0))
    return a1 * d_theta + a2 * d_p, d_p, d_theta


def phase(x: torch.Tensor, mu: float, slope: float) -> torch.Tensor:
    """D(x; mu, l) = 1 / (1 + exp(-5 (x - mu) / l)), eq. (12).

    1 means "far away, this is a locomotion problem", 0 means "close, this is a
    manipulation problem". The paper's tuning rule is mu = l = 2 * arm span,
    which for the Z1 (0.694 m of reach) puts the midpoint at about 1.4 m.
    """
    return torch.sigmoid(5.0 * (x - mu) / slope)


class RewardFusion(ManagerTermBase):
    """The fused (1 - D) * r_mani + D * r_loco term.

    r_basic (action rate, smoothness, collision, alive, power, torque) is NOT
    included here: it is added linearly in the paper too, so those stay as
    ordinary robot_lab reward terms and this class only implements the part that
    actually fuses.
    """

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.command_name = cfg.params["command_name"]
        p = cfg.params
        self.sigma_p = p.get("sigma_p", 0.25)          # position kernel width, m
        self.sigma_o = p.get("sigma_o", 0.5)           # orientation kernel width, rad
        self.sigma_reg = p.get("sigma_reg", 1.0)
        # Widened from 0.5: the schedule-matching reward is exp(-|eps_ref - eps|
        # / sigma_dw), and at 0.5 a robot that has fallen a metre behind scores
        # 0.02 -- a flat, vanishing gradient exactly where it most needs to be
        # told "you are getting warmer". Measured: the policy instead learned to
        # stand perfectly still collecting reg_loco ~ 1.0 and ignore the target.
        self.sigma_dw = p.get("sigma_dw", 0.5)
        # Back to 0.5. It was widened to 1.5 when the policy stood still ignoring
        # far targets -- before the closing-speed reward existed. That reward now
        # carries the approach, and at 1.5 a robot parked 0.7 m away still kept
        # r_dw ~ 0.63 for doing nothing; at 0.5 that falls to ~0.3.
        self.micro_m = p.get("micro_m", 2.0)           # M in r* = r + r^M
        self.gamma = p.get("gamma", 0.1)               # slack in the reference tracker
        # The paper clips the cumulative penalty at 20, but its kappa and its eps
        # are on a different scale to ours: measured with probe_rfm.py, our
        # e_cb reached 11 within 8 s and buried every other term (r_mani = -10.3).
        # It exists to break ties and nudge the agent out of local minima, so it
        # is scaled here to contribute at most ~2 against an r_mani whose useful
        # range is about 0-5.
        self.kappa_scale = p.get("kappa_scale", 0.002)
        self.cb_clip = p.get("cb_clip", 2.0)
        self.w_static_arm = p.get("w_static_arm", 0.15)
        self.w_progress = p.get("w_progress", 20.0)
        self.height_floor = p.get("height_floor", 0.30)   # crouch is fine, squatting is not
        # 40 was too soft to bite: the robot still squatted to 0.217 m for a
        # 0.45 m target, which is what "lowering its body too much" looks like
        # from outside. At 150 the floor is close to hard. 0.30 m costs no reach
        # -- wb15 served a 0.15 m target from 0.307 m of base height.
        self.w_height_floor = p.get("w_height_floor", 150.0)
        self.w_base_still = p.get("w_base_still", 2.0)
        # Phase from the ARM'S WORKSPACE (see __call__): a sphere of radius
        # reach_frac * arm_reach around the manipulator's mount.
        self.mount_idx = self.robot.find_bodies([p.get("mount_body", "link00")])[0][0]
        self.arm_reach = p.get("arm_reach", 0.694)
        self.reach_frac = p.get("reach_frac", 0.5)
        self.phase_width = p.get("phase_width", 0.25)
        self.blend_start_frac = p.get("blend_start_frac", 1.0)   # blend begins at full reach
        self.w_face = p.get("w_face", 1.0)         # unused since the camping fix
        self.w_face_prog = p.get("w_face_prog", 50.0)
        # "sphere": phase from the shoulder's distance to the target, blended
        #           from full reach down to half reach (our design).
        # "paper":  phase from the hand's SE(3) reference distance on its
        #           schedule, midpoint = slope = 2 x arm span (Jiang et al.,
        #           eqs. 11-12, taken literally). Everything else is identical,
        #           so the two can be compared on the phase alone.
        self.phase_mode = p.get("phase_mode", "sphere")
        # hold the arm still once the hand is on the target -- it shook
        self.w_arm_still = p.get("w_arm_still", 0.05)
        # rfm15: hand speed scales the TRACKING BONUS (None = off, as in rfm14)
        self.sigma_hand_v = p.get("sigma_hand_v", None)
        self.w_hand_ang = p.get("w_hand_ang", 0.05)
        # Tilt allowed for free while reaching, and the price of going past it.
        self.tilt_free = p.get("tilt_free", 0.26)        # ~15 deg
        self.height_ceiling = p.get("height_ceiling", 0.65)  # tallest measured stance
        self.w_height_track = p.get("w_height_track", 10.0)
        self.w_tilt = p.get("w_tilt", 20.0)
        self.nominal_height = p.get("nominal_height", 0.55)
        self.arm_joint_ids, _ = self.robot.find_joints(p.get("arm_joints", ["joint[1-6]"]))
        self.arm_stow = self.robot.data.default_joint_pos[:, self.arm_joint_ids].clone()
        # per-environment state
        self.e_cb = torch.zeros(env.num_envs, device=env.device)
        self.prev_eps = torch.zeros(env.num_envs, device=env.device)
        self.has_prev = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.prev_facing = torch.zeros(env.num_envs, device=env.device)
        self.has_prev_face = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.e_cb[env_ids] = 0.0
        self.has_prev[env_ids] = False
        self.has_prev_face[env_ids] = False
        return {}

    def __call__(
        self,
        env,
        asset_cfg: SceneEntityCfg,
        command_name: str,
        sigma_p: float = 0.25,
        sigma_o: float = 0.5,
        sigma_reg: float = 1.0,
        sigma_dw: float = 0.5,
        micro_m: float = 2.0,
        gamma: float = 0.1,
        kappa_scale: float = 0.02,
        cb_clip: float = 20.0,
        w_static_arm: float = 0.15,
        w_progress: float = 20.0,
        height_floor: float = 0.30,
        w_height_floor: float = 150.0,
        w_base_still: float = 2.0,
        mount_body: str = "link00",
        arm_reach: float = 0.694,
        reach_frac: float = 0.5,
        phase_width: float = 0.25,
        blend_start_frac: float = 1.0,
        w_face: float = 1.0,
        w_face_prog: float = 50.0,
        phase_mode: str = "sphere",
        w_arm_still: float = 0.05,
        sigma_hand_v: float | None = None,
        w_hand_ang: float = 0.05,
        tilt_free: float = 0.26,
        w_tilt: float = 20.0,
        height_ceiling: float = 0.65,
        w_height_track: float = 10.0,
        nominal_height: float = 0.55,
        arm_joints: list[str] | None = None,
    ) -> torch.Tensor:
        # Isaac Lab matches cfg.params against this signature by name, so every
        # configurable value has to appear here even though __init__ already
        # cached it.
        cmd = env.command_manager.get_term(self.command_name)
        d = self.robot.data

        body_idx = cmd.body_idx

        # A NEW TARGET IS A NEW EPISODE as far as this term is concerned. Without
        # this, the progress term sees the distance jump when the command is
        # resampled and reads it as enormous progress (or loss), and the
        # cumulative penalty carries a grudge from a target that no longer
        # exists.
        fresh = cmd.age <= env.step_dt * 1.5
        if bool(fresh.any()):
            self.e_cb[fresh] = 0.0
            self.has_prev[fresh] = False
            self.has_prev_face[fresh] = False

        # --- tracking errors, in the world frame where the target lives
        pos_err = cmd.pose_command_w[:, :3] - d.body_pos_w[:, body_idx]
        eps, d_p, d_theta = se3_distance(
            pos_err, d.body_quat_w[:, body_idx], cmd.pose_command_w[:, 3:]
        )

        # --- phase: how much of this is still a locomotion problem
        #
        # The paper drives the phase from eps_ref alone, which decays on a timer.
        # That is fine for a competent policy, because r_dw keeps the ACTUAL
        # distance near the reference. Early in training it is actively harmful:
        # measured, the reference had decayed to 0.04 while the target was still
        # 1.89 m away, so the policy was told it was in the manipulation phase
        # and paid to reach for something two metres off -- which is precisely
        # the lunge-and-topple failure this whole task exists to remove.
        #
        # Taking the max of the two keeps the paper's smooth schedule whenever
        # the robot is keeping up, and falls back on the truth when it is not.
        eps_ref = cmd.reference_distance()          # eq. (11), still paces the approach via r_dw

        # THE PHASE COMES FROM THE ARM'S WORKSPACE, measured from its MOUNT.
        #
        # The paper derives the phase from the HAND's distance to the target, with
        # its midpoint at 2 x arm span (1.4 m here). Two problems with that on our
        # robot, both visible in the videos:
        #   * the policy moves the hand itself, so it can shift its own phase by
        #     stretching the arm -- a hidden incentive to reach at full extension;
        #   * 1.4 m is twice what the Z1 can reach, so manipulation rewards took
        #     over while the target was still out of reach.
        # Instead: a sphere of radius reach_frac * arm_reach centred on the
        # manipulator's mount, which only moves when the BODY moves. Outside it,
        # drive; inside it, reach. At reach_frac = 0.5 the arm never has to work
        # past half its reach, which leaves it room to match orientation too.
        # Kept smooth (a narrow sigmoid, not an if/else): the paper's ablation
        # shows a hard switch costs 99% -> 82% success and makes motion jittery.
        mount = d.body_pos_w[:, self.mount_idx]
        d_mount = torch.norm(cmd.pose_command_w[:, :3] - mount, dim=1)
        # A WIDE blend, from full reach down to half reach. The first version
        # switched in a narrow band around 0.35 m, so the arm stayed idle until
        # the very end of the approach -- measured: the arm moved less, and mid-
        # height targets fell to 4/12. The paper blends early and gradually,
        # which is what lets the arm start preparing while the robot is still
        # driving. This keeps the sphere's two wins (measured from the shoulder,
        # from the real state rather than a clock) and gets that overlap back:
        # D ~ 0.95 at full reach, 0.5 halfway, ~0.05 at half reach.
        blend_start = self.blend_start_frac * self.arm_reach
        switch = self.reach_frac * self.arm_reach          # where the blend completes
        mid = 0.5 * (blend_start + switch)
        width = 5.0 * (blend_start - mid) / math.log(19.0)
        if self.phase_mode == "paper":
            dphase = phase(eps_ref, 2.0 * self.arm_reach, 2.0 * self.arm_reach)
        else:
            dphase = phase(d_mount, mid, width)

        # --- manipulation rewards, combined by PRIORITIZATION (eqs. 3-5, 7)
        r_ep = torch.exp(-d_p / self.sigma_p)
        r_eo = torch.exp(-d_theta / self.sigma_o)
        r_ep_m = r_ep + r_ep.pow(self.micro_m)      # micro-enhancement, eq. (7)
        r_eo_m = r_eo + r_eo.pow(self.micro_m)

        # Smoothness, a FLOOR under the crouch, and stillness of the base.
        #
        # Three things, each from watching the robot rather than the numbers:
        #
        #  * posture is still not regularised upward -- the crouch and lean that
        #    buy reach must stay free -- but squatting to 0.217 m is not reach,
        #    it is the cheapest way to drop the hand, and it looks wrong. Only
        #    height BELOW the floor is penalised; anything above costs nothing.
        #
        #  * the base was doing the fine positioning: 0.99 rad of yaw travel
        #    while serving targets well inside arm's reach, with the arm nearly
        #    still. Penalising planar and yaw motion once the phase is
        #    manipulation makes holding position the cheap option, so the ARM has
        #    to cover the last few centimetres -- which is the whole point of
        #    having one.
        too_low = torch.clamp(self.height_floor - (d.root_pos_w[:, 2] - env.scene.env_origins[:, 2]), min=0.0)
        base_shuffle = (torch.sum(torch.square(d.root_lin_vel_b[:, :2]), dim=1)
                        + torch.square(d.root_ang_vel_b[:, 2]))
        # TILT beyond ~15 deg. Measured on the 19,300-iteration policy: pitch up to
        # +44 deg, nose up and rear down, and it did this even for HIGH targets --
        # so it was a habit, not a reach strategy. Posture was otherwise free in
        # this phase (only the 0.30 m floor), and the policy settled into a
        # stance that looks broken. A modest lean stays free; past it, the
        # penalty multiplies straight into the tracking reward via reg_mani, so
        # a sat-back robot earns nothing for reaching.
        tilt = torch.acos(torch.clamp(-d.projected_gravity_b[:, 2], -1.0, 1.0))
        tilt_excess = torch.clamp(tilt - self.tilt_free, min=0.0)
        # A CALM HOLD. Watching the policy: once on the target the arm shook
        # and never settled. Micro-enhancement makes the reward gradient
        # steepest exactly there, so the policy keeps chasing sub-millimetre
        # error. Arm joint speed now enters the stability gate, scaled by r_ep so
        # it only bites when the hand is ALREADY close -- reaching still moves
        # the arm freely -- and a jittering arm earns less for being on target.
        arm_speed2 = torch.sum(torch.square(d.joint_vel[:, self.arm_joint_ids]), dim=1)
        reg_mani = torch.exp(
            -(torch.sum(torch.square(d.root_ang_vel_b[:, :2]), dim=1)
              + torch.square(d.root_lin_vel_b[:, 2])
              + self.w_height_floor * torch.square(too_low)
              + self.w_base_still * base_shuffle
              + self.w_tilt * torch.square(tilt_excess)
              + self.w_arm_still * r_ep * arm_speed2) / self.sigma_reg
        )

        # cumulative (integral) error, eq. (8): grows while the target is missed,
        # so a policy parked in a local minimum sees its value decay
        kappa = (1.0 - dphase) * self.kappa_scale
        self.e_cb = torch.clamp(self.e_cb + kappa * eps, max=self.cb_clip)

        # potential-based progress: positive when this step closed the gap
        progress = torch.where(self.has_prev, self.prev_eps - eps, torch.zeros_like(eps))
        self.prev_eps = eps.detach().clone()
        self.has_prev |= True

        # all wheels on the ground, a cheap proxy for "planted while working"
        # A STILL HAND, as a permission rather than a penalty (rfm15).
        #
        # The first attempt was a separate penalty on hand speed, counted only
        # near the target, and the policy found the cheap way out within 900
        # iterations: it stopped going near the target. Measured, the hand
        # hovered ~25 cm short of every mid and high target -- outside the gate
        # -- still moving at 50 cm/s. Any penalty switched on by proximity is a
        # reason to keep away.
        # As a factor on the tracking BONUS it cannot be: being on target and
        # shaking still earns at least as much as being away (the bonus is never
        # negative), and being on target and still earns the most. Same
        # prioritisation logic as the rest of RFM -- tracking pays in full only
        # once the hand is calm.
        if self.sigma_hand_v is not None:
            hv2 = torch.sum(torch.square(d.body_lin_vel_w[:, body_idx]), dim=1)
            hw2 = torch.sum(torch.square(d.body_ang_vel_w[:, body_idx]), dim=1)
            s_still = torch.exp(-(hv2 + self.w_hand_ang * hw2) / self.sigma_hand_v)
        else:
            s_still = torch.ones_like(r_ep)
        r_mani = reg_mani + reg_mani * s_still * (r_ep_m + r_ep * r_eo_m) - self.e_cb

        # --- locomotion rewards: here posture IS regularised (level, nominal height)
        flat = torch.sum(torch.square(d.projected_gravity_b[:, :2]), dim=1)
        # HEIGHT: bring the SHOULDER to the target's height, not the body to a
        # fixed nominal one.
        #
        # With the phase decided by the workspace sphere around the shoulder,
        # a fixed nominal height (0.55 m) deadlocked on anything below the
        # shoulder. Measured: for a 0.45 m target the robot stood at 0.61 m, the
        # shoulder sat 0.35 m above the target -- vertically outside the sphere --
        # so the phase stayed "locomotion", and the locomotion reward then
        # PUNISHED the crouch that was the only way in. 0/12 reached.
        # Locomotion is now "move the sphere onto the target" in all three
        # dimensions: the wheels close x/y, the legs close z. The desired height
        # is clamped to what the body can actually do.
        base_h = d.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
        shoulder_above_base = d.body_pos_w[:, self.mount_idx, 2] - d.root_pos_w[:, 2]
        target_h = cmd.pose_command_w[:, 2] - env.scene.env_origins[:, 2]
        desired_h = torch.clamp(target_h - shoulder_above_base, self.height_floor, self.height_ceiling)
        height = torch.square(base_h - desired_h)
        reg_loco = torch.exp(-(flat + self.w_height_track * height) / self.sigma_reg)

        # follow the RETREATING reference rather than sprinting at the target:
        # eq. (13). This is what makes the approach smooth instead of a lunge.
        err_ref = torch.clamp((eps_ref - eps).abs() - self.gamma, min=0.0)
        r_dw = torch.exp(-err_ref / self.sigma_dw)

        # CLOSING SPEED -- our addition, and the one that makes driving worth
        # doing at all.
        #
        # r_dw and the progress term both reward closing the gap, but neither is
        # dense enough to beat the baseline. Measured at iteration 600: standing
        # level pays reg_loco ~ 1.0 EVERY step, so doing nothing for ten seconds
        # earns ~500 while driving three metres earns ~60 of progress shaping.
        # The policy correctly concluded that standing still was the better deal
        # -- 0/12 targets approached, though also 0/12 falls.
        #
        # This rewards the component of base velocity pointing AT the target,
        # normalised by the reference speed, so it pays from the first step in
        # the right direction instead of only after the distance has closed.
        to_target = cmd.pose_command_w[:, :2] - d.root_pos_w[:, :2]
        gap = torch.norm(to_target, dim=1, keepdim=True).clamp(min=1e-6)
        closing = torch.sum(d.root_lin_vel_w[:, :2] * (to_target / gap), dim=1)
        # SIGNED, and that detail is the whole thing. Clamped at min=0 the reward
        # pays for approaching and charges nothing for drifting away, so the
        # cheapest policy is to cycle in and out forever, collecting the approach
        # reward on every inbound leg and never arriving. Measured exactly that:
        # r_appr 0.84 while the distance oscillated 1.3 -> 1.9 -> 1.5 m and the
        # closest approach never beat the starting distance.
        v_ref = cmd.reference_speed(gap.squeeze(-1)) if hasattr(cmd, "reference_speed") \
            else torch.full_like(closing, cmd.cfg.approach_speed)
        r_approach = torch.clamp(closing / v_ref, min=-1.0, max=1.0)
        # ... but only while there is somewhere to go: within arm's reach the
        # robot should stop and reach, not keep rolling at the target.
        # (no geometric gate here any more: r_approach lives inside r_loco, which
        # is weighted by D, so it fades as the phase moves to manipulation in
        # EITHER mode -- keeping the comparison about the phase alone)

        # FACE THE TARGET. Closing speed was measured in the world frame, so
        # reversing onto a target paid exactly as much as driving at it -- and
        # for a target behind, reversing also skips the turn this robot is bad
        # at. But the arm is mounted at the FRONT: arriving backwards leaves the
        # target behind the body, to be reached over the back or turned to at
        # the last moment. So approaching only pays in proportion to facing the
        # target (receding still costs in full), and facing it pays on its own,
        # which is the signal that makes turning round worth doing.
        c_h, s_h = torch.cos(d.heading_w), torch.sin(d.heading_w)
        fwd_err = c_h * to_target[:, 0] + s_h * to_target[:, 1]
        lat_err = -s_h * to_target[:, 0] + c_h * to_target[:, 1]
        facing = torch.cos(torch.atan2(lat_err, fwd_err))          # +1 facing, -1 back to it
        r_approach = torch.where(r_approach > 0, r_approach * facing.clamp(min=0.0), r_approach)
        r_face = 0.5 * (1.0 + facing)

        # keep the arm stowed while travelling
        r_sa = torch.sum(
            torch.abs(d.joint_pos[:, self.arm_joint_ids] - self.arm_stow), dim=1
        )
        # NO IDLE VALUE in the locomotion group.
        #
        # The paper's locomotion group is r_reg + r_reg * r_dw: it pays for being
        # stable whether or not the robot is going anywhere. With the phase decided
        # by the workspace sphere, that created a VALLEY. Measured on the sphere
        # policy parked 0.67 m from every target (phase ~0.90):
        #   parked, locomotion-dominated          1.44 per step
        #   moving in, hand not yet on target    ~0.85 (manipulation's idle value)
        #   hand on target                        up to ~4
        # Success pays more, but the path there goes DOWN first, and PPO does not
        # cross a valley from a settled optimum: 0/12 reached, 0/84 on the map.
        # The paper never meets this because its phase is ~0 throughout that
        # region. So locomotion pays only for PROGRESS -- following the schedule
        # and closing on the target -- and an idle robot earns roughly what an
        # idle manipulator does. Applied identically in both phase modes.
        r_loco = reg_loco * (r_dw + r_approach) - self.w_static_arm * r_sa

        # --- fusion, eq. (10), plus one addition of our own
        #
        # Closing the distance is shaped OUTSIDE the phase blend. In the paper
        # the potential-based term sits inside r_mani, so it is scaled by (1 - D)
        # and all but vanishes exactly when the target is far -- which is when
        # the robot most needs to be told that approaching is the right idea.
        # Since this is potential-based (Phi = -eps), adding it globally cannot
        # change the optimal policy, only how quickly it is found.
        # FACING, as progress rather than as a state.
        #
        # The first version paid ~1 per step simply for FACING the target. Inside
        # the locomotion group that became a place to camp: measured at 29k
        # iterations, the sphere policy drove forwards (reverse driving 93% ->
        # 5%) and then parked 0.73 m from every target -- just outside reach,
        # where the phase is still ~95% locomotion -- earning ~2.6 per step for
        # standing still, facing it. Crossing into manipulation could only lose
        # that. 0/12 reached.
        # Locomotion rewards must pay for PROGRESS, never for idling. This pays
        # for turning TOWARD the target and nothing for holding a heading, so it
        # still teaches turning round and has no value to sit on. Potential-
        # based, so it cannot change the optimal policy, only how fast it is
        # found; and outside the blend, so it is identical in both phase modes.
        face_prog = torch.where(self.has_prev_face, facing - self.prev_facing, torch.zeros_like(facing))
        self.prev_facing = facing.detach().clone()
        self.has_prev_face |= True
        fused = ((1.0 - dphase) * r_mani + dphase * r_loco
                 + self.w_progress * progress + self.w_face_prog * face_prog)

        # Keep the pieces around so they can be inspected without recomputing
        # them (probe_rfm.py). A fused reward is opaque by construction --
        # one number hides six -- so this is the only way to see which part is
        # actually driving the policy.
        self.last = {
            "eps": eps, "eps_ref": eps_ref, "D": dphase, "d_mount": d_mount, "tilt": tilt,
            "base_h": base_h, "desired_h": desired_h, "facing": facing,
            "d_p": d_p, "d_theta": d_theta,
            "r_ep": r_ep, "r_eo": r_eo, "reg_mani": reg_mani, "s_still": s_still, "reg_loco": reg_loco,
            "r_dw": r_dw, "r_appr": r_approach, "r_sa": r_sa, "e_cb": self.e_cb, "progress": progress,
            "r_mani": r_mani, "r_loco": r_loco, "fused": fused,
        }
        return fused


def target_radius_curriculum(
    env,
    env_ids,
    command_name: str = "ee_pose",
    start: float = 1.0,
    end: float = 3.0,
    step: float = 0.2,
    success_error: float = 0.20,
    success_frac: float = 0.5,
) -> torch.Tensor:
    """Grow how far away targets are placed, as the policy earns it.

    The paper says in one line that it "adopt[s] the idea of Curriculum Learning
    and Decaying Entropy Coefficient", and skipping that turned out to matter:
    trained directly on targets 0.3-3.0 m away, our policy learned the stable
    half of the task (0/12 falls, tilt 8 deg) and never learned the approach at
    all (0/12 arrivals, median closest 126 cm). Standing still simply pays better
    than a three-metre drive the policy has not yet discovered.

    Starting at arm's length makes the first success reachable -- the robot
    already knows how to reach -- and each widening is a small extrapolation from
    something that already works.

    Widens by `step` whenever at least `success_frac` of the environments ending
    an episode finished within `success_error` of their target.
    """
    cmd = env.command_manager.get_term(command_name)
    lo, hi = cmd.cfg.radial_range
    err = cmd.metrics["position_error"][env_ids]
    if err.numel() > 0:
        frac = (err < success_error).float().mean()
        if float(frac) > success_frac and hi < end:
            cmd.cfg.radial_range = (lo, min(end, hi + step))
    return torch.tensor(cmd.cfg.radial_range[1], device=env.device)



class ActionSmoothness(ManagerTermBase):
    """|a_{t-2} - 2 a_{t-1} + a_t|^2 -- the paper's "action smoothness" (Table II).

    I implemented only the first-difference action-rate term at first; this
    second-difference one penalises reversals rather than changes, which is
    precisely what a shaking arm is doing.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        n = env.action_manager.total_action_dim
        self.prev1 = torch.zeros(env.num_envs, n, device=env.device)
        self.prev2 = torch.zeros(env.num_envs, n, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.prev1[env_ids] = 0.0
        self.prev2[env_ids] = 0.0
        return {}

    def __call__(self, env) -> torch.Tensor:
        a = env.action_manager.action
        out = torch.sum(torch.square(self.prev2 - 2.0 * self.prev1 + a), dim=1)
        self.prev2 = self.prev1.clone()
        self.prev1 = a.clone()
        return out


# ---------------------------------------------------------------------------
# A CALM ARM (rfm15). Measured with evaluation/shake_diag.py on the rfm14 policies:
# on target, the hand wobbles ~15 mm RMS at ~1 Hz, and the arm's ACTIONS change
# by ~3.0 (RMS, action units) from one 20 ms step to the next -- 1.5 rad of joint
# target, every step. The policy is driving the arm bang-bang: its targets flick
# between extremes and only their average positions the hand, smoothed by the
# soft arm PD (kp 100, kd 4). The learned arm action std had grown to ~2.0 from
# 1.0, because nothing in the reward noticed the flicking (the 22-wide action
# rate at -0.003 charged ~0.03 per step for it) while the entropy bonus paid for
# more noise. A play-time low-pass filter only halved the wobble, so this is
# fixed where it was learned. Three terms, each aimed at the arm alone so the
# legs and wheels keep the freedom locomotion needs.

class ArmActionRate(ManagerTermBase):
    """First and second difference of the ARM's actions only.

        sum (a_t - a_{t-1})^2  +  w2 * sum (a_t - 2 a_{t-1} + a_{t-2})^2

    A purposeful 1 rad/s arm motion changes the action by ~0.04 per step and
    costs nothing measurable; the bang-bang hold (~3.0 per step) costs ~9.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        name = cfg.params.get("term_name", "arm_action")
        self.term = env.action_manager.get_term(name)
        self.use_processed = cfg.params.get("use_processed", False)
        off = 0
        for n, term in zip(env.action_manager.active_terms, env.action_manager._terms.values()):
            if n == name:
                self.sl = slice(off, off + term.action_dim)
            off += term.action_dim
        k = self.sl.stop - self.sl.start
        self.prev1 = torch.zeros(env.num_envs, k, device=env.device)
        self.prev2 = torch.zeros(env.num_envs, k, device=env.device)
        self.fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.fresh[env_ids] = True
        return {}

    def __call__(self, env, term_name: str = "arm_action", w2: float = 0.5,
                 use_processed: bool = False) -> torch.Tensor:
        # use_processed: charge the joint TARGETS that reach the motors (after
        # FilteredJointPositionAction's low-pass), in action units, instead of the
        # raw policy output. Charging the raw output taxes exploration noise the
        # filter removes anyway: rfm16 collapsed the arm's std to 0.15-0.20 within
        # 400 iterations (legs 0.25-0.33, wheels 0.8), choking arm exploration.
        if self.use_processed:
            a = self.term.processed_actions / self.term.cfg.scale
        else:
            a = env.action_manager.action[:, self.sl]
        # the first step after a reset has no history: don't charge the jump
        # from the previous episode's last action
        self.prev1 = torch.where(self.fresh.unsqueeze(1), a, self.prev1)
        self.prev2 = torch.where(self.fresh.unsqueeze(1), a, self.prev2)
        d1 = torch.sum(torch.square(a - self.prev1), dim=1)
        d2 = torch.sum(torch.square(a - 2.0 * self.prev1 + self.prev2), dim=1)
        self.prev2 = self.prev1.clone()
        self.prev1 = a.clone()
        self.fresh[:] = False
        return d1 + w2 * d2


class ArmTorqueRate(ManagerTermBase):
    """The paper's "relative torques rate", sum |tau_t - tau_{t-1}| / tau_limit
    (Jiang et al., Table II, weight -0.1) -- the one r_basic term I had not
    implemented. Restricted to the arm: on the legs, torque swings ARE walking.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.ids, _ = self.robot.find_joints(cfg.params.get("joints", ["joint[1-6]"]), preserve_order=True)
        lim = self.robot.data.joint_effort_limits[0, self.ids].clone()
        self.limit = torch.where(torch.isfinite(lim) & (lim > 0), lim, torch.full_like(lim, 30.0))
        self.prev = torch.zeros(env.num_envs, len(self.ids), device=env.device)
        self.fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.fresh[env_ids] = True
        return {}

    def __call__(self, env, asset_cfg: SceneEntityCfg, joints: list[str] | None = None) -> torch.Tensor:
        tau = self.robot.data.applied_torque[:, self.ids]
        self.prev = torch.where(self.fresh.unsqueeze(1), tau, self.prev)
        out = torch.sum(torch.abs(tau - self.prev) / self.limit, dim=1)
        self.prev = tau.clone()
        self.fresh[:] = False
        return out


def hand_still_on_target(
    env, command_name: str = "ee_pose", radius: float = 0.05, w_ang: float = 0.05
) -> torch.Tensor:
    """Hand speed, counted only near the target. DO NOT USE -- kept as the record
    of a failure: within 900 iterations the policy learned to hover ~25 cm short
    of the target, outside the gate, where shaking is free. Superseded by
    RewardFusion's sigma_hand_v, which scales the tracking bonus instead.

    What the user actually SEES:

        exp(-d_p / radius) * (|v_hand|^2 + w_ang |omega_hand|^2)

    Joint-speed stillness (w_arm_still, inside the stability gate) saturates --
    exp(-x) stops caring once x is large -- and ignores the body: a hand riding
    on a swaying base is still a shaking hand. This is linear in the motion and
    measured at the hand, whatever moves it. The gate keeps it out of the way
    of the approach: at 30 cm it is down to 0.2%.
    """
    cmd = env.command_manager.get_term(command_name)
    robot = cmd.robot
    i = cmd.body_idx
    d_p = torch.norm(cmd.pose_command_w[:, :3] - robot.data.body_pos_w[:, i], dim=1)
    v2 = torch.sum(torch.square(robot.data.body_lin_vel_w[:, i]), dim=1)
    w2 = torch.sum(torch.square(robot.data.body_ang_vel_w[:, i]), dim=1)
    return torch.exp(-d_p / radius) * (v2 + w_ang * w2)


# ---------------------------------------------------------------------------
# A LOW-PASS FILTER IN THE ARM'S ACTUATION (rfm15d).
#
# Part of the controller, trained with it and deployed with it -- the common
# practice on real legged robots -- not a play-time patch. Measured on rfm14
# before any retraining: a play-time EMA of 0.2 on the arm alone halved the hand
# wobble (15 -> 9 mm RMS) and cut the step-to-step action jump from 2.9 to 0.75.
# Inside training the policy can no longer drive the arm bang-bang -- a command
# that flips every step is averaged away before it reaches the motor -- so the
# fine-tune starts from a calmer arm and learns to track THROUGH the filter
# instead of having its old strategy penalised away.
#
#   target_t = alpha * (scale * a_t + offset) + (1 - alpha) * target_{t-1}
#
# alpha 0.25 at 50 Hz is a ~80 ms time constant: invisible on a reach that takes
# a second, decisive against a 25 Hz flip.
from isaaclab.envs.mdp.actions import JointPositionActionCfg as _JPACfg  # noqa: E402
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction as _JPA  # noqa: E402
from isaaclab.utils import configclass as _configclass  # noqa: E402


class FilteredJointPositionAction(_JPA):
    cfg: "FilteredJointPositionActionCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._filtered = self._offset_tensor().clone()

    def _offset_tensor(self) -> torch.Tensor:
        if isinstance(self._offset, torch.Tensor):
            return self._offset
        return torch.full_like(self._raw_actions, float(self._offset))

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        a = self.cfg.alpha
        self._filtered = a * self._processed_actions + (1.0 - a) * self._filtered
        self._processed_actions = self._filtered.clone()

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        # start from where the joints actually are, not the stow pose, so a reset
        # mid-reach does not yank the arm
        self._filtered[env_ids] = self._asset.data.joint_pos[env_ids][:, self._joint_ids]


@_configclass
class FilteredJointPositionActionCfg(_JPACfg):
    class_type: type = FilteredJointPositionAction
    alpha: float = 0.25
    """EMA weight of the NEW target per policy step (1.0 = no filter)."""


# ---------------------------------------------------------------------------
# INTEGRAL ACTION FOR THE HAND (rfm17).
#
# With the bang-bang hold trained out (rfm15d), the arm became still -- 11/16
# shake_diag cases under 3 mm -- but stopped 4-11 cm short where rfm14 got
# within 2-4 cm. The reason is the domain randomisation: every link but the base
# has its mass scaled 0.7-1.3 at startup (the arm included), the base is pushed,
# and the policy has no memory, so a smooth controller can only be PROPORTIONAL
# and a proportional controller leaves a steady sag error under an unknown load.
# Bang-bang had hidden this: it saturates the torque, so the load hardly matters.
#
# The textbook cure is an integral term, and the paper reaches for the same idea
# twice: its cumulative error e_cb (eq. 8, "emulate the integral portion of a PID
# controller") and a long-history encoder that estimates hidden properties such
# as mass. This hands the policy the integral directly: the hand's pose error in
# the base frame, accumulated while the hand is near the target and leaking away
# otherwise (anti-windup), clipped. A persistent 3 cm error grows this signal
# until the policy cancels it; a calm, on-target hand leaves it constant.
from isaaclab.utils.math import axis_angle_from_quat, quat_apply_inverse, quat_inv, quat_mul  # noqa: E402


class HandErrorIntegral(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.integ = torch.zeros(env.num_envs, 6, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.integ[env_ids] = 0.0
        return {}

    def __call__(self, env, command_name: str = "ee_pose", tau_leak: float = 5.0, gate: float = 0.3,
                 limit: float = 0.2, rot_scale: float = 0.3, gain: float = 5.0) -> torch.Tensor:
        cmd = env.command_manager.get_term(command_name)
        d = cmd.robot.data
        i = cmd.body_idx
        dt = env.step_dt
        fresh = cmd.age <= dt * 1.5                    # a new target owes nothing to the old one
        self.integ[fresh] = 0.0
        pos_w = cmd.pose_command_w[:, :3] - d.body_pos_w[:, i]
        rot_w = axis_angle_from_quat(quat_mul(cmd.pose_command_w[:, 3:], quat_inv(d.body_quat_w[:, i])))
        err = torch.cat([quat_apply_inverse(d.root_quat_w, pos_w),
                         rot_scale * quat_apply_inverse(d.root_quat_w, rot_w)], dim=1)
        near = (torch.norm(pos_w, dim=1) < gate).unsqueeze(1)
        grown = self.integ * (1.0 - dt / tau_leak) + dt * err
        drained = self.integ * (1.0 - dt / 0.5)        # away from the target: forget within ~0.5 s
        self.integ = torch.clamp(torch.where(near, grown, drained), -limit, limit)
        return gain * self.integ


# ---------------------------------------------------------------------------
# INCREMENTAL ARM COMMANDS (rfm19).
#
# Every fix that kept the arm's ABSOLUTE joint-target action -- penalties
# (rfm15/17), a low-pass filter, fixed gains and masses (rfm18 A/B/F/G) --
# either left the relay in place (12-22 mm wobble, raw command jumps of 1.6-2.9
# per step) or calmed the arm by giving up precision (4-6 cm, ori ~1 rad). With
# absolute targets, flicking between two of them is a perfectly usable controller,
# and nothing short of a large penalty makes it worse than the alternative.
#
# Here the policy commands a CHANGE of the target instead:
#
#   target_t = clamp(target_{t-1} + delta_scale * a_t)
#
#   * holding still is a = 0 -- the natural resting output, not a fine balance;
#   * a flick +a, -a cancels to nothing, so a relay has nothing left to gain;
#   * the target keeps moving while an error remains, whatever the motor gains
#     or link masses: integral action for free, which is what the smooth
#     absolute-target policies lacked to reach full precision.
# The target is kept within `window` of the measured joint position (anti-windup:
# a blocked arm does not wind its target off to the joint limit) and inside the
# soft joint limits.
class DeltaJointPositionAction(_JPA):
    cfg: "DeltaJointPositionActionCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._target = self._asset.data.joint_pos[:, self._joint_ids].clone()

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        a = torch.clamp(actions, -self.cfg.action_clip, self.cfg.action_clip)
        q = self._asset.data.joint_pos[:, self._joint_ids]
        lim = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        t = self._target + self.cfg.delta_scale * a
        t = torch.max(torch.min(t, q + self.cfg.window), q - self.cfg.window)
        self._target = torch.max(torch.min(t, lim[..., 1]), lim[..., 0])
        self._processed_actions = self._target.clone()

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._target[env_ids] = self._asset.data.joint_pos[env_ids][:, self._joint_ids]


@_configclass
class DeltaJointPositionActionCfg(_JPACfg):
    class_type: type = DeltaJointPositionAction
    delta_scale: float = 0.05
    """Target change per policy step at |a| = 1, rad (0.05 at 50 Hz = 2.5 rad/s)."""
    window: float = 0.4
    """How far the target may run ahead of the measured joint, rad."""
    action_clip: float = 2.0


def arm_action_l2(env, term_name: str = "arm_action") -> torch.Tensor:
    """sum a^2 over the arm's incremental commands: moving costs a little, so the
    resting output on target is exactly zero."""
    return torch.sum(torch.square(env.action_manager.get_term(term_name).raw_actions), dim=1)
