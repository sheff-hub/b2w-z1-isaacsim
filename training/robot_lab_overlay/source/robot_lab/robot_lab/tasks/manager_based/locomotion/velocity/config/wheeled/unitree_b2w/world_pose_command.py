# Local addition (not upstream robot_lab).
#
# A hand-target command that stays put in the WORLD instead of riding along with
# the robot.
#
# Isaac Lab's UniformPoseCommand samples the target in the BASE frame and
# recomputes its world position from the current root pose on every step
# (pose_command.py:_update_metrics), so the target is rigidly bolted to the body:
# if the robot crouches, the target crouches with it. Under that command a
# whole-body policy can never gain reach by moving its body, so "does the dog
# help the arm?" is unanswerable by construction.
#
# Here the target is sampled ONCE in the world and held there:
#   * x, y are sampled in the robot's heading frame at sampling time, so the
#     target always appears somewhere sensible in front of it;
#   * z is an ABSOLUTE world height -- this is the part that makes the body
#     matter. With the shoulder around 0.80 m when standing and 0.69 m of arm,
#     a target at 0.15 m is at the very edge of reach unless the robot crouches.
#
# The policy still OBSERVES the target in its own body frame (recomputed each
# step), which is what makes the problem well posed and translation invariant --
# but now that vector changes when the body moves, which is exactly the coupling
# we want it to discover.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch

from isaaclab.envs.mdp.commands import UniformPoseCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    compute_pose_error,
    quat_apply,
    quat_from_euler_xyz,
    quat_unique,
    subtract_frame_transforms,
    yaw_quat,
)


class WorldPoseCommand(UniformPoseCommand):
    """Pose command fixed in the world, reported to the policy in the base frame."""

    cfg: WorldPoseCommandCfg

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # SE(3) distance when the current target was drawn, and how long ago
        self.eps0 = torch.zeros(self.num_envs, device=self.device)
        self.age = torch.zeros(self.num_envs, device=self.device)
        # The reference distance as a STATE, integrated each step under a speed
        # profile, instead of a closed-form constant-speed ramp.
        self.eps_ref = torch.zeros(self.num_envs, device=self.device)

    def reference_distance(self) -> torch.Tensor:
        """eps_ref = max(eps0 - v * t, 0)  -- the paper's eq. (11).

        A reference that retreats at a set speed. The robot is rewarded for
        keeping its ACTUAL distance near this one, which is what turns "get
        there" into "get there at a sensible pace" and stops the lunge.
        """
        return self.eps_ref

    def reference_speed(self, dist: torch.Tensor) -> torch.Tensor:
        """Speed profile: fast when far, easing in as the target nears.

        v = clamp(gain * (dist - stop), v_min, v_max)

        The paper's reference retreats at ONE constant speed. We had set it to
        0.35 m/s, which after the fused reward made 0.35 m/s the optimal speed
        everywhere -- a 3 m approach took ~9 s where the earlier velocity-command
        policies drove at 1.37 m/s. A single fast speed would bring back the
        overshoot that made wb10 give up on world-fixed targets, so the pace
        now depends on how far there is to go -- the same law as the waypoint
        follower.
        """
        c = self.cfg
        return torch.clamp(c.approach_gain * (dist - c.approach_stop),
                           min=c.approach_speed_min, max=c.approach_speed_max)

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        root_pos = self.robot.data.root_pos_w[env_ids]
        root_quat = self.robot.data.root_quat_w[env_ids]

        # x, y in the robot's heading frame at sampling time -> world
        offset = torch.zeros(len(env_ids), 3, device=self.device)
        if self.cfg.radial_range is not None:
            # OMNI-DIRECTIONAL placement: a distance and a bearing, rather than a
            # box in front. The box quietly biases every target forward, so the
            # robot never has to turn around -- and turning to face a target is
            # most of the locomotion half of loco-manipulation.
            dist = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.radial_range)
            ang = torch.empty(len(env_ids), device=self.device).uniform_(-torch.pi, torch.pi)
            offset[:, 0] = dist * torch.cos(ang)
            offset[:, 1] = dist * torch.sin(ang)
        else:
            offset[:, 0] = r.uniform_(*self.cfg.ranges.pos_x)
            offset[:, 1] = r.uniform_(*self.cfg.ranges.pos_y)
        height = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.ranges.pos_z)

        # Never ask for a point inside the robot's own hardware.
        #
        # The target box reaches back to x = 0.25 and down to z = 0.15, and the
        # lidar mast occupies x 0.27-0.42, |y| < 0.08, z 0.00-0.34 in the base
        # frame -- so without this a slice of targets would land INSIDE the mast.
        # Those are unreachable however well the policy learns, and unreachable
        # targets are how wb10 learned to ignore the arm altogether. Offending
        # draws are simply redrawn.
        if self.cfg.keepout:
            for _ in range(self.cfg.keepout_tries):
                z_b = height - (root_pos[:, 2] - self._env.scene.env_origins[env_ids, 2])
                pts = torch.stack([offset[:, 0], offset[:, 1], z_b], dim=1)
                bad = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
                for centre, half in self.cfg.keepout:
                    c = torch.tensor(centre, device=self.device)
                    h = torch.tensor(half, device=self.device) + self.cfg.keepout_margin
                    bad |= ((pts - c).abs() <= h).all(dim=1)
                if not bool(bad.any()):
                    break
                n = int(bad.sum())
                offset[bad, 0] = torch.empty(n, device=self.device).uniform_(*self.cfg.ranges.pos_x)
                offset[bad, 1] = torch.empty(n, device=self.device).uniform_(*self.cfg.ranges.pos_y)
                height[bad] = torch.empty(n, device=self.device).uniform_(*self.cfg.ranges.pos_z)

        offset_w = quat_apply(yaw_quat(root_quat), offset)
        self.pose_command_w[env_ids, 0:2] = root_pos[:, 0:2] + offset_w[:, 0:2]

        # z is an absolute height above this environment's ground
        self.pose_command_w[env_ids, 2] = self._env.scene.env_origins[env_ids, 2] + height

        euler = torch.zeros(len(env_ids), 3, device=self.device)
        euler[:, 0].uniform_(*self.cfg.ranges.roll)
        euler[:, 1].uniform_(*self.cfg.ranges.pitch)
        euler[:, 2].uniform_(*self.cfg.ranges.yaw)
        quat = quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
        self.pose_command_w[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat

        # Snapshot the SE(3) distance at the moment of sampling. The Loco-Mani
        # Fusion reference retreats from THIS value at a fixed speed, so the
        # robot is asked to close the gap steadily instead of all at once.
        from .rfm import se3_distance

        eps0, _, _ = se3_distance(
            self.pose_command_w[env_ids, :3] - self.robot.data.body_pos_w[env_ids, self.body_idx],
            self.robot.data.body_quat_w[env_ids, self.body_idx],
            self.pose_command_w[env_ids, 3:],
        )
        self.eps0[env_ids] = eps0
        self.eps_ref[env_ids] = eps0
        self.age[env_ids] = 0.0

    def _update_command(self):
        """World target -> base frame, every step. The inverse of the parent's job.


        Optionally re-draws a target the robot has driven away from. A world-fixed
        target is unreachable once the robot has left it behind, and at 0.8 m/s
        that takes about a second -- which poisons training for any policy that is
        also supposed to drive. With max_distance set, a target further than that
        is replaced by a fresh one near the robot, so reaching stays a live task at
        any speed. Leave it None for the stationary study (wb9), where drifting
        away is not possible anyway.
        """
        self.age += self._env.step_dt
        self.eps_ref = torch.clamp(
            self.eps_ref - self.reference_speed(self.eps_ref) * self._env.step_dt, min=0.0
        )

        if self.cfg.max_distance is not None:
            too_far = (
                torch.norm(self.pose_command_w[:, :3] - self.robot.data.root_pos_w, dim=1)
                > self.cfg.max_distance
            )
            if bool(too_far.any()):
                self._resample_command(too_far.nonzero().flatten())

        self.pose_command_b[:, :3], self.pose_command_b[:, 3:] = subtract_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
        )

    def _update_metrics(self):
        """Error against the WORLD target.

        The parent recomputes pose_command_w from pose_command_b here, which would
        drag the target back onto the body and undo everything above.
        """
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.robot.data.body_pos_w[:, self.body_idx],
            self.robot.data.body_quat_w[:, self.body_idx],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)


@configclass
class WorldPoseCommandCfg(UniformPoseCommandCfg):
    """Ranges mean: pos_x/pos_y are offsets in the robot's heading frame at sampling
    time; pos_z is an ABSOLUTE height above the ground."""

    class_type: type = WorldPoseCommand
    asset_name: str = MISSING
    body_name: str = MISSING

    max_distance: float | None = None
    """Re-draw the target if the robot gets further than this from it, in metres.

    None keeps a target until its normal resampling time."""

    keepout: tuple = ()
    """Volumes in the BASE frame the target may not land in, as
    ((centre_xyz, half_extent_xyz), ...). Used to keep targets out of the robot's
    own body and lidar mast; see self_clearance.KEEPOUT_BOXES."""

    keepout_margin: float = 0.04
    """Inflation of the keep-out volumes, metres."""

    keepout_tries: int = 8
    """How many times an offending draw is redrawn before it is accepted."""

    radial_range: tuple[float, float] | None = None
    """(min, max) distance for OMNI-DIRECTIONAL sampling: the target is placed at
    a uniform bearing around the robot instead of inside the pos_x/pos_y box.
    None keeps the box."""

    approach_speed: float = 0.25
    """Legacy constant pace (m/s). Superseded by the speed profile below; kept so
    older configs still load."""

    approach_speed_max: float = 1.0
    """Cruise pace when far from the target, m/s. The B2W drove 1.37 m/s under
    the velocity-command policies, so 1.0 is inside what it has shown it can do."""

    approach_speed_min: float = 0.3
    """Pace when close, m/s."""

    approach_gain: float = 1.0
    """Deceleration: 1/s of pace per metre of remaining distance."""

    approach_stop: float = 0.35
    """Distance at which the pace bottoms out at approach_speed_min, m (the
    arm-workspace sphere's radius)."""

    phase_mu: float = 1.4
    """Midpoint of the locomotion/manipulation phase sigmoid, metres. The paper
    recommends 2 x arm span; the Z1 reaches 0.694 m."""

    phase_slope: float = 1.4
    """Slope l of that sigmoid; the paper recommends 2 x arm span as well."""
