"""Small helpers shared by the RFM demo scripts."""

import torch


def restart_schedule(cmd, robot):
    """Re-arm the approach schedule after a target is set by hand.

    The RFM policy is fed a reference distance that retreats from wherever the
    target was when it was drawn (eps_ref = max(eps0 - v t, 0)). A script that
    writes pose_command_w directly bypasses _resample_command, so eps0 and the
    age would still describe the PREVIOUS target and the phase variable would be
    nonsense. This recomputes them.
    """
    if not hasattr(cmd, "eps0"):
        return
    from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.unitree_b2w.rfm import (
        se3_distance,
    )

    with torch.inference_mode():
        eps0, _, _ = se3_distance(
            cmd.pose_command_w[:, :3] - robot.data.body_pos_w[:, cmd.body_idx],
            robot.data.body_quat_w[:, cmd.body_idx],
            cmd.pose_command_w[:, 3:],
        )
        cmd.eps0.copy_(eps0)
        if hasattr(cmd, "eps_ref"):
            cmd.eps_ref.copy_(eps0)
        cmd.age.zero_()

