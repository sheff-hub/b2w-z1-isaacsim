# Local addition (not upstream robot_lab).
#
# Keep the arm out of the robot's own structure.
#
# WHY THIS IS NOT JUST enabled_self_collisions=True
# =================================================
# That was tried first and it does not work on this asset. PhysX only filters
# collider pairs that are joined BY A JOINT, and this robot has two pairs that
# overlap by construction while being two joints apart, so switching
# self-collisions on jams them permanently (measured with probe_pairs.py):
#
#     head_Link  <-> lidar_link    24.6 kN   both bolted to the base, overlapping
#     link00     <-> base_link     19.2 kN   the arm's base is sunk into the body
#
# Those forces do not decay: the joint stops moving, the arm never reaches, and
# training learns nothing (hand error sat at 0.55 m where wb9 was at 0.04 m by
# the same iteration). The pairs we actually care about -- the arm's moving links
# against the lidar mast -- read 0.0 N at rest, correctly.
#
# Fixing that properly means editing collision geometry in the asset, which is
# follow-up work. This term gets the BEHAVIOUR we want meanwhile: a penalty on
# the arm entering the volume the body occupies, so the policy routes around
# instead of through. It changes no physics, so it cannot destabilise anything.
#
# THE BOXES ARE MEASURED, NOT GUESSED
# ===================================
# Both come from the collision geometry in the USD, cross-checked against the
# contact forces above (evaluation/workspace.py, probe_contacts.py):
#
#   body  |x| <= 0.25, |y| <= 0.14, |z| <= 0.075   the base_link collision box
#   mast  x in [0.266, 0.418], |y| <= 0.077,
#         z in [0.00, 0.34]                        the lidar capsule, r 0.076
#
# The mast sits directly in front of the shoulder, which is why the arm met it
# constantly. Note the body box is only 0.50 x 0.28 x 0.15 m -- much smaller than
# the visible shell, because that is what the importer generated.

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

# (centre, half-extent) in the BASE frame, metres.
KEEPOUT_BOXES = (
    ((0.000, 0.0, 0.000), (0.250, 0.140, 0.075)),   # body
    ((0.342, 0.0, 0.170), (0.076, 0.077, 0.170)),   # lidar mast
)


def arm_body_clearance(
    env,
    asset_cfg: SceneEntityCfg,
    margin: float = 0.04,
) -> torch.Tensor:
    """Penalise arm links that enter the robot's own volume.

    For each monitored link and each keep-out box, the penetration depth is

        depth = min_axis( half_extent + margin - |offset_axis| )

    which is positive only when the point is inside the inflated box, and grows
    the further in it goes. Summing those depths gives a penalty with a gradient
    that always points out of the box by the shortest route -- so the policy is
    pushed around the obstacle rather than merely away from one face.

    Positions are taken in the BASE frame, so this is correct whatever the robot's
    orientation: the boxes are bolted to the body and travel with it.

    asset_cfg selects the links to monitor (the moving arm links; link00 and
    link01 are excluded by the caller since they are inside the mount by design).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w.unsqueeze(1)
    root_quat = asset.data.root_quat_w
    pts_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]

    # world -> base frame, for every monitored link at once
    n_links = pts_w.shape[1]
    rel = (pts_w - root_pos).reshape(-1, 3)
    quat = root_quat.repeat_interleave(n_links, dim=0)
    pts_b = quat_apply_inverse(quat, rel).reshape(-1, n_links, 3)

    penalty = torch.zeros(env.num_envs, device=asset.device)
    for centre, half in KEEPOUT_BOXES:
        c = torch.tensor(centre, device=asset.device)
        h = torch.tensor(half, device=asset.device) + margin
        depth = (h - (pts_b - c).abs()).min(dim=-1).values     # (envs, links)
        penalty += depth.clamp(min=0.0).sum(dim=1)
    return penalty
