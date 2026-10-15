# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
#
# Local addition (not upstream robot_lab): the flat B2W task with its ground
# plane replaced by NVIDIA's warehouse. Only the scene changes -- robot asset,
# observations, actions and rewards are exactly those of UnitreeB2WFlatEnvCfg,
# so a policy trained on the flat task plays here unmodified.

from isaaclab.utils import configclass

from .flat_env_cfg import UnitreeB2WFlatEnvCfg

# Same asset the Isaac Sim stages in robots_usd pull in. Remote: the
# first load downloads it, later loads come from the Kit cache.
WAREHOUSE_USD = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
    "/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd"
)


@configclass
class UnitreeB2WWarehouseEnvCfg(UnitreeB2WFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # TerrainImporterCfg supports "generator" | "plane" | "usd". With "usd" the
        # file is referenced at /World/ground and its own colliders are used.
        # Note the configured physics_material is NOT applied for "usd" terrain
        # (terrain_importer_cfg.py:97), so wheel grip comes from the warehouse
        # floor's own material rather than the task's friction settings.
        self.scene.terrain.terrain_type = "usd"
        self.scene.terrain.usd_path = WAREHOUSE_USD

        # One robot. Environment origins are laid out on an env_spacing grid over
        # the single shared warehouse, so extra robots could spawn inside shelves.
        # (0, 0) is clear floor -- the wheel-only teleop test drove from there.
        self.scene.num_envs = 1

        # terrain_out_of_bounds (isaaclab_tasks .../velocity/mdp/terminations.py)
        # accepts only "plane" or "generator" and raises on "usd". On a plane it
        # returns False every step, so removing it reproduces the flat task's
        # behaviour exactly; the warehouse walls bound the robot anyway.
        self.terminations.terrain_out_of_bounds = None

        # Start upright, still, at the origin, facing +X. For recovery training the
        # B2W config (rough_env_cfg.py:109) resets to ANY orientation -- roll, pitch
        # and yaw over the full +/-3.14 -- offset up to 0.5 m, dropped from up to
        # 0.2 m, with random velocity. In an open field the policy just rights
        # itself; here it tumbles into the racks first. play.py already removes
        # the pushes and external forces but leaves this reset pose alone.
        # (0, 0) facing +X is clear floor: the wheel-only teleop test drove 2.6 m
        # forward from exactly this pose.
        still = (0.0, 0.0)
        self.events.randomize_reset_base.params = {
            "pose_range": {k: still for k in ("x", "y", "z", "roll", "pitch", "yaw")},
            "velocity_range": {k: still for k in ("x", "y", "z", "roll", "pitch", "yaw")},
        }

        # flat_env_cfg only prunes zero-weight rewards when the class name is
        # exactly "UnitreeB2WFlatEnvCfg"; a subclass would silently skip it.
        self.disable_zero_weight_rewards()
