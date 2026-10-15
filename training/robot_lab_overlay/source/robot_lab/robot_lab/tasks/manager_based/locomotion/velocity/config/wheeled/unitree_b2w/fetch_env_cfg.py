# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
#
# Local addition (not upstream robot_lab). Phase 5, step 1: the warehouse fetch
# scene -- the B2W + Z1-with-gripper drives to a table and picks up a box.
#
# Not a training task. Driven by project/fetch_box.py, which splits the job in two:
#
#   locomotion   model_2600.pt (the B2W flat policy: 57 inputs -> 12 legs +
#                4 wheels) tracks velocity commands from a scripted go-to-pose
#                controller. Its observation group is the cargo task's, i.e.
#                exactly what it was trained on.
#   manipulation the arm takes absolute joint targets from the script, which
#                does its own IK (no arm policy); the gripper a binary open/close.
#
# Scene: NVIDIA's warehouse (the aisle in front of the spawn rack is clear from
# x 0.4 to 9.5 m, y -2.8 to 2.8 m), a table built from primitives (top at 0.60 m
# -- the packing-table asset's 0.99 m top is above the arm's reach), and a small
# box on it. The box pose is read from the simulator: the "oracle" perception.

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import mdp as base_mdp
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from .arm_cargo_env_cfg import ARM_STOW, UNITREE_B2W_Z1_CFG, UnitreeB2WZ1FlatEnvCfg
from .loco_reach_env_cfg import DRIVE_ACTIONS, last_actions_of
from .warehouse_env_cfg import WAREHOUSE_USD

# Built by tools/build_b2w_z1_gripper_asset.py (container path).
B2W_Z1_GRIPPER_USD = os.environ.get("B2W_Z1_GRIPPER_USD", "/workspace/robots_usd/b2w_z1_gripper.usd")

# ------------------------------------------------------------------ the scene files
# The geometry lives in the repository's scenes/*.yaml, not here: scenes/warehouse_one_table.yaml for the fetch task,
# scenes/warehouse_three_tables.yaml for Fetch3. FETCH_SCENE=<file> swaps in another scene file of the same layout
# (one_table / multi_table), e.g. tables moved or a different environment USD, without touching this code.
SCENES_DIR = os.environ.get("SCENES_DIR", "/workspace/scenes")


def load_scene(default_name: str) -> dict:
    import yaml

    default = os.path.join(SCENES_DIR, f"{default_name}.yaml")
    with open(default) as fh:
        scene = yaml.safe_load(fh)
    override = os.environ.get("FETCH_SCENE", "")
    if override:
        path = override if os.path.isabs(override) or override.endswith(".yaml") else os.path.join(SCENES_DIR, f"{override}.yaml")
        if not os.path.isabs(path) and not os.path.exists(path):
            path = os.path.join(SCENES_DIR, os.path.basename(path))
        with open(path) as fh:
            other = yaml.safe_load(fh)
        if other.get("layout") == scene.get("layout"):
            scene = other
    return scene


SCENE_ONE = load_scene("warehouse_one_table")
SCENE_MULTI = load_scene("warehouse_three_tables")
# fetch_box.py's docking and grasp geometry uses one table and one box size for both scenes: refuse files that differ
for _k in ("size", "height", "top_thickness", "leg"):
    assert SCENE_ONE["table"][_k] == SCENE_MULTI["table"][_k], f"scene files differ in table.{_k}: keep them equal"
for _k in ("size", "mass", "from_edge"):
    assert SCENE_ONE["box"][_k] == SCENE_MULTI["box"][_k], f"scene files differ in box.{_k}: keep them equal"

# ------------------------------------------------------------------ geometry (from SCENE_ONE)
# Robot spawn, in the aisle, facing +x.
SPAWN_POS = tuple(SCENE_ONE["robot"]["spawn"])
# Table: long side along y, facing the robot's approach from -x.
TABLE_CENTER = tuple(next(iter(SCENE_ONE["tables"].values()))["center"])
TABLE_SIZE = tuple(SCENE_ONE["table"]["size"])          # depth (x), width (y)
TABLE_HEIGHT = SCENE_ONE["table"]["height"]             # top surface
TABLE_TOP_THICKNESS = SCENE_ONE["table"]["top_thickness"]
LEG = SCENE_ONE["table"]["leg"]                         # square legs
# Box: 5 x 5 cm footprint (the gripper opens ~8 cm at -1.2 rad), 8 cm tall.
BOX_SIZE = tuple(SCENE_ONE["box"]["size"])
BOX_MASS = SCENE_ONE["box"]["mass"]
BOX_FROM_EDGE = SCENE_ONE["box"]["from_edge"]           # box centre behind the table's near edge
DROP_ZONE = tuple(SCENE_ONE["drop_zone"])               # where fetch_box.py --deliver carries the box
TABLE_NEAR_EDGE_X = TABLE_CENTER[0] - TABLE_SIZE[0] / 2
BOX_POS = (TABLE_NEAR_EDGE_X + BOX_FROM_EDGE, TABLE_CENTER[1], TABLE_HEIGHT + BOX_SIZE[2] / 2)

GRIP_KP = 20.0   # N*m/rad
JAW_FRICTION = (1.2, 1.0)  # static, dynamic: rubber-pad-like (PhysX default is 0.5)
GRIP_KD = 1.0    # N*m*s/rad

# Gripper open/closed (jointGripper: 0 closed, -pi/2 fully open).
GRIPPER_OPEN = -1.4  # wide: the moving jaw rotates about a pivot, so near its root it stays close
GRIPPER_CLOSED = 0.0

# IK control point (used by the script): between the fingertips, in the gripperStator frame. x along
# the fingers (tips at 0.149 m), z is the side the moving jaw sits on; the fixed
# jaw's inner face is at z = -0.009. A box of half-width 0.025 centred at
# z = +0.031 clears it by 15 mm on the way down (at 5 mm, 1-3 cm of IK tracking
# error put the fixed jaw on the box's top edge and knocked it over); the moving
# jaw then slides the box the 15 mm onto the fixed jaw as it closes.
JAW_CLEARANCE = 0.010
GRASP_POINT = (0.12, 0.0, -0.009 + BOX_SIZE[0] / 2 + JAW_CLEARANCE)

UNITREE_B2W_Z1_GRIPPER_CFG = UNITREE_B2W_Z1_CFG.replace(
    spawn=UNITREE_B2W_Z1_CFG.spawn.replace(
        usd_path=B2W_Z1_GRIPPER_USD,
        # self-collision on: the arm must not pass through the robot. The asset
        # filters the two pairs that overlap by design (head/lidar, arm base/body).
        articulation_props=UNITREE_B2W_Z1_CFG.spawn.articulation_props.replace(enabled_self_collisions=True),
    ),
    init_state=UNITREE_B2W_Z1_CFG.init_state.replace(
        pos=(SPAWN_POS[0], SPAWN_POS[1], UNITREE_B2W_Z1_CFG.init_state.pos[2]),
        joint_pos={**UNITREE_B2W_Z1_CFG.init_state.joint_pos, **ARM_STOW, "jointGripper": GRIPPER_CLOSED},
    ),
    actuators={
        **UNITREE_B2W_Z1_CFG.actuators,
        # Soft grip. At the asset's kp 100 the drive saturated its 30 N*m on
        # contact -- a few hundred N through convex-decomposed jaws -- and PhysX
        # resolved it by squeezing the box out of the angled jaw (the jaw kept
        # closing through it, -0.33 -> 0 rad, in every dropped grasp). kp 20 gives
        # ~6 N*m at the ~0.3 rad the box stops it: ~60 N squeeze for a 2 N box.
        "gripper": ImplicitActuatorCfg(joint_names_expr=["jointGripper"], stiffness=GRIP_KP, damping=GRIP_KD),
    },
)

# partners reported by the diagnostic front-hip contact sensor (one prim each)
HIP_PARTNERS = ["{ENV_REGEX_NS}/Table_top"] + [f"{{ENV_REGEX_NS}}/Table_leg{i}" for i in range(4)] + [
    "{ENV_REGEX_NS}/Box", "{ENV_REGEX_NS}/Robot/lidar_link", "{ENV_REGEX_NS}/Robot/head_Link",
    "{ENV_REGEX_NS}/Robot/base_link", "{ENV_REGEX_NS}/Robot/FR_thigh", "{ENV_REGEX_NS}/Robot/FR_calf",
    "{ENV_REGEX_NS}/Robot/FL_hip", "{ENV_REGEX_NS}/Robot/f_dc_link", "{ENV_REGEX_NS}/Robot/f_oc_link"] + [
    f"{{ENV_REGEX_NS}}/Robot/z1/{n}" for n in ("link00", "link02", "link03", "link04", "link06", "gripperStator", "gripperMover")
] + ["/World/ground"]

WOOD = sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(next(iter(SCENE_ONE["tables"].values()))["rgb"]), roughness=0.7)
CARDBOARD = sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(SCENE_ONE["box"]["rgb"]), roughness=0.9)


def _table_part(name, size, pos):
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Table_{name}",  # flat: a nested path needs its parent prim first
        spawn=sim_utils.CuboidCfg(size=size, collision_props=sim_utils.CollisionPropertiesCfg(), visual_material=WOOD),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


@configclass
class UnitreeB2WZ1GripperFetchEnvCfg(UnitreeB2WZ1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        s = self.scene
        s.num_envs = 1  # one warehouse, one robot
        s.robot = UNITREE_B2W_Z1_GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        s.terrain.terrain_type = "usd"
        s.terrain.usd_path = SCENE_ONE.get("environment_usd", WAREHOUSE_USD)
        self.terminations.terrain_out_of_bounds = None

        # ---- table (static colliders) and box (dynamic)
        tx, ty = TABLE_CENTER
        dx, dy = TABLE_SIZE
        top_z = TABLE_HEIGHT - TABLE_TOP_THICKNESS / 2
        s.table_top = _table_part("top", (dx, dy, TABLE_TOP_THICKNESS), (tx, ty, top_z))
        leg_h = TABLE_HEIGHT - TABLE_TOP_THICKNESS
        for i, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
            pos = (tx + sx * (dx / 2 - LEG), ty + sy * (dy / 2 - LEG), leg_h / 2)
            setattr(s, f"table_leg{i}", _table_part(f"leg{i}", (LEG, LEG, leg_h), pos))
        s.box = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Box",
            spawn=sim_utils.CuboidCfg(
                size=BOX_SIZE,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=BOX_MASS),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
                visual_material=CARDBOARD,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=BOX_POS),
        )

        # diagnostic: what the front hips touch, per partner (table parts, box, floor)
        s.hip_contacts = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/FR_hip",
            filter_prim_paths_expr=HIP_PARTNERS,
        )

        # ---- actions: the drive policy's two terms (unchanged), then the arm and gripper
        # Absolute joint targets. project/fetch_box.py does the inverse kinematics
        # itself: large moves (unfolding from the stow) go in joint space, because
        # differential IK from the fold wanders into joint limits; the short moves
        # near the box are damped-least-squares IK on the GRASP_POINT frame.
        self.actions.arm_action = JointPositionActionCfg(
            asset_name="robot", joint_names=list(ARM_STOW), scale=1.0, use_default_offset=False, preserve_order=True,
        )
        # Absolute gripper target (rad), so the script can close it gradually: a
        # binary action slams the jaw shut at full speed and pops the box out.
        self.actions.gripper_action = JointPositionActionCfg(
            asset_name="robot", joint_names=["jointGripper"], scale=1.0, use_default_offset=False,
        )
        # the arm is driven now, so it no longer needs holding at stow
        self.events.hold_arm_stow = None

        # the drive policy's "last action" input: its own 16 numbers only, as trained
        for group in (self.observations.policy, self.observations.critic):
            group.actions.func = last_actions_of
            group.actions.params = {"names": DRIVE_ACTIONS}
            group.enable_corruption = False

        # ---- events: start still at the spawn; the box position is randomised on the table
        still = (0.0, 0.0)
        self.events.randomize_reset_base.params = {
            "pose_range": {k: still for k in ("x", "y", "z", "roll", "pitch", "yaw")},
            "velocity_range": {k: still for k in ("x", "y", "z", "roll", "pitch", "yaw")},
        }
        self.events.reset_box = EventTerm(
            func=base_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("box"),
                "pose_range": {"x": (-0.02, 0.03), "y": (-0.25, 0.25), "yaw": (-0.4, 0.4)},
                "velocity_range": {},
            },
        )
        # Rubber-pad friction on the jaws only (default PhysX material elsewhere).
        # Applied once at startup; with one bucket it is a fixed value, not a range.
        self.events.gripper_pads = EventTerm(
            func=base_mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["gripperStator", "gripperMover"]),
                "static_friction_range": (JAW_FRICTION[0], JAW_FRICTION[0]),
                "dynamic_friction_range": (JAW_FRICTION[1], JAW_FRICTION[1]),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 1,
            },
        )
        for name in ("randomize_rigid_body_material", "randomize_rigid_body_mass_base",
                     "randomize_rigid_body_mass_others", "randomize_com_positions",
                     "randomize_apply_external_force_torque", "randomize_actuator_gains",
                     "randomize_push_robot"):
            if hasattr(self.events, name):
                setattr(self.events, name, None)

        # ---- commands: base velocity is written by the script, never resampled
        cmd = self.commands.base_velocity
        cmd.heading_command = False
        cmd.rel_standing_envs = 0.0
        cmd.rel_heading_envs = 0.0
        cmd.resampling_time_range = (1.0e6, 1.0e6)
        cmd.debug_vis = False
        cmd.ranges.heading = None

        self.episode_length_s = 1.0e4
        self.terminations.time_out = None

        # camera follows the robot (videos, livestream)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.eye = (-2.2, -2.4, 1.9)
        self.viewer.lookat = (0.6, 0.3, 0.4)


# ------------------------------------------------------------------ three tables, nine boxes
# Three tables in a row down the aisle (long side along y, approached from -x like the first one),
# 5.5 m apart so each has its own clear 2.1 m docking line along its near edge in both directions
# (the Nav2 staging pose S is 2.1 m before the pick point). The yellow one stands where the single
# table was. On each table a red, a yellow and a green box, 0.3 m apart along the table, their order
# shuffled at every reset (place_boxes_shuffled).
# (from SCENE_MULTI, scenes/warehouse_three_tables.yaml)
TABLES3 = {k: tuple(v["center"]) for k, v in SCENE_MULTI["tables"].items()}     # centre (x, y)
COLORS3 = tuple(SCENE_MULTI["tables"])                                          # table names
TABLE_RGB = {k: tuple(v["rgb"]) for k, v in SCENE_MULTI["tables"].items()}      # painted wood
BOX_RGB = {k: tuple(v) for k, v in SCENE_MULTI["box"]["colors"].items()}         # saturated
BOX_COLORS = tuple(BOX_RGB)                                                     # one box of each colour per table
BOX_SLOTS_Y = tuple(SCENE_MULTI["box"]["slots_y"])  # along the table from its centre
DROP_ZONE3 = tuple(SCENE_MULTI["drop_zone"])


def box_name(table, color):
    return f"box_{table}_{color}"


def table_rect3(table):
    """(x0, x1, y0, y1) of a table top."""
    tx, ty = TABLES3[table]
    return (tx - TABLE_SIZE[0] / 2, tx + TABLE_SIZE[0] / 2, ty - TABLE_SIZE[1] / 2, ty + TABLE_SIZE[1] / 2)


def place_boxes_shuffled(env, env_ids, jitter_x=(-0.02, 0.03), jitter_y=(-0.04, 0.04), yaw=(-0.4, 0.4)):
    """Reset event: on every table, the three colours in a random order on the three slots, each box with
    a little position and yaw jitter, at rest on the table top."""
    import math
    import torch
    for table in COLORS3:
        tx, ty = TABLES3[table]
        order = torch.randperm(len(BOX_COLORS)).tolist()
        for slot, ci in zip(BOX_SLOTS_Y, order):
            obj = env.scene[box_name(table, BOX_COLORS[ci])]
            n = len(env_ids)
            dev = obj.data.root_pos_w.device
            pos = torch.zeros(n, 3, device=dev)
            pos[:, 0] = tx - TABLE_SIZE[0] / 2 + BOX_FROM_EDGE + torch.empty(n, device=dev).uniform_(*jitter_x)
            pos[:, 1] = ty + slot + torch.empty(n, device=dev).uniform_(*jitter_y)
            pos[:, 2] = TABLE_HEIGHT + BOX_SIZE[2] / 2
            pos += env.scene.env_origins[env_ids]
            a = torch.empty(n, device=dev).uniform_(*yaw)
            quat = torch.stack([torch.cos(a / 2), torch.zeros_like(a), torch.zeros_like(a), torch.sin(a / 2)], dim=1)
            obj.write_root_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids=env_ids)
            obj.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)


@configclass
class UnitreeB2WZ1GripperFetch3EnvCfg(UnitreeB2WZ1GripperFetchEnvCfg):
    """The fetch scene with three coloured tables (red, yellow, green) and on each a red, a yellow and a
    green box -- fetch_box.py picks one (--target table/box) and DualVLN is told which table."""

    def __post_init__(self):
        super().__post_init__()
        s = self.scene
        for name in ["table_top"] + [f"table_leg{i}" for i in range(4)] + ["box"]:
            setattr(s, name, None)
        s.terrain.usd_path = SCENE_MULTI.get("environment_usd", WAREHOUSE_USD)
        dx, dy = TABLE_SIZE
        top_z = TABLE_HEIGHT - TABLE_TOP_THICKNESS / 2
        leg_h = TABLE_HEIGHT - TABLE_TOP_THICKNESS
        for table in COLORS3:
            tx, ty = TABLES3[table]
            paint = sim_utils.PreviewSurfaceCfg(diffuse_color=TABLE_RGB[table], roughness=0.6)
            setattr(s, f"table_{table}_top", AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Table_{table}_top",
                spawn=sim_utils.CuboidCfg(size=(dx, dy, TABLE_TOP_THICKNESS), collision_props=sim_utils.CollisionPropertiesCfg(),
                                          visual_material=paint),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(tx, ty, top_z))))
            for i, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
                setattr(s, f"table_{table}_leg{i}", AssetBaseCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Table_{table}_leg{i}",
                    spawn=sim_utils.CuboidCfg(size=(LEG, LEG, leg_h), collision_props=sim_utils.CollisionPropertiesCfg(),
                                              visual_material=paint),
                    init_state=AssetBaseCfg.InitialStateCfg(pos=(tx + sx * (dx / 2 - LEG), ty + sy * (dy / 2 - LEG), leg_h / 2))))
            for slot, color in zip(BOX_SLOTS_Y, BOX_COLORS):
                setattr(s, box_name(table, color), RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Box_{table}_{color}",
                    spawn=sim_utils.CuboidCfg(
                        size=BOX_SIZE,
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                        mass_props=sim_utils.MassPropertiesCfg(mass=BOX_MASS),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=BOX_RGB[color], roughness=0.8),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(tx - dx / 2 + BOX_FROM_EDGE, ty + slot, TABLE_HEIGHT + BOX_SIZE[2] / 2))))
        # the hip contact diagnostic: its partners are the table parts and boxes by name
        partners = [f"{{ENV_REGEX_NS}}/Table_{t}_top" for t in COLORS3] + \
            [f"{{ENV_REGEX_NS}}/Table_{t}_leg{i}" for t in COLORS3 for i in range(4)] + \
            [f"{{ENV_REGEX_NS}}/Box_{t}_{c}" for t in COLORS3 for c in BOX_COLORS] + \
            [p for p in HIP_PARTNERS if "/Robot/" in p or p.endswith("/World/ground")]
        s.hip_contacts = s.hip_contacts.replace(filter_prim_paths_expr=partners)
        self.events.reset_box = EventTerm(func=place_boxes_shuffled, mode="reset", params={})
