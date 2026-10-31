"""High-resolution stills of the B2W + Z1 + gripper standing in a reconstructed (splat + collider) scene, from fixed
world cameras: for title pages and figures. The robot is the one of scene_tour.py --robot b2w_z1 (fetch gripper task,
ETH stance fine-tune standing still, NuRec render settings copied from the stage, "quality" rendering mode).

    ISAAC_IMAGE=isaac-lab:2.3.2 CONTAINER_NAME=isaac-still ./isaac/run_script.sh project/scene_still.py --device cpu \\
        --stage /workspace/scenes/spatialAI/new_office_scene/robot/new_office_robot_ground.usda \\
        --pose=-15.2,2.4,-70 --view "a:-15.0,0.3,1.45:-15.3,4.0,0.6" --out /workspace/renders/stills/kitchen

--pose x,y,yaw (deg) of the robot; --view name:eye x,y,z:look-at x,y,z (world, repeatable); --arm six joint targets
(default: the stow pose of the tours). Writes <out>/<name>.png and <out>/views.json (robot pose, cameras).
"""

import argparse
import json
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--stage", required=True, help="ground USD (container path)")
parser.add_argument("--pose", required=True, help="robot x,y,yaw_deg")
parser.add_argument("--view", action="append", required=True, help="name:eye x,y,z:lookat x,y,z (world)")
parser.add_argument("--res", default="2048x1884")
parser.add_argument("--hfov", type=float, default=65.0, help="deg, horizontal")
parser.add_argument("--arm", default=None, help="joint1..joint6 targets (rad); default ARM_STOW")
parser.add_argument("--gripper", type=float, default=None, help="gripper target (rad; default closed)")
parser.add_argument("--settle", type=float, default=2.5, help="s standing still before the stills")
parser.add_argument("--out", required=True)
parser.add_argument("--task", default="RobotLab-Isaac-Fetch-Warehouse-Unitree-B2W-Z1-Gripper-v0")
parser.add_argument("--policy", default="/workspace/eth/policy_stance_v5_600.pt")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.headless = True
app = AppLauncher(args).app

import omni.kit.app  # noqa: E402

omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("omni.usd.schema.omni_nurec_types", True)

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.envs import mdp as base_mdp  # noqa: E402
from isaaclab.managers import EventTermCfg as EventTerm  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import robot_lab.tasks  # noqa: E402,F401
import eth_loco  # noqa: E402
from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.unitree_b2w.arm_cargo_env_cfg import ARM_STOW  # noqa: E402
from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.unitree_b2w.fetch_env_cfg import GRIPPER_CLOSED  # noqa: E402


def say(*a):
    print("STILL|", *a, flush=True)


os.makedirs(args.out, exist_ok=True)
px, py, pyaw = (float(v) for v in args.pose.split(","))
pyaw = math.radians(pyaw)
W, H = (int(v) for v in args.res.split("x"))

# ---------------------------------------------------------------- env: as scene_tour.py --robot b2w_z1, standing still
cfg = parse_env_cfg(args.task, device=args.device or "cpu", num_envs=1)
cfg.scene.terrain.usd_path = args.stage
cfg.scene.sky_light = None  # the stage has its own dome light
for name in list(vars(cfg.terminations)):
    if not name.startswith("_"):
        setattr(cfg.terminations, name, None)
cfg.episode_length_s = 1e5
for ev in ("randomize_rigid_body_mass_base", "randomize_rigid_body_mass_others", "randomize_com_positions",
           "randomize_apply_external_force_torque", "randomize_actuator_gains", "randomize_push_robot",
           "randomize_reset_joints"):
    if hasattr(cfg.events, ev):
        setattr(cfg.events, ev, None)
for name in ("table_top", "table_leg0", "table_leg1", "table_leg2", "table_leg3", "box", "hip_contacts"):
    setattr(cfg.scene, name, None)
cfg.events.reset_box = None
cfg.events.tour_wheel_friction = EventTerm(
    func=base_mdp.randomize_rigid_body_material, mode="startup",
    params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_foot"]), "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0), "restitution_range": (0.0, 0.0), "num_buckets": 1})
still = (0.0, 0.0)
cfg.events.randomize_reset_base.params = {
    "pose_range": {k: still for k in ("x", "y", "z", "roll", "pitch", "yaw")},
    "velocity_range": {k: still for k in ("x", "y", "z", "roll", "pitch", "yaw")},
}
cfg.commands.base_velocity.debug_vis = False
eth_loco.patch_cfg(cfg)
cfg.scene.robot = cfg.scene.robot.replace(init_state=cfg.scene.robot.init_state.replace(
    pos=(px, py, eth_loco.SPAWN_Z), rot=(math.cos(pyaw / 2), 0.0, 0.0, math.sin(pyaw / 2))))
cfg.sim.render_interval = cfg.decimation                        # every step: the camera is current when saved
cfg.sim.render.rendering_mode = "quality"
ap_h = 20.955
cfg.scene.still_cam = CameraCfg(
    prim_path="{ENV_REGEX_NS}/StillCam", width=W, height=H, data_types=["rgb"], update_period=0.0,
    spawn=sim_utils.PinholeCameraCfg(focal_length=W / 2 / math.tan(math.radians(args.hfov) / 2) * ap_h / W,
                                     horizontal_aperture=ap_h, clipping_range=(0.05, 60.0)))

env = gym.make(args.task, cfg=cfg)
u = env.unwrapped
dt = u.step_dt
robot = u.scene["robot"]
cam = u.scene["still_cam"]
# the splat's NuRec render settings (scene_tour.py): only the scene's own layers, kept alive while read
import carb.settings  # noqa: E402
from pxr import Usd  # noqa: E402

_rs, _scene = {}, Usd.Stage.Open(args.stage)
for _layer in _scene.GetUsedLayers():
    _rs.update(_layer.customLayerData.get("renderSettings", {}))
del _scene
for _k, _v in _rs.items():
    carb.settings.get_settings().set("/" + _k.replace(":", "/"), _v)
say(f"render settings from the stage: {_rs}")

driver = eth_loco.Driver(u, onnx_path=args.policy)
_sl, _st = {}, 0
for _n, _w in zip(u.action_manager.active_terms, u.action_manager.action_term_dim):
    _sl[_n], _st = slice(_st, _st + _w), _st + _w
arm_names = u.action_manager.get_term("arm_action")._joint_names
arm_q = [float(v) for v in args.arm.split(",")] if args.arm else [ARM_STOW[j] for j in arm_names]
HOLD = [(_sl["arm_action"], torch.tensor(arm_q)),
        (_sl["gripper_action"], torch.tensor([GRIPPER_CLOSED if args.gripper is None else args.gripper]))]


def step():
    with torch.inference_mode():
        a = torch.zeros(1, u.action_manager.total_action_dim, device=u.device)
        driver.fill(a, [0.0, 0.0, 0.0])
        for sl, val in HOLD:
            a[0, sl] = val.to(a.device)
        env.step(a)


with torch.inference_mode():
    env.reset()
driver.reset()
for _ in range(int(args.settle / dt)):
    step()
p = robot.data.root_pos_w[0].cpu().numpy()
info = {"robot": {"x": float(p[0]), "y": float(p[1]), "z": float(p[2]), "yaw_deg": math.degrees(float(robot.data.heading_w[0])),
                  "tilt_deg": eth_loco.tilt_deg(robot), "arm": dict(zip(arm_names, arm_q))},
        "res": [W, H], "hfov_deg": args.hfov, "views": {}}
say(f"settled at ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.3f}), yaw {info['robot']['yaw_deg']:.1f} deg, tilt {info['robot']['tilt_deg']:.1f} deg")
for spec in args.view:
    name, eye, look = spec.split(":")
    eye = [float(v) for v in eye.split(",")]
    look = [float(v) for v in look.split(",")]
    cam.set_world_poses_from_view(torch.tensor([eye], dtype=torch.float32, device=u.device),
                                  torch.tensor([look], dtype=torch.float32, device=u.device))
    for _ in range(12):  # the robot keeps balancing; the renderer settles on the new view
        step()
    img = cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
    Image.fromarray(img).save(f"{args.out}/{name}.png")
    info["views"][name] = {"eye": eye, "lookat": look}
    say(f"view {name}: eye {eye} lookat {look} -> {args.out}/{name}.png")
json.dump(info, open(f"{args.out}/views.json", "w"), indent=1)
env.close()
app.close()
