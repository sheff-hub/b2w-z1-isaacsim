"""Render views of an Isaac Lab task scene from fixed world cameras, with the robot hidden (or shown), without
running an episode. For choosing figure cameras: is a view clear of racks and ceiling fixtures, what does it frame.

    ISAAC_IMAGE=isaac-lab:2.3.2 ./run_script.sh scene_views.py \\
        --view "A:4.1,-7.6,7.6:4.1,2.3,0.0" --view "B:10.5,-3.5,8.5:3.9,2.6,0.0" [--show_robot]
Images: renders/report/views/<name>.png (1920 x 1080, the viewer camera: 60 deg horizontal field of view)
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="RobotLab-Isaac-Fetch3-Warehouse-Unitree-B2W-Z1-Gripper-v0")
parser.add_argument("--view", action="append", required=True, help="name:eye x,y,z:lookat x,y,z (world)")
parser.add_argument("--res", default="1920x1080")
parser.add_argument("--out", default="/workspace/renders/report/views")
parser.add_argument("--show_robot", action="store_true")
parser.add_argument("--seed", type=int, default=43)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.headless = True
launcher = AppLauncher(args)

import gymnasium as gym  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import UsdGeom  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import robot_lab.tasks  # noqa: E402,F401

cfg = parse_env_cfg(args.task, device=args.device or "cuda:0", num_envs=1)
cfg.viewer.origin_type = "world"
cfg.viewer.resolution = tuple(int(v) for v in args.res.split("x"))
env = gym.make(args.task, cfg=cfg, render_mode="rgb_array")
u = env.unwrapped
env.reset(seed=args.seed)
robot = omni.usd.get_context().get_stage().GetPrimAtPath("/World/envs/env_0/Robot")
if not args.show_robot:
    UsdGeom.Imageable(robot).MakeInvisible()
os.makedirs(args.out, exist_ok=True)
u.render()  # the first call only creates the viewport capture (and returns an empty image)
for spec in args.view:
    name, eye, look = spec.split(":")
    eye = [float(v) for v in eye.split(",")]
    look = [float(v) for v in look.split(",")]
    u.viewport_camera_controller.update_view_location(eye, look)
    for _ in range(6):  # let the renderer settle on the new view
        u.sim.render()
    img = u.render()
    Image.fromarray(img[..., :3]).save(f"{args.out}/{name}.png")
    print(f"VIEW| {name}: eye {eye} lookat {look} -> {args.out}/{name}.png", flush=True)
env.close()
launcher.app.close()
