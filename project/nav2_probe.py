"""Nav2 groundwork, without ROS: build the warehouse occupancy map from the simulator's colliders
and take one 3-height scan at the spawn, then draw both (with the table and robot) to check them.

    ISAAC_IMAGE=isaac-lab:2.3.2 ./run_script.sh nav2_probe.py --device cpu
Output: nav2/maps/warehouse.{pgm,yaml}, renders/nav2/map_check.png
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="RobotLab-Isaac-Fetch-Warehouse-Unitree-B2W-Z1-Gripper-v0")
parser.add_argument("--res", type=float, default=0.05)
parser.add_argument("--margin", type=float, default=0.5, help="m added round the floor's bounding box")
parser.add_argument("--map", default="warehouse", help="map name: nav2/maps/<map>.{pgm,yaml} (warehouse3 for the three-table scene)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import robot_lab.tasks  # noqa: E402,F401
import eth_loco  # noqa: E402
import nav2_link  # noqa: E402

cfg = parse_env_cfg(args.task, device=args.device or "cpu", num_envs=1)
eth_loco.patch_cfg(cfg)
nav2_link.enable_queries(cfg)
env = gym.make(args.task, cfg=cfg)
u = env.unwrapped
env.reset()
for _ in range(10):
    u.sim.step()
robot = u.scene["robot"]

import carb  # noqa: E402
from omni.physx import get_physx_scene_query_interface  # noqa: E402
qi = get_physx_scene_query_interface()
from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.unitree_b2w import fetch_env_cfg as F0  # noqa: E402
rc = qi.raycast_closest(carb.Float3(3.0, 1.0, 0.57), carb.Float3(1.0, 0.0, 0.0), 5.0)
print("PROBE DEBUG raycast_closest toward table top:", rc, flush=True)
hits = []
n = qi.overlap_box(carb.Float3(0.1, 0.1, 0.1), carb.Float3(F0.TABLE_CENTER[0], F0.TABLE_CENTER[1], 0.57), carb.Float4(0, 0, 0, 1),
                   lambda h: hits.append(h.collision) or True, False)
print("PROBE DEBUG overlap at table top: n", n, "hits", hits[:5], flush=True)
rc2 = qi.raycast_closest(carb.Float3(1.0, -1.5, 0.35), carb.Float3(0.0, 1.0, 0.0), 20.0)
print("PROBE DEBUG raycast from spawn +y:", rc2, flush=True)
stage = omni.usd.get_context().get_stage()
for pth in ("/World/ground", "/World/envs/env_0/Table_top"):
    pr = stage.GetPrimAtPath(pth)
    print("PROBE DEBUG prim", pth, pr.IsValid(), [a for a in pr.GetAppliedSchemas()][:6] if pr.IsValid() else None, flush=True)
bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
rng = bbox.ComputeWorldBound(stage.GetPrimAtPath("/World/ground")).ComputeAlignedRange()
lo, hi = rng.GetMin(), rng.GetMax()
print(f"PROBE ground bbox x {lo[0]:.2f}..{hi[0]:.2f}  y {lo[1]:.2f}..{hi[1]:.2f}  z {lo[2]:.2f}..{hi[2]:.2f}", flush=True)
xr = (math.floor(lo[0] - args.margin), math.ceil(hi[0] + args.margin))
yr = (math.floor(lo[1] - args.margin), math.ceil(hi[1] + args.margin))
occ = nav2_link.build_map(f"/workspace/nav2/maps/{args.map}", xr, yr, res=args.res)

p = robot.data.root_pos_w[0].cpu().numpy()
q = robot.data.root_quat_w[0].cpu().numpy()
yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
sc = nav2_link.Scanner()
import time  # noqa: E402
t0 = time.time()
r = sc.scan(float(p[0]), float(p[1]), yaw)
print(f"PROBE scan at ({p[0]:.2f},{p[1]:.2f}) yaw {math.degrees(yaw):.0f}: {np.isfinite(r).sum()}/{len(r)} hits, "
      f"min {np.nanmin(np.where(np.isfinite(r), r, np.nan)):.2f} m, {1000 * (time.time() - t0):.0f} ms", flush=True)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(occ, origin="lower", cmap="gray_r", extent=(xr[0], xr[1], yr[0], yr[1]))
ang = yaw + sc.angles
ok = np.isfinite(r)
ax.scatter(p[0] + r[ok] * np.cos(ang[ok]), p[1] + r[ok] * np.sin(ang[ok]), s=3, c="red", label="scan hits")
ax.plot(p[0], p[1], "b^", ms=10, label="robot")
from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.unitree_b2w import fetch_env_cfg as F  # noqa: E402
tx, ty = F.TABLE_CENTER
dx, dy = F.TABLE_SIZE
for (tx_, ty_) in (F.TABLES3.values() if "Fetch3" in args.task else [(tx, ty)]):
    ax.add_patch(plt.Rectangle((tx_ - dx / 2, ty_ - dy / 2), dx, dy, fill=False, ec="green", lw=2))
ax.legend(loc="upper right")
ax.set_title("occupancy map from colliders (black) + one 3-height scan")
import os  # noqa: E402
os.makedirs("/workspace/renders/nav2", exist_ok=True)
fig.savefig(f"/workspace/renders/nav2/map_check_{args.map}.png", dpi=110, bbox_inches="tight")
print(f"PROBE wrote /workspace/renders/nav2/map_check_{args.map}.png", flush=True)
env.close()
app.close()
