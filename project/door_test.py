"""The door task's physics, without the robot doing anything: are the doors where the scene says, painted, latched,
and do they open once the lever is turned? (scenes/hospital_doors.yaml)

    FETCH_SCENE=/workspace/scenes/hospital_doors.yaml ./isaac/run_script.sh project/door_test.py --device cpu

1. renders/door_test/doors.png: the three doors from the hall, the middle one blue.
2. A 60 N push on door b's leaf near the handle, latched: the door must stay shut.
3. The lever held pressed 40 deg: the latch lets go (fetch_box.py's rule), the same push opens the door.
4. Push off, lever released: the door swings back no further than it was pushed (no spring) and stays unlatched.
"""
import argparse
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="RobotLab-Isaac-Door-Hospital-Unitree-B2W-Z1-Gripper-v0")
parser.add_argument("--out", default="/workspace/renders/door_test")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402
from pxr import Gf  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import robot_lab.tasks  # noqa: E402,F401
import eth_loco  # noqa: E402
from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.unitree_b2w import fetch_env_cfg as F  # noqa: E402

cfg = parse_env_cfg(args.task, device=args.device or "cpu", num_envs=1)
eth_loco.patch_cfg(cfg)
cfg.sim.physx.enable_scene_query_support = True  # ray casts and overlaps for the diagnosis
sc = F.SCENE_DOORS
cfg.viewer.origin_type = "world"
cfg.viewer.eye = tuple(sc["camera"]["fixed"]["eye"])
cfg.viewer.lookat = tuple(sc["camera"]["fixed"]["lookat"])
env = gym.make(args.task, cfg=cfg, render_mode="rgb_array")
u = env.unwrapped
env.reset()
os.makedirs(args.out, exist_ok=True)
stage = omni.usd.get_context().get_stage()
dt = u.sim.get_physics_dt()


def paint(k, rgb):
    sh = stage.GetPrimAtPath(f"/World/envs/env_0/Door_{k}/Looks/Paint/MI_Door_01")
    sh.GetAttribute("inputs:ColorAlbedo").Set(Gf.Vec4f(*rgb, 1.0))
    return sh.IsValid()


cols = F.DOOR["colors"]
print("DOORTEST paint", paint("a", cols["white"]), paint("b", cols["blue"]), paint("c", cols["yellow"]), flush=True)
doors = {k: u.scene[F.door_name(k)] for k in F.DOORS}
for k, d in doors.items():
    leaf = d.find_bodies("leaf")[0][0]
    print(f"DOORTEST door {k}: joints {d.joint_names} leaf at {[round(float(v), 3) for v in d.data.body_pos_w[0, leaf]]}"
          f" root {[round(float(v), 3) for v in d.data.root_pos_w[0]]}", flush=True)


def sim(seconds, push=None, press=None):
    """Step physics only (the robot is not driven: its pose is irrelevant here); push: N along +x on door b's leaf at
    the handle's height near its free edge; press: the lever's joint target (rad); the latch rule each step."""
    d = doors["b"]
    leaf = d.find_bodies("leaf")[0]
    hinge, handle = d.find_joints("hinge")[0], d.find_joints("handle_joint")[0]
    for _ in range(int(seconds / dt)):
        if push is not None:
            f = torch.zeros(1, 1, 3, device=d.device)
            f[..., 0] = push
            # at the leaf's centre of mass the force has no lever arm about the hinge: give it one (torque about z)
            arm = 0.9  # m from the hinge
            t = torch.zeros(1, 1, 3, device=d.device)
            t[..., 2] = push * arm * 0.5  # the extra moment a push 0.9 m out would add over one at the centre (~0.6 m)
            d.set_external_force_and_torque(f, t, body_ids=leaf)
        else:
            d.set_external_force_and_torque(torch.zeros(1, 1, 3, device=d.device), torch.zeros(1, 1, 3, device=d.device),
                                            body_ids=leaf)
        if press is not None:
            q = d.data.joint_pos_target.clone()
            q[0, handle[0]] = press
            d.set_joint_position_target(q)
        # the virtual latch (fetch_box.py): unlatch once the lever is turned past latch_release
        lever = float(d.data.joint_pos[0, handle[0]])
        if lever < -math.radians(F.DOOR["latch_release"]) and float(d.data.joint_stiffness[0, hinge[0]]) > 0:
            d.write_joint_stiffness_to_sim(torch.zeros(1, 1, device=d.device), joint_ids=hinge)
            print("DOORTEST unlatched; hinge stiffness now", float(d.data.joint_stiffness[0, hinge[0]]), flush=True)
        d.write_data_to_sim()
        u.sim.step()
        d.update(dt)
    return math.degrees(float(d.data.joint_pos[0, hinge[0]])), math.degrees(float(d.data.joint_pos[0, handle[0]]))


import carb  # noqa: E402
from omni.physx import get_physx_scene_query_interface  # noqa: E402
_qi = get_physx_scene_query_interface()
for _k in ("b",):
    _o = F.DOORS[_k]["origin"]
    _hits = []
    # the leaf's box collider, closed: centre and half extents (asset frame + origin), inset as built
    _c = carb.Float3(_o[0] + 0.023, _o[1] - 0.618, _o[2] + 1.119)
    _qi.overlap_box(carb.Float3(0.023, 0.603, 1.069), _c, carb.Float4(0, 0, 0, 1),
                    lambda h: _hits.append(h.rigid_body + " | " + h.collision) or True, False)
    print(f"DOORTEST overlaps of door {_k}'s leaf box:", sorted(set(_hits)), flush=True)
    for _dx in (-0.3, -0.05, 0.0, 0.023, 0.046, 0.08, 0.12, 0.3):
        _r = _qi.raycast_closest(carb.Float3(_o[0] + _dx, _o[1] - 0.6, 0.5), carb.Float3(0, 0, -1), 2.0)
        print(f"DOORTEST floor at x {_o[0] + _dx:.2f}: hit {_r.get('hit')} z {0.5 - _r.get('distance', 0):.4f} {str(_r.get('collision')).split('/')[-1]}",
              flush=True)
u.sim.render()
for _ in range(3):
    u.sim.step()
    u.sim.render()
imageio.imwrite(f"{args.out}/doors.png", u.render())
print("DOORTEST latched, pushed 60 N for 2 s: hinge %.2f deg, lever %.1f deg" % sim(2.0, push=60.0), flush=True)
print("DOORTEST lever pressed to -40 deg, no push, 1 s: hinge %.2f deg, lever %.1f deg"
      % sim(1.0, press=-math.radians(40)), flush=True)
print("DOORTEST lever pressed, pushed 60 N for 2 s: hinge %.2f deg, lever %.1f deg"
      % sim(2.0, push=60.0, press=-math.radians(40)), flush=True)
print("DOORTEST push off, lever released, 2 s: hinge %.2f deg, lever %.1f deg" % sim(2.0, press=F.DOOR["handle_preload"]), flush=True)
# diagnosis: drive the hinge open with its own motor (stiffness 50 towards +30 deg): blocked, or not pushed?
_d = doors["b"]
_h = _d.find_joints("hinge")[0]
_d.write_joint_stiffness_to_sim(torch.full((1, 1), 50.0, device=_d.device), joint_ids=_h)
_q = _d.data.joint_pos_target.clone()
_q[0, _h[0]] = math.radians(30)
_d.set_joint_position_target(_q)
for _ in range(int(2.0 / dt)):
    _d.write_data_to_sim()
    u.sim.step()
    _d.update(dt)
print("DOORTEST hinge driven towards +30 deg, 2 s: hinge %.2f deg, velocity %.3f, effort %.1f" % (
    math.degrees(float(_d.data.joint_pos[0, _h[0]])), float(_d.data.joint_vel[0, _h[0]]),
    float(_d.data.applied_torque[0, _h[0]])), flush=True)
_q[0, _h[0]] = math.radians(-30)
_d.set_joint_position_target(_q)
for _ in range(int(2.0 / dt)):
    _d.write_data_to_sim()
    u.sim.step()
    _d.update(dt)
print("DOORTEST hinge driven towards -30 deg, 2 s: hinge %.2f deg" % math.degrees(float(_d.data.joint_pos[0, _h[0]])), flush=True)
u.sim.render()
imageio.imwrite(f"{args.out}/door_b_open.png", u.render())
env.close()
app.close()
