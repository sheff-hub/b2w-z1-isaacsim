"""High-resolution stills of the robot for the report, with the pixel position of each sensor.

The robot stands on the plain ground stage (legs held in ETH's stance), the arm in a display pose, and
each --shot is rendered at full resolution. For every shot the projected pixel positions of the sensors
(and a few body parts) are written to <name>.json, and <name>_debug.png marks them, so figure labels can
point at the right spot (report Fig. "sensor setup").

    ISAAC_IMAGE=isaac-lab:2.3.2 ./run_script.sh report_shots.py --arm_pose 0,1.6,-1.3,0.2,0,-1.57 \\
        --shot "sensors:2.3,1.7,1.35:0.1,0.0,0.55:30:2400x1500"
Shot spec  name:eye:look-at:horizontal FOV deg:WxH; eye / look-at are x,y,z (world; the robot stands at the origin
facing +x) or @anchor[+dx,dy,dz] relative to a sensor or part (e.g. @ring+0.4,0.3,0.1)
Images: renders/report/<name>.png, <name>_debug.png, <name>.json
"""

import argparse
import json
import math
import os

p = argparse.ArgumentParser()
p.add_argument("--stage", default="/workspace/robots_usd/b2w_z1_gripper_on_ground.usd")
p.add_argument("--robot", default="/World/b2w_z1")
p.add_argument("--out", default="/workspace/renders/report")
p.add_argument("--arm_pose", default="0,0.15,-0.15,0,0,0", help="joint1..joint6 (rad); default = the pipeline's stow")
p.add_argument("--grip", type=float, default=-0.6, help="gripper angle (rad, 0 closed, -1.4 wide open)")
p.add_argument("--stance", default="0.0,0.8,-1.5", help="hip,thigh,calf (rad); robot_lab's stance holds with the asset's gains (ETH's 0.4/-1.3 tips over without its controller)")
p.add_argument("--shot", action="append", required=True, help="name:eye:lookat:hfov:WxH (see the docstring)")
p.add_argument("--sun", type=float, default=2500.0, help="distant light intensity (0 = dome light only)")
args = p.parse_args()

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "width": 1280, "height": 720})

import numpy as np
import omni.replicator.core as rep
from isaacsim.core.api import SimulationContext
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import get_current_stage, open_stage
from PIL import Image, ImageDraw
from pxr import Gf, UsdGeom, UsdLux


def say(*a):
    print("SHOT|", *a, flush=True)


open_stage(args.stage)
stage = get_current_stage()
UsdLux.DomeLight.Define(stage, "/World/dome").CreateIntensityAttr(900)
if args.sun > 0:
    sun = UsdLux.DistantLight.Define(stage, "/World/sun")
    sun.CreateIntensityAttr(args.sun)
    sun.CreateAngleAttr(1.0)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(35.0, 25.0, 40.0))

R = args.robot
# sensor / part anchors: (rigid body, prim whose origin is the anchor, extra offset in that prim's frame)
ANCHORS = {
    "front_camera": (f"{R}/f_oc_link", f"{R}/f_oc_link/front_camera", (0, 0, 0)),
    "rear_camera": (f"{R}/r_oc_ink", f"{R}/r_oc_ink/rear_camera", (0, 0, 0)),
    "lidar": (f"{R}/lidar_link", f"{R}/lidar_link", (0, 0, 0)),
    "imu": (f"{R}/imu_link", f"{R}/imu_link", (0, 0, 0)),
    "wrist_camera": (f"{R}/z1/link06", f"{R}/z1/link06/end_effector_camera", (0, 0, 0)),
    "d405_body": (f"{R}/z1/link06", f"{R}/z1/link06/wrist_d405/d405", (-0.008, -0.009, 0.0)),
    "ring": (f"{R}/z1/link06", f"{R}/z1/link06", (0.0345, 0.0, 0.0)),
    "gripper": (f"{R}/z1/gripperStator", f"{R}/z1/gripperStator", (0.11, 0.0, 0.0)),
    "arm_link03": (f"{R}/z1/link03", f"{R}/z1/link03", (0.0, 0.0, 0.0)),
    "arm_link04": (f"{R}/z1/link04", f"{R}/z1/link04", (0.0, 0.0, 0.0)),
    "arm_base": (f"{R}/z1/link01", f"{R}/z1/link01", (0.0, 0.0, 0.0)),
    "base": (f"{R}/base_link", f"{R}/base_link", (0.0, 0.0, 0.0)),
    "wheel_FL": (f"{R}/FL_foot", f"{R}/FL_foot", (0.0, 0.0, 0.0)),
    "wheel_RL": (f"{R}/RL_foot", f"{R}/RL_foot", (0.0, 0.0, 0.0)),
    "knee_FL": (f"{R}/FL_calf", f"{R}/FL_calf", (0.0, 0.0, 0.0)),
    "hip_FL": (f"{R}/FL_thigh", f"{R}/FL_thigh", (0.0, 0.0, 0.0)),
}

# static offsets of each anchor in its body's frame, from the authored stage (bodies and sensors are rigidly fixed)
xc = UsdGeom.XformCache()
LOCAL = {}
for key, (body, prim, extra) in ANCHORS.items():
    if not stage.GetPrimAtPath(prim).IsValid() or not stage.GetPrimAtPath(body).IsValid():
        say(f"skip anchor {key}: {prim} not found")
        continue
    m = xc.GetLocalToWorldTransform(stage.GetPrimAtPath(prim)) * xc.GetLocalToWorldTransform(stage.GetPrimAtPath(body)).GetInverse()
    LOCAL[key] = np.array(m.Transform(Gf.Vec3d(*extra)))

sim = SimulationContext(stage_units_in_meters=1.0)
robot = Articulation(R)
sim.reset()
robot.initialize()
# link poses straight from the articulation's physics view (RigidPrim views on its links upset it)
LINK = {n: i for i, n in enumerate(robot.body_names)}
say(f"{len(LINK)} links; anchors on {sorted({ANCHORS[k][0].split('/')[-1] for k in LOCAL})}")

names = list(robot.dof_names)
hip, thigh, calf = (float(v) for v in args.stance.split(","))
target = np.zeros(len(names), dtype=np.float32)
for i, n in enumerate(names):
    if n.endswith("_hip_joint"):
        target[i] = hip
    elif n.endswith("_thigh_joint"):
        target[i] = thigh
    elif n.endswith("_calf_joint"):
        target[i] = calf
arm = [names.index(f"joint{k}") for k in range(1, 7)]
STOW = np.zeros(6, dtype=np.float32)  # as check_gripper_robot.py, which stands (0.606 m)
POSE = np.array([float(v) for v in args.arm_pose.split(",")], dtype=np.float32)
target[arm] = STOW  # start stowed; the arm is then driven to POSE (teleporting it into place can explode)
jg = names.index("jointGripper")
target[jg] = 0.0
robot.set_joint_positions(target[None, :])
robot.set_joint_velocities(np.zeros((1, len(names)), dtype=np.float32))
robot.set_velocities(np.zeros((1, 6), dtype=np.float32))
legs = [i for i, n in enumerate(names) if n.endswith(("_hip_joint", "_thigh_joint", "_calf_joint"))]
cmd = np.array(legs + arm + [jg])  # wheels left free (position targets on wheels tip the robot)


def hold(steps, render=False):
    for _ in range(steps):
        robot.set_joint_position_targets(target[None, cmd], joint_indices=cmd)
        sim.step(render=render)


for n_ in (5, 25, 70, 200):
    hold(n_)
    say(f"stowed, after {n_} more steps: base height {float(robot.get_world_poses()[0][0][2]):.3f} m")
for k in range(1, 401):  # 2 s, smooth
    a_ = 0.5 - 0.5 * math.cos(math.pi * k / 400)
    target[arm] = STOW + a_ * (POSE - STOW)
    target[jg] = a_ * args.grip
    hold(1)
hold(300)
pos, q = robot.get_world_poses()
say(f"settled: base at {np.round(pos[0], 3).tolist()}, arm {np.round(robot.get_joint_positions()[0][arm], 2).tolist()}")


def quat_rot(qw, v):
    w, x, y, z = qw
    u = np.array([x, y, z])
    return v + 2 * np.cross(u, np.cross(u, v) + w * v)


def anchors_world():
    tf = robot._physics_view.get_link_transforms()  # (1, links, 7): x y z qx qy qz qw
    tf = tf.numpy() if hasattr(tf, "numpy") else np.asarray(tf)
    out = {}
    for key, local in LOCAL.items():
        t = tf[0, LINK[ANCHORS[key][0].split("/")[-1]]]
        out[key] = t[:3] + quat_rot([float(t[6]), float(t[3]), float(t[4]), float(t[5])], local)
    return out


shots = []
WORLD0 = anchors_world()


def point(spec_):
    """'x,y,z' (world) or '@anchor' or '@anchor+dx,dy,dz' (offset in world axes from a sensor/part)."""
    if spec_.startswith("@"):
        key, _, off = spec_[1:].partition("+")
        return WORLD0[key] + (np.array([float(v) for v in off.split(",")]) if off else 0.0)
    return np.array([float(v) for v in spec_.split(",")])


for spec in args.shot:
    name, eye, look, hfov, res = spec.split(":")
    eye, look = point(eye), point(look)
    W, H = (int(v) for v in res.split("x"))
    hfov = float(hfov)
    ha = 20.955
    focal = ha / 2 / math.tan(math.radians(hfov / 2))
    cam = rep.create.camera(position=tuple(eye), look_at=tuple(look), focal_length=focal, horizontal_aperture=ha,
                            clipping_range=(0.01, 100.0))
    rp = rep.create.render_product(cam, (W, H))
    an = rep.AnnotatorRegistry.get_annotator("rgb")
    an.attach(rp)
    shots.append((name, eye, look, hfov, W, H, an))

hold(60, render=True)
world = anchors_world()
os.makedirs(args.out, exist_ok=True)
for name, eye, look, hfov, W, H, an in shots:
    img = np.asarray(an.get_data())[..., :3]
    Image.fromarray(img).save(f"{args.out}/{name}.png")
    f = look - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, [0.0, 0.0, 1.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    fx = W / 2 / math.tan(math.radians(hfov / 2))
    px = {}
    for key, pw in world.items():
        v = pw - eye
        z = float(v @ f)
        if z <= 0:
            continue
        px[key] = [round(W / 2 + fx * float(v @ r) / z, 1), round(H / 2 - fx * float(v @ u) / z, 1)]
    with open(f"{args.out}/{name}.json", "w") as fh:
        json.dump({"size": [W, H], "pixels": px, "world": {k: np.round(v, 4).tolist() for k, v in world.items()},
                   "eye": eye.tolist(), "look": look.tolist(), "hfov": hfov, "arm_pose": args.arm_pose}, fh, indent=1)
    dbg = Image.fromarray(img).copy()
    d = ImageDraw.Draw(dbg)
    for key, (x, y) in px.items():
        d.ellipse([x - 8, y - 8, x + 8, y + 8], outline=(255, 0, 0), width=3)
        d.text((x + 10, y - 10), key, fill=(255, 0, 0))
    dbg.save(f"{args.out}/{name}_debug.png")
    say(f"{name}: {W}x{H} -> {args.out}/{name}.png (+ _debug.png, .json)")
app.close()
