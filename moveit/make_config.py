#!/usr/bin/env python3
"""Make MoveIt's robot description of the Z1 arm + gripper from the URDF the simulator's robot was built from.

    python3 moveit/make_config.py            -> moveit/config/z1.urdf

What changes from robots_usd/z1_gripper_description/urdf/z1_gripper_local.urdf:
- the root is link00 (the arm's base on the B2W): MoveIt plans in the arm base frame, the simulator sends the
  table, the box and the robot's own body in that frame;
- joint limits are the SIMULATOR's (robots_usd/b2w_z1_gripper.usd: narrower than Unitree's URDF, e.g. joint3
  -165..0 deg, not -274..0), so MoveIt never plans a motion the simulated arm cannot follow;
- a fixed `grasp_point` link: the point between the fingertips where the box centre ends up, the same point the
  scripted IK pick uses (fetch_env_cfg.GRASP_POINT: 12 cm along the gripper, 2.6 cm towards the fixed jaw);
- the wrist RealSense D405 on its ring clamp, as three boxes on link06: not in Unitree's URDF, but a collider in
  the simulated robot, 16-22 cm off the wrist axis (it caught a shelf's lip that MoveIt thought the arm cleared);
- mesh paths point into the container (/moveit/meshes, mounted from the description's meshes/).
"""
import math
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "robots_usd/z1_gripper_description/urdf/z1_gripper_local.urdf"
OUT = HERE / "config/z1.urdf"
# the simulator's limits (deg), from b2w_z1_gripper.usd (physics:lowerLimit / upperLimit)
SIM_LIMITS_DEG = {"joint1": (-150, 150), "joint2": (0, 170), "joint3": (-165, 0), "joint4": (-87, 87),
                  "joint5": (-77, 77), "joint6": (-160, 160), "jointGripper": (-90, 0)}
MAX_VEL = math.pi  # rad/s: the USD's maxJointVelocity (180 deg/s); MoveIt scales it down per request
GRASP_POINT = (0.12, 0.0, -0.009 + 0.05 / 2 + 0.010)  # gripperStator frame (fetch_env_cfg.GRASP_POINT)
# (name, centre, size) in link06's frame: bounding boxes of the collider cubes under link06/wrist_d405 in
# robots_usd/b2w_z1_gripper.usd (camera with its cradle; the bracket to the ring; the ring round the flange + ears)
WRIST_D405 = [("d405_camera", (0.017, -0.1875, 0.015), (0.052, 0.059, 0.050)),
              ("d405_bracket", (0.015, -0.101, 0.003), (0.034, 0.132, 0.018)),
              ("d405_ring", (0.0345, 0.0, 0.0045), (0.017, 0.082, 0.091))]


def main():
    if not SRC.exists():
        raise SystemExit(f"{SRC} missing: run ./setup.sh (robot assets) first")
    tree = ET.parse(SRC)
    robot = tree.getroot()
    robot.set("name", "z1")
    for el in list(robot):  # world anchor, transmissions, gazebo tags: not for MoveIt
        if el.tag in ("transmission", "gazebo") or (el.tag == "link" and el.get("name") == "world") or \
                (el.tag == "joint" and el.get("name") == "base_static_joint"):
            robot.remove(el)
    for mesh in robot.iter("mesh"):
        mesh.set("filename", "file:///moveit/meshes/" + Path(mesh.get("filename")).name)
    for j in robot.iter("joint"):
        lim = j.find("limit")
        if j.get("name") in SIM_LIMITS_DEG and lim is not None:
            lo, hi = SIM_LIMITS_DEG[j.get("name")]
            lim.set("lower", f"{math.radians(lo):.6f}")
            lim.set("upper", f"{math.radians(hi):.6f}")
            lim.set("velocity", f"{MAX_VEL:.6f}")
    link06 = next(lk for lk in robot.iter("link") if lk.get("name") == "link06")
    for name, c, s in WRIST_D405:
        col = ET.SubElement(link06, "collision", name=name)
        ET.SubElement(col, "origin", xyz=" ".join(f"{v:.4f}" for v in c), rpy="0 0 0")
        ET.SubElement(ET.SubElement(col, "geometry"), "box", size=" ".join(f"{v:.4f}" for v in s))
    gp_joint = ET.SubElement(robot, "joint", name="grasp_point_joint", type="fixed")
    ET.SubElement(gp_joint, "origin", xyz=" ".join(f"{v:.4f}" for v in GRASP_POINT), rpy="0 0 0")
    ET.SubElement(gp_joint, "parent", link="gripperStator")
    ET.SubElement(gp_joint, "child", link="grasp_point")
    ET.SubElement(robot, "link", name="grasp_point")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(OUT, encoding="unicode", xml_declaration=True)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
