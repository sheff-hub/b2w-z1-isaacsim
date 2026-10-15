"""Make z1_gripper_local.urdf from Unitree's z1_gripper.urdf so it imports correctly.

Two defects in the shipped URDF:
1. Meshes are referenced as package://z1_description/..., which the Isaac Sim
   importer cannot resolve here and silently skips -> invisible robot.
2. Neither jaw can touch anything: gripperMover has no <collision> at all, and
   gripperStator's collider is link00's base cylinder (r 3.25 cm, centred on the
   link origin) while its fixed jaw reaches x = 0.149 m.

Fix: relative mesh paths, and each jaw's visual mesh reused as its collider.
Import with convex decomposition (tools/convert_z1_gripper.py) so the colliders
follow the finger shapes instead of a hull that would fill the gap between them.

    python3 tools/make_z1_gripper_urdf.py
"""

import os
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
URDF_DIR = os.path.join(HERE, "..", "robots_usd", "z1_gripper_description", "urdf")
SRC = os.path.join(URDF_DIR, "z1_gripper.urdf")
DST = os.path.join(URDF_DIR, "z1_gripper_local.urdf")

JAW_MESHES = {"gripperStator": "z1_GripperStator.dae", "gripperMover": "z1_GripperMover.dae"}

tree = ET.parse(SRC)
root = tree.getroot()

for mesh in root.iter("mesh"):
    mesh.set("filename", mesh.get("filename").replace("package://z1_description/meshes/", "../meshes/"))

for link in root.findall("link"):
    name = link.get("name")
    if name not in JAW_MESHES:
        continue
    for c in link.findall("collision"):
        link.remove(c)
    coll = ET.Element("collision")
    ET.SubElement(coll, "origin", rpy="0 0 0", xyz="0 0 0")
    geom = ET.SubElement(coll, "geometry")
    ET.SubElement(geom, "mesh", filename=f"../meshes/{JAW_MESHES[name]}", scale="1 1 1")
    # keep URDF element order visual -> collision -> inertial
    link.insert(list(link).index(link.find("visual")) + 1, coll)

tree.write(DST, xml_declaration=True, encoding="utf-8")
unresolved = sum("package://" in m.get("filename") for m in root.iter("mesh"))
print(f"wrote {DST}; package:// left: {unresolved}; jaw colliders: {list(JAW_MESHES)}")
