"""Stage 9 - the Isaac Sim stages: splat + invisible collider, laid out like NVIDIA's own NuRec scenes.

  isaac/<name>.usda           splat (by reference to export/scene.usdz) + the LiDAR collider, invisible: the volume's
                              `proxy` targets it, it is a matte object, and a dome light lights inserted assets.
                              Frame: metres, Z-up, floor at z = 0 (the phone's frame after prep).
  isaac/<name>_visible.usda   the same with the collider drawn (orange), to check alignment
  isaac/<name>_robot.usda     <name>.usda + a friction-1.0 physics material on the collider. Use this one as the
                              ground of a robot: Isaac Lab applies its terrain material only to its own terrains,
                              so a USD scene used as terrain otherwise has the default material.
The stages reference ../export/scene.usdz: move the scene folder as a whole.
"""
from __future__ import annotations

import json
from pathlib import Path

from .common import Log, tool

# The data is already metric, Z-up, floor at 0; this undoes the NuRec volume's axis flip (B = rot^T), see make_stage.py.
LEVEL_IDENTITY = {"level_frame": {"e1": [-1, 0, 0], "e2": [0, 0, -1], "up": [0, -1, 0], "origin": [0, 0, 0]},
                  "note": "data frame is already metric Z-up with floor at 0; this undoes the NuRec volume flip"}

ROBOT_WRAPPER = '''#usda 1.0
(
    defaultPrim = "World"
    doc = """{name} as robot ground: {name}.usda + a friction-1.0 physics material on the collider.
Isaac Lab binds its terrain material only to terrains it builds itself, so a USD scene used as terrain
(terrain_type="usd") would otherwise have PhysX's default material."""
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World" (
    prepend references = @./{name}.usda@
)
{{
    over "room"
    {{
        over "collision" (
            prepend apiSchemas = ["MaterialBindingAPI"]
        )
        {{
            rel material:binding:physics = </World/ground_material> (
                bindMaterialAs = "weakerThanDescendants"
            )
        }}
    }}

    def Material "ground_material" (
        prepend apiSchemas = ["PhysicsMaterialAPI"]
    )
    {{
        float physics:dynamicFriction = {friction}
        float physics:restitution = 0
        float physics:staticFriction = {friction}
    }}
}}
'''


def run_stage(cfg: dict, scene: Path, name: str, log: Log, friction: float = 1.0) -> dict:
    log.rule("STAGE 9 - Isaac Sim stages (splat + collider)")
    isaac = scene / "isaac"
    isaac.mkdir(exist_ok=True)
    (isaac / "level.json").write_text(json.dumps(LEVEL_IDENTITY, indent=1))
    out = {}
    for visible in (False, True):
        st = isaac / f"{name}{'_visible' if visible else ''}.usda"
        tool(cfg, "make_stage.py", ["--splat", "../export/scene.usdz", "--mesh", "../mesh/collider.ply", "--out", st.name,
                                    "--level", "level.json", "--nvidia-layout"] + (["--visible"] if visible else []), log,
             cwd=str(isaac))
        out["visible_usda" if visible else "scene_usda"] = str(st)
    rb = isaac / f"{name}_robot.usda"
    rb.write_text(ROBOT_WRAPPER.format(name=name, friction=friction))
    out["robot_usda"] = str(rb)
    for k, v in out.items():
        log(f"{k:<14}: {v}")
    return out
