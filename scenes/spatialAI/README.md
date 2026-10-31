# SpatialAI scenes

One folder per scene. Each folder holds what SpatialAI gave for that room, the Isaac Sim stage built from it, and
its checks. The build script next to this README is shared by all of them.

| Scene | Open in Isaac Sim | Details |
|---|---|---|
| bedroom | `bedroom/spatialAI_bedroom.usdz` | [bedroom/README.md](bedroom/README.md) |
| new_office | `new_office_scene/spatialAI_new_office.usdz` | [new_office_scene/README.md](new_office_scene/README.md); B2W + Z1 drive video in `new_office_scene/robot/` |

**Not in git:** the SpatialAI inputs (`*.ply`, `*.usdc`), the built stages (`*.usdz`) and the drive videos (`*.mp4`)
are 10-490 MB each (`.gitignore` here). This folder holds the build script, the per-scene READMEs, transforms,
checks and routes; put the two SpatialAI files back into a scene folder and rebuild (step 3 below) to get its stage.

**File > Open** a stage and it is ready. **Dragging** one into another stage (e.g. one with your robot) adds it as a
reference, and Isaac then skips the NuRec render settings stored in it, so the splat looks washed out. Run
`apply_splat_render_settings.py` once in Window > Script Editor to fix that. Checked on new_office: it then matches
File > Open to within 0.3/255.

```
spatialAI/
  README.md                  this file
  build_spatialai_stage.py   builds the stage of one scene folder
  apply_splat_render_settings.py   Script Editor fix after dragging a stage into another one
  bedroom/
    spatialAi_splat.ply                 input: splat from SpatialAI
    spatialAi_mesh.usdc                 input: mesh from SpatialAI, via Blender
    spatialAI_bedroom.usdz              the stage (open this)
    spatialAI_bedroom_check.usda        alignment check wrapper
    spatialAI_bedroom_transform.json    transforms + fit numbers
    check/                              Isaac renders, drop test, views.json
    README.md                           what was done for this scene, and the results
  new_office_scene/                     same layout (stage: spatialAI_new_office.usdz)
    robot/                              B2W + Z1 loop: videos, route, ground wrapper for Isaac Lab
```

## Adding a scene

1. Make a folder named after the room, e.g. `kitchen/`. The name ends up in the output file names.
2. Put the two SpatialAI files in it: the splat `.ply` and the mesh `.usdc` (exported from Blender). Any file names
   work, as long as there is exactly one of each. A mesh rotated or moved in Blender is fine: its object transform
   is baked in.
3. Build it (about 4 min):

   ```
   cd <repository>/scenes/spatialAI
   <paths.threedgrut>/.venv/bin/python build_spatialai_stage.py kitchen
   ```

   This writes `spatialAI_kitchen.usdz`, `spatialAI_kitchen_check.usda`, `spatialAI_kitchen_transform.json`,
   `check/views.json` and `check/up_preview.jpg` into `kitchen/`. Options: `--no-yaw`, `--keep-origin`, `--out`,
   `--up-checked`, `--help` for the rest. A big scene (new_office, 8.8M gaussians) takes about 8 min.
4. Optional Isaac Sim check. `check/renders` must be writable by uid 1234:

   ```
   mkdir -p kitchen/check/renders && chmod 777 kitchen/check/renders
   ../../isaac/run_render_views.sh /workspace/scenes/spatialAI/kitchen/spatialAI_kitchen.usdz /workspace/scenes/spatialAI/kitchen/check/views.json /workspace/scenes/spatialAI/kitchen/check/renders "--splat-prim /World/volume/volume --collider-prim /World/volume/mesh"
   ```

5. Add a row for it to the table above.

The script checks every capture before it builds (that the mesh fits the splat, which way is up, the floor fit). If a
new capture does not pass one of these, it stops with a message instead of writing a wrong stage.

The "which way is up" check is a heuristic: furniture tops should make the lower half of the room hold more horizontal
area than the upper half. It was wrong for new_office, where a lowered ceiling section outweighs the few desks. If it
stops there, open `check/up_preview.jpg` (a point render from the capture start, 4 headings). If the ceiling is on top
and the furniture stands on the floor, rerun with `--up-checked`.
