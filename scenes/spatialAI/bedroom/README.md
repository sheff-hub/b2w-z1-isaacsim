# SpatialAI (Echo-3) bedroom: splat + mesh in one Isaac Sim stage

**Open `spatialAI_bedroom.usdz`** (110 MB, one self-contained file) in Isaac Sim 5.1: File > Open, or add it as a
reference. You see the splat; the SpatialAI mesh is under it as an invisible collider. As a reference (or dragged in),
run `../apply_splat_render_settings.py` in the Script Editor, or the splat looks washed out (see `../README.md`).

| File | What it is |
|---|---|
| `spatialAi_splat.ply`, `spatialAi_mesh.usdc` | inputs, untouched (splat from SpatialAI, mesh via Blender) |
| `spatialAI_bedroom.usdz` | **the stage**: NuRec splat + mesh collider + floor collider + dome light |
| `spatialAI_bedroom_check.usda` | 1 KB wrapper around the .usdz: the mesh is visible in its vertex colours, so you can check the fit (toggle `/World/volume/volume` to hide the splat; add a light, the closed room keeps the dome light out) |
| `spatialAI_bedroom_transform.json` | 4x4s from the SpatialAI splat / mesh frames to the stage frame, plus every fit number below |
| `../build_spatialai_stage.py` | rebuilds all of the above (about 4 min); shared by all scenes in `spatialAI/` |
| `check/` | Isaac Sim renders and the drop test (`views.json` drives (from the repository root) `./isaac/run_render_views.sh`) |

## Stage frame

Z-up, metres, floor at z = 0, walls along X/Y (long side along X), origin = the most open floor spot
(0.70 m to the nearest obstacle, so a robot can be spawned at (0, 0)). The room box is x -2.9..3.8, y -2.1..1.7. The
main ceiling is at 2.60 m.

```
/World                  (defaultPrim)
  volume/volume         NuRec Volume (the splat), proxy -> /World/volume/mesh
  volume/mesh           SpatialAI mesh: invisible, matte, triangle-mesh collider (approximation none), vertex colours kept
  volume/floor          invisible box collider, top at z = 0, under the whole room
  dome_light            1000, hidden from camera rays (lights robots/objects you add)
```
Inside the package: `default.usda` (root), `volume.usda` + `volume.nurec` (3DGRUT NuRec export), `mesh.usdc`. The
layout is the one NVIDIA's NuRec `stage.usdz` scenes use.

## What did not line up, and what was done

1. **Mesh vs splat orientation.** The splat PLY is Y-up (glTF convention). The Blender mesh is the splat turned
   180 deg about X (y and z negated), so in Blender's Z-up stage it is upside down, with height along -Y. Found
   by trying all 24 axis rotations: this flip leaves a 1.7 cm median gap, against 15.8 cm for the next best. ICP
   on top of it moves nothing (scale 1.0004, 0.016 deg, 1.2 mm), so scale and position already agreed.
2. **Which way is up** was checked by rendering the splat both ways (+Y up shows the bed, door and pictures
   upright). The script re-checks it on every run: furniture tops face the camera, so the lower half of the room
   holds more horizontal area (8.4 vs 2.0 m^2).
3. **Levelled and squared.** The floor plane (fit rms 0.7 cm) was tilted 0.85 deg. The walls were 22.9 deg off
   the axes.
4. **Splat values.** 1,011,357 gaussians had opacity exactly 1, stored as logit +inf. NuRec stores float16
   pre-activation values, so they were clamped to logit(0.9999). The positions and rotations of all 4,434,931
   gaussians were baked into the stage frame, so the volume transform is identity.
5. **Floor pits.** The floor is glossy, so the reconstruction put reflection "bubbles" 10-30 cm under it. Where
   the real floor surface sits above them they are harmless. In the dark closet corner (0.71 m^2, 26 cm deep), and
   in 19 tiny spots, only the bubble is left. The flat floor box covers them; the mesh itself is unchanged.
6. **Blender leftovers** (default Light, Camera, env_light) are not carried over.
7. **Size.** 3DGRUT writes the `.nurec` gzip stream at compression level 0. Recompressing it took the file from
   546 to 110 MB with the data unchanged. Half of the raw payload is the zero SH bands 1-3, because the PLY has SH
   degree 0 only.

## Checks (28 Oct 2026)

- Gaussian centres to mesh surface, in the stage frame: median 1.42 cm, p90 2.65 cm.
- Splat rotations: the flat (2D) splats lie in the mesh surface. Median |cos(thin axis, mesh normal)| is 0.96;
  leaving the quaternions un-rotated gives 0.24. This was read back from the `.nurec` inside the .usdz; all values
  are finite.
- USD: no composition errors, the payload resolves inside the package, and the proxy is wired.
- Isaac Sim 5.1, headless, opening the .usdz directly. Exit 0, no errors.
  - Six views render photoreal and upright (`check/isaac_splat_views.jpg`).
  - The splat's edges fall on the mesh's edges: bed, wardrobe, door frame and room corners
    (`check/isaac_alignment_edges.jpg`).
- Drop test: 6/6 balls (r = 0.10 m) come to rest with their centre 0.100-0.104 m above z = 0. That includes one
  dropped into the closet pit.
- The check wrapper renders the mesh with the splat and without it (`check/isaac_check_wrapper.jpg`).

Not tested: a robot driving in this stage (only balls were dropped).

## Notes

- To bring other SpatialAI data (camera poses, points) into the stage, use `splat_to_stage` from the JSON:
  `p_stage = M @ [p, 1]`. For anything in the Blender mesh frame use `mesh_to_stage`.
- SH degree 0 means no view-dependent effects: floor reflections are baked into the colours.
- Rebuild, from `scenes/spatialAI`: `<paths.threedgrut>/.venv/bin/python build_spatialai_stage.py bedroom`
  - `--no-yaw` keeps SpatialAI's heading.
  - `--keep-origin` keeps SpatialAI's origin, the capture start, which lies 1.31 m above the floor at stage
    (0.41, -0.32).
  - Re-run the Isaac check with
    (from the repository root) `./isaac/run_render_views.sh /workspace/scenes/spatialAI/bedroom/spatialAI_bedroom.usdz /workspace/scenes/spatialAI/bedroom/check/views.json /workspace/scenes/spatialAI/bedroom/check/renders "--splat-prim /World/volume/volume --collider-prim /World/volume/mesh"`.
    `check/renders` must be writable by uid 1234.
