# SpatialAI (Echo-3) new office: splat + mesh in one Isaac Sim stage

**Use `spatialAI_new_office.usdz`** (179 MB, one self-contained file) in Isaac Sim 5.1. You see the splat; the
SpatialAI mesh is under it as an invisible collider.

- **File > Open** it: ready as it is.
- **Drag it into another stage** (e.g. one with your robot): it comes in as a reference under `/World`. Then run
  `../apply_splat_render_settings.py` once in Window > Script Editor. Without it the splat looks washed out, because
  Isaac only applies render settings (here: the NuRec tone mapping) from the stage's own root layer. With it, the
  render matches File > Open to within 0.3/255 on average (`check/isaac_dragdrop_vs_open.jpg`). The prim's translate
  should be (0, 0, 0) for the floor to sit at z = 0.

| File | What it is |
|---|---|
| `new_office.ply`, `new_office_mesh_simp.usdc` | inputs, untouched (splat from SpatialAI, simplified mesh via Blender) |
| `textures/` | Blender's environment-light texture that came with the mesh export; not used |
| `spatialAI_new_office.usdz` | **the stage**: NuRec splat + mesh collider + floor collider + dome light |
| `spatialAI_new_office_check.usda` | 1 KB wrapper around the .usdz: the mesh is visible in its vertex colours, to check the fit |
| `spatialAI_new_office_transform.json` | 4x4s from the SpatialAI splat / mesh frames to the stage frame, plus every fit number below |
| `../build_spatialai_stage.py` | rebuilds all of the above (about 8 min); shared by all scenes in `spatialAI/` |
| `check/` | Isaac Sim renders, drop test, alignment and drag-and-drop comparisons, the up preview |
| `robot/` | the B2W + Z1 driving a loop: **`b2w_z1_office_side_by_side.mp4`**, plus the route and the ground wrapper (see below) |

## Stage frame

Z-up, metres, floor at z = 0, the main corridors along X/Y. The origin is the most open spot on floor that was
actually reconstructed: a wide part of the corridor ring with high tables, 1.60 m from the nearest obstacle, so a
robot can be spawned at (0, 0). The building box is x -20.0..9.7, y -5.0..14.1. Ceiling at 2.82 m.

The capture is a ring of corridors with side rooms, around a 78 m^2 block of rooms that was never entered. The floor
box covers that block too, but there is nothing to see there.

```
/World                  (defaultPrim)
  volume/volume         NuRec Volume (the splat, 8,751,250 gaussians), proxy -> /World/volume/mesh
  volume/mesh           SpatialAI mesh: invisible, matte, triangle-mesh collider (approximation none), vertex colours kept
  volume/floor          invisible box collider, top at z = 0, under the whole building box
  dome_light            1000, hidden from camera rays (lights robots/objects you add)
```

## What did not line up, and what was done

1. **The mesh had a Blender transform.** The mesh prim carries a +90 deg rotation about X (it was stood upright in
   Blender). The builder used to refuse that; it now bakes the transform into the points. After that, the mesh is
   the splat turned from Y-up to Z-up. The 24-rotation search finds this with a 1.6 cm median gap, against 67 cm for
   the next best rotation. ICP on top of it moves nothing (scale 1.0002, 0.007 deg, 0.4 mm).
2. **The fit check was too coarse for a big scene.** At 865 m^2 of mesh, the builder's fixed 200k surface samples
   were 6.6 cm apart, and that alone pushed the measured gap to 3.6 cm, over the 3 cm limit. The sample count now
   scales with the mesh area.
3. **The which-way-is-up heuristic was wrong here.** A lowered ceiling section (15.5 m^2) outweighs the few desks, so
   "more horizontal area in the lower half" failed (8.6 vs 21.9 m^2). A point render from the capture start
   (`check/up_preview.jpg`) shows the ceiling grid on top and the tables standing on the floor. The floor (143 m^2)
   and ceiling normals agree. Built with `--up-checked`; every build now writes that preview.
4. **Spawn point in a room that was never entered.** Filling holes in the floor map also filled the unscanned block
   inside the corridor ring, and the "most open spot" landed in the middle of it, with no floor but the box. Spawn
   and drop points now come only from reconstructed floor (172 of the 272 m^2 footprint).
5. **Splat values.** 4,194,964 gaussians had opacity exactly 1 (logit +inf); they were clamped to logit(0.9999). 40
   had opacity exactly 0 (logit -inf), which the bedroom did not have; they add nothing and were dropped. The build
   now reads the packaged `.nurec` back and fails if any value is not finite.
6. **Levelled and squared.** The floor was tilted 0.21 deg (fit rms 0.7 cm). The walls were turned by 86.9 deg. Not
   all corridors are square to each other: one long corridor and the wing off it run about 14 deg off the axes.
7. No floor pits (the bedroom had reflection pits under a glossy floor; this carpet has none). The floor box stays as
   a safety net.
8. Blender leftovers (Light, Camera, env_light and its texture) are not carried over.

## Checks (29 Oct 2026)

- Gaussian centres to mesh surface, in the stage frame: median 1.33 cm, p90 3.82 cm.
- Splat rotations, read back from the `.nurec` inside the .usdz: the flat splats lie in the mesh surface. Median
  |cos(thin axis, mesh normal)| is 0.968; leaving the quaternions un-rotated gives 0.164. All values are finite.
- USD: no composition errors, the payload resolves inside the package, the proxy is wired, the volume transform is
  identity.
- Isaac Sim 5.1, headless, opening the .usdz directly: exit 0, no errors.
  - Six views render photoreal and upright (`check/isaac_splat_views.jpg`).
  - The collider's edges fall on the splat: corridor corners, door frames, the high-table top, the chairs
    (`check/isaac_alignment_edges.jpg`).
- Drop test: 6/6 balls (r = 0.10 m), spread over the whole building up to 20 m from the origin, come to rest with
  their centre 0.100-0.112 m above z = 0.
- Referenced into a plain stage, as a drag and drop does: renders, and the balls land the same. Colours are only
  right after `apply_splat_render_settings.py` (`check/isaac_dragdrop_vs_open.jpg`).

The mesh is simplified (174k triangles for the whole building), so thin things such as chair legs and table poles are
coarse in the collider. A robot drive is in the next section.

## B2W + Z1 driving one loop of the corridor ring (29 Oct)

**Video:** `robot/b2w_z1_office_side_by_side.mp4`, 1920 x 1080, 25 fps, 108 s:

```
+----------------------+----------------------+
|                      | front camera         |
|  third-person view   | (D455 on the head)   |
|  (+ minimap)         +----------------------+
|                      | end-effector camera  |
|                      | (D405 on the wrist)  |
+----------------------+----------------------+
```

The three views are also there on their own (`b2w_z1_office_{chase,front,wrist}.mp4`), with `b2w_z1_office_track.json`
(pose, command, cross-track error, contact force, every 20 ms) and a still. These are H.264 re-encodes of the renders
(CRF 20, 45 dB PSNR against the originals, which were deleted to save space).

**Result:** the robot drives the whole 46.0 m loop in 106 s and stops where it started, with no recoveries.
- Speed: up to 0.6 m/s, mean 0.44 m/s. It slows to about 0.2 m/s at the three tight spots, where the collider leaves 0.40 m
  from the route to the nearest obstacle.
- At most 4.5 cm from the route (95% within 3 cm). Tilt at most 3.2 deg.
- Nothing but the wheels touches the scene. The only contact is the arm on itself: link02 taps link06 on 31 steps (up to 54 N). This is the known
  property of the stow pose `ARM_STOW` while driving (the M1 loop had 63 such taps).

**Setup** (`project/scene_tour.py --robot b2w_z1 --layout three`, the same tool and robot as the M1 loop in
`scenes/nurec/mine/record3d_2laps`):
- **Robot:** `robots_usd/b2w_z1_gripper.usd`, the fetch robot. It is walked by the stance fine-tune of ETH's policy
  (`eth/policy_stance_v5_600.pt`). The arm is held stowed and the gripper closed.
- **Route:** `robot/route.json` / `route.png`, from `reconstruction_with_mesh/tools/robot_route.py`.
  - The planner works on `robot/collider_stage.ply`, the stage's collider mesh written out in the stage frame.
  - Obstacles are 0.12–1.2 m high (the arm reaches 1.05 m).
  - The waypoints are (0, 0), (-15, -1), (-15.5, 6.2) and (-1, 9.4): clockwise round the ring, through the kitchenette
    corridor and the long angled one.
- **Ground:** `robot/new_office_robot_ground.usda`. It is the stage plus a friction-1.0 material on its mesh and floor
  colliders, because Isaac Lab does not apply its terrain material to USD terrain.
- **Cameras:**
  - The third-person camera rides on the route up to 2.4 m behind the robot, 1.5 m high, 65 deg wide, at 960 x 1080.
  - The front camera is the recordings' D455 lens at the head. It is 90 deg wide, cut to 16:9.
  - The end-effector camera is the asset's own `end_effector_camera`: the D405 in its holder on link06, 87 x 56 deg.
    With the arm stowed it looks forward past the gripper jaw and the lidar, so those fill its lower left.
- **Colours:** the task references the stage, so `scene_tour.py` now copies the stage's NuRec render settings into the
  renderer itself. That is the same fix as `../apply_splat_render_settings.py`; without it the splat would look washed out.

**What to know when watching:**
- **The robot casts no shadow on the splat**, so it can look as if it floats a little. This is how NVIDIA's NuRec layout behaves (the
  proxy mesh is invisible), and two ways round it failed:
  - Showing the proxy mesh as a matte object: its ceiling then blocks the dome light, and the robot renders black.
  - Showing the floor box as a matte shadow catcher: it covers the splat floor with dark speckle.
- **The door frame at about 80–84 s:** the third-person camera passes close to an open glass door, and its left edge
  shows the frame blurred for about 3 s.

**Re-run** (about 7 min on the Ada card; rendering is 0.09 s per frame):

```
cd <repository>
S=/workspace/scenes/spatialAI/new_office_scene
CONTAINER_NAME=isaac-office-tour ./isaac/run_script.sh project/scene_tour.py --device cpu --robot b2w_z1 \
    --layout three --arm_partners --stage $S/robot/new_office_robot_ground.usda --route $S/robot/route.json \
    --out /workspace/renders/office_tour/<name>
```

The output folder is written by uid 1234; delete it through the image (`docker run --rm -v ...:/d --entrypoint bash
isaac-lab:2.3.2 -c "rm -rf /d/<name>"`). To re-plan the route:
`<paths.threedgrut>/.venv/bin/python reconstruction_with_mesh/tools/robot_route.py robot/collider_stage.ply robot/route.json --zmax 1.2 --closed --waypoints "0,0; -15,-1; -15.5,6.2; -1,9.4"`.

## Notes

- To bring other SpatialAI data (camera poses, points) into the stage, use `splat_to_stage` from the JSON:
  `p_stage = M @ [p, 1]`. `mesh_to_stage` applies to the mesh points with its Blender transform already applied
  (`mesh_object_transform_baked` in the JSON).
- SH degree 0 means no view-dependent effects: reflections are baked into the colours.
- Rebuild, from `scenes/spatialAI`:
  `<paths.threedgrut>/.venv/bin/python build_spatialai_stage.py new_office_scene --out new_office_scene/spatialAI_new_office.usdz --up-checked`
  - Re-run the Isaac check with
    (from the repository root) `./isaac/run_render_views.sh /workspace/scenes/spatialAI/new_office_scene/spatialAI_new_office.usdz /workspace/scenes/spatialAI/new_office_scene/check/views.json /workspace/scenes/spatialAI/new_office_scene/check/renders "--splat-prim /World/volume/volume --collider-prim /World/volume/mesh"`.
    `check/renders` must be writable by uid 1234.
