# scenes/: what is where

A scene file describes one environment for the fetch tasks: the stage, the tables and boxes, the Nav2 map, where the
robot starts and where it delivers. The Isaac Lab task (`training/.../fetch_env_cfg.py`) builds the scene from it,
and `project/live.sh`, `run.py` and `project/fetch_box.py` read it, so moving a table or swapping the building needs
no code change.

| File | Scene | Task | Nav2 map |
|---|---|---|---|
| `warehouse_one_table.yaml` | S1: NVIDIA's warehouse, one wooden table, one box | `RobotLab-Isaac-Fetch-Warehouse-Unitree-B2W-Z1-Gripper-v0` | `warehouse` |
| `warehouse_three_tables.yaml` | S2/S3: red, yellow and green table, a red, yellow and green box on each | `RobotLab-Isaac-Fetch3-Warehouse-Unitree-B2W-Z1-Gripper-v0` | `warehouse3` |
| `hospital_reception.yaml` | NVIDIA's hospital, the reception: a red, a yellow and a green box on a tray of the medicine shelf behind the desk; the robot takes the named one out and puts it on the side table (layout `shelf`) | `RobotLab-Isaac-Shelf-Hospital-Unitree-B2W-Z1-Gripper-v0` | `hospital_reception` |
| `hospital_hall.yaml` + `.usda` | The reception's shelf and side table moved into the open hall in front of the robot (cleared of their bottles and trays, the wheelchair removed), the boxes on the shelf's side facing the robot: a green box on the low tray, a yellow one at 0.9 m, a red one above; the robot crouches for green, stands taller for red. "Take the green box from the shelf and put it on the table"; DualVLN walks to the shelf | same task (`FETCH_SCENE` picks the file) | `hospital_hall` |
| `hospital_doors.yaml` + `.usda` | The hospital's east corridor: three hinged doors with lever handles (`assets/door_hinged.usda`), one painted blue at random; the robot walks to the blue one, turns its handle, pushes it open and goes into the room | `RobotLab-Isaac-Door-Hospital-Unitree-B2W-Z1-Gripper-v0` | `hospital_doors` |

`spatialAI/` is not a fetch scene: it builds Isaac stages (splat + invisible collider) of rooms captured with
SpatialAI, a bedroom and the new office; see [spatialAI/README.md](spatialAI/README.md).

## Format

```yaml
name: my_scene
layout: multi_table            # one_table | multi_table: which scene class builds it
task: RobotLab-Isaac-Fetch3-Warehouse-Unitree-B2W-Z1-Gripper-v0
environment_usd: <any Isaac Sim stage: a URL or /workspace/... path>
nav2_map: my_map               # nav2/maps/my_map.{pgm,yaml}
robot:
  spawn: [x, y]                # where the task spawns the robot (facing +x)
  live_start: [x, y, yaw_deg]  # where live.sh starts
  demo_starts: [[x, y, yaw], ...]
  random_start: [x0, x1, y0, y1]   # run.py fetch: a random start in this rectangle, any heading
table: {size: [depth, width], height: 0.60, top_thickness: 0.04, leg: 0.04}
tables:                        # name -> centre and colour; the name is what a task says ("the red table")
  red: {center: [x, y], rgb: [r, g, b]}
box: {size: [x, y, z], mass: 0.2, from_edge: 0.10, slots_y: [-0.3, 0.0, 0.3],
      colors: {red: [r, g, b], ...}}
drop_zone: [x, y]
```

### Layout `shelf`: take a box from a shelf, put it on a table

Several shelves and tables: `shelves:` and `tables:` (name -> entry; its `label` is what a task says: "the shelf"),
what all shelves share in `shelf_defaults:` (`hospital_hall.yaml`). A shelf with `levels:` holds one box per tray,
each with its own pick settings: `top` (the tray), `stance` (ETH's body height: 1 = 6 cm crouched, 0 = default,
-0.75 = 1.8 cm taller -- the most the policy gives), `gap`, `pitch`, `jaw_flip`, `yaw_jitter`, `retreat`, and
`place_depth` (a box grasped nearly level from a high tray goes down this far onto the table: at the edge the arm
cannot hold the hand that level; the place also turns the hand about the vertical if it must, `PLACE_TRIES` in
`fetch_box.py`). `python3 moveit/shelf_reach.py <scene> --level <colour>` checks a tray. A file with one `shelf:` and one
`place:` (`hospital_reception.yaml`, below) is read as one of each. Furniture that is not in the building can come
from a USD layer on top of it, like `hospital_hall.usda`: it references the building, switches prims off
(`active = false`) and moves its props (an `over` with a new translate and orient).

The building's own furniture (static colliders in the environment USD), so the task adds only the boxes.
`hospital_reception.yaml` is the example; its numbers were measured from hospital.usd.

```yaml
layout: shelf
nav2_area: [x0, x1, y0, y1]    # map only this part of a big building (nav2_probe.py --area)
nav2_staging: 3.0              # Nav2's staging pose this far before the docking point (default 2.1)
shelf:                         # where the boxes stand
  near_x: -0.724               # the shelf's open face (x); face: -1 = the robot works from its -x side, +1 from +x
  face: -1
  top: 0.878                   # the tray the boxes stand on
  box_x: -0.624                # box centres: depth...
  slots_y: [4.55, 4.73, 4.91]  # ...and positions along the shelf (colours shuffled per reset)
  sides: [right]               # which side of the robot the shelf may be on when it docks (right, left)
  abeam: 0.15                  # the box this far ahead of the robot's centre when docked
  gap: 0.09                    # robot's hips to the face
  pitch: [45, 50, 40]          # grasp approach below horizontal, tried in turn
  jaw_flip: true               # gripper rolled 180 deg: the wrist RealSense hangs down, clear of the tray above
  pre_dist: 0.16               # pre-grasp distance along the approach
  lift: 0.05                   # lift before pulling the box out (over the tray's lip)...
  retreat: 0.20                # ...then straight out
  crouch: false
  carry: [0.0, 0.6, -1.0, 1.0, 0.0]    # carry pose (joints 1-5, joint1 relative to the arm's side)
  collision: {name: [x0, y0, z0, x1, y1, z1], ...}   # boxes for MoveIt's planning scene (world frame)
place:                         # where the box goes: same keys for docking, and
  spot: [x, y]                 # the box centre on the table
  rect: [x0, x1, y0, y1]       # the table top (a placed box must end inside)
  collision: {...}             # the table and what stands on it
obstacles: {name: [x0, x1, y0, y1], ...}   # footprints for the scripted fallback drive and random starts
camera: {fixed: {eye, lookat}, follow: {eye, lookat}}   # fetch_box.py --cam scene / scene_follow
```

How the numbers were found (and what to check in a new building) is in `moveit/README.md`, "Shelves".

Tables are approached from -x (their long side faces the robot), like in the warehouse. `table` and `box` geometry
must be the same in all scene files: the pick's docking and grasp geometry uses one set (the task refuses files
that differ).

### Layout `doors`: open a door and go in

`hospital_doors.yaml`: `doors:` (name -> the asset's origin in its opening -- the leaf's corridor face, its hinge
edge, the floor -- and the `room` behind it) and `door:` what they share: the leaf, hinge and spindle in the asset's
frame, where the jaws close on the lever (`grip`), how far it is turned (`handle_press`) and how far before the
latch lets go (`latch_release`), the hinge's damping and latch stiffness, the lever's spring, and the colours
(`target_color` is the one the task names). `dock:` is where the robot parks to open a door. The doors are
articulations spawned by the task (fetch_env_cfg `_door_cfg`), not part of the building: the layer switches the
building's static doors off. `tools/build_door_asset.py` makes the asset from the hospital's own SM_Door_01;
`moveit/door_reach.py` tells from where, and at what pitch, the arm can grasp the lever.

## A new scene

1. Copy a file here and edit it. Keep the file inside the repository: the container only sees the repository.
2. Build its Nav2 map from the colliders and name it in the file:
   `./isaac/run_script.sh project/nav2_probe.py --task <its task> --map my_map` (writes `nav2/maps/my_map.*`).
3. Run it: `python3 run.py fetch --scene my_scene`, or `SCENE=my_scene NAV2=1 ./project/live.sh fetch`.

`FETCH_SCENE=<file>` tells the task which scene file to build (run.py and live.sh set it). A room reconstructed
from photos (`reconstruction_no_mesh/`) has no collision geometry: use it for the look, not as `environment_usd` to
drive in. One recorded with an iPhone's LiDAR (`reconstruction_with_mesh/`) has an invisible collider: its
`isaac/<name>_robot.usda` carries the robot (tested driving a route); as a fetch scene's `environment_usd`, with a
Nav2 map and tables, it is not tested yet.

`nurec/` (not in git) is where NuRec scene assets go when `paths.nurec` in `config.yaml` points here.
