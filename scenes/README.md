# scenes/: what is where

A scene file describes one environment for the fetch tasks: the stage, the tables and boxes, the Nav2 map, where the
robot starts and where it delivers. The Isaac Lab task (`training/.../fetch_env_cfg.py`) builds the scene from it,
and `project/live.sh`, `run.py` and `project/fetch_box.py` read it, so moving a table or swapping the building needs
no code change.

| File | Scene | Task | Nav2 map |
|---|---|---|---|
| `warehouse_one_table.yaml` | S1: NVIDIA's warehouse, one wooden table, one box | `RobotLab-Isaac-Fetch-Warehouse-Unitree-B2W-Z1-Gripper-v0` | `warehouse` |
| `warehouse_three_tables.yaml` | S2/S3: red, yellow and green table, a red, yellow and green box on each | `RobotLab-Isaac-Fetch3-Warehouse-Unitree-B2W-Z1-Gripper-v0` | `warehouse3` |

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

Tables are approached from -x (their long side faces the robot), like in the warehouse. `table` and `box` geometry
must be the same in all scene files: the pick's docking and grasp geometry uses one set (the task refuses files
that differ).

## A new scene

1. Copy a file here and edit it. Keep the file inside the repository: the container only sees the repository.
2. Build its Nav2 map from the colliders and name it in the file:
   `./isaac/run_script.sh project/nav2_probe.py --task <its task> --map my_map` (writes `nav2/maps/my_map.*`).
3. Run it: `python3 run.py fetch --scene my_scene`, or `SCENE=my_scene NAV2=1 ./project/live.sh fetch`.

`FETCH_SCENE=<file>` tells the task which scene file to build (run.py and live.sh set it). A reconstructed room
(`reconstruction/`) has no collision geometry yet: use it for the look, not as `environment_usd` to drive in.

`nurec/` (not in git) is where NuRec scene assets go when `paths.nurec` in `config.yaml` points here.
