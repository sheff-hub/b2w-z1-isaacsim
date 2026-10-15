# Nav2 for the simulated B2W

ROS 2 Jazzy + Nav2 in Docker (`b2w-nav2:jazzy`, built from `Dockerfile`). The image also has RViz and
rqt. Isaac Sim talks to it through its own ROS 2 bridge libraries, over UDP DDS on the host network
(`isaac/fastdds.xml`), on the ROS domain of `config.yaml` (`ros.domain_id`, 37). Both sides read it:
`isaac/isaac_env.sh` and `run_nav2.sh`. The map comes from the scene file (`scenes/*.yaml`, `nav2_map`).

## Use

    python3 run.py nav2 start --scene warehouse_three_tables   # start Nav2 with the scene's map
    NAV2=1 SCENE=3 ./project/live.sh fetchdemo  # watch it in the Isaac window (DualVLN, Nav2, pick, delivery)
    ./project/process.sh                        # follow the run's progress from another terminal
    nav2/rviz.sh                      # RViz: map, costmaps, lidar cloud, Nav2's plan, footprint
    nav2/ros2_shell.sh                # a ROS 2 terminal; from a desktop terminal rviz2 / rqt work in it
    nav2/run_nav2.sh stop

Headless:

    python3 run.py fetch --no-vln --episodes 4          # Nav2 parks, picks and delivers (run.py starts Nav2)
    ./isaac/run_script.sh project/fetch_box.py --device cpu --nav2 --episodes 4 \
        --starts "1.0,-1.5,30; 0.5,1.0,0; 6.0,1.0,0; 2.0,3.0,-90"     # the same, by hand (Nav2 already running)

Only one simulator may talk to Nav2 at a time: both would publish /clock and /tf.

Options:
- `NAV2_CONTROLLER=mppi nav2/run_nav2.sh`: the holonomic controller (see below).
- `NAV2_SENSOR=scan ...`: the ray-cast scan instead of the lidar. Pass the matching
  `--nav2_sensor scan` to `fetch_box.py`.
- `fetch_box.py --nav2_goal entry`: Nav2 only to E, then the scripted E → P → D (the first setup).

## What Nav2 does in the fetch pipeline (`fetch_box.py --nav2`, default `--nav2_goal dock`)

1. DualVLN (if `--vln`) walks to the table, and the robot backs off, as before.
2. **Side choice:** Nav2's planner computes a path to the staging pose S on both sides
   (ComputePathToPose); the shorter wins.
3. **NavigateToPose(S):** Regulated Pure Pursuit drives there collision-free. S is 2.1 m before the
   pick point D on its line along the table, heading along it, ≥ 1.7 m from the table, so there is room
   to turn.
4. **DockRobot(D):** Nav2's docking server (`SimpleNonChargingDock`, dock pose from the simulator)
   drives the last 2.1 m and stops at D, 4 cm from the dock pose, aligned with its heading.
5. The scripted pick starts there. If any Nav2 step fails, the scripted route takes over from
   wherever the robot is.

Results:
- **24 random starts, seed 7** (`renders/fetch_nav2/results_val24_dock.json`):
  - **24/24 docked by Nav2** (no fallback used), **24/24 picks**, 0 falls, 0 table contacts;
  - pick pose reached within 2.8 cm (max 3.5) and 3.7° (max 8.3°);
  - staging 11.2 s, docking 7.4 s; 43.0 s per episode.
  - The scripted route on the same starts: 24/24, parking error 4.5 cm, 40.8 s per episode.
- **With DualVLN, 3 demo starts** (`renders/fetch_nav2_dock/`, videos): 3/3 docked by Nav2
  (2.2 cm), 3/3 picked and delivered.
- Compare: `python3 evaluation/compare_nav2.py renders/fetch_ik_fix/results_val24.json renders/fetch_nav2/results_val24_dock.json`

## Nav2 in the DualVLN walk (`--vln_follower`, `live.sh VLN_FOLLOWER=`)

DualVLN outputs a short trajectory (1–2 m) about every 0.4 s of sim time; something has to follow it:
- **`pp`** (default): our pure pursuit, `vln_nav.follow`.
- **`follow`** (option A): each trajectory goes to Nav2's controller server as `FollowPath`. RPP follows
  DualVLN's own path with its collision checking: "collision ahead" stops it near the table, like the
  depth bumper does.
- **`goal`** (option B): each trajectory's end, with its last segment's heading, becomes a
  `NavigateToPose` goal, and Nav2 plans its own path there.

In all three, the bumper still ends the walk ("BLOCKED"). Discrete moves (turn, step) use our small
controller. Nav2's commands go through the same dead-zone shaping as the parking.

Comparison, 6 starts × 3 followers, Nav2 parking in all (`python3 evaluation/compare_vln_followers.py --videos`
→ `renders/vln_follow_compare/`: `comparison.md`, `side_by_side_ep0..5.mp4`):

| | pure pursuit | A: FollowPath | B: NavigateToPose |
|---|---|---|---|
| DualVLN arrived (< 1 m) | 4/6 | **6/6** | 1/6 |
| end distance to the table (mean) | 1.22 m | **0.58 m** | 1.87 m |
| walk time / distance | **15.9 s** / 3.6 m | 29.4 s / 3.6 m | 34.4 s / 7.6 m |
| rms acceleration (smoothness) | **0.35** | 0.63 | 0.61 |
| picks | 6/6 | 6/6 | 6/6 |

A brings DualVLN to the table most reliably. Its stops near obstacles ("collision ahead") and RPP's
slowing at the end of every short path make it slower and less smooth. B re-plans to a new short
goal every 0.4 s and turns toward each one's heading: twice the distance, and DualVLN then stops in
the wrong places.

## Three tables (`SCENE=3`, task `…Fetch3…`, map `warehouse3`)

Red, yellow and green tables, each with a red, a yellow and a green box (order shuffled per episode).
`fetch_box.py --target green/red` or `auto`. Start Nav2 with `NAV2_MAP=warehouse3`; `live.sh` with
`SCENE=3` does this itself.

## What it sees: the robot's own lidar (`project/os0_lidar.py`)

The OS0 RTX lidar on `lidar_link` (128 channels, ±44.5°, 0.3–50 m) is read through the
`IsaacExtractRTXSensorPointCloudNoAccumulator` annotator.
- It is published as PointCloud2 `/points` in frame `os0`, with a `base_link → os0` transform every
  step, at 10 Hz.
- Points are 8 cm voxels within 6 m and −0.3–2 m height.
- Obstacles are points 0.15–1.35 m above the floor.
- The local costmap is a 3D voxel layer plus the static map; the global costmap is an obstacle layer
  plus the static map.

The lidar cannot see within about 0.9 m at floor level, nor behind the robot (its own body is in the
way). The static map keeps the table and shelves in both costmaps.

## Settings that matter, and why

- **The lidar pose comes from physics, not USD.** Isaac Lab does not update the robot links' USD
  transforms, so the USD pose of the sensor stayed at the spawn. Every cloud was placed back there:
  "Sensor origin … out of map bounds", false obstacles, no plans.
- **Rendering must be on for an RTX sensor that comes with the robot's USD.** Set
  `/isaaclab/render/rtx_sensors`, and enable `isaacsim.sensors.rtx` at start-up.
- **No unknown space in the global costmap.** The lidar's blind zone stayed UNKNOWN and blocked plans.
- **Footprint 1.1 × 0.60 m** (hip housings |y| 0.287 m). Wider, the pick point itself is in collision.
- **The Nav2 pick point is 4 cm further from the table than the script's.** At 5 cm, the footprint
  sat in the costmap cell the lidar marks at the table edge.
- **Docking collision threshold 2.3 m.** The approach line runs inside the table's inscribed band;
  at 0.4 m every docking failed with "Failed to get control" (905).
- **The map is built with cells 2 mm smaller than the grid**, so obstacles are not grown by a cell.
- **Behaviour tree server timeout 200 ms** (default 20). Plans were aborted while the planner was
  busy with the point clouds.
- **Dead-zone inverse on the commands** (`fetch_box.nav2_relay`). The ETH controller ignores sideways
  commands below about 0.12 m/s and turn commands below about 0.1 rad/s, so those two axes get the dead
  zone added. A breakaway kick handles anything still stuck.
- **RPP, not MPPI, for the drive to S.** Holonomic MPPI (forward, backward, sideways; still available)
  crawled at 0.02–0.1 m/s and wandered: 9.8 m driven for a 1.5 m path, one timeout in 4.
- **PoseProgressChecker:** turning in place counts as progress.
- Earlier traps:
  - PhysX scene queries are off by default in Isaac Lab;
  - enable the ROS 2 bridge at start-up (`--kit_args`), or `env.seed()` breaks;
  - wait for `/lifecycle_manager_navigation/is_active`;
  - pace the sim to real time while Nav2 drives.

## Files

| file | what |
|---|---|
| `Dockerfile` | ROS 2 Jazzy + navigation2 + nav2_bringup + simple_commander + slam_toolbox + rviz2 + rqt |
| `make_params.py` | writes `b2w_nav2_params.yaml` from Nav2's defaults (`nav2_params_default.yaml`). Edit this, not the yaml |
| `b2w_nav2.launch.py` | map_server (static map) + nav2_bringup `navigation_launch.py`, sim time, no AMCL |
| `run_nav2.sh` | start / stop the Nav2 container (`NAV2_SENSOR`, `NAV2_CONTROLLER`) |
| `rviz.sh`, `b2w_view.rviz` | RViz in a container |
| `ros2_shell.sh` | interactive ROS 2 shell on the same network; `rviz2`, `rqt`, `b2w_rviz` work from a desktop terminal |
| `project/nav2_link.py` | simulator side: /clock, /tf, /odom, /points, /scan out; /cmd_vel in; Nav2 action clients (navigate, path, dock); `build_map()` |
| `project/os0_lidar.py` | the lidar reader |
| `project/nav2_probe.py` | builds `nav2/maps/<map>.{pgm,yaml}` from the scene's colliders (`--task`, `--map`): run it for a new scene |
| `project/process.sh`, `progress.py` | live progress of a run in another terminal |
| `evaluation/compare_nav2.py` | compares two fetch result files episode by episode |

## On the real robot

Replace the simulator's ground-truth TF with odometry plus localization: AMCL (`localization_launch.py`)
or slam_toolbox in the launch file. The OS0 then publishes its real cloud on `/points`. The dock pose
must come from perception (the detected table or box) instead of the simulator.
