# project/: the application

The code that runs inside the Isaac Lab container (`isaac/run_script.sh`, `run.py`, `live.sh`). The Python files
import each other as siblings, so they live together in this folder. Everything that differs between environments
comes from a scene file (`scenes/*.yaml`); everything that differs between machines comes from `config.yaml`.

## Entry points

| File | What it does |
|---|---|
| `live.sh` | Interactive demos in an Isaac Sim window (or WebRTC stream): `vln`, `fetch`, `fetchdemo`, `pick`, `goto_mine`, `goto_paper`, `eth`, `say`, `stop`. Run `./project/live.sh` for the list. `SCENE=<name>` picks the scene file, `NAV2=1` lets Nav2 park, `STREAM=1` streams instead of a window. |
| `fetch_box.py` | The fetch pipeline: DualVLN walk (`--vln`), Nav2 (`--nav2`) or scripted route, the side pick planned by MoveIt (`--arm moveit`, default; `moveit/`) or scripted with damped least-squares IK (`--arm ik`), delivery (`--deliver`, `--deliver_frac`), recording (`--record`). In a shelf scene (`scenes/hospital_reception.yaml`) it takes the named box out of the shelf and puts it on the table (`place_moveit`). `--cam scene` / `scene_follow`: the scene file's camera. `run.py fetch` calls it with the final settings.; door scenes: walk to the named door, Nav2 parks, MoveIt grasps and turns the lever, the door pushed open round its hinge, into the room (`--probe --target b`: set down at the door) |
| `rfm_goto.py` | Give the whole-body RFM policy a 6D hand target and watch it drive, crouch and reach (`live.sh goto_mine`). |
| `vln_drive.py` | DualVLN driving on its own, from typed instructions (`live.sh vln`). |
| `eth_compare.py` | A/B of the walking controllers: ETH's policy vs robot_lab's, keyboard or the test sequence (`live.sh eth`). |
| `b2w_teleop.py`, `run_teleop.sh` | Drive the B2W from the keyboard (no RL, wheels only). |
| `show.sh`, `goto.py`, `demo_fetch.py`, `demo_reach_video.py`, `waypoint.py`, `loco_reach.py`, `play_task.py` | The earlier whole-body results (velocity-command policies wb9/wb12/wb15, two-policy drive + reach). |
| `process.sh`, `progress.py` | Follow a running fetch from another terminal. |
| `nav2_probe.py` | Build a Nav2 occupancy map from a scene's colliders: `--task <scene task> --map <name>` writes `nav2/maps/<name>.{pgm,yaml}`; `--area x0,x1,y0,y1` for a part of a big building. |
| `door_test.py` | the door task's physics without the robot: the doors in place and painted, a latched door holds a 60 N push, unlatched it opens (`FETCH_SCENE=/workspace/scenes/hospital_doors.yaml ./isaac/run_script.sh project/door_test.py --device cpu`) |
| `report_shots.py`, `rfm_shots.py`, `scene_views.py` | Stills for the report's figures (robot with sensor positions, whole-body reaching, camera views). |

## Modules

| File | What it is |
|---|---|
| `eth_loco.py` | ETH's walking policy as the base controller (default: our stance fine-tune `eth/policy_stance_v5_600.pt`; `ETH_POLICY=/workspace/eth/policy_force_new.onnx` for the original). |
| `nav2_link.py` | ROS 2 link to Nav2: clock, TF, odometry, lidar out; velocity commands in; Nav2 action clients. |
| `os0_lidar.py` | The robot's Ouster OS0-128 as a point cloud for Nav2. |
| `vln_nav.py` | DualVLN client: cameras, requests, path following, bumper. |
| `recorder.py` | The episode recorder (raw format; `tools/raw_to_lerobot.py` converts it). |
| `moveit_link.py` | Client of the MoveIt bridge (`moveit/moveit_server.py`), and the robot's body as boxes for MoveIt's planning scene. |
| `scene_tour.py` | Drives the B2W (or the B2W + Z1, `--robot b2w_z1`) round a reconstructed splat + collider scene on a route from `reconstruction_with_mesh/tools/robot_route.py`, with ETH's controller and pure pursuit, and films it: chase, head and wrist cameras, track and contacts per step. |
| `scene_still.py` | High-resolution stills of the B2W + Z1 standing in a reconstructed scene, from fixed world cameras (title pages, figures). |
| `rfm_helpers.py`, `free_camera.py` | Shared helpers (RFM target schedule, a movable viewer camera). |

Runtime files written here: `instruction.txt` and `target.txt` (what you type into a live demo; not in git).
