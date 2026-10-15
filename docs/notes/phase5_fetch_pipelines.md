> Development record, written during the project on the lab workstation. Paths such as `~/Firdavs/Isaac/...`
> refer to the layout before this repository: `Isaac/` is now `isaac/` + `project/` (+ `evaluation/`, `training/`),
> `repos/robot_lab` holds the tasks that are now in `training/robot_lab_overlay/`.

# Phase 5 — Fetch pipelines, Nav2, language and the VLA dataset: complete record

Unitree B2W wheeled quadruped + Z1 arm **with gripper**, NVIDIA Isaac Sim 5.1 / Isaac Lab
2.3.2. Work done 27 September – 12 October 2026.

This file collects all Phase 5 work in one place. The detailed working notes it draws on
are still there and give more depth:

| document | content |
|---|---|
| `PHASE5_STEP1_FETCH.md` | the scripted fetch, version by version (front pick → side pick) |
| `PHASE5_ETH_AND_DATASET.md` | ETH controller, first recorder, the 460-episode overnight dataset |
| `Isaac/PHASE5_ETH_STANCE_FINETUNE.md` | RL fine-tune of ETH's policy (rear-wheel stance) |
| `nav2/README.md` | Nav2 setup, the OS0 lidar, docking, DualVLN followers, three tables |

Where they disagree, this file wins: it is the latest.

---

## 0. Status as of 12 October 2026

### What Phase 5 achieved

- **An integrated quadruped-manipulator in simulation:** B2W + Z1 + gripper as one USD
  (23 DOFs), with RealSense cameras (a D455 at the head, a D405 in a holder on the gripper) and the
  Ouster OS0 RTX lidar,
  self-collision on, 0 composition errors.
- **A warehouse fetch task** (drive to a table, pick up a box, carry it back), first fully
  scripted with oracle box pose: **side pick 68/72 (94 %)**, 0 falls, 0 robot-link contacts.
- **A better walking controller:** ETH's real-robot SRU policy runs our robot. It drives
  sideways, backwards and turns far better than our own (`model_2600`), and the fetch rose to
  **69/72 (96 %)**. An RL fine-tune (`policy_stance_v5_600.pt`) fixed its narrowing rear
  stance and is the default since 9 October (fetch **24/24**).
- **Language → fetch:** DualVLN (a pretrained vision-language navigation model) walks to the
  table from a typed instruction, then the robot parks and picks. 230/240 delivered over the
  recording runs.
- **Nav2 instead of scripted parking:** ROS 2 Jazzy Nav2 in Docker, fed by the robot's own
  OS0 lidar, drives the robot to the pick point *in the right heading*: **24/24 docked, 2.8 cm,
  3.7°**, no fallback. Nav2 also carries the box back, and can follow DualVLN's trajectories
  (FollowPath: DualVLN arrived **6/6** vs our pure pursuit 4/6).
- **Scene variety:** three tables (red, yellow, green), each with a red, a yellow and a green
  box in shuffled order. The task names a table and a box ("Pick up the red box on the green
  table."). Scripted 12/12, Nav2 12/12.
- **A VLA data pipeline:** a recorder for vision (RGB-D front + wrist), language (task
  sentence + navigation sentence), proprioception (all 23 joints, base state, EE pose),
  actions (whole-body command, arm joints, previous action) and labels (phase, Nav2's planned
  path, target, success), converted to **LeRobot v2.1** (what openpi's π0 / π0.5 fine-tuning
  reads). Datasets: 444 episodes / 7.5 h (format 1, single table), and the format-2 demo set
  on three tables: **25/25 picked and delivered**, 38 min, validated with the official
  LeRobot loader (Section 7). **The training set is `demo_v4_mix`** (Section 7b): 30/30, half
  pick-only and half delivery with matching sentences, and the RealSense D405 wrist (on a ring clamp)
  and D455 head cameras.

### Timeline

| date | work |
|---|---|
| 27 Sep | Z1 gripper URDF repaired (invisible meshes, broken jaw colliders); `b2w_z1_gripper.usd` built |
| 27–28 Sep | scripted fetch v1 → v8 (front pick 65/72), then the side pick (68/72) |
| 1–3 Oct | DualVLN server in Docker; matching the Habitat conventions; first table test |
| 3–6 Oct | depth bumper; language → fetch (2/3 → 5/6); ETH controller A/B test |
| 6–7 Oct | ETH made the default (69/72); recorder v1; 460 episodes overnight (444 good); DualVLN per-client sessions |
| 7–8 Oct | ETH stance fine-tune (v1 – v8); v5 @ 600 chosen |
| 8 Oct | arm shake during the pick found and fixed (IK gain 0.3 → 0.1) |
| 9–11 Oct | Nav2 + OS0 lidar; staging + DockRobot; 24/24 |
| 11–12 Oct | DualVLN followers (pure pursuit / FollowPath / NavigateToPose); three-table scene; Nav2 delivery; recorder v2; demo dataset |
| 12–13 Oct | pick-only / delivery task mix with matching sentences; RealSense cameras (D455 lens at the head with the DualVLN crop; D405 first on a side bracket, then on a ring clamp round link06 at the user's pose); `demo_v4_mix` 30/30 |

### Which configuration to use

| piece | default | alternative |
|---|---|---|
| walking controller | ETH fine-tune `Isaac/eth/policy_stance_v5_600.pt` (`--loco eth`) | `ETH_POLICY=…/policy_force_new.onnx` (ETH original), `--loco ours` (model_2600) |
| parking | Nav2 staging + DockRobot (`--nav2`) | scripted (no `--nav2`) |
| DualVLN follower | `pp` (pure pursuit, fastest, smoothest) | `follow` (Nav2 FollowPath: arrives most reliably; used for the demo dataset) |
| pick | scripted side pick, DLS IK, oracle box pose | — |
| physics | `--device cpu` (the Ada card is shared with CARLA) | GPU |

---

## 1. Application scenarios (Isaac Sim)

All scenes are the Isaac warehouse (`full_warehouse.usd`) with racks along both sides and
pillars at y ≈ ±3. Tables are built from primitives with the top at 0.60 m. The packing
table from the asset library (top 0.99 m) is above the arm's reach.

| scenario | task id | content | used for |
|---|---|---|---|
| **S1 single table** | `RobotLab-Isaac-Fetch-Warehouse-Unitree-B2W-Z1-Gripper-v0` | one wooden table, one box (±25 cm, ±23° random), a drop zone across the aisle | scripted fetch, ETH, Nav2 docking, 460-episode dataset |
| **S2 three tables** | `RobotLab-Isaac-Fetch3-Warehouse-Unitree-B2W-Z1-Gripper-v0` | red / yellow / green tables at x 4.3, y −4.5 / 1.0 / 6.5; a red, a yellow and a green box on each (slots y −0.3 / 0 / +0.3, order shuffled every reset, jitter) | target selection by colour, Nav2, demo dataset |
| **S3 language** | either of the above + `--vln "…"` | a typed instruction; DualVLN walks, the robot parks and picks | language → fetch, Stage-2-style data |
| **S4 live** | `./live.sh fetch` (GUI, WebRTC) | type tasks ("go to the green table and pick the red box", "bring it back", "home") | demonstrations |

Variety per episode: random start position and heading (`--random_start`), random target
(`--target auto`), random box order and pose, 5 pick and 4 pick-and-deliver wordings
(`--task_variants`). Not varied: lighting, distractor objects, table positions.

Nav2 maps (`nav2/maps/warehouse`, `warehouse3`) are built from the scene's colliders by
`Isaac/nav2_probe.py`.

---

## 2. Integrated quadruped-manipulator setup

### Robot asset: `robots_usd/b2w_z1_gripper.usd`

Built by `tools/build_b2w_z1_gripper_asset.py` from the user's `b2w_z1.usd` (read only).

- **23 DOFs:** 12 leg, 4 wheel (named `*_foot_joint`), 6 arm, 1 gripper.
- **Gripper:** Unitree's Z1-gripper URDF referenced `package://` meshes the importer could not
  find, so the robot imported **invisible** without an error. Fixed with a local URDF
  (`tools/make_z1_gripper_urdf.py`), and each jaw mesh made its own collider (convex
  decomposition). Grasp test: 3 cm cube, 0.0 mm slip over 2 s.
- **Drives:** the newer importer writes acceleration drives (joint4 sagged 51° under the
  gripper). Force drives copied from the Z1 asset (1.6°). Gripper kp 20, ramped close over 1 s,
  jaw friction 1.2: harder squeezed the box out of the angled jaws.
- **Arm moved 0.11 m back:** at the original mount the stowed gripper sat 3 cm inside the
  lidar cage roof.
- **Self-collision on**, with two pairs filtered (head↔lidar, link00↔base) and an invisible top
  cover on the body. Lidar mast collider replaced by a box matching the visible cage (the old
  capsule was 7 cm taller; the arm leaned on it).
- **Composition errors:** 0. Dead importer references removed by `Isaac/usd_fixes.py`.

### Sensors

The cameras are models of real, buyable RealSense cameras, so the same setup can go on the real
robot (since 13 Oct):

| sensor | model | where | recorded / used |
|---|---|---|---|
| front (head) camera | **RealSense D455** colour camera (OV9782): 90 × 65°, 16:10, the lens of Isaac Sim's own `rsd455.usd` | head (`f_oc_link`) | VLA vision, RGB + depth, 320 × 200. DualVLN gets its exact 79° 4:3 centre crop at 640 × 480, its training camera |
| look-down view | the same D455 lens, pitched 30° | head | DualVLN's "↓" view only (virtual) |
| rear camera | RealSense D455 colour lens | tail (`r_oc_ink`) | not used yet |
| wrist camera | **RealSense D405**: 87 × 56°, 16:9, depth from 7 cm (closer reads 0, as on the real sensor) | on a ring clamp round link06, 17.6 cm out to the side, looking at the gripper | VLA vision, RGB + depth, 320 × 180 |
| Ouster OS0, RTX lidar, 128 channels | | `lidar_link` | Nav2 (`/points`), self-filtered |
| proprioception | | all joints, base IMU-like state | VLA state |

**The wrist camera holder** (`tools/wrist_d405.py`, built into the robot by `build_b2w_z1_gripper_asset.py`).
The idea and the camera pose are the user's (the pose was placed by hand in
`robots_usd/b2w_z1_gripper_reference.usd`), after the 3D-printed Z1 end mounts people share
(MakerWorld 430201, a D455 + AprilTag mount for the Z1):
- **Camera body:** Intel's own D405 mesh (`robots_usd/RealSense/d405.usd`, from the realsense-ros
  description package). Intel's official CAD is SolidWorks only.
- **Ring:** a 3D-printed split collar clamped round link06 (bore 33 mm, 7.5 mm wall, 17 mm wide), with
  two ears and an M3 bolt at the split.
- **Arm:** a 9 × 18 mm arm out to the gripper's −y side.
- **Cradle:** around the camera, with a front lip that stays clear of the lenses, and a pad where the
  arm joins.
- **Mass:** 116 g with the camera (printed parts at ~60 % solid), added to link06's mass and inertia.
  The wrist's roll inertia rises about 13×; the IK pick still reaches the grasp within 1–2 mm.

Where it sits, and why:
- **Camera pose:** link06 (0.021, −0.176, 0.014), 17.6 cm out to the side, turned 27.7° toward the
  gripper and 5.5° down. The whole gripper is in view, about a quarter of the image, and the box
  between the jaws.
- **Wrist depth:** mostly valid now. On the earlier side mount the gripper's near side was closer
  than 7 cm and read 0.
- **Side:** −y. At every recorded grasp the gripper's +y side faces down toward the box.
- **Ring position:** on link06's outer part (x 26–43 mm), not the middle. With the arm stowed, link02
  tucks under link06's first 25 mm, 1 mm away, and a centred ring pressed on it at 18 N
  (`Isaac/probe_self_collision.py`). At the stowed pose the pipeline uses, there is no contact now.
- **Clearances:** at least 31 mm from the moving jaw over its whole range and 23 mm from link05 over a
  full wrist roll; no part of the holder is in the camera's view.
- **Replay of 55 recorded picks with the holder attached** (`Isaac/tools/wrist_d405_replay.py`): at
  least 152 mm above the table and 99 mm from the box before it is gripped.
- **Stowed:** above the body, behind the lidar cage.

Two simulator traps:
- Isaac Lab does not update the robot links' USD transforms, so the lidar pose must come
  from physics.
- An RTX sensor that ships in the robot USD needs rendering on (`/isaaclab/render/rtx_sensors`)
  and `isaacsim.sensors.rtx` enabled at start-up.

### Control stack

```
 language ──► DualVLN (VLN, server)          ──► trajectory / discrete moves ─┐
 goal pose ─► Nav2 (planner, RPP, docking)   ──► /cmd_vel ────────────────────┤
 scripted states (back-off, pick, carry)     ──► base velocity ───────────────┤
                                                                               ▼
                             dead-zone shaping + breakaway kick (fetch_box.nav2_relay)
                                                                               ▼
                 ETH SRU walking policy, 50 Hz (LSTM, 60 obs → 12 legs + 4 wheels)
 arm: DLS IK on joint targets (gain 0.1) ──► Z1 joint PD     gripper: ramped position
```

**ETH controller** (`Isaac/eth_loco.py`): ETH's real-robot B2W policy, run with ETH's
motor settings (PD 100 / 3.5, wheel damping 3.0) and stance (body 0.75 m, 15 cm taller than
ours).

A/B test with the arm on, 15 tests each (`Isaac/eth_compare.py`):

| test (% of command followed) | ETH | ours (model_2600) |
|---|---|---|
| sideways 0.2 m/s left / right | **89 / 68** | 0 / 0 |
| slowing into a diagonal | **94** | 39 |
| backward 0.3 | **93** | 55 |
| turning 0.5 rad/s | **101** | 65 |
| 180° turn: turned / creep | **179° / 0.10 m** | 112° / 0.55 m |
| peak tilt at 1 m/s | 17.7° | **0.9°** |
| drift standing 10 s with the arm out | 9.5 cm | **0.5 cm** |

**Stance fine-tune** (`Isaac/PHASE5_ETH_STANCE_FINETUNE.md`): ETH pulls the rear hips
inward after the first turn (rear track 0.20 m). An RL fine-tune with an imitation reward
toward a frozen ETH copy, stance terms only while driving straight, and a 35 % straight /
15 % spin command mix widened it to **0.31 m** with velocity tracking unchanged.

Its one regression: sideways from standstill dropped to 0–35 %. Nav2's dead-zone shaping
covers it in the pipeline.

---

## 3. Task pipelines

### 3.1 Manipulation: the side pick (`fetch_box.py`, scripted, oracle box pose)

1. The robot parks parallel to the table, its hips about 5 cm from the edge. The ETH stance
   puts the hips above the 0.6 m top.
2. It crouches to ETH's "crouched" profile (6 cm lower), ramped in.
3. The arm unfolds in joint space to a ready pose. Differential IK from the folded stow runs
   into joint limits.
4. DLS IK moves to the pre-grasp, then the grasp (pitched 60°, 1.5 cm above the box centre).
   The IK step accumulates on joint targets, which removes the PD drive's gravity sag. It
   servos until < 5 mm.
5. The gripper closes, ramped over 1 s. The arm lifts (IK top-up to 12 cm) and goes to a
   carry pose.
6. Throughout the pick, a proportional base and heading hold is on. With the arm out, ETH
   drifts up to 18 cm otherwise.

**Why side, not front:** from the front, the hips touched the table in 35/72 picks
(up to 161 N). From the side: 0/72.

**Arm shake (8 Oct):** the IK loop oscillated at 2.3 Hz (±5 cm) whenever it held a pose.
The cause was a gain of 0.3 integrating on targets that the PD drive follows about 6°
behind. Gain 0.1 brought the wobble from 5.7 cm to 0.01 cm.

### 3.2 Navigation, three ways

| pipeline | how the robot gets to the pick point | where |
|---|---|---|
| **scripted** | pre-dock point → entry point (turn there, away from the table) → straight approach → closed-loop distance trim; routed round all tables by a visibility graph | `fetch_box.navigate` |
| **Nav2** | side chosen by ComputePathToPose length → NavigateToPose (RPP) to a staging pose S 2.1 m before the pick point D → **DockRobot(D)** (SimpleNonChargingDock) | `fetch_box.nav2_drive`, `nav2/` |
| **language** | DualVLN walks toward the named table (front RGB-D + look-down) until STOP / bumper / timeout → back-off → Nav2 or scripted parking | `vln_nav.py`, `--vln` |

Delivery: Nav2 NavigateToPose to the drop zone (`--nav2`), else scripted. If any Nav2 step
fails, the scripted route takes over from wherever the robot is.

**Nav2 setup** (ROS 2 Jazzy in Docker `b2w-nav2:jazzy`, `ROS_DOMAIN_ID` 37, UDP FastDDS):
- **Sensing:** the OS0 cloud as `/points`, self-filtered against every robot link and the
  carried box (0.22 m). Without that filter, the raised arm and box read as "collision ahead"
  and deliveries timed out.
- **Costmaps:** a voxel layer + static map in the local costmap, an obstacle layer + static
  map in the global one, with unknown space off.
- **Robot model:** footprint 1.1 × 0.6 m; ground-truth TF from the simulator.
- **Command shaping:** ETH ignores sideways commands below 0.12 m/s and turns below
  0.1 rad/s, so the relay adds a dead-zone inverse and a breakaway kick.
- **Settings that cost time:** docking collision threshold 2.3 m; BT server timeout 200 ms.
  The full list of traps and their fixes is in `nav2/README.md`.

**DualVLN conventions that matter** (each mismatch degrades silently):
- The look-down camera answers the model's "↓" action.
- 79° FOV; System 1 compares frames from the same camera.
- Sentences are capitalised and end with a full stop.
- The sim is frozen while System 2 thinks: async went 0/5.
- A depth bumper at 0.35 m, because DualVLN, trained for a human-height agent, drives the
  head into table tops.
- One session per client: parallel recordings had corrupted each other's memory.

### 3.3 Language interface

Typed or scripted tasks are split into a navigation sentence for DualVLN and a task
sentence. Three-table example:

- **Task:** "Pick up the red box on the green table."
- **DualVLN:** "Walk to the green table and stop in front of it."

`live.sh fetch` parses "<colour> table" / "<colour> box", "bring it back", "home" and "stop".

**The task sentence must match what the robot does.** Since 12 Oct there are two template
sets (`fetch_box.target_texts`). Each is checked against the live parser's own pick and
deliver patterns:

| task kind | templates (× 3 tables × 3 box colours = 36 sentences each) |
|---|---|
| **pick only** | "Pick up the {box} box on the {table} table." · "Go to the {table} table and pick up the {box} box." · "Grab the {box} box from the {table} table." · "Lift the {box} box off the {table} table." |
| **pick and deliver** | "Bring me the {box} box from the {table} table." · "Pick up the {box} box on the {table} table and bring it to me." · "Go to the {table} table, grab the {box} box and bring it back." · "Fetch the {box} box from the {table} table and carry it to the drop zone." |

`--deliver_frac 0.5` flips a coin per episode, drawn from its own random stream, so targets
and starts are unchanged. Pick-only episodes end after the lift. Each episode records
`deliver_task`. A VLA trained on the mix has to read the sentence to know whether to come
back.

---

## 4. Evaluation across scenarios and tasks

All results come from the result JSONs named below, with seeded random starts. 0 falls in
every row.

### 4.1 Fetch (navigate + pick [+ deliver])

| scenario | navigation | controller | episodes | picked | delivered | notes | file |
|---|---|---|---|---|---|---|---|
| S1 | scripted, front pick (v8) | ours | 72 | 65 (90 %) | 65 | hips hit the table in 35 | `renders/fetch/v8_seed*_deliver.json` |
| S1 | scripted, side pick | ours | 72 | 68 (94 %) | 68 | 0 robot contacts | `PHASE5_STEP1_FETCH.md` |
| S1 | scripted, side pick + crouch | ETH | 72 | **69 (96 %)** | 69 | 48.6 s / episode | `renders/fetch/eth_v3_seed*_deliver.json` |
| S1 | scripted | ETH fine-tune | 24 | 24 | — | | `PHASE5_ETH_STANCE_FINETUNE.md` |
| S1 | **Nav2 docking** | ETH fine-tune | 24 | **24** | — | 24/24 docked by Nav2, 2.8 cm / 3.7° (max 3.5 cm / 8.3°), 43.0 s | `renders/fetch_nav2/results_val24_dock.json` |
| S1 | scripted, same 24 starts | ETH fine-tune | 24 | 24 | — | 4.5 cm, 40.8 s | `renders/fetch_ik_fix/results_val24.json` |
| S2 | scripted, random target | ETH fine-tune | 12 | **12** | — | neighbouring boxes never moved | `renders/scene3/results_script12b.json` |
| S2 | **Nav2 dock + Nav2 delivery** | ETH fine-tune | 12 | **12** | **12** | docked 1.7–3.4 cm, 0 contacts, 60 s | `renders/scene3/results_nav2_12.json` |
| S3 (S1) | DualVLN + scripted parking (recording runs) | ETH | 240 | 230 (96 %) | 230 | DualVLN reached the table 135/240 | `datasets/raw/eth_v2_[a-f]` |
| S1 | scripted expert (recording runs) | ETH | 220 | 214 (97 %) | 214 | | `datasets/raw/eth_v2_scripted*` |
| S3 (S2) | DualVLN (FollowPath) + Nav2 | ETH fine-tune | 6 | 6 | 6 | right table 4/6, within 1 m 2/6; Nav2 docked 5/6 | `renders/scene3_vln/results.json` |
| S3 (S2) | **demo dataset:** DualVLN (FollowPath) + Nav2 dock + Nav2 delivery, random targets, whole hall | ETH fine-tune | 25 | **25** | **25** | Nav2 docked 25/25, 2.6 cm / 2.7°; DualVLN right table 13/25, < 1 m 3/25; 90 s | `datasets/raw/demo_v2_results.json` |
| S3 (S2), RealSense robot | **demo_v4_mix:** as above, half pick-only / half deliver, D405 on the wrist ring + D455 head | ETH fine-tune | 30 | **30** | **15/15** | Nav2 docked 30/30, 2.9 cm / 3.1°; DualVLN right table 18/30, < 1 m 12/30 | `datasets/raw/demo_v4_mix*_results.json` |
| S3 (S2), RealSense robot (side-mounted D405, superseded) | camera test, 6 episodes; then 30 episodes (data deleted) | ETH fine-tune | 6 + 30 | 5 + 30 | 1/1 + 15/15 | the one failure: Nav2 aborted after DualVLN stopped behind the table; fallback parked 1.24 m off | — |

Remaining pick failures (S1, ETH): 3/72, all the arm not reaching the grasp pose.

### 4.2 The DualVLN walk: who follows its trajectory

Six starts, the same instruction for each follower, Nav2 parking in all of them:

| | pure pursuit | A: Nav2 FollowPath | B: Nav2 NavigateToPose |
|---|---|---|---|
| DualVLN arrived (< 1 m from the table) | 4/6 | **6/6** | 1/6 |
| distance to the table at the end (mean) | 1.22 m | **0.58 m** | 1.87 m |
| walk time / distance | **15.9 s** / 3.6 m | 29.4 s / 3.6 m | 34.4 s / 7.6 m |
| rms acceleration (smoothness) | **0.35** | 0.63 | 0.61 |
| picks | 6/6 | 6/6 | 6/6 |

`renders/vln_follow_compare/comparison.md`, side-by-side videos `side_by_side_ep0..5.mp4`.

- **A** stops short of obstacles, and RPP slows at the end of every short path, so it is
  slower.
- **B** re-plans to a new goal every 0.4 s, turns toward each goal's heading, and drives
  twice the distance.

### 4.3 Locomotion

The controller A/B test is in §2. Stance fine-tune: rear track 0.20 → 0.31 m; sideways
from standstill 68–89 % → 0–35 % (the known regression).

### 4.4 What the numbers do not show

- **Perception is an oracle:** the box pose comes from the simulator, and so does Nav2's TF
  (no localisation).
- **DualVLN is the weakest link:** from random headings it reaches the table on its own in
  about 56 % of episodes. The pipeline still completes because Nav2 or the scripted parking
  takes over. In the three-table scene it heads for the right colour about 2 times in 3.
- **Small samples:** the Nav2 and three-table rows are 12–24 episodes; the follower
  comparison is 6. A 95 % interval on 24/24 still reaches down to about 86 %.
- **Simulation runs** on CPU PhysX with one robot, at about 1–3× real time.

---

## 5. The data pipeline (VLA recording)

The end goal (TUM "Towards Whole-Body VLA") has two stages:
- **Stage 1:** scripted expert demonstrations.
- **Stage 2:** rollouts with pretrained VLN/VLA as data generators.

Both then train one whole-body VLA: (images, proprioception, language) → base velocity + EE
pose, executed by the low-level controller.

```
Isaac Sim episode (fetch_box.py --record DIR)
   └─ Isaac/recorder.py, 10 Hz ─► raw: ep_XXXXXX/{front.mp4, wrist.mp4, data.npz, meta.json}
          └─ tools/raw_to_lerobot.py (host venv with pyarrow) ─► LeRobot v2.1
                 meta/{info,tasks,episodes,episodes_stats}.json(l)
                 data/chunk-000/episode_XXXXXX.parquet
                 videos/chunk-000/observation.images.{front,wrist}/episode_XXXXXX.mp4
          └─ tools/plot_episode.py ─► review PNG per episode (path by phase, commands, plan, frames)
```

### Recorded per frame (format 2, 10 Hz)

| group | LeRobot key | shape | content |
|---|---|---|---|
| **Vision** | `observation.images.front` / `.wrist` | front 200×320×3 (D455, 16:10), wrist 180×320×3 (D405, 16:9) video | RGB, h264. Recordings before 13 Oct: 240×320, a 79° front and a bare 60° wrist camera |
| | `observation.depth.front` / `.wrist` | half the image size, uint16 | depth in mm, 0 = no return, ≤ 10 m; wrist 0 below 7 cm (the D405's minimum) |
| **Language** | `task` (→ `meta/tasks.jsonl`) | — | task sentence, e.g. "Pick up the red box on the green table." |
| | `nav_instruction` (`meta/episodes.jsonl`) | — | DualVLN's sentence, e.g. "Walk to the green table and stop in front of it." |
| **Proprioception** | `observation.state` | 63 | joint pos 23 + joint vel 23 + base lin vel 3 + ang vel 3 + projected gravity 3 + height 1 + EE pose in base frame 7 |
| | `observation.joint_pos` / `.joint_vel` | 23 each | the same, by joint (legs, wheels, arm, gripper) |
| **Action** | `action` | 11 | **whole-body command:** base velocity [vx, vy, wz] + EE target pose in the base frame (pos 3, quat wxyz 4; 5-frame smoothed) + gripper |
| | `action.arm_joints` | 7 | arm joint targets + gripper (alternative action space) |
| | `observation.prev_action` | 11 | the previous frame's action |
| **Labels** | `label.phase` | 1 | idle / walk / backoff / dock / pick / carry / deliver / nav2 / nav2_dock |
| | `label.plan` | 20 | Nav2's (or DualVLN's) planned path ahead: 10 points every 0.2 m in the base frame, NaN where none |
| | `observation.base_pose_world` | 3 | x, y, yaw (evaluation only) |
| | per episode | — | target table / box, success, delivered, DualVLN arrived / right table, Nav2 docked, fell, start pose, seed |

**History and future** ("last states", "future states/actions") are not duplicated in each
frame. They are neighbouring frames, read at load time:

```python
LeRobotDataset(root, delta_timestamps={"observation.state": [-0.2, -0.1, 0.0],
                                       "action": [i / 10 for i in range(50)]})  # 5 s action chunk
```

### Does it fit VLA training?

Yes:
- openpi's π0 / π0.5 fine-tuning reads LeRobot v2.1 directly, and so do SmolVLA and GR00T N1.5.
- OpenVLA needs RLDS (a conversion step).
- π0.5 uses the RGB images, `observation.state` and `action`. Depth and labels are extra
  keys that it ignores, kept for auxiliary losses and evaluation.

---

## 6. Datasets

### 6.1 Format 1, single table (night of 6–7 Oct)

No depth, 51-dim state, and no Nav2 plan.

| dataset | episodes | frames | duration | size |
|---|---|---|---|---|
| `fetch_eth_v2` (all successes) | 444 | 268,719 | 7.5 h | 923 MiB |
| `fetch_eth_v2_vln` (DualVLN walk + scripted parking) | 230 | 161,066 | 4.5 h | 576 MiB |
| `fetch_eth_v2_scripted` (scripted expert) | 214 | 107,653 | 3.0 h | 347 MiB |

Recordings from before the DualVLN session fix are kept as `*_MIXED_SESSIONS`. Their walk
part is not representative.

### 6.2 Format 2, three tables: pilot

`datasets/raw/pilot_v2` → `datasets/lerobot/pilot_v2`: 1 episode, 833 frames,
idle → nav2 → nav2_dock → pick → deliver.
- The plan label is filled in about 90 % of the driving frames.
- Front depth median 3.8 m.
- About 21 MB raw (`data.npz` with depth), 25 MiB in LeRobot.

---

## 7. Demo dataset (format 2, three tables): `demo_v2`

This demonstrates the full pipeline of §5 end to end. It was recorded on 12 Oct, 18:01–20:50
(about 6.8 min wall time per episode), with this command:

```bash
NAV2_MAP=warehouse3 ~/Firdavs/nav2/run_nav2.sh
ISAAC_IMAGE=isaac-lab:2.3.2 ./run_script.sh fetch_box.py --device cpu --loco eth --nav2 --vln_follower follow \
    --task RobotLab-Isaac-Fetch3-Warehouse-Unitree-B2W-Z1-Gripper-v0 --target auto --task_variants \
    --episodes 25 --seed 31 --vln "x" --random_start=0.9,7.4,-7.5,9.5 --deliver \
    --record datasets/raw/demo_v2 --out datasets/raw/demo_v2_results.json
```

`--vln "x"` turns the DualVLN walk on. In the three-table scene the sentence itself is built
from the target ("Walk to the <colour> table and stop in front of it.").

Each episode runs:
1. DualVLN walks, its trajectories followed by Nav2 FollowPath.
2. Back-off.
3. Nav2 to the staging pose, then DockRobot to the pick point.
4. The side pick.
5. Nav2 NavigateToPose carries the box to the drop zone.

### Results (`datasets/raw/demo_v2_results.json`)

| metric | result |
|---|---|
| **picked, lifted, held** | **25/25**, 0 falls |
| **delivered** (all by Nav2) | **25/25** |
| Nav2 docked at the pick point (no scripted fallback) | 25/25, position error **2.6 cm** mean (1.3–3.5), heading **2.7°** mean (max 7.8°) |
| side of the table chosen by Nav2 | left 13, right 12 |
| box lift | 26.6 cm mean (23.6–29.3) |
| episode time (sim) | 90.2 s mean (63–139) |
| DualVLN headed for the **named** table | 13/25 (52 %) |
| DualVLN ended < 1 m from the table | 3/25; ended by STOP 17, timeout (60 s) 8; end distance 3.3 m mean (1.2 m when it chose the right table) |

The random starts cover the whole hall (y −7.5 to 9.5), often 6–12 m from the named table
and facing away. DualVLN therefore reaches the table less often here than from the
single-table starts in §4.2 (6/6). Nav2 recovers every time, from wherever DualVLN stops.
For a VLA these episodes are still useful: the walk part shows realistic VLN behaviour, and
Nav2 supplies an expert correction.

**Language caveat:** demo_v2 was recorded with the old templates. All 25 episodes deliver, but
only the "Bring me…" sentences (8/25) say so; "Pick up…", "Go to… and pick up…" and
"Fetch…" do not. Use `demo_v4_mix` (§7b) for training. demo_v2 remains a valid
pipeline demonstration.

### Content

| | |
|---|---|
| target combinations | 8 of the 9 (box colour × table colour); tables: yellow 10, red 9, green 6 |
| task sentences | 18 distinct of the 36 possible (4 wording templates × 3 tables × 3 box colours, `fetch_box.target_texts`); see `meta/tasks.jsonl` |
| navigation sentences | 3 ("Walk to the <colour> table and stop in front of it.") |
| frames | 22,548 at 10 Hz = 37.6 min |
| phase shares | walk 37 %, pick 25 %, deliver 15 %, nav2 12 %, nav2_dock 9 %, idle / backoff 2 % |
| `label.plan` filled | walk 88 % (DualVLN trajectory), nav2 99 %, nav2_dock 94 %, deliver 77 % (Nav2's plan); empty in pick, as intended |
| front depth valid pixels | 70 % (the rest is beyond 10 m, stored as 0) |

### Files

| dataset | episodes | frames | size |
|---|---|---|---|
| raw `datasets/raw/demo_v2` | 25 | 22,548 | 610 MB (depth is ~90 % of it) |
| LeRobot `datasets/lerobot/demo_v2` (all) | 25 | 22,548 | 652 MiB |
| LeRobot `datasets/lerobot/demo_v2_right_table` (`--only_right_table`) | 13 | 11,921 | 337 MiB |
| review plots `datasets/raw/demo_v2_review/` | one PNG per episode + `overview.png` (all paths) | | |

**Validated** with the official loader, `LeRobotDataset` from lerobot 0.3.3
(`tools/check_lerobot.py`).
- It decodes both videos and returns depth, state history `(3, 63)` and action chunks
  `(10, 11)` with padding masks.
- Each frame comes with its task sentence.
- With `--no_depth` (π0.5 uses RGB only), the dataset is 75 MiB instead of 652 MiB.

---

## 7b. Mixed pick / deliver dataset with the RealSense cameras: `demo_v4_mix` (the training set)

This dataset has three properties the earlier ones lacked:
- **Matching language** (§3.3): half the episodes are pick-only, half pick-and-deliver, and each
  sentence says which.
- **Real camera models** (§2): D455 at the head (320 × 200); D405 on the ring clamp at the wrist (320 × 180).
- **The user's wrist camera pose.**

Command: as for demo_v2 (§7), plus `--deliver_frac 0.5`, with 30 episodes at seed 43 (15 / 15 by the
coin), recorded on 13 Oct, 15:21–17:20.

**Two episodes replaced.** While it ran, a second simulator was started with Nav2 for about 10 minutes
(15:44–15:54).
- **What happened:** both sims published /clock and /tf, and Nav2 logged 36,401 "jump back in time"
  warnings.
- **Effect:** episodes 5 and 6 fell in that window. Nav2 aborted, the scripted parking took over, and
  their plan labels may be the other robot's.
- **What was done:** they were removed and replaced by 2 pick-only episodes at seed 143
  (`datasets/raw/demo_v4_mix_extra`). No warnings after 15:55.
- **Lesson:** only one simulator per Nav2.

### Results (`datasets/raw/demo_v4_mix_results.json` without episodes 5 and 6, plus `demo_v4_mix_extra_results.json`)

| metric | result |
|---|---|
| **picked, lifted, held** | **30/30**, 0 falls |
| pick-only / pick-and-deliver episodes | 15 / 15; **all 15 deliveries done** (by Nav2) |
| Nav2 docked at the pick point (no scripted fallback) | 30/30, position error **2.9 cm** mean (2.0–3.9), heading 3.1° mean (max 6.3°) |
| grasp point vs box centre (horizontal) | 0.1 cm mean (max 0.4) |
| box lift | 26.5 cm mean (22.3–29.4) |
| episode time (sim) | pick-only 80 s, with delivery 83 s (mean) |
| DualVLN headed for the named table | **18/30** (60 %) |
| DualVLN ended < 1 m from the table | 12/30; ended by STOP 23, timeout 7; end distance 3.0 m mean |

The heavier wrist (camera on a 17.6 cm arm, roll inertia ×13) did not hurt the IK pick: pre-grasp
and grasp reached within 1–2 mm, nothing touched.

### Content

| | |
|---|---|
| target combinations | all 9 (box colour × table colour); 10 episodes per table |
| task sentences | 27 distinct, from the 4 pick-only + 4 deliver templates (`meta/tasks.jsonl`) |
| frames | 24,416 at 10 Hz = 40.7 min |
| phase shares | walk 36 %, pick 28 %, nav2 15 %, nav2_dock 10 %, deliver 9 %, idle / backoff 2 % |
| `label.plan` filled | walk 78 %, nav2 99 %, nav2_dock 94 %, deliver 79 % |
| wrist depth = 0 | 2 % of pixels during the pick, 0 % while delivering, ~20 % while driving. On the earlier side mount it was 48 % during the pick, where the gripper's near side was closer than the D405's 7 cm |

### Files

| dataset | episodes | frames | size |
|---|---|---|---|
| raw `datasets/raw/demo_v4_mix` (28) + `demo_v4_mix_extra` (2) | 30 | 24,416 | 509 MB |
| LeRobot `datasets/lerobot/demo_v4_mix` (all) | 30 | 24,416 | 681 MiB |
| LeRobot `datasets/lerobot/demo_v4_mix_right_table` (DualVLN headed for the named table) | 18 | 13,432 | 368 MiB |
| review plots `datasets/raw/demo_v4_mix_review/`, `demo_v4_mix_extra_review/` | one PNG per episode + overview | | |

Conversion: `raw_to_lerobot.py --raw datasets/raw/demo_v4_mix --raw datasets/raw/demo_v4_mix_extra --out ...`.

**Validated** with the official `LeRobotDataset` loader (lerobot 0.3.3, `tools/check_lerobot.py`):
- videos: front 200 × 320, wrist 180 × 320;
- depth: 100 × 160 and 90 × 160;
- state history (3, 63) and action chunks (10, 11).

The earlier recordings with the previous wrist-camera positions (`demo_v3_mix`, the 8-episode partial
run, the 6-episode camera test) were deleted at the user's request.

## 8. Code base

| path | role |
|---|---|
| `repos/robot_lab/.../unitree_b2w/fetch_env_cfg.py` | fetch tasks S1 and S2: scene, robot, actions, reset events (`place_boxes_shuffled`) |
| `Isaac/fetch_box.py` | the pipeline: DualVLN walk, scripted / Nav2 navigation, IK pick, delivery, recording, videos, live mode |
| `Isaac/eth_loco.py`, `Isaac/eth/` | ETH controller (ONNX original + fine-tune `policy_stance_v5_600.pt`) |
| `Isaac/eth_compare.py`, `probe_track.py`, `train_eth_ft.py` | controller A/B test, stance probe, fine-tune training / eval |
| `Isaac/vln_nav.py`, `Isaac/vln/` | DualVLN client (Navigator, followers, bumper) and server (Docker) |
| `Isaac/nav2_link.py`, `os0_lidar.py`, `nav2_probe.py` | simulator ↔ Nav2 bridge, lidar reader, map builder |
| `nav2/` | Nav2 image, parameters (`make_params.py`), launch, `run_nav2.sh`, `rviz.sh`, `ros2_shell.sh` |
| `Isaac/recorder.py` | per-frame recording (format 2) |
| `Isaac/tools/raw_to_lerobot.py`, `plot_episode.py`, `check_lerobot.py` | LeRobot conversion, review plots, load check with the official loader |
| `tools/wrist_d405.py`, `tools/make_d405_usd.py`, `robots_usd/RealSense/` | the D405 wrist camera and its holder (Intel's mesh, cradle, bracket, checks); `rsd455.usd` (Isaac's D455, the head cameras' lens) |
| `Isaac/tools/wrist_d405_replay.py` | replays recorded picks with the holder attached: clearance to table and boxes |
| `Isaac/summarize_fetch.py`, `tools/compare_nav2.py`, `tools/compare_vln_followers.py` | evaluation summaries |
| `Isaac/live.sh`, `process.sh`, `progress.py` | live GUI demos, run progress viewer |
| `tools/build_b2w_z1_gripper_asset.py`, `tools/make_z1_gripper_urdf.py`, `Isaac/usd_fixes.py` | robot asset build |

### How to run

```bash
# headless fetch, three tables, Nav2, random targets, with delivery (from ~/Firdavs/Isaac)
NAV2_MAP=warehouse3 ~/Firdavs/nav2/run_nav2.sh
ISAAC_IMAGE=isaac-lab:2.3.2 ./run_script.sh fetch_box.py --device cpu --loco eth --nav2 \
    --task RobotLab-Isaac-Fetch3-Warehouse-Unitree-B2W-Z1-Gripper-v0 --target auto \
    --episodes 12 --seed 11 --random_start=0.9,7.4,-7.5,9.5 --deliver --out renders/x.json
#   + --vln "x" --vln_follower follow      language walk first (needs the DualVLN server)
#   + --record datasets/raw/<run> --task_variants      record VLA data
python3 summarize_fetch.py renders/x.json

# convert and review
~/Firdavs/.venvs/lerobot/bin/python tools/raw_to_lerobot.py --raw datasets/raw/<run> \
    --out datasets/lerobot/<run> [--only_success] [--only_right_table] [--no_depth]
~/Firdavs/.venvs/lerobot/bin/python tools/plot_episode.py datasets/raw/<run>

# live (workstation desktop)
SCENE=3 NAV2=1 ./live.sh fetch          # type: "go to the green table and pick the red box"
./process.sh                            # progress, another terminal
~/Firdavs/nav2/rviz.sh                  # Nav2's map, costmaps, lidar, plan
```

Container outputs are written as uid 1234: new output folders must be `a+rwX`.

---

## 9. Limitations and next steps

**Limitations:**
- Oracle perception: the box pose, and the dock pose Nav2 uses, come from the simulator.
- The pick is scripted.
- DualVLN reaches the table on its own about half the time.
- No lighting or distractor variety.
- The ETH fine-tune lost sideways driving from standstill.
- Single-robot CPU simulation, so recording is slow: about 3–10 min per episode with
  DualVLN and Nav2.

**Next steps, in order of value for the end goal:**
1. **Record at scale** in the three-table scene, in format 2: a few hundred to 1000 episodes,
   Stage 1 (scripted + Nav2) and Stage 2 (DualVLN + Nav2).
2. **Fine-tune π0.5 (openpi) on Slurm**, with the whole-body action
   (base velocity + EE pose + gripper), and evaluate it closed-loop in the same scenes.
3. **Perception instead of the oracle:** a detector on the front / wrist camera gives the box
   and table pose.
4. **Optional:** rfm19 whole-body reaching in place of the IK pick (plan kept); localisation
   (AMCL / slam_toolbox) for the real robot.
