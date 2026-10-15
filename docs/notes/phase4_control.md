> Development record, written during the project on the lab workstation. Paths such as `~/Firdavs/Isaac/...`
> refer to the layout before this repository: `Isaac/` is now `isaac/` + `project/` (+ `evaluation/`, `training/`),
> `repos/robot_lab` holds the tasks that are now in `training/robot_lab_overlay/`.

# Phase 4 — Control, Navigation and Loco-Manipulation: complete record

Unitree B2W wheeled quadruped + Z1 6-DoF arm, NVIDIA Isaac Sim 5.1 / Isaac Lab
2.3.2, reinforcement learning with RSL-RL PPO. Work done 21–27 September 2026.

This file combines the four Phase 4 documents into one. **Section 0 is new and is
the current state**; the four parts after it are the original documents, unchanged
except that their headings are one level deeper. They were written at different
points in the week, so where they disagree, Section 0 and the later parts win.

---

## 0. Status as of 9 October 2026

### What Phase 4 achieved

- **Low-level control:** a single RL policy for all 22 joints (12 leg, 4 wheel,
  6 arm) that drives, turns and reaches at the same time.
- **A to B:** waypoint follower, 16/16 arrivals on a 12 m square.
- **The body helps the arm:** with targets fixed in the world, the robot crouches
  ~30 cm and leans to reach low targets — the hand cannot go more than 8.6 cm
  below its own shoulder, so low reach is won by the body.
- **One-command 6D loco-manipulation:** give a hand position *and* orientation
  anywhere within 3 m at any bearing, and one policy decides whether that means
  turning round, driving, crouching or extending the arm — no velocity command,
  no scripted mode switch. Implemented from Jiang et al., RA-L 2024
  (arXiv:2412.03012), then modified with a shoulder-centred workspace-sphere phase
  criterion and compared against the paper's own criterion.
- **A still hand:** the arm shook on target (~15 mm wobble) because the policy
  drove it bang-bang, an answer to randomised arm gains and masses it could not
  observe. Incremental arm commands removed it (0.1 mm) and improved precision
  (Part 4 §6h).

### Timeline

| date | step | where |
|---|---|---|
| 21–22 Sep | Isaac Lab + robot_lab in Docker; B2W and B2W+Z1 assets fixed; keyboard teleop; warehouse scene | Part 1 |
| 22 Sep | arm as cargo; arm reach policy (1.4 cm); two policies run side by side | Part 1 |
| 22–23 Sep | one whole-body policy (wb1–wb8); forward-speed ceiling traced to wheel damping | Parts 1, 2 |
| 23 Sep | world-frame targets (wb9): body crouches 29 cm to help the arm; waypoint follower | Part 1 |
| 23–24 Sep | workspace measured; posture and lidar-mast clearance (wb12–wb15); why physical self-collision can't be enabled | Part 3 |
| 24 Sep | coordinates typed while the sim runs (`goto.py`); free camera; `show.sh` | Part 3 |
| 24–25 Sep | Reward Fusion Module from the paper; five reward bugs; distance curriculum; 6D omni-directional targets | Part 4 §1–6d |
| 25 Sep | faults seen on screen fixed (squat, base shuffling, lidar, roll-over); sit-back diagnosed; shoulder-sphere phase | Part 4 §6b–6e |
| 26 Sep | controlled comparison, sphere vs paper; facing, speed profile, anti-shake; the reward-valley finding | Part 4 §6f–6g |
| 8–9 Oct | shaking hand diagnosed (bang-bang arm) and cured with incremental arm commands (rfm19) | Part 4 §6h |

### Which policy to use

Checkpoint paths are relative to `repos/robot_lab/logs/rsl_rl/`.

| use | policy | checkpoint | task id (prefix `RobotLab-Isaac-`) |
|---|---|---|---|
| driving, A to B | wb8 | `b2w_z1_wholebody_sharptrack/2026-09-23_09-19-26_wb8/model_8500.pt` | `WholeBody-SharpTrack-Unitree-B2W-Z1-v0` |
| scripted drive-then-reach (`goto.py`) | wb9 | `b2w_z1_wholebody_worldtarget/2026-09-23_13-25-50_wb9/model_7700.pt` | `WholeBody-WorldTarget-Unitree-B2W-Z1-v0` |
| standing reach, tidy legs, clear of the lidar | wb15 | `b2w_z1_wholebody_tidy/2026-09-24_06-32-21_wb15/model_9500.pt` | `WholeBody-TidyFirm-Unitree-B2W-Z1-v0` |
| one-command 6D, most accurate (but reverses to targets and shakes) | RFM | `b2w_z1_rfm/2026-09-25_18-18-29_rfm10_height/model_27600.pt` | `RFM-Unitree-B2W-Z1-v0` |
| **one-command 6D, shoulder-sphere phase, still hand** (`./live.sh goto_mine`) | RFM-Delta-Sphere (rfm19) | `b2w_z1_rfm/2026-10-08_19-58-51_rfm19_delta_sphere/model_37600.pt` | `RFM-Delta-Sphere-Unitree-B2W-Z1-v0` |
| **one-command 6D, paper phase, still hand** (`./live.sh goto_paper`) | RFM-Delta-Paper (rfm19) | `b2w_z1_rfm/2026-10-08_20-28-32_rfm19_delta_paper/model_36700.pt` | `RFM-Delta-Paper-Unitree-B2W-Z1-v0` |
| previous sphere / paper (arm shakes) | RFM-Sphere / RFM-Paper (rfm14) | `b2w_z1_rfm/2026-09-26_04-38-25_rfm14_{sphere,paper}/model_33800.pt` | `RFM-Sphere-…` / `RFM-Paper-…` |

Interactive 6D targets for any RFM policy:
`PLAY_SCRIPT=/workspace/Isaac/rfm_goto.py CONTAINER_NAME=isaac-play ./run_train.sh --play --stream --task <task> --checkpoint /workspace/repos/robot_lab/logs/rsl_rl/<checkpoint>`,
then type `x y z` or `x y z roll pitch yaw`.

### Sphere vs paper, where it stands

Measured after 1,100 iterations each from the same checkpoint (Part 4 §6g): the
**sphere keeps the arm in the middle of its workspace** (target 0.32 m from the
shoulder vs 0.56 m — the paper's robot stretches the arm to high targets), is more
precise (0.7 vs 1.2 cm) and covers more of the reach map (45 vs 30 of 84). The
**paper holds the arm calmer** (1.9 vs 3.2 rad/s on target) and sits back less.
Both stopped driving backwards (93–96 % → 0–14 %).

Both then trained to iteration **33,800**, where they were stopped externally on
26 Sep 17:37 (not a crash; checkpoints intact). Training-average hand error was
0.093 m for both and still falling slowly (−4 to −7 % over the last 1,000
iterations) — nearly converged.

On 8–9 October both were superseded by **rfm19** (incremental arm commands,
§6h), fully evaluated: sphere 11/12 mid-height and 12/12 high targets, reach map
56/84 at 0.6 cm; paper 5/12 mid-height (stops 6–11 cm short, shoulder too close
to the target) and 12/12 high, 54/84 at 1.6 cm. Both hold the hand still
(0.0–0.1 mm on target, orientation 0.05 rad).

### Open problems

1. **Mid-height (0.45 m) targets** — solved for the sphere in rfm19 (11/12,
   median 2.3 cm); the paper phase rule still stops 6–11 cm short (5/12).
2. **Arm shaking on target** — solved in rfm19 (wobble 15 mm → 0.1 mm), §6h.
3. **Physical self-collision is off:** two collider pairs overlap by construction
   and jam at 20–25 kN (Part 3 §5); clearance from the lidar mast is enforced by a
   reward penalty instead.
4. Flat ground only; no perception or obstacle avoidance; simulation only.

### Lessons worth carrying into Phase 5

- **Watch the robot, not only the numbers.** A policy scoring 11/12 at 3.4 cm
  squatted, shuffled its base instead of using its arm, and fell over sideways;
  only the video showed it.
- **With a fused phase reward, compare the two groups' idle values.** If standing
  still in one phase pays more than starting the task in the other, the policy
  parks at the phase boundary.
- **Reward progress, not states.** A reward for *facing* a target made the robot
  park facing it; an *unsigned* closing-speed reward was farmed by cycling in and
  out forever.
- **Measure the test, too.** Re-placing a target relative to a driving robot every
  step scored a good policy at 10 %; a pitch sign error nearly reversed a
  conclusion.
- **Find why a controller needs to shake before penalising it.** The arm's
  bang-bang was the policy's answer to randomised gains and masses it could not
  see; penalties alone either left it or bought calm with precision.
  Incremental arm commands made calm the default.
- **Instrument fused rewards.** `Isaac/probe_rfm.py`, which prints every
  component, found most of the reward bugs.

### Document map

| part | original file | covers |
|---|---|---|
| 1 | `PHASE4_SUMMARY.md` | results overview up to 24 Sep: setup, whole-body policy, world targets, waypoints |
| 2 | `PHASE4_OVERNIGHT.md` | the wb1–wb8 study of the forward-speed ceiling |
| 3 | `PHASE4_WORKSPACE.md` | measured workspace, self-collision geometry, posture and clearance (wb12–wb15), `goto.py` |
| 4 | `PHASE4_RFM.md` | the Reward Fusion Module, its bugs, and the sphere-vs-paper comparison |

---


---

## Part 1 — Phase 4 — Control and Navigation Baselines: results

*Original file: `PHASE4_SUMMARY.md`*

Unitree B2W (wheeled quadruped) + Z1 arm, NVIDIA Isaac Sim 5.1 / Isaac Lab 2.3.2,
reinforcement learning with RSL-RL PPO. Written 23 September 2026.

Companion documents: `PHASE4_OVERNIGHT.md` (the eight-run whole-body study),
`claude project/PROJECT_CONTEXT.md` (the original plan).

---

### 1. What Phase 4 set out to do, and what happened

The plan asked for three things: a working low-level controller, spatial mapping,
and driving from A to B. Mapping was deferred by agreement. The other two are
done, and the arm was added on top, which was not in the original plan.

| Goal | Status |
|---|---|
| Low-level locomotion control | **done** — RL policy, drives and turns on command |
| Drive from A to B | **done** — waypoint follower, 16/16 arrivals on a 12 m square |
| Spatial mapping (octomap) | deferred |
| *Added:* arm reaching while driving | **done** — one policy, 22 joints |
| *Added:* reach a point in the room | **done** — world-frame targets, robot crouches to reach low |

---

### 2. The approach: one change at a time

Every step changed exactly one thing, so that any change in behaviour had exactly
one candidate cause. This caught several bugs that would otherwise have been
blamed on "the policy learning badly".

```
Rung 0b   wheels only, legs locked, keyboard          0.97 m/s  -- proves the hardware model
Rung 1    inherited policy (model_2600) drives        0.76 m/s
Rung 2    same policy, arm added as dead weight       unchanged -- arm costs nothing
Rung 3a   arm learns to reach, body held still        1.4 cm
Rung 3b   both policies running together              1.0-1.7 cm while driving
Rung 3c   ONE policy for all 22 joints                better at both
Rung 4    world-frame targets                         the body starts helping the arm
Rung 5    waypoint follower                           A to B
```

---

### 3. Headline results

#### 3.1 One policy beats two

Same driving script, same measurements, 8 robots each:

| | two separate policies | **one whole-body policy** |
|---|---|---|
| Hand error while driving | 1.0 – 1.7 cm | **0.6 – 1.1 cm** |
| Turn in place (1.0 rad/s asked) | 0.64 / 0.68 | **1.03 / 0.90** |
| Turn during an arc (0.6 asked) | 0.31 | **0.63** |
| Sideways (0.5 m/s asked) | 0.27 | **0.46** |
| Forward (1.0 m/s asked) | 0.76 | **0.80** |
| Falls | 0 / 8 | 0 / 8 |

#### 3.2 The body really does help the arm — once the target stops moving with it

Isaac Lab's stock hand target is defined in the robot's own frame, so it travels
with the robot: crouching moves the target down too, and the body can never buy
reach. Replacing it with a target fixed in the world changed the behaviour
completely.

| Target height (world) | body-frame policy: base height | **world-frame policy: base height** |
|---|---|---|
| 0.15 m | 0.585 m | **0.325 m** |
| 0.45 m | 0.589 m | 0.533 m |
| 0.95 m | 0.601 m | **0.616 m** |
| **total change** | **1.6 cm** | **29.1 cm** |

The world-frame policy lowers its whole body by nearly 30 cm to put its hand on a
low point, and holds 0.5 – 2.1 cm accuracy across the entire 0.15 – 0.95 m range.
This is the crouch-to-reach behaviour reported for whole-body MPC (Molnar et al.,
RA-L 2025) obtained here by reinforcement learning.

#### 3.3 Driving from A to B

Square course, 4 robots, arrival radius 0.30 m:

| Leg | Arrivals | Time | Path / straight line | Final error |
|---|---|---|---|---|
| (3, 0) | 4/4 | 3.3 s | 2.74 / 3.00 m | 0.27 m |
| (3, 3) | 4/4 | 4.3 s | 2.99 / 2.99 m | 0.25 m |
| (0, 3) | 4/4 | 4.1 s | 3.07 / 3.07 m | 0.25 m |
| (0, 0) | 4/4 | 4.2 s | 3.10 / 3.08 m | 0.25 m |

**16/16 arrivals**, paths within 2% of the straight line, ~0.75 m/s including turns.

#### 3.5 Tidy legs, a measured workspace, and an arm that stays out of the robot

The wb9 policy proved the science but was not presentable: its legs wandered, and
the arm swept straight through the front lidar mount. Three things were changed
together (all measured, see `PHASE4_WORKSPACE.md`): posture rewards that do **not**
reference the default stance, so they cannot forbid crouching; a keep-out penalty
on the arm entering the robot's own measured volume; and target ranges taken from
the measured workspace instead of chosen by hand.

| | wb9 (before) | wb12 | wb15 (recommended) |
|---|---|---|---|
| targets reached, 84-cell map | 62 (74 %) | **71 (85 %)** | 65 (77 %) |
| arm enters the robot's volume | **36 (43 %)** | 22 (26 %) | **9 (11 %)** |
| mean intrusion depth | 4.4 cm | 2.3 cm | 2.2 cm |
| median hand error | 1.5 cm | 1.7 cm | **1.5 cm** |
| vertical span reached | 110 cm | **130 cm** | **130 cm** |
| crouch (base travel, map) | 36.4 cm | 36.3 cm | 36.3 cm |
| unnecessary leg deviation at 0.75 m | 0.11 rad | **0.07 rad** | 0.08 rad |

The crouch survives the posture rewards, which was the point: on the height sweep
wb12 drops its body from 0.61 m to **0.307 m** for a 0.15 m target and holds
0.9-2.3 cm across the whole 0.15-0.95 m range (28.4 cm of travel, against wb9's
29.1 cm).

**The collision is reduced, not yet eliminated.** A fourth run (wb14) reached
0/84 intrusion but refused the two nearest columns of targets, because its penalty
shell was inflated past the shell the target sampler avoided -- it was being asked
to reach points it was punished for approaching. wb15 fixes that inconsistency and
takes intrusion to 13 % of cells. Removing the last of it means fixing the
collision geometry in the asset, which is written up as follow-up work.

---

#### 3.4 End to end: "put your hand on that point"

A world coordinate is given; the robot drives there and places its hand on it.

| Target (x, y, z) | Navigate | Hand error | Base height |
|---|---|---|---|
| (2.0, 1.5, 0.85) | arrived, 2.4 s | **0.7 cm** | 0.581 m (standing) |
| (2.5, 0.0, 0.25) | arrived, 2.0 s | 5.6 cm | **0.353 m (crouched)** |

---

### 4. Findings worth keeping

1. **The forward-speed ceiling was the wheel drive, not the learning.** Every
   policy reached ~75 % of a commanded forward speed. Six explanations were
   tested; only one moved it. robot_lab drives the wheels with damping
   1.0 N·m·s/rad, so torque = 1.0 × (target − actual): holding any load requires a
   standing velocity error. At damping 20, speed went 0.76 → 0.80 m/s. Direct
   wheel commands reach 0.97 m/s, so a gap remains, but it is now measured rather
   than mysterious.
2. **Isaac Lab drives un-actuated joints to zero, not to their initial pose.**
   Joints without an action term keep a position target of 0
   (`articulation.py:1641`). For held legs that means "straight", which launched
   the robot 30 cm into the air and tipped it over within a second, in every
   episode. Fixed with a reset event that holds them at their default pose.
3. **A latent index bug in robot_lab** (`joint_pos_rel_without_wheel`) only works
   for a bare 16-joint robot: it zeroes table *columns* using joint indices from
   the *whole* articulation. With the arm attached those indices run off the end
   (CUDA "index out of bounds"); had they landed in range it would have silently
   zeroed leg readings instead.
4. **Reward weights are not the same lever as reward tolerance.** Doubling the
   tracking weight changed nothing and degraded everything else; sharpening the
   same reward's tolerance (its exp kernel width) improved turning markedly.
5. **Ask for the distribution you intend to use.** Training samples forward and
   sideways speed independently, so full-speed straight-ahead driving is ~1 % of
   samples. Narrowing the sideways range did not fix the ceiling, but it showed
   the cost of the reverse mistake: a policy trained only to ±0.3 m/s sideways
   manages 0.07 of a 0.5 m/s command.

---

### 5. What exists now

**Robot asset** — `robots_usd/b2w_z1_rl.usd`, built by
`tools/build_b2w_z1_rl_asset.py` from the hand-placed `b2w_z1.usd`. Fixes: arm
mount joint follows the arm's actual placement, single articulation root, arm PD
gains, patched wheel drives, mass on a massless link. Verified under physics by
`Isaac/check_b2w_z1.py`.

**Environment** — `isaac-lab:2.3.2` image (Isaac Lab 2.3.2 + robot_lab 2.3.2 on
Isaac Sim 5.1.0), built by `repos/Dockerfile.isaaclab`, launched by
`Isaac/run_train.sh`, sharing the GPU pin and mounts of the existing launchers.

**Tasks** (local additions to robot_lab, `unitree_b2w/` and
`manipulation/reach_b2w_z1/`):

| Task id | What |
|---|---|
| `...-Velocity-Flat-Unitree-B2W-Z1-v0` | driving with the arm as cargo |
| `...-Isaac-Reach-B2W-Z1-v0` | arm reaching, body held |
| `...-LocoReach-Flat-Unitree-B2W-Z1-v0` | two policies side by side |
| `...-WholeBody-{Flat,SoftPenalty,Best,SharpTrack}-...` | one policy, body-frame targets |
| `...-WholeBody-WorldTarget-...` | one policy, world-frame targets (the crouch result) |
| `...-WholeBody-WorldDrive-...` | world targets at full driving speed |
| `...-Velocity-Warehouse-Unitree-B2W-v0` | the B2W in the NVIDIA warehouse |

**Tools** (`Isaac/`): `workspace.py` (where the hand can reach, and which target
boxes are honest), `reach_map.py` (which targets a policy actually reaches, with
base height and pitch per cell), `probe_contacts.py` / `probe_pairs.py` (what is
touching what, and which pair is pushing), `check_keepout.py` (are sampled targets
outside the robot's own body), `run_script.sh` (run any of these in the container),
`inspect_task.py` (what a policy sees and commands),
`stand_test.py` (can the robot stand at all, zero actions), `eval_locomotion.py`
(fixed driving script, tracking/tilt/falls/hand), `crouch_test.py` (does the body
help the arm), `waypoint.py` (A to B), `demo_fetch.py` (drive then reach),
`play_task.py` (play any task), `loco_reach.py` (run two policies together).

**Policies** — `repos/robot_lab/logs/rsl_rl/`, measurements in `logs/eval/`.

| Use | Run | Checkpoint |
|---|---|---|
| Best driver | wb8 | `b2w_z1_wholebody_sharptrack/2026-09-23_09-19-26_wb8/model_8500.pt` |
| Best hand (body-frame) | wb1 | `b2w_z1_wholebody/2026-09-22_22-35-02_wb1/model_7200.pt` |
| World targets / crouching | wb9 | `b2w_z1_wholebody_worldtarget/*_wb9/model_7700.pt` |
| **Tidy + clearance (recommended)** | wb15 | `b2w_z1_wholebody_tidy/2026-09-24_06-32-21_wb15/model_9500.pt` |
| Widest reach coverage | wb12 | `b2w_z1_wholebody_tidy/2026-09-24_00-58-38_wb12/model_6200.pt` |

---

### 6. Limitations

- **Flat ground only.** The warehouse works for driving, but no policy has been
  trained or measured on rough terrain.
- **Position only.** The hand reaches points; orientation is not controlled,
  there is no gripper in the model, and nothing has been grasped.
- **The robot's own parts do not collide in physics.** Self-collisions are off,
  inherited from robot_lab's B2W settings, so the arm can pass through the lidar
  mast. Turning them on was tried and reverted: two collider pairs overlap by
  construction two joints apart, so PhysX (which filters only joint-adjacent
  pairs) jams them at 20-25 kN and the arm stops moving altogether -- see
  `PHASE4_WORKSPACE.md` section 5 for the measured pairs. wb12 trains the
  avoidance behaviour with a keep-out penalty instead; making the physics itself
  correct needs collider edits in the asset.
- **No obstacle avoidance.** The waypoint follower drives straight at its goal;
  the policy cannot see anything. That is what the deferred mapping work is for.
- **No fall recovery** in the whole-body policies: resets are upright, unlike
  robot_lab's B2W default.
- **Specialists, not one generalist — and merging them is not free.** wb8 drives
  well but cannot express a world target; wb9 reaches world targets and crouches
  but was trained to ±0.3 m/s and drives poorly (0.38 of a 1.0 m/s command, 1 fall
  in 8). **wb10 tried to merge them at full speed and failed in a specific,
  informative way**: it learned to drive and ignored the arm (hand error stuck at
  0.38 m where wb9 was at 0.07 m by the same iteration). At 1.0 m/s the robot
  passes every world-fixed target within about a second, so reaching almost never
  pays. Keeping targets nearby (re-drawing one the robot has driven past) is not
  enough — they also have to be *catchable*. **wb11 retried at ±0.6 m/s and reaching survived but learns much more slowly**:
  hand error 0.46 → 0.19 m over 2,100 iterations, against wb9's 0.041 m by
  iteration 2,000 and wb10's stuck 0.37 m. Stopped at 2,100 iterations when the
  agreed time ran out, so it is undertrained, not failed — the curve was still
  descending. The pattern across the three is consistent: the faster the robot is
  asked to drive, the harder world-frame reaching becomes, because a fixed point
  is passed sooner.
- **Sim only.** Nothing has been run on hardware, and the arm mount is a modelled
  position, not a measured bracket.

---

### 7. Suggested next steps

1. Finish the merge. wb10 (±1.0 m/s) failed, wb11 (±0.6 m/s) was learning but
   was stopped early at 2,100 iterations — resuming it is the cheapest next
   experiment, and its checkpoint is kept. If a single policy cannot do both, the honest answers are either a
   speed-dependent target (one that is placed where the robot will *be*, not
   where it *is*) or simply switching policies by mode: wb8 to travel, wb9 to
   reach on arrival — which is what `demo_fetch.py` already does in two phases.
2. Send targets from outside the simulation — a small command interface, then the
   ROS 2 bridge already proven in Phase 1/2, which is what Phase 5's VLA will use.
3. Obstacle avoidance: the deferred mapping work, so the waypoint follower stops
   driving through things.
4. Gripper and orientation, for actual manipulation (door handles, boxes).
5. Rough terrain and the reconstructed NuRec scenes.


---

## Part 2 — Overnight run, 22-23 September 2026: whole-body (option C)

*Original file: `PHASE4_OVERNIGHT.md`*

Eight training runs, each a single-variable experiment (about 12 hours of GPU). Everything is kept:
logs and checkpoints under `repos/robot_lab/logs/rsl_rl/`, measurements under
`repos/robot_lab/logs/eval/`.

### Headline

**The whole-body policy beats the two separate policies**, and by a lot on
turning. One brain drives and reaches at once, 22 joints, trained from scratch.

| Measured on the identical driving script | Two policies (option A) | **Whole-body wb1** |
|---|---|---|
| Hand error while driving | 1.0 - 1.7 cm | **0.6 - 1.1 cm** |
| Turn in place, +/-1.0 rad/s commanded | 0.64 / 0.68 | **0.85 / 0.89** |
| Turn during an arc, 0.6 commanded | 0.31 | **0.62** |
| Sideways, 0.5 m/s commanded | 0.27 | **0.40** |
| Forward, 1.0 m/s commanded | 0.76 | 0.76 (0.80 in wb7) |
| Falls | 0 / 8 | 0 / 8 |

The hand is also more precise than the standalone reach policy managed on a
completely still body (1.4 cm). Coupling the arm and the legs in one policy
helped both jobs.

**Best policy: wb8** for driving (turns 1.03/0.90, arc 0.63, forward 0.80);
**wb1/wb5** if hand precision matters most (0.6-1.0 cm vs 1.2-1.7). Every one of
them beats the two-policy setup. To drive wb8:

```bash
cd ~/Firdavs/Isaac
PLAY_SCRIPT=/workspace/Isaac/play_task.py ./run_train.sh --play \
    --task RobotLab-Isaac-WholeBody-SharpTrack-Unitree-B2W-Z1-v0 \
    --checkpoint /workspace/repos/robot_lab/logs/rsl_rl/b2w_z1_wholebody_sharptrack/<timestamp>_wb8/model_8500.pt \
    --num_envs 4 --real-time --stream
```

(That plays with automatic drive commands. Keyboard driving for the whole-body
policy would need a small addition to `loco_reach.py`, which runs two policies
rather than one.)

### The runs

| Run | Change from wb1 | Result |
|---|---|---|
| **wb1** baseline | - | best overall; converged by ~4000 iterations |
| **wb2** | forward tracking reward 3.0 -> 6.0 | forward still 0.75; turning, strafing and hand all WORSE. Rejected |
| **wb3** | wheel action scale 5.0 -> 10.0 | forward still 0.73; otherwise ~= wb1. Rejected |
| **wb4** | sideways command range +/-1.0 -> +/-0.3 | forward still 0.72; turns 0.86/0.84 and hand 0.7-0.9 cm are as good as wb1, but strafing collapsed to 0.07 (it was trained only to 0.3 and the test asks 0.5). Rejected as a ceiling explanation |
| **wb5** | action-rate, torque, power and acceleration penalties all halved | forward still 0.76, so rejected as a ceiling explanation -- but the BEST policy overall: turns 0.89/0.95 (wb1: 0.85/0.89), strafe 0.46 (wb1: 0.40), hand 0.7-1.0 cm. Only the arc turn is worse (0.54 vs 0.62) |
| **wb6** | wheel drive damping 1.0 -> 20 N*m*s/rad | **the only change that moved forward speed: 0.81** (best of all runs), strafe 0.48, turn right 0.92 -- but turn LEFT collapsed to 0.35 with roll up to 26 deg. Partly confirms the hypothesis |
| **wb7** | wb5 penalties + wb6 wheel damping | **best all-round**: forward 0.80, turns 0.89/0.93, strafe 0.46, arc 0.57, hand 1.0-1.3 cm, worst roll 8.8 deg. Keeps wb6's speed and fixes its left-turn collapse |
| **wb8** | wb7 + sharper tracking kernel (std 0.5 -> 0.25) | **best driver**: turns 1.03/0.90, arc 0.63, forward 0.80, strafe 0.46; hand slightly worse at 1.2-1.7 cm. Sharpening the kernel helped turning, not forward speed |

### The one thing that would not move: 0.76 m/s forward

Every policy measured - the inherited `model_2600`, the two-policy setup, wb1,
wb2, wb3 - reaches about **75% of a 1.0 m/s forward command**, while wb1 turns
at 85-89% of its command. It is not physical: in Rung 0b, commanding the wheels
directly drove the robot at **0.97 m/s**.

Hypotheses tested overnight:

1. **Reward ceiling** (wb2). The tracking reward is `exp(-err^2/0.25)`, so
   0.76 m/s still collects ~80% of the reward while closing the gap costs
   torque and action-rate penalties. Doubling the tracking weight should shift
   that trade. **Refuted**: forward stayed at 0.75 and everything else degraded.
2. **Action-scale ceiling** (wb3). Wheel commands are scaled by 5.0 rad/s, so
   1.0 m/s needs a policy output of 1.77 - deep in the tail of its output
   distribution. Doubling the scale makes it 0.89. **Refuted**: forward 0.73.
3. **Command distribution** (wb4). Training samples forward and sideways speed
   independently in +/-1.0, so a near-pure-forward, near-full-speed command is
   about 1% of samples: the policy barely practises what the test asks for.
   **Refuted**: forward 0.72. (It did show the cost of narrowing a command
   range: wb4 strafes at 0.07 where wb1 reaches 0.40. Train the distribution
   you intend to use.)
4. **Effort and smoothness penalties** (wb5). Action rate, torque, joint power
   and acceleration all grow with speed, while going from 0.76 to 0.95 m/s buys
   only ~0.2 more tracking reward (the exp kernel is already near its top).
   **Refuted**: forward 0.76. Worth keeping anyway -- it made everything else
   better.
5. **The wheel drive itself** (wb6). **Partly confirmed.** Four policy-side explanations are
   gone, which points at the actuator. The wheels use a velocity drive with
   damping 1.0 N*m*s/rad, so torque = 1.0 * (target - actual): holding any load
   requires a standing velocity ERROR, roughly 0.1 m/s per N*m of resistance at
   a 0.113 m wheel. Rung 0b, which hit 0.97 m/s, used damping 20 on the same
   wheels. wb6 raises it to 20 and changes nothing else: forward went 0.76 -> 0.81, the
   only movement in six runs. But 0.81 is still well short of the 0.97 m/s that
   direct wheel commands reached, so the drive is a contributor, not the whole
   explanation -- and this run picked up a left-turn asymmetry (0.35 vs 0.92
   right) with 26 deg of roll, which may be undertraining at 8400 iterations or
   may be real. wb7 combines it with wb5's softer penalties, and keeps the speed (0.80)
   while turning properly again -- so the asymmetry was wb6's alone, not the
   damping's.
6. **Reward tolerance** (wb8). Weight was ruled out, but not the SHAPE:
   `exp(-err^2/0.25)` still pays 0.92 at a 0.2 m/s error. wb8 sharpens
   std 0.5 -> 0.25. **Partly useful, but not the ceiling**: forward stayed at
   0.80 while turning improved to 1.03/0.90 and the arc to 0.63.

#### Where the ceiling investigation ended

Forward speed went **0.76 -> 0.80** and stopped, against 0.97 m/s from direct
wheel commands. Of six explanations, only the wheel drive moved it. The gap that
remains is real but small, and it is now a known, measured property rather than
a mystery: budget ~80% of commanded forward speed when building anything on top,
and prefer wb7/wb8's wheel damping of 20 over robot_lab's 1.0.

### Caveats

- Flat ground only; hand targets in the box in front of and above the body
  (x 0.30-0.60, y +/-0.25, z 0.30-0.55 in the body frame).
- Hand position only - no orientation, no gripper, nothing grasped.
- Self-collisions are still disabled on the robot, so the arm can pass through
  the body. Targets are above the body, so it has little reason to.
- Whether the body actually LEANS or CROUCHES to help the arm (the ETH paper's
  behaviour, and the reason to train one policy) is **not yet verified**. The
  numbers say the coupling helps; watching it will say whether it looks like
  whole-body motion or just a better-tuned pair of skills.
- Two rewards were deliberately relaxed to allow that behaviour: `stand_still`
  switched off and `joint_pos_penalty` reduced to a quarter weight.
- Resets are upright here, unlike robot_lab's B2W default of any orientation.
  This policy has NOT been trained to recover from falls.

### Suggested next steps

1. Watch wb5 (or wb1) and judge whether the body helps the arm.
2. The wheel-damping finding applies to the plain locomotion task too: the
   inherited `model_2600` was trained against damping 1.0. Retraining B2W
   locomotion with damping 20 would likely lift its speed as well.
3. Phase 4's stated goal is still open: the A-to-B waypoint follower. It should
   be built on whichever policy wins, and must compensate for the ~75%
   speed shortfall.


---

## Part 3 — How far can it reach, and what stops it? — measurements behind wb12

*Original file: `PHASE4_WORKSPACE.md`*

Unitree B2W + Z1, Isaac Sim 5.1 / Isaac Lab 2.3.2. Written 24 September 2026.

This document backs three numbers used elsewhere: the **reachable workspace**, the
**self-collision geometry**, and the **target ranges** wb12 trains on. Companion
documents: `PHASE4_SUMMARY.md` (results), `PHASE4_OVERNIGHT.md` (the wb1–wb8 study).

---

### 1. Why measure at all

Up to wb11 the hand target was sampled from a box chosen by hand:
`x 0.25–0.60, y ±0.30, z 0.15–0.95`. That box was a guess from "the arm is about
0.69 m long". Two things are wrong with guessing here:

- an **unreachable** target cannot be reached however well the policy learns, and
  a policy trained on many of them learns to ignore the arm. That is not
  hypothetical: it is exactly how wb10 failed;
- a **too small** box hides what the robot can do, which is the thing the whole
  world-frame experiment exists to demonstrate.

So the workspace is now measured three ways, and they disagree in an informative
direction.

---

### 2. Method 1 — the arm alone, by forward kinematics

`Isaac/workspace.py`. The base is pinned level in free air and 25,000 arm
configurations are drawn uniformly inside the **real USD joint limits**:

| joint | limit |
|---|---|
| joint1 | ±150° |
| joint2 | 0 … +170° |
| joint3 | −165° … 0 |
| joint4 | ±87° |
| joint5 | ±77° |
| joint6 | ±160° |

The hand (`link06`) position is recorded in the base frame:

| axis | range in the base frame |
|---|---|
| x (forward) | −0.504 … +0.836 m |
| y (left) | −0.677 … +0.678 m |
| z (up) | **−0.086** … +0.881 m |
| distance from the mount | max **0.694 m** |

The important entry is `z min = −0.086`. **The hand cannot drop more than 8.6 cm
below its own shoulder**, because joint2 and joint3 cannot swing back past the
shoulder (0…+170° and −165°…0). Reaching low is therefore not an arm problem at
all — it is a body problem, which is the whole premise of the task.

### 3. Method 2 — add the body, as a level lift

Sweeping that cloud over the base heights the trained policy actually uses
(0.325–0.616 m, from `crouch_test.py` on wb9) gives a reachable world volume of
roughly **z 0.33 … 1.45 m**, and, crucially, a floor: with a **level** body the
hand bottoms out at **0.239 m** and cannot go lower.

Scoring candidate target boxes against that volume — the fraction of a uniformly
sampled box that is actually reachable — gives:

| candidate box | x | z | reachable |
|---|---|---|---|
| **wb9's hand-chosen box** | 0.25–0.60 | 0.15–0.95 | **69.6 %** |
| per-axis 2–98 % envelope | 0.20–0.74 | 0.33–1.45 | 83.9 % |
| world-cloud 2–98 % | 0.25–0.60 | 0.43–1.38 | 91.8 % |
| full assisted span | 0.25–0.65 | 0.10–1.20 | 73.9 % |

So **roughly a third of wb9's training targets were unreachable by this model**.

One trap worth naming: quoting each axis' 2–98 % range separately describes a
*box*, but the workspace is a *shell*. The arm reaches 0.75 m forward, and it
reaches 1.43 m up, and it does **neither at the same time**. Sampling that box
independently — which is what `UniformPoseCommand` does — manufactures exactly
the unreachable corner targets we are trying to avoid.

### 4. Method 3 — ask the robot (and the model turns out to be too pessimistic)

`Isaac/reach_map.py` gives every environment its own target from a grid, so a
6 × 14 grid is measured in one 5-second settle. Run on wb9 (`model_7700.pt`):

```
hand error, cm            forward distance, m
              0.25   0.35   0.45   0.55   0.65   0.75
   z 0.10     ...     2.2    0.9    1.0    1.5    1.7     <- model says IMPOSSIBLE
   z 0.50     3.2     1.4    1.5    1.2    1.6    1.2
   z 0.90     1.8     0.7    0.6    0.6    1.5    1.5
   z 1.20     5.7X    5.0    4.9    4.9   23.3X  30.9X
   z 1.40    10.8X   10.6X  65.1X  35.7X  68.8X  77.1X
```

**62/84 cells reached within 5 cm**, median error **1.5 cm**, over a vertical span
of **0.10 – 1.20 m (110 cm)**.

The 0.10 m row should be impossible: the level-body model floors at 0.239 m. The
pitch column explains it —

| target height | base height | body pitch |
|---|---|---|
| 0.10 m | 0.300 m | **−10.7°** (nose down) |
| 0.50 m | 0.409 m | +0.5° |
| 1.40 m | 0.661 m | +6.4° |

The robot **leans as well as crouches**: 36.4 cm of base travel *and* up to 10°
of nose-down pitch. The geometric model underestimates the workspace precisely
**because** it assumed a level body — which is itself further evidence for
body-assisted reaching, obtained accidentally.

Lesson recorded: the geometric model is a conservative *core*, not the boundary.
Where the two disagree, believe the robot.

---

### 5. The self-collision geometry, and why `enabled_self_collisions=True` failed

The arm visibly sweeps **through** the front lidar mount. The obvious fix is to
turn self-collisions on. It was tried, measured, and reverted.

**The collision geometry** (from the USD, `Isaac/workspace.py`, base frame):

| part | volume |
|---|---|
| body (`base_link` box) | \|x\| ≤ 0.25, \|y\| ≤ 0.14, \|z\| ≤ 0.075 — only 0.50 × 0.28 × 0.15 m, much smaller than the visible shell |
| lidar mast (capsule, r 0.076) | x 0.27–0.42, \|y\| ≤ 0.077, z 0.00–0.34 |
| arm | colliders on link00, link02, link03, link04, link06 — **link01 and link05 have none at all** |

The mast sits directly in front of the shoulder, which is why the arm meets it
constantly.

**What happened when self-collisions were enabled.** PhysX only auto-filters
collider pairs joined *by a joint*. This robot has two pairs that overlap by
construction while being **two** joints apart, so they jam from the first step.
Measured pairwise with `Isaac/probe_pairs.py`:

| pair | force | why they overlap |
|---|---|---|
| `head_Link` ↔ `lidar_link` | **24.6 kN** | both bolted to the base, two joints apart |
| `link00` ↔ `base_link` | **19.2 kN** | the arm's base is sunk into the body by design |

Cross-check that the geometry above is right: `link00`'s capsule spans z
0.054–0.170 and the body box tops out at 0.075 — they overlap, exactly as the
contact force says.

The consequence was not subtle. With the arm jammed, training learned nothing:
hand error sat at **0.55 m** after 70 iterations where wb9 was at 0.04 m by
iteration 2000, and `arm_self_contacts` never improved from −3.4. Running wb9's
own policy in that environment scored **0/84** cells with the arm frozen.

Reassuringly, the pairs we actually *want* — the arm's moving links against the
mast — read **0.0 N at rest**. The geometry is fine; only those two mounts are
wrong. Fixing it properly means editing collision geometry in the asset (disable
the decorative `head_Link` collider, filter `link00`↔`base_link`), which is
follow-up work, not an overnight change.

**What wb12 does instead** (`self_clearance.py`): a penalty on the arm's moving
links (link03–link06) entering the measured body and mast volumes, plus a target
sampler that refuses to place a target inside them. It changes no physics, so it
cannot destabilise anything, and it trains the behaviour that was asked for —
route around, not through.

---

### 6. The ranges wb12 trains on

```python
pos_x = (0.25, 0.65)     # arm is 0.694 m; 0.65 keeps a margin
pos_y = (-0.30, 0.30)
pos_z = (0.15, 1.25)     # 0.15 needs crouch AND lean; 1.25 needs standing tall
keepout = body box + lidar mast   # targets are never placed inside the robot
```

The low end is kept where wb9 proved it reachable (0.10–0.15 m, via lean), and the
ceiling is raised 0.95 → 1.25 m, which is the same skill mirrored: the robot must
stand tall rather than crouch. wb9 reached 1.20 m at 4.9 cm and failed at 1.30 m,
so 1.25 m is deliberately placed at the edge of the demonstrated envelope.

---

### 7. Tools added

| tool | what it answers |
|---|---|
| `Isaac/workspace.py` | where can the hand go, and which target boxes are honest |
| `Isaac/reach_map.py` | which targets does a trained policy actually reach (a map, with base height and pitch per cell) |
| `Isaac/probe_contacts.py` | which bodies are being pushed at the spawn pose |
| `Isaac/probe_pairs.py` | which **pair** is pushing — the one that named the two bad mounts |
| `Isaac/run_script.sh` | run any of these in the Isaac container with the standard GPU pin and mounts |

---

### 8. A hypothesis that was wrong: the action-rate weight

wb12 reached much more slowly than wb9 at first — hand error 0.374 m at iteration
400 against wb9's 0.137 m. The suspect was mine: the posture package doubled
`action_rate_l2` (−0.005 → −0.01), and that term taxes exactly the fast joint
motion the arm needs to chase a target resampled every 5 s.

wb13 tested it — identical to wb12 except the action-rate weight back at −0.005,
one variable:

| iteration | wb12 (−0.01) | wb13 (−0.005) | wb9 |
|---|---|---|---|
| 400 | 0.374 | 0.279 | 0.137 |
| 600 | 0.278 | 0.274 | 0.085 |
| 800 | 0.213 | 0.166 | 0.084 |
| 900 | 0.185 | 0.171 | 0.080 |

**Refuted.** wb13 is marginally ahead but nowhere near closing the gap to wb9, so
the action-rate weight is not what delayed reaching. What the curves actually show
is a *later phase transition*: all three runs sit together near 0.35 m at
iteration 300, then wb9 drops sharply (0.355 → 0.137 between 300 and 400) while
wb12 holds its plateau ~200 iterations longer before making the same drop —
0.185 (900) → 0.134 (1000) → 0.071 (1500), against wb9's 0.049 at 1500.

The honest reading is that wb12's task is simply harder than wb9's, in three ways
it was *asked* to be harder: a target box with a 1.1 m vertical span instead of
0.8 m, an arm that must route around the mast rather than through it, and posture
terms constraining the legs. Discovery takes longer; the converged result is what
matters. wb13 was stopped at iteration 900 once it had answered its question, and
its checkpoints are kept.

---

### 9. Results: wb12, wb14, wb15

All measured with `Isaac/reach_map.py` on the same 6 × 14 grid (84 targets, one
per environment, 5 s settle, 5 cm tolerance), and `crouch_test.py` /
`eval_locomotion.py` for the rest. "Arm enters the robot" is the geometric
intrusion metric: any of link03–link06 inside the measured body or mast volume.

| | wb9 (before) | wb12 | wb14 | wb15 |
|---|---|---|---|---|
| clearance weight / margin | none | −15 / 0.04 | −50 / **0.08** | −50 / 0.04 |
| targets reached | 62/84 (74 %) | **71/84 (85 %)** | 54/84 (64 %) | *see below* |
| arm enters the robot | 36/84 (43 %) | 22/84 (26 %) | **0/84 (0 %)** | |
| mean intrusion depth | 4.4 cm | 2.3 cm | **0.0 cm** | |
| median hand error | 1.5 cm | 1.7 cm | 2.0 cm | |
| crouch (base travel) | 36.4 cm | 36.3 cm | 36.3 cm | |
| vertical span reached | 110 cm | **130 cm** | 120 cm | |

**wb14's lesson is worth more than its policy.** It removed the collision
completely, but lost the two nearest columns of the map — x = 0.25 and 0.35 went
from 1.5–9 cm error to 10–25 cm. That was not a trade-off, it was a contradiction
I introduced: the shell the *reward* punished was inflated to 8 cm while the shell
the *target sampler* avoided stayed at 4 cm, so targets were being placed inside
the penalised zone. Refusing to reach them is the correct response to a reward
that punishes what it also asks for. wb15 keeps weight −50 and puts the margin
back to 0.04 so the two shells are the same volume.

#### Driving, and an honest regression

`eval_locomotion.py` at `--speed 0.3`, the range these policies are trained for
(at the script's default 1.0 they are extrapolating 3× and fall — wb12 fell 6/8
there, which measures the test, not the policy):

| | wb9 | wb12 |
|---|---|---|
| forward, 0.3 commanded | 0.14 | **0.21** |
| turn, 0.3 commanded | **0.14–0.17** | 0.07–0.09 |
| hand error while driving | 0.8–1.3 cm | 1.1–3.5 cm |
| falls | 0/8 | 0/8 |

wb12 drives forward better and **turns noticeably worse**, and that shows up
downstream: on the waypoint square it reached 11/16 corners, because the follower
asks for a turn the robot only partly delivers. The posture terms are the likely
cause — hip deviation and diagonal symmetry both resist the asymmetric leg
postures that help a wheeled quadruped rotate.

This does not undo Phase 4's A-to-B result, which was obtained with **wb8**, the
dedicated driver. It reinforces the split `demo_fetch.py` already uses: wb8 to
travel, a world-target policy to arrive and reach.

---

### 10. Did the legs actually get tidier?

"The robot moves weirdly" is a visual complaint, so it needs a number. The one
that fits is the **residual leg deviation from the default stance** at target
heights where no crouch is needed — mean |q − q_default| over the twelve leg
joints, from `crouch_test.py`:

| target height | wb9 | wb12 | wb15 |
|---|---|---|---|
| 0.15 m | 0.41 | 0.43 | 0.43 |
| 0.25 m | 0.32 | 0.34 | 0.35 |
| 0.35 m | 0.24 | 0.25 | 0.24 |
| 0.55 m | 0.14 | **0.09** | 0.11 |
| 0.75 m | 0.11 | **0.07** | 0.08 |
| 0.95 m | 0.12 | **0.08** | 0.10 |

Read it in two halves. At the **low** targets all three deviate by the same large
amount — that is the crouch itself, and the posture terms were deliberately
chosen so they could not suppress it. At **normal** heights, where the legs have
no job except to stand, wb9 keeps 0.11–0.14 rad (6–8°) of deviation it does not
need, while wb12 settles back to 0.07–0.08 rad. That is roughly **40 % less
pointless leg motion**, and it is what "tidier" means here.

This is exactly what the posture package was designed to do: constrain the legs
where the task does not need them, without touching the crouch. Had
`stand_still` been restored instead, the first three rows would have been
flattened too — and the crouch with them.

---

### 11. Final numbers and which policy to use

wb15 trained to iteration 9500 (fine-tuned from wb12's 6200) and converged:
65/84 cells, 9/84 intrusion, median 1.5 cm, 130 cm span, 36.3 cm of body travel.

| | wb9 (before) | wb12 | wb14 | **wb15** |
|---|---|---|---|---|
| clearance weight / margin | — | −15 / 0.04 | −50 / 0.08 | −50 / 0.04 |
| targets reached | 62 (74 %) | **71 (85 %)** | 54 (64 %) | 65 (77 %) |
| arm enters the robot | 36 (43 %) | 22 (26 %) | **0 (0 %)** | 9 (11 %) |
| median hand error | 1.5 cm | 1.7 cm | 2.0 cm | **1.5 cm** |
| vertical span | 110 cm | **130 cm** | 120 cm | **130 cm** |
| leg deviation at 0.75 m | 0.11 rad | **0.07** | — | 0.08 |

**Use wb15** (`b2w_z1_wholebody_tidy/2026-09-24_06-32-21_wb15/model_9500.pt`). It
has the best hand accuracy, the full 130 cm vertical span, tidy legs, and a
quarter of wb9's collision rate.

Use **wb12** instead if reach coverage matters more than the last of the
collisions — it reaches 6 more of the 84 cells but enters the mast more than twice
as often.

Use **wb8** for driving. None of these is a good traveller: they are trained to
±0.3 m/s, and wb12 turns at only ~0.09 rad/s of a 0.3 command, which cost it the
waypoint square (11/16 corners). `demo_fetch.py`'s two-phase split — wb8 to
travel, a world-target policy to arrive and reach — remains the right structure.

#### What is left

1. **The last 11 % of collisions** needs the asset fixed rather than the reward:
   disable the decorative `head_Link` collider and filter `link00`↔`base_link`,
   then `enabled_self_collisions=True` becomes usable and the physics enforces
   what the penalty currently only encourages.
2. **Turning** regressed under the posture terms. The suspect is
   `joint_deviation_hip_l1` (−0.5): hip abduction is how a wheeled quadruped
   helps itself rotate. Halving it is the obvious next single-variable test.
3. The intrusion metric uses link **origins**, so a link whose origin is just
   outside the box can still graze it. A capsule-aware version would be stricter.

---

### 12. Sending a coordinate by hand: `Isaac/goto.py`

Targets can now be given WHILE the simulation runs, instead of as a command-line
argument fixed before launch — either typed at the prompt or written to
`Isaac/target.txt` from any other terminal (the same last-value-wins contract a
ROS 2 topic gives, which is what replaces it in Phase 5).

Two things had to be learned the hard way, both worth keeping:

1. **What the hand is told to do while driving decides whether the robot stays
   upright — and my first conclusion here was wrong.** Parking the hand target
   0.40 m in front of the robot made *wb15* sit with its wheels at exactly
   0 rad/s, and I concluded that parking was the problem and pointed the hand at
   the far goal instead. That was a misattribution: the stall is wb15's weak
   turning, and pointing the hand at a distant goal is actively dangerous.

   Measured, wb9, target 9.4 m away, hand aimed at it: base height collapses
   0.55 → 0.12 m and tilt spikes to 68° **within one second**. The arm stretches
   to full extension after a point the policy has never seen — training only ever
   sampled targets 0.25–0.65 m away — and the robot topples.

   The correct behaviour is to PARK the arm while driving (`--drive_arm park`,
   now the default) so the hand target stays inside the distribution the policy
   was trained on. Same robot, same 2.5 m target: drives at 1.37 m/s, tilt 8°,
   then crouches to 0.338 m and reaches to 1.1 cm.

2. **It needs wb9, not wb15.** With wb9: 2.0 m → 0.49 m, then a crouch to 0.37 m
   and 0.7–1.3 cm hand error. With wb15 the same script stalls around 1.3 m,
   drifting off heading with its wheels spinning — the turning regression in
   section 9, met head-on. Use wb15 for reaching quality and clearance, wb9 (or
   wb8) when the robot has to travel.


---

## Part 4 — Reward Fusion: implementing Jiang et al. on our robot

*Original file: `PHASE4_RFM.md`*

Unitree B2W + Z1, Isaac Lab 2.3.2. Written 24 September 2026.

Implementation of the Reward Fusion Module from **K. Jiang, Z. Fu, J. Guo, W.
Zhang, H. Chen, "Learning Whole-Body Loco-Manipulation for Omni-Directional Task
Space Pose Tracking with a Wheeled-Quadrupedal-Manipulator", IEEE RA-L 2024**
(arXiv:2412.03012), PDF in `papers/`. Their platform is a LimX W1 wheeled
quadruped with a 6-DoF AIRBOT arm — the same class of machine as ours, doing the
same job.

Companion documents: `PHASE4_WORKSPACE.md` (workspace, posture, clearance),
`PHASE4_SUMMARY.md` (results so far).

---

### 1. Why this paper, and what it was meant to fix

Two failures of our own policies motivated it:

1. **Handed a target metres away, the robot stretched the arm at it and fell** —
   base height 0.55 → 0.12 m and 68° of tilt within a second. Reaching and
   driving were simply *added*, so reaching always paid something, and the policy
   had never seen a target it could not reach.
2. **It traded balance for reach.** Nothing stopped it risking a fall to shave a
   centimetre off the hand error.

RFM addresses both by changing *how rewards combine*, not by adding more of them.

### 2. What was implemented (`rfm.py`)

| mechanism | what it does |
|---|---|
| **Reward Prioritization** | rewards multiply: `r_mani = r_reg + r_reg·(r_ep + r_ep·r_eo)`. Orientation pays only once position is good; both pay only once stability is good. A policy about to fall earns **nothing** for chasing the target |
| **Loco-Mani Fusion** | a phase variable `D` blends whole reward groups: `r = (1−D)·r_mani + D·r_loco`. Far → locomotion, near → manipulation, continuously. Replaces the hand-written state machine in `goto.py` |
| **Micro-enhancement** | `r* = r + r^M` sharpens the gradient only where error is already small |
| **Cumulative penalty** | PID-style integral of tracking error, to escape local minima |
| **6D targets** | orientation is commanded and rewarded; the policy sees the target as a flattened rotation matrix + position (12 numbers), the paper's representation |
| **Omni-directional** | targets at a uniform bearing 0.3–3.0 m around the robot, so turning round is part of the task |
| **No velocity command** | the paper's key point: **one 6D end-effector command**, and the policy decides whether that means driving, reaching, or both |

Task id: `RobotLab-Isaac-RFM-Unitree-B2W-Z1-v0`. Observations 87 (critic 90),
actions 22.

### 3. What the paper leaves out, and what we had to decide

**Regularisation is deliberately undefined.** The paper: *"The specific meaning of
'regularization' varies according to the requirements of users and tasks … our
customized definition of 'regularization' is omitted."* It does state the
asymmetry, which matters: locomotion regularisation means *"stable base height
and flat base orientation"*, manipulation regularisation must **not**, *"because
in manipulation task, the base need to bend or stretch"*.

So `r_reg_loco` rewards a level body at nominal height, while `r_reg_mani`
rewards only smoothness (low angular rate, low vertical velocity) and says
nothing about posture — otherwise it would forbid the 36 cm crouch and 10° lean
our robot uses to reach low targets.

**`a1`, `a2` in the SE(3) distance are given no values.** This turned out to
matter a lot: `eps` drives the phase variable, whose midpoint is 1.4 m, so at
`a1 = 1` a half-turn of orientation error adds 3.14 to `eps` by itself and pins
the phase in "locomotion" forever. Position has to dominate (`a1 = 0.1`).

### 4. Five bugs, each found by measurement

Every one of these was caught by instrumenting the reward (`Isaac/probe_rfm.py`
prints all sixteen components) rather than by staring at the loss curve.

| # | symptom | cause | fix |
|---|---|---|---|
| 1 | total reward drifts structurally negative | I read Table II's −5 as a *termination* penalty; it is for **collision**. The `alive` bonus of +2.0 is what keeps the floor up | add `alive`, drop the termination penalty |
| 2 | phase pinned at "manipulation" while the target is 1.9 m away | rotation weighted 1.0 in the SE(3) distance | `a1 = 0.1` |
| 3 | `e_cb` climbs to 11 and buries every other term (`r_mani` −10.3) | cumulative penalty gain far too large for our scale | gain 0.002, clip 2.0 |
| 4 | policy stands perfectly still and ignores the target (0/12 approached, 0/12 falls) | `reg_loco` pays ~1.0 **every step** for standing level: doing nothing earns ~500 per episode, closing 3 m earned ~60 | add a dense **closing-speed** reward, widen `sigma_dw` 0.5 → 1.5 |
| 5 | `r_appr` high (0.84) while distance oscillates 1.3 → 1.9 → 1.5 m and never converges | closing speed clamped at `min=0`: approaching paid, drifting away was **free**, so cycling in and out farmed the reward forever | make it **signed** |

### 5. The curriculum the paper mentions in one line

*"We adopt the idea of Curriculum Learning and Decaying Entropy Coefficient."*
Skipping it cost a whole training run. Trained directly on 0.3–3.0 m targets the
policy learned the stable half of the task and never learned to approach at all.

`target_radius_curriculum` starts targets at arm's length (0.3–1.0 m), where the
robot already knows how to succeed, and widens by 0.2 m whenever more than half
of finishing episodes ended within 20 cm. With the signed approach reward it
reaches the 3.0 m ceiling by iteration ~600.

### 6. Where it stands

Measured with `Isaac/approach_test.py` — targets at 1/2/3 m and four bearings,
nothing steering but the policy itself:

| | wb15 (before) | RFM @600 iters |
|---|---|---|
| lunge-and-topple on a far target | 0.55 → **0.12 m**, 68° tilt | **eliminated** at 0° and 180° |
| median closest approach | never approaches | **49 cm** |
| turns round for a target behind it | no | **yes** — 3.0 m → 52 cm |
| falls | — | 6/12, all at ±90° bearings |

The structural claims of the paper reproduce: stability gating removes the
suicidal lunge, and the phase variable produces a genuine drive-then-reach
without any scripted mode switch. What is not yet solved at this checkpoint is
sideways approach (±90° rolls the robot) and the last half metre — the policy
stops where the approach reward switches off and the arm does not yet cover the
remainder.

### 6b. Four faults found by WATCHING it, not measuring it

The policy at 6,300 iterations scored well — 11/12 targets within 10 cm, median
3.4 cm — and still looked wrong on screen. Four specific complaints, each of
which turned out to be real and measurable once we knew to look:

| observed | measured | cause | fix |
|---|---|---|---|
| "dog lowers its body too much" | squats to **0.217 m** for a 0.45 m target, 9/12 cases below 0.25 m | manipulation regularisation deliberately ignores posture, so squatting is free | soft floor at 0.30 m, weight 150 |
| "base rotates to reach, arm moves less" | **0.99 rad** of base yaw travel on targets well inside arm's reach | nothing made holding the base still cheaper than shuffling it | penalise planar + yaw motion in the manipulation phase |
| "arm hits the lidar mount" | clearance term inherited at **-15**, not wb15's proven -50 | oversight | -50 with the matched margin |
| rolls over on sideways targets | 4/12 flips, tilt 130-154 deg, several AFTER arriving | I implemented only the base-height half of the paper's failure test | add the projected-gravity tilt termination |

**Result after the fixes** (iteration 10,000, resumed from 6,300):

| | before | after |
|---|---|---|
| falls | 4/12 | **0/12** (worst tilt 52 deg) |
| squats below 0.25 m | 9/12 | **0/12** (lowest 0.27 m) |
| base yaw travel | 0.99 rad | **0.33 rad** |
| arm into the lidar mast | — | **0/84** |
| reach map, fixed targets | — | **54/84**, median 2.1 cm |
| approach, median closest | 3.4 cm | 4.0 cm |

Worth keeping: **the qualitative report found things the metrics scored as
success.** Every number said the 6,300 policy was good. It squatted, shuffled its
base instead of using its arm, and fell over sideways — and only watching it
surfaced that.

### 6c. A measurement trap of my own

`reach_map.py` re-derived each target from the robot's pose on EVERY step. For a
policy that stands still that is harmless; for one that DRIVES, the target rides
along and the robot can never arrive. It scored the RFM policy at 8/84 when the
honest figure was 49/84. Fixed with `--fixed_targets`, which places each target
once. Any future evaluation of a driving policy needs that flag.

### 6d. Converged result (iteration 19,239)

| | value | paper (sim) |
|---|---|---|
| targets reached, 1-3 m, all bearings | 10/12, median closest **0.9 cm** | 99.0 % success |
| falls | **0/12** | — |
| squats below 0.25 m | **0/12** (lowest 0.25 m) | — |
| reach map, fixed targets | 53/84, median **1.5 cm** | EE position error 2.2 cm |
| orientation error | **0.091 rad** | 0.041 rad |
| arm vs base activity | arm 3.90 rad / base yaw 0.38 rad | — |
| arm into the lidar mast | **0/84** | — |

Individual approaches at 19,239 iterations: 0.1, 0.2, 0.2, 0.2, 0.9, 1.2, 1.4,
2.2, 4.1 cm — from up to 3 m away and any bearing.

**The one remaining flaw is asymmetric.** Targets at -90 deg (the robot's right)
and 2-3 m away are not approached at all (162 cm, 170 cm), while +90 deg works at
every distance and -90 deg works at 1 m. A left/right asymmetry like that is a
training artefact rather than a structural problem, and the obvious next check is
whether the omni-directional sampler is actually uniform in bearing.

Position accuracy is comparable to the paper; orientation is about twice as
loose, which is unsurprising given they train 7,000 teacher + 4,000 student
iterations with a privileged-state encoder and we train a single policy.

**What this policy does that nothing before it could:** one 6D end-effector
command, given anywhere within 3 m and at any bearing, and the robot decides for
itself whether that means turning round, driving, crouching or extending the arm
-- with no velocity command and no scripted mode switch. `Isaac/goto.py`'s
two-phase state machine is no longer needed.

Training continues; this document records the method and the traps, which are
the durable part.

### 6e. "It lowers its rear even for high targets" — the workspace sphere

The 19,300-iteration policy reached targets to sub-centimetre accuracy and still
looked wrong: it sat back on its haunches. Measured with the new pitch column in
`approach_test.py`, on **high** (1.0 m) targets:

| | 19,300-iteration policy |
|---|---|
| max pitch, nose up / rear down | median **+36 deg**, worst **+52 deg** |
| body height for a 1.0 m target | **0.27-0.36 m** |

It was reaching HIGH targets by crouching and tipping nose-up: the arm is mounted
at the front, so sitting back lifts the shoulder. "Sit up and beg" instead of
"stand up". Nothing in the manipulation phase penalised tilt, and the tilt
termination at 57 deg left plenty of room.

**The phase criterion, redesigned (the user's proposal).** The paper decides
locomotion vs manipulation from the HAND's SE(3) distance to the target, with the
switch at 2 x arm span (1.4 m). Two faults on our robot:
- the policy moves the hand itself, so it can shift its own phase by stretching
  the arm -- an incentive to work at full extension;
- 1.4 m is twice the Z1's reach, so manipulation rewards took over while the
  target was still out of reach.

Instead: a sphere of radius **k x reach (k = 0.5, 0.35 m)** centred on the
**shoulder** (link02, where the 0.694 m reach is measured from). Outside it,
locomotion; inside it, manipulation, through a narrow sigmoid rather than a hard
switch. The shoulder only moves when the body moves, and the arm never has to
work past half its reach. Centring on the shoulder rather than the mounting plate
matters: from the plate (10 cm lower), targets above ~1.10 m could never count as
inside at any body height.

Plus: tilt beyond 15 deg penalised in the manipulation phase, and the tilt
termination tightened 57 -> 34 deg.

**A deadlock this created, and its fix.** With the sphere, a 0.45 m target seen
from a standing robot is *vertically* outside it (shoulder 0.35 m above), so the
phase stays "locomotion" -- whose reward held the body at a fixed nominal 0.55 m
and so punished the only move that would get inside: crouching. Measured: 0/12
mid-height targets reached, body standing at 0.61 m. The locomotion reward now
tracks the height that puts the **shoulder level with the target** (clamped to
0.30-0.65 m), so "locomotion" means moving the sphere onto the target in all
three dimensions: wheels for x/y, legs for z.

**Result at iteration 22,400** (resumed from 19,300; `rfm10_height/model_22400.pt`):

| | 19,300 (before) | 22,400 |
|---|---|---|
| HIGH targets: max pitch (sit-back) | +36 median / +52 worst | **+16 / +17** |
| HIGH targets: body height | 0.27-0.36 m | **>= 0.53 m** (stands tall) |
| HIGH targets: target-to-shoulder at end | 0.54 m (stretched) | **0.24 m** |
| HIGH targets: reached | 11/12, median 0.2 cm | 9/12, median 0.3 cm |
| MID targets (0.45 m): reached | 10/12 | **4/12** (0/12 before the height fix) |
| MID targets: max pitch | +14 / +23 | +4 / +15 |
| reach map, fixed targets | 54/84, 1.7 cm | 54/84, **1.0 cm** |
| falls / lidar-mast intrusions | 0 / 0 | 0 / 0 |

The sit-back is gone and high targets now look right. Mid-height targets are the
open problem: they collapsed to 0/12 when the sphere was introduced and are
recovering (4/12) since the height fix, but have not caught up after ~1,500
iterations. Training was left running.

### 6f. Driving backwards, shaking, and a slow pace

Three more things seen on screen, all confirmed by measurement on the best
policy so far (`rfm10_height/model_27600`, which reaches 12/12 targets at 0.7 m
height to a median 0.2 cm):

| observed | measured | cause |
|---|---|---|
| "goes to the target backwards" | reversing for **77-99 %** of the approach on every side and rear target, 0 % on targets ahead | closing speed was rewarded in the world frame, so reversing paid exactly as much as driving forwards -- and it skipped the turn, this robot's weakest skill |
| "the arm shakes a lot on the target" | with the hand already within 3 cm: arm joints **6.3 rad/s RMS**, hand **90 cm/s** | micro-enhancement makes the gradient steepest at zero error; nothing asked the arm to hold still; and I had implemented only the first of the paper's two smoothness terms |
| "locomotion is slow" | pace capped at **0.35 m/s** | my own `approach_speed`: the schedule reward makes that pace optimal, and the closing-speed reward saturates there |

Fixes: approach reward counted only in proportion to facing the target, plus a
potential-based reward for turning toward it; arm joint speed inside the
manipulation stability gate, scaled by r_ep so it only bites once the hand is
already close; the paper's second-difference action smoothness term; and a
speed profile `v = clamp(1.0 * (d - 0.35), 0.3, 1.0) m/s` -- fast when far,
easing in near the arm's workspace, instead of one constant pace.

**A camping trap, found on the first try.** The first facing reward paid ~1 per
step simply for facing the target. At 29k iterations the sphere policy drove
forwards (reverse 93 % -> 5 %) and then *parked 0.73 m from every target*, just
outside reach, where the phase is still ~95 % locomotion -- earning ~2.6 per step
for standing still, facing it. 0/12 reached, 0/84 on the reach map. Crossing into
manipulation could only lose that. **Locomotion rewards must pay for progress,
never for idling**: facing became potential-based (pay for turning toward, nothing
for holding a heading), and both comparison runs restarted from the same
checkpoint.

**...and the facing reward was only a symptom.** With facing made potential-based,
the sphere policy *still* parked 0.66 m from every target (0/12, 0/84) while the
paper policy reached 11/12. Probing the parked policy showed a **reward valley**:

| where the robot is | phase D | reward per step |
|---|---|---|
| parked at 0.67 m from the shoulder | 0.90 (locomotion) | **1.44** = stability 0.97 x (1 + schedule 0.63) |
| moving in, hand not yet on target | -> 0 (manipulation) | **~0.85** = manipulation's stability term |
| hand on target | 0 | up to ~4 |

Success pays more, but the way there goes *down* first, and PPO does not cross a
valley from a settled optimum. The cause is the paper's own locomotion group,
`r_reg + r_reg * r_dw`, which pays for being stable whether or not the robot is
going anywhere -- plus my widening of the schedule reward's kernel (0.5 -> 1.5) to
fix an early stand-still problem, which left it paying 0.63 for doing nothing.
The paper never meets the valley because its phase is ~0 throughout that region.

Fix, applied to both modes: the locomotion group pays **only for progress**
(`r_reg * (r_dw + r_approach)`, no constant), and the kernel is back at 0.5 now
that the closing-speed reward carries the approach. An idle robot then earns
about what an idle manipulator does, and the valley is gone. Both runs restarted
from the shared checkpoint again.

The general lesson, worth more than the fix: **with a fused phase reward, compare
the two groups' *idle* values.** If standing still in one phase pays more than
starting the task in the other, the policy will park at the phase boundary.

### 6g. Phase criterion: the sphere vs the paper, controlled

Two tasks, identical in every reward, command, curriculum and termination,
differing only in how the phase D is computed:

| | `RFM-Sphere` (user's proposal) | `RFM-Paper` (Jiang et al., eqs. 11-12) |
|---|---|---|
| measured from | the **shoulder** | the **hand** |
| what is measured | position | SE(3): position + orientation |
| driven by | the actual robot state | a reference distance retreating on a schedule |
| blend | D 0.95 at full reach -> 0.05 at half reach | midpoint = slope = 2 x arm span (1.4 m) |

Both resumed from `model_27600` and trained the same number of iterations in
parallel. Evaluated with `Isaac/eval_rfm.sh` (mid 0.45 m and high 1.0 m approach
tests at 1-3 m and four bearings, plus the fixed-target reach map), tabulated by
`tools/compare_rfm.py`.

**Result after 1,100 iterations each** (`rfm14_sphere` / `rfm14_paper`,
`model_28700`; "start" is the shared checkpoint, trained far longer in its own
configuration, so it is a reference, not a competitor):

| | start | **sphere** | **paper** |
|---|---|---|---|
| high 1.0 m: reached (<10 cm) | 11/12 | 10/12 | 10/12 |
| high 1.0 m: median closest | 0.1 cm | **0.7 cm** | 1.2 cm |
| high 1.0 m: target-to-shoulder at end | 0.25 m | **0.32 m** | 0.56 m |
| high 1.0 m: lowest body | 0.52 m | 0.35 m | 0.27 m |
| high 1.0 m: sit-back pitch | +16 | +14 | **+8** |
| high 1.0 m: arm shaking on target | 6.3 rad/s | 3.2 rad/s | **1.9 rad/s** |
| high 1.0 m: hand speed on target | 91 cm/s | 68 cm/s | **25 cm/s** |
| reverse driving | 93-96 % | **0-2 %** | 0-14 % |
| mid 0.45 m: median closest | 47 cm | 19 cm | 20 cm |
| mid 0.45 m: target-to-shoulder at end | 0.48 m | 0.19 m | 0.15 m |
| reach map (fixed targets) | 60/84 | **45/84**, 2.6 cm | 30/84, 3.0 cm |
| falls / lidar-mast intrusions | 0 / 0 | 0 / 0 | 0 / 0 |

Reading it:

- **The sphere does what it was designed to do.** It keeps the arm in the middle of
  its workspace (target 0.32 m from the shoulder on high targets, against the
  paper's 0.56 m -- the paper's robot crouches to 0.27 m and *stretches the arm*
  up to a 1.0 m target, the exact failure that motivated the sphere). It is more
  precise on high targets and reaches 50 % more of the reach map.
- **The paper's criterion holds the arm calmer** -- a third of the hand speed on
  target -- and sits back less. A plausible reason: its phase is fully
  "manipulation" for the whole final approach, so the arm-stillness term acts
  for longer; the sphere hands over later.
- **Shared open problem: mid-height targets.** Both reach to within ~19-20 cm
  and no closer, with the shoulder already close (0.15-0.19 m). Same in both
  modes, so it is not the phase criterion.
- **Both fixes work in both modes:** reverse driving 93-96 % -> 0-14 %, shaking
  6.3 -> 1.9-3.2 rad/s.

Caveat: 1,100 iterations of fine-tuning is short. Both were still improving when
measured, and both were left training.

### 6h. The shaking hand: cause and cure (8-9 October)

Reported from `./live.sh goto_mine` / `goto_paper`: the robot reaches the
commanded point and turns the hand to the commanded orientation, but the hand
never settles — it keeps moving around the target.

#### Measured, not guessed (`Isaac/shake_diag.py`)

`shake_diag.py` holds the robot on 16 fixed 6D targets (three reachable points ×
four orientations, one of them impossible, plus four that need a 2 m drive
first) and scores the last 8 s: wobble amplitude, hand speed, arm joint speed,
the step-to-step change of each action group, the dominant frequency, and the
wobble split into "the shoulder moved" vs "the hand moved relative to the body".

rfm14-sphere, on target (median):

| wobble | hand speed | arm action change per 20 ms step | legs / wheels | frequency | shoulder / arm-own |
|---|---|---|---|---|---|
| 15 mm | 15 cm/s | **2.9** (≈1.5 rad of joint target) | 0.06 / 0.07 | ~1 Hz | 4 mm / 18 mm |

The arm, and only the arm, was driven **bang-bang**: its targets flicked between
extremes every step and only their average placed the hand. A relay controller
with lag cannot settle — it cycles, and that cycle was the ~1 Hz wobble.

#### Why the policy learned a relay

- **Arm motor gains were randomised 0.5–2×** at every reset
  (`randomize_actuator_gains`, joint names `.*`, the arm included). A memoryless
  policy cannot know its stiffness, so a smooth position target sags by an
  unknown amount. Saturating the torque in both directions is stiffness-proof:
  rfm14 tracks ~2 cm at kp 50, 100 and 200 alike.
- **Arm link masses were randomised 0.7–1.3×** — the same problem through the
  gravity load.
- **Nothing in the reward charged for it.** The 22-wide action-rate term at
  −0.003 cost ~0.03 per step for the flicking, while a constant entropy bonus
  (0.01 for 33,800 iterations) paid for more noise: the arm's action std grew from
  1.0 to **2.0**. The paper *decays* its entropy coefficient.
- The arm's joint-velocity observations carried the locomotion template's
  ±1.5 rad/s noise, so the policy could not see its own wobble.

#### What did not work (each measured)

| attempt | result |
|---|---|
| play-time low-pass filter on the arm | wobble halved, not removed |
| play-time "hold latch" | never held: it froze one sample of a flicking command |
| hand-speed penalty counted only near the target | robot **hovered 25 cm short** of the target — a proximity-gated penalty is a reason to stay away |
| stillness factor on the tracking bonus (σ 0.03) + arm rate/torque-rate penalties | arm still, but tracking signal gone: settled 4–6 cm short, orientation **1.0–1.3 rad** |
| fixed arm gains + filter in training (+ mild rate penalty, + fixed masses, + integral-error observation) | precise (15–16/16, ori 0.2 rad) but still 12–22 mm wobble: the relay un-learns very slowly — PPO's KL limit is in units of the action std, so capping the arm's std from 2.0 to 0.3 also cut how far its behaviour can move per update ~7× |

#### The cure: incremental arm commands (rfm19)

The arm's action now **changes** its joint target instead of setting it:

    target_t = clamp(target_{t-1} + 0.05 rad × a_t)      (rfm.py, DeltaJointPositionAction)

- holding still is a = 0, the natural resting output;
- a flick +a, −a cancels, so a relay has nothing to gain;
- the target keeps moving while an error remains, whatever the gains or masses:
  integral action for free, which is what the smooth absolute-target policies
  lacked to reach full precision.

Together with fixed arm gains and masses, clean arm velocity observations and two
small penalties (arm action² −0.01, arm action rate −0.02). Started from rfm14's
network with the arm's output weights zeroed (`Isaac/calm_init.py
--zero_arm_head`), so legs and wheels keep everything they had learned and the
arm re-learns reaching in the new command space. ~3,800 iterations (sphere) and
~2,900 (paper), about 1.5 h each.

#### Result

`shake_diag.py`, 16 targets, on target (median):

| | rfm14-sphere | **rfm19-sphere** (37,600) | **rfm19-paper** (36,700) |
|---|---|---|---|
| cases within 5 cm | 12/16 | **16/16** | **16/16** |
| wobble | 15 mm | **0.1 mm** | **0.0 mm** |
| arm action change per step | 2.9 | **0.000** | **0.000** |
| orientation error | 0.11 rad | **0.05 rad** | **0.05 rad** |

The ~2 cm/s hand speed that remains is the simulator's floor (rfm14's motionless
cases read the same).

Full battery (`eval_rfm.sh`):

| | rfm14-sphere | **rfm19-sphere** | **rfm19-paper** |
|---|---|---|---|
| mid 0.45 m, within 10 cm / median closest | 0/12, 18 cm | **11/12, 2.3 cm** | 5/12, 10.1 cm |
| high 1.0 m, within 10 cm / median closest | 10/12, 0.7 cm | **12/12, 0.2 cm** | **12/12, 0.9 cm** |
| hold on target: arm joints / hand | 3.2 rad/s / 68 cm/s | **0.18 / 2.2** | 0.37 / 2.7 |
| reach map: cells / median error / orientation | 45/84, 2.6 cm | **56/84, 0.6 cm**, 0.058 rad | 54/84, 1.6 cm, 0.059 rad |
| falls / worst tilt | 0 / 18° | 0 / 26° | 0 / 28° |
| arm into own volume (map) | 0/84 | 14/84, max **0.8 cm** deep | 0/84 |

(rfm14 column: the stored rfm14 evaluation, `model_28700`.)

- **The shake is gone in both**, and precision went *up*, not down: the
  incremental command supplies the integral action a smooth controller needs.
- **The old mid-height stall is gone for the sphere** (0/12 → 11/12). The paper's
  phase rule still stops 6–11 cm short on 0.45 m targets — its body ends with the
  shoulder only 2–7 cm from the target, and an arm cannot fold back onto its own
  base. That is the phase criterion, not the shake: the sphere keeps the target
  0.3–0.45 m from the shoulder by design.
- Interactive check (`rfm_goto.py`, target 2, 1.5, 1.0): drove 2.5 m, then held
  the hand at 0.1–0.9 cm, 0.04 rad, standing at 0.51 m.

Checkpoints: `b2w_z1_rfm/2026-10-08_19-58-51_rfm19_delta_sphere/model_37600.pt`
and `b2w_z1_rfm/2026-10-08_20-28-32_rfm19_delta_paper/model_36700.pt`, tasks
`RobotLab-Isaac-RFM-Delta-{Sphere,Paper}-Unitree-B2W-Z1-v0`. Both were still
improving slowly when stopped and can be resumed.

**Lesson:** when a learned controller shakes, look for a reason it *needs* to —
here, randomised gains and masses it could not observe — before penalising the
shaking. And let the action space make the calm behaviour the default: with
absolute targets a relay was a good controller; with incremental targets it is
pointless.

### 7. Tools

| tool | what it answers |
|---|---|
| `Isaac/probe_rfm.py` | every component of the fused reward, with or without a policy — the tool that found bugs 2–5 |
| `Isaac/approach_test.py` | can it reach a pose anywhere around it, and arrive intact |
| `Isaac/rfm_goto.py` | interactive 6D targets, no state machine |
| `Isaac/shake_diag.py` | does the hand hold still on target? wobble, hand speed, per-group action chatter, frequency, body-vs-arm split; optional play-time filter / gains / latch |


**Final checkpoint:** `logs/rsl_rl/b2w_z1_rfm/2026-09-24_23-15-45_rfm8/model_19300.pt`
