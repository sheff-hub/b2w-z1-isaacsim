# moveit/: MoveIt plans the arm's pick

The fetch pipeline's pick (`project/fetch_box.py --arm moveit`, the default) lets **MoveIt 2** plan every arm motion.
Driving and docking at the table are unchanged (scripted route or Nav2); once the robot stands beside the box,
MoveIt takes over the arm.

```
 robot docked beside the table (Nav2 or scripted)
        │
        ▼
 crouch 6 cm (walking policy)                      ┌──────────── moveit container (ROS 2 Jazzy, CPU only) ───────────┐
        │                                          │                                                                  │
        ▼            JSON over HTTP :5810          │  moveit_server.py  ──ROS 2──►  move_group (MoveIt 2.12, OMPL,    │
 fetch_box.py  ───────────────────────────────────►│  (the bridge)                  pick_ik, planning scene)          │
 (Isaac Sim)   ◄─── joint trajectory (timed) ──────│                                                                  │
        │                                          └──────────────────────────────────────────────────────────────────┘
        ▼
 1 pre-grasp   MoveIt: plan from the stowed arm to 12 cm before the box, around the table, the other boxes and the
               robot's own body (lidar cage, head, hips)                            -> collision-free, ~3 s motion
 2 open        the gripper (simulator)
 3 descend     MoveIt: straight line to the grasp point (Cartesian path); the target box leaves the scene
 4 close       the gripper (simulator)
 5 lift        MoveIt: straight line 6 cm up
 6 carry       MoveIt: plan to the carry pose, the box ATTACHED to the gripper (so it is kept clear too)
        │
        ▼
 Nav2 delivers the box (unchanged)
```

**Who does what.** MoveIt only *plans*. The simulator *executes* the timed joint trajectories at its control rate,
as a robot's joint trajectory controller would; a small integral term on the joint targets removes the standing
error the arm's PD drives need against gravity (otherwise ~3 cm at the fingertips). Isaac Sim's bundled ROS 2 has
no MoveIt messages, so the simulator talks to MoveIt through a small HTTP bridge, like the DualVLN server.

**Frames.** MoveIt plans in the arm's base frame, `link00` (fixed on the B2W body). The simulator sends the table
tops, the boxes, the robot's body boxes and the floor in that frame, and every goal pose. MoveIt's tip link
`grasp_point` is the point between the fingertips where the box centre should end up, the same point the scripted
IK pick uses. At every pick, `fetch_box.py` compares MoveIt's forward kinematics with the simulator's grasp point
(`stages.moveit.fk_check_mm` in the results): 0.0 mm.

## Results

Same episodes for both arms (one table, scripted drive, `--seed 100`, 36 episodes, with delivery):

| Pick | Picked | Delivered | Grasp-point error (median / max) | Episode time (mean) | Planning time per motion |
|---|---|---|---|---|---|
| MoveIt (`--arm moveit`) | 36/36 | 36/36 | 0.3 / 0.5 cm | 45.1 s | 0.16 s median, 1.9 s max (pre-grasp); < 0.1 s (lines, carry) |
| Scripted IK (`--arm ik`) | 36/36 | 36/36 | 0.2 / 0.6 cm | 47.1 s | – |

The IK pick corrects the hand at every control step while the base drifts, so its grasp is a little tighter.
The MoveIt pick plans each motion once; if the base drifted during the descent and the grasp point ended ≥ 4 mm
off, a short straight MoveIt move corrects it (`stages.moveit.corrections`; 15 of the 36 episodes, ≤ 2 moves).
Without that correction the same episodes ended 0.4 / 1.1 cm off (still 36/36).

With Nav2 and DualVLN (three tables, the neighbouring boxes in the planning scene):

| Run | Picked | Delivered | Grasp-point error (median / max) | Descent corrections |
|---|---|---|---|---|
| Nav2, three tables, `--seed 31`: 7 picks from the right side, 5 from the left | 12/12 | 12/12 | 0.25 / 0.4 cm | 6 of 12 episodes, one move each |
| DualVLN + Nav2, recorded, `--seed 43` | 3/3 | 3/3 | 0.2 / 0.3 cm | 2 of 3 |

Every pre-grasp plan succeeded on the first try; on the way down the gripper touched nothing and no box was pushed
(at most 1 mm). Before the third DualVLN episode the DualVLN server was killed (SIGKILL, most likely the host
running out of memory with two simulators up); Nav2 then drove the whole way, and the pick was unaffected.

## Shelves: take a box out, put it on a table (the hospital reception)

Scene layout `shelf` (`scenes/hospital_reception.yaml`): three boxes on a tray of the hospital's medicine shelf, a
tray 32 cm above them and a 3 cm lip at the front; the named box goes onto the side table 3 m away.

```
 Nav2: staging pose 3 m before the shelf, docking server along the reception desk, the shelf on the robot's right
   1 approach_ik  /approach: an IK solution for the pre-grasp from which the straight line into the shelf works,
                  at the scene's pitches in turn (45, 50, 40 deg below horizontal), the gripper rolled 180 deg
   2 pre_grasp    MoveIt: to those joints, around the shelf, the other boxes and the body
   3 open, descend (straight, 16 cm), close
   4 lift 5 cm    straight up, over the lip
   5 retreat      straight out, 20 cm, the box attached (checked against the lip and the tray above)
   6 carry        the box folded in over the robot's body
 strafe 0.5 m off the shelf; Nav2: staging pose, docking beside the table, the table on the robot's left
   7 pre_place    MoveIt, the box attached: 8 cm above its spot, the gripper turned so the BOX is upright
                  (its pose in the jaws is measured first: it may have slipped on the way)
   8 lower, open, withdraw along the approach, fold to the rest pose
```

Placed = on the table top, upright (< 15 deg), within 2 cm of the table height, the gripper 5 cm or more away.

| Run (`python3 run.py fetch --scene hospital_reception --no-vln`) | Picked | Placed | Box off its spot (median / max) | Grasp-point error (median) | Episode (median) |
|---|---|---|---|---|---|
| seeds 11 and 12, 6 episodes each, the box named at random | 12/12 | 12/12 | 0.65 / 1.5 cm, upright | 0.2 cm | 91 s |

All 24 drives (to the shelf, to the table) by Nav2; every pick at 45 deg; 59 of the 60 free-motion plans on the first
try, one on the second, no MoveIt request failed.

### One box per tray: the hall (`scenes/hospital_hall.yaml`)

The same shelf and side table moved into the open hall, in front of where the robot starts (`hospital_hall.usda`
switches their bottles and trays and the hall's wheelchair off). The shelf is open on both sides; the boxes stand on
the side facing the robot's start, which the robot docks along from the south, the shelf on its right. A green box on
the tray below the 0.9 m one, yellow on it, red on the one above -- and each is picked differently (the scene file's
`levels:`):

| Box | Tray | Body (ETH stance) | Docked | Approach | Wrist camera |
|---|---|---|---|---|---|
| green | 0.50 m | crouched 6 cm (`stance: 1`) | 12 cm further off (`gap: 0.21`) | 55 deg down | rolled under (`jaw_flip`) |
| yellow | 0.88 m | standing (`0`) | `gap: 0.09` | 45 deg | rolled under |
| red | 1.25 m | 1.8 cm taller (`-0.75`; the policy gives no more) | `gap: 0.09` | 15 deg, nearly level | on top |

| Run (`python3 run.py fetch --scene hospital_hall --target green,yellow,red`) | Picked | Placed | Box off its spot (median / max) | Grasp-point error (median) | Episode (median) |
|---|---|---|---|---|---|
| seeds 81 and 82, 6 episodes each, DualVLN walking first | 12/12 | 12/12 | 0.8 / 1.6 cm, upright | 0.3 cm | 105 s |

Every DualVLN walk ended at the shelf (11 handed over within 1 m, one stopped by itself 1.4 m off), every Nav2 drive
arrived, no falls; the body leaned 1-3 deg towards the shelf at every pick. Getting there, in the order found:

- **The low tray** is reachable only steeply from above, from a crouch and further off; a box turned more than ~10 deg
  is out of reach, so the green box stands nearly square (`yaw_jitter: 0.1`). Pulled out 16 cm, not 20: further, the
  wrist camera nears the body. The docking window is narrow: Nav2 parks ~4 cm further than asked (+-3), the arm
  reaches the box up to 0.25 m from the shelf, and from 0.17 the pulled-out wrist ends against the body (no plan to
  the carry pose; lifting the hand 6 cm at a time, at most twice, did not free it): `gap: 0.19`.
- **The high tray**: the hand goes in nearly level above the shoulder, near the end of the arm's reach. Lifting the
  box 8 cm is no measure of success there (it hits the tray above): a shelf pick counts when the box is held and out
  of the shelf. The lift is 4 cm (`lift`), over the 2.9 cm lip, and a lift cut short counts once the box clears the
  lip (one red box got only 3.2 of 5 cm). The red box stands within +-6 deg (`yaw_jitter: 0.1`): at +-17 and +-11 a
  box turned the far way was out of reach after docking.
- **The body leans after docking:** 1-4 deg of roll towards the shelf (0.4 when the robot is set down there, as in
  `--probe`). With the grasp's 15 deg below the body, not below the level, the box sat 2.4 cm higher and the approach
  3-4 deg steeper: out of reach. Shelf grasps are now pitched against the level (`quat_inv_tilt`); levelling the
  place too put it out of reach when the body leaned away from the table, so the place stays body-relative (the box
  settles).
- **A start a hair off the checked one:** `/approach` checks the straight line from the joints it returns; the arm
  stops a fraction of a degree off them, and twice MoveIt refused the first step from there (once finding the pose
  touching the shelf, once with no contact at all). Within 1 deg of the checked joints the descent now follows the
  same line unchecked; the contacts are logged (`descend_start`).
- **Putting a box down that was taken nearly level:** at the table's edge the arm cannot hold the hand that level, so
  it goes 22 cm in (`place_depth`); straight in, even that is at the edge of reach (the lowering stopped at 43-69 %).
  Turned 15-30 deg about the vertical -- the box stays upright, only its heading changes -- it is well inside:
  `place_moveit` tries the poses of `PLACE_TRIES` in turn.
- **Letting go:** when the line back out is cut short, the open jaws go straight up (they slide up the box's sides),
  and they close only 8 cm clear of the box. Before that, the arm once got 29 % of the way back, the jaws closed
  beside the box and took it off the table.
- **DualVLN and the see-through shelf:** told to walk to the shelf, DualVLN headed for it and walked on round its
  north side to the far corner (2 of 2), and a sentence naming the coloured boxes made it turn round (they are a few
  pixels from 5 m). The walk now ends once the robot is within 1 m of the named shelf (`vln_should_stop`, reason
  `near`); Nav2 knows the shelf and parks at its box side.
- **The walk that never started**, three ways: Nav2's controller, which drives DualVLN's path, (1) refused to move
  for a minute ("collision ahead", 6470 times) on marks left in its local costmap from before the reset -- the
  costmaps are now cleared at the start of every walk and whenever the robot stands 3 s while following a path; (2)
  rejected every path ("Action server is inactive", 150 of 150) because its activation had timed out while the
  lifecycle manager reported Nav2 active -- readiness now asks the controller itself; (3) never came up at all,
  started while the simulator was still booting -- halfway through the wait Nav2 is reset and started again, and a
  walk that still finds it down is followed by our own pure pursuit (`nav2_link.wait_ready`, `vln_walk`).

Earlier runs: the previous layout (the boxes on the shelf's far side, docked from the north) 12/12 picked and 11/12
placed (seeds 61, 62), then 4/4 red after the release fix; this layout before the last fixes 10/12 (seeds 71, 72).

`python3 moveit/shelf_reach.py hospital_hall --level green` checks a tray (its stance as the body height; `--abeam`,
`--gaps` for docking errors).

What made shelves hard, in the order it was found:

- **The wrist RealSense was not in MoveIt's model.** The D405 on its ring clamp sits 16-22 cm off the wrist axis and
  is a collider in the simulated robot, but not in Unitree's URDF. MoveIt planned an approach the camera could not
  follow: the arm stopped 10 cm short, its joints 12-16 deg behind the plan, and the robot's contact sensor reported
  nothing (it does not see the gripper or the camera; the joint trace in `FETCH_TRACE` did). `make_config.py` now
  adds it to link06 as three boxes (camera, bracket, ring).
- **The camera rides into the tray above.** With the jaws closing horizontally, the camera is ~27 cm above the grasp
  point, inside the shelf, at every pitch the arm can reach. Rolled 180 deg about the approach (`jaw_flip`), the
  grasp is the same but the camera hangs down and back, outside the shelf.
- **Level approaches are out of the Z1's reach.** Its hand is 27 cm long (wrist pitch joint to the grasp point);
  holding it level at shoulder height needs the wrist beyond its 87 deg. 40-50 deg work; for a box turned 12-24 deg
  towards the robot, only from 4 cm further off the shelf (`gap`).
- **A pose goal can end in a dead-end posture.** MoveIt's IK reached the pre-grasp with the elbow folded on its
  stop, from where no straight approach exists (the line failed at 12 % even without collision checks). The bridge's
  `/approach` tries IK solutions until the straight line from one works, then the pick plans to those joints.
- **The tables' carry pose reaches into the shelf** (it holds the box out to the side): a carry pose of its own,
  folded over the body. Pull-outs of 25 cm leave the arm's reach (94 %): 20 cm, and 60 % of the line is accepted.
- **After letting go, the camera hung over the robot's back** (it is rolled under): the withdraw follows the
  approach line back and up, so the arm can fold.
- **The building:** shelf and table stand against a wall the obstacle list missed (the scripted drive pushed the
  robot into it for 40 s); the Nav2 map from the colliders (`nav2_probe.py --area`) shows every wall. The shelf is
  reached along the reception desk: with the default 2.1 m, Nav2's staging pose was beside the desk, where the robot
  had no room to turn ("collision ahead", aborted); 3 m (`nav2_staging`) puts it in the open.

## Doors: turn the handle, push it open, go in (`scenes/hospital_doors.yaml`)

The hospital's east corridor with a hinged door with a lever handle in each of its three openings
(`scenes/assets/door_hinged.usda`, built from the hospital's own SM_Door_01 by `tools/build_door_asset.py`: links
frame / leaf / handle, joints `hinge` and `handle_joint`). One door is painted blue at random every episode (the
door material's own colour input; the others white and yellow). The latch is virtual: the hinge's drive is stiff
(20 000 N*m/rad) while latched and free (damping only) once the lever is turned past 25 deg, latched again when the
door is shut and the lever let go -- a 60 N push does not move a latched door, and opens it fully unlatched.

```
 DualVLN: "Walk to the blue door on the wall ..." (hands over within 1 m of the named door)
 Nav2: staging pose 2 m before the parking pose, docking straight on: facing the door, 0.70 m off the leaf,
       0.80 m from the hinge edge towards the handle
   1 approach_ik  /approach: a pre-grasp 12 cm out from the lever's bar, the approach pitched 30 deg down (15, 45, 60
                  in turn), the jaws closing round the bar in that vertical plane
   2 pre_grasp, open, straight in, close     (the jaws stop on the 3.8 cm bar at ~-0.49 rad)
   3 turn         the hand along the lever's arc round its spindle, 40 deg down (MoveIt Cartesian): unlatched
   4 push         holding the lever down, the hand's target is the grasp carried round the hinge by the commanded
                  door angle -- damped-least-squares IK every step, the base following a point 0.62 m behind the
                  hand -- the command growing 8 deg/s while the hand keeps within 4 cm, never behind the angle the
                  hand itself is at (to 65 deg)
   5 let go       the lever turned back up under the hand -- step 3 in reverse, round the spindle where the open
                  door has carried it (4 s, IK every step) -- the arm eased off the lever (1 s), the jaws opened
                  slowly (1 s); back out 10 cm, fold the arm
   6 enter        drive in through the opening (the body pushes the door further)
```

| Run (`python3 run.py fetch --scene hospital_doors --target a,b,c`) | Unlatched | Opened | In the room | Door pushed to (median) | Grasp-point error (median) | Episode (median) |
|---|---|---|---|---|---|---|
| seeds 31 and 32, 6 episodes each, DualVLN walking first | 12/12 | 12/12 | 12/12 | 64.5 deg | 0.25 cm | 78 s |
| the same before the lever was turned back up on letting go (27 Oct) | 11/12 | 11/12 | 11/12 | 64.8 deg | 0.26 cm | 68 s |
| Nav2 only (`--no-vln`), seed 5, doors c, a, b (27 Oct) | 3/3 | 3/3 | 3/3 | 65.1 deg | 0.25 cm | 58 s |
| `--probe` (set down at the door), doors a, b, c | 3/3 | 3/3 | 3/3 | 65.2 deg | 0.30 cm | |

Door by door 4/4 each. Letting go: the lever turned back up to -1.6 deg (median), the arm's pressing eased from 25
to 3 N*m (median), the hand moved 2.6 cm (median, at most 5.8) as the jaws opened, the lever dipped 8 deg (median)
and sprang back, the door went 0.8 deg further (4.4 deg letting go of the pressed lever, at most 16); 6 s more an
episode (mean). The miss on 27 Oct never reached a door: DualVLN walked 17 m the other way, to the edge of Nav2's map,
and Nav2 found no way back; since then a walk also ends 3 m further from the door than it started
(`WALK_AWAY_M`).

What it took:

- **A level hand cannot reach the lever.** 0.2 m above the arm's shoulder the Z1 holds a level hand nowhere; from
  15 to 60 deg below level it reaches the bar from 0.55-0.75 m (`moveit/door_reach.py`, the door's `shelf_reach`).
- **The door would not move, unlatched.** Its box collider sat on the doorway's raised threshold (Geo_M_DoorFloor,
  ~3 cm, in every opening of the building) and flush with the jambs: inset 1.5 cm at the edges and 5 cm off the
  floor. The lever, 0.2 kg on a 3 N*m/rad spring, rests on its stop with a 0.1 rad preload (it drooped 11 deg).
- **The push deadlocked once:** the robot's body nudged the door ahead of the command, the hand (holding the
  lever) was dragged 5 cm off its target, and the command only grew while the hand was within 4 cm. The command now
  never falls behind the door angle the hand itself measures (its bearing round the hinge).
- **Letting go swung the arm.** With the jaws opened at once on the pressed lever, its return spring snapped it up
  into them (245-535 deg/s) and the hand was knocked 12-24 cm. Turning the lever back up under the hand first (the
  turn in reverse) brought it level, yet the hand was still knocked 11 cm: while the grip holds the hand, the
  integral IK winds the arm's joint targets up to `IK_WINDUP` ahead of the joints -- 25-32 N*m pressing on the
  lever, let loose the instant the jaws open. So the targets are then eased to where the arm holds only its own
  weight: the joints plus PhysX's gravity torque over the stiffness (`get_gravity_compensation_forces`, the 6 base
  columns first; it matches the free arm's drive torque to 0.3 N*m), taken at the start of the ramp, twice. Taken
  every step instead, the targets follow the joints, the arm has no stiffness left and sagged, the lever pulled
  down 13-21 deg; holding the hand on the lever with the IK through the ramp, its targets clamped ever closer round
  the free ones, it sank as much, and kept on that weak hold while the jaws opened the arm drifted off (6-18 cm).
  Left: ~2 N*m; on doors a, b, c the hand moves 1.4-2.7 cm as the jaws open and the lever 1.5-7 deg.
  `--door_unturn_s 0 --door_ease_s 0` is the old release.
- **DualVLN and closed doors.** Facing the three doors 5.5 m away, the blue one plainly in view, DualVLN turned away
  first and walked off in 5 of 6 starts; six wordings and turning one 15-deg step at a time (instead of its four at
  once) did not change that -- the doors filling the view seem to read as a wall to it. From 8-10 m back in the hall
  it went to the named door in 4 of 6 walks (door b 2/2 handed over at 1 m, c 2/2 to 1.9 m, a 0/2), so the scene
  starts there. The Nav2 map covers the hall west to the reception: Nav2 brings a robot that wandered off back to
  the named door.

## Run

```bash
python3 run.py moveit start            # or moveit/run_moveit.sh; run.py fetch and live.sh start it when needed
python3 run.py fetch                   # the final pipeline, MoveIt picking (--arm ik: the scripted IK pick)
ARM=moveit ./project/live.sh pick      # watch it in a window
python3 run.py moveit stop
```

`docker logs -f moveit` shows the bridge; `docker exec moveit cat /tmp/move_group.log` shows move_group.

## Files

| File | What |
|---|---|
| `Dockerfile` | `b2w-moveit:jazzy`: ROS 2 Jazzy + MoveIt 2 (move_group, OMPL, pick_ik); built by `setup.sh` |
| `make_config.py` | makes `config/z1.urdf` from the Z1 + gripper URDF of the robot assets: root `link00`, the **simulator's** joint limits (narrower than Unitree's, e.g. joint3 −165…0°), the `grasp_point` tip, the wrist RealSense as boxes on link06, container mesh paths |
| `config/z1.srdf` | planning group `arm` (link00 → grasp_point), the `stow` pose, link pairs that may touch |
| `config/kinematics.yaml` | pick_ik (global search) for 6D goals |
| `config/joint_limits.yaml` | velocity/acceleration limits for MoveIt's time-optimal timing |
| `start_move_group.py` | builds move_group's parameters with MoveIt's own `MoveItConfigsBuilder` and starts it |
| `moveit_server.py` | the HTTP bridge: `/scene`, `/plan` (pose or joint goal), `/cartesian`, `/approach` (a pre-pose with a feasible straight approach), `/check`, `/fk`, `/health` |
| `entrypoint.sh`, `run_moveit.sh` | start both in the container (port and ROS domain from `config.yaml`, section `moveit`) |
| `acm_check.py` | samples 2000 arm poses for link pairs in contact (how the SRDF's allowed pairs were chosen) |
| `shelf_reach.py` | can the arm get a box out of a scene's shelf? Docking gap x approach pitch x box yaw, asked of MoveIt without the simulator (`python3 moveit/shelf_reach.py hospital_reception`; use a MoveIt no simulator is using, `--url`) |
| `door_reach.py` | can the arm grasp a door's lever? Docking distance x approach pitch, asked of MoveIt without the simulator (`python3 moveit/door_reach.py --backs 0.55,0.65,0.75`) |
| `../project/moveit_link.py` | the simulator's client, and the robot's body as boxes for the planning scene |

## Traps

- **The rest pose collides in the simplified shapes.** Unitree's URDF models the links as cylinders; with the arm
  folded (all joints 0), link02 and link06 overlap, so MoveIt refused to plan from the pose every pick starts in.
  `acm_check.py` found it as the only pair to allow; the other self-collisions it found stay checked.
- **Plans out of the rest pose grazed the lidar cage.** The stowed gripper sits ~2.6 cm from the cage. With OMPL's
  default checking resolution, 5 of 36 plans passed OMPL but a re-timed point touched the cage, and MoveIt's final
  check (`ValidateSolution`) rejected them ("MoveIt found no plan (pre_grasp)"). `config/ompl_planning.yaml` checks
  motions 5x finer (`longest_valid_segment_fraction: 0.002`) and the bridge plans again if that still happens.
- **move_group waits for /joint_states.** Without a publisher it waited 1 s on every request; the bridge publishes
  the last requested state.
- **link01 and link05 have no collision shape** (also in the simulator), so MoveIt cannot check them.
- **The lift skips collision checks.** At the start of the lift the box still rests on the table, which MoveIt
  would count as a collision; the lift is a straight 6 cm line up, so it is planned without the check. The carry
  motion after it is checked again, with the box attached.
- **One MoveIt per simulator.** Each pick replaces the planning scene before it plans, so two simulators sharing one
  MoveIt would plan in each other's scene. For a second simulator at the same time start a second MoveIt:
  `MOVEIT_CONTAINER=moveit-2 MOVEIT_PORT=5811 MOVEIT_DOMAIN_ID=48 python3 run.py fetch ...` (the same rule as one
  simulator per Nav2).
- **The robot's body is boxes.** Body, top cover and lidar cage are the measured colliders; hips, head and thighs
  are approximated around their link positions (`moveit_link.BODY_BOXES`). If MoveIt plans through a part of the
  robot, that list is where to look.
