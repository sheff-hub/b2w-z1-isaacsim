# ETH controller: rear-wheel stance fine-tune (Option B), night of 7–8 Oct 2026

## Slurm
- The Slurm node *is* this workstation (ProArt). Only the 96 GB Blackwell
  (`gpu:rtxpro6000:1`) is in Slurm.
- All night it was taken by another user's job 619 (`m2d-render`, 3-day limit), with two
  more jobs queued behind it (zjin 621, vla 617). So there was **no free Slurm GPU**.
- Your account can submit directly (test job 620 ran).
- Training therefore ran on the Ada (49 GB), which CARLA shares at 100 % load. Two
  2048-env runs fit side by side (about 17 GB in total) at about 11 s per iteration each.

## The problem, measured
ETH's policy commands the rear hips about 0.3 rad *inward* all the time.

**Fine-tune scene, fixed commands, no randomisation (ETH):**

| command | front track | rear track | followed |
|---|---|---|---|
| stand | 0.40 | 0.25 | – |
| fwd 0.1 | 0.42 | 0.23 | 0.119 |
| fwd 0.5 | 0.42 | **0.105** | 0.512 |
| fwd 1.0 | 0.47 | 0.20 | 0.941 |
| side 0.2 | 0.28 | 0.19 | 0.167 |
| yaw 0.5 | 0.29 | 0.22 | 0.507 |
| yaw −1.0 | 0.25 | 0.29 | −1.023 |

With the hips at zero the track is about 0.40 m at both ends.

Two findings shaped the fix:
1. **ETH moves sideways and turns *by* narrowing.** A fine-tune that keeps the stance
   wide everywhere loses sideways driving and turning: v3 and v4 followed sideways
   0.2 m/s at 1–4 %.
2. **The measured hip angle is the wrong signal.** A loaded wheel cannot slide sideways, so
   while driving straight, friction holds the hips near zero whatever the target is. The
   inward targets take effect in the first turn, when the wheels slip, and the stance then
   stays narrow.

## Recommended policy: `eth/policy_stance_v5_600.pt`
This is v5 at iteration 600 (SHA 154b34222a06f005). Use it through `ETH_POLICY`; nothing
else changes:

    ETH_POLICY=/workspace/eth/policy_stance_v5_600.pt ./live.sh fetch
    ETH_POLICY=/workspace/eth/policy_stance_v5_600.pt ISAAC_IMAGE=isaac-lab:2.3.2 ./run_script.sh fetch_box.py --loco eth ...

Without `ETH_POLICY`, everything still runs ETH's original ONNX.

### Stance: rear wheel track while driving straight or standing
Fine-tune scene, 512 robots, random command history (35 % straight, 15 % spin, 10 %
standing, the rest uniform), 60 s:

| policy | rear track, straight | front track | worst robot, rear | |v err| | |w err| |
|---|---|---|---|---|---|
| ETH | **0.200 m** | 0.342 | 0.043 | 0.089 | 0.044 |
| **v5 @ 600** | **0.310 m** | 0.392 | 0.139 | 0.092 | 0.050 |
| v5 @ 650 | 0.312 | 0.389 | 0.141 | 0.094 | 0.051 |
| v7 @ 650 / 799 | 0.316 / 0.312 | 0.393 / 0.399 | 0.128 / 0.124 | 0.089 | 0.059 |
| v8 @ 700 | 0.304 | 0.412 | 0.102 | 0.090 | 0.062 |

The rear track is 55 % wider and velocity tracking is unchanged.

**Warehouse probe** (`probe_track.py`, one robot: stand, forward, turn left, forward, turn
right, forward, stand). Rear track after the first turn:
- ETH: 0.16–0.18 m
- v5 @ 600: 0.27–0.34 m

Before the first turn both keep 0.38 m. A single run is chaotic, though: v5 @ 650 ended at
0.19 m. The table above is the reliable measure.

### Driving (eth_compare.py, warehouse, CPU physics: % of the command followed)

| test | ETH | v5 @ 600 |
|---|---|---|
| fwd 0.3 / 1.0 | 106 / 93 | 104 / 90 |
| small fwd 0.1 / 0.15 | 104 / 104 | 103 / 105 |
| back 0.3 | 93 | 92 |
| **left 0.2 / right 0.2 (from standing)** | **89 / 68** | **0 / 35** |
| right 0.3 | 87 | 78 |
| diagonal | 88 | 74 |
| slowdown into diag / right | 94 / 83 | 74 / 71 |
| yaw 0.5 | 101 | 96 |
| turn 180° | 179° | 171° |
| brake from 1 m/s | 0.17 m | 0.22 m |
| hold with the arm out (drift) | 0.095 m | 0.103 m |

### Fetch (fetch_box.py, 24 episodes, random starts, both sides, seed 7)
**24/24 successful, 0 falls** with v5 @ 600. ETH earlier scored 68/72 on a different seed.

### The known regression: sideways from standing
Every fine-tune lost most of ETH's sideways driving:
- Fine-tune scene: 0.2 m/s gives 0.08–0.10 (ETH 0.167).
- Warehouse, from standing: 0–35 %.

Things that did **not** fix it:
- the stance terms switched off while moving sideways (v5);
- imitating ETH's hips while moving sideways (v5);
- a tight sideways-tracking term (v6, v7);
- removing robot_lab's roll/pitch-rate, vertical-velocity and flat-orientation penalties
  (v8).

ETH's way of moving sideways uses the narrowing itself (rear track 0.19 m at 0.2 m/s), so
part of this is the price of the wider stance. The fetch pipeline doesn't depend on it.

## Runs (all on the Ada, logs in `repos/robot_lab/logs/rsl_rl/b2w_z1_eth_finetune/`)

| run | idea | outcome |
|---|---|---|
| v1 `stance1` | track-width −6 + hip pull, robot_lab rewards, command curriculum | stance fixed, driving broken (fwd 0.3 at 46 %) |
| v2 `stance2` | lighter, no curriculum, fall termination | small forward commands reversed, 0.7 m drift |
| v3 `imit3` | imitation reward toward a frozen copy of ETH | straight driving kept; stance narrowed after turns; turning 52–73 % |
| v4 `v4` | from v3; target 0.40, hip pull, tighter tracking | sideways lost, stance same |
| v5 `v5mix` | stance terms only while driving straight; ETH hips while turning or sideways; 35 % straight / 15 % spin commands | **chosen**: turning back (96 %), rear 0.31 m |
| v6/v7 | penalise the hip *targets*; tight sideways term; v7 with desired_kl 0.02 | ≈ v5; v6 alone stuck at LR 1e-5 |
| v8 | v7 without smoothness penalties, sideways ×2 | ≈ v7 |

## Lessons
- **The imitation term is what preserves ETH's driving.** Pure reward fine-tuning (v1, v2)
  rewrote the gait. The term is `eth_imitation` in `eth_finetune_env_cfg.py`: a frozen ETH
  LSTM runs alongside, with its state reset per episode.
- **A stance penalty must not apply while moving sideways or turning.** ETH does those by
  narrowing.
- **The adaptive LR with std 0.15 sits at the 1e-5 floor** (desired_kl 0.005): a tiny noise
  std makes small mean changes look like large KL. desired_kl 0.02 gives 2–7e-5.
- **robot_lab's `error_vel_xy` metric accumulates over an episode.** It rises in the critic
  warm-up with the actor frozen. Use `train_eth_ft.py --eval_only` instead (`--eval_cmd`
  with fixed commands, `--no_rand`).

## Tools added
- `train_eth_ft.py`:
  - `--eval_cmd "a,b,c;…"` runs fixed commands back to back.
  - `--no_rand`, `--rewarm`, `--desired_kl`.
  - The evaluation reports the straight-only track.
- `eth_finetune_env_cfg.py`: v3–v8 configs, `MixedVelocityCommand`, `eth_imitation`,
  `hip_target_deviation`, `track_lin_vel_y_exp`.
