# evaluation/: the scripts behind the report's numbers

Inside the container (`./isaac/run_script.sh <script> ...`) unless marked *host* (plain `python3`).

| Script | Measures | In the report |
|---|---|---|
| `summarize_fetch.py` *(host)* | success, delivery, falls, timing, dock and grasp errors of fetch result files | Table III (`run.py fetch` prints it) |
| `compare_vln_followers.py` *(host)* | who follows DualVLN's short paths: pure pursuit, Nav2 FollowPath, NavigateToPose | Table IV |
| `compare_nav2.py` *(host)* | two fetch result files episode by episode (e.g. scripted vs Nav2 parking) | Nav2 parking |
| `../project/eth_compare.py` | how much of a velocity command each walking policy follows, arm attached | Fig. 8 |
| `approach_test.py` | can the RFM policy get to a 6D target anywhere around it and arrive upright | "10 of 12 targets at 1–3 m" |
| `reach_map.py`, `crouch_test.py` | which world targets a policy reaches; target height sweep (does the body help the arm) | the crouch and reach results (Sec. III-C) |
| `workspace.py` | where the Z1's hand can go on its own, and with the body | "9 cm below the shoulder" |
| `shake_diag.py` | the shaking hand: wobble on target per case | the incremental arm actions |
| `eval_locomotion.py` | velocity tracking under a fixed command script | whole-body comparisons |
| `eval_all.sh`, `eval_rfm.sh` | the batteries above on one checkpoint; results in `repos/robot_lab/logs/eval/` | |
| `compare_rfm.py`, `plot_reach_map.py`, `tb_curve.py` *(host)* | tables and plots of those results | |
| `wrist_d405_replay.py` *(host, needs pxr)* | replays recorded picks with the wrist camera holder attached: does it touch anything | the holder check |
