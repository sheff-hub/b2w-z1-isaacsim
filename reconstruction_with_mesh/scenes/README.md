# Scene files

One short YAML per scene. It is merged over `../config.yaml` (machine + defaults), and the command line wins over
both. Only write what differs from the defaults.

```yaml
recording:
  name: 2026-10-22--16-54-08   # Record3D export folder in <paths.nurec>/mine/dataset_for_with_mesh/, or absolute
  scene_name: my_flat          # output folder under paths.scenes_root
prep:
  image_step_m: 0.06           # one image per 6 cm / 3 deg of motion (smaller = more images, slower)
train:
  caps: [1000000, 2000000]     # compare budgets (each ~25 min); or leave null for cap_ratio x the BA points
```

Several passes of one place (e.g. the phone at three heights) and a test clip:

```yaml
recording:
  passes:                      # the first is the reference frame: record it at chest height, two laps
    - my_flat/1_mid
    - my_flat/2_low
    - my_flat/3_high
  test: my_flat/test_0p8m      # never trained on; the splat is scored on it (stage test_eval)
  scene_name: my_flat_3heights
```

(A folder holding all of them, `python3 reconstruct.py my_flat`, does the same without a scene file: every export
in it is a pass, in name order, and one in a `test*` subfolder is the test clip.)

Run it by its file name:  `python3 reconstruct.py my_flat`  (`--list-scenes` lists them).

| file | recording | output |
|---|---|---|
| `record3d_2laps.yaml` | `2026-10-22--16-54-08` (407 s, two laps of a ring corridor) | `record3d_2laps` (the automated M1) |
