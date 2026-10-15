# Scene files

One short YAML per room. It is merged over `../config.yaml` (machine + defaults), and the command line wins over
both. Only write what differs from the defaults.

```yaml
dataset:
  name: my_photos            # folder under paths.dataset_root (or an absolute path)
  scene_name: my_room        # output folder under paths.scenes_root
colmap:
  max_image_size: null       # null = native below 500 photos, else 2000
train:
  downsample_factor: 2       # 2 = 2016x1512 for 12 MP photos
  cap_ratio: 2.0             # gaussians per COLMAP point; or caps: [300000, 400000] to compare
```

Run it by its file name:  `python3 reconstruct.py my_room`  (`--list-scenes` lists them).

| file | photos | output | best manual result |
|---|---|---|---|
| `home.yaml` | `room_0.5x` (428) | `room_from_script` | 25.73 dB at 400k |
| `kitchen.yaml` | `kitchen_1x` (932) | `kitchen_from_script` | 22.62 dB at 750k |
