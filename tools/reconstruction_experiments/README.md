# tools/reconstruction_experiments: the studies behind the reconstruction pipelines

Research scripts from 20-22 Oct 2026: turning a splat into a collision mesh, and the four-way comparison of
reconstruction methods on one iPhone recording (`record3d_2laps`). The winner of that comparison (called M1) became
`reconstruction_with_mesh/`; these scripts are what was tried and measured on the way. They are kept as run, not
maintained: paths in their docstrings point at folders on the lab machine (`~/Firdavs/...`, `scenes/nurec/mine/...`),
and most run in the 3DGRUT venv (`<paths.threedgrut>/.venv/bin/python`, which has torch, pxr and open3d).

| Folder | Study | Scripts |
|---|---|---|
| `tsdf/` | Collision mesh from a splat: render depth from a trained 3DGRUT model, fuse it (TSDF + marching cubes), cut at the real floor plane, and wrap splat + invisible collider into an Isaac stage. The splat's floor sagged up to 1.3 m, so the LiDAR TSDF won later. | `render_depth.py`, `check_depth.py`, `fuse_tsdf.py`, `floor_and_crop.py`, `make_collision_stage.py`, `ns_splat_to_colmap_frame.py`, `colmap_to_ns_mesh.py`, `corridor_eval.py` |
| `aria/` | A colleague's method (Depth Anything 3 depth + VDBFusion): scoring DA3 depth, splitting the walk into camera groups with consistent poses, scoring the meshes per group. | `check_da3_depth.py`, `cam_passes.py`, `pass_points.py`, `make_group.py`, `group_eval.py`, `pack_group.py`, `ns_mesh_to_colmap.py`, `turn_test.py` |
| `nurec_stereo/` | NVIDIA's NuRec stereo workflow on our phone data: Record3D export -> cuSFM input, LiDAR depth in place of FoundationStereo. Same walk, same training steps: cuSFM poses 22.4 dB, ours 26.1 dB. | `make_cusfm_input.py`, `lidar_depth_pngs.py` |
| `record3d/` | The comparison itself: metric scoring of the four meshes, Isaac render grids across methods, Record3D PLY-sequence stand-in, a textured-mesh stage (nerfstudio route). | `eval_metric.py`, `compare_grid.py`, `r3d_ply_sequence.py`, `textured_mesh_stage.py` |
| `.` | Standard 3DGS PLY (e.g. nerfstudio) -> NuRec USDZ. | `ply_to_nurec_usdz.py` |

The tools of the method that won (pose repair, loop closure, LiDAR fusion, routes, checks) are in
`reconstruction_with_mesh/tools/`.
