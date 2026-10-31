"""Refine drift-corrected Record3D poses with COLMAP: triangulate with poses fixed, bundle-adjust, align back.

    python refine_poses.py <scene dir> [base poses file in record3d/, default poses.npz]
                                                # after prep_record3d.py + feature extraction + matching
                                                # (+ loop_close.py for long walks: base poses_lc.npz)

Needs <scene>/colmap/database.db (features + matches for <scene>/images) and <scene>/sparse/0 (phone poses).
Steps:
  1. make the database's cameras/rigs/frames match the model's focal-length bins;
  2. triangulate with the phone poses fixed (reports the phone poses' reprojection error);
  3. two rounds of bundle adjustment (intrinsics fixed), dropping points over 4 px between rounds;
  4. frames with < MIN_PTS points keep their phone pose (BA has nothing to hold them);
  5. Sim3-align to the phone camera centres, so the frame stays metric, Z-up, floor at 0.
Writes <scene>/sparse/0 (refined; the phone model is moved to <scene>/sparse_phone/0) and
<scene>/record3d/poses_refined.npz (all frames: BA correction interpolated between kept frames of a pass).
"""
import shutil, sys
from pathlib import Path
import numpy as np, pycolmap
from scipy.spatial.transform import Rotation, Slerp
import r3dlib as L

MIN_PTS, MAX_ERR = 50, 4.0
scene = Path(sys.argv[1]); col = scene / "colmap"; base = sys.argv[2] if len(sys.argv) > 2 else "poses.npz"
phone_dir = scene / "sparse_phone" / "0"
if not phone_dir.exists():
    phone_dir.parent.mkdir(exist_ok=True); shutil.copytree(scene / "sparse" / "0", phone_dir)
phone = pycolmap.Reconstruction(str(phone_dir))
db = pycolmap.Database.open(str(col / "database.db"))
for cam in phone.cameras.values():
    db.update_camera(cam) if db.exists_camera(cam.camera_id) else db.write_camera(cam, use_camera_id=True)
for rig in phone.rigs.values():
    db.update_rig(rig) if db.exists_rig(rig.rig_id) else db.write_rig(rig, use_rig_id=True)
for fr in db.read_all_frames():
    rf = phone.frames[fr.frame_id]; new = pycolmap.Frame(); new.frame_id = fr.frame_id; new.rig_id = rf.rig_id
    for di in rf.data_ids: new.add_data_id(di)
    db.update_frame(new)
for img in phone.images.values():
    d = db.read_image(img.image_id); assert d.name == img.name; d.camera_id = img.camera_id; db.update_image(d)
db.close()

def err_stats(r, tag):
    e = np.array([p.error for p in r.points3D.values()])
    print(f"{tag}: {len(e):,} points, reprojection mean {e.mean():.2f} px, median {np.median(e):.2f} px")
def drop_bad(r):
    bad = [pid for pid, p in r.points3D.items() if not np.isfinite(p.error) or p.error > MAX_ERR]
    for pid in bad: r.delete_point3D(pid)
    return len(bad)

(col / "tri").mkdir(exist_ok=True)
rec = pycolmap.triangulate_points(pycolmap.Reconstruction(str(phone_dir)), str(col / "database.db"), str(scene / "images"),
                                  str(col / "tri"), clear_points=True, options=pycolmap.IncrementalPipelineOptions(),
                                  refine_intrinsics=False)
err_stats(rec, "phone poses, fixed")
o = pycolmap.BundleAdjustmentOptions(); o.refine_focal_length = o.refine_principal_point = o.refine_extra_params = False
o.ceres.solver_options.max_num_iterations = 100
for k in range(2):
    pycolmap.bundle_adjustment(rec, o); n = drop_bad(rec); err_stats(rec, f"bundle adjustment {k + 1} (dropped {n} points > {MAX_ERR:.0f} px)")
names = [im.name for im in rec.images.values()]
by_name = {im.name: im for im in phone.images.values()}
sim = pycolmap.align_reconstruction_to_locations(rec, names, np.array([by_name[n].projection_center() for n in names]), 3,
                                                 pycolmap.RANSACOptions(max_error=0.5))
rec.transform(sim)
print(f"Sim3 back to the phone frame: scale {sim.scale:.4f}, rotation {np.degrees(Rotation.from_matrix(sim.rotation.matrix()).magnitude()):.3f} deg")
kept = []
for im in rec.images.values():
    if im.num_points3D < MIN_PTS:
        kept.append(im.name); fr = rec.frames[im.frame_id]
        fr.rig_from_world = phone.frames[by_name[im.name].frame_id].rig_from_world
print(f"{len(kept)} frames with < {MIN_PTS} points keep their phone pose")
dc = np.array([np.linalg.norm(im.projection_center() - by_name[im.name].projection_center()) for im in rec.images.values()])
print(f"refined vs phone: camera position change median {np.median(dc) * 100:.1f} cm, p95 {np.percentile(dc, 95) * 100:.1f} cm, max {dc.max() * 100:.1f} cm")
out = scene / "sparse" / "0"; shutil.rmtree(out); out.mkdir(parents=True); rec.write_text(str(out)); rec.write(str(out))
err_stats(rec, "written to sparse/0")

# all-frame poses: interpolate the BA correction (refined @ inv(phone)) between kept frames, within each pass
z = dict(np.load(scene / "record3d" / base)); c2w = z["c2w"]; pl = L.Passes(z)
ref = {L.frame_of_name(im.name): np.linalg.inv(np.r_[im.cam_from_world().matrix(), [[0, 0, 0, 1]]]) for im in rec.images.values()}
D = np.tile(np.eye(4), (len(c2w), 1, 1))
for p, lo, hi in pl.ranges():
    ids = np.array(sorted(i for i in ref if lo <= i < hi))
    if len(ids) == 0:
        continue
    delta = np.stack([ref[i] @ np.linalg.inv(c2w[i]) for i in ids])
    fi = np.clip(np.arange(lo, hi), ids[0], ids[-1])
    D[lo:hi, :3, :3] = (Slerp(ids, Rotation.from_matrix(delta[:, :3, :3]))(fi).as_matrix() if len(ids) > 1
                        else np.repeat(delta[:1, :3, :3], hi - lo, 0))
    D[lo:hi, :3, 3] = np.stack([np.interp(fi, ids, delta[:, k, 3]) for k in range(3)], 1)
z["c2w"] = D @ c2w; np.savez(scene / "record3d" / "poses_refined.npz", **z)
print("wrote record3d/poses_refined.npz (all frames)")
