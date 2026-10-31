"""Pack meshes of one camera group for the corridor viewer, in that group's levelled frame (group_eval.py report).

    python pack_group.py <group sparse txt> <group_eval report.json> <viewer dir> <scene name> id=mesh.ply [id=...]
        [splat=<splat.ply>,<dataparser_transforms.json>]   # optional: opaque splat centres as a point layer

Writes <viewer>/<scene name>.json (box, COLMAP points, camera path, m_per_unit, per-mesh scores) and
<viewer>/<id>.json per mesh, in the same quantised format as corridor_eval.py.
"""
import base64
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path.home() / "Firdavs/repos/Depth-Anything-3/src"))
from depth_anything_3.utils.read_write_model import read_model  # noqa: E402

VIEW_TRIS = 200_000
sparse, rep_path, vout, scene_name = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
meshes = dict(a.split("=", 1) for a in sys.argv[5:])
splat_arg = meshes.pop("splat", None)
rep = json.loads(rep_path.read_text())
lf, mpu = rep["level_frame"], rep["frame"]["m_per_unit"]
e1, e2, up, o = (np.array(lf[k]) for k in ("e1", "e2", "up", "origin"))
_, ims, pts = read_model(sparse, ".txt")
ims = dict(sorted(ims.items(), key=lambda kv: kv[1].name))
cams = np.array([-Rotation.from_quat(np.r_[im.qvec[1:], im.qvec[0]]).as_matrix().T @ im.tvec for im in ims.values()])
sfm = np.array([p.xyz for p in pts.values()])
L = lambda p: np.c_[(p - o) @ e1, (p - o) @ e2, (p - o) @ up] * mpu
C = L(cams)
lo, hi = C[:, :2].min(0), C[:, :2].max(0); mid = (lo + hi) / 2; margin = 0.25 * (hi - lo).max()
to_view = lambda p: L(p) - np.r_[mid, 0]
box_lo = np.r_[lo - mid - margin, -1.0]; box_hi = np.r_[hi - mid + margin, 3.5]
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
quant = lambda p: np.clip(np.round((p - box_lo) / (box_hi - box_lo) * 65535), 0, 65535).astype("<u2")
stats = {}
for mid_, path in meshes.items():
    m = o3d.io.read_triangle_mesh(path)
    V, F = to_view(np.asarray(m.vertices)), np.asarray(m.triangles)
    keep = np.all((V[F] >= box_lo) & (V[F] <= box_hi), 2).all(1)
    m2 = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F[keep]))
    m2.remove_unreferenced_vertices()
    n_crop = len(m2.triangles)
    if n_crop > VIEW_TRIS:
        m2 = m2.simplify_quadric_decimation(VIEW_TRIS)
    m2.remove_unreferenced_vertices(); m2.remove_degenerate_triangles()
    V2, F2 = np.asarray(m2.vertices), np.asarray(m2.triangles).astype("<u4")
    (vout / f"{mid_}.json").write_text(json.dumps({"nv": len(V2), "nt": len(F2), "f32": False, "v": b64(quant(V2)), "f": b64(F2)}))
    s = dict(rep["meshes"][mid_]); s["viewer_tris"] = len(F2); s["tris_in_box"] = n_crop; stats[mid_] = s
    print(f"{mid_}: {n_crop:,} tris in box -> {len(F2):,} shown")
Sv = to_view(sfm); Sv = Sv[np.all((Sv >= box_lo) & (Sv <= box_hi), 1)]
extra = {}
if splat_arg:
    from plyfile import PlyData
    ply, dpt = splat_arg.split(",")
    sp = PlyData.read(ply)["vertex"]; dp = json.loads(Path(dpt).read_text())
    T, s_ = np.array(dp["transform"]), float(dp["scale"])
    G = (np.c_[sp["x"], sp["y"], sp["z"]] / s_ - T[:, 3]) @ T[:, :3]  # nerfstudio model frame -> COLMAP
    op = 1 / (1 + np.exp(-sp["opacity"]))
    col = np.clip(0.5 + 0.28209479 * np.c_[sp["f_dc_0"], sp["f_dc_1"], sp["f_dc_2"]], 0, 1)
    Gv = to_view(G); ok = (op > 0.5) & np.all((Gv >= box_lo) & (Gv <= box_hi), 1)
    idx = np.random.default_rng(0).choice(np.flatnonzero(ok), min(150_000, int(ok.sum())), replace=False)
    extra = {"splat": b64(quant(Gv[idx])), "splat_rgb": b64((col[idx] * 255).astype(np.uint8)), "n_splat": len(idx)}
    print(f"splat: {len(G):,} gaussians, {int(ok.sum()):,} opaque in box, {len(idx):,} shown")
(vout / f"{scene_name}.json").write_text(json.dumps({**extra,
    "box_lo": box_lo.tolist(), "box_hi": box_hi.tolist(), "m_per_unit": mpu, "frame": rep["frame"],
    "sfm": b64(quant(Sv)), "n_sfm": len(Sv), "cams": np.round(to_view(cams), 3).tolist(), "meshes": stats}))
print(f"{scene_name}: {len(Sv)} COLMAP points, {len(cams)} cameras")
