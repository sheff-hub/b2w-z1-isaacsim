"""Write a COLMAP text model holding only the given frame ranges (time order), with point tracks cut to them.

usage: make_group.py <src sparse txt> <out dataset dir> <images dir> 1-83,158-169,333-341
"""
import collections, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path.home() / "Firdavs/repos/Depth-Anything-3/src"))
from depth_anything_3.utils.read_write_model import read_model, write_model, Point3D
src, out, imdir, spec = sys.argv[1:5]
rng = [tuple(map(int, r.split("-"))) for r in spec.split(",")]
cams, ims, pts = read_model(src, ".txt")
keep = {i: im for i, im in ims.items() if any(a <= int(im.name.split("_")[1][:5]) <= b for a, b in rng)}
keep = collections.OrderedDict(sorted(keep.items(), key=lambda kv: kv[1].name))
newpts = {}
for pid, p in pts.items():
    m = np.isin(p.image_ids, list(keep))
    if m.sum() >= 2:
        newpts[pid] = Point3D(id=pid, xyz=p.xyz, rgb=p.rgb, error=p.error, image_ids=p.image_ids[m], point2D_idxs=p.point2D_idxs[m])
ims2 = collections.OrderedDict()
for i, im in keep.items():
    ids = np.where(np.isin(im.point3D_ids, list(newpts)), im.point3D_ids, -1)
    ims2[i] = im._replace(point3D_ids=ids)
(Path(out) / "sparse/0").mkdir(parents=True, exist_ok=True)
lk = Path(out) / "images"
if not lk.exists(): lk.symlink_to(imdir)
write_model(cams, ims2, newpts, str(Path(out) / "sparse/0"), ".txt")
print(f"{out}: {len(ims2)} images, {len(newpts)} points")
