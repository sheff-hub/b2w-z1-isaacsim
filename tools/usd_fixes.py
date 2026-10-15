"""Small USD clean-ups shared by the asset build tools.

drop_dangling_refs: Isaac Sim's URDF importer gives frame-only links (sensor
mounts, camera frames, the Z1's `world`) a `visuals` prim that references
`<..._physics.usd>/visuals/<link>`, a prim it never writes. Each shows as a red
"unresolved prim path" in the Stage tree. The reference cannot be removed from a
stronger layer (list edits don't cross a payload), so it is removed where it is
authored, and only when its target prim really does not exist.

    python3 tools/usd_fixes.py <layer.usd> [...]

Lives in Isaac/ so the Isaac container (which mounts Isaac/, not tools/) can use it.
"""

import sys
from pathlib import Path

from pxr import Sdf


def drop_dangling_refs(layer_path) -> list:
    layer = Sdf.Layer.FindOrOpen(str(layer_path))
    dropped = []

    def visit(path):
        spec = layer.GetPrimAtPath(path)
        if spec is None:
            return
        refs = list(spec.referenceList.GetAddedOrExplicitItems())
        keep = []
        for r in refs:
            if not r.assetPath:  # internal reference: check this layer
                ok = r.primPath.isEmpty or layer.GetPrimAtPath(r.primPath) is not None
            else:
                target = Sdf.Layer.FindOrOpenRelativeToLayer(layer, r.assetPath)
                ok = target is not None and (r.primPath.isEmpty or target.GetPrimAtPath(r.primPath) is not None)
            (keep if ok else dropped).append(r if ok else f"{path} -> {r.assetPath}<{r.primPath}>")
        if len(keep) != len(refs):
            spec.referenceList.ClearEdits()
            if keep:
                spec.referenceList.prependedItems = keep

    layer.Traverse(Sdf.Path.absoluteRootPath, lambda p: visit(p) if p.IsPrimPath() else None)
    if dropped:
        layer.Save()
    return dropped


if __name__ == "__main__":
    for f in sys.argv[1:]:
        d = drop_dangling_refs(Path(f))
        print(f"{f}: dropped {len(d)} dangling reference(s)")
        for x in d:
            print("   ", x)
