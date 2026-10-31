#!/bin/bash
# Create this package's tools environment (.venv): pose prep, loop closure, bundle adjustment (pycolmap), LiDAR
# fusion (vdbfusion, open3d), mesh checks and stage building (usd-core). No sudo needed.
#   ./setup_tools_env.sh            (uses uv if installed, else python3 -m venv + pip)
# vdbfusion 0.1.6 ships wheels for Python <= 3.10, so the environment is Python 3.10.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3.10}
if command -v uv >/dev/null 2>&1; then
    uv venv -p "$PY" .venv
    UV_HTTP_TIMEOUT=600 uv pip install -p .venv/bin/python -r requirements-tools.txt
else
    "$PY" -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r requirements-tools.txt
fi
OPENCV_IO_ENABLE_OPENEXR=1 .venv/bin/python -c "
import cv2, numpy, open3d, pycolmap, scipy, trimesh, vdbfusion
from pxr import Usd
print('tools environment OK: pycolmap', pycolmap.__version__, '| open3d', open3d.__version__, '| numpy', numpy.__version__)"
