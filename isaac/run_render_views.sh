#!/bin/bash
# Headless Isaac Sim 5.1: render a stage from the cameras of a views.json and drop test balls
# (reconstruction_with_mesh/tools/render_stage_views.py). Paths are inside the container: /workspace = this repository.
#   ./isaac/run_render_views.sh <stage> <views.json> <out dir> ["--splat-prim <prim> --collider-prim <prim>"]
# e.g. the SpatialAI scenes (scenes/spatialAI/README.md):
#   ./isaac/run_render_views.sh /workspace/scenes/spatialAI/kitchen/spatialAI_kitchen.usdz \
#       /workspace/scenes/spatialAI/kitchen/check/views.json /workspace/scenes/spatialAI/kitchen/check/renders \
#       "--splat-prim /World/volume/volume --collider-prim /World/volume/mesh"
# The out dir must be writable by the container's uid 1234 (mkdir -p ... && chmod 777 ...).
set -euo pipefail
[ $# -ge 3 ] || { echo "usage: $0 <stage> <views.json> <out dir> [extra args]" >&2; exit 2; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=isaac_env.sh
source "${SCRIPT_DIR}/isaac_env.sh"
preflight; ensure_dirs; handle_existing_container isaac-sim-renderviews
docker run --rm --name isaac-sim-renderviews --entrypoint bash --gpus "${GPU_SELECT}" "${COMMON_ENV[@]}" \
    --network=host --ipc=host "${WORKSPACE_MOUNTS[@]}" "${CACHE_MOUNTS[@]}" "${ISAAC_IMAGE}" \
    -c "./python.sh /workspace/reconstruction_with_mesh/tools/render_stage_views.py --stage '$1' --views '$2' --out '$3' ${4:-} 2>&1"
