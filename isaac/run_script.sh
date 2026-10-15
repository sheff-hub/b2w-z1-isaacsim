#!/bin/bash
# Run a Python script of this repository inside the Isaac Lab container, with the GPU pin and mounts of
# isaac_env.sh (config.yaml). The script is a path relative to the repository, or a bare name that is looked
# up in project/, evaluation/, training/ and tools/:
#
#   ./isaac/run_script.sh project/fetch_box.py --episodes 3
#   ./isaac/run_script.sh workspace.py --samples 20000          (= evaluation/workspace.py)
#   ISAAC_IMAGE=nvcr.io/nvidia/isaac-sim:5.1.0 ./isaac/run_script.sh ...   (plain Isaac Sim)
#
# Runs attached and removes the container afterwards. Output goes to stdout.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ISAAC_IMAGE_DEFAULT=lab   # the Isaac Lab image (built on Isaac Sim 5.1.0) runs every script here
# shellcheck source=isaac_env.sh
source "${SCRIPT_DIR}/isaac_env.sh"

[ $# -ge 1 ] || { echo "usage: $0 <script.py> [args...]" >&2; exit 2; }
PY_SCRIPT="$1"; shift
# a path relative to the repository (project/fetch_box.py), or a bare name found in one of these folders
if [[ "${PY_SCRIPT}" != */* ]]; then
    for d in project evaluation training tools isaac; do
        if [ -f "${REPO_ROOT}/${d}/${PY_SCRIPT}" ]; then PY_SCRIPT="${d}/${PY_SCRIPT}"; break; fi
    done
fi
[ -f "${REPO_ROOT}/${PY_SCRIPT}" ] || { echo "ERROR: no script ${PY_SCRIPT} in ${REPO_ROOT}" >&2; exit 2; }
# Preserve quoting when the arguments are re-expanded inside `bash -c`: a value
# containing spaces (e.g. --waypoints "2,0; 2,2") would otherwise be split and
# silently truncated to its first word.
QUOTED_ARGS=""
for a in "$@"; do QUOTED_ARGS+=" $(printf '%q' "$a")"; done

CONTAINER_NAME="${CONTAINER_NAME:-isaac-script}"

preflight
ensure_dirs
handle_existing_container "${CONTAINER_NAME}"

docker run \
    --name "${CONTAINER_NAME}" \
    --entrypoint bash \
    --rm \
    --gpus "${GPU_SELECT}" \
    "${COMMON_ENV[@]}" \
    -e "PYTHONUNBUFFERED=1" \
    --network=host \
    --ipc=host \
    "${WORKSPACE_MOUNTS[@]}" \
    "${CACHE_MOUNTS[@]}" \
    -w /workspace/repos/robot_lab \
    "${ISAAC_IMAGE}" \
    -c "/isaac-sim/python.sh /workspace/${PY_SCRIPT}${QUOTED_ARGS} 2>&1"
