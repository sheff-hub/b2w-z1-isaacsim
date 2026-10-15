#!/bin/bash
# Open a stage headlessly in Isaac Sim 5.1.0 and report problems.
#
# Unlike launch_headless.sh this does not start the livestream app: it runs
# open_stage_check.py under the container's python.sh, prints a report and
# exits. Same GPU pin and same mounts -- see isaac_env.sh.
#
#   ./run_stage_check.sh
#   ./run_stage_check.sh /workspace/scenes/nurec/nvidia/<other>/stage_volume.usdz
#
# NO_ROS_LD=1 drops the ROS 2 LD_LIBRARY_PATH override, which is useful when
# isolating library-loading failures in the RTX renderer.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=isaac_env.sh
source "${SCRIPT_DIR}/isaac_env.sh"

CONTAINER_NAME="${CONTAINER_NAME:-isaac-sim-stagecheck}"
STAGE="${1:-/workspace/scenes/nurec/nvidia/hand_hold-endeavor-livingroom/stage_volume.usdz}"

echo "Opening stage headlessly in Isaac Sim 5.1.0..."
print_banner
echo "  Stage     : ${STAGE}"
echo

preflight
ensure_dirs
handle_existing_container "${CONTAINER_NAME}"

ENV_FLAGS=("${COMMON_ENV[@]}")
if [ "${NO_ROS_LD:-0}" = "1" ]; then
    # Drop the LD_LIBRARY_PATH entry (last element of COMMON_ENV).
    ENV_FLAGS=()
    for e in "${COMMON_ENV[@]}"; do
        case "$e" in
            LD_LIBRARY_PATH=*) continue ;;
            -e) ;;
        esac
        ENV_FLAGS+=("$e")
    done
    # Re-pair the -e flags that were left dangling.
    FIXED=()
    for ((i = 0; i < ${#ENV_FLAGS[@]}; i++)); do
        if [ "${ENV_FLAGS[$i]}" = "-e" ] && [ $((i + 1)) -ge ${#ENV_FLAGS[@]} ]; then
            continue
        fi
        FIXED+=("${ENV_FLAGS[$i]}")
    done
    ENV_FLAGS=("${FIXED[@]}")
    echo "NO_ROS_LD=1: running without the ROS 2 LD_LIBRARY_PATH override"
fi

docker run --rm \
    --name "${CONTAINER_NAME}" \
    --entrypoint bash \
    --gpus "${GPU_SELECT}" \
    "${ENV_FLAGS[@]}" \
    --network=host \
    --ipc=host \
    "${WORKSPACE_MOUNTS[@]}" \
    "${CACHE_MOUNTS[@]}" \
    "${ISAAC_IMAGE}" \
    -c "./python.sh /workspace/isaac/open_stage_check.py '${STAGE}' 2>&1"
