#!/bin/bash
# Isaac Sim 5.1.0 -- headless mode with livestream, ROS 2 Jazzy.
#
# Lab workstation version. See isaac_env.sh for all machine-specific settings
# and for why the GPU is pinned rather than passed as --gpus all.
#
# The livestream address is no longer hardcoded. The previous value
# (172.23.59.19) was the OFFICE workstation's ZeroTier address and does not
# exist on this machine.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=isaac_env.sh
source "${SCRIPT_DIR}/isaac_env.sh"

CONTAINER_NAME="${CONTAINER_NAME:-isaac-sim-headless}"

echo "Starting Isaac Sim 5.1.0 (Headless / Livestream / ROS 2 Jazzy)..."
print_banner
echo "  Livestream: ${LIVESTREAM_ADDR}:${LIVESTREAM_PORT}"
echo

preflight
ensure_dirs
handle_existing_container "${CONTAINER_NAME}"
set_run_mode

# Catch the most common misconfiguration: an address that isn't actually on
# this host means the client will never be able to connect.
if ! ip -4 -br addr | grep -qw "${LIVESTREAM_ADDR}"; then
    echo "WARNING: ${LIVESTREAM_ADDR} is not an address on this machine." >&2
    echo "         Available:" >&2
    ip -4 -br addr | awk '{printf "           %-14s %s\n", $1, $3}' >&2
    echo "         Override with: LIVESTREAM_ADDR=<addr> $0" >&2
    echo >&2
fi

docker run \
    --name "${CONTAINER_NAME}" \
    --entrypoint bash \
    "${RUN_MODE_FLAGS[@]}" \
    --gpus "${GPU_SELECT}" \
    "${COMMON_ENV[@]}" \
    --network=host \
    --ipc=host \
    --pid=host \
    "${WORKSPACE_MOUNTS[@]}" \
    "${CACHE_MOUNTS[@]}" \
    "${ISAAC_IMAGE}" \
    -c "./runheadless.sh --allow-root \
        --/app/livestream/publicEndpointAddress=${LIVESTREAM_ADDR} \
        --/app/livestream/port=${LIVESTREAM_PORT}"
