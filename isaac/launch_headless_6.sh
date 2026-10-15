#!/bin/bash
# Isaac Sim 6.0.1 -- headless mode with WebRTC livestream, ROS 2 Jazzy.
#
# Side-by-side with the 5.1.0 setup: different image, different container name,
# separate cache root and separate livestream ports. See isaac_env_6.sh.
#
#   ./launch_headless_6.sh                 fresh container isaac-sim-headless-6
#   docker start -i isaac-sim-headless-6   resume it afterwards, state intact
#
# Re-running this script deletes the previous isaac-sim-headless-6 container and
# builds a new one. Use `docker start -i` to keep in-container changes.
#
# NOTE the livestream settings are configured by ENVIRONMENT VARIABLE, not by
# command-line flag. 6.0.1's own /isaac-sim/runheadless.sh reads
#   ISAACSIM_HOST, ISAACSIM_SIGNAL_PORT, ISAACSIM_STREAM_PORT
# and builds the corresponding --/exts/omni.kit.livestream.app/primaryStream/*
# flags itself, inserting them BEFORE any arguments we pass. That is the route
# NVIDIA intends for the container, so we use it rather than writing the flags
# out by hand.
#
# Do not copy 5.1.0's --/app/livestream/publicEndpointAddress and
# --/app/livestream/port here: 6.0 removed those keys, and Kit accepts unknown
# --/... settings silently, so the mistake would look like a network fault.
# Source: docs.isaacsim.omniverse.nvidia.com/6.0.1/installation/manual_livestream_clients.html
# and /isaac-sim/runheadless.sh inside the 6.0.1 image.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=isaac_env_6.sh
source "${SCRIPT_DIR}/isaac_env_6.sh"

CONTAINER_NAME="${CONTAINER_NAME:-isaac-sim-headless-6}"

# Optionally open a stage on startup:
#   STAGE=/workspace/scenes/nurec/.../stage_volume.usdz ./launch_headless_6.sh
#
# A Kit app that is already running cannot be told to open a stage from outside
# -- the HTTP service on port 8011 exposes only asset-conversion and health
# endpoints, nothing for the stage. So the stage has to be opened at startup,
# via Kit's --exec hook. Path is passed in the environment, not as an --exec
# argument, to keep it clear of the nested shell quoting in `docker run -c`.
STAGE="${STAGE:-}"
STAGE_ENV=()
RUN_CMD="./runheadless.sh"
if [ -n "${STAGE}" ]; then
    STAGE_ENV=(-e "STAGE_TO_OPEN=${STAGE}")
    RUN_CMD="${RUN_CMD} --exec /workspace/isaac/open_stage_stream.py"
fi

echo "Starting Isaac Sim 6.0.1 (Headless / Livestream / ROS 2 Jazzy)..."
print_banner_6
echo "  Container : ${CONTAINER_NAME}"
echo "  Livestream: signalling ${LIVESTREAM_ADDR}:${LIVESTREAM_SIGNAL_PORT}/tcp, media ${LIVESTREAM_STREAM_PORT}/udp"
echo "              in the Isaac Sim WebRTC Streaming Client, enter just: ${LIVESTREAM_ADDR}"
echo "  Stage     : ${STAGE:-<none, empty stage>}"
echo

preflight
require_image
ensure_dirs_6
probe_ros_bridge
# Order matters: handle_existing_container first. When a container of this same
# name is already running it is also holding the livestream ports, and its
# "already running, attach or stop it" message is the useful one. Checking ports
# first would instead report a port clash and point you at other containers.
handle_existing_container "${CONTAINER_NAME}"
check_ports_free "${LIVESTREAM_SIGNAL_PORT}/tcp" "${LIVESTREAM_STREAM_PORT}/udp"
set_run_mode

# Catch the most common misconfiguration: an address that is not actually on
# this host means the client can never connect to it.
if ! ip -4 -br addr | grep -qw "${LIVESTREAM_ADDR}"; then
    echo "WARNING: ${LIVESTREAM_ADDR} is not an address on this machine." >&2
    echo "         Available:" >&2
    ip -4 -br addr | awk '{printf "           %-14s %s\n", $1, $3}' >&2
    echo "         Override with: LIVESTREAM_ADDR=<addr> $0" >&2
    echo >&2
fi

STATUS=0
docker run \
    --name "${CONTAINER_NAME}" \
    --entrypoint bash \
    "${RUN_MODE_FLAGS[@]}" \
    --gpus "${GPU_SELECT}" \
    "${USER_FLAGS[@]}" \
    "${COMMON_ENV[@]}" \
    -e "ISAACSIM_HOST=${LIVESTREAM_ADDR}" \
    -e "ISAACSIM_SIGNAL_PORT=${LIVESTREAM_SIGNAL_PORT}" \
    -e "ISAACSIM_STREAM_PORT=${LIVESTREAM_STREAM_PORT}" \
    "${STAGE_ENV[@]}" \
    --network=host \
    --ipc=host \
    --pid=host \
    "${WORKSPACE_MOUNTS[@]}" \
    "${CACHE_MOUNTS[@]}" \
    "${ISAAC_IMAGE}" \
    -c "${RUN_CMD}" || STATUS=$?
# --allow-root is deliberately gone: it only permits Kit to run AS root, and
# USER_FLAGS pins the container to uid 1234 (isaac-sim). Carrying it over from
# the 5.1.0 script would suggest this runs as root, which it does not.

if [ "${DETACHED}" -eq 1 ]; then
    echo
    echo "Started detached (no TTY on stdin)."
    echo "  Follow startup : docker logs -f ${CONTAINER_NAME}"
    echo "  Stop           : docker stop ${CONTAINER_NAME}"
else
    print_resume_hint "${CONTAINER_NAME}"
fi

exit "${STATUS}"
