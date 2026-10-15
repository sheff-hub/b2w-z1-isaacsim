#!/bin/bash
# Isaac Sim 6.0.1 -- local GUI mode (X11) with ROS 2 Jazzy.
#
# Side-by-side with the 5.1.0 setup: different image, different container name,
# and a separate cache root. See isaac_env_6.sh for what differs from 5.1.0 and
# isaac_env.sh for the GPU pin and why it is not --gpus all.
#
#   ./launch_gui_6.sh              fresh container named isaac-sim-gui-6
#   docker start -i isaac-sim-gui-6   resume it afterwards, state intact
#
# Re-running this script deletes the previous isaac-sim-gui-6 container and
# builds a new one. Use `docker start -i` to keep in-container changes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=isaac_env_6.sh
source "${SCRIPT_DIR}/isaac_env_6.sh"

CONTAINER_NAME="${CONTAINER_NAME:-isaac-sim-gui-6}"

echo "Starting Isaac Sim 6.0.1 (Local GUI Mode / ROS 2 Jazzy)..."
print_banner_6
echo "  Container : ${CONTAINER_NAME}"
echo "  Display   : ${DISPLAY:-<unset>}"
echo

preflight
require_image
ensure_dirs_6
probe_ros_bridge
handle_existing_container "${CONTAINER_NAME}"

if [ -z "${DISPLAY:-}" ]; then
    echo "ERROR: DISPLAY is not set -- there is no X server to render into." >&2
    echo "       Use launch_headless_6.sh for remote/livestream operation." >&2
    exit 1
fi

set_run_mode

# The container runs as uid 1234 ('isaac-sim'), not root, so it cannot
# authenticate to the X server on its own -- `xhost +local:root` would grant the
# wrong identity and the window would silently never appear.
#
# The grant is deliberately NOT revoked on exit: `docker start -i` reuses the
# container's DISPLAY and X11 mount, and without the grant Isaac Sim loads with
# nowhere to draw, printing "Isaac Sim Full App is loaded." and no window.
# See launch_gui.sh for the full rationale -- this mirrors it exactly.
#
# While granted, any user logged into this machine can reach your X display.
# Revoke manually when done:  xhost -local:
XHOST_REVOKE="${XHOST_REVOKE:-0}"
XHOST_GRANTED=0
if command -v xhost >/dev/null 2>&1; then
    xhost +local: >/dev/null && XHOST_GRANTED=1
    if [ "${XHOST_REVOKE}" -eq 1 ]; then
        cleanup() {
            if [ "${XHOST_GRANTED}" -eq 1 ] && [ "${DETACHED}" -eq 0 ]; then
                xhost -local: >/dev/null 2>&1 || true
            fi
        }
        trap cleanup EXIT
    fi
else
    echo "WARNING: xhost not found; if the window fails to open, run" >&2
    echo "         'xhost +local:' manually before launching." >&2
fi

STATUS=0
docker run \
    --name "${CONTAINER_NAME}" \
    --entrypoint bash \
    "${RUN_MODE_FLAGS[@]}" \
    --gpus "${GPU_SELECT}" \
    "${USER_FLAGS[@]}" \
    "${COMMON_ENV[@]}" \
    -e "DISPLAY=${DISPLAY}" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --network=host \
    --ipc=host \
    --pid=host \
    "${WORKSPACE_MOUNTS[@]}" \
    "${CACHE_MOUNTS[@]}" \
    "${ISAAC_IMAGE}" \
    -c "./isaac-sim.sh --allow-root" || STATUS=$?

if [ "${DETACHED}" -eq 1 ]; then
    echo
    echo "Started detached (no TTY on stdin)."
    echo "  Follow startup : docker logs -f ${CONTAINER_NAME}"
    echo "  Stop           : docker stop ${CONTAINER_NAME}"
    echo "Note: 'xhost +local:' is left granted while the container runs."
else
    print_resume_hint "${CONTAINER_NAME}"
fi

exit "${STATUS}"
