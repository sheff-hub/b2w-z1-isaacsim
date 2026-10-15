#!/bin/bash
# Isaac Sim 5.1.0 -- local GUI mode (X11) with ROS 2 Jazzy.
#
# Lab workstation version. See isaac_env.sh for all machine-specific settings
# and for why the GPU is pinned rather than passed as --gpus all.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=isaac_env.sh
source "${SCRIPT_DIR}/isaac_env.sh"

CONTAINER_NAME="${CONTAINER_NAME:-isaac-sim-gui}"

echo "Starting Isaac Sim 5.1.0 (Local GUI Mode / ROS 2 Jazzy)..."
print_banner
echo "  Display   : ${DISPLAY:-<unset>}"
echo

preflight
ensure_dirs
handle_existing_container "${CONTAINER_NAME}"

if [ -z "${DISPLAY:-}" ]; then
    echo "ERROR: DISPLAY is not set -- there is no X server to render into." >&2
    echo "       Use launch_headless.sh for remote/livestream operation." >&2
    exit 1
fi

# The container runs as uid 1234 ('isaac-sim'), which does not correspond to any
# host user, so it cannot authenticate to the X server on its own. Note this is
# NOT root -- `xhost +local:root` would grant the wrong identity and the window
# would silently fail to open.
#
# `xhost +local:` allows any client connecting over the local unix socket. The
# socket is only reachable by processes on this machine, and the grant is
# revoked on exit.
set_run_mode

# The grant is NOT revoked on exit, and that is deliberate.
#
# Revoking breaks the normal resume workflow: `docker start -i isaac-sim-gui`
# reuses the container's DISPLAY and X11 mount, but without the grant the
# container (uid 1234) can no longer reach the X server -- Isaac Sim loads and
# then has nowhere to draw, printing "Isaac Sim Full App is loaded." with no
# window and no error. The original office-workstation script never revoked, which
# is why start/stop cycles worked there.
#
# The grant lasts until you log out or the X session restarts; it is not permanent.
#
# Trade-off to be aware of: while granted, any user logged into this machine can
# connect to your X display. Colleagues do have sessions on this box. To revoke
# manually when you are done for the day:
#     xhost -local:
# Or set XHOST_REVOKE=1 to restore revoke-on-exit (which disables `docker start -i`).
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

docker run \
    --name "${CONTAINER_NAME}" \
    --entrypoint bash \
    "${RUN_MODE_FLAGS[@]}" \
    --gpus "${GPU_SELECT}" \
    "${COMMON_ENV[@]}" \
    -e "DISPLAY=${DISPLAY}" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --network=host \
    --ipc=host \
    --pid=host \
    "${WORKSPACE_MOUNTS[@]}" \
    "${CACHE_MOUNTS[@]}" \
    "${ISAAC_IMAGE}" \
    -c "./isaac-sim.sh --allow-root"

if [ "${DETACHED}" -eq 1 ]; then
    echo
    echo "Started detached (no TTY on stdin)."
    echo "  Follow startup : docker logs -f ${CONTAINER_NAME}"
    echo "  Stop           : docker stop ${CONTAINER_NAME}"
    echo "Note: 'xhost +local:' is left granted while the container runs."
fi
