#!/bin/bash
# Drive the B2W around a stage on its wheels -- Rung 0b of the Phase 4 ladder.
#
# Same GPU pin and mounts as the other launchers; see isaac_env.sh.
#
#   ./run_teleop.sh --selftest     headless, scripted, prints a report and exits
#   ./run_teleop.sh                keyboard over the WebRTC livestream
#   ./run_teleop.sh --gui          keyboard in an X11 window on this machine
#
# Anything else is passed straight through to b2w_teleop.py, e.g.
#   ./run_teleop.sh --selftest --stage /workspace/robots_usd/b2w_on_ground.usd
#   ./run_teleop.sh --gui --wheel-sign -1 --max-speed 1.5

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=isaac_env.sh
source "${SCRIPT_DIR}/../isaac/isaac_env.sh"

MODE="livestream"
PASS_ARGS=()
for a in "$@"; do
    case "$a" in
        --selftest) MODE="selftest"; PASS_ARGS+=("$a") ;;
        --gui)      MODE="gui";      PASS_ARGS+=("$a") ;;
        *)          PASS_ARGS+=("$a") ;;
    esac
done

# Livestream is the default, but the flag has to reach the script explicitly so
# it knows to enable the WebRTC extension.
if [ "${MODE}" = "livestream" ]; then
    PASS_ARGS+=(--livestream)
fi

CONTAINER_NAME="${CONTAINER_NAME:-isaac-sim-teleop}"

echo "B2W wheel teleop (Isaac Sim 5.1.0, mode: ${MODE})..."
print_banner
[ "${MODE}" = "livestream" ] && echo "  Livestream: ${LIVESTREAM_ADDR}:${LIVESTREAM_PORT}"
echo

preflight
ensure_dirs
handle_existing_container "${CONTAINER_NAME}"
set_run_mode

EXTRA=()
if [ "${MODE}" = "gui" ]; then
    if [ -z "${DISPLAY:-}" ]; then
        echo "ERROR: DISPLAY is not set -- drop --gui to use the livestream." >&2
        exit 1
    fi
    # Same reasoning as launch_gui.sh: the container runs as uid 1234, which is
    # not a host user, so it needs the local X grant. Not revoked on exit, so
    # `docker start -i` still works.
    command -v xhost >/dev/null 2>&1 && xhost +local: >/dev/null || true
    EXTRA=(-e "DISPLAY=${DISPLAY}" -v /tmp/.X11-unix:/tmp/.X11-unix)
fi

# --selftest exits on its own, so let Docker clean up after it. The interactive
# modes keep the container so it can be restarted with `docker start -i`.
RM_FLAG=()
[ "${MODE}" = "selftest" ] && RM_FLAG=(--rm) && RUN_MODE_FLAGS=()

docker run \
    --name "${CONTAINER_NAME}" \
    --entrypoint bash \
    "${RM_FLAG[@]}" \
    "${RUN_MODE_FLAGS[@]}" \
    --gpus "${GPU_SELECT}" \
    "${COMMON_ENV[@]}" \
    "${EXTRA[@]}" \
    --network=host \
    --ipc=host \
    "${WORKSPACE_MOUNTS[@]}" \
    "${CACHE_MOUNTS[@]}" \
    "${ISAAC_IMAGE}" \
    -c "./python.sh /workspace/project/b2w_teleop.py ${PASS_ARGS[*]} 2>&1"
