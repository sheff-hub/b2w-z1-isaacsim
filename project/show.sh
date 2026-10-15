#!/bin/bash
# Watch the Phase 4 results, without retyping long paths.
#
#   ./show.sh                     drive to coordinates you type          (wb9)
#   ./show.sh goto                same thing, explicitly
#   ./show.sh reach               watch it reach random world targets    (wb15)
#   ./show.sh reach wb9           ... with a different policy
#   ./show.sh fetch 2.5,0,0.25    one coordinate, drive there and reach  (wb9)
#   ./show.sh square              the waypoint course                    (wb9)
#   ./show.sh send 2.5 0 0.3      send a coordinate to a RUNNING goto
#   ./show.sh stop                stop whatever is running
#
# Add --gui anywhere to open a WINDOW on the workstation instead of streaming:
#   ./show.sh goto --gui          ./show.sh reach wb9 --gui
# Run it from a terminal on the workstation's desktop, so DISPLAY is set.
#
# Without --gui everything streams over WebRTC: connect the Isaac Sim WebRTC
# Streaming Client to this machine's address. The client has no port field (it
# assumes 49100/tcp + 47998/udp), so only one of these can stream at a time --
# hence `./show.sh stop`.
#
# WHICH POLICY, AND WHY IT MATTERS
#   wb9   drives AND reaches. Use it for anything involving movement.
#   wb15  reaches best and keeps the arm out of the lidar mast, but turns at
#         only ~30 % of the commanded rate, so it stalls short of a goal it has
#         to turn towards. Use it for standing reaching.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS=/workspace/policies

WB9_TASK=RobotLab-Isaac-WholeBody-WorldTarget-Unitree-B2W-Z1-v0
WB9_CKPT=${LOGS}/b2w_z1_wholebody_worldtarget/2026-09-23_13-25-50_wb9/model_7700.pt
WB15_TASK=RobotLab-Isaac-WholeBody-TidyFirm-Unitree-B2W-Z1-v0
WB15_CKPT=${LOGS}/b2w_z1_wholebody_tidy/2026-09-24_06-32-21_wb15/model_9500.pt
WB12_TASK=RobotLab-Isaac-WholeBody-Tidy-Unitree-B2W-Z1-v0
WB12_CKPT=${LOGS}/b2w_z1_wholebody_tidy/2026-09-24_00-58-38_wb12/model_6200.pt

pick() {   # $1 = policy name -> sets TASK and CKPT
    case "${1:-wb9}" in
        wb9)  TASK=$WB9_TASK;  CKPT=$WB9_CKPT  ;;
        wb15) TASK=$WB15_TASK; CKPT=$WB15_CKPT ;;
        wb12) TASK=$WB12_TASK; CKPT=$WB12_CKPT ;;
        *) echo "unknown policy '$1' (use wb9, wb12 or wb15)" >&2; exit 2 ;;
    esac
}

launch() {  # $1 = script, rest = its arguments
    local script="$1"; shift
    echo "policy : $(basename "$(dirname "${CKPT}")")/$(basename "${CKPT}")"
    echo "task   : ${TASK}"
    echo "view   : $([ "${GUI}" = 1 ] && echo "window on this workstation" || echo "WebRTC stream")"
    echo
    if [ "${GUI}" = 1 ]; then
        window "${script}" --task "${TASK}" --checkpoint "${CKPT}" "$@"
    else
        PLAY_SCRIPT="/workspace/project/${script}" CONTAINER_NAME=isaac-play \
            "${SCRIPT_DIR}/run_train.sh" --play --stream \
            --task "${TASK}" --checkpoint "${CKPT}" "$@"
    fi
}

# A local Isaac window instead of the stream -- the same launcher as live.sh.
# run_train.sh cannot do this: it always adds --headless.
window() {  # $1 = script, rest = its arguments
    local script="$1"; shift
    [ -n "${DISPLAY:-}" ] || { echo "DISPLAY is not set: run this from a terminal on the workstation's desktop." >&2; exit 1; }
    export ISAAC_IMAGE_DEFAULT=lab
    # shellcheck source=isaac_env.sh
    source "${SCRIPT_DIR}/../isaac/isaac_env.sh"
    # the container runs as uid 1234 and must be allowed onto the X server (see launch_gui.sh)
    command -v xhost >/dev/null 2>&1 && xhost +local: >/dev/null
    preflight
    ensure_dirs
    handle_existing_container isaac-play
    local tty=(-i); [ -t 0 ] && tty=(-it)
    local quoted=""; for a in "$@"; do quoted+=" $(printf '%q' "$a")"; done
    docker run --name isaac-play --entrypoint bash --rm "${tty[@]}" \
        --gpus "${GPU_SELECT}" "${COMMON_ENV[@]}" -e PYTHONUNBUFFERED=1 \
        -e "DISPLAY=${DISPLAY}" -v /tmp/.X11-unix:/tmp/.X11-unix \
        --network=host --ipc=host \
        "${WORKSPACE_MOUNTS[@]}" "${CACHE_MOUNTS[@]}" \
        -w /workspace/repos/robot_lab "${ISAAC_IMAGE}" \
        -c "/isaac-sim/python.sh /workspace/project/${script}${quoted} 2>&1"
}

# --gui may appear anywhere on the command line; take it out before parsing.
GUI=0
REST=()
for a in "$@"; do
    if [ "$a" = "--gui" ]; then GUI=1; else REST+=("$a"); fi
done
set -- ${REST[@]+"${REST[@]}"}

MODE="${1:-goto}"; shift || true

case "${MODE}" in
    goto)
        pick "${1:-wb9}"
        # A stale command would otherwise fire the moment goto starts.
        : > "${SCRIPT_DIR}/target.txt"
        echo "Type coordinates in this terminal, e.g.  2.5 0 0.3"
        echo "or from another terminal:  ./show.sh send 2.5 0 0.3"
        echo
        launch goto.py
        ;;
    reach)
        pick "${1:-wb15}"
        launch play_task.py --num_envs 4 --real-time
        ;;
    fetch)
        TGT="${1:-2.5,0,0.25}"; [ $# -gt 0 ] && shift
        pick "${1:-wb9}"
        launch demo_fetch.py --target "${TGT}" --real-time
        ;;
    square)
        pick "${1:-wb9}"
        launch waypoint.py --vmax 0.3 --waypoints "2,0; 2,2; 0,2; 0,0" --real-time
        ;;
    send)
        [ $# -ge 2 ] || { echo "usage: ./show.sh send X Y [Z]" >&2; exit 2; }
        echo "$*" > "${SCRIPT_DIR}/target.txt"
        echo "sent '$*' -> $(basename "${SCRIPT_DIR}")/target.txt"
        ;;
    stop)
        # window and stream both run as isaac-play, so one stop covers either
        docker stop isaac-play >/dev/null 2>&1 && echo "stopped" || echo "nothing was running"
        ;;
    *)
        sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
        exit 2
        ;;
esac
