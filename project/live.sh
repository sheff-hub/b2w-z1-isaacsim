#!/bin/bash
# Watch the Phase 5 robot in an Isaac Sim WINDOW on this workstation (STREAM=1: WebRTC livestream, goto modes).
# Run it from a terminal on the workstation's desktop, so DISPLAY is set.
#
#   ./live.sh vln                  language navigation: type instructions, watch DualVLN drive
#   ./live.sh say "walk to the table"   send an instruction from ANOTHER terminal
#   ./live.sh pick [N]             the scripted sideways pick + delivery, N episodes (default 3)
#   ./live.sh fetch                TYPE TASKS: "go to the wooden table and pick the box",
#                                  "... and bring it back"; DualVLN walks, then parking + pick (+ delivery)
#   ./live.sh fetchdemo [N]        the same, automatic: N episodes from 3 start poses (default 3)
#   ./live.sh goto_mine            Phase 4 one-command 6D reach, OUR phase rule (shoulder workspace sphere):
#                                  type "x y z" or "x y z roll pitch yaw"; one policy drives AND reaches
#   ./live.sh goto_paper           the same with the PAPER's phase rule (Jiang et al. 2024), to compare
#   ./live.sh eth [keys|tests] [eth|ours]
#                                  locomotion A/B (eth_compare.py): drive with the keyboard (W/S A/D Q/E,
#                                  click the window first) or watch the test sequence; ETH's controller
#                                  by default, "ours" for model_2600
#   ./live.sh stop                 close the simulator
#   ./live.sh stop all             ... and stop the DualVLN server (frees ~21 GB of GPU memory)
#   QUANT=8bit ./live.sh vln       the 8-bit model instead of full precision (6 GB less)
#   LOCO=ours ./live.sh fetch      our old base controller (model_2600) instead of ETH's
#   NAV2=1 ./live.sh fetch         Nav2 drives to the pick point instead of the scripted route
#                                  (fetch / fetchdemo; starts the Nav2 container, nav2/README.md)
#   NAV2=1 VLN_FOLLOWER=follow ...  Nav2 also drives DualVLN's walk: follow = FollowPath on DualVLN's path,
#                                  goal = NavigateToPose to each path's end (default pp = our pure pursuit)
#   SCENE=3 NAV2=1 ./live.sh fetch three tables (SCENE=<name>: any scene file scenes/<name>.yaml), on each a red, yellow and green box:
#                                  type "go to the green table and pick the red box"; TARGET=green/red for
#                                  fetchdemo (default auto: random per episode)
#   STREAM=1 ./live.sh goto_mine   headless + WebRTC livestream instead of a window (goto_mine/goto_paper
#                                  only); connect the Isaac Sim WebRTC Streaming Client to LIVESTREAM_ADDR
#
# In `vln` and `fetch`, type in the terminal:
#   any sentence   a new instruction (interrupts the current walk)
#   stop           halt and wait
#   home           put the robot (and in `fetch` the box) back at the start
#   q              quit
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ISAAC_IMAGE_DEFAULT=lab   # config.yaml images.isaac_lab
# shellcheck source=isaac_env.sh
source "${SCRIPT_DIR}/../isaac/isaac_env.sh"
NAME="${LIVE_NAME:-${LIVE_CONTAINER}}"
QUANT="${QUANT:-${DUALVLN_QUANT}}"   # DualVLN precision (config dualvln.quant): none (bf16) or 8bit
LOCO="${LOCO:-eth}"             # base controller: eth (ETH's SRU policy, default) or ours (model_2600)
NAV2="${NAV2:-0}"               # 1: Nav2 plans and drives to the entry point E (fetch / fetchdemo)
SCENE="${SCENE:-1}"             # a scene file scenes/<name>.yaml (or a path); 1 = warehouse_one_table, 3 = warehouse_three_tables
TARGET="${TARGET:-auto}"        # multi-table scenes: which box, table/box (green/red), or auto (random per episode)
VLN_FOLLOWER="${VLN_FOLLOWER:-pp}"  # who drives DualVLN's trajectories: pp (our pure pursuit), follow (Nav2 FollowPath), goal (Nav2 NavigateToPose); follow/goal need NAV2=1
# the scene file decides the Isaac Lab task, the Nav2 map and the start poses (isaac/scene_info.py)
eval "$(python3 "${SCRIPT_DIR}/../isaac/scene_info.py" "${SCENE}")"
COMMON_ENV+=(-e "FETCH_SCENE=${SCENE_FILE_CONTAINER}")
START="${START:-${SCENE_START}}"   # x,y,yaw(deg); for the warehouse: 3.5 m from the table, which is in view
TASK_ARGS=(--task "${SCENE_TASK}")
[ "${SCENE_LAYOUT}" = "multi_table" ] && TASK_ARGS+=(--target "${TARGET}")
NAV2_MAP="${SCENE_NAV2_MAP}"
NAV2_ARGS=()
if [ "${NAV2}" = "1" ]; then
    NAV2_ARGS=(--nav2 --vln_follower "${VLN_FOLLOWER}")
    # (re)start Nav2 unless it already runs with this scene's map
    if ! docker ps --format '{{.Names}}' | grep -qx "${NAV2_CONTAINER}" || ! head -1 "${SCRIPT_DIR}/../nav2/b2w_nav2_params.yaml" 2>/dev/null | grep -q "map ${NAV2_MAP})"; then
        NAV2_MAP="${NAV2_MAP}" "${SCRIPT_DIR}/../nav2/run_nav2.sh"
    fi
fi
STREAM="${STREAM:-0}"           # 1 = headless + WebRTC livestream (goto modes), no DISPLAY needed

window() {  # $1 = script, rest = its arguments
    local script="$1"; shift
    local view=()
    if [ "${STREAM}" = "1" ]; then
        # AppLauncher: --livestream 1 forces headless and advertises PUBLIC_IP (run_train.sh)
        set -- "$@" --livestream 1
        view=(-e "PUBLIC_IP=${LIVESTREAM_ADDR}")
        echo "livestream: connect the WebRTC Streaming Client to ${LIVESTREAM_ADDR} (port ${LIVESTREAM_PORT})"
    else
        [ -n "${DISPLAY:-}" ] || { echo "DISPLAY is not set: run this from a terminal on the workstation's desktop (or use STREAM=1)." >&2; exit 1; }
        # the container runs as uid 1234 and must be allowed onto the X server (see launch_gui.sh)
        command -v xhost >/dev/null 2>&1 && xhost +local: >/dev/null
        view=(-e "DISPLAY=${DISPLAY}" -v /tmp/.X11-unix:/tmp/.X11-unix)
    fi
    preflight
    ensure_dirs
    handle_existing_container "${NAME}"
    local tty=(-i); [ -t 0 ] && tty=(-it)
    local quoted=""; for a in "$@"; do quoted+=" $(printf '%q' "$a")"; done
    docker run --name "${NAME}" --entrypoint bash --rm "${tty[@]}" \
        --gpus "${GPU_SELECT}" "${COMMON_ENV[@]}" -e PYTHONUNBUFFERED=1 \
        "${view[@]}" \
        --network=host --ipc=host \
        "${WORKSPACE_MOUNTS[@]}" "${CACHE_MOUNTS[@]}" \
        -w /workspace/repos/robot_lab "${ISAAC_IMAGE}" \
        -c "/isaac-sim/python.sh /workspace/project/${script}${quoted} 2>&1"
}

server_up() { curl -s -m 2 "http://127.0.0.1:${DUALVLN_PORT}/health" >/dev/null 2>&1; }

ensure_server() {
    if ! server_up; then
        echo "starting the DualVLN server (${QUANT}, ~16 GB full / ~10 GB 8bit on the Ada GPU) ..."
        mkdir -p "${SCRIPT_DIR}/../vln/out"
        nohup "${SCRIPT_DIR}/../vln/run_dualvln_server.sh" --quant "${QUANT}" > "${SCRIPT_DIR}/../vln/out/server.log" 2>&1 &
        for _ in $(seq 1 60); do server_up && break; sleep 3; done
        server_up || { echo "the server did not come up; see vln/out/server.log" >&2; exit 1; }
    fi
}

MODE="${1:-}"; shift || true
case "${MODE}" in goto_mine|goto_paper|stop|say|"") ;; *)
    [ "${STREAM}" = "1" ] && { echo "STREAM=1 is only wired for goto_mine/goto_paper." >&2; exit 2; } ;; esac
case "${MODE}" in
    vln)
        ensure_server
        echo "DualVLN server is up. The Isaac window takes ~1 min to open."
        : > "${SCRIPT_DIR}/instruction.txt"
        window vln_drive.py --device cpu --gui --interactive --real-time --loco "${LOCO}" --start "${START}" --max_time 90 --tag live \
            --server "http://127.0.0.1:${DUALVLN_PORT}"
        ;;
    fetch)
        ensure_server
        echo "DualVLN server is up. The Isaac window takes ~1 min to open."
        : > "${SCRIPT_DIR}/instruction.txt"
        window fetch_box.py --device cpu --gui --real-time --interactive --loco "${LOCO}" --starts "${START}" "${NAV2_ARGS[@]}" "${TASK_ARGS[@]}" \
            --vln_server "http://127.0.0.1:${DUALVLN_PORT}"
        ;;
    fetchdemo)
        ensure_server
        echo "DualVLN server is up. The Isaac window takes ~1 min to open."
        window fetch_box.py --device cpu --gui --real-time --deliver --loco "${LOCO}" --episodes "${1:-3}" \
            --vln "Walk to the wooden table and stop in front of it." \
            --starts "${SCENE_DEMO_STARTS}" --out /workspace/renders/fetch_vln/live_results.json \
            "${NAV2_ARGS[@]}" "${TASK_ARGS[@]}"
        ;;
    goto_mine|goto_paper)
        # One RFM policy does everything: no velocity command, no state machine.
        # rfm_goto.py, NOT goto.py -- goto.py is a drive-then-reach script for the
        # velocity-command policy wb9, and its "parked" hand target rides along
        # with the body, which a world-target policy chases (3.5x the commanded
        # speed, measured).
        # rfm19: the arm is commanded INCREMENTALLY (each step moves the joint
        # target a little), which removed the shaking hand -- on target the hand
        # holds within ~0.1 mm instead of wobbling ~15 mm (PHASE4_RFM.md, 6h).
        if [ "${MODE}" = "goto_mine" ]; then
            PHASE=Sphere; CKPT=2026-10-08_19-58-51_rfm19_delta_sphere/model_37600.pt
        else
            PHASE=Paper; CKPT=2026-10-08_20-28-32_rfm19_delta_paper/model_36700.pt
        fi
        : > "${SCRIPT_DIR}/target.txt"
        echo "${MODE}: type in this terminal once the window is up, e.g.  2 1.5 1.0   or   3 0 0.5 0 0.6 0"
        echo "         stop = hold still, home = back to the origin, q = quit"
        window rfm_goto.py --task "RobotLab-Isaac-RFM-Delta-${PHASE}-Unitree-B2W-Z1-v0" \
            --checkpoint "/workspace/policies/b2w_z1_rfm/${CKPT}"
        ;;
    eth)
        MODE_ETH="${1:-keys}"; CTRL="${2:-eth}"
        if [ "${MODE_ETH}" = "tests" ]; then
            window eth_compare.py --device cpu --gui --real-time --controller "${CTRL}"
        else
            window eth_compare.py --device cpu --gui --real-time --keyboard --controller "${CTRL}"
        fi
        ;;
    say)
        [ $# -ge 1 ] || { echo "usage: ./live.sh say \"walk to the wooden table\"" >&2; exit 2; }
        echo "$*" > "${SCRIPT_DIR}/instruction.txt"
        echo "sent: $*"
        ;;
    pick)
        window fetch_box.py --device cpu --gui --real-time --deliver --loco "${LOCO}" --episodes "${1:-3}" \
            --out /workspace/renders/fetch/live_results.json "${TASK_ARGS[@]}"
        ;;
    stop)
        docker stop "${NAME}" >/dev/null 2>&1 && echo "simulator stopped" || echo "no simulator running"
        if [ "${1:-}" = "all" ]; then
            docker stop "${DUALVLN_CONTAINER}" >/dev/null 2>&1 && echo "DualVLN server stopped" || echo "no server running"
        fi
        ;;
    *)
        sed -n '2,27p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
        exit 2
        ;;
esac
