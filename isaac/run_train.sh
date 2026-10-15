#!/bin/bash
# RL training in the isaac-lab:2.3.2 image (Isaac Lab 2.3.2 + robot_lab 2.3.2
# on top of Isaac Sim 5.1.0). Same GPU pin and mounts as the other launchers --
# see isaac_env.sh.
#
#   ./run_train.sh --task RobotLab-Isaac-Velocity-Flat-Unitree-B2W-v0 --num_envs 4096
#   ./run_train.sh --task RobotLab-Isaac-Velocity-Rough-Unitree-B2W-v0
#   ./run_train.sh --play --task ... --checkpoint /workspace/...   (inference)
#   ./run_train.sh --play --task ... --num_envs 16 --stream         (watch remotely)
#
# --headless is added automatically unless you pass it yourself. Everything is
# forwarded to robot_lab's script untouched.
#
# Training runs for hours, so this detaches by default: FOLLOW=1 to attach to
# the log instead. --stream always runs attached, since you are watching it.
#
# STREAMING (--stream)
# ====================
# Isaac Lab's AppLauncher takes --livestream {0,1,2}: 0 off, 1 "WebRTC public",
# 2 "WebRTC private". BOTH enable omni.services.livestream.nvcf; the difference
# is that 1 also sets --/app/livestream/publicEndpointAddress, and it reads that
# from the PUBLIC_IP environment variable (defaulting to 127.0.0.1). 2 sets no
# endpoint at all, so it is localhost-only and useless from your laptop.
#
# --stream therefore means: --livestream 1 plus PUBLIC_IP=${LIVESTREAM_ADDR}.
# Pass --livestream N yourself to override.
#
# Do not stream a training run: rendering thousands of environments to a video
# encoder destroys throughput. Use it with --play and a small --num_envs.
#
# Output lands in repos/robot_lab/logs/rsl_rl/<experiment>/<timestamp>/ (config paths.robot_lab)
# on the HOST, because the repo is bind-mounted and the script logs relative to
# its working directory. The container is uid 1234 and your files are uid 1005,
# which is why that logs/ directory is chmod a+rwX -- same reasoning as the
# cache directories in ensure_dirs().

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ISAAC_IMAGE_DEFAULT=lab   # config.yaml images.isaac_lab
# shellcheck source=isaac_env.sh
source "${SCRIPT_DIR}/isaac_env.sh"

CONTAINER_NAME="${CONTAINER_NAME:-isaac-lab-train}"

MODE="train"
ARGS=()
HAS_HEADLESS=0
STREAM=0
HAS_LIVESTREAM=0
for a in "$@"; do
    case "$a" in
        --play)       MODE="play" ;;
        --headless)   HAS_HEADLESS=1; ARGS+=("$a") ;;
        --stream)     STREAM=1 ;;   # convenience flag, not passed through
        --livestream) STREAM=1; HAS_LIVESTREAM=1; ARGS+=("$a") ;;
        --livestream=*) STREAM=1; HAS_LIVESTREAM=1; ARGS+=("$a") ;;
        *)            ARGS+=("$a") ;;
    esac
done
[ "${HAS_HEADLESS}" -eq 0 ] && ARGS+=(--headless)
# --stream on its own means livestream mode 1; an explicit --livestream wins.
if [ "${STREAM}" -eq 1 ] && [ "${HAS_LIVESTREAM}" -eq 0 ]; then
    ARGS+=(--livestream 1)
fi

if [ "${MODE}" = "play" ]; then
    # robot_lab's play.py assumes a locomotion task (it edits terrain and velocity
    # curricula unconditionally). For anything else:
    #   PLAY_SCRIPT=/workspace/project/play_task.py ./run_train.sh --play ...
    SCRIPT="${PLAY_SCRIPT:-scripts/reinforcement_learning/rsl_rl/play.py}"
else
    SCRIPT="scripts/reinforcement_learning/rsl_rl/train.py"
fi

LOGS_DIR="${ROBOT_LAB_DIR}/logs"

echo "Isaac Lab training (${MODE})..."
print_banner
echo "  Script    : ${SCRIPT}"
echo "  Args      : ${ARGS[*]}"
echo "  Logs      : ${LOGS_DIR}/rsl_rl/<experiment>/<timestamp>/"
[ "${STREAM}" -eq 1 ] && echo "  Livestream: ${LIVESTREAM_ADDR}:${LIVESTREAM_PORT} (WebRTC)"
echo

if [ "${STREAM}" -eq 1 ] && [ "${MODE}" = "train" ]; then
    echo "WARNING: streaming a TRAINING run. Rendering every environment to the" >&2
    echo "         encoder will cripple throughput. --stream is meant for --play." >&2
    echo >&2
fi

# Same check as launch_headless.sh: an address that is not on this host means
# the client can never connect, and the failure looks like a silent hang.
if [ "${STREAM}" -eq 1 ] && ! ip -4 -br addr | grep -qw "${LIVESTREAM_ADDR}"; then
    echo "WARNING: ${LIVESTREAM_ADDR} is not an address on this machine." >&2
    ip -4 -br addr | awk '{printf "           %-14s %s\n", $1, $3}' >&2
    echo "         Override with: LIVESTREAM_ADDR=<addr> $0 ..." >&2
    echo >&2
fi

preflight
ensure_dirs
handle_existing_container "${CONTAINER_NAME}"

if ! docker image inspect "${ISAAC_IMAGE}" >/dev/null 2>&1; then
    echo "ERROR: image '${ISAAC_IMAGE}' not found." >&2
    echo "       Build it:  ./setup.sh   (isaac/docker/Dockerfile.isaaclab)" >&2
    exit 1
fi

# The container writes as uid 1234 while the host owns these as uid 1005, so
# both directories need to be group/other writable or the run dies on startup.
#
# logs/    - RSL-RL checkpoints and tensorboard events.
# outputs/ - Hydra. Isaac Lab drives its configs through Hydra, which creates
#            outputs/<date>/<time> in the WORKING DIRECTORY before the task even
#            starts. Missing this kills the run ~90s in, after Isaac Sim has
#            fully booted, with "PermissionError: [Errno 13] ... 'outputs'"
#            buried under two chained FileNotFoundErrors.
OUTPUTS_DIR="${ROBOT_LAB_DIR}/outputs"
mkdir -p "${LOGS_DIR}" "${OUTPUTS_DIR}"
chmod -R a+rwX "${LOGS_DIR}" "${OUTPUTS_DIR}" 2>/dev/null || true

# Training is long. Detach unless explicitly following, so it survives the
# shell going away -- same reasoning as set_run_mode()'s non-TTY branch.
# -it needs a real terminal. Streaming and FOLLOW both want to be attached, but
# `docker run -it` fails outright with "the input device is not a TTY" when this
# is launched from a script or an agent shell -- the same reasoning as
# set_run_mode() in isaac_env.sh. Detach instead when there is no TTY; the
# stream does not need one, only a running container.
if { [ "${FOLLOW:-0}" = "1" ] || [ "${STREAM}" -eq 1 ]; } && [ -t 0 ]; then
    RUN_FLAGS=(-it)
else
    RUN_FLAGS=(-d)
fi

# AppLauncher reads the advertised endpoint from PUBLIC_IP; without it the
# stream advertises 127.0.0.1 and the client on your laptop finds nothing.
STREAM_ENV=()
[ "${STREAM}" -eq 1 ] && STREAM_ENV=(-e "PUBLIC_IP=${LIVESTREAM_ADDR}")

docker run \
    --name "${CONTAINER_NAME}" \
    --entrypoint bash \
    "${RUN_FLAGS[@]}" \
    --gpus "${GPU_SELECT}" \
    -e "ACCEPT_EULA=Y" \
    -e "PRIVACY_CONSENT=Y" \
    -e "OMNI_KIT_ACCEPT_EULA=YES" \
    -e "PYTHONUNBUFFERED=1" \
    "${STREAM_ENV[@]}" \
    --network=host \
    --ipc=host \
    "${WORKSPACE_MOUNTS[@]}" \
    "${CACHE_MOUNTS[@]}" \
    -w /workspace/repos/robot_lab \
    "${ISAAC_IMAGE}" \
    -c "/isaac-sim/python.sh ${SCRIPT} ${ARGS[*]} 2>&1"

if [ "${STREAM}" -eq 1 ]; then
    echo
    echo "Connect the Isaac Sim WebRTC Streaming Client to ${LIVESTREAM_ADDR}"
    echo "(the client has no port field -- it assumes 49100/tcp + 47998/udp, so"
    echo " nothing else may be streaming on those ports at the same time)."
elif [ "${FOLLOW:-0}" != "1" ]; then
    echo
    echo "Started detached."
    echo "  Follow    : docker logs -f ${CONTAINER_NAME}"
    echo "  Stop      : docker stop ${CONTAINER_NAME}"
    echo "  Tensorboard: tensorboard --logdir ${LOGS_DIR}"
fi
