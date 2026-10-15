#!/bin/bash
# Shared configuration for the Isaac Sim launch scripts.
# Sourced by the launchers (isaac/, project/, evaluation/) -- not meant to be run directly.
#
# Every machine-specific value comes from config.yaml in the repository root (see config.example.yaml),
# read by isaac/config.py. An environment variable of the same name still overrides it, e.g.
#   ISAAC_GPU_UUID=GPU-... ./project/live.sh fetch
#
# Inside every container the layout is the same:
#   /workspace                     this repository (isaac/, project/, policies/, robots_usd/, renders/, ...)
#   /workspace/repos/IsaacLab      Isaac Lab 2.3.2      (paths.isaaclab)  -- the isaac-lab image's editable
#   /workspace/repos/robot_lab     robot_lab 2.3.2      (paths.robot_lab)    installs point exactly here
#   /workspace/scenes/nurec        NuRec scene assets   (paths.nurec)

_ISAAC_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
eval "$(python3 "${_ISAAC_ENV_DIR}/config.py" --shell)" || { echo "ERROR: could not read config.yaml" >&2; exit 1; }
ISAAC_DIR="${REPO_ROOT}/isaac"

# Isaac Sim cache (shaders, assets). isaac_env_6.sh points it at the 6.0.1 cache before sourcing this file.
CACHE_ROOT="${CACHE_ROOT:-${CACHE_ROOT_5}}"
SCENES_ROOT="${NUREC_ROOT}"   # (kept for older notes)

# ---------------------------------------------------------------------------
# GPU selection  -- IMPORTANT, read before changing
# ---------------------------------------------------------------------------
# Example, the lab workstation this was built on, which has two GPUs:
#
#   00000000:01:00.0  RTX PRO 6000 Blackwell, 96 GB  <-- RESERVED FOR SLURM
#   00000000:03:00.0  RTX 6000 Ada Generation, 48 GB <-- ours
#
# /etc/profile.d/gpu-firvads_ruiyang.sh confines this account to the Ada card
# via CUDA_VISIBLE_DEVICES. That variable does NOT reach containers: Docker does
# not inherit the host environment, so `--gpus all` would hand Isaac Sim the
# 96 GB card that Slurm allocates to other people's jobs.
#
# Hence the explicit pin (gpu.uuid in config.yaml). Use `all` only on a machine whose GPUs are all yours.
# Selected by UUID rather than index so it survives PCI reordering.
# The UUID comes from config.yaml (gpu.uuid).
if [ "${ISAAC_GPU_UUID}" = "all" ]; then GPU_SELECT="all"; else GPU_SELECT="device=${ISAAC_GPU_UUID}"; fi

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------
# Launchers that need Isaac Lab set ISAAC_IMAGE_DEFAULT=lab before sourcing this file (config images.isaac_lab).
if [ "${ISAAC_IMAGE_DEFAULT:-}" = "lab" ]; then
    ISAAC_IMAGE="${ISAAC_IMAGE:-${IMAGE_ISAAC_LAB}}"
else
    ISAAC_IMAGE="${ISAAC_IMAGE:-${IMAGE_ISAAC_SIM}}"
fi

# ---------------------------------------------------------------------------
# Livestream endpoint (headless mode only)
# ---------------------------------------------------------------------------
# config.yaml livestream.address. Must be an address the *viewing* client can reach. On the lab workstation:
#
#   10.147.19.106    ZeroTier ztc3q54tec   -- DEFAULT. Verified working from the
#                                             laptop (10.147.19.133) on this network.
#   10.147.20.106    ZeroTier zt6ovyltm4
#   10.162.45.149    enp12s0, TUM LAN      -- only for clients on the same LAN
#
# Pick the address on the network you share with the viewing machine. A machine can
# be a member of several ZeroTier networks at once, so check both ends:
#   sudo zerotier-cli listnetworks

# ---------------------------------------------------------------------------
# Shared docker arguments
# ---------------------------------------------------------------------------
# Cache subdirectories, created on demand by ensure_dirs().
#
# Paths follow the Isaac Sim *5.1.0* container layout, where the container's
# home is /isaac-sim. (Isaac Sim 4.x used /root/... -- do not copy mount paths
# from older tutorials or from a 4.x setup; they will silently not persist.)
# Source: docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_container.html
CACHE_MOUNTS=(
    -v "${CACHE_ROOT}/cache/main:/isaac-sim/.cache:rw"
    -v "${CACHE_ROOT}/cache/computecache:/isaac-sim/.nv/ComputeCache:rw"
    -v "${CACHE_ROOT}/logs:/isaac-sim/.nvidia-omniverse/logs:rw"
    -v "${CACHE_ROOT}/config:/isaac-sim/.nvidia-omniverse/config:rw"
    -v "${CACHE_ROOT}/data:/isaac-sim/.local/share/ov/data:rw"
    -v "${CACHE_ROOT}/pkg:/isaac-sim/.local/share/ov/pkg:rw"
)

WORKSPACE_MOUNTS=(
    -v "${REPO_ROOT}:/workspace"
    -v "${ISAACLAB_DIR}:/workspace/repos/IsaacLab"
    -v "${ROBOT_LAB_DIR}:/workspace/repos/robot_lab"
)
[ -d "${NUREC_ROOT}" ] && WORKSPACE_MOUNTS+=(-v "${NUREC_ROOT}:/workspace/scenes/nurec")

COMMON_ENV=(
    -e "ACCEPT_EULA=Y"
    -e "PRIVACY_CONSENT=Y"
    -e "PYTHONDONTWRITEBYTECODE=1"   # no root-less __pycache__ (uid 1234) inside the mounted repository
    -e "ROS_DISTRO=jazzy"
    -e "RMW_IMPLEMENTATION=rmw_fastrtps_cpp"
    -e "FASTRTPS_DEFAULT_PROFILES_FILE=/workspace/isaac/fastdds.xml"
    -e "LD_LIBRARY_PATH=/isaac-sim/exts/isaacsim.ros2.bridge/jazzy/lib"
)
# ETH's base controller weights (project/eth_loco.py): a fine-tuned checkpoint instead of the ONNX
[ -n "${ETH_POLICY:-}" ] && COMMON_ENV+=(-e "ETH_POLICY=${ETH_POLICY}")
[ -n "${FETCH_TRACE:-}" ] && COMMON_ENV+=(-e "FETCH_TRACE=${FETCH_TRACE}")
# the scene file the fetch tasks build (scenes/*.yaml, container path; run.py and live.sh set it)
[ -n "${FETCH_SCENE:-}" ] && COMMON_ENV+=(-e "FETCH_SCENE=${FETCH_SCENE}")
# the fetch robot's USD (fetch_env_cfg.B2W_Z1_GRIPPER_USD), e.g. the RealSense D405 build, container path
[ -n "${B2W_Z1_GRIPPER_USD:-}" ] && COMMON_ENV+=(-e "B2W_Z1_GRIPPER_USD=${B2W_Z1_GRIPPER_USD}")
# ROS 2 domain shared with the Nav2 container (nav2/run_nav2.sh): 37, not the default 0, so other
# users' ROS 2 nodes on this machine stay invisible
COMMON_ENV+=(-e "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ensure_dirs() {
    if [ ! -d "${ROBOT_LAB_DIR}/source/robot_lab" ] || [ ! -d "${ISAACLAB_DIR}/source" ]; then
        echo "ERROR: Isaac Lab / robot_lab not found at ${ISAACLAB_DIR} / ${ROBOT_LAB_DIR}." >&2
        echo "       Run ./setup.sh first (it clones them at the pinned versions)." >&2
        exit 1
    fi
    if [ ! -f "${REPO_ROOT}/robots_usd/b2w_z1_gripper.usd" ]; then
        echo "ERROR: the robot assets are missing (robots_usd/b2w_z1_gripper.usd). Run ./setup.sh first." >&2
        exit 1
    fi
    mkdir -p \
        "${REPO_ROOT}/renders/fetch" "${REPO_ROOT}/datasets/raw" "${REPO_ROOT}/datasets/lerobot" \
        "${REPO_ROOT}/scenes/nurec" \
        "${ROBOT_LAB_DIR}/logs" "${ROBOT_LAB_DIR}/outputs" \
        "${CACHE_ROOT}/cache/main/ov" \
        "${CACHE_ROOT}/cache/main/warp" \
        "${CACHE_ROOT}/cache/computecache" \
        "${CACHE_ROOT}/config" \
        "${CACHE_ROOT}/data/documents" \
        "${CACHE_ROOT}/data/Kit" \
        "${CACHE_ROOT}/logs" \
        "${CACHE_ROOT}/pkg"

    # The container runs as uid 1234 (user 'isaac-sim'), while these directories
    # are owned by the host user. Without this the container cannot write its
    # cache and every start recompiles shaders -- silently, with no error.
    #
    # NVIDIA's docs use `sudo chown -R 1234:1234`. We use group/other write bits
    # instead so this needs no root. That is safe here because $HOME itself is
    # mode 0750, so nothing outside this account can traverse into these paths.
    chmod -R a+rwX "${CACHE_ROOT}" 2>/dev/null || true

    # Same problem, same rootless fix, for our own scene reconstructions: the
    # container must be able to write them back out. Applied to nurec/mine only
    # -- nurec/nvidia holds NVIDIA's reference assets, which the container has no
    # business modifying, and 775/664 already lets uid 1234 read them.
    [ -d "${NUREC_ROOT}/mine" ] && chmod -R a+rwX "${NUREC_ROOT}/mine" 2>/dev/null || true

    # The repository's output folders and the live-demo command files, for the same reason (uid 1234).
    # (folders the container creates itself are its own, 755: the host user could not delete recordings in them)
    chmod a+rwX "${REPO_ROOT}/renders" "${REPO_ROOT}/renders/fetch" "${REPO_ROOT}/datasets" "${REPO_ROOT}/datasets/raw" \
        "${REPO_ROOT}/datasets/lerobot" "${REPO_ROOT}/project" "${REPO_ROOT}/nav2/maps" 2>/dev/null || true
    chmod -R a+rwX "${ROBOT_LAB_DIR}/logs" "${ROBOT_LAB_DIR}/outputs" 2>/dev/null || true

    if [ ! -f "${ISAAC_DIR}/fastdds.xml" ]; then
        echo "ERROR: ${ISAAC_DIR}/fastdds.xml is missing." >&2
        echo "       FASTRTPS_DEFAULT_PROFILES_FILE points at it inside the container;" >&2
        echo "       without it ROS 2 topics will not cross the container boundary." >&2
        exit 1
    fi
}

# The original scripts used a fixed --name, which fails with "name already in
# use" as soon as a previous container is left behind. Clear out a stopped one;
# refuse to clobber a running one.
handle_existing_container() {
    local name="$1"
    if docker ps --format '{{.Names}}' | grep -qx "${name}"; then
        echo "ERROR: a container named '${name}' is already running." >&2
        echo "       Attach to it:  docker exec -it ${name} bash" >&2
        echo "       Or stop it:    docker stop ${name}" >&2
        exit 1
    fi
    if docker ps -a --format '{{.Names}}' | grep -qx "${name}"; then
        echo "Removing stopped container '${name}' from a previous run..."
        docker rm "${name}" >/dev/null
    fi
}

preflight() {
    if ! docker info >/dev/null 2>&1; then
        echo "ERROR: cannot talk to the Docker daemon." >&2
        echo "       If you were just added to the 'docker' group, log out and back" >&2
        echo "       in (or run 'newgrp docker') for it to take effect." >&2
        exit 1
    fi
    # Intermittent false alarms here although daemon.json registers nvidia: no
    # `docker info | grep -q` under pipefail (SIGPIPE), and retried in case the daemon
    # answers empty under load.
    local runtimes="" try
    for try in 1 2 3; do
        runtimes=" $(docker info --format '{{range $k, $v := .Runtimes}}{{$k}} {{end}}' 2>/dev/null || true) "
        [[ "${runtimes}" == *" nvidia "* ]] && break
        sleep 2
    done
    if [[ "${runtimes}" != *" nvidia "* ]]; then
        echo "ERROR: the 'nvidia' runtime is not registered with Docker." >&2
        echo "       Run: sudo nvidia-ctk runtime configure --runtime=docker" >&2
        echo "            sudo systemctl restart docker" >&2
        exit 1
    fi
}

# Decides how to attach the container, based on whether stdin is a terminal.
# Sets RUN_MODE_FLAGS and DETACHED.
#
# Interactive (normal use): -it. Isaac Sim runs in the foreground and Ctrl-C or
# closing the app stops it.
#
# Non-interactive (script, cron, CI, agent shell): -d, detached. `-it` fails
# outright with "the input device is not a TTY", and an attached container is
# tied to the lifetime of the launching shell -- when that shell goes away the
# simulator goes with it. Detaching removes both problems.
set_run_mode() {
    if [ -t 0 ]; then
        RUN_MODE_FLAGS=(-it)
        DETACHED=0
    else
        RUN_MODE_FLAGS=(-d)
        DETACHED=1
    fi
}

print_banner() {
    echo "  Repository: ${REPO_ROOT}"
    echo "  Cache     : ${CACHE_ROOT}"
    echo "  Scenes    : ${NUREC_ROOT}"
    echo "  Image     : ${ISAAC_IMAGE}"
    echo "  GPU       : ${ISAAC_GPU_UUID}"
}
