#!/bin/bash
# Isaac Sim 6.0.1 -- configuration overrides layered on top of isaac_env.sh.
# Sourced by launch_gui_6.sh and launch_headless_6.sh; not meant to be run directly.
#
# Everything identical between 5.1.0 and 6.0.1 -- the GPU pin, the workspace
# mounts, the scene root, the helper functions -- stays in isaac_env.sh so there
# is exactly one copy of it. This file changes only what 6.0.1 needs:
#
#   image        nvcr.io/nvidia/isaac-sim:6.0.1
#   cache root   ~/docker/isaac-sim-6.0    (separate -- see below)
#   hub cache    ~/.cache/ov-6.0/hub       (new mount in the 6.0 container docs)
#   livestream   new settings keys; 5.1.0's --/app/livestream/* no longer exist
#
# The cache MUST be separate. Kit's shader cache, ComputeCache, extension
# registry and config are all version-specific, and sharing ~/docker/isaac-sim
# between the two versions is the single most likely way to break the working
# 5.1.0 setup. Nothing here writes to any 5.1.0 path.
#
# Container home is still /isaac-sim in 6.0.1, so the mount layout inherited
# from isaac_env.sh is still correct. Verified against:
#   docs.isaacsim.omniverse.nvidia.com/6.0.1/installation/install_container.html

# ---------------------------------------------------------------------------
# Overrides that must be set BEFORE sourcing isaac_env.sh
# ---------------------------------------------------------------------------
# isaac_env.sh builds ISAAC_IMAGE and CACHE_MOUNTS from these using
# ${VAR:-default}, so assigning them first is the whole override mechanism.
eval "$(python3 "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.py" --shell)"
ISAAC_IMAGE="${ISAAC_IMAGE:-${IMAGE_ISAAC_SIM_6}}"
CACHE_ROOT="${CACHE_ROOT:-${CACHE_ROOT_6}}"

SCRIPT_DIR_6="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=isaac_env.sh
source "${SCRIPT_DIR_6}/isaac_env.sh"

# ---------------------------------------------------------------------------
# Additions on top of what isaac_env.sh built
# ---------------------------------------------------------------------------
# The 6.0 container docs mount the Omniverse hub cache separately; 5.1.0 did not.
HUB_CACHE="${HUB_CACHE:-$HOME/.cache/ov-6.0/hub}"
CACHE_MOUNTS+=(-v "${HUB_CACHE}:/var/cache/hub:rw")

# 6.0 RENAMED this extension: isaacsim.ros2.bridge -> isaacsim.ros2.core.
# Using the 5.1.0 path here points LD_LIBRARY_PATH at a directory that does not
# exist in the 6.0.1 image, and ROS 2 topics then never appear, with no error.
#
# The image ships /isaac-sim/setup_ros_env.sh, which would set this correctly on
# its own -- but only when ROS_DISTRO is UNSET (it guards the whole block on
# `if [ -z "$ROS_DISTRO" ]`). COMMON_ENV below sets ROS_DISTRO explicitly, so
# that auto-setup is skipped and we must supply the right path ourselves.
# Verified present in nvcr.io/nvidia/isaac-sim:6.0.1.
ROS_BRIDGE_LIB="${ROS_BRIDGE_LIB:-/isaac-sim/exts/isaacsim.ros2.core/jazzy/lib}"

# Rebuilt rather than inherited so that overriding ROS_BRIDGE_LIB actually takes
# effect -- isaac_env.sh baked the 5.1.0 path into COMMON_ENV literally.
COMMON_ENV=(
    -e "ACCEPT_EULA=Y"
    -e "PRIVACY_CONSENT=Y"
    -e "PYTHONDONTWRITEBYTECODE=1"
    -e "ROS_DISTRO=jazzy"
    -e "RMW_IMPLEMENTATION=rmw_fastrtps_cpp"
    -e "FASTRTPS_DEFAULT_PROFILES_FILE=/workspace/isaac/fastdds.xml"
    -e "LD_LIBRARY_PATH=${ROS_BRIDGE_LIB}"
)

# The 6.0.1 container docs run the container as uid 1234 explicitly. That is the
# image's own 'isaac-sim' user, the same uid 5.1.0 defaults to, so the existing
# a+rwX cache permissions from ensure_dirs() already fit.
USER_FLAGS=(-u 1234:1234)

# ---------------------------------------------------------------------------
# Livestream ports
# ---------------------------------------------------------------------------
# 5.1.0 took a single --/app/livestream/port (default 49100). 6.0 splits this
# into a signalling port and a media port under a different settings prefix, so
# there are two values to move, not one:
#
#   49100  TCP  WebRTC signalling
#   47998  UDP  WebRTC media          <-- UDP, not TCP. See check_ports_free().
#
# These MUST stay at the documented defaults. An earlier version of this file
# offset them to 49200/48098 so 5.1.0 and 6.0.1 could stream side by side, which
# binds fine but is unreachable in practice: the Isaac Sim WebRTC Streaming
# Client takes a server IP and nothing else -- there is no port field -- so it
# always dials 49100/47998. Offsetting the server ports just makes Connect hang
# with no error at either end.
#
# The override is kept for the side-by-side case, but it is only usable with a
# client that can be pointed at a custom port, which the stock one cannot:
#     LIVESTREAM_SIGNAL_PORT=49200 LIVESTREAM_STREAM_PORT=48098 ./launch_headless_6.sh
# Running 5.1.0 and 6.0.1 livestreams at once therefore is not practical; run
# one at a time and let each have the standard ports.
# Source: docs.isaacsim.omniverse.nvidia.com/6.0.1/installation/manual_livestream_clients.html
LIVESTREAM_SIGNAL_PORT="${LIVESTREAM_SIGNAL_PORT:-49100}"
LIVESTREAM_STREAM_PORT="${LIVESTREAM_STREAM_PORT:-47998}"

# ---------------------------------------------------------------------------
# Helpers specific to the 6.0.1 setup
# ---------------------------------------------------------------------------

ensure_dirs_6() {
    ensure_dirs
    mkdir -p "${HUB_CACHE}"
    # Same rootless permission fix as ensure_dirs() applies to CACHE_ROOT: the
    # container is uid 1234 and must be able to write here.
    chmod -R a+rwX "${HUB_CACHE}" 2>/dev/null || true
}

require_image() {
    if docker image inspect "${ISAAC_IMAGE}" >/dev/null 2>&1; then
        return
    fi
    echo "ERROR: ${ISAAC_IMAGE} is not present locally." >&2
    echo "       Pull it first (~25 GB). This does not touch the 5.1.0 image:" >&2
    echo "         docker login nvcr.io" >&2
    echo "           username: \$oauthtoken" >&2
    echo "           password: <your NGC API key>" >&2
    echo "         docker pull ${ISAAC_IMAGE}" >&2
    exit 1
}

# 6.0 moved some extension paths. If ROS_BRIDGE_LIB is not in the image then
# LD_LIBRARY_PATH points at nothing and ROS 2 topics simply never appear, with
# no error message anywhere. Much cheaper to learn that here than halfway
# through an experiment.
probe_ros_bridge() {
    if docker run --rm --entrypoint bash "${ISAAC_IMAGE}" \
            -c "test -d '${ROS_BRIDGE_LIB}'" >/dev/null 2>&1; then
        return
    fi
    echo "WARNING: ${ROS_BRIDGE_LIB}" >&2
    echo "         does not exist in ${ISAAC_IMAGE}; the ROS 2 bridge will not load." >&2
    echo "         Find the real path with:" >&2
    echo "           docker run --rm --entrypoint bash ${ISAAC_IMAGE} \\" >&2
    echo "             -c 'ls -d /isaac-sim/exts/*ros2*/*/lib'" >&2
    echo "         then re-run with ROS_BRIDGE_LIB=<path>. Continuing anyway." >&2
    echo >&2
}

# Refuse to start if something already holds a livestream port -- most likely
# the 5.1.0 headless container. Failing to bind is silent on the server side and
# looks exactly like a client-side networking problem from the laptop.
#
# Arguments are "port/proto", e.g. 49100/tcp 47998/udp. The protocol matters:
# signalling is TCP but the media stream is UDP, and an earlier version of this
# checked both with `ss -ltn`, so a process squatting on UDP 47998 sailed
# straight through the check.
check_ports_free() {
    command -v ss >/dev/null 2>&1 || return 0
    local spec p proto flags running
    for spec in "$@"; do
        p="${spec%%/*}"
        proto="${spec##*/}"
        # -l listening, -n numeric, then t or u. UDP has no LISTEN state, so
        # `ss -lun` is the only way to see a bound-but-unconnected media socket.
        [ "${proto}" = "udp" ] && flags="-Hlun" || flags="-Hltn"
        ss "${flags}" "sport = :${p}" 2>/dev/null | grep -q . || continue

        echo "ERROR: ${proto^^} port ${p} is already in use." >&2
        # Report who actually holds it rather than guessing. An earlier version
        # of this blamed the 5.1.0 container unconditionally, which sent you
        # looking at the wrong container when a 6.0.1 one was the holder.
        #
        # Note `docker port` is useless here: --network=host puts the container
        # in the host's network namespace, so Docker records no port mapping.
        # Indent via sed, not the format string: docker strips leading
        # whitespace from --format output.
        running="$(docker ps --filter 'name=isaac-sim' \
                   --format '{{.Names}}  ({{.Image}}, {{.Status}})' 2>/dev/null \
                   | sed 's/^/         /')"
        if [ -n "${running}" ]; then
            echo "       Isaac Sim containers running right now:" >&2
            echo "${running}" >&2
            echo "       Stop whichever holds it:  docker stop <name>" >&2
        else
            echo "       No Isaac Sim container is running, so something else holds it:" >&2
            echo "         sudo ss -${proto:0:1}lnp \"sport = :${p}\"" >&2
        fi
        echo "       Or move this run's ports:" >&2
        echo "         LIVESTREAM_SIGNAL_PORT=49300 LIVESTREAM_STREAM_PORT=48198 $0" >&2
        exit 1
    done
}

print_banner_6() {
    print_banner
    echo "  Hub cache : ${HUB_CACHE}"
    echo "              (5.1.0's ~/docker/isaac-sim is not touched)"
}

# The launchers deliberately do not pass --rm, so a stopped container keeps its
# filesystem and can be resumed. handle_existing_container() only deletes it when
# you start a *fresh* run from the launcher.
print_resume_hint() {
    local name="$1"
    echo
    echo "Container '${name}' still exists (stopped)."
    echo "  Resume  : docker start -i ${name}"
    echo "  Discard : docker rm ${name}"
}
