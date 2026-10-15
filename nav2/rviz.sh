#!/bin/bash
# RViz for the simulated B2W's Nav2, in Docker (ROS 2 Jazzy does not run on this Ubuntu 22.04 host).
#   nav2/rviz.sh            open RViz (Nav2 view: map, costmaps, scan, plan, footprint)
#   nav2/rviz.sh stop       close it
# Run it from a terminal on the workstation's desktop (needs DISPLAY). Nav2 (run_nav2.sh) and the
# simulator can be started before or after; RViz just listens on ROS_DOMAIN_ID 37.
# In RViz, "2D Goal Pose" (toolbar) sends a Nav2 goal by clicking and dragging on the map --
# only while a simulator with --nav2 is running and the fetch script is not driving it itself.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC="$(cd "${HERE}/../isaac" && pwd)"
eval "$(python3 "${HERE}/../isaac/config.py" --shell)"   # config.yaml: image, container name, ROS domain
if [ "${1:-}" = "stop" ]; then docker stop "${NAV2_CONTAINER}-rviz" >/dev/null 2>&1 && echo "rviz stopped" || echo "rviz not running"; exit 0; fi
[ -n "${DISPLAY:-}" ] || { echo "DISPLAY is not set: run this from a terminal on the workstation's desktop." >&2; exit 1; }
command -v xhost >/dev/null 2>&1 && xhost +local: >/dev/null
docker rm -f "${NAV2_CONTAINER}-rviz" >/dev/null 2>&1 || true
docker run -d --name "${NAV2_CONTAINER}-rviz" --network=host --ipc=host \
    -e DISPLAY="${DISPLAY}" -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e LIBGL_ALWAYS_SOFTWARE=1 \
    -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID}" -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    -e FASTRTPS_DEFAULT_PROFILES_FILE=/fastdds.xml -v "${ISAAC}/fastdds.xml:/fastdds.xml:ro" \
    -v "${HERE}:/nav2:ro" \
    "${IMAGE_NAV2}" bash -c "source /opt/ros/jazzy/setup.bash && rviz2 -d /nav2/b2w_view.rviz --ros-args -p use_sim_time:=true" >/dev/null
echo "rviz started (window opens in a few seconds; logs: docker logs -f "${NAV2_CONTAINER}-rviz")"
