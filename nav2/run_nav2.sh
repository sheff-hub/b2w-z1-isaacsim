#!/bin/bash
# Start Nav2 (config images.nav2, b2w-nav2:jazzy) for the simulated B2W, detached, as container containers.nav2.
#   nav2/run_nav2.sh          start
#   nav2/run_nav2.sh stop     stop it
#   docker logs -f nav2       watch it
# Talks to Isaac Sim over UDP DDS on the host network (isaac/fastdds.xml), on the ROS domain of
# config.yaml (ros.domain_id) so other users' ROS 2 nodes on this machine are not seen.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC="$(cd "${HERE}/../isaac" && pwd)"
eval "$(python3 "${HERE}/../isaac/config.py" --shell)"   # config.yaml: image, container name, ROS domain
if [ "${1:-}" = "stop" ]; then docker stop "${NAV2_CONTAINER}" >/dev/null 2>&1 && echo "nav2 stopped" || echo "nav2 not running"; exit 0; fi
docker rm -f "${NAV2_CONTAINER}" >/dev/null 2>&1 || true
NAV2_SENSOR="${NAV2_SENSOR:-os0}" NAV2_CONTROLLER="${NAV2_CONTROLLER:-rpp}" NAV2_MAP="${NAV2_MAP:-warehouse}" python3 "${HERE}/make_params.py"
docker run -d --name "${NAV2_CONTAINER}" --network=host --ipc=host \
    -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID}" \
    -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    -e FASTRTPS_DEFAULT_PROFILES_FILE=/fastdds.xml \
    -v "${ISAAC}/fastdds.xml:/fastdds.xml:ro" \
    -v "${HERE}:/nav2:ro" \
    -v "${HERE}/maps:/maps:ro" \
    "${IMAGE_NAV2}" bash -c "source /opt/ros/jazzy/setup.bash && ros2 launch /nav2/b2w_nav2.launch.py" >/dev/null
echo "nav2 started (docker logs -f ${NAV2_CONTAINER})"
