#!/bin/bash
# A ROS 2 Jazzy terminal on the simulator / Nav2 network (ROS_DOMAIN_ID 37), in Docker -- Jazzy does
# not run natively on this Ubuntu 22.04 host. GUI tools work from it when started from a desktop terminal:
#   nav2/ros2_shell.sh                       interactive shell, then e.g.
#       rviz2 -d /nav2/b2w_view.rviz --ros-args -p use_sim_time:=true    (or just: b2w_rviz)
#       rqt                     rqt_graph         ros2 run rqt_tf_tree rqt_tf_tree
#       ros2 topic list         ros2 topic echo /cmd_vel         ros2 topic hz /point_cloud
#   nav2/ros2_shell.sh ros2 topic hz /scan      run one command and exit
# Several shells can be open at once (each is its own container).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC="$(cd "${HERE}/../isaac" && pwd)"
eval "$(python3 "${HERE}/../isaac/config.py" --shell)"   # config.yaml: image, container name, ROS domain
GUI=()
if [ -n "${DISPLAY:-}" ]; then
    command -v xhost >/dev/null 2>&1 && xhost +local: >/dev/null
    # software OpenGL: RViz draws on the CPU and leaves the GPU to Isaac Sim
    GUI=(-e "DISPLAY=${DISPLAY}" -v /tmp/.X11-unix:/tmp/.X11-unix -e LIBGL_ALWAYS_SOFTWARE=1 -e QT_X11_NO_MITSHM=1)
else
    echo "(DISPLAY not set: command-line tools only; run from a desktop terminal for rviz2 / rqt)" >&2
fi
TTY=(-i)
[ -t 0 ] && TTY=(-it)
RC='source /opt/ros/jazzy/setup.bash
alias b2w_rviz="rviz2 -d /nav2/b2w_view.rviz --ros-args -p use_sim_time:=true"
export PS1="(ros2 jazzy, domain ${ROS_DOMAIN_ID}) \w$ "'
if [ $# -eq 0 ]; then
    CMD="echo '${RC}' > /root/.b2wrc && exec bash --rcfile /root/.b2wrc"
else
    CMD="source /opt/ros/jazzy/setup.bash && $*"
fi
docker run --rm "${TTY[@]}" --network=host --ipc=host \
    -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID}" -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    -e FASTRTPS_DEFAULT_PROFILES_FILE=/fastdds.xml -v "${ISAAC}/fastdds.xml:/fastdds.xml:ro" \
    -v "${HERE}:/nav2:ro" "${GUI[@]}" \
    "${IMAGE_NAV2}" bash -c "${CMD}"
