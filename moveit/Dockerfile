# MoveIt 2 (ROS 2 Jazzy) for the Z1 arm: move_group + our HTTP bridge (moveit_server.py).
# Built by ./setup.sh (config images.moveit); started by moveit/run_moveit.sh.
FROM ros:jazzy-ros-base
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-moveit-ros-move-group ros-jazzy-moveit-planners-ompl ros-jazzy-moveit-kinematics \
        ros-jazzy-moveit-simple-controller-manager ros-jazzy-moveit-configs-utils ros-jazzy-pick-ik \
        ros-jazzy-robot-state-publisher ros-jazzy-moveit-msgs ros-jazzy-rmw-fastrtps-cpp python3-numpy python3-yaml \
    && rm -rf /var/lib/apt/lists/*
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp
WORKDIR /moveit
