# ROS 2 Jazzy + Nav2 for the B2W fetch pipeline. Isaac Sim (isaac-lab:2.3.2) talks to it over
# DDS on the host network with its bundled Jazzy libraries (FastDDS), so both sides match.
FROM ros:jazzy-ros-base
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-navigation2 ros-jazzy-nav2-bringup ros-jazzy-nav2-simple-commander \
        ros-jazzy-slam-toolbox ros-jazzy-rmw-fastrtps-cpp \
    && rm -rf /var/lib/apt/lists/*
# RViz (nav2/rviz.sh); its libraries already come with nav2_bringup, this adds the executable
RUN apt-get update && apt-get install -y --no-install-recommends ros-jazzy-rviz2 \
    && rm -rf /var/lib/apt/lists/*
# rqt (ros2_shell.sh: rqt, rqt_graph, rqt_tf_tree, topic monitor, plots, ...)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-rqt ros-jazzy-rqt-common-plugins ros-jazzy-rqt-tf-tree \
    && rm -rf /var/lib/apt/lists/*
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc
WORKDIR /nav2
