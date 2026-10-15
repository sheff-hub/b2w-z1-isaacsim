"""Nav2 for the simulated B2W: the static map (map_server) + nav2_bringup's navigation stack, on
sim time, no AMCL (the simulator publishes map -> odom -> base_link exactly; on the real robot
put localization_launch.py or slam_toolbox here instead)."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

PARAMS = "/nav2/b2w_nav2_params.yaml"


def generate_launch_description():
    bringup = get_package_share_directory("nav2_bringup")
    return LaunchDescription([
        Node(package="nav2_map_server", executable="map_server", name="map_server", output="screen",
             parameters=[PARAMS]),
        Node(package="nav2_lifecycle_manager", executable="lifecycle_manager", name="lifecycle_manager_map",
             output="screen",
             parameters=[{"use_sim_time": True, "autostart": True, "node_names": ["map_server"]}]),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(bringup, "launch", "navigation_launch.py")),
            launch_arguments={"params_file": PARAMS, "use_sim_time": "True", "autostart": "True"}.items(),
        ),
    ])
