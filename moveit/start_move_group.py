#!/usr/bin/env python3
"""Start MoveIt's move_group for the Z1 arm (runs inside the b2w-moveit container, see run_moveit.sh).

MoveIt's own MoveItConfigsBuilder turns config/ (URDF, SRDF, kinematics, joint limits) into move_group's
parameters, so they always have the format of the installed MoveIt version. It finds config files through a
ROS package, so a minimal one (z1_moveit_config) is made under /tmp and put on the ament path.
"""
import os
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
PKG = Path("/tmp/z1pkg")


def fake_package():
    share = PKG / "share/z1_moveit_config"
    share.mkdir(parents=True, exist_ok=True)
    (share / "package.xml").write_text('<?xml version="1.0"?><package format="3"><name>z1_moveit_config</name>'
                                       '<version>0.0.1</version><description>Z1 MoveIt config</description>'
                                       '<maintainer email="none@none">none</maintainer><license>none</license>'
                                       '</package>')
    link = share / "config"
    if not link.exists():
        link.symlink_to(HERE / "config")
    index = PKG / "share/ament_index/resource_index/packages"
    index.mkdir(parents=True, exist_ok=True)
    (index / "z1_moveit_config").write_text("")
    os.environ["AMENT_PREFIX_PATH"] = f"{PKG}:{os.environ.get('AMENT_PREFIX_PATH', '')}"


def main():
    fake_package()
    from moveit_configs_utils import MoveItConfigsBuilder

    cfg = (MoveItConfigsBuilder("z1", package_name="z1_moveit_config")
           .robot_description(file_path="config/z1.urdf")
           .robot_description_semantic(file_path="config/z1.srdf")
           .robot_description_kinematics(file_path="config/kinematics.yaml")
           .joint_limits(file_path="config/joint_limits.yaml")
           .planning_pipelines(pipelines=["ompl"], default_planning_pipeline="ompl")
           .planning_scene_monitor(publish_robot_description=True, publish_robot_description_semantic=True)
           .to_moveit_configs())
    params = cfg.to_dict()
    # plan only: the simulator executes the trajectories itself (moveit_server.py hands them over)
    params["moveit_manage_controllers"] = False
    params["publish_planning_scene"] = True
    out = Path("/tmp/move_group_params.yaml")
    out.write_text(yaml.safe_dump({"move_group": {"ros__parameters": params}}, sort_keys=False))
    print(f"[moveit] parameters -> {out}", flush=True)
    os.execvp("ros2", ["ros2", "run", "moveit_ros_move_group", "move_group", "--ros-args",
                       "--params-file", str(out)] + sys.argv[1:])


if __name__ == "__main__":
    main()
