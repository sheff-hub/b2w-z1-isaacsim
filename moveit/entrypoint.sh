#!/bin/bash
# Inside the b2w-moveit container: move_group in the background, the HTTP bridge in the foreground.
set -e
source /opt/ros/jazzy/setup.bash
python3 /moveit/start_move_group.py > /tmp/move_group.log 2>&1 &
exec python3 -u /moveit/moveit_server.py
