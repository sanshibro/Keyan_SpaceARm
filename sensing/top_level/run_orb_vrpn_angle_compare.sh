#!/usr/bin/env bash
set -e

source /opt/ros/noetic/setup.bash
source /home/uav/桌面/Test/catkin_ws/devel/setup.bash

RUN_DIR="/home/uav/桌面/Test/angle_runs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}"
echo "Saving this run to: ${RUN_DIR}"

exec roslaunch orb_slam3_ros angle_compare.launch run_dir:="${RUN_DIR}" "$@"
