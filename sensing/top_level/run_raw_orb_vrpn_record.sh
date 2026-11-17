#!/usr/bin/env bash
set -e

source /opt/ros/noetic/setup.bash
source /home/uav/桌面/Test/catkin_ws/devel/setup.bash

RUN_DIR="/home/uav/桌面/Test/raw_runs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}"
echo "Saving raw ORB/VRPN run to: ${RUN_DIR}"

exec roslaunch orb_slam3_ros raw_orb_vrpn_record.launch run_dir:="${RUN_DIR}" "$@"
