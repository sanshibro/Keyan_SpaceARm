#!/usr/bin/env bash
set -e

source /opt/ros/noetic/setup.bash
source /home/uav/桌面/Test/catkin_ws/devel/setup.bash

RUN_DIR="/home/uav/桌面/Test/orb_fai_runs/$(date +%Y%m%d_%H%M%S)_d435i_fault"
mkdir -p "${RUN_DIR}"
echo "Saving ORB fai fault run to: ${RUN_DIR}"

roslaunch orb_slam3_ros d435i_fault_stereo.launch viewer:=true rviz:=false fault_mode:=normal &
ORB_PID=$!

sleep 5
roslaunch orb_slam3_ros orb_fai_record.launch run_dir:="${RUN_DIR}" &
FAI_PID=$!

trap 'kill ${ORB_PID} ${FAI_PID} 2>/dev/null || true' INT TERM EXIT
wait
