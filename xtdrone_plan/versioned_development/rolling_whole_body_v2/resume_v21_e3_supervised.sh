#!/usr/bin/env bash
set -euo pipefail

REPO=/home/uav/桌面/uav_arm_1
RESULTS="$REPO/xtdrone_plan/experiments/method_comparison/v21_e3_final_450_commite62b6f2"

export ROS_HOME=/tmp/ros_v21_e3_resume_home
export ROS_LOG_DIR=/tmp/ros_v21_e3_resume_logs
export XTDRONE_DIR=/home/uav/桌面/XTDrone_test/XTDrone
export ROS_PACKAGE_PATH="/home/uav/PX4-Autopilot:/home/uav/PX4-Autopilot/Tools/sitl_gazebo:$XTDRONE_DIR/sitl_config:${ROS_PACKAGE_PATH:-}"
mkdir -p "$ROS_HOME" "$ROS_LOG_DIR"

cd "$REPO"
source /opt/ros/noetic/setup.bash
source xtdrone_catkin_ws/src/uav_continuum_whole_body_v2/env_v2_deps.sh
source xtdrone_catkin_ws/devel/setup.bash

python3 xtdrone_plan/versioned_development/rolling_whole_body_v2/monitor_v21_e3_progress.py \
  "$RESULTS" --interval 20 --target 450 &
monitor_pid=$!

cleanup() {
  if kill -0 "$monitor_pid" 2>/dev/null; then
    kill "$monitor_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

python3 xtdrone_plan/versioned_development/rolling_whole_body_v2/run_v21_e3_resilient_supervisor.py
supervisor_rc=$?
wait "$monitor_pid" || true
if [[ "$supervisor_rc" -eq 0 ]]; then
  python3 xtdrone_plan/versioned_development/rolling_whole_body_v2/finalize_v21_e3.py \
    "$RESULTS"
fi
exit "$supervisor_rc"
