#!/usr/bin/env bash
set -euo pipefail
SLAM_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/noetic/setup.bash
source "$SLAM_ROOT/ros_ws/devel/setup.bash"
export ROS_MASTER_URI="http://127.0.0.1:11327"
export ROS_HOSTNAME="127.0.0.1"
export PYTHONPATH="$SLAM_ROOT/../.mujoco_deps${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
roslaunch -p 11327 "$SLAM_ROOT/ros_ws/launch/mujoco_fast_lio.launch" > "$SLAM_ROOT/runs/fast_lio_mission_launch.log" 2>&1 &
SLAM_LAUNCH_PID=$!
cleanup() {
  kill "$SLAM_LAUNCH_PID" 2>/dev/null || true
  wait "$SLAM_LAUNCH_PID" 2>/dev/null || true
}
trap cleanup EXIT
for ((attempt=0; attempt<50; attempt++)); do
  if rostopic list 2>/dev/null | rg -q '^/Odometry$'; then break; fi
  if ! kill -0 "$SLAM_LAUNCH_PID" 2>/dev/null; then
    tail -n 30 "$SLAM_ROOT/runs/fast_lio_mission_launch.log" >&2
    exit 1
  fi
  sleep .2
done
if ! rostopic list 2>/dev/null | rg -q '^/Odometry$'; then
  tail -n 30 "$SLAM_ROOT/runs/fast_lio_mission_launch.log" >&2
  exit 1
fi
python3 "$SLAM_ROOT/visual_grasp_development.py" --slam --seconds "${1:-180}" "${@:2}"
