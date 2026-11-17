#!/usr/bin/env bash
set -euo pipefail
PILOT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/noetic/setup.bash
source "$PILOT_ROOT/ros_ws/devel/setup.bash"
export ROS_MASTER_URI="http://127.0.0.1:11327"
export ROS_HOSTNAME="127.0.0.1"
export PYTHONPATH="$PILOT_ROOT/../.mujoco_deps${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
PILOT_OUTPUT="${1:-$PILOT_ROOT/runs/fast_lio_pilot}"
mkdir -p "$PILOT_OUTPUT"
roslaunch -p 11327 "$PILOT_ROOT/ros_ws/launch/mujoco_fast_lio.launch" > "$PILOT_OUTPUT/fast_lio.log" 2>&1 &
PILOT_LAUNCH_PID=$!
cleanup() {
  kill "$PILOT_LAUNCH_PID" 2>/dev/null || true
  wait "$PILOT_LAUNCH_PID" 2>/dev/null || true
}
trap cleanup EXIT
for ((attempt=0; attempt<50; attempt++)); do
  if rostopic list 2>/dev/null | rg -q '^/Odometry$'; then break; fi
  if ! kill -0 "$PILOT_LAUNCH_PID" 2>/dev/null; then
    tail -n 30 "$PILOT_OUTPUT/fast_lio.log" >&2
    exit 1
  fi
  sleep .2
done
if ! rostopic list 2>/dev/null | rg -q '^/Odometry$'; then
  tail -n 30 "$PILOT_OUTPUT/fast_lio.log" >&2
  exit 1
fi
python3 "$PILOT_ROOT/fast_lio_bridge.py" --seconds "${2:-15}" --output "$PILOT_OUTPUT"
