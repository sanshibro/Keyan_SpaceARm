#!/usr/bin/env bash
set -eo pipefail
root=/home/uav/paper_revision_20261117
source /opt/ros/noetic/setup.bash
source /opt/qingyu_api_ros1/setup.bash
set -u
out="$root/revision/data/qingyu_visual_probe"
mkdir -p "$out"
export ROS_MASTER_URI=http://127.0.0.1:11484
export ROS_HOSTNAME=127.0.0.1
export ROS_LOG_DIR="$out/ros_logs"
python3 - <<'PY'
import socket
s=socket.socket();s.bind(('127.0.0.1',11484));s.close()
PY
core_pid=;visual_pid=;bridge_pid=
cleanup(){
  if [[ -n "$bridge_pid" ]]; then kill -INT "$bridge_pid" 2>/dev/null || true; wait "$bridge_pid" 2>/dev/null || true; fi
  if [[ -n "$visual_pid" ]]; then kill -INT "$visual_pid" 2>/dev/null || true; wait "$visual_pid" 2>/dev/null || true; fi
  if [[ -n "$core_pid" ]]; then kill -INT "$core_pid" 2>/dev/null || true; wait "$core_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT
roscore -p 11484 >"$out/roscore.log" 2>&1 & core_pid=$!
for i in $(seq 1 80); do if rosnode list >/dev/null 2>&1; then break; fi; sleep .1; done
rosparam load /opt/qingyu_api_ros1/share/continuum_arm/config/continuum_arm_visualize.yaml /continuum_arm_visualize
# Deliberately launch only the marker generator, not the SDK launch file that
# enables the serial sender and API bridge by default.
rosrun continuum_arm continuum_arm_visualize >"$out/visual.log" 2>&1 & visual_pid=$!
rosrun continuum_arm qingyu_api_bridge _joint_topic:=/paper_visual_probe/command _pose_topic:=/paper_visual_probe/pose >"$out/bridge.log" 2>&1 & bridge_pid=$!
timeout 25 python3 "$root/revision/scripts/probe_qingyu_visual.py"
