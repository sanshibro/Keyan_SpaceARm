#!/usr/bin/env bash
# Record one sensor-map session using a frozen route and no compared planner.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE="$(cd "$PACKAGE_DIR/../.." && pwd)"
PROJECT_ROOT="$(cd "$WORKSPACE/.." && pwd)"
XTDRONE_DIR="/home/uav/桌面/XTDrone_test/XTDrone"
SCENARIO_DIR=""
MANIFEST=""
OUTPUT_DIR=""
SENSOR_MODE=front
GAZEBO_GUI=false
TASK_TIMEOUT=900
PIDS=()
RESULT=SYSTEM_ERROR

usage() {
  cat <<'EOF'
Usage: run_independent_map_survey.sh --scenario-dir DIR --manifest FILE --output-dir DIR [options]

Options:
  --sensor-mode front|down|dual  RGB-D session (default front).
  --task-timeout S              Survey timeout (default 900).
  --gazebo-gui                  Show Gazebo.

The survey controller reads only the frozen waypoint manifest and onboard pose.
It does not load a planner map, a compared method, or oracle geometry.
EOF
}

while (($#)); do
  case "$1" in
    --scenario-dir) shift; SCENARIO_DIR="${1:-}" ;;
    --manifest) shift; MANIFEST="${1:-}" ;;
    --output-dir) shift; OUTPUT_DIR="${1:-}" ;;
    --sensor-mode) shift; SENSOR_MODE="${1:-}" ;;
    --task-timeout) shift; TASK_TIMEOUT="${1:-}" ;;
    --gazebo-gui) GAZEBO_GUI=true ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done
[[ -n "$SCENARIO_DIR" && -n "$MANIFEST" && -n "$OUTPUT_DIR" ]] || { usage >&2; exit 2; }
case "$SENSOR_MODE" in front|down|dual) ;; *) echo "Bad sensor mode" >&2; exit 2;; esac
SCENARIO_DIR="$(realpath "$SCENARIO_DIR")"
MANIFEST="$(realpath "$MANIFEST")"
OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"
ORACLE_FILE="$SCENARIO_DIR/oracle_truth.yaml"
RAW_MAP="$OUTPUT_DIR/recorded_slam_map_raw.npz"
CERTIFIED_MAP="$OUTPUT_DIR/certified_sensor_survey.npz"
CERTIFICATION_REPORT="$OUTPUT_DIR/survey_certification.json"
LOG_ROOT="$OUTPUT_DIR/logs"
mkdir -p "$LOG_ROOT"
rm -f "$RAW_MAP" "$CERTIFIED_MAP" "$CERTIFICATION_REPORT"

source /opt/ros/noetic/setup.bash
source "$PACKAGE_DIR/env_v2_deps.sh"
source "$WORKSPACE/devel/setup.bash"
export XTDRONE_DIR
export ROS_PACKAGE_PATH="/home/uav/PX4-Autopilot:/home/uav/PX4-Autopilot/Tools/sitl_gazebo:$XTDRONE_DIR/sitl_config:${ROS_PACKAGE_PATH:-}"

rosrun uav_continuum_whole_body_v2 certify_independent_survey.py \
  --help >/dev/null
python3 - "$MANIFEST" <<'PY'
import sys
from uav_continuum_whole_body_v2.survey_protocol import load_survey_manifest
load_survey_manifest(sys.argv[1])
PY
for process in rosmaster gzserver px4 rviz; do
  pgrep -x "$process" >/dev/null && { echo "Refusing to start over existing $process" >&2; exit 3; }
done

cleanup() {
  local code=$?
  set +e
  for pid in "${PIDS[@]}"; do kill -INT "$pid" 2>/dev/null; done
  sleep 3
  for pid in "${PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null; done
  for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null; done
  printf '{"result":"%s","exit_code":%d,"manifest":"%s","sensor_mode":"%s","commit":"%s"}\n' \
    "$RESULT" "$code" "$MANIFEST" "$SENSOR_MODE" "$(git -C "$PROJECT_ROOT" rev-parse HEAD)" \
    >"$OUTPUT_DIR/survey_run_report.json"
  echo "INDEPENDENT_SURVEY_RESULT=$RESULT"
  echo "CERTIFIED_MAP=$CERTIFIED_MAP"
  return "$code"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

start_launch() { local name=$1; shift; "$@" >"$LOG_ROOT/$name.log" 2>&1 & PIDS+=("$!"); }
wait_until() {
  local timeout_s=$1 description=$2; shift 2
  local deadline=$((SECONDS + timeout_s))
  while ((SECONDS < deadline)); do "$@" && return 0; sleep 1; done
  echo "Timed out: $description" >&2; return 1
}
master_ready() { rosparam get /rosversion >/dev/null 2>&1; }
mavros_connected() { timeout 3 rostopic echo -n 1 /iris_0/mavros/state 2>/dev/null | grep -q 'connected: True'; }
scene_ready() { timeout 3 rostopic echo -n 1 /experiment/simulator_scene_status/data 2>/dev/null | grep -q READY; }
survey_ready() { timeout 3 rostopic echo -n 1 /planner/independent_survey_status/data 2>/dev/null | grep -q READY; }
survey_complete() { timeout 3 rostopic echo -n 1 /planner/independent_survey_status/data 2>/dev/null | grep -q COMPLETE; }
survey_finished_safely() {
  local deadline=$((SECONDS + TASK_TIMEOUT)) status
  while ((SECONDS < deadline)); do
    status="$(timeout 3 rostopic echo -n 1 /planner/independent_survey_status/data 2>/dev/null || true)"
    grep -q COMPLETE <<<"$status" && return 0
    if grep -q ABORT_ <<<"$status"; then echo "$status" >&2; return 1; fi
    sleep 1
  done
  return 1
}
px4_ready() {
  local state; state="$(timeout 3 rostopic echo -n 1 /iris_0/mavros/state 2>/dev/null || true)"
  grep -q 'armed: True' <<<"$state" && grep -q 'mode: "OFFBOARD"' <<<"$state"
}
recorder_ready() {
  local nodes; nodes="$(rosnode list 2>/dev/null || true)"
  grep -qx /slam_map_snapshot <<<"$nodes" || return 1
  rosservice list 2>/dev/null | grep -qx /slam_map_snapshot/save_snapshot || return 1
  [[ "$SENSOR_MODE" == down ]] || grep -qx /register_depth_cloud <<<"$nodes" || return 1
  [[ "$SENSOR_MODE" == front ]] || grep -qx /register_down_depth_cloud <<<"$nodes" || return 1
}

echo "[1/6] Start simulator and hidden scene"
start_launch gazebo roslaunch xtdrone_ego_tools indoor1_depth_xtdrone.launch \
  world:="$PACKAGE_DIR/config/comparison_empty.world" x:=0 y:=0 z:=0.05 gui:="$GAZEBO_GUI"
wait_until 90 "ROS master" master_ready
wait_until 120 "MAVROS" mavros_connected
start_launch scene roslaunch uav_continuum_whole_body_v2 spawn_comparison_scene.launch \
  oracle_file:="$ORACLE_FILE" model_prefix:="survey_$(basename "$SCENARIO_DIR")_"
wait_until 60 "hidden scene" scene_ready

echo "[2/6] Start sensor-only map recorder"
enable_front=true; enable_down=true; down_points=20000
[[ "$SENSOR_MODE" == down ]] && enable_front=false
[[ "$SENSOR_MODE" == front ]] && enable_down=false
snapshot_pose=/slam/sensor_pose; clearing_pose=/slam/estimator_body_pose
[[ "$SENSOR_MODE" == down ]] && snapshot_pose=/slam/down_sensor_pose && clearing_pose=/slam/down_estimator_body_pose
start_launch recorder roslaunch uav_continuum_whole_body_v2 px4_ekf_rgbd_map_record.launch \
  snapshot_file:="$RAW_MAP" initial_x:=0 initial_y:=0 initial_z:=0.05 initial_yaw:=0 \
  footprint_clear_radius_xy:=0.28 enable_front_sensor:="$enable_front" \
  enable_down_sensor:="$enable_down" down_maximum_points:="$down_points" \
  snapshot_sensor_pose_topic:="$snapshot_pose" snapshot_clearing_pose_topic:="$clearing_pose"
wait_until 45 "sensor recorder" recorder_ready

echo "[3/6] Start frozen waypoint survey controller"
start_launch survey rosrun uav_continuum_whole_body_v2 independent_map_survey_node.py \
  _manifest:="$MANIFEST" _registered_pose_topic:="$clearing_pose"
wait_until 30 "survey controller" survey_ready

echo "[4/6] Enter OFFBOARD and execute manifest"
roslaunch xtdrone_ego_tools px4_sitl_param_fix.launch profile:=takeoff >"$LOG_ROOT/takeoff_params.log" 2>&1
rosservice call /iris_0/mavros/set_mode "{base_mode: 0, custom_mode: 'OFFBOARD'}" >/dev/null
rosservice call /iris_0/mavros/cmd/arming "{value: true}" >/dev/null
wait_until 30 "PX4 OFFBOARD" px4_ready
roslaunch xtdrone_ego_tools px4_sitl_param_fix.launch profile:=tracking >"$LOG_ROOT/tracking_params.log" 2>&1
rostopic pub -1 /planner/independent_survey_enable std_msgs/Bool 'data: true' >/dev/null
survey_finished_safely || { echo "Independent survey aborted or timed out" >&2; exit 4; }
sleep 4
flush_result=""
flush_ok=false
for attempt in 1 2 3; do
  rosservice list 2>/dev/null | grep -qx '/slam_map_snapshot/save_snapshot' || {
    flush_result="snapshot service unavailable on attempt $attempt"
    sleep 5
    continue
  }
  flush_result="$(timeout 180 rosservice call \
    /slam_map_snapshot/save_snapshot 2>&1 || true)"
  if grep -q 'success: True' <<<"$flush_result"; then
    flush_ok=true
    break
  fi
  echo "Snapshot flush attempt $attempt failed: $flush_result" >&2
  sleep 5
done
[[ "$flush_ok" == true ]] || {
  echo "Explicit map snapshot flush failed after 3 attempts: $flush_result" >&2
  exit 4
}
[[ -s "$RAW_MAP" ]] || { echo "Survey produced no map" >&2; exit 4; }

echo "[5/6] Verify measured route against frozen manifest"
rosrun uav_continuum_whole_body_v2 certify_independent_survey.py \
  "$RAW_MAP" "$MANIFEST" "$CERTIFIED_MAP" --report "$CERTIFICATION_REPORT" \
  | tee "$LOG_ROOT/certification.log"

echo "[6/6] Complete"
RESULT=PASS
