#!/usr/bin/env bash
# One-command, fail-closed closed-loop trial for one method/scenario/map tuple.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE="$(cd "$PACKAGE_DIR/../.." && pwd)"
PROJECT_ROOT="$(cd "$WORKSPACE/.." && pwd)"
XTDRONE_DIR="/home/uav/桌面/XTDrone_test/XTDrone"

METHOD=""
RESULT_METHOD_LABEL=""
ABLATION_VARIANT="full"
SWITCH_CONTINUITY_MODE="whole_body_bridge"
SCENARIO_DIR=""
MAP_FILE=""
FORMAL=false
GAZEBO_GUI=false
TASK_TIMEOUT=900
PREFLIGHT_TIMEOUT=300
METHOD_TIME_BUDGET=10.0
RECORD_SLAM_MAP=""
SLAM_DATABASE=""
MAP_LOCALIZATION_BACKEND="px4_ekf"
MAP_SENSOR_MODE="dual"
FOOTPRINT_CLEAR_RADIUS_XY="0.28"
RESULT="SYSTEM_ERROR"
RUN_DIR=""
RUN_NAME=""
OUTPUT_ROOT="$PROJECT_ROOT/xtdrone_plan/experiments/method_comparison/closed_loop"
LOG_ROOT=""
PIDS=()

usage() {
  cat <<'EOF'
Usage: run_method_comparison_closed_loop.sh --method NAME --scenario-dir DIR --map FILE [options]

Options:
  --formal                 Enforce audited non-truth map and clean relevant code.
  --gazebo-gui             Show Gazebo GUI.
  --method-time-budget S   Backend budget per planning attempt (default 10).
  --result-method-label N  Label stored in the experiment manifest.
  --ablation-variant N     full, no_seed, or fixed_phi.
  --switch-continuity N    whole_body_bridge or position_only.
  --preflight-timeout S    Live planning gate timeout (default 300).
  --task-timeout S         Whole-task timeout (default 900).
  --output-root DIR        Closed-loop run root.
  --record-slam-map FILE   Record registered RGB-D observations during this survey run.
  --map-localization NAME  px4_ekf (frozen-map default) or rtabmap_icp (online-SLAM pilot).
  --map-sensor-mode NAME   front, down, or dual (default dual).
  --slam-database FILE     RTAB-Map database for the survey (default beside map).
  --footprint-clear-radius M  Physical footprint clearing radius (default 0.28).

This command starts a clean PX4/Gazebo/MAVROS session, spawns hidden oracle
geometry, runs one common-SLAM method, executes only after the live safety gate,
and audits measured whole-body clearance against oracle_truth.yaml.
EOF
}

while (($#)); do
  case "$1" in
    --method) shift; METHOD="${1:-}" ;;
    --result-method-label) shift; RESULT_METHOD_LABEL="${1:-}" ;;
    --ablation-variant) shift; ABLATION_VARIANT="${1:-}" ;;
    --switch-continuity) shift; SWITCH_CONTINUITY_MODE="${1:-}" ;;
    --scenario-dir) shift; SCENARIO_DIR="${1:-}" ;;
    --map) shift; MAP_FILE="${1:-}" ;;
    --formal) FORMAL=true ;;
    --gazebo-gui) GAZEBO_GUI=true ;;
    --method-time-budget) shift; METHOD_TIME_BUDGET="${1:-}" ;;
    --preflight-timeout) shift; PREFLIGHT_TIMEOUT="${1:-}" ;;
    --task-timeout) shift; TASK_TIMEOUT="${1:-}" ;;
    --output-root) shift; OUTPUT_ROOT="${1:-}" ;;
    --record-slam-map) shift; RECORD_SLAM_MAP="${1:-}" ;;
    --slam-database) shift; SLAM_DATABASE="${1:-}" ;;
    --map-localization) shift; MAP_LOCALIZATION_BACKEND="${1:-}" ;;
    --map-sensor-mode) shift; MAP_SENSOR_MODE="${1:-}" ;;
    --footprint-clear-radius) shift; FOOTPRINT_CLEAR_RADIUS_XY="${1:-}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

[[ -n "$METHOD" && -n "$SCENARIO_DIR" && -n "$MAP_FILE" ]] || {
  usage >&2; exit 2;
}
case "$METHOD" in
  fixed_morphology|aet_sequential_minjerk|informed_rrtstar_wb|global_batch_wb|rolling_wb_v2) ;;
  *) echo "Unknown method: $METHOD" >&2; exit 2 ;;
esac
[[ -n "$RESULT_METHOD_LABEL" ]] || RESULT_METHOD_LABEL="$METHOD"
case "$ABLATION_VARIANT" in
  full|no_seed|fixed_phi) ;;
  *) echo "Unknown ablation variant: $ABLATION_VARIANT" >&2; exit 2 ;;
esac
case "$SWITCH_CONTINUITY_MODE" in
  whole_body_bridge|position_only) ;;
  *) echo "Unknown switch continuity mode: $SWITCH_CONTINUITY_MODE" >&2; exit 2 ;;
esac
if [[ "$METHOD" != rolling_wb_v2 ]] && {
  [[ "$ABLATION_VARIANT" != full ]] || [[ "$SWITCH_CONTINUITY_MODE" != whole_body_bridge ]];
}; then
  echo "Ablation controls are valid only for rolling_wb_v2" >&2
  exit 2
fi
case "$MAP_LOCALIZATION_BACKEND" in
  px4_ekf|rtabmap_icp) ;;
  *) echo "Unknown map localization backend: $MAP_LOCALIZATION_BACKEND" >&2; exit 2 ;;
esac
case "$MAP_SENSOR_MODE" in
  front|down|dual) ;;
  *) echo "Unknown map sensor mode: $MAP_SENSOR_MODE" >&2; exit 2 ;;
esac
SCENARIO_DIR="$(realpath "$SCENARIO_DIR")"
MAP_FILE="$(realpath "$MAP_FILE")"
# roslaunch nodes do not inherit the caller's working directory contract.
# Canonicalize the run root before passing it as a ROS parameter so the shell
# gate and metrics logger always observe the same directory.
OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"
if [[ -n "$RECORD_SLAM_MAP" ]]; then
  RECORD_SLAM_MAP="$(realpath -m "$RECORD_SLAM_MAP")"
  [[ -n "$SLAM_DATABASE" ]] || SLAM_DATABASE="${RECORD_SLAM_MAP%.npz}.db"
  SLAM_DATABASE="$(realpath -m "$SLAM_DATABASE")"
  mkdir -p "$(dirname "$RECORD_SLAM_MAP")" "$(dirname "$SLAM_DATABASE")"
fi
TASK_FILE="$SCENARIO_DIR/task.yaml"
ORACLE_FILE="$SCENARIO_DIR/oracle_truth.yaml"
[[ -f "$TASK_FILE" && -f "$ORACLE_FILE" && -f "$MAP_FILE" ]] || {
  echo "Scenario requires task.yaml, oracle_truth.yaml and the selected map" >&2; exit 2;
}
TAKEOFF_Z="$(python3 - "$TASK_FILE" <<'PY'
import sys, yaml
task = yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}
print(float(task["start_state"][2]))
PY
)"
SCENARIO_NAME="$(python3 - "$TASK_FILE" <<'PY'
from pathlib import Path
import sys, yaml
task = yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}
print(str(task.get("name", Path(sys.argv[1]).parent.name)))
PY
)"
RANDOM_SEED="$(python3 - "$TASK_FILE" <<'PY'
import sys, yaml
task = yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}
seed = task.get("seed")
if not isinstance(seed, int):
    raise SystemExit("Task requires an integer paired random seed")
print(seed)
PY
)"

source /opt/ros/noetic/setup.bash
source "$PACKAGE_DIR/env_v2_deps.sh"
source "$WORKSPACE/devel/setup.bash"
export XTDRONE_DIR
export ROS_PACKAGE_PATH="/home/uav/PX4-Autopilot:/home/uav/PX4-Autopilot/Tools/sitl_gazebo:$XTDRONE_DIR/sitl_config:${ROS_PACKAGE_PATH:-}"

env PYTHONPATH="$PACKAGE_DIR/src" python3 - "$MAP_FILE" "$FORMAL" <<'PY'
import sys
from uav_continuum_whole_body_v2.mapping import MapSnapshot
s = MapSnapshot.load(sys.argv[1])
formal = sys.argv[2].lower() == "true"
if formal and s.metadata.get("truth_derived"):
    raise SystemExit("FORMAL_REJECTED: truth-derived map")
if formal and s.metadata.get("formal_experiment_allowed") is not True:
    raise SystemExit("FORMAL_REJECTED: map not audited for formal experiments")
if formal and s.metadata.get("comparison_independent_survey") is not True:
    raise SystemExit("FORMAL_REJECTED: map survey used a compared method")
if formal and int(s.metadata.get("map_audit_schema_version", 0)) < 5:
    raise SystemExit("FORMAL_REJECTED: map audit schema is older than v5")
if formal and not s.metadata.get("survey_protocol_manifest_sha256"):
    raise SystemExit("FORMAL_REJECTED: missing frozen survey manifest hash")
print("MAP_HASH=" + s.map_hash)
print("MAP_SOURCE=" + s.source)
PY

if [[ "$FORMAL" == true ]]; then
  dirty="$(git -C "$PROJECT_ROOT" status --porcelain --untracked-files=normal -- \
    xtdrone_catkin_ws/src/uav_continuum_whole_body_v2 \
    xtdrone_catkin_ws/src/xtdrone_ego_tools \
    xtdrone_catkin_ws/src/uav_continuum_opportunity_planner/scripts/arm_shape_sim_node.py \
    xtdrone_catkin_ws/src/uav_continuum_opportunity_planner/scripts/arm_kinematics_node.py)"
  [[ -z "$dirty" ]] || { echo "FORMAL_REJECTED: relevant code is dirty" >&2; echo "$dirty" >&2; exit 3; }
fi

for process in rosmaster gzserver px4 rviz; do
  if pgrep -x "$process" >/dev/null; then
    echo "Refusing to start over existing $process" >&2; exit 3
  fi
done

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${SCENARIO_NAME}__${RESULT_METHOD_LABEL}__${STAMP}_$$"
mkdir -p "$OUTPUT_ROOT"
LOG_ROOT="/tmp/method_comparison_${RUN_NAME}"
mkdir -p "$LOG_ROOT"

cleanup() {
  local code=$?
  set +e
  if ((${#PIDS[@]})); then
    for pid in "${PIDS[@]}"; do kill -INT "$pid" 2>/dev/null; done
    sleep 3
    for pid in "${PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null; done
    for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null; done
  fi
  if [[ -n "$RUN_DIR" && -d "$RUN_DIR" ]]; then
    printf '{"result":"%s","exit_code":%d,"method":"%s","scenario":"%s"}\n' \
      "$RESULT" "$code" "$RESULT_METHOD_LABEL" "$SCENARIO_NAME" > "$RUN_DIR/trial_result.json"
  fi
  echo "METHOD_COMPARISON_RESULT=$RESULT"
  echo "RUN_DIR=$RUN_DIR"
  echo "LAUNCH_LOG_DIR=$LOG_ROOT"
  return "$code"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

start_launch() {
  local name=$1; shift
  "$@" >"$LOG_ROOT/$name.log" 2>&1 &
  PIDS+=("$!")
  echo "started $name pid=$!"
}
wait_until() {
  local timeout_s=$1 description=$2; shift 2
  local deadline=$((SECONDS + timeout_s))
  while ((SECONDS < deadline)); do
    "$@" && return 0
    sleep 1
  done
  echo "Timed out: $description" >&2
  return 1
}
master_ready() { rosparam get /rosversion >/dev/null 2>&1; }
mavros_connected() {
  timeout 3 rostopic echo -n 1 /iris_0/mavros/state 2>/dev/null | grep -q 'connected: True'
}
apply_px4_profile() {
  local profile=$1 log_file="$LOG_ROOT/${1}_params.log"
  # roslaunch exits zero even when its only parameter-setting node dies. Run
  # the node directly so a service failure propagates to this fail-closed
  # experiment script and can be classified as infrastructure, not planning.
  rosrun xtdrone_ego_tools set_px4_sitl_params.py \
    _namespace:=/iris_0 _profile:="$profile" >"$log_file" 2>&1
}
scene_ready() {
  timeout 3 rostopic echo -n 1 /experiment/simulator_scene_status/data 2>/dev/null | grep -q READY
}
map_scan_ready() {
  timeout 3 rostopic echo -n 1 /planner/map_survey_scan_status/data 2>/dev/null | grep -q READY
}
map_scan_complete() {
  timeout 3 rostopic echo -n 1 /planner/map_survey_scan_status/data 2>/dev/null | grep -q COMPLETE
}
sensor_map_recorder_ready() {
  local nodes
  nodes="$(rosnode list 2>/dev/null || true)"
  grep -qx /slam_map_snapshot <<<"$nodes" || return 1
  if [[ "$MAP_SENSOR_MODE" != down ]]; then
    grep -qx /register_depth_cloud <<<"$nodes" || return 1
    [[ "$(rostopic type /slam/registered_cloud 2>/dev/null || true)" == sensor_msgs/PointCloud2 ]] || return 1
  fi
  if [[ "$MAP_SENSOR_MODE" != front ]]; then
    grep -qx /register_down_depth_cloud <<<"$nodes" || return 1
    [[ "$(rostopic type /slam/registered_down_cloud 2>/dev/null || true)" == sensor_msgs/PointCloud2 ]] || return 1
  fi
  [[ "$(rostopic type /planner/map/metadata 2>/dev/null || true)" == std_msgs/String ]]
}
sensor_map_snapshot_ready() {
  # MapSnapshot.save publishes via atomic rename only after a complete NPZ has
  # been fsynced and re-opened successfully. The file is therefore a stronger
  # readiness gate than parsing an escaped YAML representation of JSON.
  [[ -s "$RECORD_SLAM_MAP" ]]
}
px4_ready() {
  local state
  state="$(timeout 3 rostopic echo -n 1 /iris_0/mavros/state 2>/dev/null || true)"
  grep -q 'armed: True' <<<"$state" && grep -q 'mode: "OFFBOARD"' <<<"$state"
}
takeoff_stable() {
  local line
  line="$(timeout 4 rostopic echo -p -n 1 /iris_0/mavros/local_position/pose 2>/dev/null | tail -n 1 || true)"
  python3 - "$line" "$TAKEOFF_Z" <<'PY'
import sys
try: x,y,z=map(float,sys.argv[1].split(',')[4:7])
except Exception: raise SystemExit(1)
target_z=float(sys.argv[2])
raise SystemExit(0 if abs(x)<.12 and abs(y)<.12 and abs(z-target_z)<.10 else 1)
PY
}

echo "[1/8] Start PX4/Gazebo/MAVROS in controlled empty world"
start_launch gazebo roslaunch xtdrone_ego_tools indoor1_depth_xtdrone.launch \
  world:="$PACKAGE_DIR/config/comparison_empty.world" x:=0 y:=0 z:=0.05 gui:="$GAZEBO_GUI"
wait_until 90 "ROS master" master_ready
wait_until 120 "MAVROS" mavros_connected

echo "[2/8] Spawn hidden oracle scene"
start_launch scene roslaunch uav_continuum_whole_body_v2 spawn_comparison_scene.launch \
  oracle_file:="$ORACLE_FILE" model_prefix:="paper_${SCENARIO_NAME}_"
wait_until 60 "simulator scene" scene_ready

echo "[3/8] Start common-map planner and method=$METHOD"
EXPERIMENT_CLASS=closed_loop_method_comparison_pilot
[[ "$FORMAL" == true ]] && EXPERIMENT_CLASS=closed_loop_method_comparison
if [[ -n "$RECORD_SLAM_MAP" ]]; then
  EXPERIMENT_CLASS=sensor_map_survey_not_comparison
  echo "      Start RGB-D map recorder backend=$MAP_LOCALIZATION_BACKEND: $RECORD_SLAM_MAP"
  if [[ "$MAP_LOCALIZATION_BACKEND" == px4_ekf ]]; then
    enable_front=true
    enable_down=true
    down_points=800
    [[ "$MAP_SENSOR_MODE" == down ]] && enable_front=false
    [[ "$MAP_SENSOR_MODE" == front ]] && enable_down=false
    [[ "$MAP_SENSOR_MODE" == down ]] && down_points=5000
    snapshot_pose=/slam/sensor_pose
    snapshot_clearing_pose=/slam/estimator_body_pose
    if [[ "$MAP_SENSOR_MODE" == down ]]; then
      snapshot_pose=/slam/down_sensor_pose
      snapshot_clearing_pose=/slam/down_estimator_body_pose
    fi
    start_launch slam roslaunch uav_continuum_whole_body_v2 px4_ekf_rgbd_map_record.launch \
      snapshot_file:="$RECORD_SLAM_MAP" \
      initial_x:=0 initial_y:=0 initial_z:=0.05 initial_yaw:=0 \
      footprint_clear_radius_xy:="$FOOTPRINT_CLEAR_RADIUS_XY" \
      enable_front_sensor:="$enable_front" enable_down_sensor:="$enable_down" \
      down_maximum_points:="$down_points" \
      snapshot_sensor_pose_topic:="$snapshot_pose" \
      snapshot_clearing_pose_topic:="$snapshot_clearing_pose"
  else
    start_launch slam roslaunch uav_continuum_whole_body_v2 rtabmap_rgbd_map_record.launch \
      snapshot_file:="$RECORD_SLAM_MAP" database_path:="$SLAM_DATABASE" \
      initial_x:=0 initial_y:=0 initial_z:=1 initial_yaw:=0
  fi
  if [[ "$MAP_LOCALIZATION_BACKEND" == px4_ekf ]]; then
    wait_until 30 "dual RGB-D map recorder" sensor_map_recorder_ready
  fi
fi
start_launch method roslaunch uav_continuum_whole_body_v2 method_comparison_core.launch \
  method:="$METHOD" map_mode:=frozen_slam map_snapshot_file:="$MAP_FILE" \
  result_method_label:="$RESULT_METHOD_LABEL" ablation_variant:="$ABLATION_VARIANT" \
  switch_continuity_mode:="$SWITCH_CONTINUITY_MODE" \
  task_file:="$TASK_FILE" auto_execute:=false method_time_budget:="$METHOD_TIME_BUDGET" \
  random_seed:="$RANDOM_SEED" \
  oracle_file:="$ORACLE_FILE" \
  experiment_class:="$EXPERIMENT_CLASS" logger_output_dir:="$OUTPUT_ROOT" run_name:="$RUN_NAME"
RUN_DIR="$OUTPUT_ROOT/$RUN_NAME"
wait_until 30 "metrics run directory" test -d "$RUN_DIR"

echo "[4/8] Establish the only takeoff setpoint owner"
start_launch hold roslaunch uav_continuum_whole_body_v2 safe_takeoff_hold.launch \
  x:=0 y:=0 z:="$TAKEOFF_Z"
apply_px4_profile takeoff
rosservice call /iris_0/mavros/set_mode "{base_mode: 0, custom_mode: 'OFFBOARD'}" >/dev/null
rosservice call /iris_0/mavros/cmd/arming "{value: true}" >/dev/null
wait_until 30 "PX4 OFFBOARD" px4_ready
wait_until 90 "takeoff stabilization" takeoff_stable
if [[ -n "$RECORD_SLAM_MAP" ]]; then
  wait_until 45 "first sensor map snapshot" sensor_map_snapshot_ready
fi
apply_px4_profile tracking

echo "[5/8] Start measured world pose and apply live gate"
start_launch truth roslaunch xtdrone_ego_tools gazebo_truth_for_ego.launch
if ! rosrun uav_continuum_whole_body_v2 check_method_comparison_live.py \
  --timeout "$PREFLIGHT_TIMEOUT" | tee "$LOG_ROOT/live_gate.log"; then
  RESULT=PLANNING_OR_LIVE_GATE_FAILURE
  exit 4
fi

echo "[6/8] Atomically hand setpoint ownership to the validated method"
rostopic pub -1 /planner/v2/executor_enable std_msgs/Bool 'data: true' >"$LOG_ROOT/enable.log" 2>&1 &
enable_pid=$!
sleep 0.2
rosnode kill /safe_takeoff_hold_v2 >/dev/null 2>&1 || true
wait "$enable_pid"

# A successful one-shot publish is not the handoff proof: the executor may
# immediately reject its measured pose against the frozen map. Require a
# positive execution-state acknowledgement and preserve transient ABORT events
# from the append-only logger instead of looking only at the latest status.
handoff_deadline=$((SECONDS + 15))
handoff_ready=false
while ((SECONDS < handoff_deadline)); do
  if [[ -f "$RUN_DIR/events.jsonl" ]] && grep -q '"state": "ABORT_' \
      "$RUN_DIR/events.jsonl"; then
    RESULT=EXECUTION_HANDOFF_ABORT
    exit 5
  fi
  executor="$(timeout 4 rostopic echo -n 1 \
    /planner/v2/executor_status/data 2>/dev/null || true)"
  if grep -Eq 'START_SCHEDULED|EXECUTING_' <<<"$executor"; then
    handoff_ready=true
    break
  fi
  sleep 0.25
done
if [[ "$handoff_ready" != true ]]; then
  RESULT=EXECUTION_HANDOFF_TIMEOUT
  exit 5
fi

echo "[7/8] Wait for completion"
deadline=$((SECONDS + TASK_TIMEOUT))
while ((SECONDS < deadline)); do
  status="$(timeout 4 rostopic echo -n 1 /planner/v2/status/data 2>/dev/null || true)"
  grep -q TASK_COMPLETE <<<"$status" && break
  executor="$(timeout 4 rostopic echo -n 1 /planner/v2/executor_status/data 2>/dev/null || true)"
  if grep -q ABORT_ <<<"$executor" || {
      [[ -f "$RUN_DIR/events.jsonl" ]] && grep -q '"state": "ABORT_' \
        "$RUN_DIR/events.jsonl";
    }; then RESULT=TRACKING_ABORT; exit 5; fi
  sleep 2
done
if ((SECONDS >= deadline)); then RESULT=TASK_TIMEOUT; exit 5; fi

if [[ -n "$RECORD_SLAM_MAP" ]]; then
  echo "[7a/8] Transfer setpoint ownership for terminal 360-degree map scan"
  start_launch map_scan rosrun uav_continuum_whole_body_v2 \
    terminal_yaw_map_scan_node.py _duration_s:=24.0
  wait_until 15 "terminal map scan readiness" map_scan_ready
  rosnode kill /whole_body_trajectory_executor_v2 >/dev/null 2>&1
  rostopic pub -1 /planner/map_survey_scan_enable std_msgs/Bool \
    'data: true' >"$LOG_ROOT/map_scan_enable.log" 2>&1
  # The scan node deliberately remains the sole setpoint owner after COMPLETE
  # so the vehicle cannot fall while the final map frame is integrated and
  # atomically saved. Cleanup terminates it only after the audit.
  wait_until 45 "terminal map scan completion" map_scan_complete
  # Allow the final registered depth frame and atomic snapshot replacement.
  sleep 3
fi

echo "[8/8] Independent hidden-truth audit"
sleep 2
if ! rosrun uav_continuum_whole_body_v2 analyze_v2_run.py "$RUN_DIR" \
  --oracle-file "$ORACLE_FILE" | tee "$LOG_ROOT/oracle_audit.log"; then
  RESULT=ORACLE_AUDIT_FAILURE
  exit 6
fi
RESULT=PASS
