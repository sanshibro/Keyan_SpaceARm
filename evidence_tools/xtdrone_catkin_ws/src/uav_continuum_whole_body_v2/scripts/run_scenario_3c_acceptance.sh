#!/usr/bin/env bash
# One-command, fail-closed three-wall simulation acceptance.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE="$(cd "$PACKAGE_DIR/../.." && pwd)"
PROJECT_ROOT="$(cd "$WORKSPACE/.." && pwd)"
RUN_ROOT="$PROJECT_ROOT/xtdrone_plan/experiments/v2_runs"
XTDRONE_DIR="/home/uav/桌面/XTDrone_test/XTDrone"

GAZEBO_GUI=false
LAUNCH_RVIZ=true
KEEP_OPEN=false
PREFLIGHT_TIMEOUT=180
TASK_TIMEOUT=900
MIN_FREE_GB=20
PROFILE=baseline
V2_LAUNCH=scenario_3c_v2_full.launch

usage() {
  cat <<'EOF'
Usage: run_scenario_3c_acceptance.sh [options]

Options:
  --realtime         Use adaptive-phi real-time research profile.
  --gazebo-gui       Open Gazebo GUI as well as RViz (higher CPU load).
  --no-rviz          Do not open RViz.
  --keep-open        Keep all simulation processes open after analysis.
  --task-timeout N   Whole-task wall timeout in seconds (default: 900).
  -h, --help         Show this help.

The script refuses to start over an existing ROS/PX4/Gazebo session. It always
starts with auto_execute=false and enables V2 only after check_v2_live passes.
EOF
}

while (($#)); do
  case "$1" in
    --realtime)
      PROFILE=realtime
      V2_LAUNCH=scenario_3c_v2_realtime_full.launch
      ;;
    --gazebo-gui) GAZEBO_GUI=true ;;
    --no-rviz) LAUNCH_RVIZ=false ;;
    --keep-open) KEEP_OPEN=true ;;
    --task-timeout)
      shift
      [[ $# -gt 0 ]] || { echo "--task-timeout requires seconds" >&2; exit 2; }
      TASK_TIMEOUT="$1"
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

[[ "$TASK_TIMEOUT" =~ ^[0-9]+$ ]] || { echo "Invalid task timeout" >&2; exit 2; }

LAUNCH_PIDS=()
LAUNCH_NAMES=()
LOG_ROOT="/tmp/v2_one_click_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$LOG_ROOT"
RESULT="FAIL"

cleanup() {
  local code=$?
  set +e
  if [[ "$KEEP_OPEN" == true && "$RESULT" == PASS ]]; then
    echo "V2_KEEP_OPEN=true; press Ctrl+C in the launch processes when finished."
    trap - EXIT INT TERM
    wait
    return
  fi
  if ((${#LAUNCH_PIDS[@]})); then
    echo "Stopping simulation processes..."
    for pid in "${LAUNCH_PIDS[@]}"; do kill -INT "$pid" 2>/dev/null; done
    sleep 3
    for pid in "${LAUNCH_PIDS[@]}"; do
      kill -0 "$pid" 2>/dev/null && kill -TERM "$pid" 2>/dev/null
    done
    for pid in "${LAUNCH_PIDS[@]}"; do wait "$pid" 2>/dev/null; done
  fi
  echo "V2_LAUNCH_LOG_DIR=$LOG_ROOT"
  return "$code"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

stage() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

start_launch() {
  local name=$1
  shift
  "$@" >"$LOG_ROOT/${name}.log" 2>&1 &
  local pid=$!
  LAUNCH_PIDS+=("$pid")
  LAUNCH_NAMES+=("$name")
  echo "started $name pid=$pid log=$LOG_ROOT/${name}.log"
}

wait_until() {
  local timeout_s=$1 description=$2
  shift 2
  local deadline=$((SECONDS + timeout_s))
  while ((SECONDS < deadline)); do
    if "$@"; then return 0; fi
    sleep 1
  done
  echo "Timed out waiting for: $description" >&2
  return 1
}

master_ready() { rosparam get /rosversion >/dev/null 2>&1; }

mavros_connected() {
  local value
  value="$(timeout 3 rostopic echo -n 1 /iris_0/mavros/state 2>/dev/null || true)"
  grep -q 'connected: True' <<<"$value"
}

task_boxes_ready() {
  local count
  count="$(rosservice call /gazebo/get_world_properties '{}' 2>/dev/null | grep -c 'opp_' || true)"
  ((count >= 17))
}

takeoff_stable() {
  local line
  line="$(timeout 4 rostopic echo -p -n 1 /iris_0/mavros/local_position/pose 2>/dev/null | tail -n 1 || true)"
  [[ -n "$line" ]] || return 1
  python3 - "$line" <<'PY'
import sys
try:
    fields = sys.argv[1].split(',')
    x, y, z = map(float, fields[4:7])
except (ValueError, IndexError):
    raise SystemExit(1)
print("takeoff pose x={:.3f} y={:.3f} z={:.3f}".format(x, y, z))
raise SystemExit(0 if abs(x) <= 0.10 and abs(y) <= 0.10 and abs(z - 0.65) <= 0.08 else 1)
PY
}

px4_ready() {
  local value
  value="$(timeout 3 rostopic echo -n 1 /iris_0/mavros/state 2>/dev/null || true)"
  grep -q 'connected: True' <<<"$value" \
    && grep -q 'armed: True' <<<"$value" \
    && grep -q 'mode: "OFFBOARD"' <<<"$value"
}

available_kb="$(df -Pk / | awk 'NR==2 {print $4}')"
required_kb=$((MIN_FREE_GB * 1024 * 1024))
if ((available_kb < required_kb)); then
  echo "Refusing to start: less than ${MIN_FREE_GB} GiB free on /." >&2
  df -h / >&2
  exit 3
fi

if pgrep -x rosmaster >/dev/null || pgrep -x gzserver >/dev/null \
    || pgrep -x px4 >/dev/null || pgrep -x rviz >/dev/null; then
  echo "Refusing to start over an existing ROS/PX4/Gazebo/RViz session." >&2
  echo "Close the existing launch terminals first." >&2
  exit 3
fi

source /opt/ros/noetic/setup.bash
source "$PACKAGE_DIR/env_v2_deps.sh"
source "$WORKSPACE/devel/setup.bash"
export XTDRONE_DIR
export ROS_PACKAGE_PATH="/home/uav/PX4-Autopilot:/home/uav/PX4-Autopilot/Tools/sitl_gazebo:$XTDRONE_DIR/sitl_config:${ROS_PACKAGE_PATH:-}"
mkdir -p "$RUN_ROOT"

stage "1/9 Start clean Gazebo/PX4/MAVROS (gazebo_gui=$GAZEBO_GUI)"
start_launch gazebo_px4 roslaunch xtdrone_ego_tools indoor1_depth_xtdrone.launch \
  x:=0 y:=0 z:=1 gui:="$GAZEBO_GUI"
wait_until 90 "ROS master" master_ready
wait_until 120 "MAVROS connection" mavros_connected

stage "2/9 Start shared static cloud"
start_launch static_cloud roslaunch uav_continuum_opportunity_planner static_scene_cloud.launch
timeout 60 rostopic echo -n 1 /world/static_scene_cloud >/dev/null

stage "3/9 Start EGO without its trajectory server"
start_launch ego roslaunch xtdrone_ego_tools single_uav_rviz_goal.launch enable_traj_server:=false

stage "4/9 Start V2 profile=$PROFILE fail-closed and optional RViz"
RUN_MARKER="$LOG_ROOT/run_marker"
touch "$RUN_MARKER"
start_launch v2 roslaunch uav_continuum_whole_body_v2 "$V2_LAUNCH" \
  auto_execute:=false launch_rviz:="$LAUNCH_RVIZ"
wait_until 90 "17 Gazebo task objects" task_boxes_ready

stage "5/9 Start the single takeoff setpoint owner"
start_launch takeoff_hold roslaunch uav_continuum_whole_body_v2 safe_takeoff_hold.launch \
  x:=0 y:=0 z:=0.65
sleep 3

stage "6/9 Apply takeoff profile, enter OFFBOARD, arm, and stabilize"
roslaunch xtdrone_ego_tools px4_sitl_param_fix.launch profile:=takeoff \
  >"$LOG_ROOT/takeoff_params.log" 2>&1
rosservice call /iris_0/mavros/set_mode "{base_mode: 0, custom_mode: 'OFFBOARD'}"
rosservice call /iris_0/mavros/cmd/arming "{value: true}"
wait_until 30 "PX4 armed OFFBOARD" px4_ready
wait_until 90 "stable takeoff pose" takeoff_stable
roslaunch xtdrone_ego_tools px4_sitl_param_fix.launch profile:=tracking \
  >"$LOG_ROOT/tracking_params.log" 2>&1

stage "7/9 Start truth bridge last and perform live gate"
start_launch truth roslaunch xtdrone_ego_tools gazebo_truth_for_ego.launch
if ! rosrun uav_continuum_whole_body_v2 check_v2_live.py \
    --timeout "$PREFLIGHT_TIMEOUT" 2>&1 | tee "$LOG_ROOT/live_acceptance.log"; then
  echo "V2_ONE_CLICK_RESULT=FAIL_LIVE_GATE" >&2
  exit 4
fi

RUN_DIR="$(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d -newer "$RUN_MARKER" \
  -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)"
if [[ -z "$RUN_DIR" ]]; then
  echo "Cannot identify the current V2 run directory." >&2
  exit 4
fi
echo "RUN_DIR=$RUN_DIR"

stage "8/9 Atomically hand setpoint ownership to V2"
rostopic pub -1 /planner/v2/executor_enable std_msgs/Bool 'data: true' \
  >"$LOG_ROOT/enable.log" 2>&1 &
enable_pid=$!
sleep 0.2
rosnode kill /safe_takeoff_hold_v2 >/dev/null 2>&1 || true
wait "$enable_pid"

deadline=$((SECONDS + TASK_TIMEOUT))
last_report=0
while ((SECONDS < deadline)); do
  planner_status="$(timeout 5 rostopic echo -n 1 /planner/v2/status 2>/dev/null || true)"
  if grep -q 'TASK_COMPLETE' <<<"$planner_status"; then
    echo "$planner_status" | tr '\n' ' '
    echo
    break
  fi
  executor_status="$(timeout 5 rostopic echo -n 1 /planner/v2/executor_status 2>/dev/null || true)"
  if grep -q 'ABORT_' <<<"$executor_status" \
      || grep -q 'HOLDING_DISABLED' <<<"$executor_status"; then
    echo "$executor_status" >&2
    echo "V2_ONE_CLICK_RESULT=FAIL_EXECUTOR" >&2
    exit 5
  fi
  if ((SECONDS - last_report >= 15)); then
    last_report=$SECONDS
    pose="$(timeout 4 rostopic echo -p -n 1 /iris_0/mavros/local_position/pose 2>/dev/null | tail -n 1 || true)"
    echo "progress wall=${SECONDS}s pose_csv=$pose"
  fi
  sleep 2
done

if ((SECONDS >= deadline)); then
  echo "V2_ONE_CLICK_RESULT=FAIL_TASK_TIMEOUT" >&2
  exit 5
fi

stage "9/9 Audit measured execution log"
sleep 2
if ! rosrun uav_continuum_whole_body_v2 analyze_v2_run.py "$RUN_DIR" \
    2>&1 | tee "$LOG_ROOT/offline_acceptance.log"; then
  echo "V2_ONE_CLICK_RESULT=FAIL_OFFLINE_AUDIT" >&2
  exit 6
fi

RESULT=PASS
echo "V2_ONE_CLICK_RESULT=PASS"
echo "V2_PROFILE=$PROFILE"
echo "RUN_DIR=$RUN_DIR"
echo "ACCEPTANCE_REPORT=$RUN_DIR/acceptance_report.json"
