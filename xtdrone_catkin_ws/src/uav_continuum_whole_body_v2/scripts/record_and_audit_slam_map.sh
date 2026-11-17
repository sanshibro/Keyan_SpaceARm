#!/usr/bin/env bash
# Acquire one frozen RGB-D SLAM map with a controlled survey, then audit it.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE="$(cd "$PACKAGE_DIR/../.." && pwd)"

SCENARIO_DIR=""
OUTPUT_DIR=""
SURVEY_MAP=""
METHOD="rolling_wb_v2"
METHOD_TIME_BUDGET=15
GAZEBO_GUI=false
MAP_LOCALIZATION_BACKEND="px4_ekf"
MAP_SENSOR_MODE="dual"
FOOTPRINT_CLEAR_RADIUS_XY="0.28"

usage() {
  cat <<'EOF'
Usage: record_and_audit_slam_map.sh --scenario-dir DIR --output-dir DIR [options]

Options:
  --survey-map FILE       Navigation map used only by the controlled survey.
                          Default: SCENARIO_DIR/development_map.npz.
  --method NAME           Survey controller (default rolling_wb_v2).
  --method-time-budget S  Survey planner budget (default 15).
  --map-localization NAME  px4_ekf (default) or rtabmap_icp.
  --map-sensor-mode NAME   front, down, or dual (default dual).
  --footprint-clear-radius M  Physical footprint clearing radius (default 0.28).
  --gazebo-gui            Show Gazebo during acquisition.

The survey navigation map is never copied into the recorded snapshot. The
output map contains only RGB-D returns registered by RTAB-Map poses. The hidden
oracle is read only after flight by audit_slam_map.py.
EOF
}

while (($#)); do
  case "$1" in
    --scenario-dir) shift; SCENARIO_DIR="${1:-}" ;;
    --output-dir) shift; OUTPUT_DIR="${1:-}" ;;
    --survey-map) shift; SURVEY_MAP="${1:-}" ;;
    --method) shift; METHOD="${1:-}" ;;
    --method-time-budget) shift; METHOD_TIME_BUDGET="${1:-}" ;;
    --map-localization) shift; MAP_LOCALIZATION_BACKEND="${1:-}" ;;
    --map-sensor-mode) shift; MAP_SENSOR_MODE="${1:-}" ;;
    --footprint-clear-radius) shift; FOOTPRINT_CLEAR_RADIUS_XY="${1:-}" ;;
    --gazebo-gui) GAZEBO_GUI=true ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

[[ -n "$SCENARIO_DIR" && -n "$OUTPUT_DIR" ]] || { usage >&2; exit 2; }
SCENARIO_DIR="$(realpath "$SCENARIO_DIR")"
OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"
[[ -n "$SURVEY_MAP" ]] || SURVEY_MAP="$SCENARIO_DIR/development_map.npz"
SURVEY_MAP="$(realpath "$SURVEY_MAP")"
mkdir -p "$OUTPUT_DIR"

RAW_MAP="$OUTPUT_DIR/recorded_slam_map_raw.npz"
AUDITED_MAP="$OUTPUT_DIR/audited_frozen_slam_map.npz"
DATABASE="$OUTPUT_DIR/rtabmap.db"
REPORT="$OUTPUT_DIR/map_audit_report.json"
RUNS="$OUTPUT_DIR/survey_runs"
rm -f "$RAW_MAP" "$AUDITED_MAP" "$DATABASE" "$REPORT"

command=("$SCRIPT_DIR/run_method_comparison_closed_loop.sh"
  --method "$METHOD" --scenario-dir "$SCENARIO_DIR" --map "$SURVEY_MAP"
  --method-time-budget "$METHOD_TIME_BUDGET" --task-timeout 900
  --output-root "$RUNS" --record-slam-map "$RAW_MAP" --slam-database "$DATABASE"
  --map-localization "$MAP_LOCALIZATION_BACKEND"
  --map-sensor-mode "$MAP_SENSOR_MODE"
  --footprint-clear-radius "$FOOTPRINT_CLEAR_RADIUS_XY")
[[ "$GAZEBO_GUI" == true ]] && command+=(--gazebo-gui)
"${command[@]}"

[[ -s "$RAW_MAP" ]] || { echo "MAP_NOT_READY: survey produced no snapshot" >&2; exit 4; }
source /opt/ros/noetic/setup.bash
source "$PACKAGE_DIR/env_v2_deps.sh"
source "$WORKSPACE/devel/setup.bash"
rosrun uav_continuum_whole_body_v2 audit_slam_map.py \
  "$RAW_MAP" "$SCENARIO_DIR/oracle_truth.yaml" "$SCENARIO_DIR/task.yaml" \
  --report "$REPORT" --audited-map-output "$AUDITED_MAP"

echo "SLAM_MAP_ACQUISITION_RESULT=PASS"
echo "RAW_MAP=$RAW_MAP"
echo "AUDITED_MAP=$AUDITED_MAP"
echo "AUDIT_REPORT=$REPORT"
