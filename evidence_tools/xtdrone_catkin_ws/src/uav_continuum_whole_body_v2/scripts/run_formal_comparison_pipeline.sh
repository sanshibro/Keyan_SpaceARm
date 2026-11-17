#!/usr/bin/env bash
# Resume-safe one-command schema-v6 frozen-map method comparison.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE="$(cd "$PACKAGE_DIR/../.." && pwd)"
SCENARIO_DIR=""
OUTPUT_DIR=""
METHOD_TIME_BUDGET=15
PREFLIGHT_TIMEOUT=60
TASK_TIMEOUT=600
GAZEBO_GUI=false
PREFLIGHT_ONLY=false
METHODS=(fixed_morphology aet_sequential_minjerk informed_rrtstar_wb global_batch_wb rolling_wb_v2)

usage() {
  cat <<'EOF'
Usage: run_formal_comparison_pipeline.sh --scenario-dir DIR --output-dir DIR [options]

Options:
  --method-time-budget S  Common planning budget (default 15).
  --preflight-timeout S   Live gate timeout (default 60).
  --task-timeout S        Per-trial task timeout (default 600).
  --gazebo-gui            Show Gazebo.
  --preflight-only        Freeze/verify method order without starting ROS.

Completed certified maps and method trials are reused. A method failure is
retained as data; missing evidence or an ineligible aggregate fails the command.
EOF
}

while (($#)); do
  case "$1" in
    --scenario-dir) shift; SCENARIO_DIR="${1:-}" ;;
    --output-dir) shift; OUTPUT_DIR="${1:-}" ;;
    --method-time-budget) shift; METHOD_TIME_BUDGET="${1:-}" ;;
    --preflight-timeout) shift; PREFLIGHT_TIMEOUT="${1:-}" ;;
    --task-timeout) shift; TASK_TIMEOUT="${1:-}" ;;
    --gazebo-gui) GAZEBO_GUI=true ;;
    --preflight-only) PREFLIGHT_ONLY=true ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done
[[ -n "$SCENARIO_DIR" && -n "$OUTPUT_DIR" ]] || { usage >&2; exit 2; }
SCENARIO_DIR="$(realpath "$SCENARIO_DIR")"
OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"
TASK="$SCENARIO_DIR/task.yaml"
ORACLE="$SCENARIO_DIR/oracle_truth.yaml"
[[ -f "$TASK" && -f "$ORACLE" ]] || { echo "Missing task/oracle" >&2; exit 2; }
PROJECT_ROOT="$(cd "$WORKSPACE/.." && pwd)"
RELEVANT_DIRTY="$(git -C "$PROJECT_ROOT" status --porcelain \
  --untracked-files=normal -- \
  xtdrone_catkin_ws/src/uav_continuum_whole_body_v2 \
  xtdrone_catkin_ws/src/xtdrone_ego_tools \
  xtdrone_catkin_ws/src/uav_continuum_opportunity_planner/scripts/arm_shape_sim_node.py \
  xtdrone_catkin_ws/src/uav_continuum_opportunity_planner/scripts/arm_kinematics_node.py)"
[[ -z "$RELEVANT_DIRTY" ]] || {
  echo "FORMAL_REJECTED: relevant source tree is dirty" >&2
  echo "$RELEVANT_DIRTY" >&2
  exit 2
}
mkdir -p "$OUTPUT_DIR"

# A fixed method order would confound all trials with simulator warm-up and
# long-run time drift. Rotate the same five methods with a balanced Latin-cycle
# index derived from frozen family/difficulty/seed fields. With ten seeds per
# difficulty, every method occupies every order position equally often.
TASK_SHA256="$(sha256sum "$TASK" | awk '{print $1}')"
ORDER_FILE="$OUTPUT_DIR/method_order.json"
if [[ -s "$ORDER_FILE" ]]; then
  ORDER_TEXT="$(python3 - "$ORDER_FILE" "$TASK_SHA256" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("task_sha256") != sys.argv[2]:
    raise SystemExit("Existing output has a different frozen task hash")
methods = payload.get("methods", [])
allowed = {"fixed_morphology", "aet_sequential_minjerk", "informed_rrtstar_wb",
           "global_batch_wb", "rolling_wb_v2"}
if len(methods) != 5 or len(set(methods)) != 5 or set(methods) != allowed:
    raise SystemExit("Existing output has an invalid frozen method order")
for method in methods:
    print(method)
PY
  )"
  mapfile -t METHODS <<<"$ORDER_TEXT"
else
  ORDER_POLICY="balanced_family_difficulty_seed_latin_rotation_v1"
  if find "$OUTPUT_DIR/closed_loop_runs" -name trial_result.json -print -quit \
      2>/dev/null | grep -q .; then
    # Pilot directories produced before order recording used the declaration
    # order above. Preserve that fact rather than retroactively relabeling it.
    ORDER_POLICY="legacy_fixed_order_v1"
  else
    METHOD_OFFSET="$(python3 - "$TASK" <<'PY'
import sys, yaml
task = yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}
families = {"passage_sweep": 0, "multi_passage": 1, "clutter_reach": 2}
difficulties = {"easy": 0, "medium": 1, "hard": 2}
family = task.get("family")
difficulty = task.get("difficulty")
seed = task.get("seed")
if family not in families or difficulty not in difficulties or not isinstance(seed, int):
    raise SystemExit("Task lacks registered family/difficulty/integer seed")
print((seed + difficulties[difficulty] + 3 * families[family]) % 5)
PY
    )"
    ROTATED_METHODS=()
    for ((index = 0; index < ${#METHODS[@]}; ++index)); do
      ROTATED_METHODS+=("${METHODS[$(((index + METHOD_OFFSET) % ${#METHODS[@]}))]}")
    done
    METHODS=("${ROTATED_METHODS[@]}")
  fi
  python3 - "$ORDER_FILE" "$TASK_SHA256" "$ORDER_POLICY" "${METHODS[@]}" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
payload = {
    "order_policy": sys.argv[3],
    "task_sha256": sys.argv[2],
    "methods": sys.argv[4:],
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
PY
fi
[[ ${#METHODS[@]} -eq 5 ]] || {
  echo "Frozen method order must contain exactly five methods" >&2
  exit 2
}
echo "FROZEN_METHOD_ORDER=${METHODS[*]}"
[[ "$PREFLIGHT_ONLY" == false ]] || {
  echo "FORMAL_PIPELINE_PREFLIGHT=PASS"
  exit 0
}

source /opt/ros/noetic/setup.bash
source "$PACKAGE_DIR/env_v2_deps.sh"
source "$WORKSPACE/devel/setup.bash"

MANIFEST="$OUTPUT_DIR/frozen_survey_manifest.json"
FRONT="$OUTPUT_DIR/front_survey/certified_sensor_survey.npz"
DOWN="$OUTPUT_DIR/down_survey/certified_sensor_survey.npz"
FUSED="$OUTPUT_DIR/fused_candidate.npz"
AUDITED="$OUTPUT_DIR/audited_frozen_map.npz"
RUNS="$OUTPUT_DIR/closed_loop_runs"
AGGREGATE="$OUTPUT_DIR/aggregate"
SURVEY_PROTOCOL="task_endpoint_scan_survey_v6_explicit_flush"

if [[ ! -s "$MANIFEST" ]]; then
  echo "[1/6] Generate method-independent frozen survey manifest"
  rosrun uav_continuum_whole_body_v2 generate_independent_survey_manifest.py \
    "$TASK" "$MANIFEST"
else
  python3 - "$MANIFEST" "$SURVEY_PROTOCOL" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("protocol_id") != sys.argv[2]:
    raise SystemExit(
        "Existing output uses an obsolete survey protocol; choose a new output directory"
    )
PY
  echo "[1/6] Reuse frozen survey manifest"
fi

for mode in front down; do
  target="$OUTPUT_DIR/${mode}_survey/certified_sensor_survey.npz"
  if [[ ! -s "$target" ]]; then
    echo "[2/6] Record independent $mode RGB-D survey"
    command=(rosrun uav_continuum_whole_body_v2 run_independent_map_survey.sh
      --scenario-dir "$SCENARIO_DIR" --manifest "$MANIFEST"
      --output-dir "$OUTPUT_DIR/${mode}_survey" --sensor-mode "$mode"
      --task-timeout "$TASK_TIMEOUT")
    [[ "$GAZEBO_GUI" == true ]] && command+=(--gazebo-gui)
    "${command[@]}"
  else
    python3 - "$target" "$MANIFEST" "$SURVEY_PROTOCOL" <<'PY'
import hashlib, sys
from uav_continuum_whole_body_v2.mapping import MapSnapshot
snapshot = MapSnapshot.load(sys.argv[1])
digest = hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest()
if snapshot.metadata.get("survey_protocol") != sys.argv[3]:
    raise SystemExit("Existing survey map uses an obsolete survey protocol")
if snapshot.metadata.get("survey_protocol_manifest_sha256") != digest:
    raise SystemExit("Existing survey map does not match the frozen manifest")
PY
    echo "[2/6] Reuse certified $mode survey"
  fi
done

if [[ ! -s "$AUDITED" ]]; then
  echo "[3/6] Conservatively fuse and schema-v5 audit"
  rosrun uav_continuum_whole_body_v2 fuse_slam_maps.py \
    "$FRONT" "$DOWN" --output "$FUSED" --report "$OUTPUT_DIR/fusion_report.json" \
    --sensor forward_rgbd --sensor downward_rgbd \
    --comparison-independent-survey --survey-protocol "$SURVEY_PROTOCOL" \
    --survey-manifest "$MANIFEST" --observation-support-voxels 1 \
    --corridor-xy-padding 1.2 --corridor-z-min 0.2 --corridor-z-max 2.5
  rosrun uav_continuum_whole_body_v2 audit_slam_map.py \
    "$FUSED" "$ORACLE" "$TASK" --report "$OUTPUT_DIR/map_audit_report.json" \
    --audited-map-output "$AUDITED"
else
  python3 - "$AUDITED" <<'PY'
import sys
from uav_continuum_whole_body_v2.mapping import MapSnapshot
snapshot = MapSnapshot.load(sys.argv[1])
if int(snapshot.metadata.get("map_audit_schema_version", 0)) < 5:
    raise SystemExit(
        "Existing output uses an obsolete map audit; choose a new output directory"
    )
PY
  echo "[3/6] Reuse schema-v5 audited map"
fi

mkdir -p "$RUNS"
has_trial() {
  python3 - "$RUNS" "$1" <<'PY'
import json, pathlib, sys
root, method = pathlib.Path(sys.argv[1]), sys.argv[2]
for path in root.rglob("trial_result.json"):
    try: payload = json.loads(path.read_text())
    except Exception: continue
    if (payload.get("method") == method
            and payload.get("result") != "SYSTEM_ERROR"):
        raise SystemExit(0)
raise SystemExit(1)
PY
}

echo "[4/6] Run or resume five paired closed-loop methods"
echo "      deterministic order: ${METHODS[*]}"
for method in "${METHODS[@]}"; do
  if has_trial "$method"; then
    echo "      SKIP existing $method"
    continue
  fi
  command=(rosrun uav_continuum_whole_body_v2 run_method_comparison_closed_loop.sh
    --method "$method" --scenario-dir "$SCENARIO_DIR" --map "$AUDITED"
    --method-time-budget "$METHOD_TIME_BUDGET"
    --preflight-timeout "$PREFLIGHT_TIMEOUT" --task-timeout "$TASK_TIMEOUT"
    --output-root "$RUNS" --formal)
  [[ "$GAZEBO_GUI" == true ]] && command+=(--gazebo-gui)
  if "${command[@]}"; then
    echo "      $method completed successfully"
  else
    echo "      $method failed and was retained as a comparison sample"
  fi
  has_trial "$method" || { echo "Missing trial evidence for $method" >&2; exit 5; }
done

echo "[5/6] Aggregate complete paired results"
rosrun uav_continuum_whole_body_v2 aggregate_method_comparison.py \
  "$RUNS" --output "$AGGREGATE"

echo "[6/6] Verify paper evidence eligibility"
python3 - "$AGGREGATE" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
eligibility = json.loads((root / "paper_eligibility.json").read_text())
completeness = json.loads((root / "completeness.json").read_text())
if not eligibility.get("eligible"):
    raise SystemExit("aggregate is not paper eligible")
if completeness.get("duplicates") or completeness.get("incomplete"):
    raise SystemExit("paired aggregate is incomplete or duplicated")
print("FORMAL_COMPARISON_PIPELINE=PASS")
print("MAIN_TABLE=" + str(root / "main_table.csv"))
PY
