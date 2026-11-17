#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$TASK_ROOT/.mujoco_deps${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
stage="${1:-visual-grasp}"
if (($#)); then shift; fi
case "$stage" in
  stage1) exec python3 "$TASK_ROOT/mujoco_closed_loop/stage1_preview.py" "$@" ;;
  visual-grasp) exec python3 "$TASK_ROOT/mujoco_closed_loop/visual_grasp_development.py" "$@" ;;
  navigation) exec python3 "$TASK_ROOT/mujoco_closed_loop/navigation_development.py" "$@" ;;
  mission)
    task_map="${1:?mission requires a sensor presurvey map}"
    shift
    exec python3 "$TASK_ROOT/mujoco_closed_loop/visual_grasp_development.py" --mission-map "$task_map" --seconds 180 "$@"
    ;;
  mission-slam) exec bash "$TASK_ROOT/mujoco_closed_loop/run_fast_lio_mission.sh" "$@" ;;
  survey) exec python3 "$TASK_ROOT/mujoco_closed_loop/yaw_survey_audit.py" "$@" ;;
  *) echo "Usage: $0 {stage1|visual-grasp|navigation MAP|mission MAP|mission-slam|survey} [options]" >&2; exit 2 ;;
esac
