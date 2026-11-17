#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$TASK_ROOT/.mujoco_deps${PYTHONPATH:+:$PYTHONPATH}"
exec python3 "$TASK_ROOT/mujoco_closed_loop/scan.py" "$@"
