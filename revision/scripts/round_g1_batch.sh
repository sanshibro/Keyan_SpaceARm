#!/usr/bin/env bash
set -eo pipefail
exec 9>"${TMPDIR:-/tmp}/super-qingyu-sitl-instance-6.lock"
flock -n 9
source /opt/ros/noetic/setup.bash
export PYTHONPATH="$HOME/桌面/uav_arm_1/.mujoco_deps:$PWD/mujoco_closed_loop:$PWD/xtdrone_catkin_ws/src/uav_continuum_whole_body_v2/src:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1
base=${ROUND_G1_OUTPUT:-revision/data/round_g1}
if [ ! -s "$base/plans.json" ]; then
    echo "Generate plans in this ROUND_G1_OUTPUT first." >&2
    exit 2
fi
failed=0
while read -r seed mode; do
    label="$seed-$mode"
    if [ -d "$base/runs/$label" ]; then
        echo "PRESERVED $label (existing; inspect its log or use a fresh ROUND_G1_OUTPUT)"
        continue
    fi
    echo "START $label $(date --iso-8601=seconds)"
    if python3 revision/scripts/round_g1_run.py "$seed" "$mode"; then
        echo "FINISHED $label $(date --iso-8601=seconds)"
    else
        echo "FAILED $label $(date --iso-8601=seconds)"
        failed=1
    fi
done < <(python3 -c 'import json; p=json.load(open("revision/protocols/round_g1.json")); print("241119 wide"); print("\n".join(f"{s} {v}" for s,vs in p["seed_order"].items() for v in vs))')
exit "$failed"
