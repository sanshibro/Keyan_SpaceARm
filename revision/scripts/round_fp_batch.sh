#!/usr/bin/env bash
set -eo pipefail
round_dir=${1:?Supply a new output root}
mkdir -p "$round_dir"
# One loopback PX4 instance exclusively owns the F bridge ports.
exec 9>"${TMPDIR:-/tmp}/super-qingyu-sitl-instance-6.lock"
flock -n 9
source /opt/ros/noetic/setup.bash
export PYTHONPATH="$HOME/桌面/uav_arm_1/.mujoco_deps:$PWD/mujoco_closed_loop:$PWD/xtdrone_catkin_ws/src/uav_continuum_whole_body_v2/src:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1
while read -r seed variant; do
    label="$seed-$variant"
    echo "START $label $(date --iso-8601=seconds)"
    python3 revision/scripts/round_fp_run.py "$round_dir/$label" "$seed" "$variant"
    echo "FINISHED $label $(date --iso-8601=seconds)"
done < <(python3 -c 'import json; p=json.load(open("revision/protocols/round_fp.json")); print("\n".join(f"{s} {v}" for s, vs in p["seed_order"].items() for v in vs))')
