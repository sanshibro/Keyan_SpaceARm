#!/usr/bin/env bash
set -e
cd /home/uav/paper_revision_20261117
source /opt/ros/noetic/setup.bash
export PYTHONPATH="$PWD/revision/scripts:$HOME/桌面/uav_arm_1/.mujoco_deps:$PWD/mujoco_closed_loop:$PWD/xtdrone_catkin_ws/src/uav_continuum_whole_body_v2/src:${PYTHONPATH:-}"
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
gate_out="$PWD/revision/data/round_gate_feedback"
exec >> "$gate_out/campaign.log" 2>&1
/usr/bin/python3 -u revision/scripts/gate_recover_boot.py
/usr/bin/python3 -u revision/scripts/gate_feedback_campaign.py
/usr/bin/python3 -c 'import json, pathlib; p=pathlib.Path("revision/data/round_gate_feedback"); assert json.loads((p/"campaign-progress.json").read_bytes())["completed"]==60; (p/"campaign.finished").touch()'
