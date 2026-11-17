#!/usr/bin/env python3
"""Generate a fixed survey route using task endpoints, never obstacle geometry."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--step", type=float, default=0.40)
    parser.add_argument("--takeoff-altitude", type=float, default=0.65)
    parser.add_argument("--speed", type=float, default=0.10)
    parser.add_argument("--start-scan-radius", type=float, default=0.18)
    args = parser.parse_args()
    task = yaml.safe_load(args.task.read_text(encoding="utf-8")) or {}
    start = np.asarray(task["start_state"][:4], dtype=float)
    goal = np.asarray(task["preferred_standoff"][:4], dtype=float)
    distance = float(np.linalg.norm(goal[:3] - start[:3]))
    count = max(2, int(math.ceil(distance / max(0.1, args.step))) + 1)
    poses = [[float(start[0]), float(start[1]), float(args.takeoff_altitude), float(start[3])]]
    poses.append([float(value) for value in start])
    # A measured start circle certifies the normal closed-loop tracking tube;
    # it is fixed from the task start alone and has no obstacle/map access.
    for angle in np.linspace(0.0, 2.0 * math.pi, 9)[:-1]:
        poses.append([
            float(start[0] + args.start_scan_radius * math.cos(angle)),
            float(start[1] + args.start_scan_radius * math.sin(angle)),
            float(start[2]), float(start[3]),
        ])
    poses.append([float(value) for value in start])
    for fraction in np.linspace(0.0, 1.0, count)[1:]:
        pose = (1.0 - fraction) * start + fraction * goal
        poses.append([float(value) for value in pose])
    manifest = {
        "schema_version": 1,
        "protocol_id": "task_endpoint_scan_survey_v6_explicit_flush",
        "protocol_type": "method_independent_waypoint_survey",
        "frame_id": str(task.get("frame_id", "world")),
        "scenario": str(task.get("name", args.task.parent.name)),
        "waypoint_source": "task_endpoints_plus_fixed_start_circle_no_obstacles",
        "compared_planner_access": False,
        "oracle_geometry_access": False,
        "nominal_speed_mps": float(args.speed),
        "angular_ray_policy": "retain_complete_160x120_rgbd_grid",
        "maximum_translation_between_map_updates_m": 0.05,
        "map_snapshot_policy": "incremental_fusion_low_rate_atomic_snapshot",
        "terminal_snapshot_flush": "required_ros_service_acknowledgement",
        "waypoint_tolerance_m": 0.10,
        "waypoints": [
            {"pose": pose, "dwell_s": 0.6 if index else 1.5}
            for index, pose in enumerate(poses)
        ],
        "terminal_scan": {
            "duration_s": 24.0, "turns": 1.0,
            "radius_m": 0.15, "z_offset_m": 0.10,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
