#!/usr/bin/env python3
"""Relate planned flight-load proxies to PX4 closed-loop measurements."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import spearmanr


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def correlation(a, b):
    value, pvalue = spearmanr(a, b)
    return {"spearman_rho": float(value), "p_value": float(pvalue), "count": len(a)}


def analyze(run_dir):
    events = read_jsonl(run_dir / "events.jsonl")
    commands = read_jsonl(run_dir / "command_kinematics.jsonl")
    dynamics = read_jsonl(run_dir / "flight_dynamics.jsonl")
    starts = [row["actual_start"] for row in events if row.get("state") == "START_SCHEDULED"]
    stops = [row.get("logger_stamp", row.get("stamp")) for row in events if row.get("state") == "TASK_COMPLETE"]
    if not starts or not stops:
        raise RuntimeError("missing execution boundaries in %s" % run_dir)
    lower, upper = min(starts), max(stops)
    commands = [row for row in commands if lower <= row["stamp"] <= upper]
    imu = [row for row in dynamics if row.get("kind") == "imu" and lower <= row["stamp"] <= upper]
    targets = [row for row in dynamics if row.get("kind") == "attitude_target" and lower <= row["stamp"] <= upper]
    if len(commands) < 21 or len(imu) < 21 or len(targets) < 5:
        raise RuntimeError("insufficient synchronized dynamics samples in %s" % run_dir)
    source_t = np.asarray([row["stamp"] for row in commands], dtype=float)
    source_v = np.asarray([row["linear_velocity"] for row in commands], dtype=float)
    unique = np.r_[True, np.diff(source_t) > 1e-6]
    source_t, source_v = source_t[unique], source_v[unique]
    dt = 0.05
    times = np.arange(source_t[0], source_t[-1] + 0.5 * dt, dt)
    velocities = np.column_stack([np.interp(times, source_t, source_v[:, axis]) for axis in range(3)])
    acceleration = savgol_filter(velocities, 11, 3, deriv=1, delta=dt, axis=0, mode="interp")
    gravity = 9.80665
    force = acceleration + np.asarray([0.0, 0.0, gravity])
    planned_thrust_ratio = np.linalg.norm(force, axis=1) / gravity
    planned_tilt = np.arctan2(np.linalg.norm(force[:, :2], axis=1), np.maximum(force[:, 2], 1e-6))

    imu_t = np.asarray([row["stamp"] for row in imu], dtype=float)
    imu_roll = np.asarray([row["roll_rad"] for row in imu], dtype=float)
    imu_pitch = np.asarray([row["pitch_rad"] for row in imu], dtype=float)
    imu_specific_force = np.asarray([
        np.linalg.norm(row["linear_acceleration_m_s2"]) / gravity for row in imu
    ])
    actual_tilt = np.hypot(
        np.interp(times, imu_t, imu_roll), np.interp(times, imu_t, imu_pitch)
    )
    actual_force_ratio = np.interp(times, imu_t, imu_specific_force)
    target_t = np.asarray([row["stamp"] for row in targets], dtype=float)
    target_tilt_raw = np.hypot(
        [row["roll_rad"] for row in targets], [row["pitch_rad"] for row in targets]
    )
    target_tilt = np.interp(times, target_t, target_tilt_raw)
    return {
        "run_dir": str(run_dir.resolve()), "samples": len(times),
        "planned_tilt_vs_px4_target": correlation(planned_tilt, target_tilt),
        "planned_tilt_vs_imu_actual": correlation(planned_tilt, actual_tilt),
        "px4_target_vs_imu_actual_tilt": correlation(target_tilt, actual_tilt),
        "planned_thrust_ratio_vs_imu_specific_force": correlation(
            planned_thrust_ratio, actual_force_ratio
        ),
        "tilt_rad": {
            "planned_max": float(np.max(planned_tilt)),
            "px4_target_max": float(np.max(target_tilt)),
            "imu_actual_max": float(np.max(actual_tilt)),
            "target_actual_rmse": float(np.sqrt(np.mean((target_tilt - actual_tilt) ** 2))),
        },
        "specific_force_ratio": {
            "planned_min": float(np.min(planned_thrust_ratio)),
            "planned_max": float(np.max(planned_thrust_ratio)),
            "imu_min": float(np.min(actual_force_ratio)),
            "imu_max": float(np.max(actual_force_ratio)),
        },
        "direct_motor_load_available": False,
        "motor_load_note": (
            "MAVROS target actuator channel 3 remained near 0.001 and is not treated as motor thrust; "
            "the report therefore uses IMU specific-force magnitude as the realized collective-load proxy."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runs = [analyze(path) for path in args.run_dirs]
    report = {"schema": "v21_dynamics_proxy_audit_v1", "runs": runs}
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
