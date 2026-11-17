#!/usr/bin/env python3
"""Deterministic post-flight acceptance from logged measured whole-body states."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "src"))

from uav_continuum_whole_body_v2.environment import BoxWorld  # noqa: E402
from uav_continuum_whole_body_v2.experiment_metrics import (  # noqa: E402
    executed_translation_jerk_energy, partition_executed_translation_jerk,
)
from uav_continuum_whole_body_v2.geometry import (  # noqa: E402
    coupling_metrics, world_arm_points, world_body_points,
)
from uav_continuum_whole_body_v2.types import ArmModel, Limits, MassModel  # noqa: E402


def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def yaw_from_quaternion(value):
    x, y, z, w = (float(item) for item in value)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def percentile(values, q):
    return float(np.percentile(values, q)) if len(values) else None


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Validate one completed V2 execution log.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--oracle-file", type=Path,
        help="Hidden oracle_truth.yaml for a method-comparison scenario",
    )
    parser.add_argument("--body-limit", type=float, default=0.06)
    parser.add_argument("--arm-limit", type=float, default=0.02)
    parser.add_argument("--max-tracking-error", type=float, default=0.25)
    parser.add_argument("--max-continuous-hold", type=float, default=20.0)
    parser.add_argument("--max-consecutive-rejections", type=int, default=2)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) \
        if manifest_path.is_file() else {}
    method = str(manifest.get("method", "rolling_wb_v2"))
    events = read_jsonl(run_dir / "events.jsonl")
    tracking = read_jsonl(run_dir / "tracking.jsonl")
    commands = read_jsonl(run_dir / "commands.jsonl")
    command_kinematics = read_jsonl(run_dir / "command_kinematics.jsonl")
    states = [str(row.get("state", "")) for row in events]
    task_events = [row for row in events if row.get("state") == "TASK_COMPLETE"]
    tip_tolerance = float(manifest.get("tip_tolerance_m", 0.08))
    terminal_tip_error = (
        float(task_events[-1].get("terminal_tip_error", math.nan))
        if task_events else math.nan
    )
    abort_events = [row for row in events if "ABORT" in str(row.get("state", ""))]
    unsafe_events = [row for row in events if "UNSAFE" in str(row.get("state", ""))]

    if commands:
        execution_start = min(float(row["stamp"]) for row in commands)
    else:
        execution_start = math.inf
    if task_events:
        execution_end = max(float(row.get("logger_stamp", row.get("stamp", math.inf))) for row in task_events)
    elif abort_events:
        execution_end = max(float(row.get("logger_stamp", row.get("stamp", math.inf))) for row in abort_events)
    else:
        execution_end = math.inf
    measured = [
        row for row in tracking
        if execution_start <= float(row.get("stamp", -math.inf)) <= execution_end
        and row.get("arm_deg") is not None
    ]

    oracle_file = args.oracle_file or (
        PACKAGE / "generated" / "scenario_3c_unified_boxes.yaml"
    )
    oracle_hash = sha256(oracle_file)
    with oracle_file.open("r", encoding="utf-8") as handle:
        import yaml
        world_data = yaml.safe_load(handle) or {}
    world = BoxWorld.from_dicts(world_data.get("boxes", world_data.get("world_boxes", {}).get("boxes", [])))
    arm = ArmModel()
    limits = Limits()
    mass = MassModel()
    body_min, body_obstacle = math.inf, "none"
    arm_min, arm_obstacle = math.inf, "none"
    com_offsets = []
    gravity_torques = []
    for row in measured:
        position = row["position"]
        yaw = yaw_from_quaternion(row["orientation"])
        theta, phi = np.deg2rad(row["arm_deg"][:2])
        state = np.asarray([position[0], position[1], position[2], yaw, theta, phi], dtype=float)
        coupling = coupling_metrics(state, arm, mass)
        com_offsets.append(coupling["com_offset"])
        gravity_torques.append(coupling["gravity_torque"])
        body_value, body_name, _ = world.minimum_distance(
            world_body_points(state, limits, azimuth_samples=32, vertical_samples=5)
        )
        arm_value, arm_name, _ = world.minimum_distance(world_arm_points(state, arm))
        arm_value -= arm.radius
        if body_value < body_min:
            body_min, body_obstacle = float(body_value), body_name
        if arm_value < arm_min:
            arm_min, arm_obstacle = float(arm_value), arm_name

    planning_by_id = {}
    all_planning_attempts = []
    switch_ids = set()
    switch_events = {}
    for row in events:
        if row.get("state") == "SAFE_TRAJECTORY" and "trajectory_id" in row:
            planning_by_id[int(row["trajectory_id"])] = float(
                row.get("total_planning_time", row.get("solve_time", math.nan))
            )
        if row.get("state") in (
                "SAFE_TRAJECTORY", "REJECTED_KEEP_BUFFER", "METHOD_FAILED"):
            value = float(row.get(
                "total_planning_time", row.get("solve_time", math.nan)
            ))
            if np.isfinite(value):
                all_planning_attempts.append(value)
        if row.get("state") == "SWITCHED_ROLLING" and "trajectory_id" in row:
            switch_ids.add(int(row["trajectory_id"]))
            identifier = int(row["trajectory_id"])
            previous = switch_events.get(identifier)
            if (previous is None
                    or row.get("switch_continuity_diagnostics_version") is not None):
                switch_events[identifier] = row
    planning = np.asarray([value for value in planning_by_id.values() if np.isfinite(value)], dtype=float)
    all_planning = np.asarray(all_planning_attempts, dtype=float)

    hold_stamps = sorted(
        float(row.get("logger_stamp", row.get("stamp", math.nan)))
        for row in events
        if row.get("state") == "HOLD_FINAL"
        and execution_start <= float(row.get("logger_stamp", row.get("stamp", math.nan))) <= execution_end
    )
    hold_episode_ranges = []
    if hold_stamps:
        start = previous = hold_stamps[0]
        for stamp in hold_stamps[1:]:
            if stamp - previous > 2.0:
                hold_episode_ranges.append((start, previous))
                start = stamp
            previous = stamp
        hold_episode_ranges.append((start, previous))
    hold_episodes = [end - start for start, end in hold_episode_ranges]
    maximum_hold = max(hold_episodes, default=0.0)
    # The final target dwell is task semantics, not a missed rolling deadline.
    # Count only holds that ended well before TASK_COMPLETE as buffer underruns.
    nonterminal_hold_episodes = [
        end - start for start, end in hold_episode_ranges
        if not np.isfinite(execution_end) or end < execution_end - 5.0
    ]
    maximum_nonterminal_hold = max(nonterminal_hold_episodes, default=0.0)
    buffer_underruns = sum(
        duration > 0.5 for duration in nonterminal_hold_episodes
    )

    rejection_streak = maximum_rejection_streak = 0
    for row in events:
        state = str(row.get("state", ""))
        if state == "REJECTED_KEEP_BUFFER":
            rejection_streak += 1
            maximum_rejection_streak = max(maximum_rejection_streak, rejection_streak)
        elif state == "SAFE_TRAJECTORY":
            rejection_streak = 0

    tracking_errors = []
    if commands and measured:
        command_stamps = np.asarray([float(row["stamp"]) for row in commands])
        command_positions = np.asarray([row["position"] for row in commands], dtype=float)
        for row in measured:
            stamp = float(row["stamp"])
            index = int(np.searchsorted(command_stamps, stamp))
            candidates = [i for i in (index - 1, index) if 0 <= i < len(command_stamps)]
            nearest = min(candidates, key=lambda i: abs(command_stamps[i] - stamp))
            tracking_errors.append(float(np.linalg.norm(np.asarray(row["position"]) - command_positions[nearest])))
    tracking_errors = np.asarray(tracking_errors, dtype=float)

    failures = []
    expected_oracle_hash = manifest.get("oracle_sha256")
    if expected_oracle_hash and expected_oracle_hash != oracle_hash:
        failures.append("oracle_hash_mismatch")
    if not task_events:
        failures.append("missing_task_complete")
    elif not np.isfinite(terminal_tip_error) or terminal_tip_error > tip_tolerance:
        failures.append("terminal_tip_error")
    if abort_events:
        failures.append("abort_event")
    if unsafe_events:
        failures.append("unsafe_event")
    if not measured:
        failures.append("missing_measured_whole_body_states")
    if body_min < args.body_limit:
        failures.append("measured_body_clearance")
    if arm_min < args.arm_limit:
        failures.append("measured_arm_clearance")
    if method == "rolling_wb_v2" and not switch_ids:
        failures.append("missing_rolling_switch")
    tracking_max = float(np.max(tracking_errors)) if len(tracking_errors) else None
    if tracking_max is None:
        failures.append("missing_tracking_error")
    elif tracking_max > args.max_tracking_error:
        failures.append("tracking_error")
    if maximum_hold > args.max_continuous_hold:
        failures.append("continuous_hold_timeout")
    if maximum_rejection_streak > args.max_consecutive_rejections:
        failures.append("repeated_planning_rejection")

    arm_samples = np.asarray(
        [row["arm_deg"][:2] for row in measured], dtype=float
    ) if measured else np.empty((0, 2), dtype=float)

    executed_kinematics = [
        row for row in command_kinematics
        if execution_start <= float(row.get("stamp", -math.inf)) <= execution_end
    ]
    executed_jerk_energy = None
    if len(executed_kinematics) >= 5:
        kinematic_times = np.asarray([
            float(row["stamp"]) for row in executed_kinematics
        ])
        command_velocities = np.asarray([
            row["linear_velocity"] for row in executed_kinematics
        ], dtype=float)
        executed_jerk_energy = executed_translation_jerk_energy(
            kinematic_times, command_velocities,
            resample_hz=20.0, smoothing_window_s=0.5,
        )
    executed_commands = [
        row for row in commands
        if execution_start <= float(row.get("stamp", -math.inf)) <= execution_end
    ]
    command_path_length = 0.0
    if len(executed_commands) >= 2:
        command_positions = np.asarray([
            row["position"] for row in executed_commands
        ], dtype=float)
        command_path_length = float(np.sum(np.linalg.norm(
            np.diff(command_positions, axis=0), axis=1
        )))
    normalized_jerk = (
        executed_jerk_energy / command_path_length
        if executed_jerk_energy is not None and command_path_length > 1e-6
        else None
    )
    jerk_partition = None
    scheduled_starts = [
        float(row["actual_start"]) for row in events
        if row.get("state") == "START_SCHEDULED"
        and row.get("actual_start") is not None
    ]
    switch_stamps = sorted({
        float(row.get("actual_start", row.get("stamp")))
        for row in switch_events.values()
        if row.get("actual_start", row.get("stamp")) is not None
    })
    if (len(executed_kinematics) >= 5 and scheduled_starts
            and np.isfinite(execution_end)):
        jerk_partition = partition_executed_translation_jerk(
            kinematic_times, command_velocities,
            start_stamp=min(scheduled_starts),
            stop_stamp=execution_end,
            switch_stamps=switch_stamps,
            endpoint_radius_s=2.0, switch_radius_s=0.5,
            resample_hz=20.0, smoothing_window_s=0.5,
        )
        if jerk_partition is not None:
            jerk_partition.pop("profile")
            jerk_partition["normalized_components"] = {
                key: value / max(command_path_length, 1e-12)
                for key, value in jerk_partition["components"].items()
            }
            jerk_partition["normalized_total"] = (
                jerk_partition["total"] / max(command_path_length, 1e-12)
            )
            for item in jerk_partition.get("per_switch", []):
                item["normalized_energy"] = (
                    item["energy"] / max(command_path_length, 1e-12)
                )

    continuity_fields = (
        "position_gap_m", "yaw_gap_rad", "bending_gap_rad",
        "linear_velocity_gap_m_s", "yaw_rate_gap_rad_s",
        "bending_rate_gap_rad_s", "linear_acceleration_gap_m_s2",
    )
    continuity_rows = [
        row for row in switch_events.values()
        if row.get("switch_continuity_diagnostics_version") is not None
    ]
    switch_continuity = {
        "switch_count": len(switch_events),
        "diagnosed_switch_count": len(continuity_rows),
        "missing_diagnostics_count": len(switch_events) - len(continuity_rows),
        # These are the raw old-command/new-candidate alignment gaps.  A
        # validated bridge starts exactly at the old command and must not be
        # reported as if these candidate gaps were executed discontinuities.
        "candidate_alignment_metrics": {},
        "handoff_mode_counts": {
            "direct_c1": sum(
                row.get("switch_bridge", {}).get("status")
                == "NOT_REQUIRED_DIRECT_C1" for row in continuity_rows
            ),
            "validated_c2_bridge": sum(
                row.get("switch_bridge", {}).get("status") == "SAFE"
                for row in continuity_rows
            ),
        },
    }
    for field in continuity_fields:
        values = np.asarray([
            float(row[field]) for row in continuity_rows if row.get(field) is not None
        ], dtype=float)
        switch_continuity["candidate_alignment_metrics"][field] = {
            "count": len(values),
            "median": percentile(values, 50) if len(values) else None,
            "p95": percentile(values, 95) if len(values) else None,
            "max": float(np.max(values)) if len(values) else None,
        }

    report = {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "run_dir": str(run_dir),
        "method": method,
        "oracle_file": str(oracle_file.resolve()),
        "oracle_sha256": oracle_hash,
        "expected_oracle_sha256": expected_oracle_hash,
        "execution": {
            "task_complete": bool(task_events), "abort_events": len(abort_events),
            "unsafe_events": len(unsafe_events), "rolling_switches": len(switch_ids),
            "measured_samples": len(measured),
            "task_time_s": (
                float(execution_end - execution_start)
                if np.isfinite(execution_start) and np.isfinite(execution_end)
                else None
            ),
            "maximum_continuous_hold_s": maximum_hold,
            "maximum_nonterminal_hold_s": maximum_nonterminal_hold,
            "buffer_underrun_count": buffer_underruns,
            "deadline_miss_count": buffer_underruns,
            "planning_attempt_count": len(all_planning),
            "maximum_consecutive_rejections": maximum_rejection_streak,
            "continuous_hold_limit_s": args.max_continuous_hold,
            "consecutive_rejection_limit": args.max_consecutive_rejections,
        },
        "measured_morphology_deg": {
            "theta_min": float(np.min(arm_samples[:, 0])) if len(arm_samples) else None,
            "theta_max": float(np.max(arm_samples[:, 0])) if len(arm_samples) else None,
            "theta_std": float(np.std(arm_samples[:, 0])) if len(arm_samples) else None,
            "theta_total_variation": float(np.sum(np.abs(np.diff(arm_samples[:, 0])))) if len(arm_samples) else None,
            "phi_min": float(np.min(arm_samples[:, 1])) if len(arm_samples) else None,
            "phi_max": float(np.max(arm_samples[:, 1])) if len(arm_samples) else None,
            "phi_total_variation": float(np.sum(np.abs(np.diff(arm_samples[:, 1])))) if len(arm_samples) else None,
        },
        "measured_clearance": {
            "body_min": body_min if np.isfinite(body_min) else None,
            "body_obstacle": body_obstacle, "body_limit": args.body_limit,
            "arm_min": arm_min if np.isfinite(arm_min) else None,
            "arm_obstacle": arm_obstacle, "arm_limit": args.arm_limit,
        },
        "planning_time_s": {
            "count": len(planning), "median": percentile(planning, 50),
            "p95": percentile(planning, 95), "max": float(np.max(planning)) if len(planning) else None,
        },
        "all_planning_attempt_time_s": {
            "count": len(all_planning), "median": percentile(all_planning, 50),
            "p95": percentile(all_planning, 95),
            "max": float(np.max(all_planning)) if len(all_planning) else None,
            "sum": float(np.sum(all_planning)) if len(all_planning) else 0.0,
            "values": all_planning.tolist(),
        },
        "switch_continuity": switch_continuity,
        "measured_dynamics": {
            "com_offset_max_m": float(np.max(com_offsets)) if com_offsets else None,
            "gravity_torque_max_nm": float(np.max(gravity_torques))
            if gravity_torques else None,
        },
        "tracking_error_m": {
            "count": len(tracking_errors),
            "rmse": float(np.sqrt(np.mean(tracking_errors ** 2))) if len(tracking_errors) else None,
            "p95": percentile(tracking_errors, 95), "max": tracking_max,
            "limit": args.max_tracking_error,
        },
        "terminal_tip_error_m": terminal_tip_error if np.isfinite(terminal_tip_error) else None,
        "terminal_tip_tolerance_m": tip_tolerance,
        "trajectory_quality": {
            "executed_translation_jerk_energy": executed_jerk_energy,
            "executed_translation_jerk_energy_per_meter": normalized_jerk,
            "jerk_energy_sum": normalized_jerk,
            "command_path_length_m": command_path_length,
            "command_velocity_samples": len(executed_kinematics),
            "definition": "integral_squared_filtered_executed_command_jerk_per_commanded_meter",
            "analysis_resample_hz": 20.0,
            "savgol_window_s": 0.5,
            "savgol_polynomial_order": 3,
            "jerk_partition": jerk_partition,
        },
    }
    output = run_dir / "acceptance_report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
