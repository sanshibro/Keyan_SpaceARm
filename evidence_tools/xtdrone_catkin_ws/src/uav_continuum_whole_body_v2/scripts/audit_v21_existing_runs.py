#!/usr/bin/env python3
"""Audit frozen V2 runs for jerk attribution and cumulative planning cost."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from uav_continuum_whole_body_v2.experiment_metrics import (
    partition_executed_translation_jerk,
)


METHOD = "rolling_wb_v2"


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_targets(path):
    targets = {}
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("method") != METHOD:
                continue
            targets[row["scenario"]] = {
                "scenario_family": row["scenario_family"],
                "scenario_difficulty": row["scenario_difficulty"],
                "random_seed": int(row["random_seed"]),
                "jerk_energy": float(row["jerk_energy"]),
            }
    return targets


def candidate_runs(search_root, targets, commit):
    candidates = {scenario: [] for scenario in targets}
    for manifest_path in Path(search_root).rglob("manifest.json"):
        try:
            manifest = read_json(manifest_path)
        except (OSError, ValueError):
            continue
        scenario = manifest.get("scenario_name")
        if scenario not in targets or manifest.get("method") != METHOD:
            continue
        if commit and manifest.get("commit") != commit:
            continue
        run_dir = manifest_path.parent
        acceptance_path = run_dir / "acceptance_report.json"
        command_path = run_dir / "command_kinematics.jsonl"
        event_path = run_dir / "events.jsonl"
        if not acceptance_path.is_file() or not command_path.is_file() or not event_path.is_file():
            continue
        try:
            acceptance = read_json(acceptance_path)
            observed = float(
                acceptance["trajectory_quality"]["executed_translation_jerk_energy_per_meter"]
            )
        except (KeyError, TypeError, ValueError, OSError):
            continue
        error = abs(observed - targets[scenario]["jerk_energy"])
        candidates[scenario].append((error, str(run_dir), acceptance, manifest))
    selected = {}
    for scenario, values in candidates.items():
        if not values:
            continue
        values.sort(key=lambda item: (item[0], len(item[1]), item[1]))
        selected[scenario] = values[0]
    return selected


def event_anchors(rows):
    starts = [
        float(row["actual_start"])
        for row in rows
        if row.get("state") == "START_SCHEDULED" and row.get("actual_start") is not None
    ]
    stops = [
        float(row["stamp"])
        for row in rows
        if row.get("state") == "TRAJECTORY_COMPLETE" and row.get("stamp") is not None
    ]
    if not stops:
        stops = [
            float(row["stamp"])
            for row in rows
            if row.get("state") == "TASK_COMPLETE" and row.get("stamp") is not None
        ]
    switches = {}
    for row in rows:
        if row.get("state") != "SWITCHED_ROLLING":
            continue
        trajectory_id = row.get("trajectory_id")
        stamp = row.get("actual_start", row.get("stamp"))
        if trajectory_id is None or stamp is None:
            continue
        switches[int(trajectory_id)] = min(float(stamp), switches.get(int(trajectory_id), float("inf")))
    if not starts or not stops:
        raise ValueError("missing START_SCHEDULED or TRAJECTORY_COMPLETE/TASK_COMPLETE")
    task_ends = [
        float(row.get("logger_stamp", row.get("stamp")))
        for row in rows
        if row.get("state") == "TASK_COMPLETE"
        and row.get("logger_stamp", row.get("stamp")) is not None
    ]
    if not task_ends:
        task_ends = list(stops)
    return (
        min(starts), min(stops), max(task_ends),
        [switches[key] for key in sorted(switches)],
    )


def percentile(values, q):
    return None if not values else float(np.percentile(np.asarray(values, dtype=float), q))


def summarize(records, radii):
    summary = {
        "run_count": len(records),
        "scenario_families": sorted({row["scenario_family"] for row in records}),
        "scenario_difficulties": sorted({row["scenario_difficulty"] for row in records}),
        "cumulative_planning_time_s": {},
        "planning_attempt_count": {},
        "jerk_partition_by_switch_radius_s": {},
    }
    planning = [row["cumulative_planning_time_s"] for row in records]
    attempts = [row["planning_attempt_count"] for row in records]
    summary["cumulative_planning_time_s"] = {
        "median": percentile(planning, 50), "p95": percentile(planning, 95),
        "max": max(planning) if planning else None,
    }
    summary["planning_attempt_count"] = {
        "median": percentile(attempts, 50), "p95": percentile(attempts, 95),
        "max": max(attempts) if attempts else None,
    }
    for radius in radii:
        key = "%.3f" % radius
        radius_rows = [row["jerk_partitions"][key] for row in records]
        summary["jerk_partition_by_switch_radius_s"][key] = {
            name: {
                "fraction_median": percentile(
                    [entry["fractions"][name] for entry in radius_rows], 50),
                "fraction_p95": percentile(
                    [entry["fractions"][name] for entry in radius_rows], 95),
                "normalized_energy_median": percentile(
                    [entry["normalized_components"][name] for entry in radius_rows], 50),
            }
            for name in ("start", "interior", "switch", "stop")
        }
    return summary


def audit_run(run_dir, target, acceptance, switch_radii, endpoint_radius):
    run_dir = Path(run_dir)
    commands = read_jsonl(run_dir / "command_kinematics.jsonl")
    start_stamp, stop_stamp, execution_end, switches = event_anchors(
        read_jsonl(run_dir / "events.jsonl")
    )
    pose_commands = read_jsonl(run_dir / "commands.jsonl")
    if not pose_commands:
        raise ValueError("missing commanded pose samples")
    # Match analyze_v2_run.py exactly: the pose command stream defines the
    # execution interval, while the twist stream supplies the differentiated
    # signal.
    execution_start = min(float(row["stamp"]) for row in pose_commands)
    commands = [
        row for row in commands
        if execution_start <= float(row["stamp"]) <= execution_end
    ]
    # The frozen acceptance audit ran while the ROS logger was still alive.
    # A small callback tail can therefore appear in the finalized JSONL after
    # acceptance_report.json was written.  Reproduce the exact frozen sample
    # prefix here; V2.1 will explicitly close/flush logging before analysis.
    frozen_sample_count = int(
        acceptance["trajectory_quality"].get("command_velocity_samples", len(commands))
    )
    commands = commands[:frozen_sample_count]
    stamps = [float(row["stamp"]) for row in commands]
    velocities = [row["linear_velocity"] for row in commands]
    path_length = float(acceptance["trajectory_quality"]["command_path_length_m"])
    partitions = {}
    for radius in switch_radii:
        partition = partition_executed_translation_jerk(
            stamps, velocities, start_stamp=start_stamp, stop_stamp=stop_stamp,
            switch_stamps=switches, endpoint_radius_s=endpoint_radius,
            switch_radius_s=radius,
        )
        if partition is None:
            raise ValueError("insufficient velocity samples for jerk audit")
        partition.pop("profile")
        partition["normalized_components"] = {
            name: value / max(path_length, 1e-12)
            for name, value in partition["components"].items()
        }
        partition["normalized_total"] = partition["total"] / max(path_length, 1e-12)
        partitions["%.3f" % radius] = partition

    attempts = acceptance.get("all_planning_attempt_time_s", {})
    values = [float(value) for value in attempts.get("values", [])]
    return {
        "scenario": run_dir.name.split("__rolling_wb_v2__", 1)[0],
        "scenario_family": target["scenario_family"],
        "scenario_difficulty": target["scenario_difficulty"],
        "random_seed": target["random_seed"],
        "run_dir": str(run_dir.resolve()),
        "start_stamp": start_stamp,
        "stop_stamp": stop_stamp,
        "metric_execution_start_stamp": execution_start,
        "metric_execution_end_stamp": execution_end,
        "frozen_command_velocity_samples": frozen_sample_count,
        "switch_stamps": switches,
        "switch_count": len(switches),
        "planning_attempt_count": int(attempts.get("count", len(values))),
        "cumulative_planning_time_s": float(sum(values)),
        "reported_normalized_jerk": target["jerk_energy"],
        "jerk_partitions": partitions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired-trials", required=True)
    parser.add_argument("--search-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--commit", default="4f8021431e98b044f7920f2578da7801314bc5a3")
    parser.add_argument("--endpoint-radius", type=float, default=2.0)
    parser.add_argument("--switch-radii", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    args = parser.parse_args()

    targets = load_targets(args.paired_trials)
    selected = candidate_runs(args.search_root, targets, args.commit)
    missing = sorted(set(targets) - set(selected))
    if missing:
        raise SystemExit("missing exact frozen run candidates: " + ", ".join(missing))

    records = []
    match_errors = {}
    for scenario in sorted(targets):
        error, run_dir, acceptance, _manifest = selected[scenario]
        match_errors[scenario] = error
        records.append(audit_run(
            run_dir, targets[scenario], acceptance,
            args.switch_radii, args.endpoint_radius,
        ))
    if any(error > 1e-9 for error in match_errors.values()):
        worst = max(match_errors.items(), key=lambda item: item[1])
        raise SystemExit("frozen run metric mismatch: %s %.12g" % worst)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "source_commit": args.commit,
        "paired_trials": str(Path(args.paired_trials).resolve()),
        "metric_definition_unchanged": True,
        "endpoint_radius_s": args.endpoint_radius,
        "switch_radii_s": args.switch_radii,
        "summary": summarize(records, args.switch_radii),
        "runs": records,
    }
    with (output / "v21_e0_audit.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    print("V21_E0_AUDIT=" + str((output / "v21_e0_audit.json").resolve()))


if __name__ == "__main__":
    main()
