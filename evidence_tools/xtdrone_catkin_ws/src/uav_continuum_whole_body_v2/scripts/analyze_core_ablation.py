#!/usr/bin/env python3
"""Audit and summarize Full/Seed/Phi/Switch closed-loop ablations."""

import argparse
import csv
import json
import math
from pathlib import Path


SYSTEM_RESULTS = {"SYSTEM_ERROR", "INFRASTRUCTURE_FAILURE"}


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_events(path):
    if not path.is_file():
        return []
    events = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                events.append(json.loads(line))
    return events


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def percentile(values, q):
    values = sorted(float(value) for value in values if finite(value))
    if not values:
        return None
    position = (len(values) - 1) * q / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    return values[lower] + (position - lower) * (values[upper] - values[lower])


def wilson(successes, trials, z=1.959963984540054):
    if not trials:
        return None, None
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return center - half, center + half


def nested(payload, *keys):
    value = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return float(value) if finite(value) else None


def classify_failure(result, events):
    if result == "PASS":
        return "NONE"
    rejected = [event for event in events if event.get("state") == "REJECTED_KEEP_BUFFER"]
    reasons = [str(event.get("failure", "")) for event in rejected]
    if any(reason == "terminal_tip_error" for reason in reasons):
        return "TERMINAL_CONSTRAINT"
    if any("collision" in reason for reason in reasons):
        return "VALIDATION_REJECT"
    return result


def normalize(run_dir):
    trial = load_json(run_dir / "trial_result.json")
    manifest = load_json(run_dir / "manifest.json") if (run_dir / "manifest.json").is_file() else {}
    audit = load_json(run_dir / "acceptance_report.json") if (run_dir / "acceptance_report.json").is_file() else {}
    events = load_events(run_dir / "events.jsonl")
    result = str(trial.get("result", "SYSTEM_ERROR"))
    attempts = audit.get("all_planning_attempt_time_s", {}).get("values", [])
    if not attempts:
        attempts = [event.get("total_planning_time") for event in events
                    if event.get("state") in ("SAFE_TRAJECTORY", "REJECTED_KEEP_BUFFER")]
    attempts = [float(value) for value in attempts if finite(value)]
    rejection_events = [event for event in events if event.get("state") == "REJECTED_KEEP_BUFFER"]
    planned_metrics = [event.get("metrics", {}) for event in events
                       if event.get("state") == "SAFE_TRAJECTORY"]
    switch_events = [event for event in events if event.get("state") == "SWITCHED_ROLLING"
                     and "trajectory_id" in event and "command_gap" in event]
    return {
        "run_dir": str(run_dir),
        "scenario": str(trial.get("scenario", manifest.get("scenario_name", run_dir.name))),
        "family": manifest.get("scenario_family"),
        "difficulty": manifest.get("scenario_difficulty"),
        "method": str(trial.get("method", manifest.get("method", "unknown"))),
        "ablation_variant": manifest.get("ablation_variant"),
        "switch_continuity_mode": manifest.get("switch_continuity_mode"),
        "result": result,
        "failure_category": classify_failure(result, events),
        "success": result == "PASS" and bool(audit.get("passed")),
        "planning_attempts_s": attempts,
        "validation_rejects": len(rejection_events),
        "task_time_s": nested(audit, "execution", "task_time_s"),
        "body_clearance_m": nested(audit, "measured_clearance", "body_min"),
        "arm_clearance_m": nested(audit, "measured_clearance", "arm_min"),
        "tracking_rmse_m": nested(audit, "tracking_error_m", "rmse"),
        "terminal_tip_error_m": nested(audit, "terminal_tip_error_m"),
        "com_offset_max_m": nested(audit, "measured_dynamics", "com_offset_max_m"),
        "planned_tilt_max_rad": max(
            (metrics.get("tilt_max") for metrics in planned_metrics
             if finite(metrics.get("tilt_max"))), default=None),
        "planned_thrust_ratio_max": max(
            (metrics.get("thrust_ratio_max") for metrics in planned_metrics
             if finite(metrics.get("thrust_ratio_max"))), default=None),
        "jerk_per_m": nested(audit, "trajectory_quality", "executed_translation_jerk_energy_per_meter"),
        "switch_jerk_fraction": nested(audit, "trajectory_quality", "jerk_partition", "fractions", "switch"),
        "switch_count": len(switch_events),
        "switch_command_gap_p95_m": percentile([event.get("command_gap") for event in switch_events], 95),
    }


def summarize(items):
    methods = sorted({item["method"] for item in items})
    summary = {}
    for method in methods:
        rows = [item for item in items if item["method"] == method]
        successes = [item for item in rows if item["success"]]
        low, high = wilson(len(successes), len(rows))
        attempts = [value for item in rows for value in item["planning_attempts_s"]]
        summary[method] = {
            "trials": len(rows),
            "successes": len(successes),
            "success_rate": len(successes) / len(rows),
            "success_ci95_low": low,
            "success_ci95_high": high,
            "failure_categories": {
                category: sum(item["failure_category"] == category for item in rows)
                for category in sorted({item["failure_category"] for item in rows})
            },
            "family_success": {
                family: {
                    "successes": sum(item["success"] for item in rows if item["family"] == family),
                    "trials": sum(item["family"] == family for item in rows),
                }
                for family in sorted({item["family"] for item in rows if item["family"]})
            },
            "difficulty_success": {
                difficulty: {
                    "successes": sum(item["success"] for item in rows if item["difficulty"] == difficulty),
                    "trials": sum(item["difficulty"] == difficulty for item in rows),
                }
                for difficulty in sorted({item["difficulty"] for item in rows if item["difficulty"]})
            },
            "planning_p50_s": percentile(attempts, 50),
            "planning_p95_s": percentile(attempts, 95),
            "validation_rejects": sum(item["validation_rejects"] for item in rows),
            "task_time_median_s": percentile([item["task_time_s"] for item in successes], 50),
            "body_clearance_p05_m": percentile([item["body_clearance_m"] for item in successes], 5),
            "arm_clearance_p05_m": percentile([item["arm_clearance_m"] for item in successes], 5),
            "tracking_rmse_median_m": percentile([item["tracking_rmse_m"] for item in successes], 50),
            "terminal_tip_error_median_m": percentile([item["terminal_tip_error_m"] for item in successes], 50),
            "com_offset_max_p95_m": percentile([item["com_offset_max_m"] for item in successes], 95),
            "planned_tilt_max_p95_rad": percentile([item["planned_tilt_max_rad"] for item in successes], 95),
            "planned_thrust_ratio_max_p95": percentile([item["planned_thrust_ratio_max"] for item in successes], 95),
            "jerk_per_m_median": percentile([item["jerk_per_m"] for item in successes], 50),
            "switch_jerk_fraction_median": percentile([item["switch_jerk_fraction"] for item in successes], 50),
            "switch_command_gap_p95_m": percentile([item["switch_command_gap_p95_m"] for item in successes], 95),
        }
    return summary


def show(value, digits=4):
    return "-" if value is None else f"{value:.{digits}g}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.root / "ablation_audit"
    output.mkdir(parents=True, exist_ok=True)

    runs = []
    excluded = []
    for trial_path in sorted(args.root.rglob("trial_result.json")):
        item = normalize(trial_path.parent)
        if item["result"] in SYSTEM_RESULTS:
            excluded.append(item)
        else:
            runs.append(item)
    summary = summarize(runs)
    payload = {"root": str(args.root.resolve()), "included_runs": len(runs),
               "excluded_system_runs": len(excluded), "summary": summary,
               "runs": runs, "excluded": excluded}
    (output / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fields = ["scenario", "family", "difficulty", "method", "result", "failure_category",
              "validation_rejects", "task_time_s", "body_clearance_m", "arm_clearance_m",
              "tracking_rmse_m", "terminal_tip_error_m", "com_offset_max_m", "jerk_per_m",
              "planned_tilt_max_rad", "planned_thrust_ratio_max", "switch_jerk_fraction",
              "switch_count", "switch_command_gap_p95_m", "run_dir"]
    with (output / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in runs:
            writer.writerow({field: item.get(field) for field in fields})

    lines = [
        "| Method | Success | Plan P50/P95 (s) | Rejects | Task (s) | Body/Arm P05 (m) | Tracking RMSE (m) | Jerk/m | Switch jerk share |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, item in summary.items():
        lines.append("| {} | {}/{} | {}/{} | {} | {} | {}/{} | {} | {} | {} |".format(
            method, item["successes"], item["trials"], show(item["planning_p50_s"]),
            show(item["planning_p95_s"]), item["validation_rejects"], show(item["task_time_median_s"]),
            show(item["body_clearance_p05_m"]), show(item["arm_clearance_p05_m"]),
            show(item["tracking_rmse_median_m"]), show(item["jerk_per_m_median"]),
            show(item["switch_jerk_fraction_median"])))
    lines.extend(["", f"Included algorithm runs: {len(runs)}; excluded system runs: {len(excluded)}."])
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "included": len(runs), "excluded": len(excluded)}))


if __name__ == "__main__":
    main()
