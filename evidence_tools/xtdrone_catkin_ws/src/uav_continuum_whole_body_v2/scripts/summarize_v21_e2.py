#!/usr/bin/env python3
"""Summarize a frozen V2.1 E2 batch against matching frozen-V2 runs."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("e2_root", type=Path)
    parser.add_argument("frozen_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    rows = []
    for current_path in sorted(args.e2_root.glob("*/acceptance_report.json")):
        scenario = current_path.parent.name.split("__rolling_wb_v2__", 1)[0]
        matches = sorted(args.frozen_root.glob(
            "**/%s__*/closed_loop_runs/%s__rolling_wb_v2__*/acceptance_report.json"
            % (scenario, scenario)
        ))
        if len(matches) != 1:
            raise RuntimeError("expected one frozen run for %s, found %d" % (scenario, len(matches)))
        current, frozen = load(current_path), load(matches[0])
        new_jerk = current["trajectory_quality"]["executed_translation_jerk_energy_per_meter"]
        old_jerk = frozen["trajectory_quality"]["executed_translation_jerk_energy_per_meter"]
        modes = current["switch_continuity"]["handoff_mode_counts"]
        rows.append({
            "scenario": scenario,
            "family": scenario.rsplit("_", 2)[0],
            "difficulty": scenario.rsplit("_", 2)[1],
            "v21_pass": bool(current["passed"]),
            "frozen_pass": bool(frozen["passed"]),
            "v21_jerk": float(new_jerk), "frozen_jerk": float(old_jerk),
            "jerk_reduction": float((old_jerk - new_jerk) / old_jerk),
            "v21_task_time_s": current["execution"]["task_time_s"],
            "frozen_task_time_s": frozen["execution"]["task_time_s"],
            "v21_planning_p95_s": current["planning_time_s"]["p95"],
            "v21_planning_max_s": current["planning_time_s"]["max"],
            "v21_deadline_miss": current["execution"]["deadline_miss_count"],
            "v21_body_clearance_m": current["measured_clearance"]["body_min"],
            "v21_arm_clearance_m": current["measured_clearance"]["arm_min"],
            "v21_tracking_rmse_m": current["tracking_error_m"]["rmse"],
            "direct_c1": modes["direct_c1"],
            "validated_c2_bridge": modes["validated_c2_bridge"],
            "current_report": str(current_path.resolve()),
            "frozen_report": str(matches[0].resolve()),
        })
    if len(rows) != 15:
        raise RuntimeError("E2 requires 15 reports, found %d" % len(rows))
    planning_values = []
    for path in sorted(args.e2_root.glob("*/acceptance_report.json")):
        planning_values.extend(load(path)["all_planning_attempt_time_s"]["values"])
    reductions = [row["jerk_reduction"] for row in rows]
    new_jerk = [row["v21_jerk"] for row in rows]
    old_jerk = [row["frozen_jerk"] for row in rows]
    summary = {
        "schema": "v21_e2_rate_boundary_v1",
        "count": len(rows),
        "success": sum(row["v21_pass"] for row in rows),
        "deadline_miss_total": sum(row["v21_deadline_miss"] for row in rows),
        "jerk": {
            "v21_median": percentile(new_jerk, 50),
            "v21_p95": percentile(new_jerk, 95),
            "frozen_median": percentile(old_jerk, 50),
            "paired_reduction_median": percentile(reductions, 50),
            "paired_reduction_p05": percentile(reductions, 5),
            "improved_pairs": sum(value > 0.0 for value in reductions),
        },
        "planning_s": {
            "call_count": len(planning_values),
            "p50": percentile(planning_values, 50),
            "p95": percentile(planning_values, 95),
            "max": max(planning_values),
        },
        "handoff": {
            "direct_c1": sum(row["direct_c1"] for row in rows),
            "validated_c2_bridge": sum(row["validated_c2_bridge"] for row in rows),
        },
        "safety": {
            "body_clearance_min": min(row["v21_body_clearance_m"] for row in rows),
            "arm_clearance_min": min(row["v21_arm_clearance_m"] for row in rows),
            "tracking_rmse_max": max(row["v21_tracking_rmse_m"] for row in rows),
        },
    }
    summary["accepted"] = bool(
        summary["success"] == 15
        and summary["deadline_miss_total"] == 0
        and summary["jerk"]["paired_reduction_median"] >= 0.40
        and summary["planning_s"]["p95"] <= 5.0
        and summary["safety"]["body_clearance_min"] >= 0.06
        and summary["safety"]["arm_clearance_min"] >= 0.02
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "paired_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
