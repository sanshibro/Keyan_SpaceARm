#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np
import yaml

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "src"))
from uav_continuum_whole_body_v2.dependencies import import_scipy  # noqa: E402

scipy = import_scipy()
from scipy.stats import binom_test, wilcoxon  # noqa: E402


METHODS = (
    "fixed_morphology", "aet_sequential_minjerk", "informed_rrtstar_wb",
    "global_batch_wb", "rolling_wb_v2",
)


def canonical_scenario_name(manifest, fallback):
    name = manifest.get("scenario_name")
    if name:
        return str(name)
    task_path = manifest.get("scenario")
    if task_path:
        path = Path(task_path)
        if path.is_file():
            try:
                task = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                if task.get("name"):
                    return str(task["name"])
            except (OSError, yaml.YAMLError):
                pass
    return str(fallback)


def percentile(values, value):
    return None if not values else float(np.percentile(values, value))


def wilson(success, total, z=1.959963984540054):
    if total == 0:
        return None, None
    proportion = success / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return center - half, center + half


def value(report, *keys):
    current = report
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if isinstance(current, (int, float)) and math.isfinite(float(current)):
        return float(current)
    return None


def summarize(reports):
    total = len(reports)
    successes = [item for item in reports if item.get("success")]
    success_count = len(successes)
    low, high = wilson(success_count, total)
    plan = []
    for item in reports:
        attempts = value(item, "planning_attempts_s")
        if isinstance(item.get("planning_attempts_s"), list):
            plan.extend(
                float(entry) for entry in item["planning_attempts_s"]
                if isinstance(entry, (int, float)) and math.isfinite(float(entry))
            )
        else:
            candidate = value(item, "planning_time_s")
            if candidate is not None:
                plan.append(candidate)
    task = [value(item, "trajectory_duration_s") for item in successes]
    task = [item for item in task if item is not None]
    body = [value(item, "oracle_metrics", "body_clearance_min") for item in successes]
    arm = [value(item, "oracle_metrics", "arm_clearance_min") for item in successes]
    jerk = [value(item, "derived_metrics", "jerk_energy") for item in successes]
    tip = [value(item, "oracle_metrics", "terminal_tip_error") for item in successes]
    holds = [value(item, "execution_metrics", "maximum_continuous_hold_s")
             for item in reports]
    nonterminal_holds = [
        value(item, "execution_metrics", "maximum_nonterminal_hold_s")
        for item in reports
    ]
    tracking = [value(item, "execution_metrics", "tracking_rmse_m")
                for item in successes]
    com = [value(item, "execution_metrics", "com_offset_max_m")
           for item in successes]
    deadline_misses = sum(
        int(value(item, "execution_metrics", "deadline_miss_count") or 0)
        for item in reports
    )
    planning_attempts = sum(
        int(value(item, "execution_metrics", "planning_attempt_count") or 0)
        for item in reports
    )
    collision_known = [
        item for item in reports
        if value(item, "oracle_metrics", "body_clearance_min") is not None
        and value(item, "oracle_metrics", "arm_clearance_min") is not None
    ]
    collision_free_count = sum(
        value(item, "oracle_metrics", "body_clearance_min") >= 0.0
        and value(item, "oracle_metrics", "arm_clearance_min") >= 0.0
        for item in collision_known
    )
    return {
        "trials": total,
        "success_count": success_count,
        "success_rate": None if not total else success_count / total,
        "success_ci95_low": low,
        "success_ci95_high": high,
        "oracle_collision_free_rate": (
            None if not collision_known else collision_free_count / len(collision_known)
        ),
        "planning_median_s": percentile(plan, 50),
        "planning_p95_s": percentile(plan, 95),
        "planning_max_s": max(plan) if plan else None,
        "task_time_median_s": percentile(task, 50),
        "task_time_p95_s": percentile(task, 95),
        "deadline_miss_rate": (
            None if planning_attempts == 0 else deadline_misses / planning_attempts
        ),
        "max_hold_p95_s": percentile([item for item in holds if item is not None], 95),
        "nonterminal_hold_p95_s": percentile(
            [item for item in nonterminal_holds if item is not None], 95
        ),
        "body_clearance_p05_m": percentile([item for item in body if item is not None], 5),
        "arm_clearance_p05_m": percentile([item for item in arm if item is not None], 5),
        "jerk_energy_median": percentile([item for item in jerk if item is not None], 50),
        "tracking_rmse_median_m": percentile(
            [item for item in tracking if item is not None], 50
        ),
        "com_offset_max_p95_m": percentile(
            [item for item in com if item is not None], 95
        ),
        "tip_error_median_m": percentile([
            item for item in tip if item is not None and item >= 0.0
        ], 50),
        "failures": {
            category: sum(item.get("failure_category") == category for item in reports)
            for category in sorted({item.get("failure_category", "UNKNOWN") for item in reports})
        },
    }


def closed_loop_report(path):
    audit = json.loads(path.read_text(encoding="utf-8"))
    manifest_path = path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    clearance = audit.get("measured_clearance", {})
    execution = audit.get("execution", {})
    tracking = audit.get("tracking_error_m", {})
    dynamics = audit.get("measured_dynamics", {})
    attempts = audit.get("all_planning_attempt_time_s", {}).get("values", [])
    tip = audit.get("terminal_tip_error_m")
    return {
        "scenario": canonical_scenario_name(manifest, path.parent.name),
        "manifest": manifest,
        "report": {
            "method": audit.get("method", manifest.get("method")),
            "success": bool(audit.get("passed")),
            "planning_time_s": audit.get("all_planning_attempt_time_s", {}).get("p95"),
            "planning_attempts_s": attempts,
            "trajectory_duration_s": execution.get("task_time_s"),
            "oracle_metrics": {
                "body_clearance_min": clearance.get("body_min"),
                "arm_clearance_min": clearance.get("arm_min"),
                "terminal_tip_error": tip,
            },
            "derived_metrics": {
                "jerk_energy": audit.get("trajectory_quality", {}).get("jerk_energy_sum"),
            },
            "execution_metrics": {
                "maximum_continuous_hold_s": execution.get(
                    "maximum_continuous_hold_s"
                ),
                "maximum_nonterminal_hold_s": execution.get(
                    "maximum_nonterminal_hold_s"
                ),
                "deadline_miss_count": execution.get("deadline_miss_count"),
                "planning_attempt_count": execution.get("planning_attempt_count"),
                "tracking_rmse_m": tracking.get("rmse"),
                "com_offset_max_m": dynamics.get("com_offset_max_m"),
            },
            "failure_category": "NONE" if audit.get("passed") else (
                audit.get("failures", ["SYSTEM_ERROR"])[0]
                if audit.get("failures") else "SYSTEM_ERROR"
            ),
        },
    }


def failed_closed_loop_report(path):
    trial = json.loads(path.read_text(encoding="utf-8"))
    manifest_path = path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) \
        if manifest_path.is_file() else {}
    method = trial.get("method", manifest.get("method"))
    scenario = canonical_scenario_name(
        manifest, trial.get("scenario", path.parent.name)
    )
    category = str(trial.get("result", "SYSTEM_ERROR"))
    planning_attempts = []
    events_path = path.parent / "events.jsonl"
    if events_path.is_file():
        events = []
        with events_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    events.append(json.loads(line))
        terminal = [item for item in events if item.get("state") in (
            "METHOD_FAILED", "METHOD_EXCEPTION", "TIP_TASK_FAIL"
        )]
        if terminal:
            planning_attempts = [
                float(item["solve_time"])
                for item in terminal
                if isinstance(item.get("solve_time"), (int, float))
                and math.isfinite(float(item["solve_time"]))
            ]
            state = str(terminal[-1].get("state", ""))
            reason = str(terminal[-1].get("failure", ""))
            if "timeout" in reason:
                category = "PLAN_TIMEOUT"
            elif reason in ("no_path", "invalid_start", "invalid_goal"):
                category = "NO_PATH"
            elif "collision" in reason:
                category = "VALIDATION_REJECT"
            elif state == "TIP_TASK_FAIL":
                category = "TIP_TASK_FAIL"
            elif state == "METHOD_EXCEPTION":
                category = "SYSTEM_ERROR"
    return {
        "scenario": scenario,
        "manifest": manifest,
        "report": {
            "method": method,
            "success": False,
            "planning_time_s": (
                max(planning_attempts) if planning_attempts else None
            ),
            "planning_attempts_s": planning_attempts,
            "trajectory_duration_s": None,
            "oracle_metrics": {},
            "derived_metrics": {},
            "execution_metrics": {
                "planning_attempt_count": len(planning_attempts),
                "deadline_miss_count": int(category == "PLAN_TIMEOUT"),
            },
            "failure_category": category,
        },
    }


def paired_statistics(paired_reports, allow_inference=True):
    proposed = "rolling_wb_v2"
    output = {
        "inference_enabled": bool(allow_inference),
        "success_mcnemar": {},
        "continuous_wilcoxon": [],
    }
    metric_paths = {
        "task_time_s": ("trajectory_duration_s",),
        "planning_time_s": ("planning_time_s",),
        "body_clearance_m": ("oracle_metrics", "body_clearance_min"),
        "arm_clearance_m": ("oracle_metrics", "arm_clearance_min"),
        "jerk_energy": ("derived_metrics", "jerk_energy"),
        "tip_error_m": ("oracle_metrics", "terminal_tip_error"),
    }
    for baseline in METHODS[:-1]:
        pairs = [items for items in paired_reports.values()
                 if proposed in items and baseline in items]
        ours_only = sum(
            bool(items[proposed].get("success"))
            and not bool(items[baseline].get("success")) for items in pairs
        )
        baseline_only = sum(
            bool(items[baseline].get("success"))
            and not bool(items[proposed].get("success")) for items in pairs
        )
        discordant = ours_only + baseline_only
        p_value = None
        if allow_inference:
            p_value = 1.0 if discordant == 0 else float(
                binom_test(min(ours_only, baseline_only), discordant, 0.5)
            )
        output["success_mcnemar"][baseline] = {
            "paired_trials": len(pairs), "ours_only_success": ours_only,
            "baseline_only_success": baseline_only,
            "discordant": discordant, "exact_two_sided_p": p_value,
        }
        for metric, path in metric_paths.items():
            ours_values, baseline_values = [], []
            for items in pairs:
                if not (items[proposed].get("success")
                        and items[baseline].get("success")):
                    continue
                ours = value(items[proposed], *path)
                other = value(items[baseline], *path)
                if ours is not None and other is not None:
                    ours_values.append(ours)
                    baseline_values.append(other)
            differences = np.asarray(ours_values) - np.asarray(baseline_values)
            if (not allow_inference or len(differences) < 2
                    or np.allclose(differences, 0.0)):
                p_value = None
            else:
                p_value = float(wilcoxon(
                    differences, zero_method="wilcox", alternative="two-sided"
                ).pvalue)
            output["continuous_wilcoxon"].append({
                "baseline": baseline, "metric": metric, "paired_successes": len(differences),
                "median_delta_ours_minus_baseline": (
                    float(np.median(differences)) if len(differences) else None
                ),
                "raw_p": p_value, "holm_p": None,
            })
    tested = [item for item in output["continuous_wilcoxon"]
              if item["raw_p"] is not None]
    order = sorted(range(len(tested)), key=lambda index: tested[index]["raw_p"])
    running = 0.0
    count = len(order)
    for rank, index in enumerate(order):
        adjusted = min(1.0, (count - rank) * tested[index]["raw_p"])
        running = max(running, adjusted)
        tested[index]["holm_p"] = running
    return output


def main():
    parser = argparse.ArgumentParser(description="Aggregate paired method-comparison runs")
    parser.add_argument(
        "roots", type=Path, nargs="+",
        help="one or more non-overlapping comparison-run roots",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if len(args.roots) > 1 and args.output is None:
        parser.error("--output is required when more than one root is supplied")
    by_method = {method: [] for method in METHODS}
    paired = {}
    paired_reports = {}
    manifests = []
    duplicates = []

    def add_report(scenario, report, manifest):
        method = report.get("method")
        if method not in by_method:
            return
        if method in paired_reports.setdefault(scenario, {}):
            duplicates.append({"scenario": scenario, "method": method})
            return
        report = dict(report)
        report["scenario"] = scenario
        report["scenario_family"] = manifest.get("scenario_family")
        report["scenario_difficulty"] = manifest.get("scenario_difficulty")
        report["random_seed"] = manifest.get("random_seed")
        by_method[method].append(report)
        paired.setdefault(scenario, {})[method] = bool(report.get("success"))
        paired_reports[scenario][method] = report
        if manifest:
            manifests.append(manifest)

    def unique_files(filename):
        seen = set()
        for root in args.roots:
            for path in sorted(root.rglob(filename)):
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                yield path

    for path in unique_files("summary.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        # Ignore this script's own aggregate/summary.json on resume.
        if not isinstance(payload.get("reports"), list):
            continue
        scenario = payload.get("scenario", path.parent.name)
        paired.setdefault(scenario, {})
        manifest_path = path.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) \
            if manifest_path.is_file() else {}
        for report in payload.get("reports", []):
            add_report(scenario, report, manifest)
    for path in unique_files("acceptance_report.json"):
        normalized = closed_loop_report(path)
        scenario = normalized["scenario"]
        report = normalized["report"]
        add_report(scenario, report, normalized["manifest"])
    for path in unique_files("trial_result.json"):
        if (path.parent / "acceptance_report.json").is_file():
            continue
        normalized = failed_closed_loop_report(path)
        # Infrastructure failures are retained on disk for audit and retry,
        # but are not outcomes of a compared planning method.
        if normalized["report"].get("failure_category") == "SYSTEM_ERROR":
            continue
        add_report(
            normalized["scenario"], normalized["report"], normalized["manifest"]
        )
    summary = {method: summarize(by_method[method]) for method in METHODS}
    output = args.output or args.roots[0] / "aggregate"
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    columns = [
        "method", "trials", "success_count", "success_rate",
        "success_ci95_low", "success_ci95_high", "planning_median_s",
        "planning_p95_s", "planning_max_s", "task_time_median_s",
        "task_time_p95_s", "oracle_collision_free_rate", "deadline_miss_rate",
        "max_hold_p95_s", "nonterminal_hold_p95_s",
        "body_clearance_p05_m", "arm_clearance_p05_m",
        "tracking_rmse_median_m", "com_offset_max_p95_m",
        "jerk_energy_median", "tip_error_median_m",
    ]
    with (output / "main_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for method in METHODS:
            row = {"method": method, **summary[method]}
            writer.writerow({key: row.get(key) for key in columns})
    lines = [
        "| Method | Success | Collision-free | Task med. | Plan P95 | Deadline miss | Hold P95 (any/nonterminal) | Body/Arm P05 | Tracking | COM P95 | Jerk | Tip |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = summary[method]
        success = "-" if item["success_rate"] is None else "%.1f%%" % (100.0 * item["success_rate"])
        def show(key):
            return "-" if item[key] is None else "%.4g" % item[key]
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s/%s | %s | %s | %s | %s |" % (
                method, success, show("oracle_collision_free_rate"),
                show("task_time_median_s"), show("planning_p95_s"),
                show("deadline_miss_rate"), "%s/%s" % (
                    show("max_hold_p95_s"), show("nonterminal_hold_p95_s")
                ),
                show("body_clearance_p05_m"), show("arm_clearance_p05_m"),
                show("tracking_rmse_median_m"), show("com_offset_max_p95_m"),
                show("jerk_energy_median"), show("tip_error_median_m"),
            )
        )
    (output / "main_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    trial_columns = [
        "scenario", "scenario_family", "scenario_difficulty", "random_seed",
        "method", "success", "failure_category", "planning_time_s",
        "task_time_s", "deadline_miss_count", "max_hold_s",
        "nonterminal_hold_s", "body_clearance_m", "arm_clearance_m",
        "tracking_rmse_m", "com_offset_max_m", "jerk_energy", "tip_error_m",
    ]
    with (output / "paired_trials.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=trial_columns)
        writer.writeheader()
        for scenario in sorted(paired_reports):
            for method in METHODS:
                report = paired_reports[scenario].get(method)
                if report is None:
                    continue
                writer.writerow({
                    "scenario": scenario,
                    "scenario_family": report.get("scenario_family"),
                    "scenario_difficulty": report.get("scenario_difficulty"),
                    "random_seed": report.get("random_seed"),
                    "method": method,
                    "success": bool(report.get("success")),
                    "failure_category": report.get("failure_category"),
                    "planning_time_s": value(report, "planning_time_s"),
                    "task_time_s": value(report, "trajectory_duration_s"),
                    "deadline_miss_count": value(
                        report, "execution_metrics", "deadline_miss_count"
                    ),
                    "max_hold_s": value(
                        report, "execution_metrics", "maximum_continuous_hold_s"
                    ),
                    "nonterminal_hold_s": value(
                        report, "execution_metrics", "maximum_nonterminal_hold_s"
                    ),
                    "body_clearance_m": value(
                        report, "oracle_metrics", "body_clearance_min"
                    ),
                    "arm_clearance_m": value(
                        report, "oracle_metrics", "arm_clearance_min"
                    ),
                    "tracking_rmse_m": value(
                        report, "execution_metrics", "tracking_rmse_m"
                    ),
                    "com_offset_max_m": value(
                        report, "execution_metrics", "com_offset_max_m"
                    ),
                    "jerk_energy": value(report, "derived_metrics", "jerk_energy"),
                    "tip_error_m": value(
                        report, "oracle_metrics", "terminal_tip_error"
                    ),
                })
    completeness = {
        "paired_scenarios": len(paired),
        "complete_pairs": sum(all(method in values for method in METHODS)
                              for values in paired.values()),
        "duplicates": duplicates,
        "incomplete": {
            scenario: [method for method in METHODS if method not in values]
            for scenario, values in paired.items()
            if not all(method in values for method in METHODS)
        },
    }
    (output / "completeness.json").write_text(
        json.dumps(completeness, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    artifact_eligible = bool(manifests) and all(
            item.get("experiment_class") == "closed_loop_method_comparison"
            and not item.get("reproducibility_scope_dirty", item.get("tracked_dirty", []))
            and (item.get("map_metadata") or {}).get("formal_experiment_allowed") is True
            and (item.get("map_metadata") or {}).get("truth_derived") is not True
            and (item.get("map_metadata") or {}).get("comparison_independent_survey") is True
            and int((item.get("map_metadata") or {}).get("map_audit_schema_version", 0)) >= 5
            and bool((item.get("map_metadata") or {}).get("survey_protocol_manifest_sha256"))
            for item in manifests
        )
    factor_hashes = {
        "baseline_registry_sha256": sorted({
            item.get("baseline_registry_sha256") for item in manifests
            if item.get("baseline_registry_sha256")
        }),
        "planner_config_sha256": sorted({
            item.get("planner_config_sha256") for item in manifests
            if item.get("planner_config_sha256")
        }),
        "survey_protocol": sorted({
            (item.get("map_metadata") or {}).get("survey_protocol")
            for item in manifests
            if (item.get("map_metadata") or {}).get("survey_protocol")
        }),
    }
    controlled_factors_consistent = all(
        len(values) == 1 for values in factor_hashes.values()
    )
    family_pair_counts = {}
    for items in paired_reports.values():
        families = {report.get("scenario_family") for report in items.values()
                    if report.get("scenario_family")}
        if len(families) == 1 and all(method in items for method in METHODS):
            family = next(iter(families))
            family_pair_counts[family] = family_pair_counts.get(family, 0) + 1
    minimum_pairs_per_family = 30
    statistical_claim_ready = (
        len(family_pair_counts) >= 3
        and all(count >= minimum_pairs_per_family
                for count in family_pair_counts.values())
    )
    paper_eligibility = {
        "eligible": artifact_eligible and controlled_factors_consistent,
        "artifact_eligible": artifact_eligible,
        "controlled_factors_consistent": controlled_factors_consistent,
        "controlled_factor_values": factor_hashes,
        "statistical_claim_ready": statistical_claim_ready,
        "minimum_complete_pairs_per_family": minimum_pairs_per_family,
        "complete_pairs_by_family": family_pair_counts,
        "interpretation": (
            "paper-grade artifacts; pilot only until statistical_claim_ready=true"
        ),
        "manifest_count": len(manifests),
        "rule": (
            "closed_loop + schema-v5 audited non-truth map + "
            "comparison-independent frozen survey manifest + clean relevant code"
        ),
    }
    (output / "paper_eligibility.json").write_text(
        json.dumps(paper_eligibility, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paired_statistics_report = paired_statistics(
        paired_reports, allow_inference=statistical_claim_ready
    )
    paired_statistics_report["study_level"] = (
        "formal" if statistical_claim_ready else "pilot"
    )
    paired_statistics_report["minimum_complete_pairs_per_family"] = (
        minimum_pairs_per_family
    )
    paired_statistics_report["complete_pairs_by_family"] = family_pair_counts
    paired_statistics_report["interpretation"] = (
        "p-values are intentionally withheld until every scenario family "
        "has the pre-registered number of complete paired trials"
    )
    (output / "paired_statistics.json").write_text(
        json.dumps(paired_statistics_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(completeness, indent=2, sort_keys=True))
    print("TABLE=%s" % (output / "main_table.csv"))
    return 0 if not completeness["incomplete"] and not duplicates else 2


if __name__ == "__main__":
    raise SystemExit(main())
