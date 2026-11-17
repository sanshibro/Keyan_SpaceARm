#!/usr/bin/env python3
"""Generate the pre-registered E3 integrity and statistical evidence package.

The script is deliberately read-only with respect to trial directories.  It
consumes the aggregate ``paired_trials.csv`` plus measured acceptance reports
and writes derived artifacts to ``final_evidence`` (or ``--output``).  Formal
inference is refused unless the 90-pair/450-trial design is complete.
"""

import argparse
import csv
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import statistics
import sys

import numpy as np


SCRIPT = Path(__file__).resolve()
REPO = next(parent for parent in (SCRIPT.parent,) + tuple(SCRIPT.parents)
            if (parent / ".git").exists())
PACKAGE = REPO / "xtdrone_catkin_ws/src/uav_continuum_whole_body_v2"
sys.path.insert(0, str(PACKAGE / "src"))
from uav_continuum_whole_body_v2.dependencies import import_scipy  # noqa: E402

scipy = import_scipy()
from scipy.stats import binom_test, friedmanchisquare, rankdata, wilcoxon  # noqa: E402


METHODS = (
    "fixed_morphology",
    "aet_sequential_minjerk",
    "informed_rrtstar_wb",
    "global_batch_wb",
    "rolling_wb_v2",
)
OURS = "rolling_wb_v2"
FAMILIES = ("passage_sweep", "multi_passage", "clutter_reach")
DIFFICULTIES = ("easy", "medium", "hard")
METRICS = {
    "task_time_s": "lower",
    "planning_time_s": "lower",
    "body_clearance_m": "higher",
    "arm_clearance_m": "higher",
    "tracking_rmse_m": "lower",
    "com_offset_max_m": "lower",
    "jerk_energy": "lower",
    "tip_error_m": "lower",
}


def finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def percentile(values, q):
    values = [float(item) for item in values if finite(item) is not None]
    return None if not values else float(np.percentile(values, q))


def wilson(success, total, z=1.959963984540054):
    if total <= 0:
        return None, None
    p = success / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(
        p * (1.0 - p) / total + z * z / (4.0 * total * total)
    ) / denominator
    return center - half, center + half


def holm(items, key="raw_p"):
    valid = [item for item in items if item.get(key) is not None]
    ordered = sorted(valid, key=lambda item: item[key])
    running = 0.0
    count = len(ordered)
    for index, item in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * item[key]))
        item["holm_p"] = running
    return items


def matched_rank_biserial(differences):
    values = np.asarray([item for item in differences if item != 0.0], dtype=float)
    if not len(values):
        return 0.0
    ranks = rankdata(np.abs(values), method="average")
    positive = float(np.sum(ranks[values > 0.0]))
    negative = float(np.sum(ranks[values < 0.0]))
    total = positive + negative
    return 0.0 if total == 0.0 else (positive - negative) / total


def bootstrap_median_ci(values, seed, samples=10000):
    """Deterministic paired bootstrap CI for a median difference."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None, None
    generator = np.random.default_rng(int(seed))
    medians = np.empty(samples, dtype=float)
    # Chunking bounds peak memory while retaining the exact registered draw
    # count and deterministic random stream.
    offset = 0
    while offset < samples:
        count = min(1000, samples - offset)
        indices = generator.integers(0, len(values), size=(count, len(values)))
        medians[offset:offset + count] = np.median(values[indices], axis=1)
        offset += count
    return tuple(float(item) for item in np.percentile(medians, [2.5, 97.5]))


def load_trials(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["success"] = str(row.get("success", "")).lower() == "true"
        row["random_seed"] = int(float(row["random_seed"]))
        for key in METRICS:
            row[key] = finite(row.get(key))
    return rows


def verify_design(rows):
    keys = [(row["scenario"], row["method"]) for row in rows]
    duplicates = [list(key) for key, count in Counter(keys).items() if count != 1]
    scenarios = sorted({row["scenario"] for row in rows})
    lookup = {(row["scenario"], row["method"]): row for row in rows}
    incomplete = {
        scenario: [method for method in METHODS if (scenario, method) not in lookup]
        for scenario in scenarios
        if any((scenario, method) not in lookup for method in METHODS)
    }
    scenario_metadata = {}
    inconsistent = []
    for scenario in scenarios:
        metadata = {
            (row["scenario_family"], row["scenario_difficulty"], row["random_seed"])
            for row in rows if row["scenario"] == scenario
        }
        if len(metadata) != 1:
            inconsistent.append(scenario)
        else:
            scenario_metadata[scenario] = next(iter(metadata))
    family = Counter(value[0] for value in scenario_metadata.values())
    difficulty = Counter(value[1] for value in scenario_metadata.values())
    cells = Counter((value[0], value[1]) for value in scenario_metadata.values())
    seeds = sorted({value[2] for value in scenario_metadata.values()})
    passed = (
        len(rows) == 450 and len(scenarios) == 90 and not duplicates
        and not incomplete and not inconsistent
        and all(family[item] == 30 for item in FAMILIES)
        and all(difficulty[item] == 30 for item in DIFFICULTIES)
        and all(cells[(family_name, difficulty_name)] == 10
                for family_name in FAMILIES for difficulty_name in DIFFICULTIES)
        and seeds == list(range(11, 21))
    )
    return {
        "passed": passed,
        "trial_rows": len(rows),
        "scenarios": len(scenarios),
        "duplicates": duplicates,
        "incomplete": incomplete,
        "inconsistent_scenario_metadata": inconsistent,
        "pairs_by_family": dict(family),
        "pairs_by_difficulty": dict(difficulty),
        "pairs_by_family_difficulty": {
            "%s/%s" % key: value for key, value in sorted(cells.items())
        },
        "seeds": seeds,
    }


def success_statistics(rows):
    result = {"overall": {}, "by_family": {}, "by_difficulty": {}, "mcnemar": {}}
    for method in METHODS:
        subset = [row for row in rows if row["method"] == method]
        success = sum(row["success"] for row in subset)
        low, high = wilson(success, len(subset))
        result["overall"][method] = {
            "success": success,
            "trials": len(subset),
            "rate": None if not subset else success / len(subset),
            "wilson95": [low, high],
        }
    for group_key, groups in (("by_family", FAMILIES), ("by_difficulty", DIFFICULTIES)):
        column = "scenario_family" if group_key == "by_family" else "scenario_difficulty"
        for group in groups:
            result[group_key][group] = {}
            for method in METHODS:
                subset = [row for row in rows
                          if row["method"] == method and row[column] == group]
                success = sum(row["success"] for row in subset)
                low, high = wilson(success, len(subset))
                result[group_key][group][method] = {
                    "success": success,
                    "trials": len(subset),
                    "rate": None if not subset else success / len(subset),
                    "wilson95": [low, high],
                }
    lookup = {(row["scenario"], row["method"]): row for row in rows}
    scenarios = sorted({row["scenario"] for row in rows})
    for baseline in METHODS[:-1]:
        ours_only = sum(
            lookup[(scenario, OURS)]["success"]
            and not lookup[(scenario, baseline)]["success"] for scenario in scenarios
        )
        baseline_only = sum(
            lookup[(scenario, baseline)]["success"]
            and not lookup[(scenario, OURS)]["success"] for scenario in scenarios
        )
        discordant = ours_only + baseline_only
        result["mcnemar"][baseline] = {
            "ours_only": ours_only,
            "baseline_only": baseline_only,
            "discordant": discordant,
            "exact_two_sided_p": (
                1.0 if discordant == 0 else
                float(binom_test(min(ours_only, baseline_only), discordant, 0.5))
            ),
        }
    return result


def descriptive_statistics(rows):
    output = {}
    for method in METHODS:
        successful = [row for row in rows if row["method"] == method and row["success"]]
        output[method] = {}
        for metric in METRICS:
            values = [row[metric] for row in successful if row[metric] is not None]
            output[method][metric] = {
                "n": len(values),
                "median": percentile(values, 50),
                "q1": percentile(values, 25),
                "q3": percentile(values, 75),
                "p05": percentile(values, 5),
                "p95": percentile(values, 95),
                "mean": None if not values else float(np.mean(values)),
                "std": None if len(values) < 2 else float(np.std(values, ddof=1)),
            }
    return output


def common_success_statistics(rows):
    lookup = {(row["scenario"], row["method"]): row for row in rows}
    scenarios = sorted({row["scenario"] for row in rows})
    common = [scenario for scenario in scenarios
              if all(lookup[(scenario, method)]["success"] for method in METHODS)]
    friedman = {}
    posthoc = []
    contrast_index = 0
    for metric, direction in METRICS.items():
        usable = [scenario for scenario in common
                  if all(lookup[(scenario, method)][metric] is not None for method in METHODS)]
        arrays = {
            method: np.asarray([lookup[(scenario, method)][metric] for scenario in usable])
            for method in METHODS
        }
        if len(usable) >= 3:
            test = friedmanchisquare(*(arrays[method] for method in METHODS))
            statistic, p_value = float(test.statistic), float(test.pvalue)
        else:
            statistic, p_value = None, None
        friedman[metric] = {
            "direction": direction,
            "common_success_n": len(usable),
            "statistic": statistic,
            "p": p_value,
        }
        for baseline in METHODS[:-1]:
            differences = arrays[OURS] - arrays[baseline]
            ci_low, ci_high = bootstrap_median_ci(
                differences, seed=20260804 + contrast_index
            )
            contrast_index += 1
            if len(differences) < 2 or np.allclose(differences, 0.0):
                p_value = None
            else:
                p_value = float(wilcoxon(
                    differences, zero_method="wilcox", alternative="two-sided"
                ).pvalue)
            posthoc.append({
                "metric": metric,
                "direction": direction,
                "baseline": baseline,
                "n": len(differences),
                "ours_median": percentile(arrays[OURS], 50),
                "baseline_median": percentile(arrays[baseline], 50),
                "median_delta_ours_minus_baseline": percentile(differences, 50),
                "median_delta_bootstrap95_low": ci_low,
                "median_delta_bootstrap95_high": ci_high,
                "matched_rank_biserial_ours_minus_baseline": matched_rank_biserial(differences),
                "raw_p": p_value,
                "holm_p": None,
            })
    holm(posthoc)
    return {
        "common_success_scenarios": len(common),
        "scenario_ids": common,
        "friedman": friedman,
        "wilcoxon_holm": posthoc,
        "holm_family": "all pre-registered continuous metrics x four ours-vs-baseline contrasts",
    }


def failure_statistics(rows):
    output = {}
    for method in METHODS:
        subset = [row for row in rows if row["method"] == method]
        output[method] = dict(sorted(Counter(
            "NONE" if row["success"] else row.get("failure_category", "UNKNOWN")
            for row in subset
        ).items()))
    return output


def nested(report, *keys):
    current = report
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def load_switch_diagnostics(run_dir):
    """Load one executor diagnostic per accepted rolling trajectory."""
    path = run_dir / "events.jsonl"
    if not path.is_file():
        return []
    unique = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (event.get("state") != "SWITCHED_ROLLING"
                or event.get("switch_continuity_diagnostics_version") != 2
                or event.get("trajectory_id") is None):
            continue
        identifier = int(event["trajectory_id"])
        # The executor and logger may emit the same switch once each. Prefer
        # the richer executor event; numerical continuity fields are equal.
        previous = unique.get(identifier)
        if previous is None or len(event) > len(previous):
            unique[identifier] = event
    return [unique[key] for key in sorted(unique)]


def audit_v21_runs(root):
    reports = []
    for path in root.rglob("acceptance_report.json"):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if report.get("method") == OURS:
            reports.append((path, report))
    direct = bridge = switches = missing = deadlines = attempts = 0
    hold = []
    planning = []
    switch_metrics = defaultdict(list)
    jerk_total = []
    jerk_switch_fraction = []
    body = []
    arm = []
    tracking = []
    com = []
    torque = []
    failures = []
    for path, report in reports:
        execution = report.get("execution", {})
        continuity = report.get("switch_continuity", {})
        modes = continuity.get("handoff_mode_counts", {})
        direct += int(modes.get("direct_c1", 0) or 0)
        bridge += int(modes.get("validated_c2_bridge", 0) or 0)
        switches += int(continuity.get("switch_count", 0) or 0)
        missing += int(continuity.get("missing_diagnostics_count", 0) or 0)
        deadlines += int(execution.get("deadline_miss_count", 0) or 0)
        attempts += int(execution.get("planning_attempt_count", 0) or 0)
        for value_ in report.get("all_planning_attempt_time_s", {}).get("values", []):
            if finite(value_) is not None:
                planning.append(float(value_))
        if finite(execution.get("maximum_continuous_hold_s")) is not None:
            hold.append(float(execution["maximum_continuous_hold_s"]))
        for event in load_switch_diagnostics(path.parent):
            for name in (
                    "position_gap_m", "yaw_gap_rad", "bending_gap_rad",
                    "linear_velocity_gap_m_s", "yaw_rate_gap_rad_s",
                    "bending_rate_gap_rad_s", "linear_acceleration_gap_m_s2"):
                if finite(event.get(name)) is not None:
                    switch_metrics[name].append(float(event[name]))
        quality = report.get("trajectory_quality", {})
        partition = quality.get("jerk_partition", {})
        if finite(quality.get("jerk_energy_sum")) is not None:
            jerk_total.append(float(quality["jerk_energy_sum"]))
        if finite(nested(partition, "fractions", "switch")) is not None:
            jerk_switch_fraction.append(float(partition["fractions"]["switch"]))
        clearance = report.get("measured_clearance", {})
        dynamics = report.get("measured_dynamics", {})
        tracking_report = report.get("tracking_error_m", {})
        for destination, value_ in (
            (body, clearance.get("body_min")),
            (arm, clearance.get("arm_min")),
            (tracking, tracking_report.get("rmse")),
            (com, dynamics.get("com_offset_max_m")),
            (torque, dynamics.get("gravity_torque_max_nm")),
        ):
            if finite(value_) is not None:
                destination.append(float(value_))
        if not report.get("passed"):
            failures.append({"path": str(path), "failures": report.get("failures", [])})
    system_errors = []
    for path in root.rglob("trial_result.json"):
        try:
            trial = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if trial.get("result") == "SYSTEM_ERROR":
            system_errors.append(str(path))
    return {
        "acceptance_reports": len(reports),
        "passed_reports": len(reports) - len(failures),
        "failed_reports": failures,
        "switches": switches,
        "direct_c1": direct,
        "validated_c2_bridge": bridge,
        "missing_switch_diagnostics": missing,
        "planning_attempts": attempts,
        "deadline_misses": deadlines,
        "planning_time_s": {
            "median": percentile(planning, 50), "p95": percentile(planning, 95),
            "max": None if not planning else max(planning), "count": len(planning),
        },
        "maximum_continuous_hold_s": {
            "median": percentile(hold, 50), "p95": percentile(hold, 95),
            "max": None if not hold else max(hold),
        },
        "candidate_switch_gap_statistics": {
            name: {
                "count": len(values), "median": percentile(values, 50),
                "p95": percentile(values, 95),
                "max": None if not values else max(values),
            }
            for name, values in sorted(switch_metrics.items())
        },
        "jerk_energy": {
            "median": percentile(jerk_total, 50), "p95": percentile(jerk_total, 95),
        },
        "switch_jerk_fraction": {
            "median": percentile(jerk_switch_fraction, 50),
            "p95": percentile(jerk_switch_fraction, 95),
        },
        "measured_body_clearance_min_m": None if not body else min(body),
        "measured_arm_clearance_min_m": None if not arm else min(arm),
        "tracking_rmse_p95_m": percentile(tracking, 95),
        "com_offset_p95_m": percentile(com, 95),
        "gravity_torque_p95_nm": percentile(torque, 95),
        "system_error_artifacts_retained": system_errors,
    }


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=4):
    return "-" if value is None else ("%%.%dg" % digits) % value


def write_markdown(output, integrity, success, descriptive, canonical, common, failures, v21):
    gaps = v21["candidate_switch_gap_statistics"]
    lines = [
        "# Rolling-WB-V2.1 E3 预注册实验最终证据报告",
        "",
        "## 1. 完整性与可用性",
        "",
        "- 设计完整性：`%s`（%d 个 trial，%d 个完整配对任务）。" % (
            "PASS" if integrity["passed"] else "FAIL",
            integrity["trial_rows"], integrity["scenarios"],
        ),
        "- 随机种：%s；每场景族 30 对，每难度 30 对，每“场景族×难度”单元 10 对。" % integrity["seeds"],
        "- 原始失败和系统重试均保留，本报告未删除 baseline 失败。",
        "",
        "## 2. 主要结果",
        "",
        "| Method | Success [Wilson 95% CI] | Task median [IQR] (s) | Plan P95 (s) | Body/Arm P05 (m) | Track RMSE med. (m) | Jerk med. | Tip med. (m) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = success["overall"][method]
        task = descriptive[method]["task_time_s"]
        # The pre-registered main table pools every measured planning attempt.
        # ``descriptive[...][planning_time_s]`` instead summarizes one
        # scenario-level value per successful trial and is retained only for
        # supplementary distribution analyses.
        plan_p95 = canonical[method]["planning_p95_s"]
        body = descriptive[method]["body_clearance_m"]
        arm = descriptive[method]["arm_clearance_m"]
        tracking = descriptive[method]["tracking_rmse_m"]
        jerk = descriptive[method]["jerk_energy"]
        tip = descriptive[method]["tip_error_m"]
        lines.append(
            "| %s | %d/%d, %.1f%% [%.1f, %.1f] | %s [%s, %s] | %s | %s/%s | %s | %s | %s |" % (
                method, item["success"], item["trials"], 100 * item["rate"],
                100 * item["wilson95"][0], 100 * item["wilson95"][1],
                fmt(task["median"]), fmt(task["q1"]), fmt(task["q3"]),
                fmt(plan_p95), fmt(body["p05"]), fmt(arm["p05"]),
                fmt(tracking["median"]), fmt(jerk["median"]), fmt(tip["median"]),
            )
        )
    lines.extend([
        "",
        "## 3. 配对统计",
        "",
        "- 五种方法共同成功子集：%d/90。" % common["common_success_scenarios"],
        "- McNemar 以所有 90 个配对任务为样本；连续指标只在五方法共同成功集合上作 Friedman 及 Wilcoxon–Holm，并对配对中位差给出 10000 次配对 bootstrap 95% CI，避免难度选择偏差。",
        "",
        "| Baseline | Ours-only success | Baseline-only success | Exact McNemar p |",
        "|---|---:|---:|---:|",
    ])
    for baseline in METHODS[:-1]:
        item = success["mcnemar"][baseline]
        lines.append("| %s | %d | %d | %.3g |" % (
            baseline, item["ours_only"], item["baseline_only"], item["exact_two_sided_p"]
        ))
    lines.extend([
        "",
        "## 4. V2.1 闭环、接管与动力学代理审计",
        "",
        "- V2.1 验收报告：%d；通过：%d；规划尝试：%d；deadline miss：%d。" % (
            v21["acceptance_reports"], v21["passed_reports"],
            v21["planning_attempts"], v21["deadline_misses"],
        ),
        "- 滚动接管：%d 次（直接接管 %d，使用 C2 桥接的验证接管 %d，缺失诊断 %d）。" % (
            v21["switches"], v21["direct_c1"], v21["validated_c2_bridge"],
            v21["missing_switch_diagnostics"],
        ),
        "- 桥接前候选对齐间隙 median/P95/max：位置 %s/%s/%s m，速度 %s/%s/%s m/s，加速度 %s/%s/%s m/s²。" % (
            fmt(gaps["position_gap_m"]["median"]),
            fmt(gaps["position_gap_m"]["p95"]), fmt(gaps["position_gap_m"]["max"]),
            fmt(gaps["linear_velocity_gap_m_s"]["median"]),
            fmt(gaps["linear_velocity_gap_m_s"]["p95"]),
            fmt(gaps["linear_velocity_gap_m_s"]["max"]),
            fmt(gaps["linear_acceleration_gap_m_s2"]["median"]),
            fmt(gaps["linear_acceleration_gap_m_s2"]["p95"]),
            fmt(gaps["linear_acceleration_gap_m_s2"]["max"]),
        ),
        "- 上述是可选 C2 桥接之前的候选间隙，不当作实际下发指令的不连续量，也不单独构成严格解析 C2 连续性的证明；实际命令连续性见论文补充图表。",
        "- 规划时间 median/P95/max：%s/%s/%s s；最长持续 hold P95/max：%s/%s s。" % (
            fmt(v21["planning_time_s"]["median"]), fmt(v21["planning_time_s"]["p95"]),
            fmt(v21["planning_time_s"]["max"]),
            fmt(v21["maximum_continuous_hold_s"]["p95"]),
            fmt(v21["maximum_continuous_hold_s"]["max"]),
        ),
        "- 最小实测机体/臂净空：%s/%s m；跟踪 RMSE P95：%s m。" % (
            fmt(v21["measured_body_clearance_min_m"]),
            fmt(v21["measured_arm_clearance_min_m"]),
            fmt(v21["tracking_rmse_p95_m"]),
        ),
        "- COM 偏移 P95：%s m；重力力矩代理 P95：%s N·m。这两项只作保守动力学代理，不宣称为实测电机推力/力矩。" % (
            fmt(v21["com_offset_p95_m"]), fmt(v21["gravity_torque_p95_nm"]),
        ),
        "- COM 差异只解释为与基线处于可比量级；本方法较小但满足阈值的机体净空解释为空间利用，而不宣称获得最大安全裕度。",
        "",
        "## 5. 失败分类",
        "",
        "```json",
        json.dumps(failures, indent=2, ensure_ascii=False, sort_keys=True),
        "```",
        "",
        "## 6. 结论使用边界",
        "",
        "只有当本报告第 1 节为 PASS，且 `paper_eligibility.json` 同时给出 `eligible=true` 和 `statistical_claim_ready=true`，才可将这批数据称为预注册 E3 正式结果。对于 jerk、净空和耗时，论文必须同时报告自身成功集与共同成功集，不得只选有利子集。不得宣称全局最优、完整刚柔耦合动力学保证或解析连续碰撞证明。",
        "",
    ])
    (output / "E3_final_evidence_report_cn.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--aggregate", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    aggregate = args.aggregate or args.result_root / "aggregate"
    output = args.output or args.result_root / "final_evidence"
    paired = aggregate / "paired_trials.csv"
    summary_path = aggregate / "summary.json"
    if not paired.is_file():
        raise SystemExit("missing aggregate paired_trials.csv: %s" % paired)
    if not summary_path.is_file():
        raise SystemExit("missing aggregate summary.json: %s" % summary_path)
    output.mkdir(parents=True, exist_ok=True)
    rows = load_trials(paired)
    canonical = json.loads(summary_path.read_text(encoding="utf-8"))
    if set(canonical) != set(METHODS):
        raise SystemExit("aggregate summary method set does not match E3 protocol")
    integrity = verify_design(rows)
    if not integrity["passed"] and not args.allow_incomplete:
        raise SystemExit("formal E3 design is incomplete; refusing inference: %s" % integrity)
    success = success_statistics(rows)
    descriptive = descriptive_statistics(rows)
    common = common_success_statistics(rows)
    failures = failure_statistics(rows)
    v21 = audit_v21_runs(args.result_root)
    payloads = {
        "integrity.json": integrity,
        "success_statistics.json": success,
        "descriptive_statistics.json": descriptive,
        "common_success_statistics.json": common,
        "failure_taxonomy.json": failures,
        "v21_execution_audit.json": v21,
    }
    for name, payload in payloads.items():
        (output / name).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    success_rows = []
    for method in METHODS:
        item = success["overall"][method]
        success_rows.append({
            "method": method, "success": item["success"], "trials": item["trials"],
            "rate": item["rate"], "wilson95_low": item["wilson95"][0],
            "wilson95_high": item["wilson95"][1],
        })
    write_csv(output / "success_overall.csv", success_rows)
    write_csv(output / "wilcoxon_holm.csv", common["wilcoxon_holm"])
    write_csv(output / "friedman.csv", [
        {"metric": metric, **item} for metric, item in common["friedman"].items()
    ])
    failure_rows = [
        {"method": method, "category": category, "count": count}
        for method in METHODS for category, count in failures[method].items()
    ]
    write_csv(output / "failure_taxonomy.csv", failure_rows)
    write_markdown(output, integrity, success, descriptive, canonical, common, failures, v21)
    print(json.dumps({
        "result": "PASS" if integrity["passed"] else "INCOMPLETE",
        "trial_rows": len(rows), "output": str(output),
        "common_success_scenarios": common["common_success_scenarios"],
    }, indent=2, ensure_ascii=False))
    return 0 if integrity["passed"] or args.allow_incomplete else 2


if __name__ == "__main__":
    raise SystemExit(main())
