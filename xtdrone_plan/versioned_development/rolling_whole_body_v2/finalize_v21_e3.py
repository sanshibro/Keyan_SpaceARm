#!/usr/bin/env python3
"""Recompute, audit and package the completed V2.1 E3 formal study."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


SCRIPT = Path(__file__).resolve()
REPO = next(parent for parent in (SCRIPT.parent,) + tuple(SCRIPT.parents)
            if (parent / ".git").exists())
PACKAGE = REPO / "xtdrone_catkin_ws/src/uav_continuum_whole_body_v2"
AGGREGATOR = PACKAGE / "scripts/aggregate_method_comparison.py"
EVIDENCE = SCRIPT.parent / "generate_v21_e3_final_evidence.py"
FIGURES = SCRIPT.parent / "generate_v21_e3_paper_figures.py"
METHODS = (
    "fixed_morphology", "aet_sequential_minjerk", "informed_rrtstar_wb",
    "global_batch_wb", "rolling_wb_v2",
)
FROZEN_COMMIT = "e62b6f22a20a9c2ba18d4041124427e1cfe5f59c"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command, log):
    result = subprocess.run(
        [str(item) for item in command], cwd=str(REPO), env=os.environ.copy(),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    log.write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise RuntimeError("command failed (%d), see %s" % (result.returncode, log))


def trial_integrity(root):
    valid = {}
    system_error_files = []
    for path in root.rglob("trial_result.json"):
        try:
            trial = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        result = trial.get("result") or trial.get("status") or trial.get("outcome")
        if result == "SYSTEM_ERROR":
            system_error_files.append(str(path))
            continue
        manifest_path = path.parent / "manifest.json"
        manifest = load_json(manifest_path) if manifest_path.is_file() else {}
        scenario = trial.get("scenario_id") or trial.get("scenario") \
            or manifest.get("scenario_name")
        method = trial.get("method") or manifest.get("method")
        key = (scenario, method)
        if key in valid:
            raise RuntimeError("duplicate valid formal result: %s" % (key,))
        valid[key] = {"result": result, "path": str(path)}
    scenarios = sorted({scenario for scenario, method in valid})
    incomplete = {
        scenario: [method for method in METHODS if (scenario, method) not in valid]
        for scenario in scenarios
        if any((scenario, method) not in valid for method in METHODS)
    }
    return valid, scenarios, incomplete, system_error_files


def manifest_integrity(root):
    manifests = []
    for path in root.rglob("manifest.json"):
        if "closed_loop_runs" not in path.parts:
            continue
        manifest = load_json(path)
        if manifest.get("experiment_class") == "closed_loop_method_comparison":
            trial_path = path.parent / "trial_result.json"
            if trial_path.is_file():
                trial = load_json(trial_path)
                result = trial.get("result") or trial.get("status") \
                    or trial.get("outcome")
                if result == "SYSTEM_ERROR":
                    continue
            manifests.append((path, manifest))
    commits = sorted({item.get("commit") for _, item in manifests})
    dirty = [str(path) for path, item in manifests
             if item.get("tracked_dirty") or item.get("reproducibility_scope_dirty")]
    map_hashes_by_scenario = {}
    factor_values = {
        "baseline_registry_sha256": set(), "planner_config_sha256": set(),
        "survey_protocol": set(),
    }
    for path, item in manifests:
        scenario = item.get("scenario_name")
        map_hashes_by_scenario.setdefault(scenario, set()).add(item.get("map_hash"))
        factor_values["baseline_registry_sha256"].add(item.get("baseline_registry_sha256"))
        factor_values["planner_config_sha256"].add(item.get("planner_config_sha256"))
        factor_values["survey_protocol"].add(
            (item.get("map_metadata") or {}).get("survey_protocol")
        )
    inconsistent_maps = {
        scenario: sorted(values) for scenario, values in map_hashes_by_scenario.items()
        if len(values) != 1
    }
    return {
        "manifest_count": len(manifests),
        "commits": commits,
        "dirty_manifests": dirty,
        "inconsistent_map_hashes_within_scenario": inconsistent_maps,
        "controlled_factor_values": {
            key: sorted(value for value in values if value is not None)
            for key, values in factor_values.items()
        },
    }


def pipeline_system_events(root):
    events = []
    patterns = re.compile(
        r"METHOD_COMPARISON_RESULT=SYSTEM_ERROR|INDEPENDENT_SURVEY_RESULT=FAIL|"
        r"FORMAL_COMPARISON_PIPELINE=FAIL|Timed out:|Traceback \(most recent call last\)"
    )
    for path in root.rglob("*.log"):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for number, line in enumerate(lines, 1):
            if patterns.search(line):
                events.append({"path": str(path), "line": number, "text": line.strip()})
    return events


def fusion_recovery_integrity(root):
    """Retain and verify every semantics-equivalent OOM recovery record."""
    records = []
    failures = []
    bounded_outputs = {
        path.parent for path in root.rglob("fusion_memory_bounded*.log")
    }
    record_paths = sorted(root.rglob("fusion_oom_recovery.json"))
    recorded_outputs = {path.parent for path in record_paths}
    for output in sorted(bounded_outputs - recorded_outputs):
        failures.append("missing recovery record: %s" % output)
    for path in record_paths:
        record = load_json(path)
        audit_path = path.parent / "map_audit_report.json"
        if not audit_path.is_file():
            failures.append("missing map audit for %s" % path)
            continue
        audit = load_json(audit_path)
        scenario = path.parent.name.split("__", 1)[0]
        checks = {
            "scenario": record.get("scenario") == scenario,
            "cause": record.get("cause")
                == "kernel_oom_exit_137_in_frozen_27_offset_union",
            "recovery": record.get("recovery")
                == "semantics_equivalent_one_voxel_halo_precrop_then_identical_union",
            "reference_equivalence_hash": record.get(
                "equivalence_validation_reference_hash"
            ) == "dc937cfbd9c0afc900419ff4ce634a52e96ae99c22f5cf212c65b481c11900e4",
            "record_audit_passed": record.get("audit_passed") is True,
            "audit_passed": audit.get("passed") is True,
            "schema_v5": (record.get("audit_schema_version") == 5
                           and audit.get("audit_schema_version") == 5),
            "candidate_hash": record.get("candidate_map_hash")
                == audit.get("input_map_hash"),
            "audited_hash": record.get("audited_map_hash")
                == audit.get("audited_map_hash"),
        }
        if not all(checks.values()):
            failures.append("recovery verification failed %s: %s" % (path, checks))
        records.append({
            "path": str(path.relative_to(root)),
            "scenario": scenario,
            "checks": checks,
            "record": record,
        })
    return {"count": len(records), "passed": not failures,
            "failures": failures, "records": records}


def compare_csv(left, right):
    with left.open(newline="", encoding="utf-8") as handle:
        first = list(csv.DictReader(handle))
    with right.open(newline="", encoding="utf-8") as handle:
        second = list(csv.DictReader(handle))
    return first == second


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    args = parser.parse_args()
    root = args.result_root.resolve()
    progress = load_json(root / "batch_progress.json")
    if progress.get("state") != "COMPLETE" or progress.get("complete_pairs") != 90:
        raise SystemExit("formal batch is not complete: %s" % progress)
    valid, scenarios, incomplete, system_error_files = trial_integrity(root)
    if len(valid) != 450 or len(scenarios) != 90 or incomplete:
        raise SystemExit("invalid formal trial set: %d/%d, incomplete=%s" % (
            len(valid), len(scenarios), incomplete
        ))
    manifests = manifest_integrity(root)
    if (manifests["manifest_count"] != 450
            or manifests["commits"] != [FROZEN_COMMIT]
            or manifests["dirty_manifests"]
            or manifests["inconsistent_map_hashes_within_scenario"]
            or any(len(values) != 1
                   for values in manifests["controlled_factor_values"].values())):
        raise SystemExit("manifest integrity failed: %s" % manifests)
    fusion_recoveries = fusion_recovery_integrity(root)
    if not fusion_recoveries["passed"]:
        raise SystemExit("fusion recovery integrity failed: %s" % fusion_recoveries)

    closed_loop_roots = sorted({path for path in root.rglob("closed_loop_runs")
                                if path.is_dir()})
    if len(closed_loop_roots) != 90:
        raise SystemExit("expected 90 closed_loop_runs roots, got %d" % len(closed_loop_roots))
    recomputed = root / "aggregate_recomputed"
    recomputed.mkdir(parents=True, exist_ok=True)
    run(
        [sys.executable, AGGREGATOR, *closed_loop_roots, "--output", recomputed],
        root / "aggregate_recomputed.log",
    )
    official = root / "aggregate"
    if not compare_csv(official / "paired_trials.csv", recomputed / "paired_trials.csv"):
        raise SystemExit("official and independently recomputed paired tables differ")
    eligibility = load_json(recomputed / "paper_eligibility.json")
    completeness = load_json(recomputed / "completeness.json")
    if not (eligibility.get("eligible") and eligibility.get("statistical_claim_ready")):
        raise SystemExit("paper/statistical eligibility failed: %s" % eligibility)
    if completeness.get("complete_pairs") != 90 or completeness.get("duplicates") \
            or completeness.get("incomplete"):
        raise SystemExit("recomputed completeness failed: %s" % completeness)

    run(
        [sys.executable, EVIDENCE, root, "--aggregate", recomputed,
         "--output", root / "final_evidence"],
        root / "final_evidence_generation.log",
    )
    run(
        [sys.executable, FIGURES, root, "--aggregate", recomputed,
         "--output", root / "paper_artifacts"],
        root / "paper_artifacts_generation.log",
    )

    figures = sorted((root / "paper_artifacts/figures").glob("*"))
    tables = sorted((root / "paper_artifacts/tables").glob("*"))
    expected_stems = {"fig%02d" % index for index in range(1, 9)}
    png_stems = {path.stem.split("_")[0] for path in figures if path.suffix == ".png"}
    pdf_stems = {path.stem.split("_")[0] for path in figures if path.suffix == ".pdf"}
    if png_stems != expected_stems or pdf_stems != expected_stems:
        raise SystemExit("figure set incomplete: png=%s pdf=%s" % (png_stems, pdf_stems))
    too_small = [str(path) for path in figures if path.stat().st_size < 10_000]
    if too_small or len(tables) < 5:
        raise SystemExit("artifact size/count gate failed: %s, tables=%d" % (
            too_small, len(tables)
        ))

    hashes = {}
    for directory in (recomputed, root / "final_evidence", root / "paper_artifacts"):
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                hashes[str(path.relative_to(root))] = sha256(path)
    system_events = pipeline_system_events(root)
    audit = {
        "passed": True,
        "frozen_algorithm_commit": FROZEN_COMMIT,
        "batch_progress": progress,
        "formal_results": len(valid),
        "complete_pairs": len(scenarios),
        "system_error_trial_files_retained": system_error_files,
        "pipeline_system_events_retained": system_events,
        "fusion_oom_recoveries": fusion_recoveries,
        "manifest_integrity": manifests,
        "official_recomputed_paired_tables_equal": True,
        "paper_eligibility": eligibility,
        "completeness": completeness,
        "figures": [str(path.relative_to(root)) for path in figures],
        "tables": [str(path.relative_to(root)) for path in tables],
        "artifact_sha256": hashes,
    }
    (root / "final_completion_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = """# Rolling-WB-V2.1 E3 最终完成审计

- 审计结果：**PASS**
- 冻结算法提交：`{commit}`
- 正式结果：{trials} 条，{pairs} 个五方法完整配对
- 官方聚合表与独立重算表：完全一致
- paper eligibility / statistical readiness：`true / true`
- 论文图：{png} PNG + {pdf} PDF；LaTeX 表：{tables}
- 等价地图融合 OOM 恢复：{recoveries} 次，全部通过 schema-v5 独立审计
- 证据哈希：见 `final_completion_audit.json`

系统级重试事件仅作基础设施审计，不替换、删除或篡改方法失败结果。
""".format(
        commit=FROZEN_COMMIT, trials=len(valid), pairs=len(scenarios),
        png=len([path for path in figures if path.suffix == ".png"]),
        pdf=len([path for path in figures if path.suffix == ".pdf"]),
        tables=len(tables),
        recoveries=fusion_recoveries["count"],
    )
    (root / "FINAL_COMPLETION_AUDIT_CN.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "result": "PASS", "root": str(root), "trials": len(valid),
        "pairs": len(scenarios), "system_events_retained": len(system_events),
        "artifacts_hashed": len(hashes),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
