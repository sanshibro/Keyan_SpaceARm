#!/usr/bin/env python3
"""Resume frozen E3 at the first incomplete pair without replaying old pairs.

This is an experiment-orchestration utility only.  It consumes the immutable
pre-registration manifest and invokes the unchanged formal per-scenario
pipeline with its frozen budgets.  A non-SYSTEM_ERROR result is a completed
method outcome, including an honest method failure.  SYSTEM_ERROR files remain
in place as infrastructure evidence and are retried.

The sole automatic map recovery is for a confirmed Linux OOM exit 137 in the
frozen one-voxel survey union.  It uses the separately hash-validated,
semantics-equivalent bounded-memory union and then the unchanged schema-v5 map
auditor.  No algorithm, scenario, map threshold, metric or method budget is
changed by this supervisor.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


SCRIPT = Path(__file__).resolve()
REPO = next(parent for parent in (SCRIPT.parent,) + tuple(SCRIPT.parents)
            if (parent / ".git").exists())
SCENARIOS = REPO / (
    "xtdrone_plan/experiments/method_comparison/v21_e3_final_scenarios_11_20"
)
RESULTS = REPO / (
    "xtdrone_plan/experiments/method_comparison/v21_e3_final_450_commite62b6f2"
)
MEMORY_FUSION = SCRIPT.parent / "fuse_slam_maps_memory_bounded.py"
FROZEN_COMMIT = "e62b6f22a20a9c2ba18d4041124427e1cfe5f59c"
METHODS = (
    "fixed_morphology", "aet_sequential_minjerk", "informed_rrtstar_wb",
    "global_batch_wb", "rolling_wb_v2",
)


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def run_logged(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\nCOMMAND=" + json.dumps(
            [str(item) for item in command], ensure_ascii=False
        ) + "\n")
        handle.flush()
        return subprocess.run(
            [str(item) for item in command], cwd=str(REPO), env=os.environ.copy(),
            stdout=handle, stderr=subprocess.STDOUT,
        ).returncode


def load_manifest():
    path = RESULTS / "preregistered_batch_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "commit": FROZEN_COMMIT,
        "scenario_count": 90,
        "methods": list(METHODS),
        "method_time_budget_s": 15.0,
        "preflight_timeout_s": 60,
        "task_timeout_s": 600,
        "scenario_order_policy": "task_sha256_sort_v1",
        "method_order_policy": (
            "balanced_family_difficulty_seed_latin_rotation_v1"
        ),
    }
    differences = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items() if manifest.get(key) != value
    }
    if differences:
        raise RuntimeError("frozen pre-registration changed: %s" % differences)
    scenarios = manifest.get("scenarios", [])
    if len(scenarios) != 90 or len({item["name"] for item in scenarios}) != 90:
        raise RuntimeError("pre-registration must contain 90 unique scenarios")
    return manifest, scenarios


def scenario_output(scenario):
    return RESULTS / (
        scenario["name"] + "__" + scenario["task_sha256"][:12]
    )


def valid_methods(output, scenario_name):
    found = {}
    runs = output / "closed_loop_runs"
    for path in runs.rglob("trial_result.json") if runs.is_dir() else ():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("result") == "SYSTEM_ERROR":
            continue
        if payload.get("scenario") != scenario_name:
            continue
        method = payload.get("method")
        if method in METHODS:
            if method in found:
                raise RuntimeError(
                    "duplicate valid result for %s/%s" % (scenario_name, method)
                )
            found[method] = path
    return found


def validate_order(output, scenario):
    path = output / "method_order.json"
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "order_policy": "balanced_family_difficulty_seed_latin_rotation_v1",
        "task_sha256": scenario["task_sha256"],
        "methods": scenario["method_order"],
    }
    if payload != expected:
        raise RuntimeError("frozen method order changed for %s" % scenario["name"])


def fusion_oom_confirmed(output):
    pipeline = output / "pipeline.log"
    if not pipeline.is_file():
        return False
    tail = pipeline.read_text(encoding="utf-8", errors="replace")[-20000:]
    return "fuse_slam_maps.py" in tail and "已杀死" in tail


def recover_fusion(scenario, attempt):
    name = scenario["name"]
    output = scenario_output(scenario)
    scene = Path(scenario["scenario_dir"])
    front = output / "front_survey/certified_sensor_survey.npz"
    down = output / "down_survey/certified_sensor_survey.npz"
    manifest = output / "frozen_survey_manifest.json"
    if not all(path.is_file() for path in (front, down, manifest)):
        raise RuntimeError("OOM recovery inputs are incomplete for %s" % name)
    fusion_log = output / "fusion_memory_bounded_supervisor.log"
    command = [
        sys.executable, MEMORY_FUSION, front, down,
        "--output", output / "fused_candidate.npz",
        "--report", output / "fusion_report.json",
        "--sensor", "forward_rgbd", "--sensor", "downward_rgbd",
        "--comparison-independent-survey",
        "--survey-protocol", "task_endpoint_scan_survey_v6_explicit_flush",
        "--survey-manifest", manifest,
        "--observation-support-voxels", "1",
        "--corridor-xy-padding", "1.2",
        "--corridor-z-min", "0.2", "--corridor-z-max", "2.5",
    ]
    if run_logged(command, fusion_log):
        raise RuntimeError("memory-bounded fusion failed for %s" % name)
    audit_command = [
        "rosrun", "uav_continuum_whole_body_v2", "audit_slam_map.py",
        output / "fused_candidate.npz", scene / "oracle_truth.yaml",
        scene / "task.yaml", "--report", output / "map_audit_report.json",
        "--audited-map-output", output / "audited_frozen_map.npz",
    ]
    if run_logged(audit_command, fusion_log):
        raise RuntimeError("schema-v5 map audit failed for %s" % name)
    audit = json.loads((output / "map_audit_report.json").read_text(
        encoding="utf-8"
    ))
    if not audit.get("passed"):
        raise RuntimeError("recovered map did not pass audit for %s" % name)
    record = {
        "attempt": attempt, "scenario": name, "unix_time": time.time(),
        "cause": "kernel_oom_exit_137_in_frozen_27_offset_union",
        "recovery": (
            "semantics_equivalent_one_voxel_halo_precrop_then_identical_union"
        ),
        "equivalence_validation_reference_hash": (
            "dc937cfbd9c0afc900419ff4ce634a52e96ae99c22f5cf212c65b481c11900e4"
        ),
        "candidate_map_hash": audit.get("input_map_hash"),
        "audited_map_hash": audit.get("audited_map_hash"),
        "audit_schema_version": audit.get("audit_schema_version"),
        "audit_passed": audit.get("passed"),
    }
    atomic_json(output / "fusion_oom_recovery.json", record)
    return record


def completed_scenarios(scenarios):
    completed = []
    for scenario in scenarios:
        output = scenario_output(scenario)
        if len(valid_methods(output, scenario["name"])) == len(METHODS):
            completed.append(scenario["name"])
    return completed


def update_running_progress(scenarios, current=None, returncode=None):
    completed = completed_scenarios(scenarios)
    payload = {
        "state": "RUNNING" if returncode is None else "INTERRUPTED",
        "completed": completed,
        "completed_count": len(completed),
        "scenario_count": len(scenarios),
        "valid_trial_count": sum(
            len(valid_methods(scenario_output(item), item["name"]))
            for item in scenarios
        ),
        "resume_policy": "skip_complete_pairs_from_frozen_manifest_v1",
    }
    if current is not None:
        payload["current_scenario"] = current
        if returncode is not None:
            payload["failed_scenario"] = current
            payload["returncode"] = returncode
    atomic_json(RESULTS / "batch_progress.json", payload)
    return payload


def run_pipeline(scenario):
    output = scenario_output(scenario)
    output.mkdir(parents=True, exist_ok=True)
    validate_order(output, scenario)
    command = [
        "rosrun", "uav_continuum_whole_body_v2",
        "run_formal_comparison_pipeline.sh",
        "--scenario-dir", scenario["scenario_dir"],
        "--output-dir", output,
        "--method-time-budget", "15",
        "--preflight-timeout", "60",
        "--task-timeout", "600",
    ]
    return run_logged(command, output / "pipeline.log")


def aggregate_and_finish(manifest, scenarios):
    roots = [scenario_output(item) / "closed_loop_runs" for item in scenarios]
    aggregate = RESULTS / "aggregate"
    command = [
        "rosrun", "uav_continuum_whole_body_v2",
        "aggregate_method_comparison.py", *roots, "--output", aggregate,
    ]
    if run_logged(command, RESULTS / "aggregate.log"):
        raise RuntimeError("final formal aggregate failed")
    eligibility = json.loads((aggregate / "paper_eligibility.json").read_text(
        encoding="utf-8"
    ))
    completeness = json.loads((aggregate / "completeness.json").read_text(
        encoding="utf-8"
    ))
    if not eligibility.get("eligible"):
        raise RuntimeError("final aggregate is not paper-artifact eligible")
    if completeness.get("complete_pairs") != len(scenarios):
        raise RuntimeError("final aggregate does not contain all 90 pairs")
    if manifest.get("require_statistical_ready") and not eligibility.get(
            "statistical_claim_ready"):
        raise RuntimeError("final aggregate has not reached statistical gate")
    atomic_json(RESULTS / "batch_progress.json", {
        "state": "COMPLETE",
        "completed": [item["name"] for item in scenarios],
        "completed_count": len(scenarios),
        "scenario_count": len(scenarios),
        "valid_trial_count": len(scenarios) * len(METHODS),
        "complete_pairs": completeness.get("complete_pairs"),
        "statistical_claim_ready": eligibility.get("statistical_claim_ready"),
        "main_table": str(aggregate / "main_table.csv"),
        "resume_policy": "skip_complete_pairs_from_frozen_manifest_v1",
    })


def main():
    manifest, scenarios = load_manifest()
    status_path = RESULTS / "resilient_supervisor_status.json"
    recoveries = []
    for path in sorted(RESULTS.glob("*/fusion_oom_recovery.json")):
        try:
            recoveries.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            pass
    progress = update_running_progress(scenarios)
    print("E3_DIRECT_RESUME=%d/450 valid, %d/90 pairs" % (
        progress["valid_trial_count"], progress["completed_count"]
    ), flush=True)

    for index, scenario in enumerate(scenarios, 1):
        name = scenario["name"]
        output = scenario_output(scenario)
        validate_order(output, scenario)
        methods = valid_methods(output, name)
        if len(methods) == len(METHODS):
            continue
        print("E3_NEXT_SCENARIO=%d/90 %s existing=%d/5 missing=%s" % (
            index, name, len(methods),
            ",".join(method for method in scenario["method_order"]
                     if method not in methods),
        ), flush=True)
        for attempt in range(1, 51):
            atomic_json(status_path, {
                "state": "RUNNING", "scenario_index": index,
                "scenario": name, "attempt": attempt,
                "recoveries": recoveries,
            })
            returncode = run_pipeline(scenario)
            methods = valid_methods(output, name)
            if returncode == 0 and len(methods) == len(METHODS):
                progress = update_running_progress(scenarios, current=name)
                print("E3_PAIR_COMPLETE=%d/90 valid=%d/450 %s" % (
                    progress["completed_count"], progress["valid_trial_count"], name
                ), flush=True)
                break
            update_running_progress(scenarios, current=name, returncode=returncode)
            if returncode == 137 and fusion_oom_confirmed(output):
                record = recover_fusion(scenario, attempt)
                recoveries.append(record)
                print("E3_FUSION_OOM_RECOVERED=%s" % name, flush=True)
            else:
                event = {
                    "attempt": attempt, "scenario": name,
                    "unix_time": time.time(),
                    "cause": "non_method_system_interruption",
                    "returncode": returncode,
                    "valid_methods_after_attempt": sorted(methods),
                    "action": "retry_same_frozen_scenario_without_parameter_change",
                }
                recoveries.append(event)
                print("E3_SYSTEM_RETRY rc=%d scenario=%s valid=%d/5" % (
                    returncode, name, len(methods)
                ), flush=True)
            time.sleep(10)
        else:
            atomic_json(status_path, {
                "state": "FAILED", "scenario_index": index,
                "scenario": name, "attempt": 50, "recoveries": recoveries,
            })
            return 2

    aggregate_and_finish(manifest, scenarios)
    atomic_json(status_path, {
        "state": "COMPLETE", "scenario_index": 90,
        "scenario": scenarios[-1]["name"], "recoveries": recoveries,
        "batch_progress": json.loads((RESULTS / "batch_progress.json").read_text(
            encoding="utf-8"
        )),
    })
    print("E3_RESILIENT_SUPERVISOR=PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
