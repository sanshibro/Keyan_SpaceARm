#!/usr/bin/env python3
"""Pre-registered, resume-safe multi-scenario formal comparison driver."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import yaml


PACKAGE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE.parents[2]
METHODS = [
    "fixed_morphology", "aet_sequential_minjerk", "informed_rrtstar_wb",
    "global_batch_wb", "rolling_wb_v2",
]
FAMILY_INDEX = {"passage_sweep": 0, "multi_passage": 1, "clutter_reach": 2}
DIFFICULTY_INDEX = {"easy": 0, "medium": 1, "hard": 2}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_state():
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True
    ).strip()
    dirty = subprocess.check_output([
        "git", "status", "--porcelain", "--untracked-files=normal", "--",
        "xtdrone_catkin_ws/src/uav_continuum_whole_body_v2",
        "xtdrone_catkin_ws/src/xtdrone_ego_tools",
        "xtdrone_catkin_ws/src/uav_continuum_opportunity_planner/scripts/arm_shape_sim_node.py",
        "xtdrone_catkin_ws/src/uav_continuum_opportunity_planner/scripts/arm_kinematics_node.py",
    ], cwd=str(PROJECT_ROOT), text=True).splitlines()
    return commit, dirty


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def select_scenarios(args):
    selected = []
    for task_path in sorted(args.scenario_root.rglob("task.yaml")):
        task = yaml.safe_load(task_path.read_text(encoding="utf-8")) or {}
        family = str(task.get("family", ""))
        difficulty = str(task.get("difficulty", ""))
        seed = int(task.get("seed", -1))
        if args.families and family not in args.families:
            continue
        if args.difficulties and difficulty not in args.difficulties:
            continue
        if args.seeds and seed not in args.seeds:
            continue
        oracle = task_path.parent / "oracle_truth.yaml"
        if not oracle.is_file():
            raise SystemExit("Missing oracle_truth.yaml beside %s" % task_path)
        if family not in FAMILY_INDEX or difficulty not in DIFFICULTY_INDEX:
            raise SystemExit("Unregistered family/difficulty in %s" % task_path)
        offset = (seed + DIFFICULTY_INDEX[difficulty]
                  + 3 * FAMILY_INDEX[family]) % len(METHODS)
        method_order = METHODS[offset:] + METHODS[:offset]
        selected.append({
            "name": str(task.get("name", task_path.parent.name)),
            "family": family,
            "difficulty": difficulty,
            "seed": seed,
            "scenario_dir": str(task_path.parent.resolve()),
            "task_sha256": sha256(task_path),
            "oracle_sha256": sha256(oracle),
            "method_order": method_order,
        })
    # A hash-derived order avoids always running the same family at the same
    # wall-clock phase while remaining deterministic and auditable.
    selected.sort(key=lambda item: (item["task_sha256"], item["name"]))
    return selected


def stream_command(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("COMMAND=" + json.dumps(command, ensure_ascii=False) + "\n")
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, start_new_session=True,
        )
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
            return process.wait()
        except KeyboardInterrupt:
            os.killpg(process.pid, signal.SIGINT)
            process.wait(timeout=30)
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Run/resume a pre-registered schema-v6 formal experiment batch"
    )
    parser.add_argument("scenario_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--families", nargs="+")
    parser.add_argument("--difficulties", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--method-time-budget", type=float, default=15.0)
    parser.add_argument("--preflight-timeout", type=int, default=60)
    parser.add_argument("--task-timeout", type=int, default=600)
    parser.add_argument("--require-pairs-per-family", type=int, default=0)
    parser.add_argument("--require-statistical-ready", action="store_true")
    parser.add_argument("--gazebo-gui", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.scenario_root = args.scenario_root.resolve()
    args.output_root = args.output_root.resolve()
    scenarios = select_scenarios(args)
    if not scenarios:
        raise SystemExit("No scenarios match the requested pre-registration")

    counts = {}
    for scenario in scenarios:
        counts[scenario["family"]] = counts.get(scenario["family"], 0) + 1
    if args.require_pairs_per_family > 0:
        deficient = {
            family: count for family, count in counts.items()
            if count < args.require_pairs_per_family
        }
        expected_families = set(args.families or counts)
        missing = sorted(expected_families - set(counts))
        if deficient or missing:
            raise SystemExit(
                "Pre-registration lacks required pairs per family: counts=%r missing=%r"
                % (counts, missing)
            )

    commit, dirty = git_state()
    if dirty and not args.dry_run:
        raise SystemExit("Relevant source tree is dirty: %s" % dirty)
    plan = {
        "schema_version": 1,
        "experiment_class": "schema_v6_closed_loop_method_comparison_batch",
        "commit": commit,
        "scenario_root": str(args.scenario_root),
        "scenario_count": len(scenarios),
        "complete_pairs_required_by_family": args.require_pairs_per_family,
        "require_statistical_ready": args.require_statistical_ready,
        "pairs_by_family": counts,
        "methods": METHODS,
        "method_time_budget_s": args.method_time_budget,
        "preflight_timeout_s": args.preflight_timeout,
        "task_timeout_s": args.task_timeout,
        "scenario_order_policy": "task_sha256_sort_v1",
        "method_order_policy": "balanced_family_difficulty_seed_latin_rotation_v1",
        "scenarios": scenarios,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_root / "preregistered_batch_manifest.json"
    if plan_path.is_file():
        previous = json.loads(plan_path.read_text(encoding="utf-8"))
        if previous != plan:
            raise SystemExit(
                "Existing output has a different pre-registration; use a new directory"
            )
    else:
        atomic_json(plan_path, plan)
    print("PREREGISTERED_SCENARIOS=%d" % len(scenarios))
    print("PAIRS_BY_FAMILY=" + json.dumps(counts, sort_keys=True))
    print("PLANNED_TRIALS=%d" % (len(scenarios) * len(METHODS)))
    if args.dry_run:
        print("FORMAL_BATCH_DRY_RUN=PASS")
        return 0

    progress_path = args.output_root / "batch_progress.json"
    completed = []
    for index, scenario in enumerate(scenarios, 1):
        scenario_output = args.output_root / (
            scenario["name"] + "__" + scenario["task_sha256"][:12]
        )
        scenario_output.mkdir(parents=True, exist_ok=True)
        order_path = scenario_output / "method_order.json"
        order_payload = {
            "order_policy": "balanced_family_difficulty_seed_latin_rotation_v1",
            "task_sha256": scenario["task_sha256"],
            "methods": scenario["method_order"],
        }
        if order_path.is_file():
            if json.loads(order_path.read_text(encoding="utf-8")) != order_payload:
                raise SystemExit("Frozen method order changed for %s" % scenario["name"])
        else:
            atomic_json(order_path, order_payload)
        command = [
            "rosrun", "uav_continuum_whole_body_v2",
            "run_formal_comparison_pipeline.sh",
            "--scenario-dir", scenario["scenario_dir"],
            "--output-dir", str(scenario_output),
            "--method-time-budget", str(args.method_time_budget),
            "--preflight-timeout", str(args.preflight_timeout),
            "--task-timeout", str(args.task_timeout),
        ]
        if args.gazebo_gui:
            command.append("--gazebo-gui")
        print("FORMAL_BATCH_SCENARIO=%d/%d %s" % (
            index, len(scenarios), scenario["name"]
        ))
        returncode = stream_command(command, scenario_output / "pipeline.log")
        if returncode != 0:
            atomic_json(progress_path, {
                "state": "INTERRUPTED", "completed": completed,
                "failed_scenario": scenario["name"], "returncode": returncode,
            })
            raise SystemExit(returncode)
        completed.append(scenario["name"])
        atomic_json(progress_path, {
            "state": "RUNNING", "completed": completed,
            "completed_count": len(completed), "scenario_count": len(scenarios),
        })

    roots = [
        str(args.output_root / (item["name"] + "__" + item["task_sha256"][:12])
            / "closed_loop_runs")
        for item in scenarios
    ]
    aggregate = args.output_root / "aggregate"
    command = [
        "rosrun", "uav_continuum_whole_body_v2",
        "aggregate_method_comparison.py", *roots, "--output", str(aggregate),
    ]
    returncode = stream_command(command, args.output_root / "aggregate.log")
    if returncode != 0:
        raise SystemExit(returncode)
    eligibility = json.loads(
        (aggregate / "paper_eligibility.json").read_text(encoding="utf-8")
    )
    completeness = json.loads(
        (aggregate / "completeness.json").read_text(encoding="utf-8")
    )
    if not eligibility.get("eligible"):
        raise SystemExit("Final aggregate is not paper-artifact eligible")
    if completeness.get("complete_pairs") != len(scenarios):
        raise SystemExit("Final aggregate does not contain every pre-registered pair")
    if args.require_statistical_ready and not eligibility.get(
            "statistical_claim_ready"):
        raise SystemExit("Final aggregate has not reached the statistical gate")
    atomic_json(progress_path, {
        "state": "COMPLETE", "completed": completed,
        "complete_pairs": completeness.get("complete_pairs"),
        "statistical_claim_ready": eligibility.get("statistical_claim_ready"),
        "main_table": str(aggregate / "main_table.csv"),
    })
    print("FORMAL_COMPARISON_BATCH=PASS")
    print("MAIN_TABLE=" + str(aggregate / "main_table.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
