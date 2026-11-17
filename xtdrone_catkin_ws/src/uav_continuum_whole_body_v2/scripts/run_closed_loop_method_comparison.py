#!/usr/bin/env python3
"""Resume-safe paired closed-loop comparison batch driver.

Every method/scenario pair is attempted once. Algorithm failures remain data;
only trials without an auditable trial_result.json are infrastructure failures.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import yaml


PACKAGE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE.parents[2]
METHODS = (
    "fixed_morphology", "aet_sequential_minjerk", "informed_rrtstar_wb",
    "global_batch_wb", "rolling_wb_v2",
)


def method_order(scenario_name, methods):
    """Deterministically rotate order to distribute simulator time drift."""
    digest = hashlib.sha256(scenario_name.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], "big") % len(methods)
    return methods[offset:] + methods[:offset]


def load_existing(output_root):
    existing = {}
    for result_path in sorted(output_root.rglob("trial_result.json")):
        try:
            trial = json.loads(result_path.read_text(encoding="utf-8"))
            manifest_path = result_path.parent / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) \
                if manifest_path.is_file() else {}
        except (OSError, ValueError):
            continue
        scenario = str(trial.get("scenario", manifest.get("scenario_name", "")))
        method = str(trial.get("method", manifest.get("method", "")))
        if scenario and method:
            existing.setdefault((scenario, method), []).append({
                "result": trial.get("result", "SYSTEM_ERROR"),
                "run_dir": str(result_path.parent),
            })
    return existing


def git_state():
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True
    ).strip()
    dirty = subprocess.check_output(
        [
            "git", "status", "--porcelain", "--untracked-files=normal", "--",
            "xtdrone_catkin_ws/src/uav_continuum_whole_body_v2",
            "xtdrone_catkin_ws/src/xtdrone_ego_tools",
            "xtdrone_catkin_ws/src/uav_continuum_opportunity_planner/scripts/arm_shape_sim_node.py",
            "xtdrone_catkin_ws/src/uav_continuum_opportunity_planner/scripts/arm_kinematics_node.py",
        ],
        cwd=str(PROJECT_ROOT), text=True,
    ).splitlines()
    return commit, dirty


def main():
    parser = argparse.ArgumentParser(
        description="Resume-safe paired PX4/Gazebo closed-loop comparison"
    )
    parser.add_argument("scenario_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--families", nargs="+")
    parser.add_argument("--difficulties", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--map-name", default="development_map.npz")
    parser.add_argument("--method-time-budget", type=float, default=10.0)
    parser.add_argument("--preflight-timeout", type=int, default=300)
    parser.add_argument("--task-timeout", type=int, default=900)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--gazebo-gui", action="store_true")
    args = parser.parse_args()

    methods = list(dict.fromkeys(args.methods))
    scenarios = []
    for task_path in sorted(args.scenario_root.rglob("task.yaml")):
        task = yaml.safe_load(task_path.read_text(encoding="utf-8")) or {}
        if args.families and task.get("family") not in args.families:
            continue
        if args.difficulties and task.get("difficulty") not in args.difficulties:
            continue
        if args.seeds and int(task.get("seed", -1)) not in args.seeds:
            continue
        scenarios.append((task_path.parent, task))
    if args.limit > 0:
        scenarios = scenarios[:args.limit]
    if not scenarios:
        raise SystemExit("No task.yaml matched the requested trial set")

    args.output_root.mkdir(parents=True, exist_ok=True)
    log_root = args.output_root / "_batch_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    commit, dirty = git_state()
    batch_manifest = {
        "created_wall_time": time.time(),
        "commit": commit,
        "reproducibility_scope_dirty": dirty,
        "formal": args.formal,
        "scenario_root": str(args.scenario_root.resolve()),
        "scenario_count": len(scenarios),
        "methods": methods,
        "method_time_budget_s": args.method_time_budget,
        "preflight_timeout_s": args.preflight_timeout,
        "task_timeout_s": args.task_timeout,
        "order_policy": "deterministic_sha256_cyclic_rotation",
    }
    (args.output_root / "batch_manifest.json").write_text(
        json.dumps(batch_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    existing = load_existing(args.output_root)
    scheduled = len(scenarios) * len(methods)
    completed = 0
    infrastructure_failures = []
    results = []
    index = 0
    for scenario_dir, task in scenarios:
        scenario_name = str(task.get("name", scenario_dir.name))
        map_path = scenario_dir / args.map_name
        for method in method_order(scenario_name, methods):
            index += 1
            previous = [
                item for item in existing.get((scenario_name, method), [])
                if item.get("result") != "SYSTEM_ERROR"
            ]
            if previous:
                completed += 1
                results.append({
                    "scenario": scenario_name, "method": method,
                    "status": "SKIPPED_EXISTING", "existing": previous,
                })
                print("[%d/%d] SKIP %s %s" % (
                    index, scheduled, scenario_name, method
                ), flush=True)
                continue
            if not map_path.is_file():
                failure = {
                    "scenario": scenario_name, "method": method,
                    "reason": "MAP_NOT_READY", "map": str(map_path),
                }
                infrastructure_failures.append(failure)
                results.append(failure)
                print("[%d/%d] MAP_NOT_READY %s" % (
                    index, scheduled, map_path
                ), flush=True)
                continue
            command = [
                str(PACKAGE / "scripts" / "run_method_comparison_closed_loop.sh"),
                "--method", method,
                "--scenario-dir", str(scenario_dir),
                "--map", str(map_path),
                "--method-time-budget", str(args.method_time_budget),
                "--preflight-timeout", str(args.preflight_timeout),
                "--task-timeout", str(args.task_timeout),
                "--output-root", str(args.output_root),
            ]
            if args.formal:
                command.append("--formal")
            if args.gazebo_gui:
                command.append("--gazebo-gui")
            log_path = log_root / (scenario_name + "__" + method + ".log")
            print("[%d/%d] RUN %s %s" % (
                index, scheduled, scenario_name, method
            ), flush=True)
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, check=False
                )
            after = [
                item for item in load_existing(args.output_root).get(
                    (scenario_name, method), []
                ) if item.get("result") != "SYSTEM_ERROR"
            ]
            if after:
                completed += 1
                item = {
                    "scenario": scenario_name, "method": method,
                    "status": after[-1]["result"], "returncode": process.returncode,
                    "run_dir": after[-1]["run_dir"], "log": str(log_path),
                }
                print("[%d/%d] RECORDED %s rc=%d" % (
                    index, scheduled, item["status"], process.returncode
                ), flush=True)
            else:
                item = {
                    "scenario": scenario_name, "method": method,
                    "status": "INFRASTRUCTURE_INCOMPLETE",
                    "returncode": process.returncode, "log": str(log_path),
                }
                infrastructure_failures.append(item)
                print("[%d/%d] INFRASTRUCTURE_INCOMPLETE log=%s" % (
                    index, scheduled, log_path
                ), flush=True)
            results.append(item)

    status = {
        "scheduled": scheduled, "completed": completed,
        "infrastructure_failures": infrastructure_failures,
        "results": results,
    }
    (args.output_root / "batch_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "scheduled": scheduled, "completed": completed,
        "infrastructure_failure_count": len(infrastructure_failures),
    }, indent=2, sort_keys=True))
    return 0 if not infrastructure_failures else 2


if __name__ == "__main__":
    sys.exit(main())
