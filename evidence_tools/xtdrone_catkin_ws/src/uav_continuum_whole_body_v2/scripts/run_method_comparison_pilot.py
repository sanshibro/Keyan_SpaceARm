#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import subprocess
import sys

PACKAGE = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Resume-safe paired offline comparison pilot")
    parser.add_argument("scenario_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--time-budget", type=float, default=3.0)
    parser.add_argument(
        "--reference-time-budget", type=float, default=60.0,
        help="Common front-end preparation/search budget; excluded from each method budget",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument(
        "--map-name", default="development_map.npz",
        help="MapSnapshot filename inside every scenario directory",
    )
    args = parser.parse_args()
    scenarios = sorted(path.parent for path in args.scenario_root.rglob("task.yaml"))
    if args.limit > 0:
        scenarios = scenarios[:args.limit]
    args.output_root.mkdir(parents=True, exist_ok=True)
    log_root = args.output_root / "_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    completed = 0
    failed = []
    for index, scenario in enumerate(scenarios, 1):
        output = args.output_root / scenario.name
        summary = output / "summary.json"
        if summary.is_file():
            payload = json.loads(summary.read_text(encoding="utf-8"))
            if len(payload.get("reports", [])) == 5:
                print("[%d/%d] SKIP complete %s" % (index, len(scenarios), scenario.name))
                completed += 1
                continue
        if output.exists():
            failed.append({"scenario": scenario.name, "reason": "partial_output_requires_audit"})
            print("[%d/%d] HOLD partial %s" % (index, len(scenarios), scenario.name))
            continue
        command = [
            sys.executable, str(PACKAGE / "scripts" / "run_offline_method_comparison.py"),
            str(scenario), "--method", "all", "--time-budget", str(args.time_budget),
            "--reference-time-budget", str(args.reference_time_budget),
            "--output", str(output), "--map", str(scenario / args.map_name),
        ]
        if args.formal:
            command.append("--formal")
        print("[%d/%d] RUN %s" % (index, len(scenarios), scenario.name), flush=True)
        log_path = log_root / (scenario.name + ".log")
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command, check=False, stdout=log, stderr=subprocess.STDOUT,
            )
        if result.returncode == 0:
            completed += 1
            print("[%d/%d] DONE %s log=%s" % (
                index, len(scenarios), scenario.name, log_path,
            ))
        else:
            failed.append({
                "scenario": scenario.name, "returncode": result.returncode,
                "log": str(log_path),
            })
    report = {
        "scenarios": len(scenarios), "completed": completed,
        "failed": failed, "formal": args.formal,
    }
    (args.output_root / "pilot_status.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
