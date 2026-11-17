#!/usr/bin/env python3
"""Create immutable aggregate snapshots after every ten valid E3 trials."""

import argparse
import json
from pathlib import Path
import subprocess
import time


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def valid_trials(root):
    trials = {}
    system_errors = 0
    for path in root.rglob("trial_result.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("result") == "SYSTEM_ERROR":
            system_errors += 1
            continue
        scenario = str(payload.get("scenario", ""))
        method = str(payload.get("method", ""))
        if scenario and method:
            trials[(scenario, method)] = path
    return trials, system_errors


def snapshot(root, milestone, actual_count, system_errors):
    roots = sorted({str(path.parent.parent) for path in valid_trials(root)[0].values()})
    output = root / "progress_snapshots" / ("trials_%04d" % milestone)
    output.mkdir(parents=True, exist_ok=True)
    command = [
        "rosrun", "uav_continuum_whole_body_v2",
        "aggregate_method_comparison.py", *roots, "--output", str(output),
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    payload = {
        "milestone": milestone,
        "valid_trial_count_at_snapshot": actual_count,
        "system_error_files_seen": system_errors,
        "aggregate_returncode": result.returncode,
        "aggregate_stdout": result.stdout,
        "aggregate_stderr": result.stderr,
        "main_table": str(output / "main_table.csv"),
        "created_unix_time": time.time(),
    }
    atomic_json(output / "snapshot_status.json", payload)
    print("E3_PROGRESS_SNAPSHOT=%d actual=%d returncode=%d" % (
        milestone, actual_count, result.returncode
    ), flush=True)


def completed_milestones(root):
    """Return immutable milestones already produced successfully.

    A resumed monitor must never aggregate the current (possibly incomplete)
    batch back into an older snapshot directory.
    """
    completed = []
    snapshots = root / "progress_snapshots"
    for status in snapshots.glob("trials_*/snapshot_status.json"):
        try:
            payload = json.loads(status.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("aggregate_returncode") != 0:
            continue
        milestone = payload.get("milestone")
        if isinstance(milestone, int) and milestone > 0:
            completed.append(milestone)
    return sorted(set(completed))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--interval", type=float, default=20.0)
    parser.add_argument("--target", type=int, default=450)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    existing = completed_milestones(root)
    last = existing[-1] if existing else 0
    while True:
        trials, system_errors = valid_trials(root)
        count = len(trials)
        milestone = (count // 10) * 10
        while last + 10 <= milestone:
            last += 10
            snapshot(root, last, count, system_errors)
        progress = root / "batch_progress.json"
        if count >= args.target:
            break
        if progress.is_file():
            try:
                if json.loads(progress.read_text(encoding="utf-8")).get("state") == "COMPLETE":
                    break
            except Exception:
                pass
        time.sleep(args.interval)
    trials, system_errors = valid_trials(root)
    atomic_json(root / "progress_snapshots" / "monitor_complete.json", {
        "valid_trial_count": len(trials),
        "system_error_files_seen": system_errors,
        "target": args.target,
        "complete": len(trials) >= args.target,
    })
    print("E3_PROGRESS_MONITOR_COMPLETE=%d/%d" % (len(trials), args.target))


if __name__ == "__main__":
    main()
