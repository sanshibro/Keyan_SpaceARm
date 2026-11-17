#!/usr/bin/env python3
"""Deterministically rebuild completed ten-trial E3 snapshots.

This is an evidence-tool repair only.  It never edits a trial, manifest, map,
or method output.  Each historical snapshot is reconstructed from the first N
valid trial results ordered by their original filesystem modification time.
The ten-trial milestones are valid only when those N records form complete
five-method scenario pairs.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def load_records(root):
    valid = []
    system_errors = []
    keys = set()
    for path in root.rglob("trial_result.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        stamp = path.stat().st_mtime
        if payload.get("result") == "SYSTEM_ERROR":
            system_errors.append((stamp, path))
            continue
        scenario = str(payload.get("scenario", ""))
        method = str(payload.get("method", ""))
        if not scenario or not method:
            continue
        key = (scenario, method)
        if key in keys:
            raise RuntimeError("duplicate valid trial key: %s" % (key,))
        keys.add(key)
        valid.append((stamp, scenario, method, path))
    valid.sort(key=lambda item: (item[0], str(item[3])))
    system_errors.sort(key=lambda item: (item[0], str(item[1])))
    return valid, system_errors


def rebuild(root, milestone, records, system_errors):
    selected = records[:milestone]
    counts = {}
    for _, scenario, _, _ in selected:
        counts[scenario] = counts.get(scenario, 0) + 1
    incomplete = {key: value for key, value in counts.items() if value != 5}
    if len(selected) != milestone or incomplete or len(counts) != milestone // 5:
        raise RuntimeError(
            "milestone %d is not a complete paired prefix: %s" %
            (milestone, incomplete)
        )
    roots = sorted({str(path.parent.parent) for _, _, _, path in selected})
    output = root / "progress_snapshots" / ("trials_%04d" % milestone)
    output.mkdir(parents=True, exist_ok=True)
    command = [
        "rosrun", "uav_continuum_whole_body_v2",
        "aggregate_method_comparison.py", *roots, "--output", str(output),
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(
            "aggregate failed at %d: %s%s" %
            (milestone, result.stdout, result.stderr)
        )
    cutoff = selected[-1][0]
    payload = {
        "milestone": milestone,
        "valid_trial_count_at_snapshot": milestone,
        "system_error_files_seen": sum(stamp <= cutoff for stamp, _ in system_errors),
        "aggregate_returncode": result.returncode,
        "aggregate_stdout": result.stdout,
        "aggregate_stderr": result.stderr,
        "main_table": str(output / "main_table.csv"),
        "created_unix_time": cutoff,
        "reconstructed_unix_time": time.time(),
        "reconstructed_from_original_trial_mtimes": True,
        "repair_reason": "resume_monitor_attempted_to_overwrite_immutable_snapshots",
    }
    atomic_json(output / "snapshot_status.json", payload)
    print("E3_PROGRESS_SNAPSHOT_REBUILT=%d" % milestone, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--through", type=int, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.through <= 0 or args.through % 10:
        raise SystemExit("--through must be a positive multiple of ten")
    records, system_errors = load_records(root)
    if len(records) < args.through:
        raise SystemExit("only %d valid trials exist" % len(records))
    for milestone in range(10, args.through + 1, 10):
        rebuild(root, milestone, records, system_errors)
    print("E3_PROGRESS_SNAPSHOT_REBUILD=PASS", flush=True)


if __name__ == "__main__":
    main()
