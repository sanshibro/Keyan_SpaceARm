#!/usr/bin/env python3
"""Verify publication payload and independently recalculate the paired effect."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
REL = Path("xtdrone_plan/experiments/method_comparison")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-index", action="store_true", help="Create SHA256SUMS after packaging; otherwise verify it")
    args = parser.parse_args()
    index = ROOT / "SHA256SUMS"
    if args.write_index:
        entries = []
        for path in sorted(ROOT.rglob("*")):
            relative = path.relative_to(ROOT)
            if (path.is_file() and path != index and ".git" not in relative.parts
                    and "__pycache__" not in relative.parts and ".reproduction" not in relative.parts):
                entries.append(sha256(path) + "  " + str(relative))
        index.write_text("\n".join(entries) + "\n", encoding="utf-8")
    failures = []
    count = 0
    for line in index.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        count += 1
        if not path.is_file() or sha256(path) != expected:
            failures.append(relative)
    for path in sorted((ROOT / "datasets").glob("*.manifest.json")):
        data = json.loads(path.read_text())
        if any(sha256(ROOT / part["path"]) != part["sha256"] for part in data["parts"]):
            failures.append(str(path.relative_to(ROOT)))
    formal = ROOT / REL / "v21_e3_final_450_commite62b6f2"
    with (formal / "paper_artifacts/source_data/main_results.csv").open(newline="") as handle:
        table = list(csv.DictReader(handle))
    expected = {"fixed_morphology": 55, "aet_sequential_minjerk": 61, "informed_rrtstar_wb": 68,
                "global_batch_wb": 59, "rolling_wb_v2": 89}
    if len(table) != 5 or {row["method"]: int(row["success"]) for row in table} != expected:
        failures.append("main results differ from frozen counts")
    pairs = ROOT / REL / "v21_e2_15scene_commitd9e01a2/audit/paired_runs.csv"
    with pairs.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    reductions = [1.0 - float(r["v21_jerk"]) / float(r["frozen_jerk"]) for r in rows]
    effect = statistics.median(reductions)
    if len(rows) != 15 or abs(effect - 0.727837519208926) > 1e-10:
        failures.append("paired jerk effect differs from original evidence")
    result = {"passed": not failures, "indexed_files": count, "failures": failures,
              "formal_successes": expected, "paired_handoff_tasks": len(rows),
              "paired_jerk_reduction_median": effect}
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
