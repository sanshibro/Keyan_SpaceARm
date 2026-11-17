#!/usr/bin/env python3
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import yaml

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "src"))

from uav_continuum_whole_body_v2.map_audit import audit_snapshot, sha256  # noqa: E402
from uav_continuum_whole_body_v2.mapping import MapSnapshot  # noqa: E402
from uav_continuum_whole_body_v2.configuration import (  # noqa: E402
    load_planner_configuration,
)
from uav_continuum_whole_body_v2.types import ArmModel  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Audit a recorded SLAM map and optionally promote it for formal experiments"
    )
    parser.add_argument("map", type=Path)
    parser.add_argument("oracle", type=Path)
    parser.add_argument("task", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--audited-map-output", type=Path)
    parser.add_argument(
        "--config", type=Path,
        default=PACKAGE / "config" / "planner_v2_realtime.yaml",
    )
    parser.add_argument("--surface-error-p95-max", type=float, default=0.12)
    parser.add_argument("--surface-recall-min", type=float, default=None,
                        help="deprecated schema-v1 option; reported but not gated")
    parser.add_argument("--false-positive-max", type=float, default=0.10)
    parser.add_argument("--corridor-coverage-min", type=float, default=0.98,
                        help="minimum representative whole-body route coverage")
    parser.add_argument("--state-coverage-min", type=float, default=1.0,
                        help="minimum coverage of every sampled full-body state")
    parser.add_argument("--terminal-coverage-min", type=float, default=1.0,
                        help="minimum exact terminal full-body coverage")
    args = parser.parse_args()

    snapshot = MapSnapshot.load(args.map)
    oracle = yaml.safe_load(args.oracle.read_text(encoding="utf-8")) or {}
    task = yaml.safe_load(args.task.read_text(encoding="utf-8")) or {}
    _, _planner, limits, _mass = load_planner_configuration(args.config)
    report = audit_snapshot(snapshot, oracle, task, thresholds={
        "surface_error_p95_max_m": args.surface_error_p95_max,
        "occupied_false_positive_rate_max": args.false_positive_max,
        "reference_whole_body_observed_coverage_min": args.corridor_coverage_min,
        "reference_state_observed_coverage_min": args.state_coverage_min,
        "terminal_whole_body_observed_coverage_min": args.terminal_coverage_min,
    }, arm=ArmModel(), limits=limits)
    report.update({
        "input_map": str(args.map.resolve()),
        "input_map_hash": snapshot.map_hash,
        "oracle_file": str(args.oracle.resolve()),
        "oracle_sha256": sha256(args.oracle),
        "task_file": str(args.task.resolve()),
        "task_sha256": sha256(args.task),
        "planner_config": str(args.config.resolve()),
        "planner_config_sha256": sha256(args.config),
        "audit_schema_version": 5,
    })
    if report["passed"] and args.audited_map_output:
        metadata = dict(snapshot.metadata)
        metadata.update({
            "formal_experiment_allowed": True,
            "truth_derived": False,
            "map_audit_schema_version": 5,
            "map_audit_oracle_sha256": report["oracle_sha256"],
            "map_audit_task_sha256": report["task_sha256"],
            "map_audit_metrics": report["metrics"],
            "map_audit_limits": report["limits"],
        })
        audited = replace(snapshot, source="audited_frozen_slam", metadata=metadata)
        audited.save(args.audited_map_output)
        report["audited_map_output"] = str(args.audited_map_output.resolve())
        report["audited_map_hash"] = audited.map_hash
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
