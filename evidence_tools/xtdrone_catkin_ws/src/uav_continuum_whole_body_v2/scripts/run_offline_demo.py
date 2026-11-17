#!/usr/bin/env python3
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np
import yaml

PACKAGE = Path(__file__).resolve().parents[1]
REPOSITORY = PACKAGE.parents[2]
sys.path.insert(0, str(PACKAGE / "src"))

from uav_continuum_whole_body_v2.configuration import load_planner_configuration  # noqa: E402
from uav_continuum_whole_body_v2.environment import BoxWorld, HybridDistanceWorld  # noqa: E402
from uav_continuum_whole_body_v2.optimizer import WholeBodyOptimizer  # noqa: E402
from uav_continuum_whole_body_v2.seed_planner import WholeBodySeedPlanner  # noqa: E402
from uav_continuum_whole_body_v2.types import ArmModel  # noqa: E402
from uav_continuum_whole_body_v2.validator import SweptVolumeValidator  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run a deterministic, ROS-free V2 numerical smoke test")
    parser.add_argument("--time-budget", type=float, default=4.0)
    parser.add_argument(
        "--config", type=Path, default=PACKAGE / "config" / "planner_v2.yaml",
        help="Planner YAML to benchmark.",
    )
    parser.add_argument("--output", type=Path, default=REPOSITORY / "xtdrone_plan" / "experiments" / "v2_offline_smoke.json")
    args = parser.parse_args()
    with (PACKAGE / "config" / "scenario_3c_unified.yaml").open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    passage = [item for item in manifest["task_boxes"] if item["name"].startswith("p1_")]
    world = HybridDistanceWorld(BoxWorld.from_dicts(passage))
    _, configured, limits, mass = load_planner_configuration(args.config)
    config = replace(configured, optimizer_time_budget=args.time_budget)
    arm = ArmModel()
    reference = np.zeros((24, 6), dtype=float)
    reference[:, 0] = -0.92
    reference[:, 1] = np.linspace(-0.65, -3.0, len(reference))
    reference[:, 2] = 1.30
    reference[:, 3] = -np.pi / 2.0
    started = time.monotonic()
    seed = WholeBodySeedPlanner(world, arm, mass, limits, config).plan(reference, reference[0])
    seed_time = time.monotonic() - started
    validator = SweptVolumeValidator(world, arm, mass, limits, config)
    optimize_started = time.monotonic()
    result = WholeBodyOptimizer(world, arm, mass, limits, config, validator).optimize(seed, reference)
    optimize_validate_time = time.monotonic() - optimize_started
    report = {
        "success": result.success, "status": result.status, "elapsed": result.elapsed,
        "seed_time": seed_time, "optimize_validate_time": optimize_validate_time,
        "total_planning_time": seed_time + optimize_validate_time,
        "iterations": result.iterations, "objective": result.objective,
        "risk_knots": seed.risk_knots, "risk_spans": seed.risk_spans,
        "shape_mode": seed.diagnostics.get("shape_mode", "theta_phi"),
        "fixed_phi_deg": seed.diagnostics.get("fixed_phi_deg"),
        "expanded": seed.diagnostics.get("expanded"),
        "validation": result.validation.metrics,
        "failure_reason": result.validation.failure_reason,
        "diagnostics": result.diagnostics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if result.success else 2


if __name__ == "__main__":
    raise SystemExit(main())
