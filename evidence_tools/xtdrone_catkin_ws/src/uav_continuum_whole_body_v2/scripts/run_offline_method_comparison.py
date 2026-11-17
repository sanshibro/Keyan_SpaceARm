#!/usr/bin/env python3
"""Run an auditable ROS-free algorithm pilot on one frozen map snapshot."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

PACKAGE = Path(__file__).resolve().parents[1]
REPOSITORY = PACKAGE.parents[2]
sys.path.insert(0, str(PACKAGE / "src"))

import numpy as np
import yaml

from uav_continuum_whole_body_v2.comparison_methods import (  # noqa: E402
    ComparisonProblem, METHOD_NAMES, create_method,
)
from uav_continuum_whole_body_v2.configuration import load_planner_configuration  # noqa: E402
from uav_continuum_whole_body_v2.environment import BoxWorld, HybridDistanceWorld  # noqa: E402
from uav_continuum_whole_body_v2.mapping import MapSnapshot  # noqa: E402
from uav_continuum_whole_body_v2.horizon import local_horizon  # noqa: E402
from uav_continuum_whole_body_v2.reference_planner import (  # noqa: E402
    VoxelAStarReferencePlanner, densify_path, path_states,
)
from uav_continuum_whole_body_v2.types import ArmModel  # noqa: E402
from uav_continuum_whole_body_v2.types import OptimizationResult, TrajectoryData  # noqa: E402
from uav_continuum_whole_body_v2.validator import SweptVolumeValidator  # noqa: E402


class TrajectoryReplay:
    """Dense trajectory adapter for an independent oracle re-evaluation."""

    def __init__(self, trajectory):
        self.trajectory = trajectory
        self.duration = trajectory.duration
        self.span_boundaries = trajectory.times

    def evaluate(self, times, derivative=0):
        source = (
            self.trajectory.states if derivative == 0 else
            self.trajectory.velocities if derivative == 1 else
            self.trajectory.accelerations if derivative == 2 else
            self.trajectory.jerks if derivative == 3 else
            np.zeros_like(self.trajectory.states)
        )
        times = np.asarray(times, dtype=float)
        return np.column_stack([
            np.interp(times, self.trajectory.times, source[:, dimension])
            for dimension in range(source.shape[1])
        ])


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError("not JSON serializable: %r" % (type(value),))


def git_state():
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(REPOSITORY), text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal", "--",
         "xtdrone_catkin_ws/src/uav_continuum_whole_body_v2",
         "xtdrone_catkin_ws/src/xtdrone_ego_tools",
         "xtdrone_catkin_ws/src/uav_continuum_opportunity_planner/scripts/arm_shape_sim_node.py",
         "xtdrone_catkin_ws/src/uav_continuum_opportunity_planner/scripts/arm_kinematics_node.py"],
        cwd=str(REPOSITORY), text=True,
    ).splitlines()
    return commit, status


def failure_category(result, oracle=None):
    if result.trajectory is None:
        reason = result.validation.failure_reason
        return "PLAN_TIMEOUT" if "time" in reason else "NO_PATH"
    validation = result.validation
    if not validation.feasible:
        if validation.failure_reason in ("body_collision", "arm_collision"):
            return "VALIDATION_REJECT"
        return "DYNAMIC_VIOLATION"
    if oracle is not None and not oracle.feasible:
        if oracle.failure_reason in ("body_collision", "arm_collision"):
            return "ORACLE_COLLISION"
        if oracle.failure_reason == "terminal_tip_error":
            return "TIP_TASK_FAIL"
        return "DYNAMIC_VIOLATION"
    return "NONE"


def rolling_sequence(problem, maximum_windows=20):
    method = create_method("rolling_wb_v2")
    current = problem.current_state.copy()
    parts = []
    window_reports = []
    total_objective = 0.0
    total_elapsed = 0.0
    last_result = None
    for window_index in range(maximum_windows):
        horizon, nearest = local_horizon(
            problem.reference_states, current, problem.config.horizon_distance
        )
        terminal_active = (
            np.linalg.norm(horizon[-1, :3] - problem.terminal_state[:3]) <= 0.10
        )
        if terminal_active:
            horizon[-1] = problem.terminal_state
        local_problem = ComparisonProblem(
            reference_states=horizon, current_state=current,
            terminal_state=problem.terminal_state if terminal_active else None,
            terminal_target=problem.terminal_target if terminal_active else None,
            terminal_tolerance=problem.terminal_tolerance,
            world=problem.world, arm=problem.arm, mass=problem.mass,
            limits=problem.limits, config=problem.config,
            method_time_budget=problem.method_time_budget,
            random_seed=problem.random_seed + window_index,
        )
        result = method.plan(local_problem)
        last_result = result
        total_elapsed += result.elapsed
        window_reports.append({
            "window": window_index, "nearest": nearest,
            "terminal_active": bool(terminal_active), "success": bool(result.success),
            "elapsed": result.elapsed,
            "failure": result.validation.failure_reason,
            "diagnostics": dict(result.diagnostics),
            "start_xyz": current[:3].tolist(),
            "end_xyz": None if result.trajectory is None
            else result.trajectory.states[-1, :3].tolist(),
        })
        if not result.success or result.trajectory is None:
            result.elapsed = total_elapsed
            result.diagnostics["rolling_windows"] = window_reports
            return result
        parts.append(result.trajectory)
        total_objective += result.objective
        next_state = result.trajectory.states[-1].copy()
        if np.linalg.norm(next_state[:3] - current[:3]) < 0.05:
            result.success = False
            result.status = "rejected_no_rolling_progress"
            result.validation.feasible = False
            result.validation.failure_reason = "no_rolling_progress"
            result.elapsed = total_elapsed
            result.diagnostics["rolling_windows"] = window_reports
            return result
        current = next_state
        if terminal_active:
            break
    if last_result is None or np.linalg.norm(current[:3] - problem.terminal_state[:3]) > 0.10:
        if last_result is None:
            return create_method("rolling_wb_v2").plan(problem)
        last_result.success = False
        last_result.status = "rejected_window_limit"
        last_result.validation.feasible = False
        last_result.validation.failure_reason = "window_limit"
        last_result.elapsed = total_elapsed
        last_result.diagnostics["rolling_windows"] = window_reports
        return last_result
    times = []
    states = []
    velocities = []
    accelerations = []
    jerks = []
    offset = 0.0
    for part_index, part in enumerate(parts):
        start = 0 if part_index == 0 else 1
        times.extend((part.times[start:] + offset).tolist())
        states.extend(part.states[start:])
        velocities.extend(part.velocities[start:])
        accelerations.extend(part.accelerations[start:])
        jerks.extend(part.jerks[start:])
        offset += part.duration
    trajectory = TrajectoryData(
        times=np.asarray(times), states=np.asarray(states),
        velocities=np.asarray(velocities), accelerations=np.asarray(accelerations),
        jerks=np.asarray(jerks), source="rolling_wb_v2_offline_sequence",
        objective=total_objective,
        metadata={"windows": len(parts)},
    )
    last_result.trajectory = trajectory
    last_result.objective = total_objective
    last_result.elapsed = total_elapsed
    last_result.diagnostics["rolling_windows"] = window_reports
    last_result.diagnostics["window_count"] = len(parts)
    return last_result


def evaluate(method_name, problem, oracle_world):
    precondition = None
    if method_name == "fixed_morphology":
        # Fixed-Morphology is an engineering lower bound with one morphology
        # selected on the development set.  Establish it before the task timer
        # and make the initial state truthful instead of introducing an
        # impossible theta/phi jump at t=0.
        fixed_shape = problem.terminal_state[[4, 5]].copy()
        current = problem.current_state.copy()
        current[[4, 5]] = fixed_shape
        reference = problem.reference_states.copy()
        reference[:, 4] = fixed_shape[0]
        reference[:, 5] = fixed_shape[1]
        problem = replace(problem, current_state=current, reference_states=reference)
        precondition = {
            "excluded_from_task_timer": True,
            "fixed_shape_deg": np.rad2deg(fixed_shape).tolist(),
            "initial_shape_change_deg": np.rad2deg(
                fixed_shape - np.asarray([0.0, 0.0])
            ).tolist(),
        }
    started = time.monotonic()
    result = (
        rolling_sequence(problem) if method_name == "rolling_wb_v2"
        else create_method(method_name).plan(problem)
    )
    wall_time = time.monotonic() - started
    oracle = None
    metrics = {}
    if result.trajectory is not None:
        spline_times = result.trajectory.times
        jerk_sq = np.sum(result.trajectory.jerks ** 2, axis=1)
        raw_jerk_energy = float(np.trapz(jerk_sq, spline_times))
        path_length = float(np.sum(np.linalg.norm(np.diff(
            result.trajectory.states[:, :3], axis=0
        ), axis=1)))
        metrics["jerk_energy_raw"] = raw_jerk_energy
        metrics["jerk_energy"] = raw_jerk_energy / max(path_length, 1e-6)
        metrics["command_path_length_m"] = path_length
        metrics["theta_travel_rad"] = float(np.sum(np.abs(np.diff(
            result.trajectory.states[:, 4]
        ))))
        metrics["phi_travel_rad"] = float(np.sum(np.abs(np.diff(
            np.unwrap(result.trajectory.states[:, 5])
        ))))
        oracle = SweptVolumeValidator(
            oracle_world, problem.arm, problem.mass, problem.limits,
            problem.config, terminal_target=problem.terminal_target,
            terminal_tolerance=problem.terminal_tolerance,
        ).validate(TrajectoryReplay(result.trajectory))
        metrics.update({"oracle_" + key: value for key, value in oracle.metrics.items()})
    success = bool(result.success and oracle is not None and oracle.feasible)
    report = {
        "method": method_name,
        "success": success,
        "planning_success": bool(result.success),
        "status": result.status,
        "failure_category": failure_category(result, oracle),
        "planning_time_s": wall_time,
        "reported_solver_time_s": result.elapsed,
        "trajectory_duration_s": None if result.trajectory is None else result.trajectory.duration,
        "planner_metrics": result.validation.metrics,
        "oracle_metrics": None if oracle is None else oracle.metrics,
        "derived_metrics": metrics,
        "diagnostics": result.diagnostics,
    }
    if precondition is not None:
        report["precondition"] = precondition
    return report, result
def main():
    parser = argparse.ArgumentParser(description="Offline paired method comparison pilot")
    parser.add_argument("scenario_dir", type=Path)
    parser.add_argument("--method", choices=list(METHOD_NAMES) + ["all"], default="all")
    parser.add_argument("--config", type=Path,
                        default=PACKAGE / "config" / "planner_v2_realtime.yaml")
    parser.add_argument("--baseline-registry", type=Path,
                        default=PACKAGE / "config" / "baseline_registry.yaml")
    parser.add_argument("--time-budget", type=float, default=3.0)
    parser.add_argument("--reference-time-budget", type=float, default=15.0)
    parser.add_argument(
        "--map", type=Path,
        help="Planner-visible MapSnapshot; defaults to scenario development_map.npz",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args()
    task_path = args.scenario_dir / "task.yaml"
    oracle_path = args.scenario_dir / "oracle_truth.yaml"
    map_path = args.map or (args.scenario_dir / "development_map.npz")
    if not map_path.is_file():
        raise SystemExit("MAP_NOT_READY: snapshot does not exist: %s" % map_path)
    with task_path.open("r", encoding="utf-8") as handle:
        task = yaml.safe_load(handle) or {}
    with oracle_path.open("r", encoding="utf-8") as handle:
        oracle_data = yaml.safe_load(handle) or {}
    snapshot = MapSnapshot.load(map_path)
    commit, dirty = git_state()
    if args.formal:
        if snapshot.metadata.get("truth_derived"):
            raise SystemExit("FORMAL_REJECTED: truth-derived development map")
        if snapshot.metadata.get("formal_experiment_allowed") is not True:
            raise SystemExit("FORMAL_REJECTED: map is not audited for formal use")
        if snapshot.metadata.get("comparison_independent_survey") is not True:
            raise SystemExit("FORMAL_REJECTED: map survey used a compared method")
        if int(snapshot.metadata.get("map_audit_schema_version", 0)) < 5:
            raise SystemExit("FORMAL_REJECTED: map audit schema is older than v5")
        if not snapshot.metadata.get("survey_protocol_manifest_sha256"):
            raise SystemExit("FORMAL_REJECTED: missing frozen survey manifest hash")
        if dirty:
            raise SystemExit("FORMAL_REJECTED: tracked worktree changes: %s" % dirty)
    _, configured, limits, mass = load_planner_configuration(args.config)
    config = replace(
        configured, optimizer_time_budget=args.time_budget,
        repair_attempts=configured.repair_attempts,
    )
    start = np.asarray(task["start_state"], dtype=float)
    preferred = task["preferred_standoff"]
    shape = np.radians(np.asarray(task["terminal_shape_deg"], dtype=float))
    terminal = np.asarray(preferred[:4] + shape.tolist(), dtype=float)
    target = np.asarray(task["target"][:3], dtype=float)
    terminal_tolerance = float(task.get("tip_tolerance_m", 0.08))
    reference_points, reference_diagnostics = VoxelAStarReferencePlanner(
        snapshot,
        center_clearance=limits.body_radius + limits.body_margin,
        body_radius=limits.body_radius, body_height=limits.body_height,
        body_margin=limits.body_margin,
        z_min=limits.z_min, z_max=limits.z_max,
        time_budget=args.reference_time_budget,
    ).plan(start[:3], terminal[:3])
    reference = path_states(
        densify_path(reference_points, maximum_step=0.20),
        default_shape=start[4:],
    )
    planning_start = start.copy()
    planning_start[:3] = reference_points[0]
    reference[0] = planning_start
    reference[-1, :4] = terminal[:4]
    planner_world = snapshot.world(inflation=0.02, unknown_is_occupied=True)
    oracle_world = HybridDistanceWorld(BoxWorld.from_dicts(oracle_data.get("boxes", [])))
    methods = list(METHOD_NAMES) if args.method == "all" else [args.method]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output = args.output or (
        REPOSITORY / "xtdrone_plan" / "experiments" / "method_comparison"
        / "offline_runs" / (task.get("name", args.scenario_dir.name) + "_" + timestamp)
    )
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "experiment_class": "offline_algorithm_pilot_not_closed_loop_main_result",
        "formal_requested": args.formal,
        "scenario": task.get("name", args.scenario_dir.name),
        "scenario_dir": str(args.scenario_dir.resolve()),
        "methods": methods,
        "commit": commit,
        "tracked_dirty": dirty,
        "reproducibility_scope_dirty": dirty,
        "map_hash": snapshot.map_hash,
        "map_path": str(map_path.resolve()),
        "map_sha256": sha256(map_path),
        "map_source": snapshot.source,
        "map_metadata": snapshot.metadata,
        "config": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
        "baseline_registry": str(args.baseline_registry.resolve()),
        "baseline_registry_sha256": sha256(args.baseline_registry),
        "task_sha256": sha256(task_path),
        "oracle_sha256": sha256(oracle_path),
        "time_budget_s": args.time_budget,
        "reference_time_budget_s": args.reference_time_budget,
        "reference": reference_diagnostics,
        "nominal_to_planning_start_m": float(np.linalg.norm(planning_start[:3] - start[:3])),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False,
                   default=json_default) + "\n",
        encoding="utf-8",
    )
    reports = []
    for method_name in methods:
        problem = ComparisonProblem(
            reference_states=reference, current_state=planning_start,
            terminal_state=terminal, terminal_target=target,
            terminal_tolerance=terminal_tolerance,
            world=planner_world, arm=ArmModel(), mass=mass, limits=limits,
            config=config, method_time_budget=args.time_budget,
            random_seed=int(task.get("seed", 0)),
        )
        report, result = evaluate(method_name, problem, oracle_world)
        reports.append(report)
        (output / (method_name + ".json")).write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False,
                       default=json_default) + "\n",
            encoding="utf-8",
        )
        if result.trajectory is not None:
            np.savez_compressed(
                str(output / (method_name + "_trajectory.npz")),
                times=result.trajectory.times, states=result.trajectory.states,
                velocities=result.trajectory.velocities,
                accelerations=result.trajectory.accelerations,
                jerks=result.trajectory.jerks,
            )
    summary = {
        "scenario": manifest["scenario"],
        "map_hash": snapshot.map_hash,
        "reports": reports,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False,
                   default=json_default) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False,
                     default=json_default))
    print("RUN_DIR=%s" % output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
