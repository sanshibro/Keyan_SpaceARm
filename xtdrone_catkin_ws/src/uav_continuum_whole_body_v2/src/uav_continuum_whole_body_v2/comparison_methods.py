"""Auditable implementations/adaptations of the paper comparison methods."""

from dataclasses import dataclass, replace
import math
import time

import numpy as np

from .geometry import (
    coupling_metrics, coupling_metrics_batch, interpolate_state,
    world_arm_points, world_arm_points_batch, world_body_points,
    world_body_points_batch,
)
from .optimizer import WholeBodyOptimizer
from .seed_planner import SeedPlan, WholeBodySeedPlanner
from .trajectory import SplineTrajectory, initial_span_durations, resample_states
from .types import OptimizationResult, PHI, THETA, TrajectoryData, ValidationResult, YAW
from .validator import SweptVolumeValidator


METHOD_NAMES = (
    "fixed_morphology", "aet_sequential_minjerk", "informed_rrtstar_wb",
    "global_batch_wb", "rolling_wb_v2",
)


@dataclass
class ComparisonProblem:
    reference_states: np.ndarray
    current_state: np.ndarray
    world: object
    arm: object
    mass: object
    limits: object
    config: object
    terminal_state: object = None
    terminal_target: object = None
    terminal_tolerance: float = 0.05
    method_time_budget: float = 10.0
    random_seed: int = 0

    def __post_init__(self):
        self.reference_states = np.asarray(self.reference_states, dtype=float)
        self.current_state = np.asarray(self.current_state, dtype=float)
        if self.reference_states.ndim != 2 or self.reference_states.shape[1] != 6:
            raise ValueError("reference_states must have shape (N, 6)")
        if self.current_state.shape != (6,):
            raise ValueError("current_state must have shape (6,)")
        if self.terminal_state is not None:
            self.terminal_state = np.asarray(self.terminal_state, dtype=float)
        if self.terminal_target is not None:
            self.terminal_target = np.asarray(self.terminal_target, dtype=float)

    def validator(self):
        return SweptVolumeValidator(
            self.world, self.arm, self.mass, self.limits, self.config,
            terminal_target=self.terminal_target,
            terminal_tolerance=self.terminal_tolerance,
        )


def _failure(method, started, reason, diagnostics=None):
    validation = ValidationResult(
        False, "rejected", {}, failure_reason=str(reason)
    )
    return OptimizationResult(
        False, "rejected_" + str(reason), None, float("inf"), 0,
        time.monotonic() - started, validation=validation,
        diagnostics={"method": method, **(diagnostics or {})},
    )


def _timed_spline(states, problem, source, speed_scale=1.0):
    states = np.asarray(states, dtype=float)
    path_length = float(np.sum(np.linalg.norm(np.diff(states[:, :3], axis=0), axis=1)))
    design_speed = (
        problem.limits.velocity_max * problem.config.design_velocity_ratio
        * max(float(speed_scale), 1e-3)
    )
    dynamic_control_count = int(np.ceil(
        3.0 * path_length / max(design_speed * problem.config.dt_max, 1e-6)
    )) + 3
    control_count = max(problem.config.knot_count, dynamic_control_count)
    controls = resample_states(states, control_count)
    controls[:, YAW] = np.unwrap(controls[:, YAW])
    controls[:, PHI] = np.unwrap(controls[:, PHI])
    durations = initial_span_durations(
        controls, 3,
        problem.config.nominal_speed * max(float(speed_scale), 1e-3),
        problem.config.dt_min, problem.config.dt_max,
    )
    rate_dt = max(
        3.0 * float(np.max(np.abs(np.diff(controls[:, YAW]))))
        / max(problem.limits.yaw_rate_max, 1e-6),
        3.0 * float(np.max(np.abs(np.diff(controls[:, THETA]))))
        / max(problem.limits.theta_rate_max, 1e-6),
        3.0 * float(np.max(np.abs(np.diff(controls[:, PHI]))))
        / max(problem.limits.phi_rate_max, 1e-6),
    )
    durations[:] = np.clip(
        max(float(durations[0]), 1.08 * rate_dt),
        problem.config.dt_min, problem.config.dt_max,
    )
    return SplineTrajectory(controls, durations), controls, durations


def _validated_result(method, problem, spline, started, objective=0.0,
                      diagnostics=None):
    diagnostics = dict(diagnostics or {})
    stage_times = dict(diagnostics.get("stage_times_s", {}))
    validation_started = time.monotonic()
    validation = problem.validator().validate(spline)
    stage_times["swept_volume_validation"] = time.monotonic() - validation_started
    sampling_started = time.monotonic()
    trajectory = spline.sample(
        problem.config.output_dt, source=method, objective=objective,
        metadata={"method": method},
    )
    stage_times["trajectory_sampling"] = time.monotonic() - sampling_started
    diagnostics["stage_times_s"] = stage_times
    result = OptimizationResult(
        validation.feasible,
        "validated" if validation.feasible else "rejected_" + validation.failure_reason,
        trajectory, float(objective), 0, time.monotonic() - started,
        validation=validation,
        diagnostics={"method": method, **diagnostics},
    )
    if result.elapsed > problem.method_time_budget:
        return _failure(method, started, "plan_timeout", {
            **diagnostics,
            "wall_time_s": result.elapsed,
            "method_time_budget_s": problem.method_time_budget,
            "candidate_was_feasible": bool(validation.feasible),
        })
    return result


class PiecewiseMinimumJerkTrajectory:
    """C2 quintic waypoint trajectory with analytic derivatives."""

    def __init__(self, states, problem):
        self.states = np.asarray(states, dtype=float).copy()
        self.states[:, YAW] = np.unwrap(self.states[:, YAW])
        self.states[:, PHI] = np.unwrap(self.states[:, PHI])
        differences = np.diff(self.states, axis=0)
        translation = np.linalg.norm(differences[:, :3], axis=1)
        peak_progress_rate = 1.875
        durations = np.maximum(
            translation * peak_progress_rate / max(
                problem.limits.velocity_max * problem.config.design_velocity_ratio, 1e-6
            ),
            problem.config.dt_min,
        )
        durations = np.maximum(
            durations,
            peak_progress_rate * np.abs(differences[:, YAW])
            / max(problem.limits.yaw_rate_max, 1e-6),
        )
        durations = np.maximum(
            durations,
            peak_progress_rate * np.abs(differences[:, THETA])
            / max(problem.limits.theta_rate_max, 1e-6),
        )
        durations = np.maximum(
            durations,
            peak_progress_rate * np.abs(differences[:, PHI])
            / max(problem.limits.phi_rate_max, 1e-6),
        )
        peak_progress_acceleration = 5.773502691896258
        durations = np.maximum(
            durations,
            np.sqrt(
                peak_progress_acceleration * translation
                / max(problem.limits.acceleration_max, 1e-6)
            ),
        )
        durations = np.maximum(
            durations,
            np.cbrt(
                60.0 * translation / max(problem.limits.jerk_max, 1e-6)
            ),
        )
        durations = np.maximum(
            durations,
            np.sqrt(
                peak_progress_acceleration * np.abs(differences[:, YAW])
                / max(problem.limits.yaw_acceleration_max, 1e-6)
            ),
        )
        durations = np.maximum(
            durations,
            np.sqrt(
                peak_progress_acceleration * np.abs(differences[:, THETA])
                / max(problem.limits.theta_acceleration_max, 1e-6)
            ),
        )
        durations = np.maximum(
            durations,
            np.sqrt(
                peak_progress_acceleration * np.abs(differences[:, PHI])
                / max(problem.limits.phi_acceleration_max, 1e-6)
            ),
        )
        self.durations = 1.02 * np.maximum(durations, 1e-3)
        self.span_boundaries = np.r_[0.0, np.cumsum(self.durations)]
        self.duration = float(self.span_boundaries[-1])

    def evaluate(self, times, derivative=0):
        times = np.clip(np.asarray(times, dtype=float), 0.0, self.duration)
        indices = np.searchsorted(self.span_boundaries, times, side="right") - 1
        indices = np.clip(indices, 0, len(self.durations) - 1)
        local = times - self.span_boundaries[indices]
        duration = self.durations[indices]
        s = np.clip(local / duration, 0.0, 1.0)
        if derivative == 0:
            blend = 10.0 * s ** 3 - 15.0 * s ** 4 + 6.0 * s ** 5
            return self.states[indices] + blend[:, None] * (
                self.states[indices + 1] - self.states[indices]
            )
        if derivative == 1:
            blend = (30.0 * s ** 2 - 60.0 * s ** 3 + 30.0 * s ** 4) / duration
        elif derivative == 2:
            blend = (60.0 * s - 180.0 * s ** 2 + 120.0 * s ** 3) / duration ** 2
        elif derivative == 3:
            blend = (60.0 - 360.0 * s + 360.0 * s ** 2) / duration ** 3
        else:
            return np.zeros((len(times), self.states.shape[1]), dtype=float)
        return blend[:, None] * (self.states[indices + 1] - self.states[indices])

    def sample(self, output_dt, source="minimum_jerk", objective=0.0, metadata=None):
        count = max(2, int(np.ceil(self.duration / float(output_dt))) + 1)
        times = np.linspace(0.0, self.duration, count)
        return TrajectoryData(
            times=times, states=self.evaluate(times, 0),
            velocities=self.evaluate(times, 1),
            accelerations=self.evaluate(times, 2), jerks=self.evaluate(times, 3),
            source=source, objective=float(objective), metadata=dict(metadata or {}),
        )


class FixedMorphologyMethod:
    name = "fixed_morphology"

    def __init__(self, fixed_shape=None):
        self.fixed_shape = fixed_shape

    def plan(self, problem):
        started = time.monotonic()
        states = resample_states(problem.reference_states, problem.config.knot_count)
        if self.fixed_shape is not None:
            shape = np.asarray(self.fixed_shape, dtype=float)
        elif problem.terminal_state is not None:
            # A single morphology may be selected on the development set and
            # is established before timing/execution starts.
            shape = problem.terminal_state[[THETA, PHI]]
        else:
            shape = problem.current_state[[THETA, PHI]]
        states[:, THETA] = shape[0]
        states[:, PHI] = shape[1]
        states[0, :4] = problem.current_state[:4]
        if problem.terminal_state is not None:
            states[-1, :4] = problem.terminal_state[:4]
        spline, _controls, _durations = _timed_spline(states, problem, self.name)
        return _validated_result(
            self.name, problem, spline, started,
            diagnostics={
                "adaptation": "common_slam_reference_with_shape_fixed_before_execution",
                "fixed_shape_deg": np.rad2deg(shape).tolist(),
                "rolling": False,
            },
        )


class AetSequentialMinJerkMethod:
    """AET-style waypoint minimum-jerk trajectory followed by whole-body IK.

    AET plans a quintic minimum-jerk end-effector trajectory and decomposes it
    into UAV/configuration commands by gradient descent. For the one-section
    continuum model, the decomposition is represented by a smooth transition
    from the current configuration to the task configuration while retaining
    the common map-derived waypoint path. No V2 risk segmentation, rolling
    buffer, collision repair, or time-budget recovery is used.
    """

    name = "aet_sequential_minjerk"

    @staticmethod
    def _minimum_jerk_progress(count):
        parameter = np.linspace(0.0, 1.0, count)
        return 10.0 * parameter ** 3 - 15.0 * parameter ** 4 + 6.0 * parameter ** 5

    def plan(self, problem):
        started = time.monotonic()
        count = problem.config.knot_count
        # Spatial waypoints remain uniformly distributed so narrow passages
        # are not skipped.  Each segment then uses exact quintic minimum-jerk
        # time scaling with analytic derivatives.
        states = resample_states(problem.reference_states, count)
        states[0] = problem.current_state
        if problem.terminal_state is not None:
            states[-1] = problem.terminal_state
        # AET's gradient decomposition obtains platform/configuration commands
        # for the prescribed end-effector waypoint trajectory.  On this
        # platform we solve the analogous fixed-UAV-reference configuration
        # corridor, but never move UAV waypoints or invoke V2 rolling/repair.
        validation_reserve = min(
            4.0, max(1.5, 0.25 * problem.method_time_budget)
        )
        corridor_started = time.monotonic()
        corridor, corridor_diagnostics = GlobalBatchWholeBodyMethod._configuration_corridor(
            problem, states,
            deadline=started + max(0.0, problem.method_time_budget - validation_reserve),
        )
        corridor_elapsed = time.monotonic() - corridor_started
        if corridor is None:
            return _failure(
                self.name, started,
                corridor_diagnostics.get("failure", "configuration_decomposition_failed"),
                {
                    "stage": "fixed_reference_configuration_decomposition",
                    "configuration_corridor": corridor_diagnostics,
                    "stage_times_s": {"configuration_corridor": corridor_elapsed},
                    "uav_reference_fixed": True,
                },
            )
        states = corridor
        spline = PiecewiseMinimumJerkTrajectory(states, problem)
        jerk_times = np.linspace(0.0, spline.duration, 300)
        jerk = spline.evaluate(jerk_times, 3)
        objective = float(np.trapz(np.sum(jerk ** 2, axis=1), jerk_times))
        return _validated_result(
            self.name, problem, spline, started, objective=objective,
            diagnostics={
                "adaptation": "AET_quintic_minimum_jerk_plus_continuum_configuration_decomposition",
                "reference_feature": "common_slam_waypoints",
                "joint_uav_collision_optimization": False,
                "collision_constrained_configuration_decomposition": True,
                "uav_reference_fixed": True,
                "configuration_corridor": corridor_diagnostics,
                "validation_reserve_s": validation_reserve,
                "stage_times_s": {"configuration_corridor": corridor_elapsed},
                "rolling": False,
            },
        )


class InformedRrtStarWholeBodyMethod:
    """Anytime RRT* adaptation in the six-dimensional whole-body state.

    This is deliberately named as an adaptation, not as the exact S-RGM*
    implementation from the cited paper.  It uses the common SLAM reference as
    a geometry-guided sampling distribution and keeps improving an incumbent
    after the first feasible solution.
    """

    name = "informed_rrtstar_wb"

    def __init__(self, maximum_samples=6000, goal_bias=0.18,
                 reference_bias=0.62, improvement_samples=64):
        self.maximum_samples = int(maximum_samples)
        self.goal_bias = float(goal_bias)
        self.reference_bias = float(reference_bias)
        self.improvement_samples = int(improvement_samples)

    @staticmethod
    def _metric(first, second):
        delta = np.asarray(first) - np.asarray(second)
        return float(np.linalg.norm(np.asarray([
            delta[0], delta[1], delta[2], 0.25 * delta[YAW],
            0.30 * delta[THETA], 0.15 * delta[PHI],
        ])))

    @staticmethod
    def _steer(first, target, distance=0.42):
        metric = InformedRrtStarWholeBodyMethod._metric(first, target)
        if metric <= distance:
            return np.asarray(target, dtype=float).copy()
        return interpolate_state(first, target, distance / max(metric, 1e-9))

    @staticmethod
    def _state_feasible(problem, state):
        body = problem.world.minimum_distance(
            world_body_points(state, problem.limits)
        )[0]
        arm = problem.world.minimum_distance(
            world_arm_points(state, problem.arm)
        )[0] - problem.arm.radius
        coupling = coupling_metrics(state, problem.arm, problem.mass)
        return (
            body >= problem.limits.body_margin
            and arm >= problem.limits.arm_margin
            and coupling["com_offset"] <= problem.limits.com_max
            and problem.limits.z_min <= state[2] <= problem.limits.z_max
            and problem.arm.theta_min <= state[THETA] <= problem.arm.theta_max
        )

    @classmethod
    def _edge_feasible(cls, problem, first, second):
        translation = np.linalg.norm(np.asarray(second[:3]) - np.asarray(first[:3]))
        shape = max(
            abs(second[THETA] - first[THETA]) / np.deg2rad(8.0),
            abs(math.atan2(math.sin(second[PHI] - first[PHI]),
                           math.cos(second[PHI] - first[PHI]))) / np.deg2rad(12.0),
        )
        count = max(2, int(np.ceil(max(translation / 0.06, shape))) + 1)
        return all(cls._state_feasible(problem, interpolate_state(first, second, ratio))
                   for ratio in np.linspace(0.0, 1.0, count))

    @classmethod
    def _edges_feasible(cls, problem, first_states, second_states):
        """Vectorized equivalent of _edge_feasible for corridor layers."""
        first_states = np.asarray(first_states, dtype=float)
        second_states = np.asarray(second_states, dtype=float)
        if len(first_states) == 0:
            return np.empty(0, dtype=bool)
        translation = np.linalg.norm(second_states[:, :3] - first_states[:, :3], axis=1)
        phi_delta = np.arctan2(
            np.sin(second_states[:, PHI] - first_states[:, PHI]),
            np.cos(second_states[:, PHI] - first_states[:, PHI]),
        )
        shape = np.maximum(
            np.abs(second_states[:, THETA] - first_states[:, THETA]) / np.deg2rad(8.0),
            np.abs(phi_delta) / np.deg2rad(12.0),
        )
        sample_count = max(2, int(np.ceil(np.max(np.maximum(
            translation / 0.06, shape
        )))) + 1)
        ratios = np.linspace(0.0, 1.0, sample_count)
        states = (
            first_states[:, None, :] * (1.0 - ratios[None, :, None])
            + second_states[:, None, :] * ratios[None, :, None]
        )
        for coordinate in (YAW, PHI):
            delta = np.arctan2(
                np.sin(second_states[:, coordinate] - first_states[:, coordinate]),
                np.cos(second_states[:, coordinate] - first_states[:, coordinate]),
            )
            states[:, :, coordinate] = (
                first_states[:, None, coordinate]
                + ratios[None, :] * delta[:, None]
            )
        flat = states.reshape((-1, 6))
        body_groups = world_body_points_batch(flat, problem.limits)
        arm_groups = world_arm_points_batch(flat, problem.arm)
        body, _ = problem.world.minimum_distances_grouped(body_groups)
        arm, _ = problem.world.minimum_distances_grouped(arm_groups)
        arm = arm - problem.arm.radius
        com = coupling_metrics_batch(flat, problem.arm, problem.mass)
        feasible = (
            (body >= problem.limits.body_margin)
            & (arm >= problem.limits.arm_margin)
            & (com <= problem.limits.com_max)
            & (flat[:, 2] >= problem.limits.z_min)
            & (flat[:, 2] <= problem.limits.z_max)
            & (flat[:, THETA] >= problem.arm.theta_min)
            & (flat[:, THETA] <= problem.arm.theta_max)
        )
        return np.all(feasible.reshape((len(first_states), sample_count)), axis=1)

    def _sample(self, problem, rng, goal, incumbent_cost=None):
        draw = rng.rand()
        if incumbent_cost is None and draw < self.goal_bias:
            return goal.copy()
        if draw < self.goal_bias + self.reference_bias:
            index = rng.randint(len(problem.reference_states))
            sample = problem.reference_states[index].copy()
            sample[:3] += rng.normal(0.0, [0.28, 0.28, 0.16])
            sample[YAW] += rng.normal(0.0, np.deg2rad(20.0))
            # Discrete morphology modes plus small noise are substantially
            # more useful in tight passages than drawing both shape variables
            # uniformly from their complete ranges.
            theta_modes = np.asarray([
                problem.arm.theta_min, np.deg2rad(35.0), np.deg2rad(70.0),
                np.deg2rad(100.0), problem.arm.theta_max,
            ])
            phi_modes = np.asarray([-np.pi, -0.5 * np.pi, 0.0, 0.5 * np.pi])
            sample[THETA] = np.clip(
                theta_modes[rng.randint(len(theta_modes))]
                + rng.normal(0.0, np.deg2rad(6.0)),
                problem.arm.theta_min, problem.arm.theta_max,
            )
            sample[PHI] = phi_modes[rng.randint(len(phi_modes))] + rng.normal(
                0.0, np.deg2rad(8.0)
            )
            return sample
        observed = problem.world.snapshot.observed_points
        sample = np.zeros(6)
        sample[:3] = observed[rng.randint(len(observed))]
        sample[YAW] = rng.uniform(-np.pi, np.pi)
        sample[THETA] = rng.uniform(problem.arm.theta_min, problem.arm.theta_max)
        sample[PHI] = rng.uniform(-np.pi, np.pi)
        return sample

    @classmethod
    def _reference_guides(cls, problem, start, goal):
        """Return the best feasible deterministic guide for RRT* warm start.

        The guide uses only the common SLAM reference and a fixed public shape
        bank.  It is independent of V2 risk segmentation and provides the
        geometry-biased initialization used by the sampling baseline.
        """
        reference = resample_states(
            problem.reference_states, max(problem.config.knot_count, 12)
        )
        policies = []
        terminal_shape = goal[[THETA, PHI]]
        current_shape = start[[THETA, PHI]]
        policies.append(("terminal_fixed", np.repeat(
            terminal_shape[None, :], len(reference), axis=0
        )))
        policies.append(("current_fixed", np.repeat(
            current_shape[None, :], len(reference), axis=0
        )))
        blend = np.linspace(0.0, 1.0, len(reference))[:, None]
        phi_delta = math.atan2(
            math.sin(terminal_shape[1] - current_shape[1]),
            math.cos(terminal_shape[1] - current_shape[1]),
        )
        linear = np.empty((len(reference), 2), dtype=float)
        linear[:, 0] = current_shape[0] + blend[:, 0] * (
            terminal_shape[0] - current_shape[0]
        )
        linear[:, 1] = current_shape[1] + blend[:, 0] * phi_delta
        policies.append(("linear_terminal", linear))
        for theta in np.radians([35.0, 70.0, 100.0, 118.0]):
            for phi in (0.0, -np.pi):
                policies.append((
                    "shape_bank_%d_%d" % (
                        int(round(np.degrees(theta))),
                        int(round(np.degrees(phi))),
                    ),
                    np.repeat([[theta, phi]], len(reference), axis=0),
                ))
        candidates = []
        all_first, all_second = [], []
        for label, shapes in policies:
            states = reference.copy()
            states[:, [THETA, PHI]] = shapes
            states[0] = start
            states[-1] = goal
            candidates.append((label, states))
            all_first.extend(states[:-1])
            all_second.extend(states[1:])
        all_feasible = cls._edges_feasible(problem, all_first, all_second)
        feasible_guides = []
        edge_count = len(reference) - 1
        for index, (label, states) in enumerate(candidates):
            feasible = all_feasible[index * edge_count:(index + 1) * edge_count]
            if np.all(feasible):
                cost = sum(cls._metric(a, b) for a, b in zip(states[:-1], states[1:]))
                feasible_guides.append((cost, label, states))
        if not feasible_guides:
            return None, {"candidate_count": len(policies), "feasible_count": 0}
        feasible_guides.sort(key=lambda item: item[0])
        cost, label, states = feasible_guides[0]
        return states, {
            "candidate_count": len(policies),
            "feasible_count": len(feasible_guides),
            "selected": label,
            "cost": cost,
        }

    def _shortcut(self, problem, states, deadline=None):
        states = list(states)
        changed = True
        while changed and len(states) > 2:
            changed = False
            for first in range(len(states) - 2):
                for second in range(len(states) - 1, first + 1, -1):
                    if deadline is not None and time.monotonic() >= deadline:
                        return np.asarray(states)
                    if self._edge_feasible(problem, states[first], states[second]):
                        states = states[:first + 1] + states[second:]
                        changed = True
                        break
                if changed:
                    break
        return np.asarray(states)

    def plan(self, problem):
        started = time.monotonic()
        if not hasattr(problem.world, "snapshot"):
            return _failure(self.name, started, "requires_voxel_slam_world")
        goal = (
            problem.terminal_state.copy() if problem.terminal_state is not None
            else problem.reference_states[-1].copy()
        )
        start = problem.current_state.copy()
        if not self._state_feasible(problem, start):
            return _failure(self.name, started, "invalid_start")
        if not self._state_feasible(problem, goal):
            return _failure(self.name, started, "invalid_goal")
        rng = np.random.RandomState(problem.random_seed)
        nodes = [start]
        parents = [-1]
        costs = [0.0]
        children = [set()]
        goal_indices = []
        samples = 0
        first_solution_sample = -1
        incumbent_updates = 0
        guide_started = time.monotonic()
        guide, guide_diagnostics = self._reference_guides(problem, start, goal)
        guide_elapsed = time.monotonic() - guide_started
        if guide is not None:
            for state in guide[1:]:
                parent = len(nodes) - 1
                nodes.append(state.copy())
                parents.append(parent)
                costs.append(costs[parent] + self._metric(nodes[parent], state))
                children.append(set())
                children[parent].add(len(nodes) - 1)
            goal_indices.append(len(nodes) - 1)
            first_solution_sample = 0
        elif self._edge_feasible(problem, start, goal):
            nodes.append(goal.copy())
            parents.append(0)
            costs.append(self._metric(start, goal))
            children.append(set())
            children[0].add(1)
            goal_indices.append(1)
            first_solution_sample = 0
        validation_reserve = min(
            5.0, max(2.0, 0.33 * problem.method_time_budget)
        )
        search_deadline = started + max(
            0.0, problem.method_time_budget - validation_reserve
        )
        search_started = time.monotonic()
        while samples < self.maximum_samples:
            # Reserve time for shortcutting, spline construction and the same
            # independent validator used by every method.
            if time.monotonic() >= search_deadline:
                break
            if (first_solution_sample >= 0
                    and samples - first_solution_sample >= self.improvement_samples):
                break
            incumbent_cost = (
                min(costs[index] for index in goal_indices)
                if goal_indices else None
            )
            sample = self._sample(problem, rng, goal, incumbent_cost)
            if incumbent_cost is not None and (
                    self._metric(start, sample) + self._metric(sample, goal)
                    >= 1.15 * incumbent_cost):
                samples += 1
                continue
            expandable = [
                index for index in range(len(nodes)) if index not in goal_indices
            ]
            nearest = min(expandable, key=lambda index: self._metric(nodes[index], sample))
            candidate = self._steer(nodes[nearest], sample)
            samples += 1
            if not self._edge_feasible(problem, nodes[nearest], candidate):
                continue
            radius = max(0.48, 1.1 * (math.log(len(nodes) + 1) / (len(nodes) + 1)) ** (1.0 / 6.0))
            neighbours = [
                index for index, item in enumerate(nodes)
                if index not in goal_indices
                if self._metric(item, candidate) <= radius
                and self._edge_feasible(problem, item, candidate)
            ]
            parent = nearest
            candidate_cost = costs[nearest] + self._metric(nodes[nearest], candidate)
            for index in neighbours:
                alternative = costs[index] + self._metric(nodes[index], candidate)
                if alternative < candidate_cost:
                    parent, candidate_cost = index, alternative
            nodes.append(candidate)
            parents.append(parent)
            costs.append(candidate_cost)
            children.append(set())
            new_index = len(nodes) - 1
            children[parent].add(new_index)
            for index in neighbours:
                alternative = candidate_cost + self._metric(candidate, nodes[index])
                if alternative < costs[index] and self._edge_feasible(problem, candidate, nodes[index]):
                    old_parent = parents[index]
                    if old_parent >= 0:
                        children[old_parent].discard(index)
                    parents[index] = new_index
                    children[new_index].add(index)
                    delta = alternative - costs[index]
                    costs[index] = alternative
                    descendants = list(children[index])
                    while descendants:
                        descendant = descendants.pop()
                        costs[descendant] += delta
                        descendants.extend(children[descendant])
            if self._metric(candidate, goal) <= 0.42 and self._edge_feasible(problem, candidate, goal):
                nodes.append(goal.copy())
                parents.append(new_index)
                costs.append(candidate_cost + self._metric(candidate, goal))
                children.append(set())
                children[new_index].add(len(nodes) - 1)
                goal_indices.append(len(nodes) - 1)
                if first_solution_sample < 0:
                    first_solution_sample = samples
                else:
                    incumbent_updates += 1
        search_elapsed = time.monotonic() - search_started
        if not goal_indices:
            return _failure(
                self.name, started, "no_path",
                {
                    "samples": samples, "tree_nodes": len(nodes),
                    "random_seed": problem.random_seed,
                    "reference_guide": guide_diagnostics,
                    "stage_times_s": {
                        "reference_guide": guide_elapsed,
                        "tree_improvement": search_elapsed,
                    },
                },
            )
        goal_index = min(goal_indices, key=lambda index: costs[index])
        path = []
        current = goal_index
        while current >= 0:
            path.append(nodes[current])
            current = parents[current]
        path.reverse()
        shortcut_started = time.monotonic()
        path = self._shortcut(problem, path, deadline=search_deadline)
        spline, _controls, _durations = _timed_spline(path, problem, self.name, speed_scale=0.8)
        shortcut_elapsed = time.monotonic() - shortcut_started
        return _validated_result(
            self.name, problem, spline, started, objective=costs[goal_index],
            diagnostics={
                "adaptation": "anytime_informed_RRTstar_in_UAV_yaw_continuum_configuration_space",
                "samples": samples, "tree_nodes": len(nodes),
                "path_states": len(path), "rolling": False,
                "random_seed": problem.random_seed,
                "first_solution_sample": first_solution_sample,
                "improvement_samples_run": max(0, samples - first_solution_sample),
                "incumbent_updates": incumbent_updates,
                "validation_reserve_s": validation_reserve,
                "reference_guide": guide_diagnostics,
                "stage_times_s": {
                    "reference_guide": guide_elapsed,
                    "tree_improvement": search_elapsed,
                    "shortcut_and_spline": shortcut_elapsed,
                },
            },
        )


class GlobalBatchWholeBodyMethod:
    """Non-rolling full-horizon whole-body trajectory optimization baseline."""

    name = "global_batch_wb"

    @staticmethod
    def _configuration_corridor(problem, reference, deadline=None):
        """Full-path variable-geometry initialization, not V2 risk seeding."""
        corridor_started = time.monotonic()
        theta_values = np.unique(np.radians(np.r_[
            np.linspace(0.0, 118.0, 9),
            np.degrees(problem.current_state[THETA]),
            np.degrees(problem.terminal_state[THETA])
            if problem.terminal_state is not None else 0.0,
        ]))
        phi_values = np.unique(np.radians(np.r_[
            -180.0, -90.0, 0.0, 90.0, 180.0,
            np.degrees(problem.current_state[PHI]),
            np.degrees(problem.terminal_state[PHI])
            if problem.terminal_state is not None else 0.0,
        ]))
        layers = []
        clearances = []
        for index, pose in enumerate(reference):
            if deadline is not None and time.monotonic() >= deadline:
                return None, {
                    "failure": "plan_timeout", "stage": "corridor_layers",
                    "completed_layers": len(layers),
                    "elapsed_s": time.monotonic() - corridor_started,
                }
            if index == 0:
                candidates = [problem.current_state.copy()]
            elif index == len(reference) - 1 and problem.terminal_state is not None:
                candidates = [problem.terminal_state.copy()]
            else:
                candidate_states = []
                for theta in theta_values:
                    for phi in phi_values:
                        state = pose.copy()
                        state[THETA], state[PHI] = theta, phi
                        candidate_states.append(state)
                arm_groups = np.asarray([
                    world_arm_points(state, problem.arm)
                    for state in candidate_states
                ])
                arm_clearances, _ = problem.world.minimum_distances_grouped(arm_groups)
                arm_clearances = arm_clearances - problem.arm.radius
                ranked = []
                for state, arm_clearance in zip(candidate_states, arm_clearances):
                    coupling = coupling_metrics(state, problem.arm, problem.mass)
                    theta, phi = state[THETA], state[PHI]
                    if (arm_clearance < problem.limits.arm_margin
                            or coupling["com_offset"] > problem.limits.com_max):
                        continue
                    local_cost = (
                        0.015 * theta ** 2
                        + 0.004 / max(arm_clearance, 1e-3)
                    )
                    ranked.append((local_cost, -arm_clearance, state, arm_clearance))
                ranked.sort(key=lambda item: (item[0], item[1]))
                # Preserve morphology diversity instead of allowing the local
                # low-theta cost to prune the only configuration sequence that
                # can connect a later narrow passage.
                selected = []
                seen_theta, seen_phi, seen_states = set(), set(), set()
                priority_shapes = [
                    problem.current_state[[THETA, PHI]],
                    problem.terminal_state[[THETA, PHI]]
                    if problem.terminal_state is not None else None,
                ]
                for desired in priority_shapes:
                    if desired is None:
                        continue
                    for item in ranked:
                        state = item[2]
                        phi_error = math.atan2(
                            math.sin(state[PHI] - desired[1]),
                            math.cos(state[PHI] - desired[1]),
                        )
                        if (abs(state[THETA] - desired[0]) < 1e-7
                                and abs(phi_error) < 1e-7):
                            key = (
                                round(float(state[THETA]), 8),
                                round(float(math.atan2(
                                    math.sin(state[PHI]), math.cos(state[PHI])
                                )), 8),
                            )
                            if key not in seen_states:
                                selected.append(item)
                                seen_states.add(key)
                                seen_theta.add(key[0])
                                seen_phi.add(key[1])
                            break
                for item in ranked:
                    state = item[2]
                    theta_key = round(float(state[THETA]), 8)
                    phi_key = round(float(math.atan2(
                        math.sin(state[PHI]), math.cos(state[PHI])
                    )), 8)
                    state_key = (theta_key, phi_key)
                    if theta_key not in seen_theta or phi_key not in seen_phi:
                        if state_key not in seen_states:
                            selected.append(item)
                            seen_states.add(state_key)
                        seen_theta.add(theta_key)
                        seen_phi.add(phi_key)
                    if len(selected) >= 14:
                        break
                for item in ranked:
                    state_key = (
                        round(float(item[2][THETA]), 8),
                        round(float(math.atan2(
                            math.sin(item[2][PHI]), math.cos(item[2][PHI])
                        )), 8),
                    )
                    if state_key not in seen_states:
                        selected.append(item)
                        seen_states.add(state_key)
                    if len(selected) >= 14:
                        break
                candidates = [item[2] for item in selected[:14]]
                layer_clearances = [item[3] for item in selected[:14]]
                if not candidates:
                    return None, {"failure": "empty_configuration_corridor", "layer": index}
            layers.append(candidates)
            if index in (0, len(reference) - 1):
                clearances.append([float("inf")] * len(candidates))
            else:
                clearances.append(layer_clearances)

        costs = [np.zeros(len(layers[0]), dtype=float)]
        parents = []
        tested_edges = 0
        for layer_index in range(1, len(layers)):
            if deadline is not None and time.monotonic() >= deadline:
                return None, {
                    "failure": "plan_timeout", "stage": "corridor_edges",
                    "completed_transitions": layer_index - 1,
                    "tested_edges": tested_edges,
                    "elapsed_s": time.monotonic() - corridor_started,
                }
            current_costs = np.full(len(layers[layer_index]), np.inf)
            current_parents = np.full(len(layers[layer_index]), -1, dtype=int)
            pairs = [
                (previous_index, current_index, previous, current)
                for current_index, current in enumerate(layers[layer_index])
                for previous_index, previous in enumerate(layers[layer_index - 1])
            ]
            feasible_edges = InformedRrtStarWholeBodyMethod._edges_feasible(
                problem,
                [item[2] for item in pairs],
                [item[3] for item in pairs],
            )
            tested_edges += len(pairs)
            if deadline is not None and time.monotonic() >= deadline:
                return None, {
                    "failure": "plan_timeout", "stage": "corridor_edges",
                    "completed_transitions": layer_index - 1,
                    "tested_edges": tested_edges,
                    "elapsed_s": time.monotonic() - corridor_started,
                }
            for (previous_index, current_index, previous, current), feasible in zip(
                    pairs, feasible_edges):
                if not feasible:
                    continue
                theta_delta = current[THETA] - previous[THETA]
                phi_delta = math.atan2(
                    math.sin(current[PHI] - previous[PHI]),
                    math.cos(current[PHI] - previous[PHI]),
                )
                transition = 0.8 * theta_delta ** 2 + 0.25 * phi_delta ** 2
                candidate_cost = costs[-1][previous_index] + transition
                if candidate_cost < current_costs[current_index]:
                    current_costs[current_index] = candidate_cost
                    current_parents[current_index] = previous_index
            if not np.any(np.isfinite(current_costs)):
                return None, {
                    "failure": "disconnected_configuration_corridor",
                    "layer": layer_index, "tested_edges": tested_edges,
                }
            costs.append(current_costs)
            parents.append(current_parents)
        selected = [int(np.argmin(costs[-1]))]
        for parent_layer in reversed(parents):
            selected.append(int(parent_layer[selected[-1]]))
        selected.reverse()
        states = np.asarray([
            layers[index][selected[index]] for index in range(len(layers))
        ])
        return states, {
            "layers": len(layers), "tested_edges": tested_edges,
            "maximum_layer_width": max(len(layer) for layer in layers),
            "failure": "none",
            "elapsed_s": time.monotonic() - corridor_started,
        }

    def plan(self, problem):
        started = time.monotonic()
        reference = resample_states(problem.reference_states, problem.config.knot_count)
        reference[0] = problem.current_state
        if problem.terminal_state is not None:
            goal_shape = problem.terminal_state[[THETA, PHI]]
            blend = np.linspace(0.0, 1.0, len(reference)) ** 2
            reference[:, THETA] = (
                problem.current_state[THETA]
                + blend * (goal_shape[0] - problem.current_state[THETA])
            )
            phi_delta = math.atan2(
                math.sin(goal_shape[1] - problem.current_state[PHI]),
                math.cos(goal_shape[1] - problem.current_state[PHI]),
            )
            reference[:, PHI] = problem.current_state[PHI] + blend * phi_delta
            reference[-1] = problem.terminal_state
        corridor_reserve = min(
            4.0, max(2.0, 0.27 * problem.method_time_budget)
        )
        corridor_started = time.monotonic()
        corridor, corridor_diagnostics = self._configuration_corridor(
            problem, reference,
            deadline=started + max(0.0, problem.method_time_budget - corridor_reserve),
        )
        corridor_elapsed = time.monotonic() - corridor_started
        if corridor is None:
            return _failure(
                self.name, started,
                corridor_diagnostics.get("failure", "configuration_corridor_failed"),
                {
                    "stage": "configuration_corridor",
                    "configuration_corridor": corridor_diagnostics,
                    "stage_times_s": {"configuration_corridor": corridor_elapsed},
                },
            )
        reference = corridor
        remaining_budget = problem.method_time_budget - (time.monotonic() - started)
        if remaining_budget <= 0.0:
            return _failure(self.name, started, "plan_timeout", {
                "stage": "configuration_corridor",
                "configuration_corridor": corridor_diagnostics,
                "stage_times_s": {"configuration_corridor": corridor_elapsed},
                "method_time_budget_s": problem.method_time_budget,
            })
        validation_reserve = min(
            2.0, max(0.5, 0.20 * problem.method_time_budget)
        )
        optimizer_budget = remaining_budget - validation_reserve
        if optimizer_budget <= 0.0:
            return _failure(self.name, started, "plan_timeout", {
                "stage": "configuration_corridor",
                "configuration_corridor": corridor_diagnostics,
                "stage_times_s": {"configuration_corridor": corridor_elapsed},
                "validation_reserve_s": validation_reserve,
                "method_time_budget_s": problem.method_time_budget,
            })
        batch_config = replace(
            problem.config,
            knot_count=max(
                problem.config.knot_count,
                int(np.ceil(
                    3.0 * np.sum(np.linalg.norm(
                        np.diff(reference[:, :3], axis=0), axis=1
                    )) / max(
                        problem.limits.velocity_max
                        * problem.config.design_velocity_ratio
                        * problem.config.dt_max, 1e-6,
                    )
                )) + 3,
            ),
            optimizer_time_budget=optimizer_budget,
            repair_attempts=0,
            adaptive_phi=False,
            # AM-Planner is an integrated batch optimizer; it does not use
            # the V2 risk-segment beam initializer.
            risk_margin=max(problem.config.risk_margin, 0.08),
        )
        # Preserve the exact piecewise corridor as the safety incumbent.
        # Resampling is useful for the batch optimizer, but chords introduced
        # across several corridor layers need not retain the edge certificate.
        # Only the optimizer seed is resampled; the fallback remains the path
        # whose every original transition passed the swept-volume edge test.
        corridor_reference = reference.copy()
        optimizer_reference = resample_states(reference, batch_config.knot_count)
        optimizer_reference[0] = problem.current_state
        if problem.terminal_state is not None:
            corridor_reference[-1] = problem.terminal_state
            optimizer_reference[-1] = problem.terminal_state

        # Keep a fully validated feasible incumbent before spending the
        # remaining budget on batch refinement.  This makes the comparator
        # anytime: optimization may improve the trajectory, but a deadline no
        # longer discards a safe full-horizon solution already found by the
        # configuration corridor.
        batch_problem = replace(problem, config=batch_config)
        incumbent_spline = PiecewiseMinimumJerkTrajectory(
            corridor_reference, batch_problem
        )
        incumbent_validation_started = time.monotonic()
        incumbent_validation = SweptVolumeValidator(
            problem.world, problem.arm, problem.mass, problem.limits,
            batch_config, terminal_target=problem.terminal_target,
            terminal_tolerance=problem.terminal_tolerance,
        ).validate(incumbent_spline)
        incumbent_validation_elapsed = time.monotonic() - incumbent_validation_started
        stage_times = {
            "configuration_corridor": corridor_elapsed,
            "incumbent_validation": incumbent_validation_elapsed,
        }
        incumbent = None
        if incumbent_validation.feasible:
            incumbent_trajectory = incumbent_spline.sample(
                problem.config.output_dt, source=self.name, objective=0.0,
                metadata={"method": self.name, "solution": "corridor_incumbent"},
            )
            incumbent = OptimizationResult(
                True, "validated_incumbent", incumbent_trajectory, 0.0, 0,
                time.monotonic() - started, validation=incumbent_validation,
                diagnostics={},
            )
        refinement_remaining = problem.method_time_budget - (time.monotonic() - started)
        # A complete full-horizon batch candidate evaluation can take several
        # seconds under concurrent ROS/Gazebo load.  Starting refinement with
        # a smaller window is neither useful nor deadline-safe; the validated
        # global corridor is the correct anytime answer for the realtime arm.
        minimum_refinement_budget = 8.0
        # The projected-coordinate backend checks its budget between complete
        # candidate evaluations; one dense evaluation is not interruptible.
        # Keep a wall-clock guard outside that backend so returning an already
        # validated incumbent never misses the public method deadline.
        deadline_return_guard = min(
            2.0, max(0.75, 0.10 * problem.method_time_budget)
        )
        refinement_window = max(0.0, refinement_remaining - deadline_return_guard)
        if refinement_window <= validation_reserve + minimum_refinement_budget:
            if incumbent is not None:
                incumbent.elapsed = time.monotonic() - started
                incumbent.diagnostics.update({
                    "method": self.name,
                    "rolling": False,
                    "configuration_corridor": corridor_diagnostics,
                    "fallback": "validated_configuration_corridor_incumbent",
                    "optimizer_status": "skipped_to_preserve_deadline",
                    "refinement_remaining_s": refinement_remaining,
                    "refinement_window_s": refinement_window,
                    "deadline_return_guard_s": deadline_return_guard,
                    "minimum_refinement_budget_s": minimum_refinement_budget,
                    "incumbent_control_points": len(corridor_reference),
                    "optimizer_control_points": len(optimizer_reference),
                    "stage_times_s": stage_times,
                })
                return incumbent
            return _failure(self.name, started, "plan_timeout", {
                "stage": "incumbent_validation",
                "configuration_corridor": corridor_diagnostics,
                "stage_times_s": stage_times,
            })
        batch_config = replace(
            batch_config,
            optimizer_time_budget=refinement_window - validation_reserve,
        )
        seed = SeedPlan(
            states=optimizer_reference,
            risk_knots=list(range(len(optimizer_reference))),
            risk_spans=list(range(max(0, len(optimizer_reference) - 3))),
            diagnostics={
                "shape_mode": "theta_phi",
                "edge_arm_clearance_min": float("inf"),
                "edge_body_clearance_min": float("inf"),
                "terminal_shape": None if problem.terminal_state is None
                else problem.terminal_state[[THETA, PHI]].tolist(),
            },
        )
        validator = SweptVolumeValidator(
            problem.world, problem.arm, problem.mass, problem.limits,
            batch_config, terminal_target=problem.terminal_target,
            terminal_tolerance=problem.terminal_tolerance,
        )
        result = WholeBodyOptimizer(
            problem.world, problem.arm, problem.mass, problem.limits,
            batch_config, validator,
        ).optimize(
            seed, optimizer_reference, wall_time_budget=refinement_window,
            validation_reserve=validation_reserve,
        )
        stage_times["batch_refinement"] = max(
            0.0, time.monotonic() - started - corridor_elapsed
            - incumbent_validation_elapsed
        )
        result.elapsed = time.monotonic() - started
        result.diagnostics.update({
            "method": self.name,
            "adaptation": "global_batch_joint_optimization_with_exact_continuum_swept_geometry",
            "rolling": False,
            "v2_risk_seed": False,
            "v2_repair": False,
            "configuration_corridor": corridor_diagnostics,
            "incumbent_control_points": len(corridor_reference),
            "optimizer_control_points": len(optimizer_reference),
            "deadline_return_guard_s": deadline_return_guard,
            "refinement_window_s": refinement_window,
            "validation_reserve_s": validation_reserve,
            "stage_times_s": stage_times,
        })
        if result.trajectory is not None:
            result.trajectory.source = self.name
            result.trajectory.metadata["method"] = self.name
        if result.elapsed > problem.method_time_budget:
            timeout = _failure(self.name, started, "plan_timeout", {
                **result.diagnostics,
                "wall_time_s": result.elapsed,
                "method_time_budget_s": problem.method_time_budget,
                "candidate_was_feasible": bool(result.success),
            })
            return timeout
        if not result.success and incumbent is not None:
            incumbent.elapsed = time.monotonic() - started
            incumbent.diagnostics.update({
                "method": self.name,
                "adaptation": "global_batch_joint_optimization_with_exact_continuum_swept_geometry",
                "rolling": False,
                "configuration_corridor": corridor_diagnostics,
                "incumbent_control_points": len(corridor_reference),
                "optimizer_control_points": len(optimizer_reference),
                "deadline_return_guard_s": deadline_return_guard,
                "refinement_window_s": refinement_window,
                "fallback": "validated_configuration_corridor_incumbent",
                "optimizer_status": result.status,
                "optimizer_diagnostics": result.diagnostics,
                "stage_times_s": stage_times,
            })
            if incumbent.elapsed <= problem.method_time_budget:
                return incumbent
        return result


class RollingWholeBodyV2Method:
    name = "rolling_wb_v2"

    def plan(self, problem):
        started = time.monotonic()
        config = replace(
            problem.config, optimizer_time_budget=problem.method_time_budget
        )
        terminal = problem.terminal_state
        seed = WholeBodySeedPlanner(
            problem.world, problem.arm, problem.mass, problem.limits, config
        ).plan(
            problem.reference_states, problem.current_state,
            terminal_state=terminal,
        )
        if terminal is not None:
            seed.diagnostics["terminal_shape"] = terminal[[THETA, PHI]].tolist()
        remaining_budget = problem.method_time_budget - (time.monotonic() - started)
        if remaining_budget <= 0.0:
            return _failure(self.name, started, "plan_timeout", {
                "stage": "whole_body_seed",
                "method_time_budget_s": problem.method_time_budget,
            })
        validation_reserve = min(
            3.0, max(0.75, 0.20 * problem.method_time_budget)
        )
        if remaining_budget <= validation_reserve:
            return _failure(self.name, started, "plan_timeout", {
                "stage": "whole_body_seed",
                "validation_reserve_s": validation_reserve,
                "method_time_budget_s": problem.method_time_budget,
            })
        config = replace(
            config, optimizer_time_budget=remaining_budget - validation_reserve
        )
        validator = SweptVolumeValidator(
            problem.world, problem.arm, problem.mass, problem.limits, config,
            terminal_target=problem.terminal_target,
            terminal_tolerance=problem.terminal_tolerance,
        )
        result = WholeBodyOptimizer(
            problem.world, problem.arm, problem.mass, problem.limits,
            config, validator,
        ).optimize(
            seed, problem.reference_states, wall_time_budget=remaining_budget,
            validation_reserve=validation_reserve,
        )
        result.diagnostics.update({
            "method": self.name, "rolling": True,
            "comparison_interface": "shared",
        })
        result.elapsed = time.monotonic() - started
        if result.elapsed > problem.method_time_budget:
            return _failure(self.name, started, "plan_timeout", {
                **result.diagnostics,
                "wall_time_s": result.elapsed,
                "method_time_budget_s": problem.method_time_budget,
                "candidate_was_feasible": bool(result.success),
            })
        return result


def create_method(name, **options):
    name = str(name).strip().lower().replace("-", "_")
    aliases = {
        "fixed": "fixed_morphology",
        "aet": "aet_sequential_minjerk",
        "aet_minjerk": "aet_sequential_minjerk",
        "rrtstar": "informed_rrtstar_wb",
        "rrtstar_uam": "informed_rrtstar_wb",
        "am_planner": "global_batch_wb",
        "am_planner_adapted": "global_batch_wb",
        "v2": "rolling_wb_v2",
    }
    name = aliases.get(name, name)
    if name == "fixed_morphology":
        return FixedMorphologyMethod(**options)
    if name == "aet_sequential_minjerk":
        return AetSequentialMinJerkMethod()
    if name == "informed_rrtstar_wb":
        return InformedRrtStarWholeBodyMethod(**options)
    if name == "global_batch_wb":
        return GlobalBatchWholeBodyMethod()
    if name == "rolling_wb_v2":
        return RollingWholeBodyV2Method()
    raise ValueError("unknown comparison method %r; expected %s" % (name, METHOD_NAMES))


# Import compatibility for analysis scripts tied to the schema-v5 pilot.  New
# manifests and paper tables must use the honest schema-v6 names above.
AetMinimumJerkMethod = AetSequentialMinJerkMethod
RrtStarUamMethod = InformedRrtStarWholeBodyMethod
AmPlannerAdaptedMethod = GlobalBatchWholeBodyMethod
