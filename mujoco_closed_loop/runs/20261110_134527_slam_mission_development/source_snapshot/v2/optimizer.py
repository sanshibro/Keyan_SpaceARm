import time

import numpy as np

from .geometry import coupling_metrics, world_arm_points, world_body_points, wrap_angle
from .trajectory import SplineTrajectory, initial_span_durations, resample_states
from .types import OptimizationResult, PHI, THETA, ValidationResult, YAW


class _TimeBudgetExceeded(RuntimeError):
    pass


class WholeBodyOptimizer:
    @staticmethod
    def _seed_edge_violates_limits(seed_arm, seed_body, limits):
        """Use the documented formal limits; robustness reserve is config-owned."""
        return seed_arm < limits.arm_margin or seed_body < limits.body_margin

    """Jointly optimize UAV pose, continuum shape, and non-uniform span time."""

    def __init__(self, world, arm_model, mass_model, limits, config, validator, body_world=None,
                 center_keepout_world=None):
        self.world = world
        self.body_world = body_world or world
        self.center_keepout_world = center_keepout_world
        self.arm = arm_model
        self.mass = mass_model
        self.limits = limits
        self.config = config
        self.validator = validator
        self._stationary_start = True
        self._initial_rate = None

    @property
    def _theta_design_max(self):
        return max(
            self.arm.theta_min,
            self.arm.theta_max - float(self.config.theta_design_margin),
        )

    @property
    def _velocity_design_max(self):
        return float(self.limits.velocity_max) * float(
            self.config.design_velocity_ratio
        )

    def _body_clearance(self, state):
        body = min(10.0, self.body_world.minimum_distance(world_body_points(state, self.limits))[0])
        if self.center_keepout_world is not None:
            keepout = self.center_keepout_world.signed_distance(state[:3])[0]
            body = min(body, keepout + self.limits.body_margin)
        return body

    def _clearance_samples(self, states):
        """Evaluate optimizer collision samples in two batched world queries.

        This only changes query scheduling.  The body surface points, arm
        backbone points, exact distance models, margins and collocation states
        are identical to the scalar implementation.
        """
        states = np.asarray(states, dtype=float)
        body_groups = np.stack([
            world_body_points(current, self.limits) for current in states
        ])
        arm_groups = np.stack([
            world_arm_points(current, self.arm) for current in states
        ])
        body, _ = self.body_world.minimum_distances_grouped(body_groups)
        arm, _ = self.world.minimum_distances_grouped(arm_groups)
        body = np.minimum(body, 10.0)
        arm = np.minimum(arm, 10.0) - self.arm.radius
        if self.center_keepout_world is not None:
            keepout, _ = self.center_keepout_world.minimum_distances_grouped(
                states[:, None, :3]
            )
            body = np.minimum(body, keepout + self.limits.body_margin)
        return [
            (float(body[index]), float(arm[index]),
             coupling_metrics(current, self.arm, self.mass))
            for index, current in enumerate(states)
        ]

    def _decode(self, vector):
        variable_count = self.config.optimization_basis_count * 6
        corrections = np.asarray(vector[:variable_count], dtype=float).reshape(
            self.config.optimization_basis_count, 6
        )
        controls = self._seed_base + self._basis.dot(corrections)
        durations = np.asarray(vector[variable_count:], dtype=float)
        if self._initial_rate is not None and len(controls) >= 2:
            controls[1] = controls[0] + self._initial_rate * durations[0] / 3.0
        # Correction bounds alone do not guarantee physical state bounds.
        # Project bounded control coordinates before constructing the clamped
        # B-spline; its convex-hull property then enforces these limits over
        # the complete continuous trajectory, not only at collocation points.
        controls[:, 2] = np.clip(controls[:, 2], self.limits.z_min, self.limits.z_max)
        # Control zero is the measured boundary and may already lie inside the
        # hard/design reserve.  Never move it merely to manufacture margin;
        # all future controls use the interior design ceiling.
        controls[0, THETA] = np.clip(
            controls[0, THETA], self.arm.theta_min, self.arm.theta_max
        )
        controls[1:, THETA] = np.clip(
            controls[1:, THETA], self.arm.theta_min, self._theta_design_max
        )
        return controls, durations

    def _bounds(self, duration_count):
        bounds = []
        per_dimension = (0.30, 0.30, 0.22, np.deg2rad(35.0), np.deg2rad(70.0), np.deg2rad(90.0))
        for _ in range(self.config.optimization_basis_count):
            for dimension, value in enumerate(per_dimension):
                bounds.append((0.0, 0.0) if self._phi_locked and dimension == PHI
                              else (-value, value))
        bounds.extend([(self.config.dt_min, self.config.dt_max)] * duration_count)
        return bounds

    def _make_basis(self, count):
        basis_count = self.config.optimization_basis_count
        parameter = np.linspace(0.0, 1.0, count)
        centers = np.linspace(0.18, 0.90, basis_count)
        width = max(0.20, 0.75 / max(basis_count, 1))
        basis = np.exp(-0.5 * ((parameter[:, None] - centers[None, :]) / width) ** 2)
        row_sum = np.sum(basis, axis=1)
        basis = basis / np.maximum(row_sum[:, None], 1e-9)
        basis[0, :] = 0.0
        return basis

    def _constraint_values(self, spline, risk_spans, terminal_reference, samples=None,
                           terminal_shape=None):
        if samples is None:
            times = spline.collocation_times(
                self.config.collocation_per_span,
                self.config.risk_collocation_per_span,
                risk_spans,
            )
            state = spline.evaluate(times, 0)
            velocity = spline.evaluate(times, 1)
            acceleration = spline.evaluate(times, 2)
            jerk = spline.evaluate(times, 3)
            clearance = None
        else:
            state, velocity, acceleration, jerk, clearance = samples
        result = []
        torque_limit = self.limits.resolved_torque_max(self.mass.total)
        gravity = np.array([0.0, 0.0, 9.81])
        for index, current in enumerate(state):
            if clearance is None:
                body = self._body_clearance(current)
                arm = min(10.0, self.world.minimum_distance(world_arm_points(current, self.arm))[0]) - self.arm.radius
                coupling = coupling_metrics(current, self.arm, self.mass)
            else:
                body, arm, coupling = clearance[index]
            speed = np.linalg.norm(velocity[index, :3])
            accel = np.linalg.norm(acceleration[index, :3])
            jerk_norm = np.linalg.norm(jerk[index, :3])
            thrust = acceleration[index, :3] + gravity
            thrust_ratio = np.linalg.norm(thrust) / 9.81
            tilt_margin = np.tan(self.limits.tilt_max) * max(thrust[2], 1e-3) - np.linalg.norm(thrust[:2])
            result.extend([
                body - self.limits.body_margin,
                arm - self.limits.arm_margin,
                self.limits.com_max - coupling["com_offset"],
                torque_limit - coupling["gravity_torque"],
                self._velocity_design_max - speed,
                self.limits.acceleration_max - accel,
                self.limits.jerk_max - jerk_norm,
                self.limits.yaw_rate_max - abs(velocity[index, YAW]),
                self.limits.theta_rate_max - abs(velocity[index, THETA]),
                self.limits.phi_rate_max - abs(velocity[index, PHI]),
                self.limits.yaw_acceleration_max - abs(acceleration[index, YAW]),
                self.limits.theta_acceleration_max - abs(acceleration[index, THETA]),
                self.limits.phi_acceleration_max - abs(acceleration[index, PHI]),
                tilt_margin,
                thrust_ratio - self.limits.thrust_min_ratio,
                self.limits.thrust_max_ratio - thrust_ratio,
                current[2] - self.limits.z_min,
                self.limits.z_max - current[2],
                current[THETA] - self.arm.theta_min,
                (self.arm.theta_max if index == 0 else self._theta_design_max)
                - current[THETA],
            ])
        terminal_position_tolerance = 0.06 if terminal_shape is not None else self.config.terminal_tolerance
        result.append(terminal_position_tolerance - np.linalg.norm(state[-1, :3] - terminal_reference[:3]))
        if terminal_shape is not None:
            result.append(np.deg2rad(5.0) - abs(state[-1, THETA] - terminal_shape[0]))
            result.append(np.deg2rad(8.0) - abs(wrap_angle(state[-1, PHI] - terminal_shape[1])))
            # The terminal task is defined at the tool, not at the UAV
            # center.  Previously the optimizer could use the allowed 6 cm
            # center-position slack to reduce its objective, move a 1 cm
            # reachability seed to a 7 cm tool error, and then be rejected by
            # the independent validator.  Put the validator's exact terminal
            # contract into the optimization constraints as well.
            if self.validator is not None and self.validator.terminal_target is not None:
                terminal_tip = world_arm_points(state[-1], self.arm)[-1]
                result.append(
                    self.validator.terminal_tolerance
                    - np.linalg.norm(terminal_tip - self.validator.terminal_target)
                )
        return np.asarray(result, dtype=float)

    def _collision_proxy(self, controls, span_durations, risk_spans):
        """Return conservative collocation clearance slacks for repair ranking.

        This proxy never accepts a trajectory.  It only orders a small set of
        deterministic repair candidates before the unchanged dense validator
        proves (or rejects) them.  Using both body and arm slack avoids the old
        failure mode where repairing the reported arm collision immediately
        created a body collision, or vice versa.
        """
        spline = self._spline(controls, span_durations)
        times = spline.collocation_times(
            max(5, self.config.collocation_per_span),
            max(11, self.config.risk_collocation_per_span),
            risk_spans,
        )
        clearance = self._clearance_samples(spline.evaluate(times, 0))
        body_slack = min(item[0] for item in clearance) - self.limits.body_margin
        arm_slack = min(item[1] for item in clearance) - self.limits.arm_margin
        return float(body_slack), float(arm_slack)

    @staticmethod
    def _risk_control_indices(control_count, risk_spans, preserve_terminal):
        """Controls whose cubic basis can influence a risky spline span."""
        active = sorted(set(
            index for span in risk_spans
            for index in range(max(1, int(span)), min(control_count, int(span) + 4))
        ))
        if not active:
            active = list(range(1, control_count))
        if preserve_terminal:
            active = [index for index in active if index != control_count - 1]
        return active

    def _theta_only_retract_hold(self, controls, span_durations):
        """Rate-conditioned compact morphology used by the fixed-phi set."""
        repaired = controls.copy()
        minimum_dt = max(self.config.dt_min, float(np.min(span_durations)))
        theta_delta = 0.80 * self.limits.theta_rate_max * minimum_dt / 3.0
        for index in range(1, len(repaired)):
            target = min(
                self._theta_design_max - 1e-7,
                repaired[index - 1, THETA] + theta_delta,
            )
            repaired[index, THETA] = max(repaired[index, THETA], target)
        repaired[:, THETA] = np.minimum(
            repaired[:, THETA], self._theta_design_max - 1e-7
        )
        return repaired

    def _translation_repair_candidates(self, controls, span_durations, risk_spans,
                                       preserve_terminal=False):
        """Rank generic local whole-body translations by body+arm clearance.

        The 26 directions form a scene-independent 3-D stencil.  For each
        direction we retain the smallest displacement that establishes a
        small proxy reserve, or otherwise the displacement with the best
        minimum formal slack.  The measured initial control is never moved;
        a terminal task control is also fixed so tool reachability is not
        silently changed.
        """
        active = self._risk_control_indices(
            len(controls), risk_spans, preserve_terminal
        )
        if not active:
            return []
        directions = []
        for dx in (-1.0, 0.0, 1.0):
            for dy in (-1.0, 0.0, 1.0):
                for dz in (-1.0, 0.0, 1.0):
                    direction = np.asarray([dx, dy, dz], dtype=float)
                    norm = float(np.linalg.norm(direction))
                    if norm > 0.0:
                        if self.config.repair_axis_only and np.count_nonzero(direction) != 1:
                            continue
                        directions.append(direction / norm)
        distances = (
            (0.06, 0.09, 0.12) if self.config.repair_axis_only
            else (0.03, 0.06, 0.09, 0.12)
        )
        proxy_reserve = max(0.0, float(self.config.repair_proxy_reserve))
        ranked = []
        for direction in directions:
            alternatives = []
            for distance in distances:
                repaired = controls.copy()
                repaired[active, :3] += distance * direction
                repaired[active, 2] = np.clip(
                    repaired[active, 2], self.limits.z_min, self.limits.z_max
                )
                if np.allclose(repaired, controls, atol=1e-12, rtol=0.0):
                    continue
                repaired_durations = np.clip(
                    # Four percent dynamic reserve prevents a geometrically
                    # safe translation from landing numerically just above a
                    # hard velocity/rate bound after spline reconstruction.
                    span_durations * (1.04 + 0.65 * distance),
                    self.config.dt_min, self.config.dt_max,
                )
                body_slack, arm_slack = self._collision_proxy(
                    repaired, repaired_durations, risk_spans
                )
                alternatives.append({
                    "distance": distance,
                    "controls": repaired,
                    "durations": repaired_durations,
                    "body_slack": body_slack,
                    "arm_slack": arm_slack,
                    "minimum_slack": min(body_slack, arm_slack),
                })
            if not alternatives:
                continue
            reserved = [
                item for item in alternatives
                if item["minimum_slack"] >= proxy_reserve
            ]
            if reserved:
                selected = min(reserved, key=lambda item: item["distance"])
                establishes_reserve = True
            else:
                selected = max(
                    alternatives,
                    key=lambda item: (
                        item["minimum_slack"],
                        item["body_slack"] + item["arm_slack"],
                        -item["distance"],
                    ),
                )
                establishes_reserve = False
            label = "whole_body_shift_%+.3f_%+.3f_%+.3f" % tuple(
                selected["distance"] * direction
            )
            ranked.append((
                establishes_reserve,
                selected["minimum_slack"],
                selected["body_slack"] + selected["arm_slack"],
                -selected["distance"],
                label,
                selected["controls"],
                selected["durations"],
            ))
        ranked.sort(key=lambda item: item[:4], reverse=True)
        return [
            (item[4], item[5], item[6]) for item in ranked
        ]

    def _spline(self, controls, span_durations):
        return SplineTrajectory(
            controls, span_durations,
            stationary_start=self._stationary_start,
            stationary_end=True,
        )

    def optimize(self, seed_plan, reference_states, wall_time_budget=None,
                 validation_reserve=0.0, initial_rate=None):
        started = time.monotonic()
        total_budget = (
            float("inf") if wall_time_budget is None
            else max(0.05, float(wall_time_budget))
        )
        validation_reserve = max(0.0, float(validation_reserve))
        solver_budget = min(
            self.config.optimizer_time_budget,
            max(0.01, total_budget - validation_reserve),
        )
        seed = np.asarray(seed_plan.states, dtype=float)
        self._phi_locked = seed_plan.diagnostics.get("shape_mode") == "theta_only_fixed_phi"
        reference = resample_states(reference_states, self.config.knot_count)
        reference[:, YAW] = np.unwrap(reference[:, YAW])
        reference[:, PHI] = np.unwrap(reference[:, PHI])
        terminal_shape = seed_plan.diagnostics.get("terminal_shape")
        if terminal_shape is not None:
            terminal_shape = np.asarray(terminal_shape, dtype=float)
        durations = initial_span_durations(seed, 3, self.config.nominal_speed, self.config.dt_min, self.config.dt_max)
        # Short terminal translations can still require a large yaw or arm
        # morphology change.  The unified clock must be sized by every
        # commanded degree of freedom, not UAV distance alone.
        dynamic_dt = max(
            3.0 * float(np.max(np.abs(np.diff(seed[:, YAW])))) / max(self.limits.yaw_rate_max, 1e-6),
            3.0 * float(np.max(np.abs(np.diff(seed[:, THETA])))) / max(self.limits.theta_rate_max, 1e-6),
            3.0 * float(np.max(np.abs(np.diff(seed[:, PHI])))) / max(self.limits.phi_rate_max, 1e-6),
        )
        durations[:] = np.clip(max(float(durations[0]), dynamic_dt * 1.08),
                               self.config.dt_min, self.config.dt_max)
        conditioned_initial_rate = None
        if initial_rate is not None and len(seed) >= 2:
            conditioned_initial_rate = np.asarray(initial_rate, dtype=float).reshape(6)
            if np.all(np.isfinite(conditioned_initial_rate)):
                # For a clamped cubic B-spline, qdot(0) equals
                # 3(P1-P0)/dt0.  Condition P1 so a rolling successor inherits
                # the executor-predicted full-body rate instead of restarting
                # from rest.  The unchanged dense validator remains the final
                # authority for the modified curve.
                seed = seed.copy()
                seed[1] = seed[0] + conditioned_initial_rate * durations[0] / 3.0
                seed[1, 2] = np.clip(seed[1, 2], self.limits.z_min, self.limits.z_max)
                seed[1, THETA] = np.clip(
                    seed[1, THETA], self.arm.theta_min, self._theta_design_max
                )
                self._stationary_start = False
                self._initial_rate = conditioned_initial_rate.copy()
            else:
                conditioned_initial_rate = None
                self._stationary_start = True
                self._initial_rate = None
        else:
            self._stationary_start = True
            self._initial_rate = None
        self._seed_base = seed.copy()
        self._basis = self._make_basis(len(seed))
        if self._initial_rate is not None and len(seed) >= 2:
            self._basis[1, :] = 0.0
        initial = np.concatenate((np.zeros(self.config.optimization_basis_count * 6), durations))
        best = {
            "x": initial.copy(), "objective": float("inf"),
            "constraint_min": -float("inf"),
        }

        def check_time():
            if time.monotonic() - started > solver_budget:
                raise _TimeBudgetExceeded()

        last_validation_elapsed = 0.0

        def enough_time_for_another_validation():
            expected = max(validation_reserve, 1.25 * last_validation_elapsed)
            return time.monotonic() - started + expected < total_budget

        def objective(vector):
            check_time()
            controls, span_durations = self._decode(vector)
            spline = self._spline(controls, span_durations)
            times = spline.collocation_times(
                self.config.collocation_per_span,
                self.config.risk_collocation_per_span,
                seed_plan.risk_spans,
            )
            states = spline.evaluate(times, 0)
            velocity = spline.evaluate(times, 1)
            acceleration = spline.evaluate(times, 2)
            jerk = spline.evaluate(times, 3)
            normalized = times / max(spline.duration, 1e-6)
            ref_source = np.linspace(0.0, 1.0, len(reference))
            ref = np.column_stack([np.interp(normalized, ref_source, reference[:, dim]) for dim in range(6)])
            weights = self.config.weights
            cost = weights["reference"] * np.mean(np.sum((states[:, :3] - ref[:, :3]) ** 2, axis=1))
            cost += weights["yaw_reference"] * np.mean(np.asarray([
                wrap_angle(a - b) ** 2 for a, b in zip(states[:, YAW], ref[:, YAW])
            ]))
            cost += weights["jerk"] * np.mean(np.sum(jerk[:, :3] ** 2, axis=1))
            cost += weights["arm_rate"] * np.mean(np.sum(velocity[:, [THETA, PHI]] ** 2, axis=1))
            cost += weights["arm_acceleration"] * np.mean(np.sum(acceleration[:, [THETA, PHI]] ** 2, axis=1))
            clearance_penalty = 0.0
            com_penalty = 0.0
            clearance_samples = self._clearance_samples(states)
            for body, arm, coupling in clearance_samples:
                clearance_penalty += max(0.0, self.limits.body_margin + self.config.risk_margin - body) ** 2
                clearance_penalty += max(0.0, self.limits.arm_margin + self.config.risk_margin - arm) ** 2
                com = coupling["com_offset"]
                com_penalty += (com / max(self.limits.com_max, 1e-3)) ** 2
            cost += weights["clearance"] * clearance_penalty / len(states)
            cost += weights["com"] * com_penalty / len(states)
            cost += weights["duration"] * spline.duration
            if len(span_durations) > 1:
                cost += weights["dt_smooth"] * np.mean(np.diff(span_durations) ** 2)
            cost += weights["terminal"] * np.sum((states[-1, :3] - reference[-1, :3]) ** 2)
            if terminal_shape is not None:
                terminal_shape_error = np.asarray([
                    states[-1, THETA] - terminal_shape[0],
                    wrap_angle(states[-1, PHI] - terminal_shape[1]),
                ])
                cost += 50.0 * (terminal_shape_error[0] ** 2 + 0.35 * terminal_shape_error[1] ** 2)
                if self.validator is not None and self.validator.terminal_target is not None:
                    terminal_tip = world_arm_points(states[-1], self.arm)[-1]
                    terminal_tip_error = np.linalg.norm(
                        terminal_tip - self.validator.terminal_target
                    )
                    cost += 250.0 * terminal_tip_error ** 2
            constraint_values = self._constraint_values(
                spline, seed_plan.risk_spans, reference[-1],
                (states, velocity, acceleration, jerk, clearance_samples), terminal_shape,
            )
            violations = np.minimum(constraint_values, 0.0)
            cost += 5000.0 * np.sum(violations ** 2)
            if cost < best["objective"]:
                best["objective"] = float(cost)
                best["x"] = np.asarray(vector).copy()
                best["constraint_min"] = float(np.min(constraint_values))
            return float(cost)

        status = "solver_failed"
        solver_result = None
        iterations = 0
        try:
            bounds = np.asarray(self._bounds(len(durations)), dtype=float)
            lower, upper = bounds[:, 0], bounds[:, 1]
            current = np.clip(initial.copy(), lower, upper)
            current_cost = objective(current)
            radius = 0.08 * np.maximum(upper - lower, 1e-3)
            maximum_radius = radius.copy()
            basis_count = self.config.optimization_basis_count
            shape_theta = [6 * index + THETA for index in range(basis_count)]
            shape_phi = ([] if self._phi_locked else
                         [6 * index + PHI for index in range(basis_count)])
            altitude = [6 * index + 2 for index in range(basis_count)]
            remaining = [index for index in range(len(current)) if index not in shape_theta + shape_phi + altitude]
            coordinate_order = shape_theta + shape_phi + altitude + remaining
            # Projected coordinate pattern search gives a real anytime
            # iteration in two objective calls. Shape coordinates are visited
            # first because they dominate tight-passage feasibility.
            for iteration in range(self.config.optimizer_max_iterations):
                check_time()
                coordinate = coordinate_order[iteration % len(coordinate_order)]
                direction = np.zeros(len(current))
                direction[coordinate] = 1.0
                plus = np.clip(current + radius * direction, lower, upper)
                minus = np.clip(current - radius * direction, lower, upper)
                plus_cost = objective(plus)
                minus_cost = objective(minus)
                if plus_cost < current_cost or minus_cost < current_cost:
                    if plus_cost <= minus_cost:
                        current, current_cost = plus, plus_cost
                    else:
                        current, current_cost = minus, minus_cost
                    radius[coordinate] = min(maximum_radius[coordinate], radius[coordinate] * 1.05)
                else:
                    radius[coordinate] *= 0.65
                iterations = iteration + 1
                if np.max(radius / np.maximum(maximum_radius, 1e-9)) < 0.02:
                    status = "converged"
                    break
            else:
                status = "iteration_limit"
            candidate_vector = best["x"]
        except _TimeBudgetExceeded:
            status = "time_budget"
            candidate_vector = best["x"]
        except Exception as error:
            status = "exception"
            candidate_vector = best["x"]
            solver_result = error
        controls, span_durations = self._decode(candidate_vector)
        spline = self._spline(controls, span_durations)
        # A candidate that already violates an exact planning constraint at a
        # collocation state cannot satisfy the planner's own acceptance
        # contract.  Do not spend several seconds running the independent
        # dense proof on that known-invalid candidate; validate the anytime
        # seed incumbent directly.  Final accepted trajectories still pass
        # the unchanged full-density independent validator.
        candidate_validation_skipped = best["constraint_min"] < -1e-9
        if candidate_validation_skipped:
            seed_spline = self._spline(seed, durations)
            controls = seed.copy()
            span_durations = durations.copy()
            spline = seed_spline
            # The beam search already performs exact edge collision checks.
            # If it records an unsafe seed edge, avoid proving the same known
            # failure again at hundreds of dense samples; proceed directly to
            # repair and reserve the full independent proof for the candidate
            # that may actually be accepted.
            seed_arm = float(seed_plan.diagnostics.get("edge_arm_clearance_min", float("inf")))
            seed_body = float(seed_plan.diagnostics.get("edge_body_clearance_min", float("inf")))
            # Only reject a seed edge that violates the configured formal
            # limit.  The previous hidden +1 cm requirement could label a
            # formally feasible seed (for example 0.062 m clearance against a
            # 0.060 m limit) as a collision and skip the independent dense
            # validator entirely.  Robust reserve belongs in planner_v2.yaml,
            # not in an undocumented second threshold here.
            if self._seed_edge_violates_limits(seed_arm, seed_body, self.limits):
                reason = (
                    "arm_collision" if seed_arm - self.limits.arm_margin
                    <= seed_body - self.limits.body_margin else "body_collision"
                )
                validation = ValidationResult(
                    False, "unsafe_known_seed_edge", {
                        "arm_clearance_min": seed_arm,
                        "body_clearance_min": seed_body,
                    }, failure_reason=reason,
                )
            else:
                validation_started = time.monotonic()
                validation = self.validator.validate(seed_spline)
                last_validation_elapsed = time.monotonic() - validation_started
        else:
            validation_started = time.monotonic()
            validation = self.validator.validate(spline)
            last_validation_elapsed = time.monotonic() - validation_started
        repair_used = "none"
        repair_attempts_run = 0
        repair_attempt_labels = []
        repair_attempt_results = []
        # Preserve a feasible seed as the anytime incumbent.  A lower
        # penalized objective is never allowed to replace a trajectory that
        # passed the independent dense validator with an unsafe one.
        if (not validation.feasible and not candidate_validation_skipped
                and enough_time_for_another_validation()):
            seed_spline = self._spline(seed, durations)
            validation_started = time.monotonic()
            seed_validation = self.validator.validate(seed_spline)
            last_validation_elapsed = time.monotonic() - validation_started
            if seed_validation.feasible:
                controls = seed.copy()
                span_durations = durations.copy()
                spline = seed_spline
                validation = seed_validation
                repair_used = "validated_seed_fallback"
        if not validation.feasible:
            repair_candidates = []
            arm_repair_candidates = []
            if validation.failure_reason == "arm_collision":
                if self._phi_locked and terminal_shape is None:
                    # Theta-only passage mode: build one rate-conditioned
                    # retract-and-hold candidate.  Releasing immediately after
                    # a thin obstacle can make the cubic backbone sweep back
                    # through it; retaining the compact shape to the rolling
                    # horizon end is both safer and cheaper than exploring phi.
                    repaired = self._theta_only_retract_hold(
                        controls, span_durations
                    )
                    arm_repair_candidates.append((
                        "theta_only_retract_hold", repaired, span_durations * 1.04,
                    ))
                if not self._phi_locked:
                    seed_arm = float(seed_plan.diagnostics.get(
                        "edge_arm_clearance_min", -float("inf")
                    ))
                    deficit = self.limits.arm_margin - seed_arm
                    if deficit > 0.02:
                        repair_angles = (36.0, 24.0, 12.0)
                    elif deficit > 0.005:
                        repair_angles = (24.0, 36.0, 12.0)
                    else:
                        repair_angles = (12.0, 24.0, 36.0)
                    for delta_deg in repair_angles:
                        repaired = controls.copy()
                        active = self._risk_control_indices(
                            len(repaired), seed_plan.risk_spans,
                            preserve_terminal=terminal_shape is not None,
                        )
                        repaired[active, THETA] = np.clip(
                            repaired[active, THETA] + np.deg2rad(delta_deg),
                            self.arm.theta_min, self._theta_design_max,
                        )
                        # Do not spend a dense proof on a no-op at theta_max.
                        if not np.allclose(repaired, controls, atol=1e-12, rtol=0.0):
                            arm_repair_candidates.append((
                                "arm_retract_%.0fdeg" % delta_deg,
                                repaired, span_durations * 1.03,
                            ))
                # With a proven fixed bending plane, increasing theta is the
                # lowest-dimensional collision response and must be attempted
                # before the expensive 3-D translation stencil.  In full
                # theta+phi mode retain the historical translation-first order.
                if self._phi_locked:
                    repair_candidates.extend(arm_repair_candidates)
            skip_translation = (
                self._phi_locked and validation.failure_reason == "arm_collision"
                and bool(arm_repair_candidates)
            )
            if (validation.failure_reason in ("arm_collision", "body_collision")
                    and not skip_translation):
                # Collision type is only the first failing sample reported by
                # the proof.  Repair body and arm clearance jointly, because
                # both envelopes can be active in a narrow passage.
                repair_candidates.extend(self._translation_repair_candidates(
                    controls, span_durations, seed_plan.risk_spans,
                    preserve_terminal=terminal_shape is not None,
                ))
            if validation.failure_reason == "arm_collision" and not self._phi_locked:
                repair_candidates.extend(arm_repair_candidates)
            if validation.failure_reason == "arm_collision":
                pass
            elif validation.failure_reason != "body_collision":
                metrics = validation.metrics
                scale = max(
                    1.0,
                    metrics.get("velocity_max", 0.0) / self._velocity_design_max,
                    np.sqrt(metrics.get("acceleration_max", 0.0) / self.limits.acceleration_max),
                    np.cbrt(metrics.get("jerk_max", 0.0) / self.limits.jerk_max),
                    metrics.get("theta_rate_max", 0.0) / self.limits.theta_rate_max,
                    metrics.get("phi_rate_max", 0.0) / self.limits.phi_rate_max,
                ) * self.config.dynamic_repair_reserve
                repair_candidates.append(("time_scale_%.3f" % scale, controls.copy(), span_durations * scale))
            # A repair candidate is accepted only after the unchanged dense
            # independent proof.  Limit how many such proofs one online cycle
            # may spend on already-invalid alternatives; a later rolling
            # cycle can try a new seed while the old certified trajectory
            # continues executing.  This bounds rejection latency without
            # weakening the proof required for acceptance.
            for label, repaired_controls, repaired_durations in repair_candidates[
                    :max(0, int(self.config.repair_attempts))]:
                if not enough_time_for_another_validation():
                    break
                repair_attempts_run += 1
                repair_attempt_labels.append(label)
                repaired_durations = np.clip(repaired_durations, self.config.dt_min, self.config.dt_max)
                repaired_spline = self._spline(repaired_controls, repaired_durations)
                validation_started = time.monotonic()
                repaired_validation = self.validator.validate(
                    repaired_spline, stop_on_failure=True
                )
                last_validation_elapsed = time.monotonic() - validation_started
                repair_attempt_results.append({
                    "label": label,
                    "feasible": bool(repaired_validation.feasible),
                    "failure_reason": repaired_validation.failure_reason,
                    "metrics": {
                        key: repaired_validation.metrics.get(key)
                        for key in (
                            "body_clearance_min", "arm_clearance_min",
                            "velocity_max", "theta_rate_max", "phi_rate_max",
                        ) if key in repaired_validation.metrics
                    },
                })
                if repaired_validation.feasible:
                    controls, span_durations = repaired_controls, repaired_durations
                    spline, validation, repair_used = repaired_spline, repaired_validation, label
                    break
        trajectory = spline.sample(
            self.config.output_dt, source="rolling_whole_body_v2", objective=best["objective"],
            metadata={"solver_status": status, "risk_knots": seed_plan.risk_knots,
                      "risk_spans": seed_plan.risk_spans, "span_durations": span_durations.tolist(),
                      "repair": repair_used,
                      "conditioned_initial_rate": (
                          None if conditioned_initial_rate is None
                          else conditioned_initial_rate.tolist()
                      )},
        )
        return OptimizationResult(
            success=validation.feasible,
            status=("validated_" + status) if validation.feasible else ("rejected_" + status),
            trajectory=trajectory, objective=float(best["objective"]),
            iterations=iterations, elapsed=float(time.monotonic() - started),
            validation=validation,
            diagnostics={"solver_message": str(getattr(solver_result, "message", solver_result)),
                         "backend": "projected_coordinate_pattern_anytime",
                         "seed": seed_plan.diagnostics, "repair": repair_used,
                         "repair_attempts_run": repair_attempts_run,
                         "repair_attempt_labels": repair_attempt_labels,
                         "repair_attempt_results": repair_attempt_results,
                         "candidate_collocation_min": best["constraint_min"],
                         "candidate_validation_skipped": candidate_validation_skipped,
                         "solver_time_budget_s": solver_budget,
                         "wall_time_budget_s": (
                             None if not np.isfinite(total_budget) else total_budget
                         ),
                         "validation_reserve_s": validation_reserve},
        )
