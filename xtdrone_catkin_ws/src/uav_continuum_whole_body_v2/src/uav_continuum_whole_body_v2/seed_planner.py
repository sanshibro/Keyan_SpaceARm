from dataclasses import dataclass, field

import numpy as np

from .geometry import coupling_metrics, interpolate_state, world_arm_points, world_body_points, wrap_angle
from .trajectory import resample_states
from .types import PHI, THETA, YAW


@dataclass
class SeedPlan:
    states: np.ndarray
    risk_knots: list
    risk_spans: list
    diagnostics: dict = field(default_factory=dict)


def reference_only_seed(reference_states, current_state, knot_count,
                        terminal_state=None, fixed_phi=None):
    """Build the deliberately unguided seed used only by the seed ablation."""
    states = resample_states(reference_states, knot_count)
    states[0] = np.asarray(current_state, dtype=float)
    if terminal_state is not None:
        states[-1] = np.asarray(terminal_state, dtype=float)
    if fixed_phi is not None:
        states[:, PHI] = float(fixed_phi)
    states[:, YAW] = np.unwrap(states[:, YAW])
    states[:, PHI] = np.unwrap(states[:, PHI])
    return SeedPlan(
        states=states,
        risk_knots=[],
        risk_spans=[],
        diagnostics={
            "seed_strategy": "reference_only_ablation",
            "shape_mode": "fixed_phi_ablation" if fixed_phi is not None else "reference_shape",
            "expanded": 0,
            "raw_risk_knots": [],
        },
    )


class WholeBodySeedPlanner:
    """Beam-search a collision-aware whole-body seed over the complete horizon."""

    def __init__(self, world, arm_model, mass_model, limits, config, body_world=None,
                 center_keepout_world=None, platform_geometry=None):
        self.platform_geometry = platform_geometry
        self.world = world
        self.body_world = body_world or world
        self.center_keepout_world = center_keepout_world
        self.arm = arm_model
        self.mass = mass_model
        self.limits = limits
        self.config = config
        self._clearance_cache = {}
        self._edge_cache = {}

    @property
    def _theta_design_max(self):
        return max(
            self.arm.theta_min,
            self.arm.theta_max - float(self.config.theta_design_margin),
        )

    @staticmethod
    def _state_key(state):
        return np.ascontiguousarray(state, dtype=np.float64).tobytes()

    def _clearance(self, state):
        key = self._state_key(state)
        cached = self._clearance_cache.get(key)
        if cached is not None:
            return cached
        if self.platform_geometry is not None:
            body, arm = self.platform_geometry.search_clearances(state)
            result = (body, arm, coupling_metrics(state, self.arm, self.mass))
            self._clearance_cache[key] = result
            return result
        body = self.body_world.minimum_distance(world_body_points(state, self.limits))[0]
        if self.center_keepout_world is not None:
            keepout = self.center_keepout_world.signed_distance(state[:3])[0]
            # Synthetic clearance preserves the existing body-margin test:
            # synthetic >= body_margin iff the UAV center is outside keepout.
            body = min(body, keepout + self.limits.body_margin)
        arm = self.world.minimum_distance(world_arm_points(state, self.arm))[0] - self.arm.radius
        coupling = coupling_metrics(state, self.arm, self.mass)
        result = (body, arm, coupling)
        self._clearance_cache[key] = result
        return result

    def _is_risk(self, state):
        body, arm, coupling = self._clearance(state)
        return (
            body < self.limits.body_margin + self.config.risk_margin
            or arm < self.limits.arm_margin + self.config.risk_margin
            or coupling["com_offset"] > 0.8 * self.limits.com_max
        )

    @staticmethod
    def _lateral(reference, index):
        if index == 0:
            tangent = reference[min(1, len(reference) - 1), :3] - reference[0, :3]
        elif index == len(reference) - 1:
            tangent = reference[-1, :3] - reference[-2, :3]
        else:
            tangent = reference[index + 1, :3] - reference[index - 1, :3]
        planar = tangent[:2]
        norm = np.linalg.norm(planar)
        if norm < 1e-8:
            return np.array([1.0, 0.0, 0.0])
        return np.array([-planar[1] / norm, planar[0] / norm, 0.0])

    def _shape_candidates(self, reference_state, risk, current_shape, phi_locked=False):
        if not risk:
            raw = [current_shape, (reference_state[THETA], reference_state[PHI]), (0.0, current_shape[1])]
        else:
            thetas = np.asarray([
                np.deg2rad(0.0), np.deg2rad(25.0), np.deg2rad(50.0),
                np.deg2rad(75.0), np.deg2rad(100.0), self._theta_design_max,
            ])
            phis = (
                np.asarray([current_shape[1]], dtype=float) if phi_locked else
                reference_state[PHI] + np.deg2rad([0.0, 45.0, -45.0, 90.0, -90.0, 180.0])
            )
            # Interleave bending magnitudes so truncation never removes all
            # strongly bent states needed by a tight vertical passage.
            # Task/current boundary conditions must precede the exploratory
            # grid.  Appending them after 36 grid entries and then slicing to
            # seed_shape_limit=12 silently removed a terminal 90 deg target.
            raw = [current_shape, (reference_state[THETA], reference_state[PHI])]
            raw.extend((theta, phi) for phi in phis for theta in thetas)
        unique = []
        for theta, phi in raw:
            candidate = (
                float(np.clip(theta, self.arm.theta_min, self._theta_design_max)),
                float(reference_state[PHI] + wrap_angle(phi - reference_state[PHI])),
            )
            if not any(abs(candidate[0] - item[0]) < 1e-6 and abs(wrap_angle(candidate[1] - item[1])) < 1e-6 for item in unique):
                unique.append(candidate)
        limit = (
            self.config.locked_phi_candidate_limit if phi_locked
            else self.config.seed_shape_limit
        )
        return unique[:limit]

    def _candidates(self, reference, index, current_shape, risk, phi_locked=False):
        state = reference[index]
        if index == 0:
            return [state.copy()]
        lateral = self._lateral(reference, index)
        offsets = [
            (0.0, 0.0),
            (self.config.seed_lateral_step, 0.0),
            (-self.config.seed_lateral_step, 0.0),
            (0.0, self.config.seed_z_step),
            (0.0, -self.config.seed_z_step),
        ] if risk else [(0.0, 0.0)]
        candidates = []
        for lateral_offset, z_offset in offsets[: self.config.seed_pose_limit]:
            pose = state.copy()
            pose[:3] += lateral * lateral_offset
            pose[2] = np.clip(pose[2] + z_offset, self.limits.z_min, self.limits.z_max)
            for theta, phi in self._shape_candidates(
                    state, risk, current_shape, phi_locked=phi_locked):
                item = pose.copy()
                item[THETA] = theta
                item[PHI] = phi
                candidates.append(item)
        candidates.sort(key=lambda item: self._node_cost(item, state, risk))
        shape_limit = (
            self.config.locked_phi_candidate_limit if phi_locked
            else self.config.seed_shape_limit
        )
        limit = max(shape_limit, self.config.seed_pose_limit)
        if not risk:
            return candidates[:limit]
        # Do not let point-wise COM cost remove all highly bent or alternate
        # direction states just before a swept transition.  Retain a compact,
        # deterministic morphology-diverse set for the beam search.
        selected = []

        def add(candidate):
            if len(selected) >= limit:
                return
            if not any(np.linalg.norm(candidate[:3] - item[:3]) < 1e-8
                       and abs(candidate[THETA] - item[THETA]) < 1e-8
                       and abs(wrap_angle(candidate[PHI] - item[PHI])) < 1e-8
                       for item in selected):
                selected.append(candidate)

        for candidate in candidates[:2]:
            add(candidate)
        theta_values = sorted(set(round(float(candidate[THETA]), 8) for candidate in candidates))
        for theta in theta_values:
            add(next(candidate for candidate in candidates if abs(candidate[THETA] - theta) < 1e-7))
        phi_values = []
        for candidate in candidates:
            value = round(float(candidate[PHI]), 8)
            if value not in phi_values:
                phi_values.append(value)
        for phi in phi_values:
            add(next(candidate for candidate in candidates if abs(candidate[PHI] - phi) < 1e-7))
        for candidate in candidates:
            add(candidate)
            if len(selected) >= limit:
                break
        return selected

    def _fixed_phi_feasible(self, reference, risk_knots, current_shape):
        """Prove that a fixed bending plane has a safe theta option per risk knot.

        This inexpensive active-set screen is deliberately necessary but not
        sufficient: seed edges and the independent dense validator still prove
        the continuous trajectory.  If the screen cannot establish a reserve,
        the normal theta+phi search remains available.
        """
        if not risk_knots:
            return True, float("inf")
        fixed_phi = float(current_shape[1])
        reserve = float(self.config.adaptive_phi_clearance_reserve)
        theta_values = np.asarray([
            np.deg2rad(0.0), np.deg2rad(25.0), np.deg2rad(50.0),
            np.deg2rad(75.0), np.deg2rad(100.0), self._theta_design_max,
        ])
        worst_best = float("inf")
        for index in risk_knots:
            best_slack = -float("inf")
            lateral = self._lateral(reference, index)
            offsets = [
                (0.0, 0.0),
                (self.config.seed_lateral_step, 0.0),
                (-self.config.seed_lateral_step, 0.0),
                (0.0, self.config.seed_z_step),
                (0.0, -self.config.seed_z_step),
            ][:self.config.seed_pose_limit]
            for lateral_offset, z_offset in offsets:
                for theta in theta_values:
                    candidate = reference[index].copy()
                    candidate[:3] += lateral * lateral_offset
                    candidate[2] = np.clip(
                        candidate[2] + z_offset, self.limits.z_min, self.limits.z_max
                    )
                    candidate[THETA] = theta
                    candidate[PHI] = fixed_phi
                    body, arm, coupling = self._clearance(candidate)
                    # Body clearance is independent of phi and therefore
                    # cannot justify unlocking the bending plane.  Pose/z
                    # candidates and the joint optimizer handle it separately.
                    slack = min(
                        arm - self.limits.arm_margin,
                        self.limits.com_max - coupling["com_offset"],
                    )
                    best_slack = max(best_slack, float(slack))
            worst_best = min(worst_best, best_slack)
            if best_slack < reserve:
                return False, float(worst_best)
        return True, float(worst_best)

    def _node_cost(self, state, reference, risk):
        body, arm, coupling = self._clearance(state)
        body_deficit = max(0.0, self.limits.body_margin - body)
        arm_deficit = max(0.0, self.limits.arm_margin - arm)
        infeasible = body_deficit > 0.0 or arm_deficit > 0.0 or coupling["com_offset"] > self.limits.com_max
        reference_cost = 8.0 * np.sum((state[:3] - reference[:3]) ** 2)
        yaw_cost = 0.3 * wrap_angle(state[YAW] - reference[YAW]) ** 2
        # A non-zero reference shape is an explicit whole-body task (notably
        # terminal reach-in) and must survive the beam ranking.  A zero-shape
        # transit reference is only a compactness preference; weighting it as
        # a hard reference suppresses the anticipatory bending required by a
        # tight passage and can leave every cubic swept edge unsafe.
        if abs(reference[THETA]) > np.deg2rad(1.0) or abs(wrap_angle(reference[PHI])) > np.deg2rad(1.0):
            shape_reference_cost = (
                4.0 * (state[THETA] - reference[THETA]) ** 2
                + 0.5 * wrap_angle(state[PHI] - reference[PHI]) ** 2
            )
        else:
            shape_reference_cost = 0.08 * state[THETA] ** 2
        clearance_cost = 400.0 * (body_deficit ** 2 + arm_deficit ** 2)
        if risk:
            # Keep morphology-capable states in the beam before the exact
            # collision span.  Without this guard-distance term, low-COM
            # straight-arm states dominate all beams until it is too late to
            # bend a cubic spline without sweeping through the obstacle.
            body_guard = max(0.0, self.limits.body_margin + self.config.risk_margin - body)
            arm_guard = max(0.0, self.limits.arm_margin + self.config.risk_margin - arm)
            clearance_cost += 1200.0 * (body_guard ** 2 + arm_guard ** 2)
        com_cost = 20.0 * (coupling["com_offset"] / max(self.limits.com_max, 1e-3)) ** 2
        shape_hold = 0.0 if risk else 0.1 * state[THETA] ** 2
        return (reference_cost + yaw_cost + shape_reference_cost + clearance_cost
                + com_cost + shape_hold + (1e5 if infeasible else 0.0))

    def _edge_cost(self, previous, current):
        key = (self._state_key(previous), self._state_key(current))
        cached = self._edge_cache.get(key)
        if cached is not None:
            return cached
        if self.platform_geometry is not None and not self.platform_geometry.edge(previous, current):
            self._edge_cache[key] = float("inf")
            return float("inf")
        dp = np.linalg.norm(current[:3] - previous[:3])
        dyaw = abs(wrap_angle(current[YAW] - previous[YAW]))
        dtheta = abs(current[THETA] - previous[THETA])
        dphi = abs(wrap_angle(current[PHI] - previous[PHI]))
        cost = dp ** 2 + 0.2 * dyaw ** 2 + 1.5 * dtheta ** 2 + 0.15 * dphi ** 2
        sample_count = max(2, int(np.ceil(dp / max(self.config.swept_translation_step * 2.0, 0.04))) + 1)
        sample_count = min(sample_count, max(2, self.config.seed_edge_max_samples))
        for ratio in np.linspace(0.0, 1.0, sample_count)[1:]:
            state = interpolate_state(previous, current, ratio)
            body, arm, coupling = self._clearance(state)
            if body < self.limits.body_margin or arm < self.limits.arm_margin or coupling["com_offset"] > self.limits.com_max:
                cost += 2e4
        self._edge_cache[key] = cost
        return cost

    def _condition_reference_yaw(self, reference):
        """Build a dynamically plausible yaw control polygon.

        A clamped cubic spline starts at roughly ``3 * (c1-c0) / dt``.
        Copying a path-tangent yaw jump directly into the first two control
        points therefore violates the rate limit even when the translational
        path is slow.  Limit control-polygon increments before beam search;
        the optimizer may still refine every yaw control afterwards.
        """
        conditioned = np.asarray(reference, dtype=float).copy()
        maximum_delta = 0.8 * self.limits.yaw_rate_max * self.config.dt_max / 3.0
        maximum_delta = max(maximum_delta, np.deg2rad(1.0))
        for index in range(1, len(conditioned)):
            target_delta = wrap_angle(conditioned[index, YAW] - conditioned[index - 1, YAW])
            conditioned[index, YAW] = conditioned[index - 1, YAW] + np.clip(
                target_delta, -maximum_delta, maximum_delta
            )
        return conditioned

    def _condition_shape_rates(self, states):
        conditioned = np.asarray(states, dtype=float).copy()
        # Numerical margin for non-uniform clamped spans and dense validation.
        theta_delta = 0.72 * self.limits.theta_rate_max * self.config.dt_max / 3.0
        phi_delta = 0.72 * self.limits.phi_rate_max * self.config.dt_max / 3.0
        # Backward pass anticipates an upcoming morphology while preserving
        # control 0 as the measured state.
        for index in range(len(conditioned) - 2, 0, -1):
            delta_theta = conditioned[index + 1, THETA] - conditioned[index, THETA]
            conditioned[index, THETA] = conditioned[index + 1, THETA] - np.clip(
                delta_theta, -theta_delta, theta_delta,
            )
            delta_phi = wrap_angle(conditioned[index + 1, PHI] - conditioned[index, PHI])
            conditioned[index, PHI] = conditioned[index + 1, PHI] - np.clip(
                delta_phi, -phi_delta, phi_delta,
            )
        # Forward pass also bounds release and resolves any conflict with the
        # fixed measured start state.
        for index in range(1, len(conditioned)):
            conditioned[index, THETA] = conditioned[index - 1, THETA] + np.clip(
                conditioned[index, THETA] - conditioned[index - 1, THETA],
                -theta_delta, theta_delta,
            )
            delta_phi = wrap_angle(conditioned[index, PHI] - conditioned[index - 1, PHI])
            conditioned[index, PHI] = conditioned[index - 1, PHI] + np.clip(
                delta_phi, -phi_delta, phi_delta,
            )
        return conditioned

    def plan(self, reference_states, current_state=None, terminal_state=None,
             _force_phi_unlocked=False, _force_phi_locked=False):
        # A planner instance is reused across rolling windows, but the local
        # obstacle set can change.  Cache only within one deterministic search.
        self._clearance_cache = {}
        self._edge_cache = {}
        reference = resample_states(reference_states, self.config.knot_count)
        if current_state is not None:
            reference[0] = np.asarray(current_state, dtype=float)
        reference = self._condition_reference_yaw(reference)
        raw_risk_knots = [index for index, state in enumerate(reference) if self._is_risk(state)]
        # A cubic B-spline has four-control-point local support.  Dilate by two
        # controls on both sides so the morphology is already established
        # before the obstacle-influenced span and is not released too early.
        risk_knots = sorted(set(
            neighbor for index in raw_risk_knots
            for neighbor in (index - 2, index - 1, index, index + 1, index + 2)
            if 0 < neighbor < len(reference)
        ))
        if _force_phi_locked and _force_phi_unlocked:
            raise ValueError("phi cannot be forced locked and unlocked simultaneously")
        phi_locked = bool(_force_phi_locked)
        fixed_phi_slack = -float("inf")
        if (not phi_locked and self.config.adaptive_phi and terminal_state is None
                and not _force_phi_unlocked):
            phi_locked, fixed_phi_slack = self._fixed_phi_feasible(
                reference, risk_knots, reference[0, [THETA, PHI]]
            )
        if phi_locked:
            # Keep the bending plane exactly constant throughout this horizon;
            # theta, UAV pose, yaw and timing remain whole-window variables.
            reference[:, PHI] = reference[0, PHI]
        candidates = [
            self._candidates(
                reference, index, reference[0, [THETA, PHI]], index in risk_knots,
                phi_locked=phi_locked,
            )
            for index in range(len(reference))
        ]
        beams = [(0.0, [candidates[0][0]])]
        expanded = 0
        for index in range(1, len(reference)):
            next_beams = []
            for accumulated, path in beams:
                for candidate in candidates[index]:
                    expanded += 1
                    cost = accumulated + self._edge_cost(path[-1], candidate)
                    if self.platform_geometry is not None and not np.isfinite(cost):
                        continue
                    cost += self._node_cost(candidate, reference[index], index in risk_knots)
                    next_beams.append((cost, path + [candidate]))
            next_beams.sort(key=lambda item: item[0])
            selector = getattr(self, 'select_layer', None)
            beams = (selector(next_beams, index) if selector is not None else
                     next_beams[: self.config.seed_beam_width])
        if not beams:
            raise RuntimeError("whole-body seed search produced no candidates")
        best_cost, best_path = beams[0]
        states = np.asarray(best_path, dtype=float)
        if terminal_state is not None:
            # A reach task has a certified whole-body boundary condition.
            # Beam ranking is allowed to choose the interior morphology, but
            # it must not replace the terminal pose/shape with a cheaper grid
            # state (previously 75 deg instead of the required 90 deg).
            states[-1] = np.asarray(terminal_state, dtype=float)
            if phi_locked:
                states[-1, PHI] = reference[0, PHI]
        states = self._condition_shape_rates(states)
        states[:, YAW] = np.unwrap(states[:, YAW])
        states[:, PHI] = np.unwrap(states[:, PHI])
        span_count = self.config.knot_count - 3
        risk_spans = sorted(set(max(0, min(span_count - 1, index - 1)) for index in risk_knots))
        seed_clearance = [self._clearance(state) for state in states]
        edge_clearance = []
        unsafe_edges = []
        for index, (a, b) in enumerate(zip(states[:-1], states[1:])):
            samples = [interpolate_state(a, b, ratio) for ratio in np.linspace(0.0, 1.0, 7)]
            values = [self._clearance(state) for state in samples]
            body_min = min(item[0] for item in values)
            arm_min = min(item[1] for item in values)
            edge_clearance.append((body_min, arm_min))
            if body_min < self.limits.body_margin or arm_min < self.limits.arm_margin:
                unsafe_edges.append(index)
        # The point-wise fixed-plane screen is only a necessary condition.
        # Keep its active set through continuous optimization and dense
        # theta/pose repair.  Only the caller may unlock phi after that full
        # fixed-plane attempt is independently rejected for an arm collision;
        # a sparse seed edge alone is not evidence that a new bending plane is
        # physically necessary.
        edge_body_min = float(min(item[0] for item in edge_clearance))
        edge_arm_min = float(min(item[1] for item in edge_clearance))
        return SeedPlan(
            states=states,
            risk_knots=risk_knots,
            risk_spans=risk_spans,
            diagnostics={
                "seed_cost": float(best_cost), "expanded": expanded,
                "raw_risk_knots": raw_risk_knots,
                "control_body_clearance_min": float(min(item[0] for item in seed_clearance)),
                "control_arm_clearance_min": float(min(item[1] for item in seed_clearance)),
                "control_theta_max_deg": float(np.rad2deg(np.max(states[:, THETA]))),
                "control_z_max": float(np.max(states[:, 2])),
                "edge_body_clearance_min": edge_body_min,
                "edge_arm_clearance_min": edge_arm_min,
                "unsafe_edges": unsafe_edges,
                "shape_mode": "theta_only_fixed_phi" if phi_locked else "theta_phi",
                "fixed_phi_deg": float(np.rad2deg(reference[0, PHI])) if phi_locked else None,
                "fixed_phi_screen_slack": float(fixed_phi_slack),
            },
        )
