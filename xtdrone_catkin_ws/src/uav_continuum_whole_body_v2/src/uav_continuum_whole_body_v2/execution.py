"""Pure helpers shared by the ROS whole-body executor and its unit tests."""

import bisect
import math

import numpy as np


def wrap_angle(value):
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quaternion(quaternion):
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
    )


def quaternion_from_yaw(yaw, quaternion):
    quaternion.x = 0.0
    quaternion.y = 0.0
    quaternion.z = math.sin(0.5 * yaw)
    quaternion.w = math.cos(0.5 * yaw)


def interpolation_bracket(times, elapsed):
    """Return (lower, upper, ratio), including well-defined endpoint holds."""
    if len(times) == 0:
        raise ValueError("empty trajectory time array")
    if elapsed <= times[0]:
        return 0, 0, 0.0
    if elapsed >= times[-1]:
        last = len(times) - 1
        return last, last, 0.0
    upper = bisect.bisect_right(times, elapsed)
    lower = upper - 1
    span = float(times[upper] - times[lower])
    ratio = 0.0 if span <= 0.0 else (float(elapsed) - times[lower]) / span
    return lower, upper, max(0.0, min(1.0, ratio))


def lerp(first, second, ratio):
    return float(first) + float(ratio) * (float(second) - float(first))


def lerp_angle(first, second, ratio):
    return wrap_angle(float(first) + float(ratio) * wrap_angle(float(second) - float(first)))


def position_distance(first, second):
    dx = float(first.x) - float(second.x)
    dy = float(first.y) - float(second.y)
    dz = float(first.z) - float(second.z)
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def largest_feasible_ratio(is_feasible, iterations=12):
    """Return the largest interpolation ratio accepted by a monotone guard."""
    if not is_feasible(0.0):
        return 0.0
    if is_feasible(1.0):
        return 1.0
    lower, upper = 0.0, 1.0
    for _ in range(max(1, int(iterations))):
        middle = 0.5 * (lower + upper)
        if is_feasible(middle):
            lower = middle
        else:
            upper = middle
    return lower


def tracking_progress_scale(error, full_speed_error, stop_error):
    """Continuous phase rate for a bounded whole-body tracking tube.

    The returned value is one inside the full-speed tube, zero at and beyond
    the stop boundary, and linear in between.  Keeping this helper pure makes
    the executor's safety-critical time law independently testable.
    """
    error = max(0.0, float(error))
    full_speed_error = max(0.0, float(full_speed_error))
    stop_error = float(stop_error)
    if stop_error <= full_speed_error:
        raise ValueError("stop_error must exceed full_speed_error")
    if error <= full_speed_error:
        return 1.0
    if error >= stop_error:
        return 0.0
    return (stop_error - error) / (stop_error - full_speed_error)


def clearance_tracking_budget(clearance, hard_limit, stopping_reserve, maximum):
    """Maximum geometric tracking error preserving a clearance hard line."""
    return max(0.0, min(
        float(maximum),
        float(clearance) - float(hard_limit) - float(stopping_reserve),
    ))


def clearance_progress_scale(error, preferred_full_error, budget,
                             calibrated_full_speed_error=None):
    """Phase scale for a clearance-contracted tube, including very small tubes."""
    budget = max(0.0, float(budget))
    if budget <= 0.0:
        return 0.0
    adaptive_full = min(max(0.0, float(preferred_full_error)), 0.25 * budget)
    scale = tracking_progress_scale(error, adaptive_full, budget)
    if calibrated_full_speed_error is not None:
        calibration = max(1e-6, float(calibrated_full_speed_error))
        scale = min(scale, budget / calibration)
    return max(0.0, min(1.0, scale))


def projected_phase_elapsed(observed_elapsed, observation_age, phase_rate, duration):
    """Project the executor-owned clock without assuming wall-time speed.

    The guarded executor may pause trajectory phase in a narrow passage.
    Rolling replanning must extrapolate its reported phase rate; extrapolating
    wall time would place a successor ahead of the state actually reached.
    """
    elapsed = max(0.0, float(observed_elapsed))
    age = max(0.0, float(observation_age))
    rate = max(0.0, min(1.0, float(phase_rate)))
    return max(0.0, min(float(duration), elapsed + age * rate))


def trajectory_translation_progress(times, states, elapsed):
    """Arc length executed up to a trajectory phase, using sampled XYZ states.

    Rolling-window commitment must be measured in geometric progress, not
    wall time: the safety phase governor is allowed to slow or pause inside a
    constrained passage.  This helper intentionally ignores yaw and arm
    motion because ``commit_horizon`` is specified in metres.
    """
    if len(times) != len(states):
        raise ValueError("trajectory times and states must have equal length")
    lower, upper, ratio = interpolation_bracket(times, elapsed)

    def distance(first, second):
        return math.sqrt(sum(
            (float(first[dimension]) - float(second[dimension])) ** 2
            for dimension in range(3)
        ))

    progress = sum(
        distance(states[index], states[index - 1])
        for index in range(1, lower + 1)
    )
    if upper != lower:
        progress += ratio * distance(states[upper], states[lower])
    return float(progress)


def bounded_switch_match(times, positions, targets, maximum_advance):
    """Find a certified successor phase with bounded geometric advancement.

    Rolling optimization finishes after the active vehicle has moved.  Starting
    every successor at sample zero can therefore command a backward jump.  A
    suffix of an independently validated trajectory is still validated, so the
    executor may splice at an interior sample.  The search is deliberately
    limited by XYZ arc length to prevent a self-intersecting path from skipping
    an obstacle or most of the local horizon.

    The minimax score keeps the selected sample close to every supplied target;
    the executor supplies both its previous command and the measured position.
    Returns ``(elapsed, index, advance, gaps)``.
    """
    if len(times) == 0 or len(times) != len(positions):
        raise ValueError("non-empty times and positions must have equal length")
    if not targets:
        raise ValueError("at least one switch target is required")
    maximum_advance = max(0.0, float(maximum_advance))

    def distance(first, second):
        return math.sqrt(sum(
            (float(first[dimension]) - float(second[dimension])) ** 2
            for dimension in range(3)
        ))

    best = None
    advance = 0.0
    for index, position in enumerate(positions):
        if index:
            advance += distance(position, positions[index - 1])
        if advance > maximum_advance + 1e-12:
            break
        gaps = tuple(distance(position, target) for target in targets)
        candidate = (max(gaps), index, float(advance), gaps)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:  # Defensive; index zero always has zero advancement.
        raise ValueError("no switch candidate inside maximum_advance")
    _score, index, advance, gaps = best
    return float(times[index]), int(index), float(advance), gaps


def reference_converged(final, governor_alpha, tracking_error, tolerance):
    """Return whether the current reference satisfies completion semantics.

    A moving reference must remain synchronized with the phase governor.  At
    the endpoint there is no phase left to advance, so requiring unit phase
    rate would deadlock on small steady-state controller error.  Terminal
    task accuracy remains independently checked at the measured tool tip.
    """
    return (
        float(tracking_error) <= float(tolerance)
        and (bool(final) or float(governor_alpha) >= 0.999)
    )


def continuous_switch_phase_rate(previous_rate, scheduled_restart=False):
    """Keep an immediate rolling splice continuous in reference-clock rate."""
    if scheduled_restart:
        return 0.0
    return min(1.0, max(0.0, float(previous_rate)))


def bending_vector(theta, phi):
    """Return the nonsingular planar bending vector for radian angles."""
    theta = float(theta)
    phi = float(phi)
    return np.asarray([theta * math.cos(phi), theta * math.sin(phi)], dtype=float)


def bending_vector_rate(theta, phi, theta_rate, phi_rate):
    """Return the time derivative of ``bending_vector``."""
    theta = float(theta)
    phi = float(phi)
    theta_rate = float(theta_rate)
    phi_rate = float(phi_rate)
    return np.asarray([
        theta_rate * math.cos(phi) - theta * math.sin(phi) * phi_rate,
        theta_rate * math.sin(phi) + theta * math.cos(phi) * phi_rate,
    ], dtype=float)


def bending_state_from_vector(vector, vector_rate=None):
    """Recover nonsingular ``theta, phi`` and optional rates from ``b``."""
    vector = np.asarray(vector, dtype=float).reshape(-1)
    if len(vector) != 2 or not np.all(np.isfinite(vector)):
        raise ValueError("bending vector must contain two finite values")
    theta = float(np.linalg.norm(vector))
    phi = math.atan2(float(vector[1]), float(vector[0])) if theta > 1e-10 else 0.0
    if vector_rate is None:
        return theta, phi
    vector_rate = np.asarray(vector_rate, dtype=float).reshape(-1)
    if len(vector_rate) != 2 or not np.all(np.isfinite(vector_rate)):
        raise ValueError("bending-vector rate must contain two finite values")
    if theta <= 1e-10:
        theta_rate = float(np.linalg.norm(vector_rate))
        phi_rate = 0.0
    else:
        theta_rate = float(np.dot(vector, vector_rate) / theta)
        phi_rate = float(
            (vector[0] * vector_rate[1] - vector[1] * vector_rate[0])
            / (theta * theta)
        )
    return theta, phi, theta_rate, phi_rate


def quintic_bridge_coefficients(start, start_rate, start_acceleration,
                                finish, finish_rate, finish_acceleration,
                                duration):
    """Return C2 quintic coefficients for vector boundary conditions."""
    values = [
        np.asarray(item, dtype=float).reshape(-1) for item in (
            start, start_rate, start_acceleration,
            finish, finish_rate, finish_acceleration,
        )
    ]
    if len({len(item) for item in values}) != 1:
        raise ValueError("quintic bridge boundary vectors must share one dimension")
    if not all(np.all(np.isfinite(item)) for item in values):
        raise ValueError("quintic bridge boundaries must be finite")
    duration = float(duration)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("quintic bridge duration must be positive")
    start, start_rate, start_acceleration, finish, finish_rate, finish_acceleration = values
    coefficients = np.zeros((6, len(start)), dtype=float)
    coefficients[0] = start
    coefficients[1] = start_rate
    coefficients[2] = 0.5 * start_acceleration
    t = duration
    matrix = np.asarray([
        [t ** 3, t ** 4, t ** 5],
        [3.0 * t ** 2, 4.0 * t ** 3, 5.0 * t ** 4],
        [6.0 * t, 12.0 * t ** 2, 20.0 * t ** 3],
    ], dtype=float)
    residual = np.vstack((
        finish - (coefficients[0] + coefficients[1] * t + coefficients[2] * t ** 2),
        finish_rate - (coefficients[1] + 2.0 * coefficients[2] * t),
        finish_acceleration - 2.0 * coefficients[2],
    ))
    coefficients[3:] = np.linalg.solve(matrix, residual)
    return coefficients


def evaluate_quintic_bridge(coefficients, times, derivative=0):
    """Evaluate a vector quintic bridge and its first three derivatives."""
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.ndim != 2 or coefficients.shape[0] != 6:
        raise ValueError("quintic coefficients must have shape (6,D)")
    derivative = int(derivative)
    if derivative < 0 or derivative > 3:
        raise ValueError("supported quintic bridge derivatives are zero through three")
    times = np.atleast_1d(np.asarray(times, dtype=float))
    if not np.all(np.isfinite(times)):
        raise ValueError("quintic bridge times must be finite")
    result = np.zeros((len(times), coefficients.shape[1]), dtype=float)
    for power in range(derivative, 6):
        factor = math.factorial(power) / math.factorial(power - derivative)
        result += factor * coefficients[power] * times[:, None] ** (power - derivative)
    return result


def whole_body_switch_deltas(old_state, old_rate, new_state, new_rate):
    """Measure C0/C1 command gaps using a nonsingular morphology state.

    States are ``[x, y, z, yaw, theta, phi]`` and rates use the matching
    derivatives, all angular quantities in radians.  Reporting the bending
    vector avoids a false 180-degree phi discontinuity when theta is zero.
    """
    old_state = np.asarray(old_state, dtype=float).reshape(-1)
    new_state = np.asarray(new_state, dtype=float).reshape(-1)
    old_rate = np.asarray(old_rate, dtype=float).reshape(-1)
    new_rate = np.asarray(new_rate, dtype=float).reshape(-1)
    if any(len(value) != 6 for value in (old_state, new_state, old_rate, new_rate)):
        raise ValueError("whole-body switch states and rates must have length six")
    if not all(np.all(np.isfinite(value)) for value in (
            old_state, new_state, old_rate, new_rate)):
        raise ValueError("whole-body switch states and rates must be finite")
    old_bending = bending_vector(old_state[4], old_state[5])
    new_bending = bending_vector(new_state[4], new_state[5])
    old_bending_rate = bending_vector_rate(
        old_state[4], old_state[5], old_rate[4], old_rate[5]
    )
    new_bending_rate = bending_vector_rate(
        new_state[4], new_state[5], new_rate[4], new_rate[5]
    )
    return {
        "position_gap_m": float(np.linalg.norm(new_state[:3] - old_state[:3])),
        "yaw_gap_rad": abs(float(wrap_angle(new_state[3] - old_state[3]))),
        "bending_gap_rad": float(np.linalg.norm(new_bending - old_bending)),
        "linear_velocity_gap_m_s": float(np.linalg.norm(new_rate[:3] - old_rate[:3])),
        "yaw_rate_gap_rad_s": abs(float(new_rate[3] - old_rate[3])),
        "bending_rate_gap_rad_s": float(np.linalg.norm(
            new_bending_rate - old_bending_rate
        )),
        "old_bending_vector_rad": old_bending.tolist(),
        "new_bending_vector_rad": new_bending.tolist(),
        "old_bending_rate_rad_s": old_bending_rate.tolist(),
        "new_bending_rate_rad_s": new_bending_rate.tolist(),
    }


def whole_body_switch_gate(metrics, limits):
    """Return explicit C0/C1 threshold violations for a rolling splice."""
    required = [
        "position_gap_m", "yaw_gap_rad", "bending_gap_rad",
        "linear_velocity_gap_m_s", "yaw_rate_gap_rad_s",
        "bending_rate_gap_rad_s",
    ]
    if "linear_acceleration_gap_m_s2" in limits:
        required.append("linear_acceleration_gap_m_s2")
    violations = {}
    for name in required:
        if name not in metrics or name not in limits:
            raise KeyError("missing whole-body switch gate field: %s" % name)
        value = float(metrics[name])
        limit = float(limits[name])
        if not np.isfinite(value) or not np.isfinite(limit) or limit <= 0.0:
            raise ValueError("whole-body switch gate values must be finite and positive")
        if value > limit + 1e-12:
            violations[name] = {"value": value, "limit": limit}
    return violations


def bounded_whole_body_switch_match(
        times, states, rates, old_state, old_rate, measured_position,
        maximum_advance, scales=None, accelerations=None,
        old_acceleration=None):
    """Select a bounded successor phase using normalized C0/C1/C2 gaps.

    Unlike ``bounded_switch_match``, this selector cannot prefer a zero-speed
    window endpoint merely because its XYZ position is marginally closer.  The
    scales normalize heterogeneous units for a minimax objective; they are not
    hard safety limits.  Existing spatial handoff gates remain authoritative.
    """
    times = np.asarray(times, dtype=float).reshape(-1)
    states = np.asarray(states, dtype=float)
    rates = np.asarray(rates, dtype=float)
    old_state = np.asarray(old_state, dtype=float).reshape(-1)
    old_rate = np.asarray(old_rate, dtype=float).reshape(-1)
    measured_position = np.asarray(measured_position, dtype=float).reshape(-1)
    use_acceleration = accelerations is not None or old_acceleration is not None
    if use_acceleration:
        if accelerations is None or old_acceleration is None:
            raise ValueError("candidate and old accelerations must be supplied together")
        accelerations = np.asarray(accelerations, dtype=float)
        old_acceleration = np.asarray(old_acceleration, dtype=float).reshape(-1)
    if len(times) == 0 or states.shape != (len(times), 6) or rates.shape != (len(times), 6):
        raise ValueError("times, states and rates must have shapes (N,), (N,6), (N,6)")
    if len(old_state) != 6 or len(old_rate) != 6 or len(measured_position) != 3:
        raise ValueError("invalid whole-body switch target dimensions")
    if use_acceleration and (
            accelerations.shape != (len(times), 3) or len(old_acceleration) != 3):
        raise ValueError("accelerations must have shapes (N,3) and (3,)")
    if not (np.all(np.isfinite(times)) and np.all(np.isfinite(states))
            and np.all(np.isfinite(rates)) and np.all(np.isfinite(old_state))
            and np.all(np.isfinite(old_rate)) and np.all(np.isfinite(measured_position))):
        raise ValueError("whole-body switch inputs must be finite")
    if use_acceleration and not (
            np.all(np.isfinite(accelerations))
            and np.all(np.isfinite(old_acceleration))):
        raise ValueError("whole-body switch accelerations must be finite")
    if np.any(np.diff(times) < 0.0):
        raise ValueError("switch candidate times must be monotonic")
    maximum_advance = max(0.0, float(maximum_advance))
    default_scales = {
        "position_gap_m": 0.10,
        "measured_position_gap_m": 0.10,
        "yaw_gap_rad": math.radians(10.0),
        "bending_gap_rad": math.radians(10.0),
        "linear_velocity_gap_m_s": 0.05,
        "yaw_rate_gap_rad_s": math.radians(10.0),
        "bending_rate_gap_rad_s": math.radians(10.0),
        "linear_acceleration_gap_m_s2": 0.15,
    }
    if scales is not None:
        default_scales.update({key: float(value) for key, value in scales.items()})
    if any(value <= 0.0 or not math.isfinite(value) for value in default_scales.values()):
        raise ValueError("whole-body switch normalization scales must be positive")

    best = None
    advance = 0.0
    for index in range(len(times)):
        if index:
            advance += float(np.linalg.norm(states[index, :3] - states[index - 1, :3]))
        if advance > maximum_advance + 1e-12:
            break
        deltas = whole_body_switch_deltas(
            old_state, old_rate, states[index], rates[index]
        )
        if use_acceleration:
            acceleration_gap = float(np.linalg.norm(
                accelerations[index] - old_acceleration
            ))
            deltas.update({
                "linear_acceleration_gap_m_s2": acceleration_gap,
                "old_linear_acceleration_m_s2": old_acceleration.tolist(),
                "new_linear_acceleration_m_s2": accelerations[index].tolist(),
            })
        measured_gap = float(np.linalg.norm(states[index, :3] - measured_position))
        normalized = [
            deltas["position_gap_m"] / default_scales["position_gap_m"],
            measured_gap / default_scales["measured_position_gap_m"],
            deltas["yaw_gap_rad"] / default_scales["yaw_gap_rad"],
            deltas["bending_gap_rad"] / default_scales["bending_gap_rad"],
            deltas["linear_velocity_gap_m_s"]
            / default_scales["linear_velocity_gap_m_s"],
            deltas["yaw_rate_gap_rad_s"] / default_scales["yaw_rate_gap_rad_s"],
            deltas["bending_rate_gap_rad_s"]
            / default_scales["bending_rate_gap_rad_s"],
        ]
        if use_acceleration:
            normalized.append(
                deltas["linear_acceleration_gap_m_s2"]
                / default_scales["linear_acceleration_gap_m_s2"]
            )
        score = max(normalized)
        candidate = (
            score,
            max(deltas["position_gap_m"], measured_gap),
            index,
            float(advance),
            measured_gap,
            deltas,
        )
        if best is None or candidate[:3] < best[:3]:
            best = candidate
    if best is None:
        raise ValueError("no whole-body switch candidate inside maximum_advance")
    score, _position_score, index, advance, measured_gap, deltas = best
    return (
        float(times[index]), int(index), float(advance),
        float(deltas["position_gap_m"]), float(measured_gap),
        float(score), deltas,
    )


def stationary_endpoint_feedforward(final, velocity, yaw_rate):
    """Return zero feed-forward once a stationary terminal state is reached."""
    values = tuple(float(value) for value in velocity)
    if len(values) != 3:
        raise ValueError("velocity must contain exactly three components")
    if final:
        return (0.0, 0.0, 0.0), 0.0
    return values, float(yaw_rate)
