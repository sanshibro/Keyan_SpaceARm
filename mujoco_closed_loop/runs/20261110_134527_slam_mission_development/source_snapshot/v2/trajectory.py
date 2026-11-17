import math

import numpy as np

from .dependencies import import_scipy
from .types import PHI, STATE_DIM, TrajectoryData, YAW


scipy = import_scipy()
from scipy.interpolate import BSpline  # noqa: E402


def cumulative_lengths(points):
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return np.zeros(len(points))
    return np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points[:, :3], axis=0), axis=1))))


def resample_states(states, count):
    states = np.asarray(states, dtype=float)
    if len(states) == 0:
        raise ValueError("cannot resample an empty state sequence")
    if len(states) == 1:
        return np.repeat(states, count, axis=0)
    result = states.copy()
    result[:, YAW] = np.unwrap(result[:, YAW])
    result[:, PHI] = np.unwrap(result[:, PHI])
    lengths = cumulative_lengths(result)
    if lengths[-1] < 1e-9:
        source = np.linspace(0.0, 1.0, len(result))
    else:
        source = lengths / lengths[-1]
    target = np.linspace(0.0, 1.0, count)
    return np.column_stack([np.interp(target, source, result[:, dim]) for dim in range(STATE_DIM)])


def cut_path_by_distance(points, maximum_distance):
    points = np.asarray(points, dtype=float)
    if len(points) < 2 or maximum_distance <= 0.0:
        return points.copy()
    out = [points[0].copy()]
    distance = 0.0
    for a, b in zip(points[:-1], points[1:]):
        step = float(np.linalg.norm(b[:3] - a[:3]))
        if step < 1e-9:
            continue
        if distance + step <= maximum_distance + 1e-9:
            out.append(b.copy())
            distance += step
            continue
        ratio = (maximum_distance - distance) / step
        state = (1.0 - ratio) * a + ratio * b
        out.append(state)
        break
    return np.asarray(out)


class SplineTrajectory:
    def __init__(self, control_states, span_durations, degree=3,
                 stationary_endpoints=True, boundary_ramp_duration=1.2,
                 stationary_start=None, stationary_end=None):
        self.control_states = np.asarray(control_states, dtype=float).copy()
        self.span_durations = np.asarray(span_durations, dtype=float)
        self.degree = int(degree)
        if self.control_states.ndim != 2 or self.control_states.shape[1] != STATE_DIM:
            raise ValueError("control_states must have shape (K, 6)")
        expected_spans = len(self.control_states) - self.degree
        if expected_spans < 1 or len(self.span_durations) != expected_spans:
            raise ValueError("duration count must equal K-degree")
        if np.any(self.span_durations <= 0.0):
            raise ValueError("span durations must be positive")
        geometric_duration = float(np.sum(self.span_durations))
        # Preserve the optimized geometric curve and apply one common smooth
        # reference-clock ramp.  A quintic phase-rate profile is zero at both
        # endpoints, never exceeds one, and therefore gives every B-spline
        # method a certified rest-to-rest command without moving arm or UAV
        # geometry into an unvalidated spatial blend.
        default_stationary = bool(stationary_endpoints)
        self.stationary_start = (
            default_stationary if stationary_start is None else bool(stationary_start)
        )
        self.stationary_end = (
            default_stationary if stationary_end is None else bool(stationary_end)
        )
        self.stationary_endpoints = self.stationary_start and self.stationary_end
        ramp = min(
            max(0.0, float(boundary_ramp_duration)), 0.45 * geometric_duration
        )
        self.start_ramp_duration = ramp if self.stationary_start else 0.0
        self.finish_ramp_duration = ramp if self.stationary_end else 0.0
        # Retain the historical public field for callers that only inspect
        # symmetric rest-to-rest trajectories.
        self.boundary_ramp_duration = ramp
        self.geometric_duration = geometric_duration
        self.execution_duration = geometric_duration + 0.5 * (
            self.start_ramp_duration + self.finish_ramp_duration
        )
        end = geometric_duration
        interior = np.cumsum(self.span_durations)[:-1]
        self.knots = np.concatenate((
            np.zeros(self.degree + 1),
            interior,
            np.full(self.degree + 1, end),
        ))
        self._splines = [
            BSpline(self.knots, self.control_states[:, dim], self.degree, extrapolate=False)
            for dim in range(STATE_DIM)
        ]

    @property
    def duration(self):
        return float(self.execution_duration)

    @property
    def span_boundaries(self):
        geometric = np.concatenate(([0.0], np.cumsum(self.span_durations)))
        return np.asarray([self._execution_time(value) for value in geometric])

    @staticmethod
    def _smooth_rate(u):
        return 10.0 * u ** 3 - 15.0 * u ** 4 + 6.0 * u ** 5

    @staticmethod
    def _smooth_rate_derivative(u):
        return 30.0 * u ** 2 - 60.0 * u ** 3 + 30.0 * u ** 4

    @staticmethod
    def _smooth_rate_second_derivative(u):
        return 60.0 * u - 180.0 * u ** 2 + 120.0 * u ** 3

    @staticmethod
    def _smooth_rate_integral(u):
        return 2.5 * u ** 4 - 3.0 * u ** 5 + u ** 6

    def _phase(self, times):
        """Map execution time to spline time and its first three derivatives."""
        times = np.clip(np.asarray(times, dtype=float), 0.0, self.duration)
        start_ramp = self.start_ramp_duration
        finish_ramp = self.finish_ramp_duration
        if start_ramp <= 1e-12 and finish_ramp <= 1e-12:
            return times, np.ones_like(times), np.zeros_like(times), np.zeros_like(times)
        phase = np.empty_like(times)
        rate = np.ones_like(times)
        acceleration = np.zeros_like(times)
        jerk = np.zeros_like(times)
        start = times < start_ramp if start_ramp > 1e-12 else np.zeros_like(times, dtype=bool)
        finish = (
            times > self.duration - finish_ramp
            if finish_ramp > 1e-12 else np.zeros_like(times, dtype=bool)
        )
        middle = ~(start | finish)
        if np.any(start):
            u = times[start] / start_ramp
            phase[start] = start_ramp * self._smooth_rate_integral(u)
            rate[start] = self._smooth_rate(u)
            acceleration[start] = self._smooth_rate_derivative(u) / start_ramp
            jerk[start] = self._smooth_rate_second_derivative(u) / start_ramp ** 2
        if np.any(middle):
            phase[middle] = times[middle] - 0.5 * start_ramp
        if np.any(finish):
            u = (self.duration - times[finish]) / finish_ramp
            phase[finish] = self.geometric_duration - finish_ramp * self._smooth_rate_integral(u)
            rate[finish] = self._smooth_rate(u)
            acceleration[finish] = -self._smooth_rate_derivative(u) / finish_ramp
            jerk[finish] = self._smooth_rate_second_derivative(u) / finish_ramp ** 2
        return phase, rate, acceleration, jerk

    def _execution_time(self, phase):
        """Invert the monotone phase map for geometric knot boundaries."""
        phase = min(self.geometric_duration, max(0.0, float(phase)))
        start_ramp = self.start_ramp_duration
        finish_ramp = self.finish_ramp_duration
        if start_ramp <= 1e-12 and finish_ramp <= 1e-12:
            return phase
        if start_ramp > 1e-12 and phase < 0.5 * start_ramp:
            target, reverse, ramp = phase / start_ramp, False, start_ramp
        elif (finish_ramp > 1e-12
              and phase > self.geometric_duration - 0.5 * finish_ramp):
            target = (self.geometric_duration - phase) / finish_ramp
            ramp = finish_ramp
            reverse = True
        else:
            return phase + 0.5 * start_ramp
        lower, upper = 0.0, 1.0
        for _ in range(40):
            midpoint = 0.5 * (lower + upper)
            if self._smooth_rate_integral(midpoint) < target:
                lower = midpoint
            else:
                upper = midpoint
        local = ramp * 0.5 * (lower + upper)
        return self.duration - local if reverse else local

    def evaluate(self, times, derivative=0):
        times = np.atleast_1d(np.asarray(times, dtype=float))
        times = np.clip(times, 0.0, self.duration)
        if derivative > self.degree:
            return np.zeros((len(times), STATE_DIM), dtype=float)
        phase, rate, phase_acceleration, phase_jerk = self._phase(times)
        base = lambda order: np.column_stack([
            np.asarray(spline.derivative(order)(phase), dtype=float)
            for spline in self._splines
        ])
        if derivative == 0:
            return base(0)
        first = base(1)
        if derivative == 1:
            return first * rate[:, None]
        second = base(2)
        if derivative == 2:
            return second * rate[:, None] ** 2 + first * phase_acceleration[:, None]
        third = base(3)
        return (
            third * rate[:, None] ** 3
            + 3.0 * second * rate[:, None] * phase_acceleration[:, None]
            + first * phase_jerk[:, None]
        )

    def sample(self, dt=0.05, source="spline", objective=float("inf"), metadata=None):
        count = max(2, int(math.ceil(self.duration / max(dt, 1e-3))) + 1)
        times = np.linspace(0.0, self.duration, count)
        return TrajectoryData(
            times=times,
            states=self.evaluate(times, 0),
            velocities=self.evaluate(times, 1),
            accelerations=self.evaluate(times, 2),
            jerks=self.evaluate(times, 3),
            source=source,
            objective=objective,
            metadata=metadata or {},
        )

    def collocation_times(self, normal_per_span=3, risk_per_span=7, risk_spans=None):
        risk_spans = set(risk_spans or [])
        times = [0.0]
        starts = self.span_boundaries
        for index, duration in enumerate(self.span_durations):
            count = risk_per_span if index in risk_spans else normal_per_span
            local = np.linspace(starts[index], starts[index + 1], max(2, count + 1))[1:]
            times.extend(local.tolist())
        return np.unique(np.asarray(times, dtype=float))


def initial_span_durations(control_states, degree, nominal_speed, dt_min, dt_max):
    controls = np.asarray(control_states, dtype=float)
    span_count = len(controls) - degree
    if span_count <= 0:
        raise ValueError("not enough control states")
    total_length = cumulative_lengths(controls)[-1]
    total_time = max(total_length / max(nominal_speed, 1e-3), dt_min * span_count)
    adjacent = np.linalg.norm(np.diff(controls[:, :3], axis=0), axis=1)
    endpoint_safe = degree * float(np.max(adjacent)) / max(nominal_speed, 1e-3)
    uniform = np.clip(max(total_time / span_count, endpoint_safe), dt_min, dt_max)
    return np.full(span_count, uniform, dtype=float)
