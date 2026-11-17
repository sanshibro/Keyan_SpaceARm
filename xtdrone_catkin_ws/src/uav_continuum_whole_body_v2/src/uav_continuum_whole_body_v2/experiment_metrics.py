"""Method-independent metrics used by the comparison experiment."""

import numpy as np

from .dependencies import import_scipy


scipy = import_scipy()
from scipy.signal import savgol_filter  # noqa: E402


def executed_translation_jerk_profile(
        stamps, linear_velocities, resample_hz=20.0, smoothing_window_s=0.5,
        polynomial_order=3):
    """Return the frozen executed-jerk profile on a uniform analysis grid.

    ROS callback timing is not a suitable differentiation grid: millisecond-scale
    timestamp jitter is amplified twice and can dominate the physical signal.  The
    comparison protocol therefore sorts and de-duplicates samples, interpolates
    every method onto the same grid, then applies the same Savitzky-Golay
    differentiator.  No planner-specific information is used.  The returned
    dictionary exposes the same profile used by the scalar paper metric so that
    start, rolling-switch, interior and stop contributions can be audited without
    changing the metric definition.
    """
    times = np.asarray(stamps, dtype=float).reshape(-1)
    velocities = np.asarray(linear_velocities, dtype=float)
    if velocities.ndim != 2 or velocities.shape != (len(times), 3):
        raise ValueError("linear_velocities must have shape (N, 3)")
    if resample_hz <= 0.0 or smoothing_window_s <= 0.0:
        raise ValueError("resample_hz and smoothing_window_s must be positive")
    finite = np.isfinite(times) & np.all(np.isfinite(velocities), axis=1)
    times = times[finite]
    velocities = velocities[finite]
    if len(times) < polynomial_order + 2:
        return None
    order = np.argsort(times, kind="stable")
    times = times[order]
    velocities = velocities[order]
    unique = np.r_[True, np.diff(times) > 1e-6]
    times = times[unique]
    velocities = velocities[unique]
    if len(times) < polynomial_order + 2:
        return None

    dt = 1.0 / float(resample_hz)
    duration = float(times[-1] - times[0])
    if duration < max(smoothing_window_s, (polynomial_order + 1) * dt):
        return None
    sample_count = int(np.floor(duration / dt)) + 1
    uniform_times = times[0] + np.arange(sample_count, dtype=float) * dt
    uniform_velocities = np.column_stack([
        np.interp(uniform_times, times, velocities[:, axis])
        for axis in range(3)
    ])

    requested_points = int(round(smoothing_window_s * resample_hz))
    window_points = requested_points if requested_points % 2 else requested_points + 1
    window_points = max(window_points, polynomial_order + 2)
    if window_points % 2 == 0:
        window_points += 1
    if window_points > sample_count:
        window_points = sample_count if sample_count % 2 else sample_count - 1
    if window_points <= polynomial_order:
        return None

    jerk = savgol_filter(
        uniform_velocities, window_length=window_points,
        polyorder=polynomial_order, deriv=2, delta=dt, axis=0, mode="interp",
    )
    density = np.sum(jerk ** 2, axis=1)
    return {
        "times": uniform_times,
        "velocities": uniform_velocities,
        "jerks": jerk,
        "energy_density": density,
        "energy": float(np.trapz(density, uniform_times)),
    }


def executed_translation_jerk_energy(
        stamps, linear_velocities, resample_hz=20.0, smoothing_window_s=0.5,
        polynomial_order=3):
    """Return integral ||d2 velocity / dt2||^2 on a fixed analysis grid."""
    profile = executed_translation_jerk_profile(
        stamps, linear_velocities, resample_hz=resample_hz,
        smoothing_window_s=smoothing_window_s,
        polynomial_order=polynomial_order,
    )
    return None if profile is None else float(profile["energy"])


def partition_executed_translation_jerk(
        stamps, linear_velocities, start_stamp, stop_stamp, switch_stamps=(),
        endpoint_radius_s=2.0, switch_radius_s=1.0, resample_hz=20.0,
        smoothing_window_s=0.5, polynomial_order=3):
    """Partition the unchanged jerk integral into disjoint event neighborhoods.

    Each integration interval is assigned using its midpoint.  Start/stop regions
    take precedence over rolling-switch regions, preventing overlap from double
    counting energy.  The components therefore sum numerically to ``total``.
    Neighborhood radii are explicit analysis parameters and should be accompanied
    by a sensitivity report; they are not planner-specific tuning knobs.
    """
    endpoint_radius_s = float(endpoint_radius_s)
    switch_radius_s = float(switch_radius_s)
    if endpoint_radius_s < 0.0 or switch_radius_s < 0.0:
        raise ValueError("jerk partition radii must be non-negative")
    start_stamp = float(start_stamp)
    stop_stamp = float(stop_stamp)
    if stop_stamp < start_stamp:
        raise ValueError("stop_stamp must not precede start_stamp")
    switches = np.asarray(tuple(float(value) for value in switch_stamps), dtype=float)
    profile = executed_translation_jerk_profile(
        stamps, linear_velocities, resample_hz=resample_hz,
        smoothing_window_s=smoothing_window_s,
        polynomial_order=polynomial_order,
    )
    if profile is None:
        return None

    times = profile["times"]
    density = profile["energy_density"]
    midpoints = 0.5 * (times[:-1] + times[1:])
    interval_energy = 0.5 * (density[:-1] + density[1:]) * np.diff(times)
    labels = np.full(len(midpoints), "interior", dtype=object)
    labels[np.abs(midpoints - start_stamp) <= endpoint_radius_s] = "start"
    labels[np.abs(midpoints - stop_stamp) <= endpoint_radius_s] = "stop"
    if len(switches):
        all_switch_distances = np.abs(midpoints[:, None] - switches[None, :])
        nearest_switch = np.argmin(all_switch_distances, axis=1)
        switch_distance = np.min(all_switch_distances, axis=1)
        switch_mask = switch_distance <= switch_radius_s
        labels[(labels == "interior") & switch_mask] = "switch"
    else:
        nearest_switch = np.zeros(len(midpoints), dtype=int)

    components = {
        name: float(np.sum(interval_energy[labels == name]))
        for name in ("start", "interior", "switch", "stop")
    }
    total = float(np.sum(interval_energy))
    fractions = {
        name: (value / total if total > 0.0 else 0.0)
        for name, value in components.items()
    }
    per_switch = []
    for index, stamp in enumerate(switches):
        mask = (labels == "switch") & (nearest_switch == index)
        energy = float(np.sum(interval_energy[mask]))
        per_switch.append({
            "index": int(index), "stamp": float(stamp), "energy": energy,
            "fraction_of_total": energy / total if total > 0.0 else 0.0,
        })
    return {
        "total": total,
        "components": components,
        "fractions": fractions,
        "start_stamp": start_stamp,
        "stop_stamp": stop_stamp,
        "switch_stamps": switches.tolist(),
        "endpoint_radius_s": endpoint_radius_s,
        "switch_radius_s": switch_radius_s,
        "per_switch": per_switch,
        "profile": profile,
    }
