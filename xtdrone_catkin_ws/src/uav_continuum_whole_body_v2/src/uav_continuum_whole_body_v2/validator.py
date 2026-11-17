import math

import numpy as np

from .geometry import (
    coupling_metrics_batch, world_arm_points, world_arm_points_batch,
    world_body_points_batch,
)
from .types import PHI, THETA, ValidationResult, YAW


class SweptVolumeValidator:
    """Independent dense validator; it never reuses optimizer constraints as proof."""

    def __init__(self, world, arm_model, mass_model, limits, config, body_world=None,
                 center_keepout_world=None, terminal_target=None, terminal_tolerance=0.05):
        self.world = world
        self.body_world = body_world or world
        self.center_keepout_world = center_keepout_world
        self.terminal_target = None if terminal_target is None else np.asarray(terminal_target, dtype=float)
        self.terminal_tolerance = float(terminal_tolerance)
        self.arm = arm_model
        self.mass = mass_model
        self.limits = limits
        self.config = config

    def _adaptive_times(self, spline):
        coarse = np.unique(np.concatenate((
            spline.span_boundaries,
            np.arange(0.0, spline.duration, self.config.swept_time_step),
            [spline.duration],
        )))
        states = spline.evaluate(coarse)
        times = [float(coarse[0])]
        for index in range(len(coarse) - 1):
            a, b = states[index], states[index + 1]
            counts = [
                np.linalg.norm(b[:3] - a[:3]) / max(self.config.swept_translation_step, 1e-4),
                abs(b[THETA] - a[THETA]) / max(self.config.swept_theta_step, 1e-4),
                abs(b[PHI] - a[PHI]) / max(self.config.swept_phi_step, 1e-4),
            ]
            count = max(1, int(math.ceil(max(counts))))
            times.extend(np.linspace(coarse[index], coarse[index + 1], count + 1)[1:].tolist())
        return np.unique(np.asarray(times, dtype=float))

    def validate(self, spline, stop_on_failure=False):
        times = self._adaptive_times(spline)
        state = spline.evaluate(times, 0)
        velocity = spline.evaluate(times, 1)
        acceleration = spline.evaluate(times, 2)
        jerk = spline.evaluate(times, 3)
        metrics = {
            "duration": float(spline.duration), "validation_samples": float(len(times)),
            "body_clearance_min": float("inf"), "arm_clearance_min": float("inf"),
            "com_offset_max": 0.0, "gravity_torque_max": 0.0,
            "velocity_max": 0.0, "acceleration_max": 0.0, "jerk_max": 0.0,
            "yaw_rate_max": 0.0, "theta_rate_max": 0.0, "phi_rate_max": 0.0,
            "tilt_max": 0.0, "thrust_ratio_min": float("inf"), "thrust_ratio_max": 0.0,
            "body_obstacle_min": "none", "arm_obstacle_min": "none",
            "terminal_tip_error": -1.0,
        }
        failure = None
        gravity = np.array([0.0, 0.0, 9.81])
        # Collision proof remains independent and uses the same dense samples,
        # but evaluates all samples in grouped NumPy/KD-tree batches instead
        # of issuing hundreds of small Python queries.
        body_groups = world_body_points_batch(
            state, self.limits, azimuth_samples=32, vertical_samples=5
        )
        arm_groups = world_arm_points_batch(state, self.arm)
        body_distances, body_names = self.body_world.minimum_distances_grouped(body_groups)
        arm_distances, arm_names = self.world.minimum_distances_grouped(arm_groups)
        com_offsets = coupling_metrics_batch(state, self.arm, self.mass)
        for index, current in enumerate(state):
            body = float(body_distances[index])
            body_name = body_names[index]
            if self.center_keepout_world is not None:
                keepout, keepout_name = self.center_keepout_world.signed_distance(current[:3])
                keepout_clearance = keepout + self.limits.body_margin
                if keepout_clearance < body:
                    body = keepout_clearance
                    body_name = "center_keepout:" + keepout_name
            arm = float(arm_distances[index]) - self.arm.radius
            com_offset = float(com_offsets[index])
            gravity_torque = self.mass.total * 9.81 * com_offset
            speed = float(np.linalg.norm(velocity[index, :3]))
            accel = float(np.linalg.norm(acceleration[index, :3]))
            jerk_norm = float(np.linalg.norm(jerk[index, :3]))
            thrust_vector = acceleration[index, :3] + gravity
            thrust_ratio = float(np.linalg.norm(thrust_vector) / 9.81)
            tilt = float(math.atan2(np.linalg.norm(thrust_vector[:2]), max(thrust_vector[2], 1e-9)))
            if body < metrics["body_clearance_min"]:
                metrics["body_clearance_min"] = body
                metrics["body_obstacle_min"] = body_name
            if arm < metrics["arm_clearance_min"]:
                metrics["arm_clearance_min"] = arm
                metrics["arm_obstacle_min"] = arm_names[index]
            for key, value in (
                ("com_offset_max", com_offset), ("gravity_torque_max", gravity_torque),
                ("velocity_max", speed), ("acceleration_max", accel), ("jerk_max", jerk_norm),
                ("yaw_rate_max", abs(velocity[index, YAW])), ("theta_rate_max", abs(velocity[index, THETA])),
                ("phi_rate_max", abs(velocity[index, PHI])), ("tilt_max", tilt), ("thrust_ratio_max", thrust_ratio),
            ):
                metrics[key] = max(metrics[key], value)
            metrics["thrust_ratio_min"] = min(metrics["thrust_ratio_min"], thrust_ratio)
            checks = (
                (body >= self.limits.body_margin, "body_collision"),
                (arm >= self.limits.arm_margin, "arm_collision"),
                (com_offset <= self.limits.com_max, "com_limit"),
                (gravity_torque <= self.limits.resolved_torque_max(self.mass.total), "gravity_torque_limit"),
                (speed <= self.limits.velocity_max, "velocity_limit"),
                (accel <= self.limits.acceleration_max, "acceleration_limit"),
                (jerk_norm <= self.limits.jerk_max, "jerk_limit"),
                (abs(velocity[index, YAW]) <= self.limits.yaw_rate_max, "yaw_rate_limit"),
                (abs(velocity[index, THETA]) <= self.limits.theta_rate_max, "theta_rate_limit"),
                (abs(velocity[index, PHI]) <= self.limits.phi_rate_max, "phi_rate_limit"),
                (abs(acceleration[index, YAW]) <= self.limits.yaw_acceleration_max, "yaw_acceleration_limit"),
                (abs(acceleration[index, THETA]) <= self.limits.theta_acceleration_max, "theta_acceleration_limit"),
                (abs(acceleration[index, PHI]) <= self.limits.phi_acceleration_max, "phi_acceleration_limit"),
                (tilt <= self.limits.tilt_max, "tilt_limit"),
                (self.limits.thrust_min_ratio <= thrust_ratio <= self.limits.thrust_max_ratio, "thrust_limit"),
                (self.limits.z_min <= current[2] <= self.limits.z_max, "altitude_limit"),
                (self.arm.theta_min <= current[THETA] <= self.arm.theta_max, "theta_position_limit"),
            )
            if failure is None:
                for passed, reason in checks:
                    if not passed:
                        failure = (index, float(times[index]), reason)
                        break
            if failure is not None and stop_on_failure:
                break
        if failure is None and self.terminal_target is not None:
            terminal_tip = world_arm_points(state[-1], self.arm)[-1]
            tip_error = float(np.linalg.norm(terminal_tip - self.terminal_target))
            metrics["terminal_tip_error"] = tip_error
            if tip_error > self.terminal_tolerance:
                failure = (len(times) - 1, float(times[-1]), "terminal_tip_error")
        if failure is None:
            return ValidationResult(True, "safe", metrics)
        return ValidationResult(False, "unsafe", metrics, failure[0], failure[1], failure[2])
