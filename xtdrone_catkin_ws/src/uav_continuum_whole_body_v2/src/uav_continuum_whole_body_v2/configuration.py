import math
from pathlib import Path

import yaml

from .types import Limits, MassModel, PlannerConfig


def load_planner_configuration(path):
    """Load the numerical models shared by the ROS node and offline checks."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = (yaml.safe_load(handle) or {}).get("planner_v2", {})

    limits_raw = raw.get("limits", {})
    mass_raw = raw.get("mass", {})
    validator_raw = raw.get("validator", {})
    config = PlannerConfig(
        horizon_distance=float(raw.get("horizon_distance", 2.2)),
        knot_count=int(raw.get("knot_count", 8)),
        nominal_speed=float(raw.get("nominal_speed", 0.35)),
        dt_min=float(raw.get("dt_min", 0.3)),
        dt_max=float(raw.get("dt_max", 2.5)),
        collocation_per_span=int(raw.get("collocation_per_span", 3)),
        risk_collocation_per_span=int(raw.get("risk_collocation_per_span", 7)),
        optimizer_max_iterations=int(raw.get("optimizer_max_iterations", 80)),
        optimizer_time_budget=float(raw.get("optimizer_time_budget", 4.0)),
        optimization_basis_count=int(raw.get("optimization_basis_count", 3)),
        seed_beam_width=int(raw.get("seed_beam_width", 12)),
        seed_pose_limit=int(raw.get("seed_pose_limit", 5)),
        seed_shape_limit=int(raw.get("seed_shape_limit", 12)),
        seed_edge_max_samples=int(raw.get("seed_edge_max_samples", 3)),
        repair_attempts=max(0, int(raw.get("repair_attempts", 1))),
        repair_axis_only=bool(raw.get("repair_axis_only", False)),
        repair_proxy_reserve=max(0.0, float(raw.get("repair_proxy_reserve", 0.004))),
        adaptive_phi=bool(raw.get("adaptive_phi", False)),
        adaptive_phi_clearance_reserve=max(
            0.0, float(raw.get("adaptive_phi_clearance_reserve", 0.004))
        ),
        locked_phi_candidate_limit=max(
            2, int(raw.get("locked_phi_candidate_limit", 6))
        ),
        design_velocity_ratio=min(
            0.99, max(0.50, float(raw.get("design_velocity_ratio", 0.94)))
        ),
        theta_design_margin=math.radians(max(
            0.0, float(raw.get("theta_design_margin_deg", 2.0))
        )),
        dynamic_repair_reserve=max(
            1.01, float(raw.get("dynamic_repair_reserve", 1.08))
        ),
        risk_margin=float(raw.get("risk_margin", 0.12)),
        output_dt=float(raw.get("output_dt", 0.05)),
        swept_translation_step=float(validator_raw.get("translation_step", 0.03)),
        swept_theta_step=math.radians(float(validator_raw.get("theta_step_deg", 3.0))),
        swept_phi_step=math.radians(float(validator_raw.get("phi_step_deg", 5.0))),
        swept_time_step=float(validator_raw.get("time_step", 0.04)),
    )
    limits = Limits(
        body_radius=float(limits_raw.get("body_radius", 0.28)),
        body_height=float(limits_raw.get("body_height", 0.25)),
        body_margin=float(limits_raw.get("body_margin", 0.06)),
        arm_margin=float(limits_raw.get("arm_margin", 0.035)),
        com_max=float(limits_raw.get("com_max", 0.08)),
        velocity_max=float(limits_raw.get("velocity_max", 0.6)),
        acceleration_max=float(limits_raw.get("acceleration_max", 1.0)),
        jerk_max=float(limits_raw.get("jerk_max", 3.0)),
        yaw_rate_max=math.radians(float(limits_raw.get("yaw_rate_max_deg", 45.0))),
        yaw_acceleration_max=math.radians(float(limits_raw.get("yaw_acceleration_max_deg", 90.0))),
        tilt_max=math.radians(float(limits_raw.get("tilt_max_deg", 15.0))),
        thrust_min_ratio=float(limits_raw.get("thrust_min_ratio", 0.5)),
        thrust_max_ratio=float(limits_raw.get("thrust_max_ratio", 1.6)),
        theta_rate_max=math.radians(float(limits_raw.get("theta_rate_max_deg", 50.0))),
        phi_rate_max=math.radians(float(limits_raw.get("phi_rate_max_deg", 60.0))),
        theta_acceleration_max=math.radians(float(limits_raw.get("theta_acceleration_max_deg", 100.0))),
        phi_acceleration_max=math.radians(float(limits_raw.get("phi_acceleration_max_deg", 120.0))),
        z_min=float(limits_raw.get("z_min", 0.75)),
        z_max=float(limits_raw.get("z_max", 2.2)),
    )
    mass = MassModel(**{key: float(value) for key, value in mass_raw.items()})
    return raw, config, limits, mass
