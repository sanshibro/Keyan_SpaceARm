from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


STATE_DIM = 6
X, Y, Z, YAW, THETA, PHI = range(STATE_DIM)


@dataclass
class ArmModel:
    arc_length: float = 0.22
    radius: float = 0.034
    segment_count: int = 18
    base_offset_z: float = 0.10
    gripper_length: float = 0.16
    mount_xyz: tuple = (0.0, 0.0, -0.03)
    mount_rpy: tuple = (0.0, np.pi, 0.0)
    theta_min: float = 0.0
    theta_max: float = np.deg2rad(120.0)
    # Optional end-tool sample cloud; appended to preserve positional arguments.
    extra_tool_points: tuple = ()


@dataclass
class MassModel:
    uav: float = 1.50
    arm_base: float = 0.30
    flexible: float = 0.25
    tool: float = 0.20

    @property
    def total(self):
        return self.uav + self.arm_base + self.flexible + self.tool


@dataclass
class Limits:
    body_radius: float = 0.28
    body_height: float = 0.25
    body_margin: float = 0.06
    arm_margin: float = 0.035
    com_max: float = 0.08
    torque_max: Optional[float] = None
    # Generic library default; deployment scenarios provide their own limit.
    velocity_max: float = 0.60
    acceleration_max: float = 1.00
    jerk_max: float = 3.00
    yaw_rate_max: float = np.deg2rad(45.0)
    yaw_acceleration_max: float = np.deg2rad(90.0)
    tilt_max: float = np.deg2rad(15.0)
    thrust_min_ratio: float = 0.50
    thrust_max_ratio: float = 1.60
    theta_rate_max: float = np.deg2rad(50.0)
    phi_rate_max: float = np.deg2rad(60.0)
    theta_acceleration_max: float = np.deg2rad(100.0)
    phi_acceleration_max: float = np.deg2rad(120.0)
    z_min: float = 0.75
    z_max: float = 2.20

    def resolved_torque_max(self, total_mass):
        if self.torque_max is not None:
            return float(self.torque_max)
        return float(total_mass) * 9.81 * self.com_max


@dataclass
class PlannerConfig:
    horizon_distance: float = 2.2
    knot_count: int = 8
    nominal_speed: float = 0.35
    dt_min: float = 0.30
    dt_max: float = 2.50
    collocation_per_span: int = 3
    risk_collocation_per_span: int = 7
    optimizer_max_iterations: int = 80
    optimizer_time_budget: float = 1.5
    optimization_basis_count: int = 3
    seed_beam_width: int = 12
    seed_pose_limit: int = 5
    seed_shape_limit: int = 12
    seed_edge_max_samples: int = 3
    seed_lateral_step: float = 0.12
    seed_z_step: float = 0.08
    risk_margin: float = 0.12
    terminal_tolerance: float = 0.30
    output_dt: float = 0.05
    swept_translation_step: float = 0.03
    swept_theta_step: float = np.deg2rad(3.0)
    swept_phi_step: float = np.deg2rad(5.0)
    swept_time_step: float = 0.04
    repair_attempts: int = 2
    repair_axis_only: bool = False
    repair_proxy_reserve: float = 0.004
    # Reduce morphology search dimension when one bending plane is already
    # sufficient for every risky knot in the current horizon.  This is a
    # geometry-derived active-set decision, not a scene/wall identifier.
    adaptive_phi: bool = False
    adaptive_phi_clearance_reserve: float = 0.004
    locked_phi_candidate_limit: int = 6
    # The optimizer stays strictly inside the independent validator's hard
    # limits.  This avoids repeatedly producing a trajectory that differs
    # from a hard velocity/theta boundary only by floating-point noise.
    design_velocity_ratio: float = 0.94
    theta_design_margin: float = np.deg2rad(2.0)
    dynamic_repair_reserve: float = 1.08
    random_seed: int = 0
    weights: Dict[str, float] = field(default_factory=lambda: {
        "reference": 12.0,
        "yaw_reference": 1.0,
        "jerk": 0.06,
        "arm_rate": 0.02,
        "arm_acceleration": 0.01,
        "clearance": 80.0,
        "com": 25.0,
        "duration": 0.35,
        "dt_smooth": 1.0,
        "terminal": 30.0,
        "risk_shape_hold": 1.5,
    })


@dataclass
class TrajectoryData:
    times: np.ndarray
    states: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    jerks: np.ndarray
    source: str = "unknown"
    objective: float = float("inf")
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=float)
        self.states = np.asarray(self.states, dtype=float)
        self.velocities = np.asarray(self.velocities, dtype=float)
        self.accelerations = np.asarray(self.accelerations, dtype=float)
        self.jerks = np.asarray(self.jerks, dtype=float)
        count = len(self.times)
        if self.states.shape != (count, STATE_DIM):
            raise ValueError("states must have shape (N, 6)")
        for name, values in (
            ("velocities", self.velocities),
            ("accelerations", self.accelerations),
            ("jerks", self.jerks),
        ):
            if values.shape != (count, STATE_DIM):
                raise ValueError("%s must have shape (N, 6)" % name)
        if count < 2 or np.any(np.diff(self.times) <= 0.0):
            raise ValueError("trajectory times must be strictly increasing")

    @property
    def duration(self):
        return float(self.times[-1] - self.times[0])


@dataclass
class ValidationResult:
    feasible: bool
    status: str
    metrics: Dict[str, float]
    failure_index: int = -1
    failure_time: float = -1.0
    failure_reason: str = "none"


@dataclass
class OptimizationResult:
    success: bool
    status: str
    trajectory: Optional[TrajectoryData]
    objective: float
    iterations: int
    elapsed: float
    validation: Optional[ValidationResult] = None
    diagnostics: Dict = field(default_factory=dict)
