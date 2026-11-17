#!/usr/bin/env python3
import json
import math
from pathlib import Path
import sys
import time

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "src"))

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import PositionTarget, State
from nav_msgs.msg import Path as RosPath
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool, Float32MultiArray, String

from uav_continuum_whole_body_v2.execution import (
    bending_state_from_vector, bending_vector, bending_vector_rate,
    bounded_switch_match, bounded_whole_body_switch_match,
    clearance_progress_scale, clearance_tracking_budget,
    continuous_switch_phase_rate, whole_body_switch_gate,
    evaluate_quintic_bridge, quintic_bridge_coefficients,
    interpolation_bracket,
    largest_feasible_ratio,
    lerp,
    lerp_angle,
    position_distance,
    quaternion_from_yaw,
    reference_converged,
    stationary_endpoint_feedforward,
    tracking_progress_scale,
    wrap_angle,
    yaw_from_quaternion,
)
from uav_continuum_whole_body_v2.environment import BoxWorld
from uav_continuum_whole_body_v2.geometry import (
    interpolate_state,
    world_arm_points,
    world_body_points,
)
from uav_continuum_whole_body_v2.msg import WholeBodyTrajectory
from uav_continuum_whole_body_v2.mapping import (
    MapSnapshot, OnlineSnapshotAssembler, validate_map_mode,
)
from uav_continuum_whole_body_v2.types import ArmModel, Limits


class WholeBodyTrajectoryExecutorNode:
    """Fail-closed, synchronized UAV/arm executor for validator-approved trajectories."""

    def __init__(self):
        self.enabled = bool(rospy.get_param("~auto_execute", False))
        self.publish_hold_when_disabled = bool(rospy.get_param("~publish_hold_when_disabled", True))
        self.require_px4_ready = bool(rospy.get_param("~require_px4_ready", True))
        self.required_mode = str(rospy.get_param("~required_mode", "OFFBOARD"))
        self.handoff_lead = max(0.2, float(rospy.get_param("~handoff_lead", 0.6)))
        self.maximum_start_gap = float(rospy.get_param("~maximum_start_gap", 0.35))
        self.maximum_switch_gap = float(rospy.get_param("~maximum_switch_gap", 0.25))
        self.rolling_switch_target_gap = min(
            self.maximum_switch_gap,
            max(0.02, float(rospy.get_param("~rolling_switch_target_gap", 0.10))),
        )
        self.rolling_switch_max_advance = max(
            self.rolling_switch_target_gap,
            float(rospy.get_param("~rolling_switch_max_advance", 0.35)),
        )
        self.rolling_switch_wait_timeout = max(
            1.0, float(rospy.get_param("~rolling_switch_wait_timeout", 15.0))
        )
        # Objective normalization is separate from the explicit C0/C1 hard
        # gate below. A low minimax score selects a candidate phase; it does
        # not by itself authorize a discontinuous handoff.
        self.switch_match_scales = {
            "position_gap_m": self.rolling_switch_target_gap,
            "measured_position_gap_m": self.rolling_switch_target_gap,
            "yaw_gap_rad": math.radians(float(rospy.get_param(
                "~switch_match_yaw_scale_deg", 10.0
            ))),
            "bending_gap_rad": math.radians(float(rospy.get_param(
                "~switch_match_bending_scale_deg", 10.0
            ))),
            "linear_velocity_gap_m_s": float(rospy.get_param(
                "~switch_match_velocity_scale", 0.05
            )),
            "yaw_rate_gap_rad_s": math.radians(float(rospy.get_param(
                "~switch_match_yaw_rate_scale_deg_s", 10.0
            ))),
            "bending_rate_gap_rad_s": math.radians(float(rospy.get_param(
                "~switch_match_bending_rate_scale_deg_s", 10.0
            ))),
            "linear_acceleration_gap_m_s2": float(rospy.get_param(
                "~switch_match_acceleration_scale_m_s2", 0.15
            )),
        }
        self.switch_gate_limits = {
            "position_gap_m": float(rospy.get_param(
                "~switch_gate_position_m", 0.06
            )),
            "yaw_gap_rad": math.radians(float(rospy.get_param(
                "~switch_gate_yaw_deg", 5.0
            ))),
            "bending_gap_rad": math.radians(float(rospy.get_param(
                "~switch_gate_bending_deg", 8.0
            ))),
            "linear_velocity_gap_m_s": float(rospy.get_param(
                "~switch_gate_velocity_m_s", 0.04
            )),
            "yaw_rate_gap_rad_s": math.radians(float(rospy.get_param(
                "~switch_gate_yaw_rate_deg_s", 5.0
            ))),
            "bending_rate_gap_rad_s": math.radians(float(rospy.get_param(
                "~switch_gate_bending_rate_deg_s", 8.0
            ))),
        }
        if any((not np.isfinite(value) or value <= 0.0)
               for value in self.switch_gate_limits.values()):
            raise ValueError("whole-body switch gate limits must be finite and positive")
        self.enable_switch_bridge = bool(rospy.get_param(
            "~enable_switch_bridge", True
        ))
        self.switch_bridge_duration = max(
            0.4, float(rospy.get_param("~switch_bridge_duration", 1.5))
        )
        self.switch_bridge_validation_dt = min(
            0.05, max(0.01, float(rospy.get_param(
                "~switch_bridge_validation_dt", 0.02
            )))
        )
        self.pending_start_expiry = max(
            0.2, float(rospy.get_param("~pending_start_expiry", 0.5))
        )
        self.maximum_tracking_error = float(rospy.get_param("~maximum_tracking_error", 0.45))
        self.tracking_error_hold = float(rospy.get_param("~tracking_error_hold", 0.75))
        self.tracking_grace = float(rospy.get_param("~tracking_grace", 1.5))
        self.first_tracking_grace = float(rospy.get_param("~first_tracking_grace", 5.0))
        # A replacement whole-body window can take several seconds to optimize
        # and validate on the target CPU.  Keep commanding the already
        # validated endpoint long enough for either PX4 to settle or a
        # measured-state replacement window to arrive.  Three seconds proved
        # shorter than one planner cycle in the clean three-wall flight and
        # caused a false fail-closed stop while the vehicle was still safely
        # converging.
        self.final_convergence_timeout = float(rospy.get_param("~final_convergence_timeout", 15.0))
        self.completion_tracking_tolerance = float(
            rospy.get_param("~completion_tracking_tolerance", 0.18)
        )
        # MAVROS State is commonly published at 1 Hz; leave scheduling margin
        # while still failing closed after a genuinely stale connection.
        self.input_timeout = float(rospy.get_param("~input_timeout", 2.5))
        self.mavros_state_timeout = float(
            rospy.get_param("~mavros_state_timeout", max(5.0, self.input_timeout))
        )
        self.history_period = max(0.05, float(rospy.get_param("~history_period", 0.10)))
        # PX4 local-position and Gazebo world frames are only approximately
        # rigid in SITL: EKF instance changes and estimator drift move the
        # local origin during a flight.  Re-estimate the translation from the
        # synchronized measured poses instead of freezing it at first handoff.
        self.transform_update_alpha = min(
            1.0, max(0.01, float(rospy.get_param("~transform_update_alpha", 1.0)))
        )
        # Position-only setpoints make PX4 create a steady spatial lag of
        # roughly velocity / position-loop gain.  That lag is not acceptable
        # when the validated whole-body clearance reserve is only a few
        # centimetres.  The optimizer already supplies velocity and yaw-rate;
        # pass them to PX4 as feed-forward terms on the same reference clock.
        self.use_velocity_feedforward = bool(
            rospy.get_param("~use_velocity_feedforward", True)
        )
        self.enable_tracking_tube = bool(
            rospy.get_param("~enable_tracking_tube", False)
        )
        # Advance along validator-approved trajectory time only while the
        # measured whole body remains inside a geometric tracking tube.  This
        # is a phase governor: pausing it keeps a fixed, already certified
        # reference state instead of chasing wall-clock time through a narrow
        # passage.
        self.enable_phase_guard = bool(rospy.get_param("~enable_phase_guard", True))
        self.phase_body_full_error = float(
            rospy.get_param("~phase_body_full_error", 0.015)
        )
        self.phase_body_stop_error_max = float(
            rospy.get_param("~phase_body_stop_error_max", 0.10)
        )
        self.phase_arm_full_error = float(
            rospy.get_param("~phase_arm_full_error", 0.008)
        )
        self.phase_arm_stop_error_max = float(
            rospy.get_param("~phase_arm_stop_error_max", 0.10)
        )
        self.phase_body_clearance_reserve = float(
            rospy.get_param("~phase_body_clearance_reserve", 0.01)
        )
        self.phase_arm_clearance_reserve = float(
            rospy.get_param("~phase_arm_clearance_reserve", 0.02)
        )
        self.phase_calibrated_full_speed_error = max(
            0.01, float(rospy.get_param("~phase_calibrated_full_speed_error", 0.06))
        )
        if (self.phase_body_stop_error_max <= self.phase_body_full_error
                or self.phase_arm_stop_error_max <= self.phase_arm_full_error):
            raise ValueError("phase stop-error maxima must exceed full-speed errors")

        # The runtime guard must use the same deployable map as the planner.
        # Exact Gazebo geometry is permitted only in the explicitly labelled
        # regression mode; the offline oracle remains independent.
        self.map_mode = validate_map_mode(rospy.get_param(
            "~map_mode", "regression_exact"
        ))
        self.map_inflation = float(rospy.get_param("~map_inflation", 0.02))
        self.online_map_assembler = OnlineSnapshotAssembler()
        self.map_snapshot = None
        if self.map_mode == "regression_exact":
            boxes_file = Path(rospy.get_param(
                "~safety_boxes_file",
                str(PACKAGE / "generated" / "scenario_3c_unified_boxes.yaml"),
            ))
            self.safety_world = BoxWorld.from_yaml_files([boxes_file])
        elif self.map_mode == "frozen_slam":
            snapshot_path = Path(rospy.get_param("~map_snapshot_file", ""))
            if not snapshot_path.is_file():
                raise FileNotFoundError(
                    "frozen_slam requires ~map_snapshot_file, got %s" % snapshot_path
                )
            self.map_snapshot = MapSnapshot.load(snapshot_path)
            self.safety_world = self.map_snapshot.world(
                inflation=self.map_inflation, unknown_is_occupied=True
            )
        else:
            self.safety_world = None
        self.arm_model = ArmModel(
            arc_length=float(rospy.get_param("/arm/arc_length", 0.22)),
            radius=float(rospy.get_param("/arm/radius", 0.034)),
            segment_count=int(rospy.get_param("/arm/segment_count", 18)),
            base_offset_z=float(rospy.get_param("/arm/base_offset_z", 0.10)),
            gripper_length=float(rospy.get_param("/arm/gripper_length", 0.16)),
            mount_xyz=tuple(rospy.get_param("/arm/mount_xyz", [0.0, 0.0, -0.03])),
            mount_rpy=tuple(rospy.get_param("/arm/mount_rpy", [0.0, math.pi, 0.0])),
        )
        self.safety_limits = Limits(
            body_radius=float(rospy.get_param("/planner_v2/limits/body_radius", 0.28)),
            body_height=float(rospy.get_param("/planner_v2/limits/body_height", 0.25)),
        )
        self.actual_body_abort_clearance = float(
            rospy.get_param("~actual_body_abort_clearance", 0.06)
        )
        self.actual_arm_abort_clearance = float(
            rospy.get_param("~actual_arm_abort_clearance", 0.035)
        )
        self.tracking_tube_body_reserve = max(
            0.0, float(rospy.get_param("~tracking_tube_body_reserve", 0.02))
        )
        self.tracking_tube_arm_reserve = max(
            0.0, float(rospy.get_param("~tracking_tube_arm_reserve", 0.015))
        )
        self.tracking_tube_geometry_factor = max(
            1.0, float(rospy.get_param("~tracking_tube_geometry_factor", 1.10))
        )
        self.tracking_tube_reaction_time = max(
            0.0, float(rospy.get_param("~tracking_tube_reaction_time", 0.25))
        )
        self.tracking_tube_braking_acceleration = max(
            0.01, float(rospy.get_param("~tracking_tube_braking_acceleration", 0.03))
        )
        self.tracking_tube_max_command_step = max(
            0.0005, float(rospy.get_param("~tracking_tube_max_command_step", 0.0025))
        )
        self.recovery_body_clearance = float(rospy.get_param(
            "~recovery_body_clearance", self.actual_body_abort_clearance + 0.05
        ))
        self.recovery_arm_clearance = float(rospy.get_param(
            "~recovery_arm_clearance", self.actual_arm_abort_clearance + 0.04
        ))
        self.safety_status_period = max(
            0.05, float(rospy.get_param("~safety_status_period", 0.20))
        )

        self.trajectory = None
        self.pending = None
        self.pending_wait_started_wall = None
        self.last_continuity_status_wall = 0.0
        self.active_start = None
        self.phase_elapsed = 0.0
        self.phase_last_wall = time.monotonic()
        self.phase_rate = 0.0
        self.phase_body_budget = 0.0
        self.phase_arm_budget = 0.0
        self.locked_id = "none"
        # Before the atomic handoff the external takeoff node owns MAVROS.
        # After the first enable request this executor owns it permanently for
        # the run, including fail-closed holds after an abort.
        self.owns_setpoint = self.enabled
        self.first_active_trajectory = False
        self.world_pose = None
        self.local_pose = None
        self.world_rx_wall = 0.0
        self.local_rx_wall = 0.0
        self.mavros_state = None
        self.mavros_rx_wall = 0.0
        self.arm_state = [0.0, 0.0]
        self.linear_speed = 0.0
        self.velocity_rx_wall = 0.0
        self.hold_local = None
        self.hold_world = None
        self.hold_arm_state = [0.0, 0.0]
        self.last_recovery_world = None
        self.last_recovery_arm_state = [0.0, 0.0]
        # Stateful reference-governor command.  It must remain fixed while
        # progress is blocked; recomputing a zero-progress target from every
        # new measurement would create a drifting hold with no restoring
        # position error for PX4.
        self.governed_state = None
        self.world_to_local = [0.0, 0.0, 0.0]
        self.bad_tracking_since = None
        self.last_status_wall = 0.0
        self.last_history_wall = 0.0
        self.last_safety_status_wall = 0.0
        self.executed_path = RosPath()
        self.executed_path.header.frame_id = "world"
        self.completion_reported_id = None
        self.final_started_wall = None
        # Preserve the reference-clock rate across a rolling splice. Resetting
        # it to 1.0 for one timer tick creates a velocity impulse even when the
        # old and new certified positions are nearly identical.
        self.last_switch_phase_rate = 0.0
        # Exact command immediately before a rolling splice.  V2.1 logs this
        # together with the successor sample to distinguish position, tangent,
        # yaw and morphology discontinuities before changing switch behavior.
        self.last_commanded_switch_state = None
        self.last_commanded_switch_rate = None
        self.last_commanded_switch_acceleration = None
        self.last_commanded_switch_stamp = None
        self.switch_bridge = None

        self.setpoint_pub = rospy.Publisher(
            rospy.get_param("~setpoint_topic", "/iris_0/mavros/setpoint_position/local"),
            PoseStamped, queue_size=10,
        )
        self.raw_setpoint_pub = rospy.Publisher(
            rospy.get_param("~raw_setpoint_topic", "/iris_0/mavros/setpoint_raw/local"),
            PositionTarget, queue_size=10,
        )
        self.arm_pub = rospy.Publisher(
            rospy.get_param("~arm_cmd_topic", "/arm/theta_phi_cmd"),
            Float32MultiArray, queue_size=10,
        )
        self.status_pub = rospy.Publisher("/planner/v2/executor_status", String, queue_size=10, latch=True)
        self.feedback_pub = rospy.Publisher("/planner/v2/execution_feedback", String, queue_size=10, latch=True)
        self.commanded_world_pub = rospy.Publisher(
            "/planner/v2/commanded_pose_world", PoseStamped, queue_size=10,
        )
        self.commanded_twist_world_pub = rospy.Publisher(
            "/planner/v2/commanded_twist_world", TwistStamped, queue_size=10,
        )
        self.executed_path_pub = rospy.Publisher(
            "/planner/v2/executed_path", RosPath, queue_size=1, latch=True,
        )
        self.safety_status_pub = rospy.Publisher(
            "/planner/v2/actual_safety_status", String, queue_size=10, latch=True,
        )

        rospy.Subscriber("/planner/v2/whole_body_trajectory", WholeBodyTrajectory, self._trajectory_cb, queue_size=2)
        rospy.Subscriber("/planner/v2/executor_enable", Bool, self._enable_cb, queue_size=2)
        rospy.Subscriber("/planner/v2/hold_pose_local", PoseStamped, self._hold_request_cb, queue_size=2)
        rospy.Subscriber(rospy.get_param("~world_pose_topic", "/iris_0/world_pose"), PoseStamped,
                         self._world_pose_cb, queue_size=10)
        rospy.Subscriber(rospy.get_param("~local_pose_topic", "/iris_0/mavros/local_position/pose"), PoseStamped,
                         self._local_pose_cb, queue_size=10)
        rospy.Subscriber(rospy.get_param(
            "~local_velocity_topic", "/iris_0/mavros/local_position/velocity_local"
        ), TwistStamped, self._velocity_cb, queue_size=10)
        rospy.Subscriber(rospy.get_param("~mavros_state_topic", "/iris_0/mavros/state"), State,
                         self._mavros_state_cb, queue_size=10)
        rospy.Subscriber(rospy.get_param("~arm_state_topic", "/arm/current_theta_phi"), Float32MultiArray,
                         self._arm_state_cb, queue_size=10)
        if self.map_mode == "online_slam":
            rospy.Subscriber(rospy.get_param(
                "~occupied_voxels_topic", "/planner/map/occupied_voxels"
            ), PointCloud2, self._online_occupied_cb, queue_size=1)
            rospy.Subscriber(rospy.get_param(
                "~observed_voxels_topic", "/planner/map/observed_voxels"
            ), PointCloud2, self._online_observed_cb, queue_size=1)
            rospy.Subscriber(rospy.get_param(
                "~map_metadata_topic", "/planner/map/metadata"
            ), String, self._online_metadata_cb, queue_size=1)

        rate = max(20.0, float(rospy.get_param("~rate_hz", 40.0)))
        self.timer = rospy.Timer(rospy.Duration(1.0 / rate), self._tick)
        self._status("WAITING_INPUT" if not self.enabled else "WAITING_TRAJECTORY")

    def _status(self, state, **fields):
        report = {"state": state, "enabled": self.enabled, "trajectory_id": self.locked_id,
                  "stamp": rospy.Time.now().to_sec(), "map_mode": self.map_mode,
                  "map_hash": None if self.map_snapshot is None else self.map_snapshot.map_hash,
                  "truth_geometry_access": self.map_mode == "regression_exact"}
        report.update(fields)
        self.status_pub.publish(String(data=json.dumps(report, sort_keys=True)))
        self.last_status_wall = time.monotonic()

    def _feedback(self, state, **fields):
        report = {"state": state, "trajectory_id": self.locked_id, "stamp": rospy.Time.now().to_sec()}
        report.update(fields)
        self.feedback_pub.publish(String(data=json.dumps(report, sort_keys=True)))

    def _world_pose_cb(self, message):
        self.world_pose = message
        self.world_rx_wall = time.monotonic()

    def _local_pose_cb(self, message):
        self.local_pose = message
        self.local_rx_wall = time.monotonic()
        if self.hold_local is None:
            self.hold_local = self._copy_pose(message, message.header.frame_id or "map")

    def _mavros_state_cb(self, message):
        self.mavros_state = message
        self.mavros_rx_wall = time.monotonic()

    def _velocity_cb(self, message):
        value = message.twist.linear
        self.linear_speed = math.sqrt(value.x * value.x + value.y * value.y + value.z * value.z)
        self.velocity_rx_wall = time.monotonic()

    def _arm_state_cb(self, message):
        if len(message.data) >= 2:
            self.arm_state = [float(message.data[0]), float(message.data[1])]

    @staticmethod
    def _message_points(message):
        return np.asarray([
            (x, y, z) for x, y, z in point_cloud2.read_points(
                message, field_names=("x", "y", "z"), skip_nans=True
            )
        ], dtype=float).reshape((-1, 3))

    def _accept_online_snapshot(self, snapshot):
        if snapshot is None:
            return
        self.map_snapshot = snapshot
        self.safety_world = snapshot.world(
            inflation=self.map_inflation, unknown_is_occupied=True
        )

    def _online_occupied_cb(self, message):
        try:
            snapshot = self.online_map_assembler.update_occupied(
                self._message_points(message), message.header.frame_id,
                message.header.stamp.to_sec(),
            )
        except ValueError as error:
            rospy.logerr_throttle(2.0, "executor rejected online map: %s", error)
            return
        self._accept_online_snapshot(snapshot)

    def _online_observed_cb(self, message):
        try:
            snapshot = self.online_map_assembler.update_observed(
                self._message_points(message), message.header.frame_id,
                message.header.stamp.to_sec(),
            )
        except ValueError as error:
            rospy.logerr_throttle(2.0, "executor rejected online map: %s", error)
            return
        self._accept_online_snapshot(snapshot)

    def _online_metadata_cb(self, message):
        try:
            metadata = json.loads(message.data)
            snapshot = self.online_map_assembler.update_metadata(metadata)
        except (ValueError, TypeError) as error:
            rospy.logerr_throttle(2.0, "executor rejected online map metadata: %s", error)
            return
        self._accept_online_snapshot(snapshot)

    @staticmethod
    def _copy_pose(message, frame_id):
        copied = PoseStamped()
        copied.header.stamp = rospy.Time.now()
        copied.header.frame_id = frame_id
        copied.pose = message.pose
        return copied

    def _enable_cb(self, message):
        requested = bool(message.data)
        if not requested:
            self._abort("DISABLED_BY_OPERATOR", disable=True)
            return
        # Take control from the external takeoff publisher at the *measured*
        # pose.  A certified trajectory may have been inspected for many
        # seconds before the operator enables execution, so its first state
        # is not automatically a safe handoff state any more.
        if self._inputs_ready() == []:
            self._capture_transform_and_hold()
        self.enabled = True
        self.owns_setpoint = True
        self._status("ENABLE_REQUESTED")

    def _hold_request_cb(self, message):
        """Allow takeoff/hover setup through the executor's single MAVROS publisher."""
        if self.enabled:
            self._status("REJECTED_HOLD_REQUEST_WHILE_ENABLED")
            return
        values = (message.pose.position.x, message.pose.position.y, message.pose.position.z,
                  message.pose.orientation.x, message.pose.orientation.y,
                  message.pose.orientation.z, message.pose.orientation.w)
        if not all(math.isfinite(value) for value in values):
            self._status("REJECTED_HOLD_REQUEST_NONFINITE")
            return
        requested = self._copy_pose(message, message.header.frame_id or "map")
        if abs(requested.pose.orientation.x) + abs(requested.pose.orientation.y) + \
                abs(requested.pose.orientation.z) + abs(requested.pose.orientation.w) < 1e-6:
            requested.pose.orientation.w = 1.0
        self.hold_local = requested
        self._status("HOLD_TARGET_UPDATED", target=[message.pose.position.x,
                                                     message.pose.position.y,
                                                     message.pose.position.z])

    def _trajectory_cb(self, message):
        count = len(message.time_from_start)
        if message.validator_status != "safe" or count < 2:
            self._status("REJECTED_MESSAGE", validator_status=message.validator_status, samples=count)
            return
        if not (len(message.uav_poses) == len(message.linear_velocities)
                == len(message.yaw_rates) == len(message.theta_deg)
                == len(message.phi_deg) == len(message.theta_rate_deg_s)
                == len(message.phi_rate_deg_s) == count):
            self._status("REJECTED_MESSAGE", reason="array_length_mismatch")
            return
        if any(message.time_from_start[index] <= message.time_from_start[index - 1] for index in range(1, count)):
            self._status("REJECTED_MESSAGE", reason="non_monotonic_time")
            return
        if self.pending is not None:
            # Protect the first certified successor until its scheduled
            # handoff is either accepted or explicitly rejected.  Replacing
            # it with every later publication can discard a perfectly
            # continuous window just before its switch time.
            rejected_id = int(message.trajectory_id)
            pending_id = int(self.pending.trajectory_id)
            if rejected_id == pending_id:
                return
            self._status(
                "REJECTED_PENDING_EXISTS", rejected_id=rejected_id,
                pending_id=pending_id,
            )
            self._feedback(
                "REJECTED_PENDING_EXISTS", trajectory_id=rejected_id,
                pending_id=pending_id, active_trajectory_id=self.locked_id,
            )
            return
        self.pending = message
        self.pending_wait_started_wall = None
        self._status("BUFFERED", pending_id=int(message.trajectory_id),
                     planned_start=message.execution_start_time.to_sec(), duration=message.duration)
        self._feedback(
            "BUFFERED", trajectory_id=int(message.trajectory_id),
            planned_start=message.execution_start_time.to_sec(),
        )

    def _inputs_ready(self):
        now = time.monotonic()
        missing = []
        if self.safety_world is None:
            missing.append("planner_map")
        if self.world_pose is None or now - self.world_rx_wall > self.input_timeout:
            missing.append("world_pose")
        if self.local_pose is None or now - self.local_rx_wall > self.input_timeout:
            missing.append("local_pose")
        if (self.enable_tracking_tube
                and (self.velocity_rx_wall <= 0.0
                     or now - self.velocity_rx_wall > self.input_timeout)):
            missing.append("local_velocity")
        if self.require_px4_ready:
            if (self.mavros_state is None
                    or now - self.mavros_rx_wall > self.mavros_state_timeout):
                missing.append("mavros_state")
            elif not self.mavros_state.connected:
                missing.append("mavros_disconnected")
            elif not self.mavros_state.armed:
                missing.append("mavros_disarmed")
            elif self.required_mode and self.mavros_state.mode != self.required_mode:
                missing.append("mode_" + self.mavros_state.mode)
        return missing

    def _capture_transform_and_hold(self):
        world = self.world_pose.pose.position
        local = self.local_pose.pose.position
        self.world_to_local = [local.x - world.x, local.y - world.y, local.z - world.z]
        self.hold_local = self._copy_pose(self.local_pose, self.local_pose.header.frame_id or "map")
        self.hold_world = self._copy_pose(self.world_pose, "world")
        self.hold_arm_state = list(self.arm_state)

    def _update_world_to_local_transform(self):
        world = self.world_pose.pose.position
        local = self.local_pose.pose.position
        measured = np.asarray([
            local.x - world.x, local.y - world.y, local.z - world.z,
        ], dtype=float)
        previous = np.asarray(self.world_to_local, dtype=float)
        alpha = self.transform_update_alpha
        self.world_to_local = ((1.0 - alpha) * previous + alpha * measured).tolist()

    def _actual_clearance(self):
        state = self._actual_state()
        body_distance, body_name, _ = self.safety_world.minimum_distance(
            world_body_points(
                state, self.safety_limits, azimuth_samples=32, vertical_samples=5
            )
        )
        arm_distance, arm_name, _ = self.safety_world.minimum_distance(
            world_arm_points(state, self.arm_model)
        )
        arm_distance -= self.arm_model.radius
        return float(body_distance), body_name, float(arm_distance), arm_name

    def _actual_state(self):
        pose = self.world_pose.pose
        return np.asarray([
            pose.position.x, pose.position.y, pose.position.z,
            yaw_from_quaternion(pose.orientation),
            math.radians(self.arm_state[0]), math.radians(self.arm_state[1]),
        ], dtype=float)

    def _govern_reference(self, desired_world, theta_deg, phi_deg, velocity,
                          yaw_rate, body_clearance, arm_clearance):
        """Restrict the command to the measured whole-body clearance tube."""
        if not self.enable_tracking_tube:
            self.governed_state = None
            return (
                desired_world, theta_deg, phi_deg,
                np.asarray(velocity, dtype=float), float(yaw_rate),
                1.0, 0.0, 0.0, 0.0,
            )
        actual = self._actual_state()
        desired = np.asarray([
            desired_world.pose.position.x, desired_world.pose.position.y,
            desired_world.pose.position.z,
            yaw_from_quaternion(desired_world.pose.orientation),
            math.radians(theta_deg), math.radians(phi_deg),
        ], dtype=float)
        base = actual.copy() if self.governed_state is None else self.governed_state.copy()
        actual_body = world_body_points(
            actual, self.safety_limits, azimuth_samples=32, vertical_samples=5
        )
        actual_arm = world_arm_points(actual, self.arm_model)
        base_body = world_body_points(
            base, self.safety_limits, azimuth_samples=32, vertical_samples=5
        )
        base_arm = world_arm_points(base, self.arm_model)
        stopping_reserve = (
            self.linear_speed * self.tracking_tube_reaction_time
            + self.linear_speed * self.linear_speed
            / (2.0 * self.tracking_tube_braking_acceleration)
        )
        body_budget = max(
            0.0, body_clearance - self.actual_body_abort_clearance
            - self.tracking_tube_body_reserve - stopping_reserve
        )
        arm_budget = max(
            0.0, arm_clearance - self.actual_arm_abort_clearance
            - self.tracking_tube_arm_reserve - stopping_reserve
        )
        factor = self.tracking_tube_geometry_factor
        cache = {}

        def displacement(ratio):
            key = round(float(ratio), 12)
            if key not in cache:
                candidate = interpolate_state(base, desired, ratio)
                candidate_body = world_body_points(
                    candidate, self.safety_limits,
                    azimuth_samples=32, vertical_samples=5,
                )
                candidate_arm = world_arm_points(candidate, self.arm_model)
                body_delta = factor * float(np.max(np.linalg.norm(
                    candidate_body - actual_body, axis=1
                )))
                arm_delta = factor * float(np.max(np.linalg.norm(
                    candidate_arm - actual_arm, axis=1
                )))
                body_step = float(np.max(np.linalg.norm(
                    candidate_body - base_body, axis=1
                )))
                arm_step = float(np.max(np.linalg.norm(
                    candidate_arm - base_arm, axis=1
                )))
                candidate_body_clearance = float(
                    self.safety_world.minimum_distance(candidate_body)[0]
                )
                candidate_arm_clearance = float(
                    self.safety_world.minimum_distance(candidate_arm)[0]
                    - self.arm_model.radius
                )
                cache[key] = (
                    candidate, body_delta, arm_delta,
                    candidate_body_clearance, candidate_arm_clearance,
                    body_step, arm_step,
                )
            return cache[key]

        def feasible(ratio):
            (_candidate, body_delta, arm_delta,
             candidate_body_clearance, candidate_arm_clearance,
             body_step, arm_step) = displacement(ratio)
            return (
                body_delta <= body_budget and arm_delta <= arm_budget
                and body_step <= self.tracking_tube_max_command_step
                and arm_step <= self.tracking_tube_max_command_step
                and candidate_body_clearance >= (
                    self.actual_body_abort_clearance + self.tracking_tube_body_reserve
                )
                and candidate_arm_clearance >= (
                    self.actual_arm_abort_clearance + self.tracking_tube_arm_reserve
                )
            )

        alpha = largest_feasible_ratio(feasible, iterations=12)
        (candidate, body_delta, arm_delta,
         _candidate_body_clearance, _candidate_arm_clearance,
         _body_step, _arm_step) = displacement(alpha)
        self.governed_state = candidate.copy()
        governed = PoseStamped()
        governed.header.stamp = rospy.Time.now()
        governed.header.frame_id = desired_world.header.frame_id or "world"
        governed.pose.position.x = float(candidate[0])
        governed.pose.position.y = float(candidate[1])
        governed.pose.position.z = float(candidate[2])
        quaternion_from_yaw(float(candidate[3]), governed.pose.orientation)
        return (
            governed, math.degrees(candidate[4]), math.degrees(candidate[5]),
            np.asarray(velocity, dtype=float) * alpha, float(yaw_rate) * alpha,
            float(alpha), float(body_delta), float(arm_delta), float(stopping_reserve),
        )

    def _phase_guard(self, desired_world, theta_deg, phi_deg):
        """Return a safe trajectory phase rate from measured geometry error."""
        if not self.enable_phase_guard:
            return 1.0, 0.0, 0.0
        desired = np.asarray([
            desired_world.pose.position.x, desired_world.pose.position.y,
            desired_world.pose.position.z,
            yaw_from_quaternion(desired_world.pose.orientation),
            math.radians(theta_deg), math.radians(phi_deg),
        ], dtype=float)
        actual = self._actual_state()
        desired_body = world_body_points(
            desired, self.safety_limits, azimuth_samples=32, vertical_samples=5
        )
        actual_body = world_body_points(
            actual, self.safety_limits, azimuth_samples=32, vertical_samples=5
        )
        desired_arm = world_arm_points(desired, self.arm_model)
        actual_arm = world_arm_points(actual, self.arm_model)
        body_error = float(np.max(np.linalg.norm(actual_body - desired_body, axis=1)))
        arm_error = float(np.max(np.linalg.norm(actual_arm - desired_arm, axis=1)))
        desired_body_clearance = float(
            self.safety_world.minimum_distance(desired_body)[0]
        )
        desired_arm_clearance = float(
            self.safety_world.minimum_distance(desired_arm)[0]
            - self.arm_model.radius
        )
        self.phase_body_budget = clearance_tracking_budget(
            desired_body_clearance, self.actual_body_abort_clearance,
            self.phase_body_clearance_reserve, self.phase_body_stop_error_max,
        )
        self.phase_arm_budget = clearance_tracking_budget(
            desired_arm_clearance, self.actual_arm_abort_clearance,
            self.phase_arm_clearance_reserve, self.phase_arm_stop_error_max,
        )
        body_scale = clearance_progress_scale(
            body_error, self.phase_body_full_error, self.phase_body_budget,
            self.phase_calibrated_full_speed_error,
        )
        arm_scale = clearance_progress_scale(
            arm_error, self.phase_arm_full_error, self.phase_arm_budget,
            self.phase_calibrated_full_speed_error,
        )
        return min(body_scale, arm_scale), body_error, arm_error

    def _reset_phase(self):
        self.phase_elapsed = 0.0
        self.phase_last_wall = time.monotonic()
        self.phase_rate = 0.0
        self.phase_body_budget = 0.0
        self.phase_arm_budget = 0.0

    def _publish_actual_safety(self, state, body, body_name, arm, arm_name):
        report = {
            "state": state, "stamp": rospy.Time.now().to_sec(),
            "body_clearance": body, "body_obstacle": body_name,
            "body_limit": self.actual_body_abort_clearance,
            "arm_clearance": arm, "arm_obstacle": arm_name,
            "arm_limit": self.actual_arm_abort_clearance,
            "trajectory_id": self.locked_id,
        }
        self.safety_status_pub.publish(String(data=json.dumps(report, sort_keys=True)))
        self.last_safety_status_wall = time.monotonic()

    def _world_to_local_pose(self, world_pose):
        local = PoseStamped()
        local.header.stamp = rospy.Time.now()
        local.header.frame_id = self.local_pose.header.frame_id or "map"
        local.pose.position.x = world_pose.pose.position.x + self.world_to_local[0]
        local.pose.position.y = world_pose.pose.position.y + self.world_to_local[1]
        local.pose.position.z = world_pose.pose.position.z + self.world_to_local[2]
        local.pose.orientation = world_pose.pose.orientation
        return local

    @staticmethod
    def _sample(message, elapsed):
        lower, upper, ratio = interpolation_bracket(message.time_from_start, elapsed)
        first, second = message.uav_poses[lower], message.uav_poses[upper]
        world = PoseStamped()
        world.header.stamp = rospy.Time.now()
        world.header.frame_id = message.header.frame_id or "world"
        world.pose.position.x = lerp(first.position.x, second.position.x, ratio)
        world.pose.position.y = lerp(first.position.y, second.position.y, ratio)
        world.pose.position.z = lerp(first.position.z, second.position.z, ratio)
        yaw = lerp_angle(yaw_from_quaternion(first.orientation), yaw_from_quaternion(second.orientation), ratio)
        quaternion_from_yaw(yaw, world.pose.orientation)
        theta = lerp(message.theta_deg[lower], message.theta_deg[upper], ratio)
        phi = lerp(message.phi_deg[lower], message.phi_deg[upper], ratio)
        first_velocity = message.linear_velocities[lower]
        second_velocity = message.linear_velocities[upper]
        velocity = np.asarray([
            lerp(first_velocity.x, second_velocity.x, ratio),
            lerp(first_velocity.y, second_velocity.y, ratio),
            lerp(first_velocity.z, second_velocity.z, ratio),
        ], dtype=float)
        yaw_rate = lerp(message.yaw_rates[lower], message.yaw_rates[upper], ratio)
        return world, theta, phi, velocity, yaw_rate, lower, upper

    @staticmethod
    def _sample_shape_rates(message, elapsed):
        lower, upper, ratio = interpolation_bracket(message.time_from_start, elapsed)
        return (
            lerp(message.theta_rate_deg_s[lower], message.theta_rate_deg_s[upper], ratio),
            lerp(message.phi_rate_deg_s[lower], message.phi_rate_deg_s[upper], ratio),
        )

    @staticmethod
    def _bridge_coordinates(state):
        state = np.asarray(state, dtype=float)
        bend = bending_vector(state[4], state[5])
        return np.asarray([
            state[0], state[1], state[2], state[3], bend[0], bend[1]
        ], dtype=float)

    @staticmethod
    def _bridge_rate(state, rate):
        state = np.asarray(state, dtype=float)
        rate = np.asarray(rate, dtype=float)
        bend_rate = bending_vector_rate(
            state[4], state[5], rate[4], rate[5]
        )
        return np.asarray([
            rate[0], rate[1], rate[2], rate[3],
            bend_rate[0], bend_rate[1],
        ], dtype=float)

    @staticmethod
    def _bridge_state(coordinates, coordinate_rate=None):
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinate_rate is None:
            theta, phi = bending_state_from_vector(coordinates[4:6])
            return np.asarray([
                coordinates[0], coordinates[1], coordinates[2],
                coordinates[3], theta, phi,
            ], dtype=float)
        theta, phi, theta_rate, phi_rate = bending_state_from_vector(
            coordinates[4:6], np.asarray(coordinate_rate, dtype=float)[4:6]
        )
        return (
            np.asarray([
                coordinates[0], coordinates[1], coordinates[2],
                coordinates[3], theta, phi,
            ], dtype=float),
            np.asarray([
                coordinate_rate[0], coordinate_rate[1], coordinate_rate[2],
                coordinate_rate[3], theta_rate, phi_rate,
            ], dtype=float),
        )

    def _prepare_switch_bridge(self, message, matched_elapsed,
                               preserved_phase_rate, pending_accelerations):
        """Construct and independently validate a short C2 handoff bridge."""
        if not self.enable_switch_bridge:
            return None, {"enabled": False, "status": "DISABLED"}
        duration = self.switch_bridge_duration
        trajectory_duration = float(message.time_from_start[-1])
        # Hold the successor trajectory phase while the C2 bridge is being
        # executed.  Advancing the target by ``duration * phase_rate`` makes
        # the bridge chase a moving point and needlessly increases jerk.  The
        # successor resumes from the matched phase after the validated bridge
        # reaches it.
        target_elapsed = min(trajectory_duration, float(matched_elapsed))
        (target_pose, target_theta_deg, target_phi_deg, target_velocity,
         target_yaw_rate, _lower, _upper) = self._sample(message, target_elapsed)
        target_theta_rate_deg_s, target_phi_rate_deg_s = self._sample_shape_rates(
            message, target_elapsed
        )
        target_state = np.asarray([
            target_pose.pose.position.x, target_pose.pose.position.y,
            target_pose.pose.position.z,
            yaw_from_quaternion(target_pose.pose.orientation),
            math.radians(target_theta_deg), math.radians(target_phi_deg),
        ], dtype=float)
        target_state[3] = self.last_commanded_switch_state[3] + wrap_angle(
            target_state[3] - self.last_commanded_switch_state[3]
        )
        target_rate = np.asarray([
            target_velocity[0] * preserved_phase_rate,
            target_velocity[1] * preserved_phase_rate,
            target_velocity[2] * preserved_phase_rate,
            target_yaw_rate * preserved_phase_rate,
            math.radians(target_theta_rate_deg_s) * preserved_phase_rate,
            math.radians(target_phi_rate_deg_s) * preserved_phase_rate,
        ], dtype=float)
        pending_times = np.asarray(message.time_from_start, dtype=float)
        target_acceleration = np.asarray([
            np.interp(target_elapsed, pending_times, pending_accelerations[:, axis])
            for axis in range(3)
        ], dtype=float)
        start_coordinates = self._bridge_coordinates(
            self.last_commanded_switch_state
        )
        finish_coordinates = self._bridge_coordinates(target_state)
        start_rate = self._bridge_rate(
            self.last_commanded_switch_state, self.last_commanded_switch_rate
        )
        finish_rate = self._bridge_rate(target_state, target_rate)
        start_acceleration = np.r_[
            self.last_commanded_switch_acceleration, np.zeros(3)
        ]
        finish_acceleration = np.r_[target_acceleration, np.zeros(3)]
        coefficients = quintic_bridge_coefficients(
            start_coordinates, start_rate, start_acceleration,
            finish_coordinates, finish_rate, finish_acceleration, duration,
        )
        count = max(3, int(math.ceil(duration / self.switch_bridge_validation_dt)) + 1)
        times = np.linspace(0.0, duration, count)
        coordinates = evaluate_quintic_bridge(coefficients, times, 0)
        rates = evaluate_quintic_bridge(coefficients, times, 1)
        accelerations = evaluate_quintic_bridge(coefficients, times, 2)
        jerks = evaluate_quintic_bridge(coefficients, times, 3)
        body_min = float("inf")
        arm_min = float("inf")
        theta_rates = []
        phi_rates = []
        states = []
        for coordinate, rate in zip(coordinates, rates):
            state, state_rate = self._bridge_state(coordinate, rate)
            states.append(state)
            theta_rates.append(state_rate[4])
            # Phi is physically undefined for a straight arm; the bending
            # vector remains smooth there and is the quantity constrained.
            phi_rates.append(0.0 if state[4] < math.radians(1.0) else state_rate[5])
            body_min = min(body_min, float(self.safety_world.minimum_distance(
                world_body_points(
                    state, self.safety_limits,
                    azimuth_samples=32, vertical_samples=5,
                )
            )[0]))
            arm_min = min(arm_min, float(self.safety_world.minimum_distance(
                world_arm_points(state, self.arm_model)
            )[0] - self.arm_model.radius))
        states = np.asarray(states, dtype=float)
        speed_max = float(np.max(np.linalg.norm(rates[:, :3], axis=1)))
        acceleration_max = float(np.max(np.linalg.norm(
            accelerations[:, :3], axis=1
        )))
        jerk_max = float(np.max(np.linalg.norm(jerks[:, :3], axis=1)))
        yaw_rate_max = float(np.max(np.abs(rates[:, 3])))
        theta_rate_max = float(np.max(np.abs(theta_rates)))
        phi_rate_max = float(np.max(np.abs(phi_rates)))
        limits = {
            "velocity": float(rospy.get_param(
                "/planner_v2/limits/velocity_max", self.safety_limits.velocity_max
            )),
            "acceleration": float(rospy.get_param(
                "/planner_v2/limits/acceleration_max", self.safety_limits.acceleration_max
            )),
            "jerk": float(rospy.get_param(
                "/planner_v2/limits/jerk_max", self.safety_limits.jerk_max
            )),
            "yaw_rate": math.radians(float(rospy.get_param(
                "/planner_v2/limits/yaw_rate_max_deg",
                math.degrees(self.safety_limits.yaw_rate_max),
            ))),
            "theta_rate": math.radians(float(rospy.get_param(
                "/planner_v2/limits/theta_rate_max_deg",
                math.degrees(self.safety_limits.theta_rate_max),
            ))),
            "phi_rate": math.radians(float(rospy.get_param(
                "/planner_v2/limits/phi_rate_max_deg",
                math.degrees(self.safety_limits.phi_rate_max),
            ))),
        }
        failures = []
        if body_min < self.actual_body_abort_clearance:
            failures.append("body_clearance")
        if arm_min < self.actual_arm_abort_clearance:
            failures.append("arm_clearance")
        if float(np.min(states[:, 2])) < self.safety_limits.z_min:
            failures.append("z_min")
        if float(np.max(states[:, 2])) > self.safety_limits.z_max:
            failures.append("z_max")
        for name, value in (
                ("velocity", speed_max), ("acceleration", acceleration_max),
                ("jerk", jerk_max), ("yaw_rate", yaw_rate_max),
                ("theta_rate", theta_rate_max), ("phi_rate", phi_rate_max)):
            if value > limits[name] + 1e-9:
                failures.append(name)
        report = {
            "enabled": True, "status": "SAFE" if not failures else "REJECTED",
            "duration_s": duration, "target_elapsed": target_elapsed,
            "phase_policy": "hold_successor_phase_during_bridge",
            "samples": count, "body_clearance_min": body_min,
            "arm_clearance_min": arm_min, "velocity_max": speed_max,
            "acceleration_max": acceleration_max, "jerk_max": jerk_max,
            "yaw_rate_max": yaw_rate_max, "theta_rate_max": theta_rate_max,
            "phi_rate_max": phi_rate_max, "limits": limits,
            "failures": failures,
        }
        if failures:
            return None, report
        return {
            "coefficients": coefficients,
            "duration": duration,
            "start_wall": time.monotonic(),
            "start_phase": float(matched_elapsed),
            "target_phase": target_elapsed,
            "report": report,
        }, report

    def _sample_switch_bridge(self, wall_now):
        if self.switch_bridge is None:
            return None
        bridge = self.switch_bridge
        elapsed = min(
            bridge["duration"], max(0.0, wall_now - bridge["start_wall"])
        )
        coordinate = evaluate_quintic_bridge(
            bridge["coefficients"], [elapsed], 0
        )[0]
        coordinate_rate = evaluate_quintic_bridge(
            bridge["coefficients"], [elapsed], 1
        )[0]
        state, rate = self._bridge_state(coordinate, coordinate_rate)
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "world"
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = state[:3]
        quaternion_from_yaw(state[3], pose.pose.orientation)
        ratio = elapsed / max(bridge["duration"], 1e-9)
        self.phase_elapsed = lerp(
            bridge["start_phase"], bridge["target_phase"], ratio
        )
        completed = elapsed >= bridge["duration"] - 1e-9
        if completed:
            self.phase_elapsed = bridge["target_phase"]
            self.switch_bridge = None
        return (
            pose, math.degrees(state[4]), math.degrees(state[5]),
            rate[:3], rate[3], 0, 0,
            math.degrees(rate[4]), math.degrees(rate[5]),
        )

    def _publish_trajectory_setpoint(self, desired_world, velocity, yaw_rate,
                                     allow_feedforward=True):
        """Publish one PX4 position reference with trajectory feed-forward.

        MAVROS accepts ROS ENU values on ``setpoint_raw/local`` and performs
        the MAVLink NED conversion internally.  Translation between the
        Gazebo world and PX4 local origins affects position only, not the
        velocity vector.
        """
        local = self._world_to_local_pose(desired_world)
        if not self.use_velocity_feedforward or not allow_feedforward:
            self.setpoint_pub.publish(local)
            return
        target = PositionTarget()
        target.header.stamp = rospy.Time.now()
        target.header.frame_id = local.header.frame_id
        target.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        target.type_mask = (
            PositionTarget.IGNORE_AFX
            | PositionTarget.IGNORE_AFY
            | PositionTarget.IGNORE_AFZ
        )
        target.position = local.pose.position
        target.velocity.x = float(velocity[0])
        target.velocity.y = float(velocity[1])
        target.velocity.z = float(velocity[2])
        target.yaw = float(yaw_from_quaternion(local.pose.orientation))
        target.yaw_rate = float(yaw_rate)
        self.raw_setpoint_pub.publish(target)

    def _start_first(self, now):
        missing = self._inputs_ready()
        if missing:
            self._status("ENABLE_BLOCKED", missing=missing)
            return False
        if self.pending is None:
            self._status("WAITING_TRAJECTORY")
            return False
        start_gap = position_distance(self.pending.uav_poses[0].position, self.world_pose.pose.position)
        if start_gap > self.maximum_start_gap:
            if self.pending_wait_started_wall is None:
                self.pending_wait_started_wall = time.monotonic()
            wait_time = time.monotonic() - self.pending_wait_started_wall
            if wait_time <= self.rolling_switch_wait_timeout:
                if time.monotonic() - self.last_continuity_status_wall >= 0.5:
                    self.last_continuity_status_wall = time.monotonic()
                    self._status(
                        "WAITING_START_CONTINUITY", pending_id=int(self.pending.trajectory_id),
                        start_gap=start_gap, limit=self.maximum_start_gap,
                        wait_time=wait_time,
                    )
                return False
            rejected_id = int(self.pending.trajectory_id)
            self._status("REJECTED_START_DISCONTINUITY", pending_id=int(self.pending.trajectory_id),
                         start_gap=start_gap, limit=self.maximum_start_gap,
                         wait_time=wait_time)
            self.pending = None
            self.pending_wait_started_wall = None
            # Stay enabled and keep publishing the captured handoff hold.
            # The planner consumes this rejection, discards the stale buffer,
            # and produces a fresh measured-state trajectory.  Disabling here
            # used to require a second manual enable and, more importantly,
            # could leave PX4 without a setpoint owner during the retry.
            self._feedback("REJECTED_START_DISCONTINUITY", start_gap=start_gap,
                           limit=self.maximum_start_gap, trajectory_id=rejected_id)
            return False
        self._capture_transform_and_hold()
        self.trajectory = self.pending
        self.pending = None
        self.pending_wait_started_wall = None
        self.locked_id = int(self.trajectory.trajectory_id)
        self.completion_reported_id = None
        self.final_started_wall = None
        self.active_start = now + rospy.Duration(self.handoff_lead)
        self._reset_phase()
        self.bad_tracking_since = None
        self.first_active_trajectory = True
        self.governed_state = None
        self.last_commanded_switch_state = None
        self.last_commanded_switch_rate = None
        self.last_commanded_switch_acceleration = None
        self.last_commanded_switch_stamp = None
        self.switch_bridge = None
        self._status("START_SCHEDULED", actual_start=self.active_start.to_sec(), start_gap=start_gap,
                     world_to_local=self.world_to_local)
        self._feedback("START_SCHEDULED", actual_start=self.active_start.to_sec())
        return True

    def _try_rolling_switch(self, now, current_desired, active_final=False):
        if self.pending is None or self.switch_bridge is not None:
            return
        if active_final:
            # If online prediction was abandoned because tracking lagged, the
            # replacement window starts at a measured state and is not
            # continuous with the old desired curve.  At the old safe endpoint
            # it may be handed off against the actual vehicle state, with a
            # fresh clock, instead of being discarded as an expired window.
            start_gap = position_distance(
                self.pending.uav_poses[0].position, self.world_pose.pose.position
            )
            if start_gap > self.maximum_start_gap:
                lateness = (now - self.pending.execution_start_time).to_sec()
                if lateness >= self.pending_start_expiry:
                    rejected_id = int(self.pending.trajectory_id)
                    self.pending = None
                    self.pending_wait_started_wall = None
                    self._status(
                        "REJECTED_START_DISCONTINUITY", trajectory_id=rejected_id,
                        start_gap=start_gap, limit=self.maximum_start_gap,
                        lateness=lateness, active_trajectory_id=self.locked_id,
                    )
                    self._feedback(
                        "REJECTED_START_DISCONTINUITY", trajectory_id=rejected_id,
                        start_gap=start_gap, limit=self.maximum_start_gap,
                        lateness=lateness, active_trajectory_id=self.locked_id,
                    )
                return
            self.trajectory = self.pending
            self.pending = None
            self.pending_wait_started_wall = None
            self.locked_id = int(self.trajectory.trajectory_id)
            self.completion_reported_id = None
            self.final_started_wall = None
            self.active_start = now + rospy.Duration(self.handoff_lead)
            self._reset_phase()
            self.switch_bridge = None
            self.bad_tracking_since = None
            self.first_active_trajectory = False
            self._status(
                "SWITCHED_AFTER_FINAL_HOLD", actual_start=self.active_start.to_sec(),
                start_gap=start_gap,
            )
            self._feedback("SWITCHED_AFTER_FINAL_HOLD", actual_start=self.active_start.to_sec())
            return
        if now < self.pending.execution_start_time:
            return
        pending_positions = [
            (pose.position.x, pose.position.y, pose.position.z)
            for pose in self.pending.uav_poses
        ]
        desired_position = (
            current_desired.pose.position.x, current_desired.pose.position.y,
            current_desired.pose.position.z,
        )
        measured_position = (
            self.world_pose.pose.position.x, self.world_pose.pose.position.y,
            self.world_pose.pose.position.z,
        )
        preserved_phase_rate = continuous_switch_phase_rate(self.phase_rate)
        continuity = {"switch_continuity_diagnostics_version": 2}
        match_score = None
        gate_violations = {}
        switch_bridge = None
        switch_bridge_report = {"enabled": False, "status": "UNAVAILABLE"}
        if (self.last_commanded_switch_state is not None
                and self.last_commanded_switch_rate is not None
                and self.last_commanded_switch_acceleration is not None):
            pending_states = []
            pending_rates = []
            for index, pose in enumerate(self.pending.uav_poses):
                pending_states.append([
                    pose.position.x, pose.position.y, pose.position.z,
                    yaw_from_quaternion(pose.orientation),
                    math.radians(self.pending.theta_deg[index]),
                    math.radians(self.pending.phi_deg[index]),
                ])
                velocity = self.pending.linear_velocities[index]
                pending_rates.append([
                    velocity.x * preserved_phase_rate,
                    velocity.y * preserved_phase_rate,
                    velocity.z * preserved_phase_rate,
                    self.pending.yaw_rates[index] * preserved_phase_rate,
                    math.radians(self.pending.theta_rate_deg_s[index])
                    * preserved_phase_rate,
                    math.radians(self.pending.phi_rate_deg_s[index])
                    * preserved_phase_rate,
                ])
            pending_times = np.asarray(self.pending.time_from_start, dtype=float)
            pending_raw_linear_rates = np.asarray([
                [velocity.x, velocity.y, velocity.z]
                for velocity in self.pending.linear_velocities
            ], dtype=float)
            edge_order = 2 if len(pending_times) >= 3 else 1
            pending_accelerations = np.gradient(
                pending_raw_linear_rates, pending_times, axis=0,
                edge_order=edge_order,
            ) * preserved_phase_rate ** 2
            (matched_elapsed, matched_index, matched_advance,
             command_gap, measured_gap, match_score,
             continuity_metrics) = bounded_whole_body_switch_match(
                self.pending.time_from_start, pending_states, pending_rates,
                self.last_commanded_switch_state,
                self.last_commanded_switch_rate,
                measured_position, self.rolling_switch_max_advance,
                scales=self.switch_match_scales,
                accelerations=pending_accelerations,
                old_acceleration=self.last_commanded_switch_acceleration,
            )
            matched_gaps = (command_gap, measured_gap)
            new_state = np.asarray(pending_states[matched_index], dtype=float)
            new_rate = np.asarray(pending_rates[matched_index], dtype=float)
            continuity.update(continuity_metrics)
            continuity.update({
                "switch_match_mode": "whole_body_c0_c1_minimax",
                "old_command_state": self.last_commanded_switch_state.tolist(),
                "new_candidate_state": new_state.tolist(),
                "old_command_rate": self.last_commanded_switch_rate.tolist(),
                "new_candidate_rate": new_rate.tolist(),
                "old_command_acceleration": self.last_commanded_switch_acceleration.tolist(),
                "new_candidate_acceleration": pending_accelerations[matched_index].tolist(),
            })
            gate_violations = whole_body_switch_gate(
                continuity_metrics, self.switch_gate_limits
            )
        else:
            (matched_elapsed, matched_index, matched_advance,
             matched_gaps) = bounded_switch_match(
                self.pending.time_from_start, pending_positions,
                (desired_position, measured_position), self.rolling_switch_max_advance,
            )
            continuity.update({
                "switch_match_mode": "position_fallback",
                "switch_continuity_diagnostics_unavailable": True,
            })
        switch_gap = max(matched_gaps)
        raw_gate_violations = dict(gate_violations)
        direct_c1_authorized = (
            switch_gap <= self.rolling_switch_target_gap
            and not gate_violations
        )
        if direct_c1_authorized:
            switch_bridge_report = {
                "enabled": self.enable_switch_bridge,
                "status": "NOT_REQUIRED_DIRECT_C1",
            }
        elif self.last_commanded_switch_acceleration is not None:
            switch_bridge, switch_bridge_report = self._prepare_switch_bridge(
                self.pending, matched_elapsed, preserved_phase_rate,
                pending_accelerations,
            )
            if (self.enable_switch_bridge and switch_bridge is None):
                gate_violations["switch_bridge_validation"] = {
                    "status": switch_bridge_report.get("status"),
                    "failures": switch_bridge_report.get("failures", []),
                }
        bridge_authorized = switch_bridge is not None
        if bridge_authorized:
            # Direct C1 matching failed.  The executed fallback splice starts
            # exactly at the old command and ends C2-matched at the successor
            # state.  Raw point-to-point gaps remain diagnostics; the
            # independently validated bridge is the handoff authority.
            gate_violations = {}
        if (not bridge_authorized
                and (switch_gap > self.rolling_switch_target_gap
                     or gate_violations)):
            if self.pending_wait_started_wall is None:
                self.pending_wait_started_wall = time.monotonic()
            wait_time = time.monotonic() - self.pending_wait_started_wall
            if wait_time <= self.rolling_switch_wait_timeout:
                if time.monotonic() - self.last_continuity_status_wall >= 0.5:
                    self.last_continuity_status_wall = time.monotonic()
                    self._status(
                        "WAITING_ROLLING_CONTINUITY", switch_gap=switch_gap,
                        target_gap=self.rolling_switch_target_gap,
                        matched_index=matched_index,
                        matched_advance=matched_advance,
                        command_gap=matched_gaps[0], measured_gap=matched_gaps[1],
                        switch_match_score=match_score,
                        switch_gate_violations=gate_violations,
                        raw_switch_gate_violations=raw_gate_violations,
                        switch_gate_limits=self.switch_gate_limits,
                        switch_bridge=switch_bridge_report,
                        wait_time=wait_time, pending_id=int(self.pending.trajectory_id),
                    )
                return
            # A timeout is evidence that the prediction has gone stale.  Do
            # not force a discontinuous handoff merely because it remains
            # just inside a loose emergency bound; reject it so the planner
            # can solve again from the latest executor/measured state.
            rejected = int(self.pending.trajectory_id)
            self.pending = None
            self.pending_wait_started_wall = None
            self._status("REJECTED_ROLLING_DISCONTINUITY", rejected_id=rejected,
                         switch_gap=switch_gap,
                         limit=self.rolling_switch_target_gap,
                         matched_index=matched_index,
                         matched_advance=matched_advance,
                         switch_match_score=match_score,
                         switch_gate_violations=gate_violations,
                         raw_switch_gate_violations=raw_gate_violations,
                         switch_gate_limits=self.switch_gate_limits,
                         switch_bridge=switch_bridge_report)
            self._feedback("REJECTED_ROLLING_DISCONTINUITY",
                           trajectory_id=rejected, switch_gap=switch_gap,
                           limit=self.rolling_switch_target_gap,
                           switch_gate_violations=gate_violations,
                           raw_switch_gate_violations=raw_gate_violations,
                           switch_gate_limits=self.switch_gate_limits,
                           switch_bridge=switch_bridge_report,
                           active_trajectory_id=self.locked_id)
            return
        self.trajectory = self.pending
        self.pending = None
        self.pending_wait_started_wall = None
        self.locked_id = int(self.trajectory.trajectory_id)
        self.completion_reported_id = None
        self.final_started_wall = None
        self.active_start = now
        self._reset_phase()
        self.phase_elapsed = matched_elapsed
        self.phase_rate = preserved_phase_rate
        self.last_switch_phase_rate = preserved_phase_rate
        self.switch_bridge = switch_bridge
        self.bad_tracking_since = None
        self.first_active_trajectory = False
        self._status(
            "SWITCHED_ROLLING", actual_start=self.active_start.to_sec(),
            switch_gap=switch_gap, matched_index=matched_index,
            matched_elapsed=matched_elapsed, matched_advance=matched_advance,
            command_gap=matched_gaps[0], measured_gap=matched_gaps[1],
            preserved_phase_rate=preserved_phase_rate,
            switch_match_score=match_score,
            switch_gate_violations=gate_violations,
            raw_switch_gate_violations=raw_gate_violations,
            switch_gate_limits=self.switch_gate_limits,
            switch_bridge=switch_bridge_report,
            **continuity,
        )
        self._feedback(
            "SWITCHED_ROLLING", actual_start=self.active_start.to_sec(),
            matched_elapsed=matched_elapsed, matched_advance=matched_advance,
            preserved_phase_rate=preserved_phase_rate,
            switch_match_score=match_score,
            switch_gate_violations=gate_violations,
            raw_switch_gate_violations=raw_gate_violations,
            switch_gate_limits=self.switch_gate_limits,
            switch_bridge=switch_bridge_report,
            **continuity,
        )

    def _abort(self, reason, disable=True, **fields):
        if self.local_pose is not None:
            self.hold_local = self._copy_pose(self.local_pose, self.local_pose.header.frame_id or "map")
        if self.world_pose is not None:
            self.hold_world = self._copy_pose(self.world_pose, "world")
            self.hold_arm_state = list(self.arm_state)
            fields["hold_pose"] = [
                self.hold_world.pose.position.x, self.hold_world.pose.position.y,
                self.hold_world.pose.position.z,
            ]
        self.trajectory = None
        self.last_commanded_switch_state = None
        self.last_commanded_switch_rate = None
        self.last_commanded_switch_acceleration = None
        self.last_commanded_switch_stamp = None
        self.switch_bridge = None
        # A disabled/aborted run must not retain a future window from the old
        # planner instance.  Otherwise a restarted planner can only receive
        # REJECTED_PENDING_EXISTS and has no way to establish a fresh,
        # measured-state handoff.
        self.pending = None
        self.pending_wait_started_wall = None
        self.active_start = None
        self._reset_phase()
        self.governed_state = None
        self.bad_tracking_since = None
        self.final_started_wall = None
        if disable:
            self.enabled = False
        self._status(reason, **fields)
        self._feedback(reason, **fields)

    def _publish_hold(self):
        if self.hold_local is None:
            return
        self.hold_local.header.stamp = rospy.Time.now()
        self.setpoint_pub.publish(self.hold_local)
        self.arm_pub.publish(Float32MultiArray(data=list(self.hold_arm_state)))

    def _refresh_owned_world_hold(self):
        """Keep an aborted vehicle fixed in world, despite PX4 local-frame drift."""
        if self.hold_world is None or self.world_pose is None or self.local_pose is None:
            return
        self._update_world_to_local_transform()
        self.hold_local = self._world_to_local_pose(self.hold_world)
        body, body_name, arm, arm_name = self._actual_clearance()
        state = (
            "UNSAFE_HOLD" if body < self.actual_body_abort_clearance
            or arm < self.actual_arm_abort_clearance else "SAFE_HOLD"
        )
        if time.monotonic() - self.last_safety_status_wall >= self.safety_status_period:
            self._publish_actual_safety(state, body, body_name, arm, arm_name)

    def _record_history(self):
        now_wall = time.monotonic()
        if self.world_pose is None or now_wall - self.last_history_wall < self.history_period:
            return
        pose = self._copy_pose(self.world_pose, "world")
        self.executed_path.header.stamp = pose.header.stamp
        self.executed_path.poses.append(pose)
        if len(self.executed_path.poses) > 10000:
            self.executed_path.poses = self.executed_path.poses[-10000:]
        self.executed_path_pub.publish(self.executed_path)
        self.last_history_wall = now_wall

    def _tick(self, _event):
        now = rospy.Time.now()
        if not self.enabled:
            if self.publish_hold_when_disabled or self.owns_setpoint:
                if self.owns_setpoint:
                    self._refresh_owned_world_hold()
                self._publish_hold()
            if time.monotonic() - self.last_status_wall > 2.0:
                self._status("HOLDING_DISABLED" if self.hold_local is not None else "WAITING_INPUT")
            return

        if self.trajectory is None and not self._start_first(now):
            if self.publish_hold_when_disabled or self.owns_setpoint:
                self._publish_hold()
            return

        missing = self._inputs_ready()
        if missing:
            self._abort("ABORT_INPUT_OR_PX4_STATE", missing=missing)
            self._publish_hold()
            return

        self._update_world_to_local_transform()
        body_clearance, body_name, arm_clearance, arm_name = self._actual_clearance()
        if (body_clearance < self.actual_body_abort_clearance
                or arm_clearance < self.actual_arm_abort_clearance):
            self._publish_actual_safety(
                "UNSAFE", body_clearance, body_name, arm_clearance, arm_name
            )
            self._abort(
                "ABORT_ACTUAL_CLEARANCE", body_clearance=body_clearance,
                body_obstacle=body_name, body_limit=self.actual_body_abort_clearance,
                arm_clearance=arm_clearance, arm_obstacle=arm_name,
                arm_limit=self.actual_arm_abort_clearance,
            )
            self._publish_hold()
            return
        if (body_clearance >= self.recovery_body_clearance
                and arm_clearance >= self.recovery_arm_clearance):
            self.last_recovery_world = self._copy_pose(self.world_pose, "world")
            self.last_recovery_arm_state = list(self.arm_state)
        if time.monotonic() - self.last_safety_status_wall >= self.safety_status_period:
            self._publish_actual_safety(
                "SAFE", body_clearance, body_name, arm_clearance, arm_name
            )

        wall_now = time.monotonic()
        before_start = now < self.active_start
        bridge_shape_rates_deg_s = None
        nominal_world, nominal_theta, nominal_phi, nominal_velocity, nominal_yaw_rate, lower, upper = self._sample(
            self.trajectory, self.phase_elapsed
        )
        if self.switch_bridge is not None:
            before_start = False
            self.phase_last_wall = wall_now
            phase_body_error = 0.0
            phase_arm_error = 0.0
            bridge_sample = self._sample_switch_bridge(wall_now)
            (nominal_world, nominal_theta, nominal_phi, nominal_velocity,
             nominal_yaw_rate, lower, upper, bridge_theta_rate,
             bridge_phi_rate) = bridge_sample
            bridge_shape_rates_deg_s = (bridge_theta_rate, bridge_phi_rate)
        elif before_start:
            self.phase_last_wall = wall_now
            self.phase_rate = 0.0
            phase_body_error = 0.0
            phase_arm_error = 0.0
        else:
            phase_dt = max(0.0, min(0.10, wall_now - self.phase_last_wall))
            self.phase_last_wall = wall_now
            (self.phase_rate, phase_body_error,
             phase_arm_error) = self._phase_guard(
                nominal_world, nominal_theta, nominal_phi
            )
            duration = float(self.trajectory.time_from_start[-1])
            self.phase_elapsed = min(
                duration, self.phase_elapsed + phase_dt * self.phase_rate
            )
            nominal_world, nominal_theta, nominal_phi, nominal_velocity, nominal_yaw_rate, lower, upper = self._sample(
                self.trajectory, self.phase_elapsed
            )
        elapsed = self.phase_elapsed
        switch_reference = nominal_world
        if self.governed_state is not None:
            switch_reference = PoseStamped()
            switch_reference.header.stamp = now
            switch_reference.header.frame_id = "world"
            switch_reference.pose.position.x = float(self.governed_state[0])
            switch_reference.pose.position.y = float(self.governed_state[1])
            switch_reference.pose.position.z = float(self.governed_state[2])
            quaternion_from_yaw(
                float(self.governed_state[3]), switch_reference.pose.orientation
            )
        active_final = elapsed >= float(self.trajectory.time_from_start[-1])
        if self.switch_bridge is not None:
            active_final = False
        previous_identifier = self.locked_id
        self._try_rolling_switch(now, switch_reference, active_final=active_final)
        if self.trajectory is None:
            self._publish_hold()
            return
        if self.locked_id != previous_identifier:
            bridge_sample = self._sample_switch_bridge(wall_now)
            if bridge_sample is not None:
                (nominal_world, nominal_theta, nominal_phi, nominal_velocity,
                 nominal_yaw_rate, lower, upper, bridge_theta_rate,
                 bridge_phi_rate) = bridge_sample
                bridge_shape_rates_deg_s = (
                    bridge_theta_rate, bridge_phi_rate
                )
            else:
                nominal_world, nominal_theta, nominal_phi, nominal_velocity, nominal_yaw_rate, lower, upper = self._sample(
                    self.trajectory, self.phase_elapsed
                )
            elapsed = self.phase_elapsed
            # A normal rolling splice already preserved the old phase rate in
            # _try_rolling_switch. Only a scheduled restart remains at zero.
            if now < self.active_start:
                self.phase_rate = continuous_switch_phase_rate(
                    self.phase_rate, scheduled_restart=True
                )
            phase_body_error = 0.0
            phase_arm_error = 0.0
        bridge_command_active = bridge_shape_rates_deg_s is not None
        reference_rate_scale = 1.0 if bridge_command_active else self.phase_rate
        (desired_world, theta, phi, velocity, yaw_rate, tube_alpha,
         tube_body, tube_arm, stopping_reserve) = self._govern_reference(
            nominal_world, nominal_theta, nominal_phi,
            nominal_velocity * reference_rate_scale,
            nominal_yaw_rate * reference_rate_scale,
            body_clearance, arm_clearance,
        )
        governor_alpha = reference_rate_scale * tube_alpha
        tracking_error = position_distance(desired_world.pose.position, self.world_pose.pose.position)
        nominal_tracking_error = position_distance(
            nominal_world.pose.position, self.world_pose.pose.position
        )
        final = elapsed >= float(self.trajectory.time_from_start[-1])

        # A sampled spline endpoint may have a non-zero one-sided derivative.
        # Once no phase remains, publishing that derivative next to a fixed
        # position makes PX4 fly through the endpoint and pull back. The final
        # whole-body task state is stationary by definition.
        velocity, yaw_rate = stationary_endpoint_feedforward(
            final, velocity, yaw_rate
        )
        velocity = np.asarray(velocity, dtype=float)

        grace = self.first_tracking_grace if self.first_active_trajectory else self.tracking_grace
        if (not final and elapsed >= grace
                and tracking_error > self.maximum_tracking_error):
            if self.bad_tracking_since is None:
                self.bad_tracking_since = time.monotonic()
            elif time.monotonic() - self.bad_tracking_since >= self.tracking_error_hold:
                self._abort("ABORT_TRACKING_ERROR", tracking_error=tracking_error,
                            limit=self.maximum_tracking_error)
                self._publish_hold()
                return
        else:
            self.bad_tracking_since = None

        if final:
            if self.final_started_wall is None:
                self.final_started_wall = time.monotonic()
            final_wait = time.monotonic() - self.final_started_wall
            if (self.completion_reported_id != self.locked_id
                    and final_wait >= self.final_convergence_timeout
                    and tracking_error > self.completion_tracking_tolerance):
                self._abort(
                    "ABORT_FINAL_CONVERGENCE", tracking_error=tracking_error,
                    tolerance=self.completion_tracking_tolerance,
                    timeout=self.final_convergence_timeout,
                )
                self._publish_hold()
                return
        else:
            self.final_started_wall = None

        # Phase scaling preserves the derivative relationship: the setpoint
        # is sampled at phase time and its feed-forward is multiplied by the
        # same d(phase)/dt.  A frozen phase therefore publishes a fixed
        # position with zero feed-forward instead of a moving wall-clock goal.
        self._publish_trajectory_setpoint(
            desired_world, velocity, yaw_rate,
            allow_feedforward=(not self.enable_tracking_tube or tube_alpha >= 0.999),
        )
        self.arm_pub.publish(Float32MultiArray(data=[theta, phi]))
        self.commanded_world_pub.publish(desired_world)
        commanded_twist = TwistStamped()
        commanded_twist.header.stamp = desired_world.header.stamp
        commanded_twist.header.frame_id = "world"
        commanded_twist.twist.linear.x = float(velocity[0])
        commanded_twist.twist.linear.y = float(velocity[1])
        commanded_twist.twist.linear.z = float(velocity[2])
        commanded_twist.twist.angular.z = float(yaw_rate)
        self.commanded_twist_world_pub.publish(commanded_twist)
        if bridge_shape_rates_deg_s is None:
            theta_rate_deg_s, phi_rate_deg_s = self._sample_shape_rates(
                self.trajectory, elapsed
            )
        else:
            theta_rate_deg_s, phi_rate_deg_s = bridge_shape_rates_deg_s
        morphology_rate_scale = 0.0 if final else governor_alpha
        next_switch_state = np.asarray([
            desired_world.pose.position.x,
            desired_world.pose.position.y,
            desired_world.pose.position.z,
            yaw_from_quaternion(desired_world.pose.orientation),
            math.radians(theta), math.radians(phi),
        ], dtype=float)
        next_switch_rate = np.asarray([
            velocity[0], velocity[1], velocity[2], yaw_rate,
            math.radians(theta_rate_deg_s) * morphology_rate_scale,
            math.radians(phi_rate_deg_s) * morphology_rate_scale,
        ], dtype=float)
        command_stamp = desired_world.header.stamp.to_sec()
        if (self.last_commanded_switch_rate is not None
                and self.last_commanded_switch_stamp is not None):
            command_dt = command_stamp - self.last_commanded_switch_stamp
            if command_dt > 1e-4:
                raw_acceleration = (
                    next_switch_rate[:3] - self.last_commanded_switch_rate[:3]
                ) / command_dt
                if self.last_commanded_switch_acceleration is None:
                    self.last_commanded_switch_acceleration = raw_acceleration
                else:
                    # The command timer has small scheduling jitter. A short
                    # causal low-pass preserves real phase changes while
                    # preventing one late callback from dominating C2 match.
                    self.last_commanded_switch_acceleration = (
                        0.25 * raw_acceleration
                        + 0.75 * self.last_commanded_switch_acceleration
                    )
        if self.last_commanded_switch_acceleration is None:
            self.last_commanded_switch_acceleration = np.zeros(3, dtype=float)
        self.last_commanded_switch_state = next_switch_state
        self.last_commanded_switch_rate = next_switch_rate
        self.last_commanded_switch_stamp = command_stamp
        self._record_history()

        reference_has_converged = reference_converged(
            final, governor_alpha, tracking_error,
            self.completion_tracking_tolerance,
        )
        if (final and reference_has_converged
                and self.completion_reported_id != self.locked_id):
            self._feedback(
                "TRAJECTORY_COMPLETE", method=self.trajectory.method,
                final_tracking_error=tracking_error, theta_deg=theta, phi_deg=phi,
            )
            self.completion_reported_id = self.locked_id
        state = (
            ("HOLD_FINAL" if reference_has_converged else "HOLD_FINAL_GOVERNED")
            if final else (
                "WAITING_START" if before_start else
                ("EXECUTING_SYNC" if governor_alpha >= 0.999
                 else "EXECUTING_GOVERNED")
            )
        )
        if time.monotonic() - self.last_status_wall > 0.5:
            self._status(state, elapsed=max(0.0, elapsed), duration=self.trajectory.duration,
                         lower_index=lower, upper_index=upper, tracking_error=tracking_error,
                         nominal_tracking_error=nominal_tracking_error,
                         governor_alpha=governor_alpha,
                         tracking_tube_body_displacement=tube_body,
                         tracking_tube_arm_displacement=tube_arm,
                         tracking_tube_stopping_reserve=stopping_reserve,
                         phase_body_error=phase_body_error,
                         phase_arm_error=phase_arm_error,
                         phase_body_budget=self.phase_body_budget,
                         phase_arm_budget=self.phase_arm_budget,
                         phase_rate=self.phase_rate,
                         measured_speed=self.linear_speed,
                         theta_deg=theta, phi_deg=phi)


if __name__ == "__main__":
    rospy.init_node("whole_body_trajectory_executor_node")
    WholeBodyTrajectoryExecutorNode()
    rospy.spin()
