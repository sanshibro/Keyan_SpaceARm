#!/usr/bin/env python3
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import traceback
from dataclasses import replace

PACKAGE = Path(__file__).resolve().parents[1]
REPOSITORY = PACKAGE.parents[2]
sys.path.insert(0, str(PACKAGE / "src"))
local_deps = REPOSITORY / ".v2_deps" / "root" / "usr" / "lib" / "python3" / "dist-packages"
if local_deps.is_dir():
    sys.path.insert(0, str(local_deps))
    library = REPOSITORY / ".v2_deps" / "root" / "usr" / "lib" / "x86_64-linux-gnu"
    os.environ["LD_LIBRARY_PATH"] = str(library) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")

import numpy as np
import rospy
import yaml
from geometry_msgs.msg import PoseStamped, Vector3
from nav_msgs.msg import Path as RosPath
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray, String
from tf.transformations import quaternion_from_euler

from uav_continuum_whole_body_v2.configuration import load_planner_configuration
from uav_continuum_whole_body_v2.environment import BoxWorld, EsdfGrid, HybridDistanceWorld, PointCloudWorld
from uav_continuum_whole_body_v2.execution import (
    projected_phase_elapsed, trajectory_translation_progress,
)
from uav_continuum_whole_body_v2.geometry import world_arm_points
from uav_continuum_whole_body_v2.horizon import local_horizon, path_to_states, yaw_from_quaternion
from uav_continuum_whole_body_v2.mapping import (
    MapSnapshot, OnlineSnapshotAssembler, validate_map_mode,
)
from uav_continuum_whole_body_v2.msg import WholeBodyTrajectory
from uav_continuum_whole_body_v2.optimizer import WholeBodyOptimizer
from uav_continuum_whole_body_v2.seed_planner import WholeBodySeedPlanner
from uav_continuum_whole_body_v2.types import ArmModel
from uav_continuum_whole_body_v2.validator import SweptVolumeValidator


def nested(data, key, default):
    value = data
    for part in key.split("/"):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


class RollingWholeBodyPlannerNode:
    def __init__(self):
        config_path = Path(rospy.get_param("~planner_config", str(PACKAGE / "config" / "planner_v2.yaml")))
        raw, self.config, self.limits, self.mass = load_planner_configuration(config_path)
        self.arm = ArmModel()
        self.map_mode = validate_map_mode(rospy.get_param("~map_mode", "regression_exact"))
        if self.map_mode == "regression_exact":
            manifest_default = PACKAGE / "config" / "scenario_3c_unified.yaml"
        else:
            manifest_default = PACKAGE / "config" / "scenario_3c_task.yaml"
        manifest_path = Path(rospy.get_param("~task_file", rospy.get_param(
            "~scenario_manifest", str(manifest_default)
        )))
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = yaml.safe_load(handle) or {}
        self.task_target = np.asarray(manifest.get("target", [0.0, 0.0, 0.0])[:3], dtype=float)
        self.map_snapshot = None
        self.online_map_assembler = OnlineSnapshotAssembler()
        if self.map_mode == "regression_exact":
            generated_boxes = Path(rospy.get_param(
                "~boxes_file", str(PACKAGE / "generated" / "scenario_3c_unified_boxes.yaml")
            ))
            truth = BoxWorld.from_yaml_files([generated_boxes])
            self.task_world = BoxWorld.from_dicts(manifest.get("task_boxes", []))
            self.body_keepout_world = BoxWorld.from_dicts(manifest.get("body_keepout_boxes", []))
            esdf_path = Path(rospy.get_param(
                "~esdf_file", str(PACKAGE / "generated" / "scenario_3c_unified_esdf.npz")
            ))
            if esdf_path.exists():
                self.esdf = EsdfGrid.load(esdf_path)
                map_source = "generated_esdf"
            else:
                self.esdf = None
                map_source = "exact_boxes_fallback"
            self.truth_world = HybridDistanceWorld(box_world=truth)
            self.truth_body_world = HybridDistanceWorld(box_world=truth)
            truth_box_count = len(truth.boxes)
        else:
            # Paper modes may read task goals, but never analytic obstacle
            # boxes, generated truth ESDFs, or task-specific keepout boxes.
            self.task_world = BoxWorld([])
            self.body_keepout_world = BoxWorld([])
            self.esdf = None
            self.truth_world = None
            self.truth_body_world = None
            truth_box_count = 0
            if self.map_mode == "frozen_slam":
                snapshot_path = Path(rospy.get_param("~map_snapshot_file", ""))
                if not snapshot_path.is_file():
                    raise FileNotFoundError(
                        "frozen_slam requires ~map_snapshot_file, got %s" % snapshot_path
                    )
                self.map_snapshot = MapSnapshot.load(snapshot_path)
                map_source = self.map_snapshot.source
            else:
                map_source = "waiting_online_slam"
        self.reference = None
        self.pose = None
        self.shape_deg = [0.0, 0.0]
        self.selected_reach_goal = None
        self.selected_reach_shape_deg = None
        self.terminal_trajectory_ids = set()
        self.terminal_execution_finished = False
        self.terminal_convergence_since = None
        self.task_complete = False
        self.executor_enabled = False
        self.executor_pending_id = None
        self.safe_waiting_for_enable = False
        self.busy = False
        self.lock = threading.Lock()
        self.sequence = 0
        self.execution_lead = float(raw.get("execution_lead_time", 0.15))
        self.latency_reserve = float(raw.get("planning_latency_reserve", 10.0))
        self.method_time_budget = float(rospy.get_param("~method_time_budget", 0.0))
        self.maximum_tracking_error = float(raw.get("maximum_tracking_error", 0.35))
        self.maximum_reference_start_gap = float(raw.get("maximum_reference_start_gap", 0.75))
        self.enable_recovery_windows = bool(raw.get("enable_recovery_windows", False))
        self.recovery_after_failures = max(1, int(raw.get("recovery_after_failures", 2)))
        vehicle_scale_horizon = 2.0 * (self.limits.body_radius + self.limits.body_margin)
        self.recovery_horizon_distance = min(
            self.config.horizon_distance,
            max(vehicle_scale_horizon, float(raw.get(
                "recovery_horizon_distance", vehicle_scale_horizon
            ))),
        )
        self.consecutive_plan_failures = 0
        transit_shape_deg = raw.get("transit_reference_shape_deg", [0.0, 0.0])
        if not isinstance(transit_shape_deg, (list, tuple)) or len(transit_shape_deg) < 2:
            raise ValueError("transit_reference_shape_deg must contain theta and phi")
        self.transit_reference_shape = np.radians(np.asarray(
            transit_shape_deg[:2], dtype=float
        ))
        self.failure_slowdown_step = min(
            1.0, max(0.50, float(raw.get("failure_slowdown_step", 0.88)))
        )
        self.failure_slowdown_levels = max(
            0, int(raw.get("failure_slowdown_levels", 2))
        )
        task_tip_tolerance = float(manifest.get(
            "tip_tolerance_m", raw.get("terminal_actual_tip_tolerance", 0.08)
        ))
        self.terminal_actual_tip_tolerance = min(
            task_tip_tolerance,
            float(raw.get("terminal_precision_target", task_tip_tolerance)),
        )
        self.terminal_convergence_dwell = max(
            0.0, float(raw.get("terminal_convergence_dwell", 1.0))
        )
        self.safe_buffer = None
        self.executor_phase_observation = None
        self.trajectory_buffers = {}
        self.point_world = None
        self.point_cloud_inflation = float(raw.get("point_cloud_inflation", 0.02))
        self.point_cloud_max_points = int(raw.get("point_cloud_max_points", 100000))
        self.point_cloud_static_once = bool(raw.get("point_cloud_static_once", False))
        self.point_cloud_local_padding = max(
            0.75, float(raw.get("point_cloud_local_padding", 1.0))
        )
        self.minimum_replan_wall_interval = max(
            0.0, float(raw.get("minimum_replan_wall_interval", 1.0))
        )
        self.commit_horizon = max(
            0.0, float(raw.get("commit_horizon", 0.35))
        )
        self.last_plan_finish_wall = 0.0
        self.trajectory_pub = rospy.Publisher("/planner/v2/whole_body_trajectory", WholeBodyTrajectory, queue_size=1, latch=True)
        self.path_pub = rospy.Publisher("/planner/v2/uav_path", RosPath, queue_size=1, latch=True)
        self.status_pub = rospy.Publisher("/planner/v2/status", String, queue_size=10, latch=True)
        self.accepted_status_pub = rospy.Publisher(
            "/planner/v2/accepted_status", String, queue_size=1, latch=True
        )
        rospy.Subscriber(rospy.get_param("~reference_topic", "/planner/ego_reference_trajectory"), RosPath, self._reference_cb, queue_size=1)
        rospy.Subscriber(rospy.get_param("~pose_topic", "/iris_0/world_pose"), PoseStamped, self._pose_cb, queue_size=1)
        rospy.Subscriber(rospy.get_param("~arm_state_topic", "/arm/current_theta_phi"), Float32MultiArray, self._arm_cb, queue_size=10)
        rospy.Subscriber("/planner/selected_reach_goal", PoseStamped, self._selected_goal_cb, queue_size=1)
        rospy.Subscriber("/planner/selected_reach_theta_phi", Float32MultiArray,
                         self._selected_shape_cb, queue_size=1)
        rospy.Subscriber("/planner/v2/execution_feedback", String, self._execution_feedback_cb, queue_size=10)
        rospy.Subscriber("/planner/v2/executor_status", String, self._executor_status_cb, queue_size=10)
        if self.map_mode == "regression_exact":
            rospy.Subscriber(rospy.get_param(
                "~point_cloud_topic", raw.get("point_cloud_topic", "/world/static_scene_cloud")
            ), PointCloud2, self._cloud_cb, queue_size=1)
        elif self.map_mode == "online_slam":
            rospy.Subscriber(rospy.get_param(
                "~occupied_voxels_topic", "/planner/map/occupied_voxels"
            ), PointCloud2, self._online_occupied_cb, queue_size=1)
            rospy.Subscriber(rospy.get_param(
                "~observed_voxels_topic", "/planner/map/observed_voxels"
            ), PointCloud2, self._online_observed_cb, queue_size=1)
            rospy.Subscriber(rospy.get_param(
                "~map_metadata_topic", "/planner/map/metadata"
            ), String, self._online_metadata_cb, queue_size=1)
        rate = max(0.02, float(raw.get("replan_rate_hz", 0.2)))
        self.timer = rospy.Timer(rospy.Duration(1.0 / rate), self._timer_cb)
        self._status(
            "WAITING_INPUT", map_mode=self.map_mode, map_source=map_source,
            boxes=truth_box_count,
            map_hash=None if self.map_snapshot is None else self.map_snapshot.map_hash,
            truth_geometry_access=self.map_mode == "regression_exact",
        )

    def _reference_cb(self, message):
        with self.lock:
            self.reference = message

    def _pose_cb(self, message):
        with self.lock:
            self.pose = message

    def _arm_cb(self, message):
        if len(message.data) >= 2:
            with self.lock:
                self.shape_deg = [float(message.data[0]), float(message.data[1])]

    def _selected_goal_cb(self, message):
        with self.lock:
            self.selected_reach_goal = message

    def _selected_shape_cb(self, message):
        if len(message.data) >= 2:
            with self.lock:
                self.selected_reach_shape_deg = [float(message.data[0]), float(message.data[1])]

    def _execution_feedback_cb(self, message):
        """Align the planner's prediction clock with the executor's guarded first handoff."""
        try:
            report = json.loads(message.data)
            identifier = int(report.get("trajectory_id"))
        except (TypeError, ValueError, KeyError):
            return
        state = report.get("state")
        if state == "BUFFERED":
            with self.lock:
                self.executor_pending_id = identifier
            return
        if state == "TRAJECTORY_COMPLETE":
            with self.lock:
                if identifier in self.terminal_trajectory_ids:
                    self.terminal_execution_finished = True
                    self.terminal_convergence_since = None
            return
        if state in ("REJECTED_START_DISCONTINUITY", "ABORT_TRACKING_ERROR",
                     "ABORT_INPUT_OR_PX4_STATE", "DISABLED_BY_OPERATOR",
                     "REJECTED_EXPIRED_ROLLING_WINDOW",
                     "REJECTED_ROLLING_DISCONTINUITY",
                     "REJECTED_PENDING_EXISTS"):
            with self.lock:
                self.trajectory_buffers.pop(identifier, None)
                if self.executor_pending_id == identifier:
                    self.executor_pending_id = None
                if self.safe_buffer is not None and self.safe_buffer[2] == identifier:
                    self.safe_buffer = None
                self.safe_waiting_for_enable = False
                if identifier in self.terminal_trajectory_ids:
                    self.terminal_trajectory_ids.discard(identifier)
                    self.terminal_execution_finished = False
                    self.terminal_convergence_since = None
            return
        if state not in ("START_SCHEDULED", "SWITCHED_ROLLING", "SWITCHED_AFTER_FINAL_HOLD"):
            return
        try:
            actual_start = rospy.Time.from_sec(float(report.get("actual_start")))
        except (TypeError, ValueError, KeyError):
            return
        with self.lock:
            buffered = self.trajectory_buffers.get(identifier)
            if buffered is not None:
                self.safe_buffer = (buffered[0], actual_start, identifier)
                self.executor_pending_id = None
                rospy.loginfo("V2 buffer clock aligned id=%d actual_start=%.3f", identifier, actual_start.to_sec())

    def _executor_status_cb(self, message):
        try:
            report = json.loads(message.data)
            enabled = bool(report.get("enabled", False))
        except (TypeError, ValueError):
            return
        with self.lock:
            self.executor_enabled = enabled
            if not enabled:
                # A published trajectory is not a predicted state until the
                # guarded executor has actually accepted control authority.
                self.safe_buffer = None
                self.executor_pending_id = None
                self.executor_phase_observation = None
            else:
                self.safe_waiting_for_enable = False
                # The executor owns a tracking-aware trajectory phase rather
                # than blindly equating progress with wall time.  Re-anchor
                # the prediction clock from its reported phase so successor
                # windows start from the state that will actually be reached.
                try:
                    identifier = int(report.get("trajectory_id"))
                    elapsed = max(0.0, float(report.get("elapsed")))
                    stamp = float(report.get("stamp"))
                except (TypeError, ValueError):
                    return
                if self.safe_buffer is not None and self.safe_buffer[2] == identifier:
                    trajectory = self.safe_buffer[0]
                    self.safe_buffer = (
                        trajectory, rospy.Time.from_sec(stamp - elapsed), identifier
                    )
                    self.executor_phase_observation = (
                        identifier, elapsed, rospy.Time.from_sec(stamp),
                        max(0.0, min(1.0, float(report.get("phase_rate", 0.0)))),
                    )

    def _cloud_cb(self, message):
        with self.lock:
            if self.point_cloud_static_once and self.point_world is not None:
                return
        points = np.asarray([
            (x, y, z) for x, y, z in point_cloud2.read_points(
                message, field_names=("x", "y", "z"), skip_nans=True
            )
        ], dtype=float)
        if len(points) > self.point_cloud_max_points:
            stride = int(math.ceil(len(points) / float(self.point_cloud_max_points)))
            points = points[::stride]
        cloud = PointCloudWorld(points, self.point_cloud_inflation)
        with self.lock:
            self.point_world = cloud
        rospy.loginfo_throttle(10.0, "V2 point cloud updated: %d points", len(points))

    @staticmethod
    def _message_points(message):
        return np.asarray([
            (x, y, z) for x, y, z in point_cloud2.read_points(
                message, field_names=("x", "y", "z"), skip_nans=True
            )
        ], dtype=float).reshape((-1, 3))

    def _online_occupied_cb(self, message):
        points = self._message_points(message)
        with self.lock:
            snapshot = self.online_map_assembler.update_occupied(
                points, message.header.frame_id, message.header.stamp.to_sec()
            )
            if snapshot is not None:
                self.map_snapshot = snapshot

    def _online_observed_cb(self, message):
        points = self._message_points(message)
        with self.lock:
            snapshot = self.online_map_assembler.update_observed(
                points, message.header.frame_id, message.header.stamp.to_sec()
            )
            if snapshot is not None:
                self.map_snapshot = snapshot

    def _online_metadata_cb(self, message):
        try:
            metadata = json.loads(message.data)
        except ValueError:
            rospy.logerr_throttle(2.0, "invalid /planner/map/metadata JSON")
            return
        if metadata.get("truth_geometry_access") is not False:
            rospy.logerr_throttle(2.0, "online map rejected: truth access audit flag missing")
            return
        with self.lock:
            try:
                snapshot = self.online_map_assembler.update_metadata(metadata)
            except ValueError as error:
                rospy.logerr_throttle(2.0, "online map rejected: %s", error)
                return
            if snapshot is not None:
                self.map_snapshot = snapshot

    def _status(self, state, **fields):
        report = {"state": state, "stamp": rospy.Time.now().to_sec()}
        report.update(fields)
        message = String(data=json.dumps(report, sort_keys=True))
        self.status_pub.publish(message)
        if state == "SAFE_TRAJECTORY":
            self.accepted_status_pub.publish(message)

    def _current_state(self, pose, shape_deg):
        p = pose.pose.position
        return np.asarray([p.x, p.y, p.z, yaw_from_quaternion(pose.pose.orientation),
                           math.radians(shape_deg[0]), math.radians(shape_deg[1])], dtype=float)

    def _publish(self, result, planning_stamp, nearest, requested_start,
                 reference_start_gap, reference_points, terminal_reach=False,
                 seed_time=0.0, optimize_validate_time=0.0):
        trajectory = result.trajectory
        msg = WholeBodyTrajectory()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "world"
        msg.trajectory_id = self.sequence
        msg.planning_state_stamp = planning_stamp
        msg.execution_start_time = max(
            requested_start, rospy.Time.now() + rospy.Duration(self.execution_lead)
        )
        msg.method = (
            "rolling_whole_body_v2_terminal_reach" if terminal_reach
            else "rolling_whole_body_v2_full_horizon"
        )
        msg.validator_status = result.validation.status
        msg.duration = trajectory.duration
        for index, time_value in enumerate(trajectory.times):
            state = trajectory.states[index]
            pose = PoseStamped().pose
            pose.position.x, pose.position.y, pose.position.z = state[:3]
            q = quaternion_from_euler(0.0, 0.0, state[3])
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q
            linear_velocity = Vector3()
            linear_velocity.x, linear_velocity.y, linear_velocity.z = trajectory.velocities[index, :3]
            msg.time_from_start.append(float(time_value))
            msg.uav_poses.append(pose)
            msg.linear_velocities.append(linear_velocity)
            msg.yaw_rates.append(float(trajectory.velocities[index, 3]))
            msg.theta_deg.append(math.degrees(state[4]))
            msg.phi_deg.append(math.degrees(state[5]))
            msg.theta_rate_deg_s.append(math.degrees(trajectory.velocities[index, 4]))
            msg.phi_rate_deg_s.append(math.degrees(trajectory.velocities[index, 5]))
        self.trajectory_pub.publish(msg)
        with self.lock:
            identifier = int(msg.trajectory_id)
            self.trajectory_buffers[identifier] = (trajectory, msg.execution_start_time)
            if len(self.trajectory_buffers) > 24:
                for old_id in sorted(self.trajectory_buffers)[:-24]:
                    if self.safe_buffer is None or old_id != self.safe_buffer[2]:
                        self.trajectory_buffers.pop(old_id, None)
            self.safe_waiting_for_enable = not self.executor_enabled
        path = RosPath()
        path.header = msg.header
        for pose in msg.uav_poses:
            item = PoseStamped()
            item.header = msg.header
            item.pose = pose
            path.poses.append(item)
        self.path_pub.publish(path)
        if terminal_reach:
            with self.lock:
                self.terminal_convergence_since = None
                self.terminal_trajectory_ids.add(int(msg.trajectory_id))
                if len(self.terminal_trajectory_ids) > 16:
                    self.terminal_trajectory_ids = set(sorted(self.terminal_trajectory_ids)[-16:])
        self._status("SAFE_TRAJECTORY", trajectory_id=msg.trajectory_id, duration=msg.duration,
                     solve_time=result.elapsed, objective=result.objective, nearest_reference_index=nearest,
                     reference_start_gap=reference_start_gap, reference_points=reference_points,
                     solver=result.status, metrics=result.validation.metrics,
                     diagnostics=result.diagnostics,
                     seed_time=seed_time, optimize_validate_time=optimize_validate_time,
                     total_planning_time=seed_time + optimize_validate_time)

    def _buffer_state(self, when):
        if self.safe_buffer is None:
            return None
        trajectory, start, identifier = self.safe_buffer
        observation = self.executor_phase_observation
        if observation is not None and observation[0] == identifier:
            elapsed = projected_phase_elapsed(
                observation[1], (when - observation[2]).to_sec(),
                observation[3], trajectory.duration,
            )
        else:
            elapsed = (when - start).to_sec()
        if elapsed < 0.0 or elapsed > trajectory.duration:
            return None
        return np.asarray([
            np.interp(elapsed, trajectory.times, trajectory.states[:, dim]) for dim in range(6)
        ], dtype=float)

    def _buffer_rate(self, when):
        """Predict the executor-owned full-body rate at a future handoff."""
        if self.safe_buffer is None:
            return None
        trajectory, start, identifier = self.safe_buffer
        observation = self.executor_phase_observation
        if observation is not None and observation[0] == identifier:
            elapsed = projected_phase_elapsed(
                observation[1], (when - observation[2]).to_sec(),
                observation[3], trajectory.duration,
            )
        else:
            elapsed = (when - start).to_sec()
        if elapsed < 0.0 or elapsed > trajectory.duration:
            return None
        return np.asarray([
            np.interp(elapsed, trajectory.times, trajectory.velocities[:, dim])
            for dim in range(6)
        ], dtype=float)

    def _timer_cb(self, _event):
        with self.lock:
            if (self.busy or self.pose is None
                    or time.monotonic() - self.last_plan_finish_wall < self.minimum_replan_wall_interval):
                return
            # One independently certified successor is already reserved by
            # the executor.  Preserve it and leave CPU to Gazebo/PX4 until
            # the guarded handoff accepts or rejects it.
            if self.executor_pending_id is not None:
                return
            # Do not replace a newly accepted rolling window before it has
            # delivered the configured geometric commitment.  The previous
            # implementation loaded commit_horizon in YAML but never used it;
            # repeated future-window handoffs reset executor phase near a
            # tight passage and could keep the vehicle oscillating around the
            # same wall indefinitely.  Gate on executor-owned trajectory
            # phase and sampled XYZ arc length, never on wall-clock time.
            if (self.executor_enabled and self.safe_buffer is not None
                    and not self.terminal_execution_finished):
                trajectory, _start, identifier = self.safe_buffer
                observation = self.executor_phase_observation
                if observation is None or observation[0] != identifier:
                    return
                committed_distance = trajectory_translation_progress(
                    trajectory.times, trajectory.states, observation[1]
                )
                if committed_distance + 1e-9 < self.commit_horizon:
                    return
            # In operator-gated mode one fresh certified window is enough for
            # inspection.  Replanning it continuously wastes CPU and can
            # leave a pre-enable solve in flight during control handoff.  The
            # executor reschedules this first window from the actual enable
            # time, then its status callback releases rolling replanning.
            if self.safe_waiting_for_enable and not self.executor_enabled:
                return
            if self.task_complete:
                return
            # Once a certified terminal trajectory has finished, completion
            # depends only on the measured UAV/arm state and the task target.
            # The rolling global path naturally collapses to zero or one point
            # near its goal; requiring two reference poses here used to stop
            # terminal tip-error monitoring exactly when it was needed.
            if (not self.terminal_execution_finished
                    and (self.reference is None or len(self.reference.poses) < 2)):
                return
            self.busy = True
            pose, reference, shape = self.pose, self.reference, list(self.shape_deg)
            selected_goal = self.selected_reach_goal
            selected_shape_deg = None if self.selected_reach_shape_deg is None else list(self.selected_reach_shape_deg)
            point_world = self.point_world
            map_snapshot = self.map_snapshot
            terminal_execution_finished = self.terminal_execution_finished
            executor_enabled = self.executor_enabled
            consecutive_plan_failures = self.consecutive_plan_failures
        try:
            measured = self._current_state(pose, shape)
            if terminal_execution_finished:
                terminal_tip_error = float(np.linalg.norm(
                    world_arm_points(measured, self.arm)[-1] - self.task_target
                ))
                now_wall = time.monotonic()
                if terminal_tip_error <= self.terminal_actual_tip_tolerance:
                    with self.lock:
                        if self.terminal_convergence_since is None:
                            self.terminal_convergence_since = now_wall
                        dwell = now_wall - self.terminal_convergence_since
                    if dwell >= self.terminal_convergence_dwell:
                        with self.lock:
                            self.task_complete = True
                            self.safe_buffer = None
                        self._status(
                            "TASK_COMPLETE", terminal_tip_error=terminal_tip_error,
                            terminal_dwell=dwell,
                            measured_theta_deg=shape[0], measured_phi_deg=shape[1],
                        )
                    else:
                        self._status(
                            "WAITING_TERMINAL_DWELL", terminal_tip_error=terminal_tip_error,
                            terminal_dwell=dwell,
                            required_dwell=self.terminal_convergence_dwell,
                            tolerance=self.terminal_actual_tip_tolerance,
                        )
                else:
                    with self.lock:
                        self.terminal_convergence_since = None
                    self._status(
                        "WAITING_TERMINAL_CONVERGENCE",
                        terminal_tip_error=terminal_tip_error,
                        tolerance=self.terminal_actual_tip_tolerance,
                    )
                return
            if self.terminal_trajectory_ids:
                self._status(
                    "WAITING_TERMINAL_EXECUTION",
                    trajectory_ids=sorted(self.terminal_trajectory_ids),
                )
                return
            now = rospy.Time.now()
            # A flight reference carries UAV topology, not a command to hold
            # the arm's currently measured bend forever.  Preserve the
            # measured shape only at the local-horizon boundary; the remaining
            # transit reference is neutral so collision, COM and rate costs
            # decide when bending is actually useful.
            reference_states = path_to_states(
                reference, self.transit_reference_shape
            )
            reference_start_gap = float(np.min(np.linalg.norm(reference_states[:, :3] - measured[:3], axis=1)))
            if reference_start_gap > self.maximum_reference_start_gap:
                self._status(
                    "REJECTED_REFERENCE_DISCONTINUITY",
                    reference_start_gap=reference_start_gap,
                    maximum_reference_start_gap=self.maximum_reference_start_gap,
                    reference_points=len(reference_states),
                )
                return
            predicted_now = self._buffer_state(now) if executor_enabled else None
            tracking_error = 0.0 if predicted_now is None else float(np.linalg.norm(predicted_now[:3] - measured[:3]))
            requested_start = now + rospy.Duration(self.latency_reserve) if predicted_now is not None else now
            predicted_start = self._buffer_state(requested_start)
            predicted_start_rate = self._buffer_rate(requested_start)
            if predicted_now is not None and tracking_error > self.maximum_tracking_error:
                self._status("BUFFER_RESET_TRACKING_ERROR", tracking_error=tracking_error)
                predicted_start = None
                predicted_start_rate = None
                # Use the measured state for this solve, but retain the
                # executor-accepted buffer.  A single transient tracking
                # spike must not permanently destroy the only truthful
                # prediction clock: doing so makes every later candidate
                # measured-state based and therefore discontinuous at a
                # future rolling handoff.  The executor remains the hard
                # authority and will abort sustained tracking error.
            current = predicted_start if predicted_start is not None else measured
            if predicted_start is None:
                requested_start = now
            recovery_window = (
                self.enable_recovery_windows
                and consecutive_plan_failures >= self.recovery_after_failures
            )
            active_horizon_distance = (
                self.recovery_horizon_distance if recovery_window
                else self.config.horizon_distance
            )
            retry_level = min(
                consecutive_plan_failures, self.failure_slowdown_levels
            )
            retry_speed_scale = self.failure_slowdown_step ** retry_level
            cycle_config = replace(
                self.config,
                nominal_speed=self.config.nominal_speed * retry_speed_scale,
                dt_max=self.config.dt_max / max(retry_speed_scale, 1e-6),
            )
            horizon, nearest = local_horizon(reference_states, current, active_horizon_distance)
            terminal_active = False
            terminal_shape = None
            if selected_goal is not None and selected_shape_deg is not None:
                goal = np.asarray([
                    selected_goal.pose.position.x, selected_goal.pose.position.y,
                    selected_goal.pose.position.z,
                ], dtype=float)
                if np.linalg.norm(horizon[-1, :3] - goal) <= 0.45:
                    terminal_active = True
                    terminal_shape = np.radians(np.asarray(selected_shape_deg, dtype=float))
                    # The global/EGO layer only supplies terminal topology.
                    # Once its local horizon reaches the selected standoff,
                    # make that certified reachability candidate the exact
                    # whole-body boundary condition.  Merely checking that
                    # the coarse reference ends "near" this pose can leave a
                    # 30--40 cm tool-tip error even with the right arm shape.
                    horizon[-1, :3] = goal
                    horizon[-1, 3] = yaw_from_quaternion(selected_goal.pose.orientation)
                    phi_delta = math.atan2(
                        math.sin(terminal_shape[1] - current[5]),
                        math.cos(terminal_shape[1] - current[5]),
                    )
                    for index in range(len(horizon)):
                        ratio = float(index) / float(max(1, len(horizon) - 1))
                        blend = ratio * ratio * (3.0 - 2.0 * ratio)
                        horizon[index, 4] = current[4] + blend * (terminal_shape[0] - current[4])
                        horizon[index, 5] = current[5] + blend * phi_delta
            self._status("PLANNING", reference_points=len(horizon), nearest_reference_index=nearest,
                         rolling_prediction=predicted_start is not None, tracking_error=tracking_error,
                         requested_start=requested_start.to_sec(), terminal_reach=terminal_active,
                         recovery_window=recovery_window,
                         horizon_distance=active_horizon_distance,
                         retry_level=retry_level,
                         retry_speed_scale=retry_speed_scale,
                         consecutive_plan_failures=consecutive_plan_failures)
            roi_minimum = np.min(horizon[:, :3], axis=0)
            roi_maximum = np.max(horizon[:, :3], axis=0)
            local_keepout = self.body_keepout_world.filtered(
                roi_minimum, roi_maximum, padding=0.75
            )
            if self.map_mode == "regression_exact":
                local_task = self.task_world.filtered(
                    roi_minimum, roi_maximum, padding=0.75
                )
                local_truth = self.truth_world.box_world.filtered(
                    roi_minimum, roi_maximum, padding=1.0
                )
                local_point_world = None
                if point_world is not None:
                    local_point_world = point_world.filtered(
                        roi_minimum, roi_maximum,
                        padding=self.point_cloud_local_padding,
                    )
                planning_world = HybridDistanceWorld(
                    box_world=local_task if self.esdf is not None else self.truth_world.box_world,
                    esdf=self.esdf,
                    point_world=local_point_world,
                )
                body_box_world = (
                    local_task if self.esdf is not None else self.truth_body_world.box_world
                )
                body_planning_world = HybridDistanceWorld(
                    box_world=body_box_world, esdf=self.esdf,
                    point_world=local_point_world,
                )
                validator_world = HybridDistanceWorld(
                    box_world=local_truth, point_world=local_point_world
                )
                validator_body_world = HybridDistanceWorld(
                    box_world=local_truth, point_world=local_point_world
                )
                local_point_count = 0 if local_point_world is None else len(local_point_world.points)
                active_map_hash = "regression_exact"
            else:
                if map_snapshot is None:
                    self._status(
                        "WAITING_SLAM_MAP", map_mode=self.map_mode,
                        truth_geometry_access=False,
                    )
                    return
                # Planner and runtime validator intentionally share the same
                # deployable map. Exact Gazebo geometry is evaluated only by
                # the separate oracle after execution.
                slam_world = map_snapshot.world(
                    inflation=self.point_cloud_inflation,
                    unknown_is_occupied=True,
                ).filtered(
                    roi_minimum, roi_maximum,
                    padding=self.point_cloud_local_padding,
                )
                planning_world = slam_world
                body_planning_world = slam_world
                validator_world = slam_world
                validator_body_world = slam_world
                local_point_count = len(slam_world.points)
                active_map_hash = map_snapshot.map_hash
            validator = SweptVolumeValidator(
                validator_world, self.arm, self.mass, self.limits, cycle_config,
                body_world=validator_body_world, center_keepout_world=local_keepout,
                terminal_target=self.task_target if terminal_active else None,
                terminal_tolerance=self.terminal_actual_tip_tolerance,
            )
            seed_planner = WholeBodySeedPlanner(
                planning_world, self.arm, self.mass, self.limits, cycle_config,
                body_world=body_planning_world, center_keepout_world=local_keepout,
            )
            optimizer = WholeBodyOptimizer(
                planning_world, self.arm, self.mass, self.limits, cycle_config, validator,
                body_world=body_planning_world, center_keepout_world=local_keepout,
            )
            stage_started = time.monotonic()
            seed = seed_planner.plan(
                horizon, current,
                terminal_state=horizon[-1] if terminal_active else None,
            )
            if terminal_shape is not None:
                seed.diagnostics["terminal_shape"] = terminal_shape.tolist()
                seed.diagnostics["terminal_position_tolerance"] = 0.06
            seed_elapsed = time.monotonic() - stage_started
            rospy.loginfo("V2 seed diagnostics: %s", json.dumps(seed.diagnostics, sort_keys=True))
            optimize_started = time.monotonic()
            result = optimizer.optimize(
                seed, horizon, initial_rate=predicted_start_rate
            )
            result.diagnostics["local_point_count"] = local_point_count
            result.diagnostics["map_mode"] = self.map_mode
            result.diagnostics["map_hash"] = active_map_hash
            result.diagnostics["truth_geometry_access"] = (
                self.map_mode == "regression_exact"
            )
            optimize_elapsed = time.monotonic() - optimize_started
            if (not result.success and not terminal_active
                    and result.validation.failure_reason == "arm_collision"
                    and seed.diagnostics.get("shape_mode") == "theta_only_fixed_phi"):
                # Active-set fallback: a point-wise fixed-plane screen is only
                # a necessary condition.  If both its continuous optimizer and
                # dense repair proof fail, unlock phi once in this same rolling
                # cycle instead of repeating the identical reduced problem.
                fallback_seed_started = time.monotonic()
                fallback_seed = seed_planner.plan(
                    horizon, current, terminal_state=None,
                    _force_phi_unlocked=True,
                )
                fallback_seed.diagnostics["active_set_fallback"] = (
                    "fixed_phi_optimizer_rejected"
                )
                fallback_seed_elapsed = time.monotonic() - fallback_seed_started
                rospy.loginfo(
                    "V2 active-set retry seed diagnostics: %s",
                    json.dumps(fallback_seed.diagnostics, sort_keys=True),
                )
                fallback_optimize_started = time.monotonic()
                fallback_result = optimizer.optimize(
                    fallback_seed, horizon, initial_rate=predicted_start_rate
                )
                fallback_result.diagnostics["local_point_count"] = local_point_count
                fallback_result.diagnostics["map_mode"] = self.map_mode
                fallback_result.diagnostics["map_hash"] = active_map_hash
                fallback_result.diagnostics["truth_geometry_access"] = (
                    self.map_mode == "regression_exact"
                )
                fallback_result.diagnostics["active_set_retry"] = "theta_phi"
                fallback_optimize_elapsed = time.monotonic() - fallback_optimize_started
                seed_elapsed += fallback_seed_elapsed
                optimize_elapsed += fallback_optimize_elapsed
                result = fallback_result
            rospy.loginfo(
                "V2 planning stages: seed=%.3fs optimize_and_validate=%.3fs total=%.3fs success=%s",
                seed_elapsed, optimize_elapsed, seed_elapsed + optimize_elapsed, str(result.success),
            )
            deadline_missed = (
                self.method_time_budget > 0.0
                and seed_elapsed + optimize_elapsed > self.method_time_budget
            )
            self.sequence += 1
            if result.success and not deadline_missed:
                with self.lock:
                    self.consecutive_plan_failures = 0
                self._publish(
                    result, requested_start, nearest, requested_start,
                    reference_start_gap, len(reference_states), terminal_active,
                    seed_elapsed, optimize_elapsed,
                )
            else:
                with self.lock:
                    self.consecutive_plan_failures += 1
                self._status("REJECTED_KEEP_BUFFER", solver=result.status,
                             failure=("plan_timeout" if deadline_missed
                                      else result.validation.failure_reason),
                             failure_index=result.validation.failure_index,
                             failure_time=result.validation.failure_time,
                             metrics=result.validation.metrics,
                             diagnostics=result.diagnostics,
                             solve_time=result.elapsed, seed_time=seed_elapsed,
                             optimize_validate_time=optimize_elapsed,
                             total_planning_time=seed_elapsed + optimize_elapsed,
                             method_time_budget=self.method_time_budget,
                             deadline_missed=deadline_missed,
                             recovery_window=recovery_window,
                             horizon_distance=active_horizon_distance,
                             consecutive_plan_failures=self.consecutive_plan_failures)
        except Exception as error:
            rospy.logerr("rolling_whole_body_planner_node: %s\n%s", error, traceback.format_exc())
            self._status("PLANNER_EXCEPTION", error=str(error))
        finally:
            with self.lock:
                self.busy = False
                self.last_plan_finish_wall = time.monotonic()


if __name__ == "__main__":
    rospy.init_node("rolling_whole_body_planner_node")
    RollingWholeBodyPlannerNode()
    rospy.spin()
