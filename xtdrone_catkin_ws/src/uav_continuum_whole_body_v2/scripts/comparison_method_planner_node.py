#!/usr/bin/env python3
"""ROS adapter exposing every non-V2 baseline through one trajectory API."""

import json
import math
from pathlib import Path
import sys
import threading
import time

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "src"))

import numpy as np
import rospy
import yaml
from geometry_msgs.msg import PoseStamped, Vector3
from nav_msgs.msg import Path as RosPath
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray, String
from tf.transformations import quaternion_from_euler

from uav_continuum_whole_body_v2.comparison_methods import (
    ComparisonProblem, create_method,
)
from uav_continuum_whole_body_v2.configuration import load_planner_configuration
from uav_continuum_whole_body_v2.geometry import world_arm_points
from uav_continuum_whole_body_v2.horizon import path_to_states, yaw_from_quaternion
from uav_continuum_whole_body_v2.mapping import (
    MapSnapshot, OnlineSnapshotAssembler, validate_map_mode,
)
from uav_continuum_whole_body_v2.msg import WholeBodyTrajectory
from uav_continuum_whole_body_v2.types import ArmModel, YAW


class ComparisonMethodPlannerNode:
    def __init__(self):
        self.method_name = str(rospy.get_param("~method")).strip().lower().replace("-", "_")
        if self.method_name == "rolling_wb_v2":
            raise ValueError("rolling_wb_v2 must use rolling_whole_body_planner_node.py")
        self.method = create_method(self.method_name)
        self.map_mode = validate_map_mode(rospy.get_param("~map_mode", "frozen_slam"))
        if self.map_mode == "regression_exact":
            raise ValueError("formal comparison methods require frozen_slam or online_slam")
        config_path = Path(rospy.get_param(
            "~planner_config", str(PACKAGE / "config" / "planner_v2_realtime.yaml")
        ))
        raw, self.config, self.limits, self.mass = load_planner_configuration(config_path)
        self.arm = ArmModel()
        task_path = Path(rospy.get_param(
            "~task_file", str(PACKAGE / "config" / "scenario_3c_task.yaml")
        ))
        with task_path.open("r", encoding="utf-8") as handle:
            self.task = yaml.safe_load(handle) or {}
        self.target = np.asarray(self.task.get("target", [0.0, 0.0, 0.0])[:3], dtype=float)
        preferred = self.task.get("preferred_standoff", [0.0, 0.0, 1.0, 0.0])
        terminal_shape = self.task.get("terminal_shape_deg", [90.0, -180.0])
        self.terminal_state = np.asarray([
            preferred[0], preferred[1], preferred[2],
            preferred[3] if len(preferred) >= 4 else 0.0,
            math.radians(terminal_shape[0]), math.radians(terminal_shape[1]),
        ], dtype=float)
        self.method_time_budget = float(rospy.get_param("~method_time_budget", 10.0))
        self.random_seed = int(rospy.get_param("~random_seed", 0))
        self.map_inflation = float(rospy.get_param("~map_inflation", 0.02))
        self.execution_lead = float(rospy.get_param("~execution_lead_time", 0.25))
        task_tip_tolerance = float(self.task.get(
            "tip_tolerance_m",
            rospy.get_param("~terminal_actual_tip_tolerance", 0.08),
        ))
        self.terminal_tolerance = min(
            task_tip_tolerance,
            float(raw.get("terminal_precision_target", task_tip_tolerance)),
        )
        self.terminal_convergence_dwell = max(
            0.0, float(raw.get("terminal_convergence_dwell", 1.0))
        )
        self.lock = threading.Lock()
        self.pose = None
        self.shape_deg = [0.0, 0.0]
        self.reference = None
        self.snapshot = None
        self.assembler = OnlineSnapshotAssembler()
        self.busy = False
        self.planned_map_hash = None
        self.trajectory_id = 0
        self.execution_finished = False
        self.task_complete = False
        self.terminal_convergence_since = None
        self.last_terminal_active = False
        self.fixed_ready_since = None
        self.fixed_precondition_tolerance = float(rospy.get_param(
            "~fixed_precondition_tolerance_deg", 2.0
        ))
        self.fixed_precondition_dwell = float(rospy.get_param(
            "~fixed_precondition_dwell_s", 0.5
        ))

        self.trajectory_pub = rospy.Publisher(
            "/planner/v2/whole_body_trajectory", WholeBodyTrajectory,
            queue_size=1, latch=True,
        )
        self.path_pub = rospy.Publisher(
            "/planner/v2/uav_path", RosPath, queue_size=1, latch=True
        )
        self.status_pub = rospy.Publisher(
            "/planner/v2/status", String, queue_size=10, latch=True
        )
        self.accepted_pub = rospy.Publisher(
            "/planner/v2/accepted_status", String, queue_size=1, latch=True
        )
        self.arm_command_pub = rospy.Publisher(
            rospy.get_param("~arm_cmd_topic", "/arm/theta_phi_cmd"),
            Float32MultiArray, queue_size=10, latch=True,
        )
        rospy.Subscriber(rospy.get_param(
            "~reference_topic", "/planner/experiment/reference"
        ), RosPath, self._reference_cb, queue_size=1)
        rospy.Subscriber(rospy.get_param(
            "~pose_topic", "/iris_0/world_pose"
        ), PoseStamped, self._pose_cb, queue_size=1)
        rospy.Subscriber(rospy.get_param(
            "~arm_state_topic", "/arm/current_theta_phi"
        ), Float32MultiArray, self._shape_cb, queue_size=10)
        rospy.Subscriber(
            "/planner/v2/execution_feedback", String,
            self._execution_feedback_cb, queue_size=10,
        )
        if self.map_mode == "frozen_slam":
            snapshot_path = Path(rospy.get_param("~map_snapshot_file", ""))
            if not snapshot_path.is_file():
                raise FileNotFoundError(
                    "frozen_slam requires ~map_snapshot_file, got %s" % snapshot_path
                )
            self.snapshot = MapSnapshot.load(snapshot_path)
        else:
            rospy.Subscriber(
                "/planner/map/occupied_voxels", PointCloud2,
                self._occupied_cb, queue_size=1,
            )
            rospy.Subscriber(
                "/planner/map/observed_voxels", PointCloud2,
                self._observed_cb, queue_size=1,
            )
            rospy.Subscriber(
                "/planner/map/metadata", String, self._metadata_cb, queue_size=1,
            )
        self.timer = rospy.Timer(rospy.Duration(0.25), self._timer_cb)
        self._status("WAITING_INPUT")

    def _status(self, state, **fields):
        report = {
            "state": state, "stamp": rospy.Time.now().to_sec(),
            "method": self.method_name, "map_mode": self.map_mode,
            "map_hash": None if self.snapshot is None else self.snapshot.map_hash,
            "truth_geometry_access": False,
        }
        report.update(fields)
        message = String(data=json.dumps(report, sort_keys=True))
        self.status_pub.publish(message)
        if state == "SAFE_TRAJECTORY":
            self.accepted_pub.publish(message)

    def _pose_cb(self, message):
        with self.lock:
            self.pose = message

    def _shape_cb(self, message):
        if len(message.data) >= 2:
            with self.lock:
                self.shape_deg = list(message.data[:2])

    def _reference_cb(self, message):
        with self.lock:
            self.reference = message

    @staticmethod
    def _points(message):
        return np.asarray([
            item[:3] for item in point_cloud2.read_points(
                message, field_names=("x", "y", "z"), skip_nans=True
            )
        ], dtype=float).reshape((-1, 3))

    def _set_snapshot(self, snapshot):
        if snapshot is not None:
            with self.lock:
                self.snapshot = snapshot

    def _occupied_cb(self, message):
        try:
            self._set_snapshot(self.assembler.update_occupied(
                self._points(message), message.header.frame_id,
                message.header.stamp.to_sec(),
            ))
        except ValueError as error:
            self._status("MAP_REJECTED", error=str(error))

    def _observed_cb(self, message):
        try:
            self._set_snapshot(self.assembler.update_observed(
                self._points(message), message.header.frame_id,
                message.header.stamp.to_sec(),
            ))
        except ValueError as error:
            self._status("MAP_REJECTED", error=str(error))

    def _metadata_cb(self, message):
        try:
            self._set_snapshot(self.assembler.update_metadata(json.loads(message.data)))
        except (ValueError, TypeError) as error:
            self._status("MAP_REJECTED", error=str(error))

    def _execution_feedback_cb(self, message):
        try:
            report = json.loads(message.data)
        except ValueError:
            return
        if report.get("state") == "TRAJECTORY_COMPLETE":
            self.execution_finished = True

    def _current_state(self, pose, shape_deg):
        position = pose.pose.position
        return np.asarray([
            position.x, position.y, position.z,
            yaw_from_quaternion(pose.pose.orientation),
            math.radians(shape_deg[0]), math.radians(shape_deg[1]),
        ], dtype=float)

    def _publish(self, result, planning_stamp):
        trajectory = result.trajectory
        self.trajectory_id += 1
        message = WholeBodyTrajectory()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = "world"
        message.trajectory_id = self.trajectory_id
        message.planning_state_stamp = planning_stamp
        message.execution_start_time = rospy.Time.now() + rospy.Duration(self.execution_lead)
        message.method = self.method_name
        message.validator_status = result.validation.status
        message.duration = trajectory.duration
        for index, elapsed in enumerate(trajectory.times):
            state = trajectory.states[index]
            pose = PoseStamped().pose
            pose.position.x, pose.position.y, pose.position.z = state[:3]
            quaternion = quaternion_from_euler(0.0, 0.0, state[3])
            pose.orientation.x, pose.orientation.y = quaternion[:2]
            pose.orientation.z, pose.orientation.w = quaternion[2:]
            velocity = Vector3()
            velocity.x, velocity.y, velocity.z = trajectory.velocities[index, :3]
            message.time_from_start.append(float(elapsed))
            message.uav_poses.append(pose)
            message.linear_velocities.append(velocity)
            message.yaw_rates.append(float(trajectory.velocities[index, YAW]))
            message.theta_deg.append(math.degrees(state[4]))
            message.phi_deg.append(math.degrees(state[5]))
            message.theta_rate_deg_s.append(math.degrees(trajectory.velocities[index, 4]))
            message.phi_rate_deg_s.append(math.degrees(trajectory.velocities[index, 5]))
        self.trajectory_pub.publish(message)
        path = RosPath(header=message.header)
        for item in message.uav_poses:
            path.poses.append(PoseStamped(header=message.header, pose=item))
        self.path_pub.publish(path)

    def _timer_cb(self, _event):
        with self.lock:
            pose, reference, snapshot = self.pose, self.reference, self.snapshot
            shape_deg = list(self.shape_deg)
            if self.busy:
                return
        if self.task_complete:
            return
        if self.execution_finished:
            if self.map_mode == "online_slam" and not self.last_terminal_active:
                self.execution_finished = False
                self.planned_map_hash = None
                self._status("WAITING_NEXT_OBSERVED_WINDOW")
                return
            if pose is None:
                return
            state = self._current_state(pose, shape_deg)
            tip_error = float(np.linalg.norm(world_arm_points(state, self.arm)[-1] - self.target))
            if tip_error <= self.terminal_tolerance:
                now_wall = time.monotonic()
                if self.terminal_convergence_since is None:
                    self.terminal_convergence_since = now_wall
                dwell = now_wall - self.terminal_convergence_since
                if dwell >= self.terminal_convergence_dwell:
                    self.task_complete = True
                    self._status(
                        "TASK_COMPLETE", terminal_tip_error=tip_error,
                        terminal_dwell=dwell,
                    )
                else:
                    self._status(
                        "WAITING_TERMINAL_DWELL", terminal_tip_error=tip_error,
                        terminal_dwell=dwell,
                        required_dwell=self.terminal_convergence_dwell,
                        tolerance=self.terminal_tolerance,
                    )
            else:
                self.terminal_convergence_since = None
                self._status(
                    "WAITING_TERMINAL_CONVERGENCE", terminal_tip_error=tip_error,
                    tolerance=self.terminal_tolerance,
                )
            return
        if pose is None or reference is None or len(reference.poses) < 2 or snapshot is None:
            self._status(
                "WAITING_INPUT", missing_pose=pose is None,
                missing_reference=reference is None or len(reference.poses) < 2,
                missing_map=snapshot is None,
            )
            return
        if self.method_name == "fixed_morphology":
            fixed_shape = np.rad2deg(self.terminal_state[[4, 5]])
            self.arm_command_pub.publish(Float32MultiArray(data=fixed_shape.tolist()))
            error = float(np.max(np.abs(np.asarray(shape_deg) - fixed_shape)))
            now = time.monotonic()
            if error > self.fixed_precondition_tolerance:
                self.fixed_ready_since = None
                self._status(
                    "PRECONDITIONING_FIXED_MORPHOLOGY",
                    target_shape_deg=fixed_shape.tolist(), shape_error_deg=error,
                    excluded_from_task_timer=True,
                )
                return
            if self.fixed_ready_since is None:
                self.fixed_ready_since = now
            dwell = now - self.fixed_ready_since
            if dwell < self.fixed_precondition_dwell:
                self._status(
                    "PRECONDITIONING_FIXED_MORPHOLOGY",
                    target_shape_deg=fixed_shape.tolist(), shape_error_deg=error,
                    dwell_s=dwell, excluded_from_task_timer=True,
                )
                return
        if (self.planned_map_hash is not None
                and (self.map_mode == "frozen_slam"
                     or self.planned_map_hash == snapshot.map_hash)):
            return
        with self.lock:
            self.busy = True
        try:
            current = self._current_state(pose, shape_deg)
            if self.method_name == "fixed_morphology":
                current[[4, 5]] = self.terminal_state[[4, 5]]
            states = path_to_states(reference, default_shape=current[4:])
            terminal_active = bool(
                np.linalg.norm(states[-1, :3] - self.terminal_state[:3]) <= 0.12
            )
            if terminal_active:
                states[-1] = self.terminal_state
            world = snapshot.world(
                inflation=self.map_inflation, unknown_is_occupied=True
            )
            problem = ComparisonProblem(
                reference_states=states, current_state=current,
                terminal_state=self.terminal_state if terminal_active else None,
                terminal_target=self.target if terminal_active else None,
                terminal_tolerance=self.terminal_tolerance,
                world=world, arm=self.arm, mass=self.mass,
                limits=self.limits, config=self.config,
                method_time_budget=self.method_time_budget,
                random_seed=self.random_seed,
            )
            started = time.monotonic()
            self._status("PLANNING")
            result = self.method.plan(problem)
            elapsed = time.monotonic() - started
            if not result.success:
                self.planned_map_hash = snapshot.map_hash
                self._status(
                    "METHOD_FAILED", failure=result.validation.failure_reason,
                    solve_time=elapsed, diagnostics=result.diagnostics,
                    metrics=result.validation.metrics,
                )
                return
            self._publish(result, pose.header.stamp)
            self.planned_map_hash = snapshot.map_hash
            self.last_terminal_active = terminal_active
            self._status(
                "SAFE_TRAJECTORY", trajectory_id=self.trajectory_id,
                duration=result.trajectory.duration, solve_time=elapsed,
                objective=result.objective, metrics=result.validation.metrics,
                diagnostics=result.diagnostics, terminal_active=terminal_active,
            )
        except Exception as error:
            self.planned_map_hash = snapshot.map_hash
            self._status("METHOD_EXCEPTION", error=str(error))
            rospy.logerr("comparison method %s failed: %s", self.method_name, error)
        finally:
            with self.lock:
                self.busy = False


if __name__ == "__main__":
    rospy.init_node("comparison_method_planner_node")
    ComparisonMethodPlannerNode()
    rospy.spin()
