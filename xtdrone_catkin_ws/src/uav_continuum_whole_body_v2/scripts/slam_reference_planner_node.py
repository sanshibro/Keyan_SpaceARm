#!/usr/bin/env python3
"""Publish the common body-clear reference from the planner-visible map."""

import json
import math
from pathlib import Path
import sys
import threading

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "src"))

import numpy as np
import rospy
import yaml
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as RosPath
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray, String
from tf.transformations import quaternion_from_euler

from uav_continuum_whole_body_v2.mapping import (
    MapSnapshot, OnlineSnapshotAssembler, validate_map_mode,
)
from uav_continuum_whole_body_v2.reference_planner import (
    ReferencePlanningError, VoxelAStarReferencePlanner, densify_path,
)


class SlamReferencePlannerNode:
    def __init__(self):
        self.map_mode = validate_map_mode(rospy.get_param("~map_mode", "frozen_slam"))
        if self.map_mode == "regression_exact":
            raise ValueError("slam_reference_planner_node cannot run in regression_exact mode")
        task_path = Path(rospy.get_param(
            "~task_file", str(PACKAGE / "config" / "scenario_3c_task.yaml")
        ))
        with task_path.open("r", encoding="utf-8") as handle:
            self.task = yaml.safe_load(handle) or {}
        preferred = self.task.get("preferred_standoff")
        if not isinstance(preferred, (list, tuple)) or len(preferred) < 3:
            raise ValueError("task file needs preferred_standoff")
        self.goal = np.asarray(preferred[:3], dtype=float)
        self.goal_yaw = float(preferred[3]) if len(preferred) >= 4 else 0.0
        self.terminal_shape_deg = [float(value) for value in self.task.get(
            "terminal_shape_deg", [90.0, -180.0]
        )[:2]]
        self.center_clearance = float(rospy.get_param("~center_clearance", 0.38))
        self.body_radius = float(rospy.get_param("~body_radius", 0.28))
        self.body_height = float(rospy.get_param("~body_height", 0.25))
        self.body_margin = float(rospy.get_param("~body_margin", 0.10))
        self.z_min = float(rospy.get_param("~z_min", 0.75))
        self.z_max = float(rospy.get_param("~z_max", 2.20))
        self.replan_movement = max(0.05, float(rospy.get_param(
            "~replan_movement_threshold", 0.50
        )))
        self.lock = threading.Lock()
        self.pose = None
        self.snapshot = None
        self.assembler = OnlineSnapshotAssembler()
        self.last_plan_pose = None
        self.last_map_hash = None
        self.planning = False
        self.reference_planner = None
        self.reference_planner_hash = None
        self.path_pub = rospy.Publisher(
            "/planner/experiment/reference", RosPath, queue_size=1, latch=True
        )
        self.goal_pub = rospy.Publisher(
            "/planner/selected_reach_goal", PoseStamped, queue_size=1, latch=True
        )
        self.shape_pub = rospy.Publisher(
            "/planner/selected_reach_theta_phi", Float32MultiArray,
            queue_size=1, latch=True,
        )
        self.status_pub = rospy.Publisher(
            "/planner/experiment/reference_status", String, queue_size=1, latch=True
        )
        rospy.Subscriber(rospy.get_param(
            "~pose_topic", "/iris_0/world_pose"
        ), PoseStamped, self._pose_cb, queue_size=1)
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
        self.timer = rospy.Timer(rospy.Duration(0.5), self._timer_cb)

    def _status(self, state, **fields):
        report = {
            "state": state, "stamp": rospy.Time.now().to_sec(),
            "map_mode": self.map_mode, "truth_geometry_access": False,
        }
        report.update(fields)
        self.status_pub.publish(String(data=json.dumps(report, sort_keys=True)))

    def _pose_cb(self, message):
        with self.lock:
            self.pose = message

    @staticmethod
    def _points(message):
        return np.asarray([
            item[:3] for item in point_cloud2.read_points(
                message, field_names=("x", "y", "z"), skip_nans=True
            )
        ], dtype=float).reshape((-1, 3))

    def _occupied_cb(self, message):
        try:
            snapshot = self.assembler.update_occupied(
                self._points(message), message.header.frame_id,
                message.header.stamp.to_sec(),
            )
        except ValueError as error:
            self._status("MAP_REJECTED", error=str(error))
            return
        if snapshot is not None:
            with self.lock:
                self.snapshot = snapshot

    def _observed_cb(self, message):
        try:
            snapshot = self.assembler.update_observed(
                self._points(message), message.header.frame_id,
                message.header.stamp.to_sec(),
            )
        except ValueError as error:
            self._status("MAP_REJECTED", error=str(error))
            return
        if snapshot is not None:
            with self.lock:
                self.snapshot = snapshot

    def _metadata_cb(self, message):
        try:
            snapshot = self.assembler.update_metadata(json.loads(message.data))
        except (ValueError, TypeError) as error:
            self._status("MAP_REJECTED", error=str(error))
            return
        if snapshot is not None:
            with self.lock:
                self.snapshot = snapshot

    def _publish_goal(self, frame_id):
        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = frame_id
        goal.pose.position.x, goal.pose.position.y, goal.pose.position.z = self.goal
        quaternion = quaternion_from_euler(0.0, 0.0, self.goal_yaw)
        goal.pose.orientation.x, goal.pose.orientation.y = quaternion[:2]
        goal.pose.orientation.z, goal.pose.orientation.w = quaternion[2:]
        self.goal_pub.publish(goal)
        self.shape_pub.publish(Float32MultiArray(data=self.terminal_shape_deg))

    def _publish_path(self, points, frame_id):
        message = RosPath()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = frame_id
        for index, point in enumerate(points):
            pose = PoseStamped()
            pose.header = message.header
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = point
            if index + 1 < len(points):
                delta = points[index + 1] - point
            else:
                delta = point - points[index - 1]
            yaw = math.atan2(delta[1], delta[0]) if np.linalg.norm(delta[:2]) > 1e-9 else 0.0
            quaternion = quaternion_from_euler(0.0, 0.0, yaw)
            pose.pose.orientation.x, pose.pose.orientation.y = quaternion[:2]
            pose.pose.orientation.z, pose.pose.orientation.w = quaternion[2:]
            message.poses.append(pose)
        self.path_pub.publish(message)

    def _timer_cb(self, _event):
        with self.lock:
            if self.planning:
                return
            self.planning = True
        try:
            self._plan_cycle()
        finally:
            with self.lock:
                self.planning = False

    def _plan_cycle(self):
        with self.lock:
            pose = self.pose
            snapshot = self.snapshot
        if pose is None or snapshot is None:
            self._status(
                "WAITING_INPUT", missing_pose=pose is None,
                missing_map=snapshot is None,
            )
            return
        position = np.asarray([
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z,
        ], dtype=float)
        if (self.last_plan_pose is not None and self.last_map_hash == snapshot.map_hash
                and np.linalg.norm(position - self.last_plan_pose) < self.replan_movement):
            self._publish_goal(snapshot.frame_id)
            return
        try:
            if (self.reference_planner is None
                    or self.reference_planner_hash != snapshot.map_hash):
                self._status("PREPARING_REFERENCE_MAP", map_hash=snapshot.map_hash)
                self.reference_planner = VoxelAStarReferencePlanner(
                    snapshot, center_clearance=self.center_clearance,
                    body_radius=self.body_radius, body_height=self.body_height,
                    body_margin=self.body_margin,
                    z_min=self.z_min, z_max=self.z_max,
                    time_budget=float(rospy.get_param(
                        "~planning_time_budget", 30.0
                    )),
                )
                self.reference_planner_hash = snapshot.map_hash
            points, diagnostics = self.reference_planner.plan(
                position, self.goal,
                allow_partial=self.map_mode == "online_slam",
            )
        except ReferencePlanningError as error:
            self._status(
                "NO_REFERENCE", error=str(error), map_hash=snapshot.map_hash
            )
            return
        points = densify_path(points, maximum_step=float(rospy.get_param(
            "~reference_step", 0.20
        )))
        self._publish_path(points, snapshot.frame_id)
        self._publish_goal(snapshot.frame_id)
        self.last_plan_pose = position
        self.last_map_hash = snapshot.map_hash
        self._status("REFERENCE_READY", **diagnostics)


if __name__ == "__main__":
    rospy.init_node("slam_reference_planner_node")
    SlamReferencePlannerNode()
    rospy.spin()
