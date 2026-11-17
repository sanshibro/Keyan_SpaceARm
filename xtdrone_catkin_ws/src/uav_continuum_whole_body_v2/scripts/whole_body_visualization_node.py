#!/usr/bin/env python3
import json
import math
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "src"))

import numpy as np
import rospy
from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import Float32MultiArray, String
from visualization_msgs.msg import Marker, MarkerArray

from uav_continuum_whole_body_v2.geometry import coupling_metrics, world_arm_sections
from uav_continuum_whole_body_v2.msg import WholeBodyTrajectory
from uav_continuum_whole_body_v2.types import ArmModel, Limits, MassModel


def color(marker, red, green, blue, alpha):
    marker.color.r = float(red)
    marker.color.g = float(green)
    marker.color.b = float(blue)
    marker.color.a = float(alpha)


def point(value):
    return Point(x=float(value[0]), y=float(value[1]), z=float(value[2]))


class WholeBodyVisualizationNode:
    """Publish a truthful RViz preview of every accepted local whole-body trajectory."""

    def __init__(self):
        self.arm = ArmModel()
        self.mass = MassModel()
        self.limits = Limits()
        self.max_ghosts = max(2, int(rospy.get_param("~max_ghosts", 30)))
        self.label_stride = max(1, int(rospy.get_param("~label_stride", 3)))
        self.body_alpha = float(rospy.get_param("~body_alpha", 0.20))
        self.arm_width = float(rospy.get_param("~arm_width", 0.018))
        self.executed_ghost_period = max(0.1, float(rospy.get_param("~executed_ghost_period", 0.25)))
        self.executed_max_ghosts = max(5, int(rospy.get_param("~executed_max_ghosts", 60)))
        self.current_arm_deg = [0.0, 0.0]
        self.executed_states = []
        self.last_executed_stamp = rospy.Time(0)
        self.marker_pub = rospy.Publisher(
            "/planner/v2/whole_body_markers", MarkerArray, queue_size=1, latch=True
        )
        self.status_pub = rospy.Publisher(
            "/planner/v2/visualization_status", String, queue_size=1, latch=True
        )
        self.executed_marker_pub = rospy.Publisher(
            "/planner/v2/executed_whole_body_markers", MarkerArray, queue_size=1, latch=True
        )
        rospy.Subscriber(
            "/planner/v2/whole_body_trajectory", WholeBodyTrajectory, self._trajectory_cb, queue_size=1
        )
        rospy.Subscriber("/planner/v2/commanded_pose_world", PoseStamped, self._executed_pose_cb, queue_size=10)
        rospy.Subscriber("/arm/current_theta_phi", Float32MultiArray, self._arm_state_cb, queue_size=10)
        self.status_pub.publish(String(data=json.dumps({"state": "WAITING_TRAJECTORY"})))

    def _arm_state_cb(self, message):
        if len(message.data) >= 2:
            self.current_arm_deg = [float(message.data[0]), float(message.data[1])]

    def _executed_pose_cb(self, message):
        now = rospy.Time.now()
        if self.last_executed_stamp != rospy.Time(0) and (now - self.last_executed_stamp).to_sec() < self.executed_ghost_period:
            return
        q = message.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        p = message.pose.position
        state = np.asarray([p.x, p.y, p.z, yaw,
                            math.radians(self.current_arm_deg[0]), math.radians(self.current_arm_deg[1])])
        self.executed_states.append(state)
        if len(self.executed_states) > self.executed_max_ghosts:
            self.executed_states = self.executed_states[-self.executed_max_ghosts:]
        self.last_executed_stamp = now
        self._publish_executed_markers(message)

    def _publish_executed_markers(self, pose_message):
        markers = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        marker_id = 0
        for index, state in enumerate(self.executed_states):
            base, flexible, tool = world_arm_sections(state, self.arm)
            alpha = 0.25 + 0.65 * float(index + 1) / float(len(self.executed_states))
            for namespace, section, rgb in (
                ("v2_executed_base", base, (0.25, 1.0, 0.25)),
                ("v2_executed_flexible", flexible, (1.0, 0.2, 0.85)),
                ("v2_executed_tool", tool, (1.0, 0.55, 0.08)),
            ):
                marker = Marker()
                marker.header.stamp = pose_message.header.stamp
                marker.header.frame_id = "world"
                marker.ns = namespace
                marker.id = marker_id
                marker_id += 1
                marker.type = Marker.LINE_STRIP
                marker.action = Marker.ADD
                marker.pose.orientation.w = 1.0
                marker.scale.x = self.arm_width
                marker.points = [point(value) for value in section]
                color(marker, rgb[0], rgb[1], rgb[2], alpha)
                markers.markers.append(marker)
        self.executed_marker_pub.publish(markers)

    @staticmethod
    def _header(marker, message, namespace, marker_id, marker_type):
        marker.header = message.header
        marker.ns = namespace
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.lifetime = rospy.Duration(0.0)

    def _indices(self, count):
        if count <= self.max_ghosts:
            return list(range(count))
        return sorted(set(np.linspace(0, count - 1, self.max_ghosts).round().astype(int).tolist()))

    @staticmethod
    def _state(message, index):
        pose = message.uav_poses[index]
        q = pose.orientation
        # Published V2 poses are yaw-only; this avoids a tf dependency here.
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return np.asarray([
            pose.position.x, pose.position.y, pose.position.z, yaw,
            math.radians(message.theta_deg[index]), math.radians(message.phi_deg[index]),
        ], dtype=float)

    def _trajectory_cb(self, message):
        count = min(len(message.time_from_start), len(message.uav_poses), len(message.theta_deg), len(message.phi_deg))
        if message.validator_status != "safe" or count < 2:
            self.status_pub.publish(String(data=json.dumps({
                "state": "REJECTED_TRAJECTORY", "validator_status": message.validator_status, "samples": count,
            }, sort_keys=True)))
            return

        markers = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        selected = self._indices(count)
        marker_id = 0
        com_offsets = []

        for ghost_number, index in enumerate(selected):
            state = self._state(message, index)
            ratio = float(ghost_number) / float(max(1, len(selected) - 1))
            red, green, blue = ratio, 0.85 - 0.35 * ratio, 1.0 - 0.75 * ratio

            body = Marker()
            self._header(body, message, "v2_body_ghost", marker_id, Marker.CYLINDER)
            marker_id += 1
            body.pose = message.uav_poses[index]
            body.scale.x = 2.0 * self.limits.body_radius
            body.scale.y = 2.0 * self.limits.body_radius
            body.scale.z = self.limits.body_height
            color(body, red, green, blue, self.body_alpha)
            markers.markers.append(body)

            base, flexible, tool = world_arm_sections(state, self.arm)
            for namespace, section, section_color in (
                ("v2_arm_base", base, (0.25, 0.85, 0.25)),
                ("v2_arm_flexible", flexible, (red, green, blue)),
                ("v2_arm_tool", tool, (1.0, 0.55, 0.08)),
            ):
                arm_marker = Marker()
                self._header(arm_marker, message, namespace, marker_id, Marker.LINE_STRIP)
                marker_id += 1
                arm_marker.scale.x = self.arm_width
                arm_marker.points = [point(value) for value in section]
                color(arm_marker, section_color[0], section_color[1], section_color[2], 0.88)
                markers.markers.append(arm_marker)

            coupling = coupling_metrics(state, self.arm, self.mass)
            com_offsets.append(coupling["com_offset"])
            com = Marker()
            self._header(com, message, "v2_system_com", marker_id, Marker.SPHERE)
            marker_id += 1
            com.pose.position = point(coupling["system_com"])
            com.scale.x = com.scale.y = com.scale.z = 0.045
            color(com, 1.0, 0.1, 0.8, 0.90)
            markers.markers.append(com)

            com_line = Marker()
            self._header(com_line, message, "v2_com_offset", marker_id, Marker.LINE_LIST)
            marker_id += 1
            com_line.scale.x = 0.008
            com_line.points = [point(state[:3]), point(coupling["system_com"])]
            color(com_line, 1.0, 0.1, 0.8, 0.75)
            markers.markers.append(com_line)

            if ghost_number % self.label_stride == 0 or ghost_number == len(selected) - 1:
                label = Marker()
                self._header(label, message, "v2_shape_labels", marker_id, Marker.TEXT_VIEW_FACING)
                marker_id += 1
                label.pose.position.x = state[0]
                label.pose.position.y = state[1]
                label.pose.position.z = state[2] + 0.24
                label.scale.z = 0.11
                label.text = "t=%.1f  th=%.0f  ph=%.0f" % (
                    message.time_from_start[index], message.theta_deg[index], message.phi_deg[index]
                )
                color(label, 1.0, 1.0, 1.0, 0.92)
                markers.markers.append(label)

        self.marker_pub.publish(markers)
        self.status_pub.publish(String(data=json.dumps({
            "state": "READY", "trajectory_id": int(message.trajectory_id), "samples": count,
            "ghosts": len(selected), "duration": float(message.duration),
            "com_offset_max_preview": max(com_offsets) if com_offsets else 0.0,
            "meaning": "accepted_local_whole_body_window_not_global_tail",
        }, sort_keys=True)))


if __name__ == "__main__":
    rospy.init_node("whole_body_visualization_node")
    WholeBodyVisualizationNode()
    rospy.spin()
