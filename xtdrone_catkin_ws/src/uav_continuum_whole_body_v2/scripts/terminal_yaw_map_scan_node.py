#!/usr/bin/env python3
"""Single-owner terminal yaw scan used only while acquiring a sensor map."""

import json
import math
import time

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, String


class TerminalYawMapScan:
    def __init__(self):
        self.duration = max(5.0, float(rospy.get_param("~duration_s", 20.0)))
        self.hold_after = max(1.0, float(rospy.get_param("~hold_after_s", 3.0)))
        self.scan_radius = max(0.0, float(rospy.get_param("~scan_radius_m", 0.15)))
        self.scan_z_offset = max(0.0, float(rospy.get_param("~scan_z_offset_m", 0.10)))
        self.rate_hz = max(20.0, float(rospy.get_param("~rate_hz", 30.0)))
        self.pose = None
        self.hold_position = None
        self.initial_yaw = None
        self.started_wall = None
        self.enabled = False
        self.complete = False
        self.publisher = rospy.Publisher(
            "/iris_0/mavros/setpoint_position/local", PoseStamped,
            queue_size=10,
        )
        self.status = rospy.Publisher(
            "/planner/map_survey_scan_status", String, queue_size=1, latch=True
        )
        rospy.Subscriber(
            "/iris_0/mavros/local_position/pose", PoseStamped,
            self._pose_cb, queue_size=20,
        )
        rospy.Subscriber(
            "/planner/map_survey_scan_enable", Bool,
            self._enable_cb, queue_size=1,
        )
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self._tick)

    @staticmethod
    def _yaw(pose):
        q = pose.orientation
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

    def _pose_cb(self, message):
        self.pose = message
        if not self.enabled:
            self.status.publish(String(data=json.dumps({
                "state": "READY", "setpoint_owner": False,
            }, sort_keys=True)))

    def _enable_cb(self, message):
        if not message.data or self.enabled or self.pose is None:
            return
        self.enabled = True
        self.started_wall = time.monotonic()
        position = self.pose.pose.position
        self.hold_position = (position.x, position.y, position.z)
        self.initial_yaw = self._yaw(self.pose.pose)
        self.status.publish(String(data=json.dumps({
            "state": "SCANNING", "duration_s": self.duration,
            "scan_radius_m": self.scan_radius,
            "scan_z_offset_m": self.scan_z_offset,
            "setpoint_owner": True,
        }, sort_keys=True)))

    def _tick(self, _event):
        if not self.enabled:
            return
        elapsed = time.monotonic() - self.started_wall
        phase = min(1.0, elapsed / self.duration)
        yaw = self.initial_yaw + 2.0 * math.pi * phase
        message = PoseStamped()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = self.pose.header.frame_id or "map"
        # Radially enter a small circle, survey one revolution, then return to
        # the exact task endpoint. The actually measured platform footprint,
        # not this command, is what the mapper clears as observed free space.
        if phase < 0.10:
            radial = phase / 0.10
            angle = 0.0
        elif phase < 0.90:
            radial = 1.0
            angle = 2.0 * math.pi * (phase - 0.10) / 0.80
        else:
            radial = max(0.0, (1.0 - phase) / 0.10)
            angle = 2.0 * math.pi
        message.pose.position.x = (
            self.hold_position[0] + self.scan_radius * radial * math.cos(angle)
        )
        message.pose.position.y = (
            self.hold_position[1] + self.scan_radius * radial * math.sin(angle)
        )
        # The depth camera has no upward return in open space. A small, slow
        # climb while circling makes the measured physical body sweep certify
        # the top layer of the planning envelope without inventing free rays.
        message.pose.position.z = self.hold_position[2] + self.scan_z_offset * radial
        message.pose.orientation.z = math.sin(0.5 * yaw)
        message.pose.orientation.w = math.cos(0.5 * yaw)
        self.publisher.publish(message)
        if elapsed >= self.duration + self.hold_after and not self.complete:
            self.complete = True
            self.status.publish(String(data=json.dumps({
                "state": "COMPLETE", "duration_s": elapsed,
                "setpoint_owner": True,
            }, sort_keys=True)))


if __name__ == "__main__":
    rospy.init_node("terminal_yaw_map_scan")
    TerminalYawMapScan()
    rospy.spin()
