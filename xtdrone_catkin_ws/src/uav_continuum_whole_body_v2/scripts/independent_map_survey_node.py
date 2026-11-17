#!/usr/bin/env python3
"""Execute a frozen waypoint survey without loading a planner map or oracle."""

import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, String

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "src"))
from uav_continuum_whole_body_v2.survey_protocol import load_survey_manifest  # noqa: E402


class IndependentMapSurvey:
    def __init__(self):
        path = rospy.get_param("~manifest")
        self.manifest, self.manifest_sha256 = load_survey_manifest(path)
        self.waypoints = self.manifest["waypoints"]
        self.speed = max(0.03, float(self.manifest.get("nominal_speed_mps", 0.12)))
        # Keep a fixed margin below the independent certification limit while
        # remaining reachable under PX4's measured altitude steady-state
        # error. The controller and certifier both use registered-world pose.
        self.tolerance = max(
            0.03,
            float(self.manifest.get("waypoint_tolerance_m", 0.25)) * 0.75,
        )
        self.rate_hz = max(20.0, float(rospy.get_param("~rate_hz", 30.0)))
        self.pose = None
        self.registered_pose = None
        self.local_minus_registered = None
        self.enabled = False
        self.complete = False
        self.aborted = False
        self.index = 0
        self.segment_start = None
        self.segment_goal = np.asarray(self.waypoints[0]["pose"], dtype=float)
        self.segment_started = None
        self.arrived_at = None
        self.scan_started = None
        self.last_command = None
        self.deviation_started = None
        self.publisher = rospy.Publisher(
            "/iris_0/mavros/setpoint_position/local", PoseStamped, queue_size=10,
        )
        self.status = rospy.Publisher(
            "/planner/independent_survey_status", String, queue_size=1, latch=True,
        )
        rospy.Subscriber(
            "/iris_0/mavros/local_position/pose", PoseStamped,
            self._pose_cb, queue_size=30,
        )
        rospy.Subscriber(
            rospy.get_param("~registered_pose_topic", "/slam/estimator_body_pose"),
            PoseStamped, self._registered_pose_cb, queue_size=30,
        )
        rospy.Subscriber(
            "/planner/independent_survey_enable", Bool,
            self._enable_cb, queue_size=1,
        )
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self._tick)

    def _publish_status(self, state, **extra):
        payload = {
            "state": state, "waypoint_index": self.index,
            "waypoint_count": len(self.waypoints),
            "manifest_sha256": self.manifest_sha256,
            "setpoint_owner": bool(
                self.pose is not None and self.registered_pose is not None
            ),
        }
        payload.update(extra)
        self.status.publish(String(data=json.dumps(payload, sort_keys=True)))

    def _pose_cb(self, message):
        self.pose = message
        if not self.enabled and self.registered_pose is not None:
            self._publish_status("READY")

    def _registered_pose_cb(self, message):
        self.registered_pose = message
        if not self.enabled and self.pose is not None:
            self._publish_status("READY")

    def _enable_cb(self, message):
        if (not message.data or self.enabled or self.pose is None
                or self.registered_pose is None):
            return
        self.enabled = True
        local = self.pose.pose.position
        registered = self.registered_pose.pose.position
        self.local_minus_registered = np.asarray([
            local.x - registered.x,
            local.y - registered.y,
            local.z - registered.z,
        ], dtype=float)
        current = registered
        self.segment_start = np.asarray([current.x, current.y, current.z], dtype=float)
        self.segment_started = time.monotonic()
        self._publish_status("EXECUTING")

    @staticmethod
    def _yaw_quaternion(yaw):
        return math.sin(0.5 * yaw), math.cos(0.5 * yaw)

    def _setpoint(self, xyz, yaw):
        if self.local_minus_registered is None:
            if self.pose is None or self.registered_pose is None:
                return
            local = self.pose.pose.position
            registered = self.registered_pose.pose.position
            self.local_minus_registered = np.asarray([
                local.x - registered.x,
                local.y - registered.y,
                local.z - registered.z,
            ], dtype=float)
        local_xyz = np.asarray(xyz, dtype=float) + self.local_minus_registered
        message = PoseStamped()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = self.pose.header.frame_id or "map"
        message.pose.position.x, message.pose.position.y, message.pose.position.z = local_xyz
        message.pose.orientation.z, message.pose.orientation.w = self._yaw_quaternion(yaw)
        self.last_command = local_xyz
        self.publisher.publish(message)

    def _advance(self, now):
        self.index += 1
        self.arrived_at = None
        if self.index >= len(self.waypoints):
            self.scan_started = now
            self._publish_status("SCANNING")
            return
        current = self.registered_pose.pose.position
        self.segment_start = np.asarray([current.x, current.y, current.z], dtype=float)
        self.segment_goal = np.asarray(self.waypoints[self.index]["pose"], dtype=float)
        self.segment_started = now
        self._publish_status("EXECUTING")

    def _scan_setpoint(self, now):
        scan = self.manifest.get("terminal_scan") or {}
        duration = max(1.0, float(scan.get("duration_s", 20.0)))
        elapsed = now - self.scan_started
        phase = min(1.0, elapsed / duration)
        final = np.asarray(self.waypoints[-1]["pose"], dtype=float)
        radius = max(0.0, float(scan.get("radius_m", 0.0)))
        z_offset = max(0.0, float(scan.get("z_offset_m", 0.0)))
        turns = max(0.0, float(scan.get("turns", 1.0)))
        envelope = math.sin(math.pi * phase)
        angle = 2.0 * math.pi * turns * phase
        xyz = final[:3] + np.asarray([
            radius * envelope * math.cos(angle),
            radius * envelope * math.sin(angle), z_offset * envelope,
        ])
        self._setpoint(xyz, final[3] + angle)
        if elapsed >= duration and not self.complete:
            self.complete = True
            self._publish_status("COMPLETE", elapsed_s=elapsed)

    def _tick(self, _event):
        if self.pose is None or self.registered_pose is None:
            return
        now = time.monotonic()
        if not self.enabled:
            target = np.asarray(self.waypoints[0]["pose"], dtype=float)
            self._setpoint(target[:3], target[3])
            return
        current = np.asarray([
            self.registered_pose.pose.position.x,
            self.registered_pose.pose.position.y,
            self.registered_pose.pose.position.z,
        ])
        if self.last_command is not None and self.index > 0:
            local = self.pose.pose.position
            local_position = np.asarray([local.x, local.y, local.z], dtype=float)
            deviation = float(np.linalg.norm(local_position - self.last_command))
            if deviation > 0.45:
                if self.deviation_started is None:
                    self.deviation_started = now
                elif now - self.deviation_started > 3.0 and not self.aborted:
                    self.aborted = True
                    self._publish_status("ABORT_TRACKING", tracking_error_m=deviation)
            else:
                self.deviation_started = None
        if self.aborted:
            self._setpoint(current, float(self.waypoints[max(0, self.index - 1)]["pose"][3]))
            return
        if self.complete:
            target = np.asarray(self.waypoints[-1]["pose"], dtype=float)
            self._setpoint(target[:3], target[3])
            return
        if self.scan_started is not None:
            self._scan_setpoint(now)
            return
        target = np.asarray(self.waypoints[self.index]["pose"], dtype=float)
        distance = float(np.linalg.norm(target[:3] - current))
        if distance <= self.tolerance:
            if self.arrived_at is None:
                self.arrived_at = now
            dwell = max(0.0, float(self.waypoints[self.index].get("dwell_s", 0.5)))
            self._setpoint(target[:3], target[3])
            if now - self.arrived_at >= dwell:
                self._advance(now)
            return
        elapsed = max(0.0, now - self.segment_started)
        length = float(np.linalg.norm(target[:3] - self.segment_start))
        fraction = min(1.0, elapsed * self.speed / max(length, 1e-6))
        commanded = self.segment_start + fraction * (target[:3] - self.segment_start)
        self._setpoint(commanded, target[3])


if __name__ == "__main__":
    rospy.init_node("independent_map_survey")
    IndependentMapSurvey()
    rospy.spin()
