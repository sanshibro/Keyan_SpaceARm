#!/usr/bin/env python3
"""Single-source, lead-limited OFFBOARD takeoff/hold setpoint publisher."""

import math
import json
import time

import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from std_msgs.msg import String


class SafeTakeoffHold:
    def __init__(self):
        self.target = [
            float(rospy.get_param("~x", 0.0)),
            float(rospy.get_param("~y", 0.0)),
            float(rospy.get_param("~z", 0.65)),
        ]
        self.xy_rate = max(0.01, float(rospy.get_param("~xy_rate", 0.12)))
        self.z_rate = max(0.01, float(rospy.get_param("~z_rate", 0.10)))
        self.xy_lead = max(0.02, float(rospy.get_param("~xy_lead", 0.10)))
        self.z_lead = max(0.02, float(rospy.get_param("~z_lead", 0.10)))
        self.abort_xy_error = max(0.10, float(rospy.get_param("~abort_xy_error", 0.50)))
        self.abort_tilt = math.radians(max(10.0, float(rospy.get_param("~abort_tilt_deg", 35.0))))
        self.measured = None
        self.vehicle_state = None
        self.command = None
        self.aborting = False
        self.abort_reason = "none"
        self.last_wall = time.monotonic()
        self.publisher = rospy.Publisher(
            rospy.get_param("~setpoint_topic", "/iris_0/mavros/setpoint_position/local"),
            PoseStamped, queue_size=10,
        )
        self.status = rospy.Publisher("/planner/v2/takeoff_hold_status", String, queue_size=1, latch=True)
        rospy.Subscriber(
            rospy.get_param("~local_pose_topic", "/iris_0/mavros/local_position/pose"),
            PoseStamped, self._pose_cb, queue_size=10,
        )
        self.namespace = rospy.get_param("~mavros_namespace", "/iris_0/mavros").rstrip("/")
        rospy.Subscriber(self.namespace + "/state", State, self._state_cb, queue_size=10)
        self.set_mode = rospy.ServiceProxy(self.namespace + "/set_mode", SetMode)
        self.arm = rospy.ServiceProxy(self.namespace + "/cmd/arming", CommandBool)
        rate = max(20.0, float(rospy.get_param("~rate_hz", 30.0)))
        self.timer = rospy.Timer(rospy.Duration(1.0 / rate), self._tick)

    def _pose_cb(self, message):
        self.measured = message
        if self.command is None:
            # For this scenario the spawn point and hold target must already
            # agree in x/y. Lock those axes immediately. Following the measured
            # x/y with a rate-limited command created a ratchet: a small PX4
            # drift moved the command in the same direction and could grow into
            # a runaway even though the nominal target stayed at the origin.
            self.command = [
                self.target[0], self.target[1], message.pose.position.z,
            ]

    def _state_cb(self, message):
        self.vehicle_state = message

    @staticmethod
    def _tilt(pose):
        q = pose.orientation
        body_z_dot_world_z = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        return math.acos(max(-1.0, min(1.0, body_z_dot_world_z)))

    def _trigger_abort(self, reason, xy_error, tilt):
        if self.aborting:
            return
        self.aborting = True
        self.abort_reason = reason
        rospy.logerr(
            "safe_takeoff_hold EMERGENCY_ABORT reason=%s xy_error=%.3f tilt_deg=%.1f",
            reason, xy_error, math.degrees(tilt),
        )
        self.status.publish(String(data=json.dumps({
            "state": "EMERGENCY_ABORT", "reason": reason,
            "xy_error": xy_error, "tilt_deg": math.degrees(tilt),
        }, sort_keys=True)))
        try:
            response = self.set_mode(base_mode=0, custom_mode="AUTO.LAND")
            rospy.logerr("safe_takeoff_hold AUTO.LAND requested accepted=%s", response.mode_sent)
        except rospy.ServiceException as exc:
            rospy.logerr("safe_takeoff_hold AUTO.LAND service failed: %s", exc)
        # A vehicle already beyond 60 degrees cannot produce controlled lift.
        # Request disarm after AUTO.LAND; PX4 may reject it while genuinely airborne.
        if tilt >= math.radians(60.0):
            try:
                response = self.arm(value=False)
                rospy.logerr("safe_takeoff_hold emergency disarm success=%s", response.success)
            except rospy.ServiceException as exc:
                rospy.logerr("safe_takeoff_hold disarm service failed: %s", exc)

    @staticmethod
    def _advance(current, target, maximum_step):
        return current + max(-maximum_step, min(maximum_step, target - current))

    def _tick(self, _event):
        if self.measured is None or self.command is None:
            return
        measured_pose = self.measured.pose
        xy_error = math.hypot(
            measured_pose.position.x - self.target[0],
            measured_pose.position.y - self.target[1],
        )
        tilt = self._tilt(measured_pose)
        if (self.vehicle_state is not None and self.vehicle_state.armed
                and (xy_error > self.abort_xy_error or tilt > self.abort_tilt)):
            reason = "xy_runaway" if xy_error > self.abort_xy_error else "excessive_tilt"
            self._trigger_abort(reason, xy_error, tilt)
        if self.aborting:
            return
        now_wall = time.monotonic()
        dt = max(0.0, min(0.10, now_wall - self.last_wall))
        self.last_wall = now_wall
        measured = self.measured.pose.position
        lead = (self.xy_lead, self.xy_lead, self.z_lead)
        rates = (self.xy_rate, self.xy_rate, self.z_rate)
        values = (measured.x, measured.y, measured.z)
        self.command[0] = self.target[0]
        self.command[1] = self.target[1]
        bounded_z = max(values[2] - lead[2], min(values[2] + lead[2], self.target[2]))
        self.command[2] = self._advance(self.command[2], bounded_z, rates[2] * dt)
        message = PoseStamped()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = self.measured.header.frame_id or "map"
        message.pose.position.x, message.pose.position.y, message.pose.position.z = self.command
        message.pose.orientation.w = 1.0
        self.publisher.publish(message)
        error = math.sqrt(sum((self.target[i] - values[i]) ** 2 for i in range(3)))
        rospy.loginfo_throttle(
            2.0, "safe_takeoff_hold: measured=[%.2f %.2f %.2f] command=[%.2f %.2f %.2f] error=%.2f",
            values[0], values[1], values[2], self.command[0], self.command[1], self.command[2], error,
        )
        self.status.publish(String(data=json.dumps({
            "state": "HOLDING", "armed": bool(self.vehicle_state and self.vehicle_state.armed),
            "mode": "unknown" if self.vehicle_state is None else self.vehicle_state.mode,
            "xy_error": xy_error, "tilt_deg": math.degrees(tilt), "target_error": error,
        }, sort_keys=True)))


if __name__ == "__main__":
    rospy.init_node("safe_takeoff_hold_node")
    SafeTakeoffHold()
    rospy.spin()
