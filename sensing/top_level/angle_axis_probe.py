#!/usr/bin/env python3
import math
import time
from collections import deque

import numpy as np
import rospy
from nav_msgs.msg import Odometry


class AxisProbe:
    def __init__(self):
        rospy.init_node(f"vins_axis_probe_{int(time.time())}")
        self.pose_topic = rospy.get_param("~pose_topic", "/vins_estimator/odometry")
        self.zero_samples = int(rospy.get_param("~zero_samples", 30))
        self.theta_deadband_deg = float(rospy.get_param("~theta_deadband_deg", 2.0))

        self.axes = {
            "x": np.array([1.0, 0.0, 0.0]),
            "y": np.array([0.0, 1.0, 0.0]),
            "z": np.array([0.0, 0.0, 1.0]),
            "-x": np.array([-1.0, 0.0, 0.0]),
            "-y": np.array([0.0, -1.0, 0.0]),
            "-z": np.array([0.0, 0.0, -1.0]),
        }
        self.zero_rotations = deque(maxlen=self.zero_samples)
        self.R0 = None
        self.sub = rospy.Subscriber(self.pose_topic, Odometry, self.callback, queue_size=50)
        rospy.loginfo("Axis probe started. Hold zero pose still.")
        rospy.loginfo("pose_topic=%s zero_samples=%d", self.pose_topic, self.zero_samples)

    def callback(self, msg):
        q = np.array([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        ], dtype=float)
        if abs(np.linalg.norm(q) - 1.0) > 0.05:
            return
        R = self.quat_to_rotation(q)

        if self.R0 is None:
            self.zero_rotations.append(R)
            if len(self.zero_rotations) < self.zero_samples:
                rospy.loginfo_throttle(1.0, "Calibrating zero pose: %d/%d", len(self.zero_rotations), self.zero_samples)
                return
            self.R0 = self.average_rotation(list(self.zero_rotations))
            rospy.loginfo("Zero pose calibrated. Bend toward a known direction now.")
            return

        R_rel = self.R0.T @ R
        parts = []
        for name, axis in self.axes.items():
            theta, phi = self.angles_for_axis(R_rel, axis)
            if theta < self.theta_deadband_deg:
                phi = 0.0
            parts.append(f"{name}: theta={theta:6.2f}, phi={phi:6.1f}")
        rospy.loginfo_throttle(0.3, " | ".join(parts))

    @staticmethod
    def angles_for_axis(R_rel, axis):
        tangent = R_rel @ axis
        tangent = tangent / max(np.linalg.norm(tangent), 1e-12)
        theta = math.degrees(math.acos(float(np.clip(np.dot(tangent, axis), -1.0, 1.0))))
        # Express bend direction in the zero-frame XY plane for easy comparison.
        xy_norm = math.hypot(float(tangent[0]), float(tangent[1]))
        if xy_norm < 1e-8:
            phi = 0.0
        else:
            phi = math.degrees(math.atan2(float(tangent[1]), float(tangent[0]))) % 360.0
        return theta, phi

    @staticmethod
    def quat_to_rotation(q):
        x, y, z, w = q
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])

    @staticmethod
    def average_rotation(rotations):
        M = np.mean(np.stack(rotations, axis=0), axis=0)
        U, _, Vt = np.linalg.svd(M)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        return R


if __name__ == "__main__":
    AxisProbe()
    rospy.spin()
