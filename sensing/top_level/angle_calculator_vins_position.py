#!/usr/bin/env python3
import math
import time
from collections import deque

import numpy as np
import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray


class VinsPositionBendCalculator:
    def __init__(self):
        rospy.init_node(f"vins_position_bend_{int(time.time())}")

        self.pose_topic = rospy.get_param("~pose_topic", "/vins_estimator/odometry")
        self.zero_samples = int(rospy.get_param("~zero_samples", 30))
        self.theta_deadband_deg = float(rospy.get_param("~theta_deadband_deg", 2.0))
        self.phi_offset_deg = float(rospy.get_param("~phi_offset_deg", 0.0))
        self.phi_sign = float(rospy.get_param("~phi_sign", 1.0))
        self.forward_axis = self.parse_axis(rospy.get_param("~forward_axis", "z"))

        self.zero_positions = deque(maxlen=self.zero_samples)
        self.zero_rotations = deque(maxlen=self.zero_samples)
        self.p0 = None
        self.R0 = None

        self.angle_pub = rospy.Publisher("/continuum_angles", Float32MultiArray, queue_size=10)
        self.length_pub = rospy.Publisher("/continuum_length", Float32MultiArray, queue_size=10)
        self.pose_sub = rospy.Subscriber(self.pose_topic, Odometry, self.pose_callback, queue_size=50)

        rospy.loginfo("VINS position bend calculator started.")
        rospy.loginfo("Hold the straight zero pose still until calibration finishes.")
        rospy.loginfo(
            "pose_topic=%s zero_samples=%d forward_axis=%s",
            self.pose_topic,
            self.zero_samples,
            self.forward_axis,
        )

    def pose_callback(self, msg):
        p = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        ], dtype=float)
        q = np.array([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        ], dtype=float)
        if abs(np.linalg.norm(q) - 1.0) > 0.05:
            return
        R = self.quat_to_rotation(q)

        if self.p0 is None:
            self.zero_positions.append(p)
            self.zero_rotations.append(R)
            if len(self.zero_positions) < self.zero_samples:
                rospy.loginfo_throttle(1.0, "Calibrating zero pose: %d/%d", len(self.zero_positions), self.zero_samples)
                return

            self.p0 = np.mean(np.stack(self.zero_positions, axis=0), axis=0)
            self.R0 = self.average_rotation(list(self.zero_rotations))
            rospy.loginfo("Zero pose calibrated. Start bending now.")
            return

        p_rel_world = p - self.p0
        p_rel = self.R0.T @ p_rel_world
        theta_deg, phi_deg, length = self.calculate_from_position(p_rel)

        if theta_deg < self.theta_deadband_deg:
            phi_deg = 0.0

        self.publish(self.angle_pub, [theta_deg, phi_deg])
        self.publish(self.length_pub, [length])
        rospy.loginfo_throttle(
            0.2,
            "theta=%.2f deg, phi=%.2f deg, rel=[%.4f %.4f %.4f], chord=%.4f",
            theta_deg,
            phi_deg,
            p_rel[0],
            p_rel[1],
            p_rel[2],
            np.linalg.norm(p_rel),
        )

    def calculate_from_position(self, p_rel):
        z = float(np.dot(p_rel, self.forward_axis))
        lateral_vec = p_rel - z * self.forward_axis
        lateral = float(np.linalg.norm(lateral_vec))

        theta = 2.0 * math.degrees(math.atan2(lateral, max(abs(z), 1e-9)))
        if lateral < 1e-8:
            phi = 0.0
        else:
            phi = math.degrees(math.atan2(float(lateral_vec[1]), float(lateral_vec[0])))
            phi = (self.phi_sign * phi + self.phi_offset_deg) % 360.0

        theta_rad = math.radians(theta)
        chord = float(np.linalg.norm(p_rel))
        if theta_rad < 1e-6:
            length = abs(z)
        else:
            length = chord * theta_rad / (2.0 * math.sin(theta_rad / 2.0))

        return theta, phi, length

    @staticmethod
    def publish(pub, values):
        msg = Float32MultiArray()
        msg.data = [float(v) for v in values]
        pub.publish(msg)

    @staticmethod
    def parse_axis(axis_name):
        axes = {
            "x": np.array([1.0, 0.0, 0.0]),
            "y": np.array([0.0, 1.0, 0.0]),
            "z": np.array([0.0, 0.0, 1.0]),
            "-x": np.array([-1.0, 0.0, 0.0]),
            "-y": np.array([0.0, -1.0, 0.0]),
            "-z": np.array([0.0, 0.0, -1.0]),
        }
        if axis_name not in axes:
            raise ValueError("forward_axis must be one of: x, y, z, -x, -y, -z")
        return axes[axis_name]

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
    VinsPositionBendCalculator()
    rospy.spin()
