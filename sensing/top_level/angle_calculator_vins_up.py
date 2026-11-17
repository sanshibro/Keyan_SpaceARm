#!/usr/bin/env python3
import math
import time
from collections import deque

import numpy as np
import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray


class VinsBendAngleCalculator:
    def __init__(self):
        rospy.init_node(f"vins_bend_angle_{int(time.time())}")

        self.pose_topic = rospy.get_param("~pose_topic", "/vins_estimator/odometry")
        self.zero_samples = int(rospy.get_param("~zero_samples", 30))
        self.phi_offset_deg = float(rospy.get_param("~phi_offset_deg", 0.0))
        self.phi_sign = float(rospy.get_param("~phi_sign", 1.0))
        self.theta_deadband_deg = float(rospy.get_param("~theta_deadband_deg", 2.0))
        self.bend_axis = self.parse_axis(rospy.get_param("~bend_axis", "z"))
        self.angle_mode = rospy.get_param("~angle_mode", "axis")

        self.zero_rotations = deque(maxlen=self.zero_samples)
        self.R0 = None
        self.p0 = None

        self.angle_pub = rospy.Publisher("/continuum_angles", Float32MultiArray, queue_size=10)
        self.length_pub = rospy.Publisher("/continuum_length", Float32MultiArray, queue_size=10)
        self.pose_sub = rospy.Subscriber(self.pose_topic, Odometry, self.pose_callback, queue_size=50)

        rospy.loginfo("VINS bend angle calculator started.")
        rospy.loginfo("Hold the camera/arm at the straight zero pose until calibration finishes.")
        rospy.loginfo(
            "pose_topic=%s zero_samples=%d theta_deadband_deg=%.2f bend_axis=%s angle_mode=%s",
            self.pose_topic,
            self.zero_samples,
            self.theta_deadband_deg,
            self.bend_axis,
            self.angle_mode,
        )

    def pose_callback(self, msg):
        q = np.array([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        ], dtype=float)
        p = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        ], dtype=float)

        if not self.valid_quaternion(q):
            return

        R = self.quat_to_rotation(q)

        if self.R0 is None:
            self.zero_rotations.append(R)
            if len(self.zero_rotations) < self.zero_samples:
                rospy.loginfo_throttle(1.0, "Calibrating zero pose: %d/%d", len(self.zero_rotations), self.zero_samples)
                return

            self.R0 = self.average_rotation(list(self.zero_rotations))
            self.p0 = p
            rospy.loginfo("Zero pose calibrated. Start bending now.")
            return

        R_rel = self.R0.T @ R
        p_rel = self.R0.T @ (p - self.p0)

        if self.angle_mode == "rotation":
            theta_deg, phi_deg = self.calculate_rotation_angles(R_rel)
        else:
            theta_deg, phi_deg = self.calculate_bend_angles(R_rel)
        if theta_deg < self.theta_deadband_deg:
            phi_deg = 0.0
        length = self.calculate_length(theta_deg, p_rel)

        self.publish(self.angle_pub, [theta_deg, phi_deg])
        self.publish(self.length_pub, [length])

        rospy.loginfo_throttle(
            0.2,
            "theta=%.2f deg, phi=%.2f deg, vins_rel=[%.4f %.4f %.4f], length=%.4f",
            theta_deg,
            phi_deg,
            p_rel[0],
            p_rel[1],
            p_rel[2],
            length,
        )

    def calculate_bend_angles(self, R_rel):
        # Current selected body/camera axis expressed in the startup zero frame.
        tangent = R_rel @ self.bend_axis
        tangent = tangent / max(np.linalg.norm(tangent), 1e-12)

        theta = math.degrees(math.acos(float(np.clip(tangent[2], -1.0, 1.0))))
        xy_norm = math.hypot(float(tangent[0]), float(tangent[1]))
        if xy_norm < 1e-8:
            phi = 0.0
        else:
            phi = math.degrees(math.atan2(float(tangent[1]), float(tangent[0])))
            phi = (self.phi_sign * phi + self.phi_offset_deg) % 360.0

        return theta, phi

    def calculate_rotation_angles(self, R_rel):
        cos_angle = float(np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0))
        theta = math.degrees(math.acos(cos_angle))

        axis = np.array([
            R_rel[2, 1] - R_rel[1, 2],
            R_rel[0, 2] - R_rel[2, 0],
            R_rel[1, 0] - R_rel[0, 1],
        ])
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-8:
            return theta, 0.0
        axis = axis / axis_norm

        bend_direction = np.cross(axis, self.bend_axis)
        bend_norm = np.linalg.norm(bend_direction)
        if bend_norm < 1e-8:
            phi = 0.0
        else:
            bend_direction = bend_direction / bend_norm
            phi = math.degrees(math.atan2(float(bend_direction[1]), float(bend_direction[0])))
            phi = (self.phi_sign * phi + self.phi_offset_deg) % 360.0

        return theta, phi

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
            raise ValueError("bend_axis must be one of: x, y, z, -x, -y, -z")
        return axes[axis_name]

    @staticmethod
    def calculate_length(theta_deg, p_rel):
        # Uses displacement along the zero forward axis. This is only meaningful
        # if VINS position is tracking the moving end point relative to the base.
        theta_rad = math.radians(theta_deg)
        if abs(math.sin(theta_rad)) < 1e-6:
            return abs(float(p_rel[2]))
        return abs((float(p_rel[2]) * theta_rad) / math.sin(theta_rad))

    @staticmethod
    def publish(pub, values):
        msg = Float32MultiArray()
        msg.data = [float(v) for v in values]
        pub.publish(msg)

    @staticmethod
    def valid_quaternion(q):
        norm = np.linalg.norm(q)
        if abs(norm - 1.0) > 0.05:
            rospy.logwarn_throttle(1.0, "Invalid quaternion norm: %.4f", norm)
            return False
        return True

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
    VinsBendAngleCalculator()
    rospy.spin()
