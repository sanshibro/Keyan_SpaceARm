#!/usr/bin/env python3
import math
import time
from collections import deque

import numpy as np
import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray, Int32


class OrbAngleCalculator:
    def __init__(self):
        rospy.init_node(f"orb_angle_calculator_{int(time.time())}")

        self.pose_topic = rospy.get_param("~pose_topic", "/orb_slam3/odometry")
        self.state_topic = rospy.get_param(
            "~state_topic", "/orb_slam3/tracking_state"
        )
        self.zero_samples = int(rospy.get_param("~zero_samples", 30))
        self.theta_deadband_deg = float(
            rospy.get_param("~theta_deadband_deg", 2.0)
        )
        self.phi_offset_deg = float(rospy.get_param("~phi_offset_deg", 0.0))
        self.phi_sign = float(rospy.get_param("~phi_sign", 1.0))
        self.phi_range = rospy.get_param("~phi_range", "0_360")
        self.bend_axis = self.parse_axis(rospy.get_param("~bend_axis", "z"))
        self.use_tracking_gate = bool(rospy.get_param("~use_tracking_gate", False))
        self.max_bad_states = int(rospy.get_param("~max_bad_states", 5))

        self.zero_rotations = deque(maxlen=self.zero_samples)
        self.R0 = None
        self.tracking_state = None
        self.bad_state_count = 0
        self.reference_invalid = False

        self.angle_pub = rospy.Publisher(
            "/continuum_angles", Float32MultiArray, queue_size=10
        )
        self.valid_pub = rospy.Publisher(
            "/continuum_angles_valid", Int32, queue_size=10, latch=True
        )
        self.state_sub = rospy.Subscriber(
            self.state_topic, Int32, self.state_callback, queue_size=20
        )
        self.pose_sub = rospy.Subscriber(
            self.pose_topic, Odometry, self.pose_callback, queue_size=20
        )

        self.publish_valid(False)
        rospy.loginfo(
            "ORB angle calculator: pose=%s state=%s axis=%s use_tracking_gate=%s",
            self.pose_topic,
            self.state_topic,
            self.bend_axis,
            self.use_tracking_gate,
        )
        rospy.loginfo(
            "Hold the arm at its straight zero pose until calibration finishes."
        )

    def state_callback(self, msg):
        self.tracking_state = msg.data
        if not self.use_tracking_gate:
            return

        if msg.data == 2:
            self.bad_state_count = 0
            return

        self.bad_state_count += 1
        self.publish_valid(False)
        if self.R0 is not None and self.bad_state_count >= self.max_bad_states:
            self.reference_invalid = True
            rospy.logerr_throttle(
                1.0,
                "ORB tracking was lost. Angle reference is invalid; "
                "return to the straight pose and restart this script.",
            )

    def pose_callback(self, msg):
        if self.use_tracking_gate and self.tracking_state != 2:
            rospy.logwarn_throttle(
                1.0, "Waiting for valid ORB tracking_state=2."
            )
            return
        if self.use_tracking_gate and self.reference_invalid:
            return

        q = np.array([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        ], dtype=float)
        norm = np.linalg.norm(q)
        if abs(norm - 1.0) > 0.05:
            self.publish_valid(False)
            return

        R = self.quat_to_rotation(q / norm)
        if self.R0 is None:
            self.zero_rotations.append(R)
            if len(self.zero_rotations) < self.zero_samples:
                rospy.loginfo_throttle(
                    1.0,
                    "Calibrating zero pose: %d/%d",
                    len(self.zero_rotations),
                    self.zero_samples,
                )
                return
            self.R0 = self.average_rotation(list(self.zero_rotations))
            rospy.loginfo("Zero pose calibrated. Start bending.")
            return

        R_rel = self.R0.T @ R
        tangent = R_rel @ self.bend_axis
        tangent /= max(np.linalg.norm(tangent), 1e-12)

        theta = math.degrees(math.acos(float(np.clip(
            np.dot(tangent, self.bend_axis), -1.0, 1.0
        ))))

        lateral = tangent - np.dot(tangent, self.bend_axis) * self.bend_axis
        if np.linalg.norm(lateral) < 1e-8 or theta < self.theta_deadband_deg:
            phi = 0.0
        else:
            phi = math.degrees(
                math.atan2(float(lateral[1]), float(lateral[0]))
            )
            phi = self.normalize_phi(self.phi_sign * phi + self.phi_offset_deg)

        msg_out = Float32MultiArray()
        msg_out.data = [float(theta), float(phi)]
        self.angle_pub.publish(msg_out)
        self.publish_valid(True)
        rospy.loginfo_throttle(
            0.2, "theta=%.2f deg, phi=%.2f deg, tracking=OK", theta, phi
        )

    def publish_valid(self, valid):
        msg = Int32()
        msg.data = 1 if valid else 0
        self.valid_pub.publish(msg)

    def normalize_phi(self, phi):
        phi = phi % 360.0
        if self.phi_range == "signed" and phi > 180.0:
            phi -= 360.0
        return phi

    @staticmethod
    def parse_axis(name):
        axes = {
            "x": np.array([1.0, 0.0, 0.0]),
            "y": np.array([0.0, 1.0, 0.0]),
            "z": np.array([0.0, 0.0, 1.0]),
            "-x": np.array([-1.0, 0.0, 0.0]),
            "-y": np.array([0.0, -1.0, 0.0]),
            "-z": np.array([0.0, 0.0, -1.0]),
        }
        if name not in axes:
            raise ValueError("bend_axis must be x, y, z, -x, -y or -z")
        return axes[name]

    @staticmethod
    def quat_to_rotation(q):
        x, y, z, w = q
        return np.array([
            [1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
            [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
            [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)],
        ])

    @staticmethod
    def average_rotation(rotations):
        mean = np.mean(np.stack(rotations), axis=0)
        U, _, Vt = np.linalg.svd(mean)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        return R


if __name__ == "__main__":
    OrbAngleCalculator()
    rospy.spin()
