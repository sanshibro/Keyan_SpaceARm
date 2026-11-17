#!/usr/bin/env python3
"""Anchor a non-truth flight-state estimate and publish the optical camera pose.

The first estimator pose is aligned with the task's known start pose.  No
Gazebo pose or obstacle geometry is consumed.  This is intended for fair
frozen sensor-map acquisition, not for claiming standalone SLAM localization.
"""

import json
import math

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String


def quaternion_normalize(q):
    q = np.asarray(q, dtype=float)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("zero quaternion")
    return q / norm


def quaternion_multiply(left, right):
    x1, y1, z1, w1 = quaternion_normalize(left)
    x2, y2, z2, w2 = quaternion_normalize(right)
    return quaternion_normalize([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def rotate_vector(q, vector):
    x, y, z, w = quaternion_normalize(q)
    rotation = np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=float)
    return rotation.dot(np.asarray(vector, dtype=float))


def quaternion_from_rpy(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return quaternion_normalize([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ])


def quaternion_yaw(q):
    x, y, z, w = quaternion_normalize(q)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


class AnchoredEstimatorCameraPose:
    def __init__(self):
        self.world_frame = str(rospy.get_param("~world_frame", "world"))
        self.start_position = np.asarray([
            float(rospy.get_param("~initial_x", 0.0)),
            float(rospy.get_param("~initial_y", 0.0)),
            float(rospy.get_param("~initial_z", 1.0)),
        ], dtype=float)
        self.start_yaw = float(rospy.get_param("~initial_yaw", 0.0))
        self.camera_translation = np.asarray([
            float(rospy.get_param("~camera_x", 0.10)),
            float(rospy.get_param("~camera_y", 0.0)),
            float(rospy.get_param("~camera_z", 0.035)),
        ], dtype=float)
        # Gazebo OpenNI publishes optical coordinates while labelling the
        # cloud camera_link: z forward, x right, y down.
        self.camera_orientation = quaternion_from_rpy(
            float(rospy.get_param("~camera_roll", -math.pi / 2.0)),
            float(rospy.get_param("~camera_pitch", 0.0)),
            float(rospy.get_param("~camera_yaw", -math.pi / 2.0)),
        )
        self.initial_local_position = None
        self.initial_local_orientation = None
        self.publisher = rospy.Publisher(
            rospy.get_param("~output_topic", "/slam/estimator_camera_pose"),
            PoseStamped, queue_size=20,
        )
        self.body_publisher = rospy.Publisher(
            rospy.get_param("~body_output_topic", "/slam/estimator_body_pose"),
            PoseStamped, queue_size=20,
        )
        self.status = rospy.Publisher(
            rospy.get_param("~status_topic", "/slam/localization_status"),
            String, queue_size=1, latch=True
        )
        rospy.Subscriber(
            rospy.get_param(
                "~estimator_pose_topic", "/iris_0/mavros/local_position/pose"
            ),
            PoseStamped, self._pose_cb, queue_size=50,
        )

    @staticmethod
    def _components(message):
        p = message.pose.position
        q = message.pose.orientation
        return np.asarray([p.x, p.y, p.z], dtype=float), quaternion_normalize(
            [q.x, q.y, q.z, q.w]
        )

    def _pose_cb(self, message):
        local_position, local_orientation = self._components(message)
        if self.initial_local_position is None:
            self.initial_local_position = local_position.copy()
            self.initial_local_orientation = local_orientation.copy()
            self.status.publish(String(data=json.dumps({
                "state": "ANCHORED",
                "localization_source": "px4_ekf_sensor_fusion",
                "truth_pose_access": False,
                "initial_local_position": local_position.tolist(),
                "known_world_start": self.start_position.tolist(),
                "alignment": "translation_plus_yaw_gravity_aligned",
            }, sort_keys=True)))

        # The onboard estimator is already gravity aligned. Using the full
        # first-frame inverse would freeze transient ground roll/pitch into
        # the world axes, converting long horizontal travel into false height
        # drift. Anchor only the unobservable yaw and translation.
        alignment_orientation = quaternion_from_rpy(
            0.0, 0.0,
            self.start_yaw - quaternion_yaw(self.initial_local_orientation),
        )
        body_orientation = quaternion_multiply(
            alignment_orientation, local_orientation
        )
        body_position = self.start_position + rotate_vector(
            alignment_orientation, local_position - self.initial_local_position
        )
        camera_position = body_position + rotate_vector(
            body_orientation, self.camera_translation
        )
        camera_orientation = quaternion_multiply(
            body_orientation, self.camera_orientation
        )

        body_output = PoseStamped()
        body_output.header.stamp = message.header.stamp
        body_output.header.frame_id = self.world_frame
        body_output.pose.position.x = float(body_position[0])
        body_output.pose.position.y = float(body_position[1])
        body_output.pose.position.z = float(body_position[2])
        body_output.pose.orientation.x = float(body_orientation[0])
        body_output.pose.orientation.y = float(body_orientation[1])
        body_output.pose.orientation.z = float(body_orientation[2])
        body_output.pose.orientation.w = float(body_orientation[3])
        self.body_publisher.publish(body_output)

        output = PoseStamped()
        output.header.stamp = message.header.stamp
        output.header.frame_id = self.world_frame
        output.pose.position.x, output.pose.position.y, output.pose.position.z = (
            camera_position.tolist()
        )
        output.pose.orientation.x = float(camera_orientation[0])
        output.pose.orientation.y = float(camera_orientation[1])
        output.pose.orientation.z = float(camera_orientation[2])
        output.pose.orientation.w = float(camera_orientation[3])
        self.publisher.publish(output)


if __name__ == "__main__":
    rospy.init_node("anchored_estimator_camera_pose")
    AnchoredEstimatorCameraPose()
    rospy.spin()
