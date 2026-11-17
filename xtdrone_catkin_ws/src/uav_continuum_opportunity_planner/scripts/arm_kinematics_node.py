#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import rospy
from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker

from uav_continuum_opportunity_planner.common import (
    color,
    load_arm_params,
    make_pose_stamped,
    marker_header,
    point_msg,
    world_arm_points,
)


class ArmKinematicsNode:
    def __init__(self):
        self.frame_id = rospy.get_param("/frame_id", "world")
        self.arm_params = load_arm_params()
        self.theta_phi = [0.0, 0.0]
        self.uav_pose = None

        uav_pose_topic = rospy.get_param("~uav_pose_topic", "/iris_0/mavros/local_position/pose")
        self.tip_pub = rospy.Publisher("/arm/tip_pose", PoseStamped, queue_size=10)
        self.points_pub = rospy.Publisher("/arm/backbone_points", Float32MultiArray, queue_size=10)
        self.marker_pub = rospy.Publisher("/arm/backbone_marker", Marker, queue_size=10)

        rospy.Subscriber(uav_pose_topic, PoseStamped, self._pose_cb, queue_size=1)
        rospy.Subscriber("/arm/current_theta_phi", Float32MultiArray, self._shape_cb, queue_size=10)
        rospy.loginfo("arm_kinematics_node: pose=%s, arm hangs from mount_xyz=%s", uav_pose_topic, self.arm_params["mount_xyz"])

    def _pose_cb(self, msg):
        self.uav_pose = msg.pose

    def _shape_cb(self, msg):
        if len(msg.data) >= 2:
            self.theta_phi = [float(msg.data[0]), float(msg.data[1])]

    def _publish(self):
        if self.uav_pose is None:
            return
        stamp = rospy.Time.now()
        pts = world_arm_points(self.uav_pose, self.theta_phi[0], self.theta_phi[1], self.arm_params)
        tip = pts[-1]

        self.tip_pub.publish(make_pose_stamped(self.frame_id, stamp, tip))

        arr = Float32MultiArray()
        for p in pts:
            arr.data.extend([p[0], p[1], p[2]])
        self.points_pub.publish(arr)

        marker = Marker()
        marker_header(marker, self.frame_id, stamp, "arm_backbone", 0, Marker.LINE_STRIP)
        marker.scale.x = self.arm_params["radius"] * 2.0
        marker.color = color(0.0, 0.8, 0.2, 1.0)
        marker.points = [Point(x=p[0], y=p[1], z=p[2]) for p in pts]
        self.marker_pub.publish(marker)

        tip_marker = Marker()
        marker_header(tip_marker, self.frame_id, stamp, "arm_tip", 1, Marker.SPHERE)
        tip_marker.pose.position = point_msg(*tip)
        tip_marker.pose.orientation.w = 1.0
        tip_marker.scale.x = 0.07
        tip_marker.scale.y = 0.07
        tip_marker.scale.z = 0.07
        tip_marker.color = color(1.0, 0.4, 0.0, 1.0)
        self.marker_pub.publish(tip_marker)

    def spin(self):
        rate = rospy.Rate(float(rospy.get_param("~rate_hz", 30.0)))
        while not rospy.is_shutdown():
            self._publish()
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("arm_kinematics_node")
    try:
        ArmKinematicsNode().spin()
    except rospy.ROSInterruptException:
        pass
