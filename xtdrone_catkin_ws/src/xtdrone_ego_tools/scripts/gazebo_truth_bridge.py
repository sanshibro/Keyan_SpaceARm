#!/usr/bin/env python3
import rospy
import copy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import tf.transformations as tft


class GazeboTruthBridge:
    def __init__(self):
        self.model_name = rospy.get_param("~model_name", "iris_0")
        self.frame_id = rospy.get_param("~frame_id", "world")
        self.child_frame_id = rospy.get_param("~child_frame_id", self.model_name)
        self.use_initial_pose_as_origin = rospy.get_param("~use_initial_pose_as_origin", True)
        self.camera_offset_x = rospy.get_param("~camera_offset_x", 0.1)
        self.camera_offset_y = rospy.get_param("~camera_offset_y", 0.0)
        self.camera_offset_z = rospy.get_param("~camera_offset_z", 0.035)
        self.origin = None
        self.last_state_time = rospy.Time(0)
        self.last_missing_warn = rospy.Time(0)
        self.last_no_state_warn = rospy.Time(0)

        odom_topics = rospy.get_param(
            "~odom_topics",
            ["/vins_estimator/odometry", "/xtdrone/iris_0/vins_estimator/odometry"],
        )
        pose_topics = rospy.get_param("~pose_topics", ["/iris_0/camera_pose"])
        world_pose_topics = rospy.get_param("~world_pose_topics", ["/iris_0/world_pose"])
        extrinsic_topics = rospy.get_param("~extrinsic_topics", ["/vins_estimator/extrinsic"])

        self.odom_pubs = [rospy.Publisher(topic, Odometry, queue_size=10) for topic in odom_topics]
        self.pose_pubs = [rospy.Publisher(topic, PoseStamped, queue_size=10) for topic in pose_topics]
        self.world_pose_pubs = [rospy.Publisher(topic, PoseStamped, queue_size=10) for topic in world_pose_topics]
        self.extrinsic_pubs = [rospy.Publisher(topic, Odometry, queue_size=10) for topic in extrinsic_topics]

        rospy.Subscriber("/gazebo/model_states", ModelStates, self._states_cb, queue_size=1)
        rospy.Timer(rospy.Duration(2.0), self._watchdog_cb)
        rospy.loginfo(
            "gazebo_truth_bridge: model %s -> odom %s, pose %s, world_pose %s",
            self.model_name,
            ", ".join(odom_topics),
            ", ".join(pose_topics),
            ", ".join(world_pose_topics),
        )

    def _states_cb(self, msg):
        self.last_state_time = rospy.Time.now()
        try:
            idx = msg.name.index(self.model_name)
        except ValueError:
            now = rospy.Time.now()
            if (now - self.last_missing_warn).to_sec() > 5.0:
                rospy.logwarn(
                    "gazebo_truth_bridge: model '%s' not found in /gazebo/model_states. Available models: %s",
                    self.model_name,
                    ", ".join(msg.name[:30]),
                )
                self.last_missing_warn = now
            return

        model_pose = msg.pose[idx]
        if self.origin is None:
            self.origin = model_pose.position
            rospy.loginfo(
                "gazebo_truth_bridge: origin set to x=%.3f y=%.3f z=%.3f",
                self.origin.x,
                self.origin.y,
                self.origin.z,
            )

        stamp = rospy.Time.now()

        body_quat = [
            model_pose.orientation.x,
            model_pose.orientation.y,
            model_pose.orientation.z,
            model_pose.orientation.w,
        ]
        body_tf = tft.quaternion_matrix(body_quat)
        body_tf[0, 3] = model_pose.position.x
        body_tf[1, 3] = model_pose.position.y
        body_tf[2, 3] = model_pose.position.z

        # EGO projects depth pixels in optical camera coordinates:
        # x right, y down, z forward. The PX4 iris body convention used here is
        # x forward, y left, z up, so this is the camera-to-body rotation.
        cam_to_body = tft.identity_matrix()
        cam_to_body[0:3, 0:3] = [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
        cam_to_body[0, 3] = self.camera_offset_x
        cam_to_body[1, 3] = self.camera_offset_y
        cam_to_body[2, 3] = self.camera_offset_z
        camera_tf = body_tf.dot(cam_to_body)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.frame_id
        odom.child_frame_id = self.child_frame_id
        odom.pose.pose = copy.deepcopy(model_pose)
        if self.use_initial_pose_as_origin:
            odom.pose.pose.position.x = model_pose.position.x - self.origin.x
            odom.pose.pose.position.y = model_pose.position.y - self.origin.y
            odom.pose.pose.position.z = model_pose.position.z - self.origin.z
        odom.twist.twist = msg.twist[idx]

        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self.frame_id
        pose.pose.position.x = camera_tf[0, 3]
        pose.pose.position.y = camera_tf[1, 3]
        pose.pose.position.z = camera_tf[2, 3]
        if self.use_initial_pose_as_origin:
            pose.pose.position.x -= self.origin.x
            pose.pose.position.y -= self.origin.y
            pose.pose.position.z -= self.origin.z
        cam_quat = tft.quaternion_from_matrix(camera_tf)
        pose.pose.orientation.x = cam_quat[0]
        pose.pose.orientation.y = cam_quat[1]
        pose.pose.orientation.z = cam_quat[2]
        pose.pose.orientation.w = cam_quat[3]

        for pub in self.odom_pubs:
            pub.publish(odom)
        for pub in self.pose_pubs:
            pub.publish(pose)

        world_pose = PoseStamped()
        world_pose.header.stamp = stamp
        world_pose.header.frame_id = self.frame_id
        world_pose.pose = copy.deepcopy(model_pose)
        for pub in self.world_pose_pubs:
            pub.publish(world_pose)

        extrinsic = Odometry()
        extrinsic.header.stamp = stamp
        extrinsic.header.frame_id = self.child_frame_id
        extrinsic.child_frame_id = "camera"
        extrinsic.pose.pose.position.x = self.camera_offset_x
        extrinsic.pose.pose.position.y = self.camera_offset_y
        extrinsic.pose.pose.position.z = self.camera_offset_z
        cam_to_body_quat = tft.quaternion_from_matrix(cam_to_body)
        extrinsic.pose.pose.orientation.x = cam_to_body_quat[0]
        extrinsic.pose.pose.orientation.y = cam_to_body_quat[1]
        extrinsic.pose.pose.orientation.z = cam_to_body_quat[2]
        extrinsic.pose.pose.orientation.w = cam_to_body_quat[3]
        for pub in self.extrinsic_pubs:
            pub.publish(extrinsic)

    def _watchdog_cb(self, _event):
        now = rospy.Time.now()
        if self.last_state_time == rospy.Time(0) and (now - self.last_no_state_warn).to_sec() > 5.0:
            rospy.logwarn("gazebo_truth_bridge: waiting for /gazebo/model_states")
            self.last_no_state_warn = now


if __name__ == "__main__":
    rospy.init_node("gazebo_truth_bridge")
    GazeboTruthBridge()
    rospy.spin()
