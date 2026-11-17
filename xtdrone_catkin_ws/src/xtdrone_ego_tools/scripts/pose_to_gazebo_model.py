#!/usr/bin/env python3
import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Pose, PoseStamped


class PoseToGazeboModel:
    def __init__(self):
        self.model_name = rospy.get_param("~model_name", "iris_0")
        self.reference_frame = rospy.get_param("~reference_frame", "world")
        self.input_topic = rospy.get_param("~input_topic", "/xtdrone/iris_0/cmd_pose_enu")
        self.last_pose = None
        self.set_model_state = None

        self.pose_sub = rospy.Subscriber(self.input_topic, Pose, self._pose_cb, queue_size=1)
        self.pose_stamped_sub = rospy.Subscriber(
            self.input_topic + "_stamped", PoseStamped, self._pose_stamped_cb, queue_size=1
        )
        rospy.loginfo("pose_to_gazebo_model: %s -> Gazebo model %s", self.input_topic, self.model_name)

    def _connect(self):
        if self.set_model_state is not None:
            return True
        try:
            rospy.wait_for_service("/gazebo/set_model_state", timeout=0.2)
            self.set_model_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
            return True
        except rospy.ROSException:
            return False

    def _pose_cb(self, msg):
        self._set_pose(msg)

    def _pose_stamped_cb(self, msg):
        self._set_pose(msg.pose)

    def _set_pose(self, pose):
        self.last_pose = pose
        if not self._connect():
            return
        state = ModelState()
        state.model_name = self.model_name
        state.pose = pose
        state.reference_frame = self.reference_frame
        try:
            self.set_model_state(state)
        except rospy.ServiceException as exc:
            rospy.logwarn_throttle(1.0, "set_model_state failed: %s", exc)
            self.set_model_state = None

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    rospy.init_node("pose_to_gazebo_model")
    PoseToGazeboModel().spin()
