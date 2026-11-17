#!/usr/bin/env python3
import rospy
from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Odometry


class ModelStateToOdom:
    def __init__(self):
        self.model_name = rospy.get_param("~model_name", "iris_0")
        self.frame_id = rospy.get_param("~frame_id", "world")
        self.child_frame_id = rospy.get_param("~child_frame_id", self.model_name)
        self.output_topic = rospy.get_param("~output_topic", "/vins_estimator/odometry")
        self.odom_pub = rospy.Publisher(self.output_topic, Odometry, queue_size=10)
        rospy.Subscriber("/gazebo/model_states", ModelStates, self._states_cb, queue_size=1)
        rospy.loginfo("model_state_to_odom: Gazebo model %s -> %s", self.model_name, self.output_topic)

    def _states_cb(self, msg):
        try:
            idx = msg.name.index(self.model_name)
        except ValueError:
            return
        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = self.frame_id
        odom.child_frame_id = self.child_frame_id
        odom.pose.pose = msg.pose[idx]
        odom.twist.twist = msg.twist[idx]
        self.odom_pub.publish(odom)


if __name__ == "__main__":
    rospy.init_node("model_state_to_odom")
    ModelStateToOdom()
    rospy.spin()
