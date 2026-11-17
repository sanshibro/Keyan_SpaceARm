#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import rospy
from std_msgs.msg import String
from visualization_msgs.msg import Marker

from uav_continuum_opportunity_planner.common import color, marker_header


class OpportunityVisualizationNode:
    def __init__(self):
        self.frame_id = rospy.get_param("/frame_id", "world")
        self.last_text = "waiting for opportunity detector"
        self.pub = rospy.Publisher("/opportunity/status_marker", Marker, queue_size=10)
        rospy.Subscriber("/opportunity/mode_hint", String, self._status_cb, queue_size=10)

    def _status_cb(self, msg):
        self.last_text = msg.data

    def spin(self):
        rate = rospy.Rate(float(rospy.get_param("~rate_hz", 2.0)))
        while not rospy.is_shutdown():
            marker = Marker()
            marker_header(marker, self.frame_id, rospy.Time.now(), "opportunity_status", 0, Marker.TEXT_VIEW_FACING)
            marker.pose.position.x = 0.0
            marker.pose.position.y = 4.0
            marker.pose.position.z = 1.8
            marker.pose.orientation.w = 1.0
            marker.scale.z = 0.18
            marker.color = color(1.0, 1.0, 1.0, 1.0)
            marker.text = self.last_text
            self.pub.publish(marker)
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("opportunity_visualization_node")
    try:
        OpportunityVisualizationNode().spin()
    except rospy.ROSInterruptException:
        pass
