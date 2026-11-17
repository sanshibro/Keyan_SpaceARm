#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import rospy
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from uav_continuum_opportunity_planner.common import (
    boxes_from_items,
    color,
    load_body_keepout_boxes,
    load_boxes,
    load_task_boxes_raw,
    load_world_boxes_raw,
    marker_header,
    point_msg,
)


class ObstacleGeometryNode:
    def __init__(self):
        self.frame_id = rospy.get_param("/frame_id", "world")
        self.world_boxes = boxes_from_items(load_world_boxes_raw())
        self.task_boxes = boxes_from_items(load_task_boxes_raw())
        self.boxes = load_boxes()
        self.body_keepout_boxes = load_body_keepout_boxes()
        self.marker_pub = rospy.Publisher("/world/obstacle_markers", MarkerArray, queue_size=1, latch=True)
        self.world_marker_pub = rospy.Publisher("/world/world_box_markers", MarkerArray, queue_size=1, latch=True)
        self.task_marker_pub = rospy.Publisher("/world/task_obstacle_markers", MarkerArray, queue_size=1, latch=True)
        self.info_pub = rospy.Publisher("/world/known_obstacles", String, queue_size=1, latch=True)
        rospy.loginfo(
            "obstacle_geometry_node: loaded %d boxes (%d world_sdf, %d task)",
            len(self.boxes),
            len(self.world_boxes),
            len(self.task_boxes),
        )

    def _outline_marker(self, box, stamp, marker_id, ns, rgba):
        outline = Marker()
        marker_header(outline, self.frame_id, stamp, ns, marker_id, Marker.LINE_LIST)
        outline.pose.orientation.w = 1.0
        outline.scale.x = 0.025
        outline.color = rgba

        cx, cy, cz = box.center
        hx, hy, hz = box.half
        corners = [
            (cx - hx, cy - hy, cz - hz),
            (cx + hx, cy - hy, cz - hz),
            (cx + hx, cy + hy, cz - hz),
            (cx - hx, cy + hy, cz - hz),
            (cx - hx, cy - hy, cz + hz),
            (cx + hx, cy - hy, cz + hz),
            (cx + hx, cy + hy, cz + hz),
            (cx - hx, cy + hy, cz + hz),
        ]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        for a, b in edges:
            outline.points.append(point_msg(*corners[a]))
            outline.points.append(point_msg(*corners[b]))
        return outline

    def publish(self):
        stamp = rospy.Time.now()
        arr = MarkerArray()
        world_arr = MarkerArray()
        task_arr = MarkerArray()
        marker_id = 0
        for box in self.world_boxes:
            marker = box.marker(self.frame_id, stamp, marker_id, color(0.8, 0.1, 0.1, 0.42))
            marker.ns = "world_sdf_obstacles"
            arr.markers.append(marker)
            world_arr.markers.append(marker)
            marker_id += 1
            outline = self._outline_marker(box, stamp, marker_id, "world_sdf_obstacle_edges", color(0.05, 0.05, 0.05, 0.9))
            arr.markers.append(outline)
            world_arr.markers.append(outline)
            marker_id += 1
        for box in self.task_boxes:
            marker = box.marker(self.frame_id, stamp, marker_id, color(1.0, 0.45, 0.0, 0.45))
            marker.ns = "task_obstacles"
            arr.markers.append(marker)
            task_arr.markers.append(marker)
            marker_id += 1
            outline = self._outline_marker(box, stamp, marker_id, "task_obstacle_edges", color(1.0, 0.85, 0.0, 0.95))
            arr.markers.append(outline)
            task_arr.markers.append(outline)
            marker_id += 1
        for idx, box in enumerate(self.body_keepout_boxes):
            marker = box.marker(self.frame_id, stamp, marker_id + idx, color(0.1, 0.35, 1.0, 0.25))
            marker.ns = "body_keepout"
            arr.markers.append(marker)
            task_arr.markers.append(marker)
            outline = self._outline_marker(box, stamp, marker_id + idx + 10000, "body_keepout_edges", color(0.2, 0.65, 1.0, 0.95))
            arr.markers.append(outline)
            task_arr.markers.append(outline)
        self.marker_pub.publish(arr)
        self.world_marker_pub.publish(world_arr)
        self.task_marker_pub.publish(task_arr)
        lines = []
        lines.append("world_sdf_boxes=%d task_boxes=%d total_boxes=%d" % (
            len(self.world_boxes),
            len(self.task_boxes),
            len(self.boxes),
        ))
        for box in self.world_boxes:
            lines.append("WORLD_SDF %s center=%s size=%s" % (box.name, list(box.center), list(box.size)))
        for box in self.task_boxes:
            lines.append("TASK %s center=%s size=%s" % (box.name, list(box.center), list(box.size)))
        for box in self.body_keepout_boxes:
            lines.append("BODY_KEEP OUT %s center=%s size=%s" % (box.name, list(box.center), list(box.size)))
        self.info_pub.publish(String(data="\n".join(lines)))

    def spin(self):
        rate = rospy.Rate(float(rospy.get_param("~rate_hz", 1.0)))
        while not rospy.is_shutdown():
            self.publish()
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("obstacle_geometry_node")
    try:
        ObstacleGeometryNode().spin()
    except rospy.ROSInterruptException:
        pass
