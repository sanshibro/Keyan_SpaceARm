#!/usr/bin/env python3
import math
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2


class EgoInputCheck:
    def __init__(self):
        self.odom = None
        self.camera_pose = None
        self.depth = None
        self.inflate = None
        rospy.Subscriber("/vins_estimator/odometry", Odometry, self._odom_cb, queue_size=1)
        rospy.Subscriber("/iris_0/camera_pose", PoseStamped, self._pose_cb, queue_size=1)
        rospy.Subscriber("/iris_0/realsense/depth_camera/depth/image_raw", Image, self._depth_cb, queue_size=1)
        rospy.Subscriber(
            "/iris_0_ego_planner_node/grid_map/occupancy_inflate",
            PointCloud2,
            self._inflate_cb,
            queue_size=1,
        )

    def _odom_cb(self, msg):
        self.odom = msg

    def _pose_cb(self, msg):
        self.camera_pose = msg

    def _depth_cb(self, msg):
        self.depth = msg

    def _inflate_cb(self, msg):
        self.inflate = msg

    def run(self):
        deadline = rospy.Time.now() + rospy.Duration(3.0)
        rate = rospy.Rate(20)
        while rospy.Time.now() < deadline and not rospy.is_shutdown():
            if self.odom and self.camera_pose and self.depth and self.inflate:
                break
            rate.sleep()

        print("EGO input check")
        print("  odom:", "OK" if self.odom else "MISSING")
        print("  camera_pose:", "OK" if self.camera_pose else "MISSING")
        print("  depth:", "OK" if self.depth else "MISSING")
        print("  occupancy_inflate:", "OK" if self.inflate else "MISSING")

        if not self.odom or not self.inflate:
            return

        pos = self.odom.pose.pose.position
        radius = rospy.get_param("~radius", 0.7)
        count = 0
        nearest = None
        for pt in pc2.read_points(self.inflate, field_names=("x", "y", "z"), skip_nans=True):
            dx = pt[0] - pos.x
            dy = pt[1] - pos.y
            dz = pt[2] - pos.z
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if nearest is None or dist < nearest:
                nearest = dist
            if dist <= radius:
                count += 1
        print("  odom position: x={:.3f} y={:.3f} z={:.3f}".format(pos.x, pos.y, pos.z))
        print("  nearest inflated point distance:", "none" if nearest is None else "{:.3f} m".format(nearest))
        print("  inflated points within {:.2f} m: {}".format(radius, count))
        if count > 0:
            print("  RESULT: current UAV position is inside/too close to inflated obstacles")
        else:
            print("  RESULT: current UAV position is clear in inflated map")


if __name__ == "__main__":
    rospy.init_node("check_ego_inputs")
    EgoInputCheck().run()
