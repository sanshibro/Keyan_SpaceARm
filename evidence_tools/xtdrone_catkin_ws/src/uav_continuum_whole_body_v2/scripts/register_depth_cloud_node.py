#!/usr/bin/env python3
"""Register a depth point cloud using an externally supplied camera pose.

This node performs no localization.  The experiment manifest must identify
whether the pose topic comes from Gazebo truth (sensor-mapping experiment) or
from an audited SLAM/VIO/LIO estimator (formal online-SLAM experiment).
"""

from collections import deque
import math
from pathlib import Path
import sys
import threading

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "src"))

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

from uav_continuum_whole_body_v2.mapping import transform_points


class RegisterDepthCloudNode:
    def __init__(self):
        self.output_frame = str(rospy.get_param("~output_frame", "world"))
        self.pose_source = str(rospy.get_param("~pose_source", "topic")).strip().lower()
        if self.pose_source not in ("topic", "tf"):
            raise ValueError("~pose_source must be topic or tf")
        self.maximum_pose_age = float(rospy.get_param("~maximum_pose_age_s", 0.10))
        self.tf_timeout = float(rospy.get_param("~tf_lookup_timeout_s", 0.10))
        self.maximum_points = max(100, int(rospy.get_param("~maximum_points", 30000)))
        self.maximum_free_rays = max(0, int(rospy.get_param("~maximum_free_rays", 120)))
        self.free_ray_range = max(1.0, float(rospy.get_param("~free_ray_range_m", 8.0)))
        self.horizontal_fov = float(rospy.get_param("~horizontal_fov_rad", 1.02974))
        self.lock = threading.Lock()
        self.poses = deque(maxlen=200)
        self.cloud_pub = rospy.Publisher(
            rospy.get_param("~registered_cloud_topic", "/slam/registered_cloud"),
            PointCloud2, queue_size=1,
        )
        self.free_ray_pub = rospy.Publisher(
            rospy.get_param(
                "~registered_free_ray_topic", "/slam/registered_free_ray_endpoints"
            ), PointCloud2, queue_size=1,
        )
        self.pose_pub = rospy.Publisher(
            rospy.get_param("~registered_pose_topic", "/slam/sensor_pose"),
            PoseStamped, queue_size=10,
        )
        self.tf_buffer = None
        self.tf_listener = None
        if self.pose_source == "tf":
            self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        else:
            rospy.Subscriber(
                rospy.get_param("~camera_pose_topic", "/iris_0/camera_pose"),
                PoseStamped, self._pose_cb, queue_size=100,
            )
        rospy.Subscriber(
            rospy.get_param("~depth_cloud_topic", "/camera/depth/points"),
            PointCloud2, self._cloud_cb, queue_size=1,
            buff_size=32 * 1024 * 1024,
        )

    @staticmethod
    def _xyz_array(message):
        """Return organized XYZ data without a 307k-element Python loop.

        Gazebo's depth camera publishes a conventional FLOAT32 PointCloud2.
        Keeping the organized layout is important because no-return rays are
        selected by image pixel.  A standards-compliant generator fallback is
        retained for unusual field layouts.
        """
        fields = {field.name: field for field in message.fields}
        xyz = [fields.get(name) for name in ("x", "y", "z")]
        if all(
            field is not None
            and field.datatype == PointField.FLOAT32
            and field.count == 1
            for field in xyz
        ):
            endian = ">" if message.is_bigendian else "<"
            dtype = np.dtype({
                "names": ["x", "y", "z"],
                "formats": [endian + "f4"] * 3,
                "offsets": [field.offset for field in xyz],
                "itemsize": int(message.point_step),
            })
            height = max(1, int(message.height))
            width = int(message.width)
            organized = np.ndarray(
                shape=(height, width), dtype=dtype, buffer=message.data,
                strides=(int(message.row_step), int(message.point_step)),
            )
            return np.column_stack((
                organized["x"].ravel(),
                organized["y"].ravel(),
                organized["z"].ravel(),
            ))
        return np.asarray([
            row[:3] for row in point_cloud2.read_points(
                message, field_names=("x", "y", "z"), skip_nans=False,
            )
        ], dtype=float).reshape((-1, 3))

    def _pose_cb(self, message):
        if (message.header.frame_id or self.output_frame) != self.output_frame:
            rospy.logerr_throttle(
                2.0, "camera pose must be in %s, got %s",
                self.output_frame, message.header.frame_id,
            )
            return
        with self.lock:
            self.poses.append(message)

    def _nearest_pose(self, stamp):
        with self.lock:
            poses = list(self.poses)
        if not poses:
            return None, float("inf")
        pose = min(poses, key=lambda item: abs((item.header.stamp - stamp).to_sec()))
        return pose, abs((pose.header.stamp - stamp).to_sec())

    def _tf_pose(self, source_frame, stamp):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.output_frame, source_frame, stamp,
                rospy.Duration(self.tf_timeout),
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as error:
            rospy.logwarn_throttle(
                2.0, "depth registration waiting for SLAM TF %s <- %s: %s",
                self.output_frame, source_frame, error,
            )
            return None
        pose = PoseStamped()
        pose.header = transform.header
        pose.pose.position.x = transform.transform.translation.x
        pose.pose.position.y = transform.transform.translation.y
        pose.pose.position.z = transform.transform.translation.z
        pose.pose.orientation = transform.transform.rotation
        return pose

    def _cloud_cb(self, message):
        if rospy.is_shutdown():
            return
        if self.pose_source == "tf":
            pose = self._tf_pose(message.header.frame_id, message.header.stamp)
            if pose is None:
                return
        else:
            pose, age = self._nearest_pose(message.header.stamp)
            if pose is None or age > self.maximum_pose_age:
                rospy.logwarn_throttle(
                    2.0, "depth registration waiting for synchronized camera pose (age %.3fs)",
                    age,
                )
                return
        raw_points = self._xyz_array(message)
        finite = np.all(np.isfinite(raw_points), axis=1)
        ranges = np.full(len(raw_points), np.inf, dtype=float)
        ranges[finite] = np.linalg.norm(raw_points[finite], axis=1)
        no_return = (~finite) | (ranges >= self.free_ray_range - 1e-3)
        hit_mask = finite & ~no_return
        if message.width > 1 and message.height > 1:
            # Preserve the camera's two-dimensional angular coverage. A flat
            # 1-D stride creates diagonal stripes (especially at 160x120) and
            # leaves false unknown holes even when every depth pixel returned.
            hit_grid_step = max(1, int(math.ceil(math.sqrt(
                float(message.width * message.height) / self.maximum_points
            ))))
            hit_rows = np.arange(0, int(message.height), hit_grid_step, dtype=int)
            hit_cols = np.arange(0, int(message.width), hit_grid_step, dtype=int)
            hit_vv, hit_uu = np.meshgrid(hit_rows, hit_cols, indexing="ij")
            hit_flat = (
                hit_vv.ravel() * int(message.width) + hit_uu.ravel()
            ).astype(int)
            hit_flat = hit_flat[(hit_flat < len(hit_mask)) & hit_mask[hit_flat]]
            points = raw_points[hit_flat]
        else:
            points = raw_points[hit_mask]
            stride = max(1, int(np.ceil(float(len(points)) / self.maximum_points)))
            points = points[::stride]
        if not len(points):
            points = np.empty((0, 3), dtype=float)
        position = pose.pose.position
        orientation = pose.pose.orientation
        registered = transform_points(
            points,
            [position.x, position.y, position.z],
            [orientation.x, orientation.y, orientation.z, orientation.w],
        )
        header = Header(stamp=message.header.stamp, frame_id=self.output_frame)
        if rospy.is_shutdown():
            return
        # Publish the synchronized origin first.  The mapper keeps a short
        # timestamped origin history and will not silently register a cloud
        # against an unrelated latest pose.
        registered_pose = PoseStamped(header=header, pose=pose.pose)
        self.pose_pub.publish(registered_pose)
        if self.maximum_free_rays and message.width > 1 and message.height > 1:
            grid_step = max(1, int(math.ceil(math.sqrt(
                float(message.width * message.height) / self.maximum_free_rays
            ))))
            rows = np.arange(0, int(message.height), grid_step, dtype=int)
            cols = np.arange(0, int(message.width), grid_step, dtype=int)
            vv, uu = np.meshgrid(rows, cols, indexing="ij")
            flat = (vv.ravel() * int(message.width) + uu.ravel()).astype(int)
            flat = flat[flat < len(finite)]
            missing = flat[no_return[flat]]
            if len(missing):
                u = (missing % int(message.width)).astype(float)
                v = (missing // int(message.width)).astype(float)
                focal = 0.5 * float(message.width) / math.tan(0.5 * self.horizontal_fov)
                directions = np.column_stack((
                    (u - 0.5 * (message.width - 1)) / focal,
                    (v - 0.5 * (message.height - 1)) / focal,
                    np.ones(len(missing)),
                ))
                local_free = directions * (
                    self.free_ray_range
                    / np.linalg.norm(directions, axis=1)
                )[:, None]
                registered_free = transform_points(
                    local_free,
                    [position.x, position.y, position.z],
                    [orientation.x, orientation.y, orientation.z, orientation.w],
                )
                self.free_ray_pub.publish(point_cloud2.create_cloud_xyz32(
                    header, registered_free.tolist()
                ))
        self.cloud_pub.publish(point_cloud2.create_cloud_xyz32(
            header, registered.tolist()
        ))
        if len(registered):
            rospy.loginfo_throttle(
                5.0,
                "depth registration raw=%d hits=%d no_return=%d "
                "hit_range_p95=%.2f registered_z=[%.2f,%.2f]",
                len(raw_points), int(np.count_nonzero(finite & ~no_return)),
                int(np.count_nonzero(no_return)),
                float(np.percentile(ranges[finite & ~no_return], 95.0)),
                float(np.min(registered[:, 2])), float(np.max(registered[:, 2])),
            )


if __name__ == "__main__":
    rospy.init_node("register_depth_cloud_node")
    RegisterDepthCloudNode()
    rospy.spin()
