#!/usr/bin/env python3
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import rospy
import tf.transformations as tft
from geometry_msgs.msg import PoseStamped
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, String
from visualization_msgs.msg import Marker, MarkerArray

from uav_continuum_opportunity_planner.common import (
    BoxObstacle,
    VoxelOccupancy,
    color,
    frange,
    load_arm_params,
    load_body_keepout_boxes,
    load_boxes,
    load_target,
    marker_header,
    point_distance,
    point_msg,
    world_arm_points,
)


class ReachabilityNode:
    def __init__(self):
        self.frame_id = rospy.get_param("/frame_id", "world")
        self.arm_params = load_arm_params()
        self.boxes = load_boxes()
        self.body_keepout_boxes = load_body_keepout_boxes()
        self.target = load_target()
        self.uav_pose = None
        self.current_theta_phi = [0.0, 0.0]

        vehicle = rospy.get_param("/vehicle", {})
        reach = rospy.get_param("/reachability", {})
        self.body_radius = float(vehicle.get("body_radius", 0.34))
        self.body_height = float(vehicle.get("body_height", self.body_radius * 2.0))
        self.body_clearance = float(vehicle.get("body_clearance", 0.12))
        self.keepout_uses_body_envelope = bool(rospy.get_param("~keepout_uses_body_envelope", False))
        self.arm_clearance = float(vehicle.get("arm_clearance", 0.04))
        self.use_static_cloud = bool(rospy.get_param("~use_static_cloud", False))
        self.require_static_cloud = bool(rospy.get_param("~require_static_cloud", False))
        self.static_cloud_topic = rospy.get_param("~static_cloud_topic", "/world/static_scene_cloud")
        self.static_cloud_search_radius = float(rospy.get_param("~static_cloud_search_radius", 0.35))
        self.static_cloud_query_padding = float(rospy.get_param("~static_cloud_query_padding", 0.05))
        self.static_cloud_stride = max(1, int(rospy.get_param("~static_cloud_stride", 1)))
        static_resolution = float(rospy.get_param("~static_cloud_resolution", 0.1))
        static_inflation = float(rospy.get_param("~static_cloud_inflation", 0.1))
        static_max_points = int(rospy.get_param("~static_cloud_max_points", 250000))
        self.static_occupancy = VoxelOccupancy(static_resolution, static_inflation, static_max_points)
        self.target_tolerance = float(reach.get("target_tolerance", 0.08))
        self.standoff_min = float(reach.get("candidate_standoff_min", 0.18))
        self.standoff_max = float(reach.get("candidate_standoff_max", 0.55))
        self.standoff_step = float(reach.get("candidate_standoff_step", 0.08))
        self.angle_step_deg = float(reach.get("candidate_angle_step_deg", 15.0))
        self.z_offsets = [float(v) for v in reach.get("candidate_z_offsets", [-0.15, 0.0, 0.15])]
        self.max_candidates = int(reach.get("max_candidates_to_publish", 80))
        self.body_margin_target = float(reach.get("candidate_body_margin_target", 0.14))
        self.auto_execute = bool(rospy.get_param("/reachability/auto_execute", reach.get("auto_execute", False)))
        self.last_best = None
        self.last_candidates = []
        self.last_safe_pose_count = 0
        self.last_diag = {}
        self.last_search_time = rospy.Time(0)
        self.search_period = float(rospy.get_param("~search_period", 20.0))
        self.logged_static_cloud = False
        self.static_cloud_loaded = False

        uav_pose_topic = rospy.get_param("~uav_pose_topic", "/iris_0/mavros/local_position/pose")
        rospy.Subscriber(uav_pose_topic, PoseStamped, self._pose_cb, queue_size=1)
        rospy.Subscriber("/arm/current_theta_phi", Float32MultiArray, self._shape_cb, queue_size=10)
        if self.use_static_cloud:
            rospy.Subscriber(self.static_cloud_topic, PointCloud2, self._static_cloud_cb, queue_size=1, buff_size=16 * 1024 * 1024)

        self.status_pub = rospy.Publisher("/opportunity/mode_hint", String, queue_size=10, latch=True)
        self.marker_pub = rospy.Publisher("/opportunity/reachability_markers", MarkerArray, queue_size=10)
        self.goal_pub = rospy.Publisher("/planner/uav_goal", PoseStamped, queue_size=10, latch=True)
        self.arm_cmd_pub = rospy.Publisher("/planner/arm_theta_phi_cmd", Float32MultiArray, queue_size=10, latch=True)
        self.candidates_pub = rospy.Publisher("/planner/reach_candidates", Float32MultiArray, queue_size=10, latch=True)
        self.exec_arm_pub = rospy.Publisher("/arm/theta_phi_cmd", Float32MultiArray, queue_size=10)

        rospy.loginfo(
            "reachability_node: target=%s, body radius %.2f, arm clearance %.2f",
            self.target["pose"],
            self.body_radius,
            self.arm_clearance,
        )

    def _pose_cb(self, msg):
        self.uav_pose = msg.pose

    def _shape_cb(self, msg):
        if len(msg.data) >= 2:
            self.current_theta_phi = [float(msg.data[0]), float(msg.data[1])]

    def _static_cloud_cb(self, msg):
        if self.static_cloud_loaded:
            return
        points = []
        for idx, point in enumerate(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)):
            if idx % self.static_cloud_stride == 0:
                points.append((point[0], point[1], point[2]))
        self.static_occupancy.update(points, msg.header.frame_id, msg.header.stamp)
        self.static_cloud_loaded = True
        if not self.logged_static_cloud:
            self.logged_static_cloud = True
            rospy.loginfo(
                "reachability_node: static cloud received points=%d occupied_voxels=%d frame=%s",
                len(points),
                len(self.static_occupancy.occupied),
                msg.header.frame_id,
            )

    def _pose_from_xyz_yaw(self, x, y, z, yaw):
        pose = PoseStamped()
        pose.header.frame_id = self.frame_id
        pose.header.stamp = rospy.Time.now()
        pose.pose.position = point_msg(x, y, z)
        q = tft.quaternion_from_euler(0.0, 0.0, yaw)
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]
        return pose

    def _pose_yaw(self, pose):
        q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        return tft.euler_from_quaternion(q)[2]

    def _clearance_ok(self, point, clearance):
        for box in self.boxes:
            if box.signed_distance(point) < clearance:
                return False
        return True

    def _expanded_box_margin(self, point, box, xy_clearance, z_clearance):
        dx = abs(point[0] - box.center[0]) - (box.half[0] + xy_clearance)
        dy = abs(point[1] - box.center[1]) - (box.half[1] + xy_clearance)
        dz = abs(point[2] - box.center[2]) - (box.half[2] + z_clearance)
        ox = max(dx, 0.0)
        oy = max(dy, 0.0)
        oz = max(dz, 0.0)
        outside = math.sqrt(ox * ox + oy * oy + oz * oz)
        inside = min(max(dx, max(dy, dz)), 0.0)
        return outside + inside

    def _box_clearance_ok(self, point, clearance):
        xy_clearance = self.body_radius + self.body_clearance
        z_clearance = 0.5 * self.body_height + self.body_clearance
        for box in self.boxes:
            if self._expanded_box_margin(point, box, xy_clearance, z_clearance) < 0.0:
                return False
        return True

    def _static_clearance_ok(self, point, clearance):
        if not self.use_static_cloud:
            return True
        if self.require_static_cloud and not self.static_occupancy.ready():
            return False
        query_radius = min(
            max(clearance + self.static_cloud_query_padding, self.static_occupancy.resolution),
            self.static_cloud_search_radius,
        )
        dist, _, _ = self.static_occupancy.signed_distance(
            point,
            query_radius,
        )
        return dist >= clearance

    def _full_clearance_ok(self, point, clearance):
        if not self._clearance_ok(point, clearance):
            return False
        if self.use_static_cloud:
            if not self._static_clearance_ok(point, clearance):
                return False
        return True

    def _body_safe(self, point):
        if not self._box_clearance_ok(point, self.body_radius + self.body_clearance):
            return False
        if self.use_static_cloud and not self._static_clearance_ok(point, self.body_radius + self.body_clearance):
            return False
        for box in self.body_keepout_boxes:
            if self._keepout_margin(point, box) < 0.0:
                return False
        return True

    def _body_margin(self, point):
        xy_clearance = self.body_radius + self.body_clearance
        z_clearance = 0.5 * self.body_height + self.body_clearance
        best = float("inf")
        for box in self.boxes:
            best = min(best, self._expanded_box_margin(point, box, xy_clearance, z_clearance))
        if self.use_static_cloud and self.static_occupancy.ready():
            query_radius = min(
                max(xy_clearance + self.static_cloud_query_padding, self.static_occupancy.resolution),
                self.static_cloud_search_radius,
            )
            dist, _, _ = self.static_occupancy.signed_distance(point, query_radius)
            best = min(best, dist - xy_clearance)
        for box in self.body_keepout_boxes:
            best = min(best, self._keepout_margin(point, box))
        return best

    def _keepout_margin(self, point, box):
        if not self.keepout_uses_body_envelope:
            return box.signed_distance(point)
        return self._expanded_box_margin(
            point, box,
            self.body_radius + self.body_clearance,
            0.5 * self.body_height + self.body_clearance,
        )

    def _arm_collision_free(self, points):
        threshold = self.arm_params["radius"] + self.arm_clearance
        for point in points:
            if not self._clearance_ok(point, threshold):
                return False
        return True

    def _arm_static_collision_free(self, points):
        threshold = self.arm_params["radius"] + self.arm_clearance
        if not self.use_static_cloud:
            return True
        sample_count = min(len(points), 6)
        if sample_count <= 0:
            return True
        for idx in range(sample_count):
            point = points[int(round(idx * (len(points) - 1) / float(max(sample_count - 1, 1))))]
            if not self._static_clearance_ok(point, threshold):
                return False
        return True

    def _candidate_poses(self):
        target = self.target["pose"]
        tx, ty, tz = target[0], target[1], target[2]
        poses = []
        diag = {
            "pose_total": 0,
            "pose_box_reject": 0,
            "pose_static_reject": 0,
            "pose_keepout_reject": 0,
            "pose_safe": 0,
        }
        body_clearance = self.body_radius + self.body_clearance
        for radius in frange(self.standoff_min, self.standoff_max, self.standoff_step):
            angle = -180.0
            while angle < 180.0 - 1e-9:
                rad = math.radians(angle)
                x = tx + radius * math.cos(rad)
                y = ty + radius * math.sin(rad)
                yaw = math.atan2(ty - y, tx - x)
                for dz in self.z_offsets:
                    z = tz + dz
                    point = (x, y, z)
                    diag["pose_total"] += 1
                    if not self._box_clearance_ok(point, body_clearance):
                        diag["pose_box_reject"] += 1
                        continue
                    if self.use_static_cloud and not self._static_clearance_ok(point, body_clearance):
                        diag["pose_static_reject"] += 1
                        continue
                    keepout_hit = False
                    for box in self.body_keepout_boxes:
                        if self._keepout_margin(point, box) < 0.0:
                            keepout_hit = True
                            break
                    if keepout_hit:
                        diag["pose_keepout_reject"] += 1
                        continue
                    diag["pose_safe"] += 1
                    poses.append(self._pose_from_xyz_yaw(x, y, z, yaw))
                angle += self.angle_step_deg
        self.last_diag.update(diag)
        return poses

    def _search(self):
        target_xyz = tuple(self.target["pose"][:3])
        best = None
        candidates = []
        safe_poses = self._candidate_poses()
        self.last_safe_pose_count = len(safe_poses)
        diag = {
            "arm_tests": 0,
            "arm_box_reject": 0,
            "tip_reject": 0,
            "arm_static_reject": 0,
            "reach_candidates": 0,
        }
        theta_values = frange(
            self.arm_params["theta_min_deg"],
            self.arm_params["theta_max_deg"],
            self.arm_params["theta_step_deg"],
        )
        phi_values = frange(
            self.arm_params["phi_min_deg"],
            self.arm_params["phi_max_deg"],
            self.arm_params["phi_step_deg"],
        )
        for pose_stamped in safe_poses:
            for theta in theta_values:
                for phi in phi_values:
                    diag["arm_tests"] += 1
                    pts = world_arm_points(pose_stamped.pose, theta, phi, self.arm_params)
                    tip_error = point_distance(pts[-1], target_xyz)
                    if tip_error > self.target_tolerance:
                        diag["tip_reject"] += 1
                        continue
                    if not self._arm_collision_free(pts):
                        diag["arm_box_reject"] += 1
                        continue
                    if not self._arm_static_collision_free(pts):
                        diag["arm_static_reject"] += 1
                        continue
                    body_dist = point_distance(
                        (
                            pose_stamped.pose.position.x,
                            pose_stamped.pose.position.y,
                            pose_stamped.pose.position.z,
                        ),
                        target_xyz,
                    )
                    body_margin = self._body_margin(
                        (
                            pose_stamped.pose.position.x,
                            pose_stamped.pose.position.y,
                            pose_stamped.pose.position.z,
                        )
                    )
                    body_deficit = max(0.0, self.body_margin_target - body_margin) / max(self.body_margin_target, 1e-3)
                    score = (
                        tip_error
                        + 0.05 * theta / max(self.arm_params["theta_max_deg"], 1.0)
                        - 0.02 * body_dist
                        + 2.5 * body_deficit * body_deficit
                        - 0.15 * min(body_margin, self.body_margin_target)
                    )
                    item = {
                        "pose": pose_stamped,
                        "theta": theta,
                        "phi": phi,
                        "tip_error": tip_error,
                        "body_dist": body_dist,
                        "body_margin": body_margin,
                        "score": score,
                    }
                    diag["reach_candidates"] += 1
                    candidates.append(item)
                    if best is None or score < best["score"]:
                        best = item
        candidates.sort(key=lambda c: c["score"])
        self.last_diag.update(diag)
        rospy.loginfo(
            "reachability_node: search summary safe_poses=%d arm_tests=%d tip_reject=%d arm_box_reject=%d arm_static_reject=%d reach_candidates=%d",
            len(safe_poses),
            int(diag["arm_tests"]),
            int(diag["tip_reject"]),
            int(diag["arm_box_reject"]),
            int(diag["arm_static_reject"]),
            int(diag["reach_candidates"]),
        )
        return best, candidates

    def _publish_markers(self, best, candidates):
        stamp = rospy.Time.now()
        arr = MarkerArray()

        target_marker = Marker()
        marker_header(target_marker, self.frame_id, stamp, "opportunity_target", 0, Marker.SPHERE)
        target_marker.pose.position = point_msg(*self.target["pose"][:3])
        target_marker.pose.orientation.w = 1.0
        target_marker.scale.x = 0.10
        target_marker.scale.y = 0.10
        target_marker.scale.z = 0.10
        target_marker.color = color(1.0, 0.0, 1.0, 1.0)
        arr.markers.append(target_marker)

        for idx, cand in enumerate(candidates[: self.max_candidates]):
            marker = Marker()
            marker_header(marker, self.frame_id, stamp, "reach_candidates", idx, Marker.SPHERE)
            marker.pose = cand["pose"].pose
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            marker.color = color(0.0, 0.6, 1.0, 0.5)
            arr.markers.append(marker)

        if best is not None:
            best_marker = Marker()
            marker_header(best_marker, self.frame_id, stamp, "selected_reach_candidate", 10000, Marker.ARROW)
            best_marker.pose = best["pose"].pose
            best_marker.scale.x = 0.35
            best_marker.scale.y = 0.05
            best_marker.scale.z = 0.05
            best_marker.color = color(0.0, 1.0, 0.0, 1.0)
            arr.markers.append(best_marker)

            pts = world_arm_points(best["pose"].pose, best["theta"], best["phi"], self.arm_params)
            arm_marker = Marker()
            marker_header(arm_marker, self.frame_id, stamp, "selected_arm_shape", 10001, Marker.LINE_STRIP)
            arm_marker.scale.x = self.arm_params["radius"] * 2.0
            arm_marker.color = color(1.0, 1.0, 0.0, 1.0)
            arm_marker.points = [point_msg(*p) for p in pts]
            arr.markers.append(arm_marker)

        self.marker_pub.publish(arr)

    def _publish_result(self, best, candidates):
        publish_candidates = candidates[: self.max_candidates]
        cand_msg = Float32MultiArray()
        cand_msg.layout.dim = [
            MultiArrayDimension(label="candidate", size=len(publish_candidates), stride=len(publish_candidates) * 9),
            MultiArrayDimension(label="x_y_z_yaw_theta_phi_tip_error_body_dist_score", size=9, stride=9),
        ]
        cand_msg.layout.data_offset = 0
        for cand in publish_candidates:
            pose = cand["pose"].pose
            cand_msg.data.extend(
                [
                    pose.position.x,
                    pose.position.y,
                    pose.position.z,
                    self._pose_yaw(pose),
                    cand["theta"],
                    cand["phi"],
                    cand["tip_error"],
                    cand["body_dist"],
                    cand["score"],
                ]
            )
        self.candidates_pub.publish(cand_msg)
        try:
            rospy.set_param(
                "/planner/latest_reach_candidates_cache",
                {
                    "stamp": rospy.Time.now().to_sec(),
                    "count": len(publish_candidates),
                    "stride": 9,
                    "data": list(cand_msg.data),
                },
            )
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "reachability_node: failed to cache reach candidates param: %s", exc)

        if self.use_static_cloud and self.require_static_cloud and not self.static_occupancy.ready():
            self.status_pub.publish(String(data="WAITING_STATIC_SCENE_CLOUD"))
            return
        if best is None:
            self.status_pub.publish(
                String(
                    data=(
                        "NO_REACH_IN_OPPORTUNITY safe_uav_poses=%d reach_candidates=0 static_cloud=%s "
                        "pose_total=%d pose_box_reject=%d pose_static_reject=%d pose_keepout_reject=%d "
                        "arm_tests=%d arm_box_reject=%d tip_reject=%d arm_static_reject=%d"
                    )
                    % (
                        self.last_safe_pose_count,
                        "ready" if self.static_occupancy.ready() else "disabled",
                        int(self.last_diag.get("pose_total", 0)),
                        int(self.last_diag.get("pose_box_reject", 0)),
                        int(self.last_diag.get("pose_static_reject", 0)),
                        int(self.last_diag.get("pose_keepout_reject", 0)),
                        int(self.last_diag.get("arm_tests", 0)),
                        int(self.last_diag.get("arm_box_reject", 0)),
                        int(self.last_diag.get("tip_reject", 0)),
                        int(self.last_diag.get("arm_static_reject", 0)),
                    )
                )
            )
            return
        msg = (
            "STANDOFF_REACH_IN candidates=%d theta=%.2f phi=%.2f tip_error=%.3f body_dist=%.3f static_cloud=%s"
            % (
                len(candidates),
                best["theta"],
                best["phi"],
                best["tip_error"],
                best["body_dist"],
                "ready" if self.static_occupancy.ready() else "disabled",
            )
        )
        self.status_pub.publish(String(data=msg))
        self.goal_pub.publish(best["pose"])
        cmd = Float32MultiArray()
        cmd.data = [best["theta"], best["phi"]]
        self.arm_cmd_pub.publish(cmd)
        if self.auto_execute:
            self.exec_arm_pub.publish(cmd)

    def spin(self):
        rate = rospy.Rate(float(rospy.get_param("~rate_hz", 5.0)))
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            if (
                (self.last_search_time == rospy.Time(0)
                 or (now - self.last_search_time).to_sec() >= self.search_period)
                and not (self.use_static_cloud and self.require_static_cloud and not self.static_occupancy.ready())
            ):
                self.last_search_time = now
                self.status_pub.publish(String(data="STATIC_CLOUD_READY_SEARCHING"))
                rospy.loginfo("reachability_node: search start")
                self.last_best, self.last_candidates = self._search()
                self.last_search_time = rospy.Time.now()
                rospy.loginfo(
                    "reachability_node: search done, candidates=%d, best=%s",
                    len(self.last_candidates),
                    "yes" if self.last_best is not None else "no",
                )
            self._publish_markers(self.last_best, self.last_candidates)
            self._publish_result(self.last_best, self.last_candidates)
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("reachability_node")
    try:
        ReachabilityNode().spin()
    except rospy.ROSInterruptException:
        pass
