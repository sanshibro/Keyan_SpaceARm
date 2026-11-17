#!/usr/bin/env python3
import heapq
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import rospy
import tf.transformations as tft
from geometry_msgs.msg import Pose, PoseStamped
from nav_msgs.msg import Path
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, String
from visualization_msgs.msg import Marker, MarkerArray

from uav_continuum_opportunity_planner.common import (
    VoxelOccupancy,
    color,
    load_arm_params,
    load_body_keepout_boxes,
    load_boxes,
    marker_header,
    point_distance,
    point_msg,
)


class GlobalPathNode:
    def __init__(self):
        self.frame_id = rospy.get_param("/frame_id", "world")
        self.arm_params = load_arm_params()
        self.boxes = load_boxes()
        self.body_keepout_boxes = load_body_keepout_boxes()
        self.scenario_name = rospy.get_param("/scenario/name", "unknown")

        vehicle = rospy.get_param("/vehicle", {})
        global_cfg = rospy.get_param("/global_path", {})
        self.body_radius = float(global_cfg.get("body_radius", vehicle.get("body_radius", 0.34)))
        self.body_height = float(global_cfg.get("body_height", vehicle.get("body_height", self.body_radius * 2.0)))
        self.clearance = float(global_cfg.get("clearance", vehicle.get("body_clearance", 0.04)))
        self.arm_clearance = float(vehicle.get("arm_clearance", 0.025))
        self.include_arm_transit_envelope = bool(
            global_cfg.get("include_arm_transit_envelope", rospy.get_param("~include_arm_transit_envelope", False))
        )
        self.transit_lower_extent = self._transit_lower_extent()
        self.transit_upper_extent = 0.5 * self.body_height + self.clearance
        self.resolution = float(global_cfg.get("resolution", rospy.get_param("~resolution", 0.30)))
        self.xy_resolution = float(global_cfg.get("xy_resolution", rospy.get_param("~xy_resolution", self.resolution)))
        self.z_resolution = float(global_cfg.get("z_resolution", rospy.get_param("~z_resolution", self.resolution)))
        self.z_min = float(global_cfg.get("z_min", rospy.get_param("~z_min", 0.90)))
        self.z_max = float(global_cfg.get("z_max", rospy.get_param("~z_max", 2.05)))
        self.xy_padding = float(global_cfg.get("xy_padding", rospy.get_param("~xy_padding", 2.5)))
        self.max_expansions = int(global_cfg.get("max_expansions", rospy.get_param("~max_expansions", 80000)))
        self.allow_preferred_goal_fallback = bool(rospy.get_param("~allow_preferred_goal_fallback", False))
        self.preferred_goal = [float(v) for v in global_cfg.get("preferred_goal", [0.0, 0.0, 1.0, 0.0])]
        self.path_step = float(rospy.get_param("~path_step", 0.20))
        self.replan_period = float(rospy.get_param("~replan_period", 2.0))
        # A periodic topology search driven directly by a noisy pose stream can
        # continuously replace an otherwise valid path while the vehicle is
        # stationary.  Keep the legacy behaviour when this is zero; V2 enables
        # the movement gate explicitly from its launch file.
        self.replan_movement_threshold = float(rospy.get_param("~replan_movement_threshold", 0.0))
        self.max_candidate_attempts = int(rospy.get_param("~max_candidate_attempts", global_cfg.get("max_candidate_attempts", 24)))
        self.max_successful_paths = int(rospy.get_param("~max_successful_paths", global_cfg.get("max_successful_paths", 3)))
        self.heuristic_weight = float(global_cfg.get("heuristic_weight", rospy.get_param("~heuristic_weight", 1.0)))
        self.clearance_cost_weight = float(
            global_cfg.get("clearance_cost_weight", rospy.get_param("~clearance_cost_weight", 3.0))
        )
        self.clearance_cost_target = float(
            global_cfg.get("clearance_cost_target", rospy.get_param("~clearance_cost_target", 0.08))
        )
        self.use_passage_landmarks = bool(
            global_cfg.get("use_passage_landmarks", rospy.get_param("~use_passage_landmarks", True))
        )
        self.candidate_cluster_resolution = float(
            rospy.get_param("~candidate_cluster_resolution", global_cfg.get("candidate_cluster_resolution", 0.16))
        )
        self.candidate_preselect_limit = int(
            rospy.get_param("~candidate_preselect_limit", global_cfg.get("candidate_preselect_limit", 60))
        )

        self.use_static_cloud = bool(rospy.get_param("~use_static_cloud", True))
        self.require_static_cloud = bool(rospy.get_param("~require_static_cloud", False))
        self.static_cloud_topic = rospy.get_param("~static_cloud_topic", "/world/static_scene_cloud")
        self.static_cloud_search_radius = float(rospy.get_param("~static_cloud_search_radius", 0.45))
        self.static_cloud_query_padding = float(rospy.get_param("~static_cloud_query_padding", 0.05))
        self.static_cloud_stride = max(1, int(rospy.get_param("~static_cloud_stride", 1)))
        static_resolution = float(rospy.get_param("~static_cloud_resolution", 0.10))
        static_inflation = float(rospy.get_param("~static_cloud_inflation", 0.05))
        static_max_points = int(rospy.get_param("~static_cloud_max_points", 250000))
        self.static_occupancy = VoxelOccupancy(static_resolution, static_inflation, static_max_points)
        self.static_cloud_loaded = False

        self.current_pose = None
        self.reach_candidates = []
        self.last_plan_time = rospy.Time(0)
        self.last_plan_position = None
        self.last_path_msg = None
        self.path_republish_period = float(rospy.get_param("~path_republish_period", 0.5))

        uav_pose_topic = rospy.get_param("~uav_pose_topic", "/iris_0/world_pose")
        rospy.Subscriber(uav_pose_topic, PoseStamped, self._pose_cb, queue_size=1)
        rospy.Subscriber("/planner/reach_candidates", Float32MultiArray, self._reach_candidates_cb, queue_size=10)
        if self.use_static_cloud:
            rospy.Subscriber(self.static_cloud_topic, PointCloud2, self._static_cloud_cb, queue_size=1, buff_size=16 * 1024 * 1024)

        self.path_pub = rospy.Publisher("/planner/global_path", Path, queue_size=1, latch=True)
        self.path_array_pub = rospy.Publisher("/planner/global_path_array", Float32MultiArray, queue_size=1, latch=True)
        self.goal_pub = rospy.Publisher("/planner/selected_reach_goal", PoseStamped, queue_size=1, latch=True)
        self.selected_arm_pub = rospy.Publisher(
            "/planner/selected_reach_theta_phi", Float32MultiArray, queue_size=1, latch=True
        )
        self.report_pub = rospy.Publisher("/planner/global_path_report", String, queue_size=1, latch=True)
        self.marker_pub = rospy.Publisher("/planner/global_path_markers", MarkerArray, queue_size=1, latch=True)
        rospy.Timer(rospy.Duration(self.path_republish_period), self._republish_path_cb)

        rospy.loginfo("global_path_node: scenario=%s waiting for pose, reach candidates, static map", self.scenario_name)

    def _publish_path(self, poses):
        msg = self._path_msg(poses)
        self.last_path_msg = msg
        self.path_pub.publish(msg)
        array_msg = self._path_array_msg(msg)
        self.path_array_pub.publish(array_msg)
        try:
            rospy.set_param(
                "/planner/latest_global_path_array_cache",
                {
                    "stamp": rospy.Time.now().to_sec(),
                    "frame_id": msg.header.frame_id,
                    "count": len(msg.poses),
                    "data": list(array_msg.data),
                },
            )
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "global_path_node: failed to cache global path param: %s", exc)

    def _republish_path_cb(self, _event):
        if self.last_path_msg is None:
            return
        self.last_path_msg.header.stamp = rospy.Time.now()
        for pose_stamped in self.last_path_msg.poses:
            pose_stamped.header.stamp = self.last_path_msg.header.stamp
        self.path_pub.publish(self.last_path_msg)
        self.path_array_pub.publish(self._path_array_msg(self.last_path_msg))

    def _pose_cb(self, msg):
        self.current_pose = msg.pose

    def _static_cloud_cb(self, msg):
        if self.static_cloud_loaded:
            return
        points = []
        for idx, point in enumerate(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)):
            if idx % self.static_cloud_stride == 0:
                points.append((point[0], point[1], point[2]))
        self.static_occupancy.update(points, msg.header.frame_id, msg.header.stamp)
        self.static_cloud_loaded = True
        rospy.loginfo(
            "global_path_node: static cloud received points=%d occupied_voxels=%d frame=%s",
            len(points),
            len(self.static_occupancy.occupied),
            msg.header.frame_id,
        )

    def _reach_candidates_cb(self, msg):
        candidates = []
        stride = 9
        if msg.layout.dim and len(msg.layout.dim) >= 2:
            stride = int(msg.layout.dim[1].size)
        if stride < 9:
            return
        for offset in range(0, len(msg.data) - stride + 1, stride):
            values = [float(v) for v in msg.data[offset : offset + stride]]
            pose = self._make_pose(values[0], values[1], values[2], values[3])
            candidates.append(
                {
                    "pose": pose,
                    "theta_phi": [values[4], values[5]],
                    "tip_error": values[6],
                    "body_dist": values[7],
                    "score": values[8],
                }
            )
        candidates.sort(key=lambda c: c["score"])
        self.reach_candidates = candidates

    def _make_pose(self, x, y, z, yaw):
        pose = Pose()
        pose.position = point_msg(x, y, z)
        q = tft.quaternion_from_euler(0.0, 0.0, yaw)
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]
        return pose

    def _pose_xyzyaw(self, pose):
        q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        yaw = tft.euler_from_quaternion(q)[2]
        return (pose.position.x, pose.position.y, pose.position.z, yaw)

    def _transit_lower_extent(self):
        body_drop = 0.5 * self.body_height + self.clearance
        if not self.include_arm_transit_envelope:
            return body_drop
        mount_z = abs(float(self.arm_params.get("mount_xyz", [0.0, 0.0, 0.0])[2]))
        arm_drop = (
            mount_z
            + float(self.arm_params.get("base_offset_z", 0.0))
            + float(self.arm_params.get("arc_length", 0.0))
            + float(self.arm_params.get("gripper_length", 0.0))
            + float(self.arm_params.get("radius", 0.0))
            + self.arm_clearance
        )
        return max(body_drop, arm_drop)

    def _auto_passage_landmarks(self, start, goal):
        if not self.use_passage_landmarks:
            return []

        y0, y1 = start[1], goal[1]
        y_min = min(y0, y1)
        y_max = max(y0, y1)
        y_groups = {}
        for box in self.boxes:
            if box.center[1] < y_min or box.center[1] > y_max:
                continue
            if box.size[0] < 6.0 or box.size[1] > 0.8:
                continue
            key = round(box.center[1] / 0.20) * 0.20
            y_groups.setdefault(key, []).append(box)

        landmarks = []
        for _, boxes in sorted(y_groups.items(), key=lambda item: item[0], reverse=(y0 > y1)):
            best = None
            for lower in boxes:
                lower_top = lower.center[2] + lower.half[2]
                if lower_top > self.z_max:
                    continue
                for upper in boxes:
                    upper_bottom = upper.center[2] - upper.half[2]
                    if upper_bottom < self.z_min or upper_bottom <= lower_top:
                        continue
                    free_low = max(self.z_min, lower_top + self.transit_lower_extent)
                    free_high = min(self.z_max, upper_bottom - self.transit_upper_extent)
                    if free_high < free_low:
                        continue
                    gap = free_high - free_low
                    if best is None or gap > best[0]:
                        best = (gap, lower.center[1], 0.5 * (free_low + free_high))
            if best is None:
                continue
            y = best[1]
            z = best[2]
            x = start[0]
            yaw = math.atan2(goal[1] - start[1], goal[0] - start[0])
            landmarks.append((x, y, z, yaw))
        return landmarks

    def _path_via_landmarks(self, start, goal):
        landmarks = self._auto_passage_landmarks(start, goal)
        if not landmarks:
            return None, "NO_PASSAGE_LANDMARKS", 0, 0
        current = start
        path = []
        total_expansions = 0
        for segment_idx, waypoint in enumerate(landmarks + [goal]):
            segment, status, expansions = self._astar(current, waypoint)
            total_expansions += expansions
            if segment is None:
                return None, "LANDMARK_SEG%d_%s" % (segment_idx, status), total_expansions, len(landmarks)
            if path:
                segment = segment[1:]
            path.extend(segment)
            current = waypoint
        return path, "GLOBAL_PATH_READY", total_expansions, len(landmarks)

    def _interp_yaw(self, a, b, ratio):
        delta = math.atan2(math.sin(b - a), math.cos(b - a))
        return a + ratio * delta

    def _key(self, xyz):
        return (
            int(round(xyz[0] / self.xy_resolution)),
            int(round(xyz[1] / self.xy_resolution)),
            int(round(xyz[2] / self.z_resolution)),
        )

    def _point(self, key):
        return (key[0] * self.xy_resolution, key[1] * self.xy_resolution, key[2] * self.z_resolution)

    def _heuristic(self, key, goal_key):
        return math.sqrt(
            ((key[0] - goal_key[0]) * self.xy_resolution) ** 2
            + ((key[1] - goal_key[1]) * self.xy_resolution) ** 2
            + ((key[2] - goal_key[2]) * self.z_resolution) ** 2
        )

    def _box_margin(self, point, clearance):
        best = float("inf")
        name = "none"
        xy_clearance = self.body_radius + self.clearance
        for box in self.boxes:
            margin = self._expanded_box_margin(point, box, xy_clearance, self.transit_lower_extent, self.transit_upper_extent)
            if margin < best:
                best = margin
                name = box.name
        for box in self.body_keepout_boxes:
            margin = box.signed_distance(point)
            if margin < best:
                best = margin
                name = box.name
        return best, name

    def _expanded_box_margin(self, point, box, xy_clearance, z_lower_clearance, z_upper_clearance):
        dx = abs(point[0] - box.center[0]) - (box.half[0] + xy_clearance)
        dy = abs(point[1] - box.center[1]) - (box.half[1] + xy_clearance)
        if point[2] >= box.center[2]:
            dz = point[2] - box.center[2] - (box.half[2] + z_lower_clearance)
        else:
            dz = box.center[2] - point[2] - (box.half[2] + z_upper_clearance)
        ox = max(dx, 0.0)
        oy = max(dy, 0.0)
        oz = max(dz, 0.0)
        outside = math.sqrt(ox * ox + oy * oy + oz * oz)
        inside = min(max(dx, max(dy, dz)), 0.0)
        return outside + inside

    def _cloud_margin(self, point, clearance):
        if not self.use_static_cloud:
            return float("inf"), "static_cloud:disabled"
        if self.require_static_cloud and not self.static_occupancy.ready():
            return -float("inf"), "static_cloud:not_ready"
        if not self.static_occupancy.ready():
            return float("inf"), "static_cloud:not_ready"
        query_radius = min(
            max(clearance + self.static_cloud_query_padding, self.static_occupancy.resolution),
            self.static_cloud_search_radius,
        )
        dist, _, _ = self.static_occupancy.signed_distance(point, query_radius)
        return dist - clearance, "static_scene_cloud"

    def _body_margin(self, point):
        cloud_clearance = self.body_radius + self.clearance
        box_margin, box_name = self._box_margin(point, cloud_clearance)
        cloud_margin, cloud_name = self._cloud_margin(point, cloud_clearance)
        if cloud_margin < box_margin:
            return cloud_margin, cloud_name
        return box_margin, box_name

    def _safe_point(self, point):
        margin, _ = self._body_margin(point)
        return margin >= 0.0

    def _clearance_cost_factor(self, point):
        if self.clearance_cost_weight <= 0.0 or self.clearance_cost_target <= 1e-6:
            return 1.0
        margin, _ = self._body_margin(point)
        deficit = max(0.0, self.clearance_cost_target - margin) / self.clearance_cost_target
        return 1.0 + self.clearance_cost_weight * deficit * deficit

    def _safe_segment(self, a, b):
        dist = point_distance(a, b)
        step = max(min(self.xy_resolution, self.z_resolution) * 0.5, 0.02)
        samples = max(2, int(math.ceil(dist / step)) + 1)
        for idx in range(samples):
            ratio = idx / float(samples - 1)
            point = (
                a[0] + (b[0] - a[0]) * ratio,
                a[1] + (b[1] - a[1]) * ratio,
                a[2] + (b[2] - a[2]) * ratio,
            )
            if not self._safe_point(point):
                return False
        return True

    def _nearest_safe_key(self, center_key, bounds, max_shell=3):
        if self._safe_point(self._point(center_key)):
            return center_key
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        best = None
        best_d2 = float("inf")
        for shell in range(1, max_shell + 1):
            for dx in range(-shell, shell + 1):
                for dy in range(-shell, shell + 1):
                    for dz in range(-shell, shell + 1):
                        if max(abs(dx), abs(dy), abs(dz)) != shell:
                            continue
                        key = (center_key[0] + dx, center_key[1] + dy, center_key[2] + dz)
                        if key[0] < x_min or key[0] > x_max or key[1] < y_min or key[1] > y_max:
                            continue
                        if key[2] < z_min or key[2] > z_max:
                            continue
                        if not self._safe_point(self._point(key)):
                            continue
                        d2 = dx * dx + dy * dy + dz * dz
                        if d2 < best_d2:
                            best = key
                            best_d2 = d2
            if best is not None:
                return best
        return None

    def _segment_poses(self, a, b):
        ax, ay, az, ayaw = a
        bx, by, bz, byaw = b
        dist = math.sqrt((bx - ax) ** 2 + (by - ay) ** 2 + (bz - az) ** 2)
        steps = max(2, int(math.ceil(dist / max(self.path_step, 1e-3))) + 1)
        poses = []
        for i in range(steps):
            r = i / float(steps - 1)
            poses.append(self._make_pose(ax + (bx - ax) * r, ay + (by - ay) * r, az + (bz - az) * r, self._interp_yaw(ayaw, byaw, r)))
        return poses

    def _route_from_grid_points(self, points):
        poses = []
        for i in range(len(points) - 1):
            seg = self._segment_poses(points[i], points[i + 1])
            if poses:
                seg = seg[1:]
            poses.extend(seg)
        return poses

    def _reconstruct_path(self, came_from, start_key, goal_key, start, goal):
        keys = [goal_key]
        while keys[-1] != start_key:
            keys.append(came_from[keys[-1]])
        keys.reverse()
        points = []
        syaw = start[3]
        gyaw = goal[3]
        for idx, key in enumerate(keys):
            x, y, z = self._point(key)
            r = idx / float(max(len(keys) - 1, 1))
            points.append((x, y, z, self._interp_yaw(syaw, gyaw, r)))
        points[0] = start
        points[-1] = goal
        return self._route_from_grid_points(points)

    def _multigoal_astar(self, start, candidate_items):
        if not candidate_items:
            return [], "NO_CANDIDATES", 0

        sx, sy, sz, _ = start
        start_key = self._key((sx, sy, sz))
        goals = []
        for idx, candidate in candidate_items:
            goal = self._pose_xyzyaw(candidate["pose"])
            goals.append({"idx": idx, "candidate": candidate, "goal": goal, "center_key": self._key(goal[:3])})

        z_min_key = int(math.floor(self.z_min / self.z_resolution))
        z_max_key = int(math.ceil(self.z_max / self.z_resolution))
        x_pad = int(math.ceil(self.xy_padding / self.xy_resolution))
        y_pad = int(math.ceil(self.xy_padding / self.xy_resolution))
        x_keys = [start_key[0]] + [item["center_key"][0] for item in goals]
        y_keys = [start_key[1]] + [item["center_key"][1] for item in goals]
        bounds = (
            min(x_keys) - x_pad,
            max(x_keys) + x_pad,
            min(y_keys) - y_pad,
            max(y_keys) + y_pad,
            z_min_key,
            z_max_key,
        )

        start_safe_key = self._nearest_safe_key(start_key, bounds, max_shell=6)
        if start_safe_key is None:
            return [], "START_NOT_SAFE", 0
        start_key = start_safe_key
        snapped_start = self._point(start_key)
        start = (snapped_start[0], snapped_start[1], snapped_start[2], start[3])

        goal_items_by_key = {}
        skipped_goals = 0
        for item in goals:
            safe_key = self._nearest_safe_key(item["center_key"], bounds, max_shell=6)
            if safe_key is None:
                skipped_goals += 1
                continue
            snapped_goal = self._point(safe_key)
            safe_item = dict(item)
            safe_item["goal"] = (snapped_goal[0], snapped_goal[1], snapped_goal[2], item["goal"][3])
            goal_items_by_key.setdefault(safe_key, []).append(safe_item)
        if not goal_items_by_key:
            return [], "GOAL_NOT_SAFE", 0

        goal_keys = set(goal_items_by_key.keys())

        def heuristic(key):
            best = float("inf")
            for gkey in goal_keys:
                h = self._heuristic(key, gkey)
                if h < best:
                    best = h
            return best

        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    cost = math.sqrt(
                        (dx * self.xy_resolution) ** 2
                        + (dy * self.xy_resolution) ** 2
                        + (dz * self.z_resolution) ** 2
                    )
                    neighbors.append((dx, dy, dz, cost))

        open_heap = [(self.heuristic_weight * heuristic(start_key), 0.0, start_key)]
        came_from = {}
        g_score = {start_key: 0.0}
        closed = set()
        safe_cache = {}
        segment_cache = {}
        clearance_factor_cache = {}
        reached_goal_keys = set()
        solutions = []
        expansions = 0
        x_min, x_max, y_min, y_max, z_min, z_max = bounds

        while open_heap and expansions < self.max_expansions:
            _, current_g, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            closed.add(current)
            expansions += 1

            if current in goal_keys and current not in reached_goal_keys:
                reached_goal_keys.add(current)
                for item in goal_items_by_key[current]:
                    path = self._reconstruct_path(came_from, start_key, current, start, item["goal"])
                    solutions.append(
                        {
                            "idx": item["idx"],
                            "candidate": item["candidate"],
                            "goal": item["goal"],
                            "path": path,
                            "expansions": expansions,
                        }
                    )
                if len(solutions) >= max(1, self.max_successful_paths):
                    return solutions, "GLOBAL_PATH_READY", expansions
                if len(reached_goal_keys) >= len(goal_keys):
                    return solutions, "GLOBAL_PATH_READY", expansions

            for dx, dy, dz, step_cost in neighbors:
                nb = (current[0] + dx, current[1] + dy, current[2] + dz)
                if nb in closed:
                    continue
                if nb[0] < x_min or nb[0] > x_max or nb[1] < y_min or nb[1] > y_max or nb[2] < z_min or nb[2] > z_max:
                    continue
                if nb not in safe_cache:
                    safe_cache[nb] = self._safe_point(self._point(nb))
                if not safe_cache[nb]:
                    continue
                segment_key = (current, nb)
                if segment_key not in segment_cache:
                    segment_cache[segment_key] = self._safe_segment(self._point(current), self._point(nb))
                if not segment_cache[segment_key]:
                    continue
                if nb not in clearance_factor_cache:
                    clearance_factor_cache[nb] = self._clearance_cost_factor(self._point(nb))
                tentative = current_g + step_cost * clearance_factor_cache[nb]
                if tentative >= g_score.get(nb, float("inf")):
                    continue
                came_from[nb] = current
                g_score[nb] = tentative
                heapq.heappush(open_heap, (tentative + self.heuristic_weight * heuristic(nb), tentative, nb))

        if skipped_goals and not solutions:
            return [], "GOAL_NOT_SAFE", expansions
        return solutions, "GLOBAL_PATH_BLOCKED", expansions

    def _astar(self, start, goal):
        sx, sy, sz, syaw = start
        gx, gy, gz, gyaw = goal
        start_key = self._key((sx, sy, sz))
        goal_key = self._key((gx, gy, gz))
        z_min_key = int(math.floor(self.z_min / self.z_resolution))
        z_max_key = int(math.ceil(self.z_max / self.z_resolution))
        x_pad = int(math.ceil(self.xy_padding / self.xy_resolution))
        y_pad = int(math.ceil(self.xy_padding / self.xy_resolution))
        bounds = (
            min(start_key[0], goal_key[0]) - x_pad,
            max(start_key[0], goal_key[0]) + x_pad,
            min(start_key[1], goal_key[1]) - y_pad,
            max(start_key[1], goal_key[1]) + y_pad,
            z_min_key,
            z_max_key,
        )
        start_key = self._nearest_safe_key(start_key, bounds, max_shell=8)
        goal_key = self._nearest_safe_key(goal_key, bounds, max_shell=8)
        if start_key is None:
            return None, "START_NOT_SAFE", 0
        if goal_key is None:
            return None, "GOAL_NOT_SAFE", 0

        snapped_start = self._point(start_key)
        snapped_goal = self._point(goal_key)
        start = (snapped_start[0], snapped_start[1], snapped_start[2], syaw)
        goal = (snapped_goal[0], snapped_goal[1], snapped_goal[2], gyaw)

        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    cost = math.sqrt(
                        (dx * self.xy_resolution) ** 2
                        + (dy * self.xy_resolution) ** 2
                        + (dz * self.z_resolution) ** 2
                    )
                    neighbors.append((dx, dy, dz, cost))

        open_heap = [(self.heuristic_weight * self._heuristic(start_key, goal_key), 0.0, start_key)]
        came_from = {}
        g_score = {start_key: 0.0}
        closed = set()
        safe_cache = {}
        segment_cache = {}
        clearance_factor_cache = {}
        expansions = 0
        while open_heap and expansions < self.max_expansions:
            _, current_g, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == goal_key:
                keys = [current]
                while keys[-1] != start_key:
                    keys.append(came_from[keys[-1]])
                keys.reverse()
                points = []
                for idx, key in enumerate(keys):
                    x, y, z = self._point(key)
                    r = idx / float(max(len(keys) - 1, 1))
                    points.append((x, y, z, self._interp_yaw(syaw, gyaw, r)))
                points[0] = start
                points[-1] = goal
                return self._route_from_grid_points(points), "GLOBAL_PATH_READY", expansions

            closed.add(current)
            expansions += 1
            for dx, dy, dz, step_cost in neighbors:
                nb = (current[0] + dx, current[1] + dy, current[2] + dz)
                if nb in closed:
                    continue
                x_min, x_max, y_min, y_max, z_min, z_max = bounds
                if nb[0] < x_min or nb[0] > x_max or nb[1] < y_min or nb[1] > y_max or nb[2] < z_min or nb[2] > z_max:
                    continue
                if nb not in safe_cache:
                    safe_cache[nb] = self._safe_point(self._point(nb))
                if not safe_cache[nb]:
                    continue
                segment_key = (current, nb)
                if segment_key not in segment_cache:
                    segment_cache[segment_key] = self._safe_segment(self._point(current), self._point(nb))
                if not segment_cache[segment_key]:
                    continue
                if nb not in clearance_factor_cache:
                    clearance_factor_cache[nb] = self._clearance_cost_factor(self._point(nb))
                tentative = current_g + step_cost * clearance_factor_cache[nb]
                if tentative >= g_score.get(nb, float("inf")):
                    continue
                came_from[nb] = current
                g_score[nb] = tentative
                heapq.heappush(open_heap, (tentative + self.heuristic_weight * self._heuristic(nb, goal_key), tentative, nb))
        return None, "GLOBAL_PATH_BLOCKED", expansions

    def _path_msg(self, poses):
        msg = Path()
        msg.header.frame_id = self.frame_id
        msg.header.stamp = rospy.Time.now()
        for pose in poses:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose = pose
            msg.poses.append(ps)
        return msg

    def _path_array_msg(self, path_msg):
        msg = Float32MultiArray()
        count = len(path_msg.poses)
        msg.layout.dim = [
            MultiArrayDimension(label="waypoint", size=count, stride=count * 4),
            MultiArrayDimension(label="xyzyaw", size=4, stride=4),
        ]
        msg.layout.data_offset = 0
        for ps in path_msg.poses:
            q = [ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w]
            yaw = tft.euler_from_quaternion(q)[2]
            msg.data.extend([ps.pose.position.x, ps.pose.position.y, ps.pose.position.z, yaw])
        return msg

    def _path_length(self, poses):
        total = 0.0
        for a, b in zip(poses, poses[1:]):
            total += point_distance((a.position.x, a.position.y, a.position.z), (b.position.x, b.position.y, b.position.z))
        return total

    def _path_min_margin(self, poses):
        best = float("inf")
        name = "none"
        for pose in poses:
            margin, obstacle = self._body_margin((pose.position.x, pose.position.y, pose.position.z))
            if margin < best:
                best = margin
                name = obstacle
        return best, name

    def _candidate_cluster_key(self, candidate):
        pose = candidate["pose"]
        res = max(self.candidate_cluster_resolution, 1e-3)
        return (
            int(round(pose.position.x / res)),
            int(round(pose.position.y / res)),
            int(round(pose.position.z / res)),
        )

    def _candidate_rank_score(self, candidate, start):
        pose = candidate["pose"]
        xyz = (pose.position.x, pose.position.y, pose.position.z)
        start_xyz = (start[0], start[1], start[2])
        direct_dist = point_distance(start_xyz, xyz)
        margin, _ = self._body_margin(xyz)
        theta = abs(float(candidate.get("theta_phi", [0.0, 0.0])[0]))
        tip_error = max(float(candidate.get("tip_error", 0.0)), 0.0)
        reach_score = float(candidate.get("score", 0.0))
        margin_bonus = max(min(margin, 0.6), -0.6)
        return (
            4.0 * tip_error
            + 0.04 * direct_dist
            + 0.0015 * theta
            + 0.35 * reach_score
            - 0.30 * margin_bonus
        )

    def _preselect_candidates(self, candidates, start):
        if not candidates:
            return []

        representatives = {}
        for candidate in candidates:
            ranked = dict(candidate)
            ranked["selector_score"] = self._candidate_rank_score(candidate, start)
            key = self._candidate_cluster_key(candidate)
            previous = representatives.get(key)
            if previous is None or ranked["selector_score"] < previous["selector_score"]:
                representatives[key] = ranked

        selected = sorted(representatives.values(), key=lambda c: c["selector_score"])
        limit = max(1, self.candidate_preselect_limit)
        return selected[:limit]

    def _path_selection_cost(self, candidate, poses):
        length = self._path_length(poses)
        min_margin, _ = self._path_min_margin(poses)
        tip_error = max(float(candidate.get("tip_error", 0.0)), 0.0)
        theta = abs(float(candidate.get("theta_phi", [0.0, 0.0])[0]))
        margin_penalty = max(0.0, self.clearance_cost_target - min_margin)
        return length + 6.0 * tip_error + 0.002 * theta + self.clearance_cost_weight * 2.0 * margin_penalty

    def _select_landmark_solution(self, start, candidates, attempts):
        best_solution = None
        last_status = "NO_PASSAGE_LANDMARKS"
        total_expansions = 0
        landmark_count = 0
        for idx, candidate in enumerate(candidates[:attempts]):
            goal = self._pose_xyzyaw(candidate["pose"])
            path, status, expansions, count = self._path_via_landmarks(start, goal)
            total_expansions += expansions
            landmark_count = max(landmark_count, count)
            last_status = status
            if path is None:
                rospy.loginfo(
                    "global_path_node: landmark candidate %d failed status=%s expansions=%d landmarks=%d goal=[%.3f, %.3f, %.3f]",
                    idx,
                    status,
                    expansions,
                    count,
                    goal[0],
                    goal[1],
                    goal[2],
                )
                continue
            path_cost = self._path_selection_cost(candidate, path)
            path_margin, path_obstacle = self._path_min_margin(path)
            rospy.loginfo(
                "global_path_node: landmark candidate %d reached waypoints=%d expansions=%d landmarks=%d cost=%.3f min_margin=%.3f obstacle=%s goal=[%.3f, %.3f, %.3f]",
                idx,
                len(path),
                expansions,
                count,
                path_cost,
                path_margin,
                path_obstacle,
                goal[0],
                goal[1],
                goal[2],
            )
            if path_margin < 0.0:
                continue
            if best_solution is None or path_cost < best_solution["cost"]:
                best_solution = {
                    "idx": idx,
                    "candidate": candidate,
                    "goal": goal,
                    "path": path,
                    "cost": path_cost,
                    "expansions": expansions,
                    "min_margin": path_margin,
                    "min_obstacle": path_obstacle,
                    "landmark_count": count,
                }
        return best_solution, last_status, total_expansions, landmark_count

    def _markers(self, poses, selected_idx):
        stamp = rospy.Time.now()
        arr = MarkerArray()
        line = Marker()
        marker_header(line, self.frame_id, stamp, "global_path_line", 0, Marker.LINE_STRIP)
        line.pose.orientation.w = 1.0
        line.scale.x = 0.045
        line.color = color(0.1, 0.7, 1.0, 0.95)
        line.points = [point_msg(p.position.x, p.position.y, p.position.z) for p in poses]
        arr.markers.append(line)
        for idx, cand in enumerate(self.reach_candidates[:50]):
            marker = Marker()
            marker_header(marker, self.frame_id, stamp, "global_candidate_goals", idx + 10, Marker.SPHERE)
            marker.pose = cand["pose"]
            marker.scale.x = 0.07
            marker.scale.y = 0.07
            marker.scale.z = 0.07
            marker.color = color(0.0, 1.0, 0.2, 0.9) if idx == selected_idx else color(0.2, 0.4, 1.0, 0.35)
            arr.markers.append(marker)
        return arr

    def _publish_report(self, status, **kwargs):
        lines = ["status=%s" % status, "scenario=%s" % self.scenario_name]
        for key in sorted(kwargs):
            lines.append("%s=%s" % (key, kwargs[key]))
        self.report_pub.publish(String(data="\n".join(lines)))

    def _goal_pose_stamped(self, pose):
        msg = PoseStamped()
        msg.header.frame_id = self.frame_id
        msg.header.stamp = rospy.Time.now()
        msg.pose = pose
        return msg

    def _plan_once(self):
        if self.current_pose is None:
            self._publish_report("WAITING_CURRENT_POSE")
            return
        if self.use_static_cloud and self.require_static_cloud and not self.static_occupancy.ready():
            self._publish_report("WAITING_STATIC_CLOUD")
            return

        start = self._pose_xyzyaw(self.current_pose)
        start = (start[0], start[1], max(start[2], self.z_min), start[3])
        candidates = list(self.reach_candidates)
        if not candidates and self.allow_preferred_goal_fallback:
            candidates = [{"pose": self._make_pose(*self.preferred_goal), "score": 999.0, "tip_error": -1.0, "body_dist": -1.0, "theta_phi": [0.0, 0.0]}]
        if not candidates:
            self._publish_report("WAITING_REACH_CANDIDATES", candidate_count=0)
            return

        raw_candidate_count = len(candidates)
        candidates = self._preselect_candidates(candidates, start)
        attempts = min(len(candidates), max(1, self.max_candidate_attempts))
        rospy.loginfo(
            "global_path_node: planning start raw_candidates=%d representatives=%d attempts=%d start=[%.3f, %.3f, %.3f]",
            raw_candidate_count,
            len(candidates),
            attempts,
            start[0],
            start[1],
            start[2],
        )
        attempt_items = list(enumerate(candidates[:attempts]))
        solutions, search_status, expansions = self._multigoal_astar(start, attempt_items)
        best_solution = None
        for solution in solutions:
            idx = solution["idx"]
            candidate = solution["candidate"]
            path = solution["path"]
            goal = solution["goal"]
            path_cost = self._path_selection_cost(candidate, path)
            path_margin, path_obstacle = self._path_min_margin(path)
            rospy.loginfo(
                "global_path_node: candidate %d reached waypoints=%d search_expansions=%d cost=%.3f min_margin=%.3f obstacle=%s goal=[%.3f, %.3f, %.3f]",
                idx,
                len(path),
                solution["expansions"],
                path_cost,
                path_margin,
                path_obstacle,
                goal[0],
                goal[1],
                goal[2],
            )
            if path_margin < 0.0:
                continue
            if best_solution is None or path_cost < best_solution["cost"]:
                best_solution = {
                    "idx": idx,
                    "candidate": candidate,
                    "goal": goal,
                    "path": path,
                    "cost": path_cost,
                    "expansions": expansions,
                    "min_margin": path_margin,
                    "min_obstacle": path_obstacle,
                }

        if best_solution is not None:
            candidate = best_solution["candidate"]
            path = best_solution["path"]
            goal = best_solution["goal"]
            rospy.loginfo(
                "global_path_node: selected candidate=%d cost=%.3f waypoints=%d goal=[%.3f, %.3f, %.3f]",
                best_solution["idx"],
                best_solution["cost"],
                len(path),
                goal[0],
                goal[1],
                goal[2],
            )
            self.goal_pub.publish(self._goal_pose_stamped(candidate["pose"]))
            self.selected_arm_pub.publish(Float32MultiArray(data=list(candidate.get("theta_phi", [0.0, 0.0]))))
            self._publish_path(path)
            self.marker_pub.publish(self._markers(path, best_solution["idx"]))
            self._publish_report(
                "GLOBAL_PATH_READY",
                raw_candidate_count=raw_candidate_count,
                representative_count=len(candidates),
                attempted_candidates=attempts,
                selected_candidate=best_solution["idx"],
                selected_goal="[%.3f, %.3f, %.3f]" % (goal[0], goal[1], goal[2]),
                selected_cost="%.3f" % best_solution["cost"],
                waypoints=len(path),
                length="%.3f" % self._path_length(path),
                min_margin="%.3f" % best_solution["min_margin"],
                min_obstacle=best_solution["min_obstacle"],
                heuristic_weight="%.3f" % self.heuristic_weight,
                clearance_cost_weight="%.3f" % self.clearance_cost_weight,
                clearance_cost_target="%.3f" % self.clearance_cost_target,
                max_expansions=self.max_expansions,
                transit_lower_extent="%.3f" % self.transit_lower_extent,
                transit_upper_extent="%.3f" % self.transit_upper_extent,
                resolution="%.3f" % self.resolution,
                xy_resolution="%.3f" % self.xy_resolution,
                z_resolution="%.3f" % self.z_resolution,
                expansions=best_solution["expansions"],
                start="[%.3f, %.3f, %.3f]" % (start[0], start[1], start[2]),
                goal="[%.3f, %.3f, %.3f]" % (goal[0], goal[1], goal[2]),
            )
            return

        landmark_solution, landmark_status, landmark_expansions, landmark_count = self._select_landmark_solution(start, candidates, attempts)
        if landmark_solution is not None:
            candidate = landmark_solution["candidate"]
            path = landmark_solution["path"]
            goal = landmark_solution["goal"]
            rospy.loginfo(
                "global_path_node: selected landmark candidate=%d cost=%.3f waypoints=%d landmarks=%d goal=[%.3f, %.3f, %.3f]",
                landmark_solution["idx"],
                landmark_solution["cost"],
                len(path),
                landmark_solution["landmark_count"],
                goal[0],
                goal[1],
                goal[2],
            )
            self.goal_pub.publish(self._goal_pose_stamped(candidate["pose"]))
            self.selected_arm_pub.publish(Float32MultiArray(data=list(candidate.get("theta_phi", [0.0, 0.0]))))
            self._publish_path(path)
            self.marker_pub.publish(self._markers(path, landmark_solution["idx"]))
            self._publish_report(
                "GLOBAL_PATH_READY",
                planner="auto_passage_landmarks",
                raw_candidate_count=raw_candidate_count,
                representative_count=len(candidates),
                attempted_candidates=attempts,
                selected_candidate=landmark_solution["idx"],
                selected_goal="[%.3f, %.3f, %.3f]" % (goal[0], goal[1], goal[2]),
                selected_cost="%.3f" % landmark_solution["cost"],
                waypoints=len(path),
                length="%.3f" % self._path_length(path),
                min_margin="%.3f" % landmark_solution["min_margin"],
                min_obstacle=landmark_solution["min_obstacle"],
                heuristic_weight="%.3f" % self.heuristic_weight,
                clearance_cost_weight="%.3f" % self.clearance_cost_weight,
                clearance_cost_target="%.3f" % self.clearance_cost_target,
                max_expansions=self.max_expansions,
                transit_lower_extent="%.3f" % self.transit_lower_extent,
                transit_upper_extent="%.3f" % self.transit_upper_extent,
                passage_landmarks=landmark_solution["landmark_count"],
                landmark_expansions=landmark_solution["expansions"],
                start="[%.3f, %.3f, %.3f]" % (start[0], start[1], start[2]),
                goal="[%.3f, %.3f, %.3f]" % (goal[0], goal[1], goal[2]),
            )
            return

        self._publish_report(
            search_status,
            raw_candidate_count=raw_candidate_count,
            representative_count=len(candidates),
            attempted_candidates=attempts,
            expansions=expansions,
            heuristic_weight="%.3f" % self.heuristic_weight,
            clearance_cost_weight="%.3f" % self.clearance_cost_weight,
            clearance_cost_target="%.3f" % self.clearance_cost_target,
            landmark_status=landmark_status,
            landmark_expansions=landmark_expansions,
            passage_landmarks=landmark_count,
            max_expansions=self.max_expansions,
            transit_lower_extent="%.3f" % self.transit_lower_extent,
            transit_upper_extent="%.3f" % self.transit_upper_extent,
            successful_paths=len(solutions),
            xy_resolution="%.3f" % self.xy_resolution,
            z_resolution="%.3f" % self.z_resolution,
        )

    def spin(self):
        rate = rospy.Rate(float(rospy.get_param("~rate_hz", 2.0)))
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            if (now - self.last_plan_time).to_sec() >= self.replan_period:
                if (self.last_path_msg is not None and self.last_plan_position is not None and
                        self.current_pose is not None and self.replan_movement_threshold > 0.0):
                    current = (
                        self.current_pose.position.x,
                        self.current_pose.position.y,
                        self.current_pose.position.z,
                    )
                    if point_distance(current, self.last_plan_position) < self.replan_movement_threshold:
                        rate.sleep()
                        continue
                self.last_plan_time = now
                if self.current_pose is not None:
                    self.last_plan_position = (
                        self.current_pose.position.x,
                        self.current_pose.position.y,
                        self.current_pose.position.z,
                    )
                self._plan_once()
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("global_path_node")
    try:
        GlobalPathNode().spin()
    except rospy.ROSInterruptException:
        pass
