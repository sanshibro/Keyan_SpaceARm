#!/usr/bin/env python3
import heapq
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import rospy
import tf.transformations as tft
from geometry_msgs.msg import Point, Pose, PoseStamped
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
    load_target,
    marker_header,
    point_distance,
    point_msg,
    world_arm_points,
)


class JointPlannerNode:
    def __init__(self):
        self.frame_id = rospy.get_param("/frame_id", "world")
        self.arm_params = load_arm_params()
        self.boxes = load_boxes()
        self.body_keepout_boxes = load_body_keepout_boxes()
        self.target = load_target()

        vehicle = rospy.get_param("/vehicle", {})
        self.body_radius = float(vehicle.get("body_radius", 0.34))
        self.body_clearance = float(vehicle.get("body_clearance", 0.12))
        self.arm_clearance = float(vehicle.get("arm_clearance", 0.04))
        self.world_box_count = len(rospy.get_param("/world_boxes/boxes", []))
        self.task_box_count = len(rospy.get_param("/boxes", []))
        self.use_static_cloud = bool(rospy.get_param("~use_static_cloud", False))
        self.require_static_cloud = bool(rospy.get_param("~require_static_cloud", False))
        self.static_cloud_topic = rospy.get_param("~static_cloud_topic", "/world/static_scene_cloud")
        self.static_cloud_search_radius = float(rospy.get_param("~static_cloud_search_radius", 0.35))
        self.static_cloud_stride = max(1, int(rospy.get_param("~static_cloud_stride", 1)))
        static_resolution = float(rospy.get_param("~static_cloud_resolution", 0.1))
        static_inflation = float(rospy.get_param("~static_cloud_inflation", 0.1))
        static_max_points = int(rospy.get_param("~static_cloud_max_points", 250000))
        self.static_occupancy = VoxelOccupancy(static_resolution, static_inflation, static_max_points)

        self.compact_theta = float(rospy.get_param("~compact_theta", self.arm_params["theta_max_deg"]))
        self.compact_phi = float(rospy.get_param("~compact_phi", 0.0))
        self.compact_arc_length = float(rospy.get_param("~compact_arc_length", 0.25))
        self.path_step = float(rospy.get_param("~path_step", 0.12))
        self.reach_steps = int(rospy.get_param("~reach_steps", 8))
        self.rate_hz = float(rospy.get_param("~rate_hz", 2.0))
        self.global_search_resolution = float(rospy.get_param("~global_search_resolution", 0.35))
        self.global_search_xy_padding = float(rospy.get_param("~global_search_xy_padding", 3.0))
        self.global_search_z_min = float(rospy.get_param("~global_search_z_min", 0.85))
        self.global_search_z_max = float(rospy.get_param("~global_search_z_max", 2.0))
        self.global_search_max_expansions = int(rospy.get_param("~global_search_max_expansions", 25000))
        self.min_planning_z = float(
            rospy.get_param(
                "~min_planning_z",
                max(self.target["pose"][2] + 0.75, self.body_radius + self.body_clearance + 0.05),
            )
        )

        self.current_pose = None
        self.current_theta_phi = [0.0, 0.0]
        self.standoff_goal = None
        self.reach_theta_phi = None
        self.reach_candidates = []
        self.last_selected_candidate = -1
        self.planned_arm_waypoints = None
        self.last_precondition = "NONE"
        self.last_raw_start_z = 0.0
        self.last_planning_start_z = 0.0
        self.logged_static_cloud = False
        self.static_cloud_loaded = False

        uav_pose_topic = rospy.get_param("~uav_pose_topic", "/iris_0/mavros/local_position/pose")
        rospy.Subscriber(uav_pose_topic, PoseStamped, self._current_pose_cb, queue_size=1)
        rospy.Subscriber("/arm/current_theta_phi", Float32MultiArray, self._current_arm_cb, queue_size=10)
        rospy.Subscriber("/planner/uav_goal", PoseStamped, self._goal_cb, queue_size=10)
        rospy.Subscriber("/planner/arm_theta_phi_cmd", Float32MultiArray, self._arm_goal_cb, queue_size=10)
        rospy.Subscriber("/planner/reach_candidates", Float32MultiArray, self._reach_candidates_cb, queue_size=10)
        if self.use_static_cloud:
            rospy.Subscriber(self.static_cloud_topic, PointCloud2, self._static_cloud_cb, queue_size=1, buff_size=16 * 1024 * 1024)

        self.status_pub = rospy.Publisher("/planner/joint_status", String, queue_size=10, latch=True)
        self.plan_pub = rospy.Publisher("/planner/task_plan", String, queue_size=10, latch=True)
        self.safety_pub = rospy.Publisher("/planner/safety_report", String, queue_size=10, latch=True)
        self.occupancy_pub = rospy.Publisher("/world/static_occupancy_report", String, queue_size=1, latch=True)
        self.path_pub = rospy.Publisher("/planner/uav_waypoints", Path, queue_size=10)
        self.arm_waypoints_pub = rospy.Publisher("/planner/arm_waypoints", Float32MultiArray, queue_size=10)
        self.marker_pub = rospy.Publisher("/planner/task_plan_markers", MarkerArray, queue_size=10)

        rospy.loginfo("joint_planner_node: waiting for current pose, standoff goal, and arm reach-in goal")

    def _compact_arm_params(self):
        params = dict(self.arm_params)
        return params

    def _static_cloud_cb(self, msg):
        if self.static_cloud_loaded:
            return
        points = []
        for idx, point in enumerate(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)):
            if idx % self.static_cloud_stride == 0:
                points.append((point[0], point[1], point[2]))
        self.static_occupancy.update(points, msg.header.frame_id, msg.header.stamp)
        self.static_cloud_loaded = True
        self.occupancy_pub.publish(String(data=self.static_occupancy.report()))
        if not self.logged_static_cloud:
            self.logged_static_cloud = True
            rospy.loginfo(
                "joint_planner_node: static cloud received points=%d occupied_voxels=%d frame=%s",
                len(points),
                len(self.static_occupancy.occupied),
                msg.header.frame_id,
            )

    def _current_pose_cb(self, msg):
        self.current_pose = msg.pose

    def _current_arm_cb(self, msg):
        if len(msg.data) >= 2:
            self.current_theta_phi = [float(msg.data[0]), float(msg.data[1])]

    def _goal_cb(self, msg):
        self.standoff_goal = msg.pose

    def _arm_goal_cb(self, msg):
        if len(msg.data) >= 2:
            self.reach_theta_phi = [float(msg.data[0]), float(msg.data[1])]

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
        self.reach_candidates = candidates
        if candidates:
            self.standoff_goal = candidates[0]["pose"]
            self.reach_theta_phi = list(candidates[0]["theta_phi"])

    def _pose_xyz_yaw(self, pose):
        q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        yaw = tft.euler_from_quaternion(q)[2]
        return (pose.position.x, pose.position.y, pose.position.z, yaw)

    def _make_pose(self, x, y, z, yaw):
        pose = Pose()
        pose.position = point_msg(x, y, z)
        q = tft.quaternion_from_euler(0.0, 0.0, yaw)
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]
        return pose

    def _signed_clearance_ok(self, point, clearance):
        for box in self.boxes:
            if box.signed_distance(point) < clearance:
                return False
        if self.use_static_cloud:
            if self.require_static_cloud and not self.static_occupancy.ready():
                return False
            dist, _, _ = self.static_occupancy.signed_distance(point, max(self.static_cloud_search_radius, clearance + 0.05))
            if dist < clearance:
                return False
        return True

    def _body_safe(self, pose):
        point = (pose.position.x, pose.position.y, pose.position.z)
        if not self._signed_clearance_ok(point, self.body_radius + self.body_clearance):
            return False
        for box in self.body_keepout_boxes:
            if box.signed_distance(point) < 0.0:
                return False
        return True

    def _arm_safe(self, pose, theta, phi):
        threshold = self.arm_params["radius"] + self.arm_clearance
        arm_params = self._compact_arm_params() if theta == self.compact_theta and phi == self.compact_phi else self.arm_params
        for point in world_arm_points(pose, theta, phi, arm_params):
            if not self._signed_clearance_ok(point, threshold):
                return False
        return True

    def _joint_state_safe(self, pose, theta, phi):
        return self._body_safe(pose) and self._arm_safe(pose, theta, phi)

    def _interp_yaw(self, start, end, ratio):
        delta = math.atan2(math.sin(end - start), math.cos(end - start))
        return start + ratio * delta

    def _segment(self, a, b):
        ax, ay, az, ayaw = a
        bx, by, bz, byaw = b
        dist = math.sqrt((bx - ax) ** 2 + (by - ay) ** 2 + (bz - az) ** 2)
        steps = max(2, int(math.ceil(dist / max(self.path_step, 1e-3))) + 1)
        return [
            self._make_pose(
                ax + (bx - ax) * i / float(steps - 1),
                ay + (by - ay) * i / float(steps - 1),
                az + (bz - az) * i / float(steps - 1),
                self._interp_yaw(ayaw, byaw, i / float(steps - 1)),
            )
            for i in range(steps)
        ]

    def _route_from_points(self, points):
        poses = []
        for i in range(len(points) - 1):
            seg = self._segment(points[i], points[i + 1])
            if poses:
                seg = seg[1:]
            poses.extend(seg)
        return poses

    def _route_safe(self, poses):
        for pose in poses:
            if not self._joint_state_safe(pose, self.compact_theta, self.compact_phi):
                return False
        return True

    def _grid_key(self, point):
        res = self.global_search_resolution
        return (
            int(round(point[0] / res)),
            int(round(point[1] / res)),
            int(round(point[2] / res)),
        )

    def _grid_point(self, key):
        res = self.global_search_resolution
        return (key[0] * res, key[1] * res, key[2] * res)

    def _heuristic_key(self, key, goal_key):
        res = self.global_search_resolution
        return res * math.sqrt(
            (key[0] - goal_key[0]) ** 2
            + (key[1] - goal_key[1]) ** 2
            + (key[2] - goal_key[2]) ** 2
        )

    def _pose_safe_at_xyz(self, xyz, yaw):
        pose = self._make_pose(xyz[0], xyz[1], xyz[2], yaw)
        return self._joint_state_safe(pose, self.compact_theta, self.compact_phi)

    def _body_safe_at_xyz(self, xyz):
        pose = self._make_pose(xyz[0], xyz[1], xyz[2], 0.0)
        return self._body_safe(pose)

    def _nearest_safe_key(self, center_key, is_safe_key, x_min, x_max, y_min, y_max, z_min_key, z_max_key, max_shell=2):
        if is_safe_key(center_key):
            return center_key
        best = None
        best_dist2 = float("inf")
        for shell in range(1, max_shell + 1):
            for dx in range(-shell, shell + 1):
                for dy in range(-shell, shell + 1):
                    for dz in range(-shell, shell + 1):
                        if max(abs(dx), abs(dy), abs(dz)) != shell:
                            continue
                        key = (center_key[0] + dx, center_key[1] + dy, center_key[2] + dz)
                        if key[0] < x_min or key[0] > x_max or key[1] < y_min or key[1] > y_max:
                            continue
                        if key[2] < z_min_key or key[2] > z_max_key:
                            continue
                        if not is_safe_key(key):
                            continue
                        dist2 = dx * dx + dy * dy + dz * dz
                        if dist2 < best_dist2:
                            best = key
                            best_dist2 = dist2
            if best is not None:
                return best
        return None

    def _astar_route(self, start, goal):
        sx, sy, sz, syaw = start
        gx, gy, gz, gyaw = goal
        start_key = self._grid_key((sx, sy, sz))
        goal_key = self._grid_key((gx, gy, gz))
        z_min_key = int(math.floor(self.global_search_z_min / self.global_search_resolution))
        z_max_key = int(math.ceil(self.global_search_z_max / self.global_search_resolution))
        x_min = min(start_key[0], goal_key[0]) - int(math.ceil(self.global_search_xy_padding / self.global_search_resolution))
        x_max = max(start_key[0], goal_key[0]) + int(math.ceil(self.global_search_xy_padding / self.global_search_resolution))
        y_min = min(start_key[1], goal_key[1]) - int(math.ceil(self.global_search_xy_padding / self.global_search_resolution))
        y_max = max(start_key[1], goal_key[1]) + int(math.ceil(self.global_search_xy_padding / self.global_search_resolution))

        rospy.loginfo_throttle(
            5.0,
            "joint_planner_node: A* transit search start=%s goal=%s res=%.2f z=[%.2f, %.2f]",
            start_key,
            goal_key,
            self.global_search_resolution,
            self.global_search_z_min,
            self.global_search_z_max,
        )

        if not self._pose_safe_at_xyz((sx, sy, sz), syaw):
            rospy.logwarn_throttle(5.0, "joint_planner_node: A* exact start pose is not safe")
            return None
        if not self._pose_safe_at_xyz((gx, gy, gz), gyaw):
            rospy.logwarn_throttle(5.0, "joint_planner_node: A* exact goal pose is not safe")
            return None

        safe_cache = {}

        def is_safe_key(key):
            cached = safe_cache.get(key)
            if cached is not None:
                return cached
            safe = self._body_safe_at_xyz(self._grid_point(key))
            safe_cache[key] = safe
            return safe

        snapped_start = self._nearest_safe_key(start_key, is_safe_key, x_min, x_max, y_min, y_max, z_min_key, z_max_key)
        snapped_goal = self._nearest_safe_key(goal_key, is_safe_key, x_min, x_max, y_min, y_max, z_min_key, z_max_key)
        if snapped_start is None:
            rospy.logwarn_throttle(5.0, "joint_planner_node: A* cannot snap start to a safe grid cell")
            return None
        if snapped_goal is None:
            rospy.logwarn_throttle(5.0, "joint_planner_node: A* cannot snap goal to a safe grid cell")
            return None
        if snapped_start != start_key or snapped_goal != goal_key:
            rospy.loginfo_throttle(
                5.0,
                "joint_planner_node: A* snapped start %s->%s goal %s->%s",
                start_key,
                snapped_start,
                goal_key,
                snapped_goal,
            )
        start_key = snapped_start
        goal_key = snapped_goal

        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    cost = self.global_search_resolution * math.sqrt(dx * dx + dy * dy + dz * dz)
                    neighbors.append((dx, dy, dz, cost))

        open_heap = []
        heapq.heappush(open_heap, (self._heuristic_key(start_key, goal_key), 0.0, start_key))
        came_from = {}
        g_score = {start_key: 0.0}
        closed = set()
        expansions = 0

        while open_heap and expansions < self.global_search_max_expansions:
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
                    x, y, z = self._grid_point(key)
                    ratio = idx / float(max(len(keys) - 1, 1))
                    yaw = self._interp_yaw(syaw, gyaw, ratio)
                    points.append((x, y, z, yaw))
                points[0] = start
                points[-1] = goal
                rospy.loginfo(
                    "joint_planner_node: A* success expansions=%d waypoints=%d",
                    expansions,
                    len(points),
                )
                return self._route_from_points(points)

            closed.add(current)
            expansions += 1
            for dx, dy, dz, step_cost in neighbors:
                nb = (current[0] + dx, current[1] + dy, current[2] + dz)
                if nb in closed:
                    continue
                if nb[0] < x_min or nb[0] > x_max or nb[1] < y_min or nb[1] > y_max:
                    continue
                if nb[2] < z_min_key or nb[2] > z_max_key:
                    continue
                if not is_safe_key(nb):
                    continue
                tentative = current_g + step_cost
                if tentative >= g_score.get(nb, float("inf")):
                    continue
                came_from[nb] = current
                g_score[nb] = tentative
                heapq.heappush(open_heap, (tentative + self._heuristic_key(nb, goal_key), tentative, nb))

        rospy.logwarn_throttle(
            5.0,
            "joint_planner_node: A* failed expansions=%d open=%d max=%d",
            expansions,
            len(open_heap),
            self.global_search_max_expansions,
        )
        return None

    def _planning_start(self, raw_start):
        sx, sy, sz, syaw = raw_start
        planning_z = max(sz, self.min_planning_z)
        self.last_raw_start_z = sz
        self.last_planning_start_z = planning_z
        if planning_z > sz + 0.05:
            self.last_precondition = "TAKEOFF_TO_PLANNING_ALTITUDE"
        else:
            self.last_precondition = "NONE"
        return (sx, sy, planning_z, syaw)

    def _point_margin(self, point, boxes, required_clearance):
        best_margin = float("inf")
        best_name = "none"
        for box in boxes:
            margin = box.signed_distance(point) - required_clearance
            if margin < best_margin:
                best_margin = margin
                best_name = box.name
        return best_margin, best_name

    def _update_margin(self, current, point, boxes, required_clearance):
        margin, name = self._point_margin(point, boxes, required_clearance)
        if margin < current["margin"]:
            current["margin"] = margin
            current["obstacle"] = name
            current["point"] = point

    def _update_cloud_margin(self, current, point, required_clearance):
        if not self.use_static_cloud:
            return
        if self.require_static_cloud and not self.static_occupancy.ready():
            margin = -float("inf")
            nearest = point
        else:
            dist, _, nearest = self.static_occupancy.signed_distance(
                point,
                max(self.static_cloud_search_radius, required_clearance + 0.05),
            )
            margin = dist - required_clearance
        if margin < current["margin"]:
            current["margin"] = margin
            current["obstacle"] = "static_scene_cloud"
            current["point"] = nearest

    def _format_margin(self, value):
        if math.isinf(value):
            return "inf"
        return "%.3f" % value

    def _safety_report(self, path, arm_waypoints):
        body = {"margin": float("inf"), "obstacle": "none", "point": (0.0, 0.0, 0.0)}
        arm = {"margin": float("inf"), "obstacle": "none", "point": (0.0, 0.0, 0.0)}

        body_required = self.body_radius + self.body_clearance
        arm_required = self.arm_params["radius"] + self.arm_clearance

        for pose in path:
            body_point = (pose.position.x, pose.position.y, pose.position.z)
            self._update_margin(body, body_point, self.boxes, body_required)
            self._update_cloud_margin(body, body_point, body_required)
            self._update_margin(body, body_point, self.body_keepout_boxes, 0.0)
            for arm_point in world_arm_points(pose, self.compact_theta, self.compact_phi, self._compact_arm_params()):
                self._update_margin(arm, arm_point, self.boxes, arm_required)
                self._update_cloud_margin(arm, arm_point, arm_required)

        if self.standoff_goal is not None:
            for _, theta, phi in arm_waypoints[2:]:
                for arm_point in world_arm_points(self.standoff_goal, theta, phi, self.arm_params):
                    self._update_margin(arm, arm_point, self.boxes, arm_required)
                    self._update_cloud_margin(arm, arm_point, arm_required)

        return {
            "world_boxes": self.world_box_count,
            "task_boxes": self.task_box_count,
            "total_boxes": len(self.boxes),
            "static_cloud_enabled": self.use_static_cloud,
            "static_cloud_ready": self.static_occupancy.ready(),
            "static_cloud_voxels": len(self.static_occupancy.occupied),
            "body_margin": body["margin"],
            "body_obstacle": body["obstacle"],
            "body_point": body["point"],
            "arm_margin": arm["margin"],
            "arm_obstacle": arm["obstacle"],
            "arm_point": arm["point"],
        }

    def _build_uav_route_to_goal(self, start, goal_pose):
        goal = self._pose_xyz_yaw(goal_pose)

        direct = self._route_from_points([start, goal])
        if self._route_safe(direct):
            return direct, "DIRECT_COMPACT_TRANSIT"

        astar = self._astar_route(start, goal)
        if astar is not None and self._route_safe(astar):
            return astar, "ASTAR_COMPACT_TRANSIT"

        sx, sy, sz, syaw = start
        gx, gy, gz, gyaw = goal
        cruise_z_values = [
            max(sz, gz, self.target["pose"][2] + 0.35),
            max(sz, gz, self.target["pose"][2] + 0.65, 1.25),
            max(sz, gz, self.target["pose"][2] + 0.95, 1.55),
        ]
        mid_y = 0.5 * (sy + gy)
        offsets = [0.8, -0.8, 1.2, -1.2, 1.6, -1.6, 2.0, -2.0, 2.5, -2.5]
        for cruise_z in cruise_z_values:
            for offset in offsets:
                via1 = (sx + offset, sy, cruise_z, syaw)
                via2 = (sx + offset, mid_y, cruise_z, math.atan2(gy - mid_y, gx - (sx + offset)))
                via3 = (gx + offset, gy, cruise_z, gyaw)
                route = self._route_from_points([start, via1, via2, via3, goal])
                if self._route_safe(route):
                    return route, "DETOUR_COMPACT_TRANSIT"

        return direct, "BLOCKED_COMPACT_TRANSIT"

    def _candidate_route_options(self):
        if self.reach_candidates:
            return list(self.reach_candidates)
        if self.standoff_goal is None or self.reach_theta_phi is None:
            return []
        return [
            {
                "pose": self.standoff_goal,
                "theta_phi": list(self.reach_theta_phi),
                "tip_error": self._tip_error(),
                "body_dist": 0.0,
                "score": 0.0,
            }
        ]

    def _build_uav_route(self):
        start = self._planning_start(self._pose_xyz_yaw(self.current_pose))
        options = self._candidate_route_options()
        self.planned_arm_waypoints = None
        if not options:
            return [], "BLOCKED_NO_REACH_CANDIDATE"

        best_blocked = None
        for idx, option in enumerate(options):
            self.standoff_goal = option["pose"]
            self.reach_theta_phi = list(option["theta_phi"])
            path, transit_mode = self._build_uav_route_to_goal(start, option["pose"])
            if "BLOCKED" in transit_mode:
                if best_blocked is None or float(option.get("score", 0.0)) < float(best_blocked[0].get("score", 0.0)):
                    best_blocked = (option, path, transit_mode, idx)
                continue
            arm_waypoints = self._search_arm_waypoints()
            if arm_waypoints is not None:
                self.planned_arm_waypoints = arm_waypoints
                self.last_selected_candidate = idx
                return path, transit_mode
            if best_blocked is None:
                best_blocked = (option, path, "BLOCKED_REACH_SEQUENCE", idx)

        option, path, transit_mode, idx = best_blocked
        self.standoff_goal = option["pose"]
        self.reach_theta_phi = list(option["theta_phi"])
        self.last_selected_candidate = idx
        return path, transit_mode

    def _angle_delta_deg(self, value, goal):
        return math.atan2(math.sin(math.radians(goal - value)), math.cos(math.radians(goal - value))) * 180.0 / math.pi

    def _arm_search_key(self, theta, phi):
        theta_step = max(float(self.arm_params.get("theta_step_deg", 10.0)), 1e-3)
        phi_step = max(float(self.arm_params.get("phi_step_deg", 20.0)), 1e-3)
        return (int(round(theta / theta_step)), int(round(phi / phi_step)))

    def _arm_key_value(self, key):
        theta_step = max(float(self.arm_params.get("theta_step_deg", 10.0)), 1e-3)
        phi_step = max(float(self.arm_params.get("phi_step_deg", 20.0)), 1e-3)
        return (key[0] * theta_step, key[1] * phi_step)

    def _arm_search_heuristic(self, key, goal_key):
        theta, phi = self._arm_key_value(key)
        goal_theta, goal_phi = self._arm_key_value(goal_key)
        return abs(theta - goal_theta) + 0.25 * abs(self._angle_delta_deg(phi, goal_phi))

    def _search_arm_waypoints(self):
        if self.standoff_goal is None or self.reach_theta_phi is None:
            return None
        start_key = self._arm_search_key(self.compact_theta, self.compact_phi)
        goal_key = self._arm_search_key(self.reach_theta_phi[0], self.reach_theta_phi[1])
        theta_min = float(self.arm_params["theta_min_deg"])
        theta_max = float(self.arm_params["theta_max_deg"])
        phi_min = float(self.arm_params["phi_min_deg"])
        phi_max = float(self.arm_params["phi_max_deg"])
        theta_step = max(float(self.arm_params.get("theta_step_deg", 10.0)), 1e-3)
        phi_step = max(float(self.arm_params.get("phi_step_deg", 20.0)), 1e-3)
        theta_min_key = int(math.ceil(theta_min / theta_step))
        theta_max_key = int(math.floor(theta_max / theta_step))
        phi_min_key = int(math.ceil(phi_min / phi_step))
        phi_max_key = int(math.floor(phi_max / phi_step))

        safe_cache = {}

        def is_safe_key(key):
            cached = safe_cache.get(key)
            if cached is not None:
                return cached
            theta, phi = self._arm_key_value(key)
            if abs(theta - self.compact_theta) < 1e-6 and abs(phi - self.compact_phi) < 1e-6:
                safe_cache[key] = True
                return True
            safe = self._arm_safe(self.standoff_goal, theta, phi)
            safe_cache[key] = safe
            return safe

        if not is_safe_key(goal_key):
            return None

        open_heap = []
        heapq.heappush(open_heap, (self._arm_search_heuristic(start_key, goal_key), 0.0, start_key))
        came_from = {}
        g_score = {start_key: 0.0}
        closed = set()

        while open_heap:
            _, current_g, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == goal_key:
                keys = [current]
                while keys[-1] != start_key:
                    keys.append(came_from[keys[-1]])
                keys.reverse()
                waypoints = [("compact_from_current", self.current_theta_phi[0], self.current_theta_phi[1])]
                for idx, key in enumerate(keys):
                    theta, phi = self._arm_key_value(key)
                    label = "compact_transit" if idx == 0 else "arm_search_%02d" % idx
                    waypoints.append((label, theta, phi))
                rospy.loginfo_throttle(3.0, "joint_planner_node: arm search success waypoints=%d", len(waypoints))
                return waypoints
            closed.add(current)
            neighbors = []
            for dt in [-1, 0, 1]:
                for dp in [-1, 0, 1]:
                    if dt == 0 and dp == 0:
                        continue
                    neighbors.append((current[0] + dt, current[1] + dp))
            for nb in neighbors:
                if nb in closed:
                    continue
                if nb[0] < theta_min_key or nb[0] > theta_max_key or nb[1] < phi_min_key or nb[1] > phi_max_key:
                    continue
                if not is_safe_key(nb):
                    continue
                step = 1.0 + 0.1 * abs(nb[1] - current[1])
                tentative = current_g + step
                if tentative >= g_score.get(nb, float("inf")):
                    continue
                came_from[nb] = current
                g_score[nb] = tentative
                heapq.heappush(open_heap, (tentative + self._arm_search_heuristic(nb, goal_key), tentative, nb))
        return None

    def _build_arm_waypoints(self):
        if self.planned_arm_waypoints is not None:
            return list(self.planned_arm_waypoints)
        theta_goal, phi_goal = self.reach_theta_phi
        waypoints = []
        waypoints.append(("compact_from_current", self.current_theta_phi[0], self.current_theta_phi[1]))
        waypoints.append(("compact_transit", self.compact_theta, self.compact_phi))
        steps = max(self.reach_steps, 1)
        phi_steps = max(2, steps // 2)
        theta_steps = max(2, steps - phi_steps)
        for i in range(1, phi_steps + 1):
            r = i / float(phi_steps)
            waypoints.append(("orient_phi_%02d" % i, self.compact_theta, self.compact_phi + (phi_goal - self.compact_phi) * r))
        for i in range(1, theta_steps + 1):
            r = i / float(theta_steps)
            waypoints.append(("extend_theta_%02d" % i, self.compact_theta + (theta_goal - self.compact_theta) * r, phi_goal))
        return waypoints

    def _reach_sequence_safe(self, arm_waypoints):
        if self.standoff_goal is None:
            return False
        for _, theta, phi in arm_waypoints[2:]:
            if not self._arm_safe(self.standoff_goal, theta, phi):
                return False
        return True

    def _tip_error(self):
        if self.standoff_goal is None or self.reach_theta_phi is None:
            return float("inf")
        pts = world_arm_points(self.standoff_goal, self.reach_theta_phi[0], self.reach_theta_phi[1], self.arm_params)
        return point_distance(pts[-1], tuple(self.target["pose"][:3]))

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

    def _arm_msg(self, waypoints):
        msg = Float32MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label="waypoint", size=len(waypoints), stride=len(waypoints) * 2),
            MultiArrayDimension(label="theta_phi", size=2, stride=2),
        ]
        msg.layout.data_offset = 0
        for _, theta, phi in waypoints:
            msg.data.extend([theta, phi])
        return msg

    def _plan_text(self, status, transit_mode, path, arm_waypoints, tip_error, safety):
        goal_xyz = self._pose_xyz_yaw(self.standoff_goal)[:3]
        lines = [
            "status=%s" % status,
            "method=opportunity_guided_joint_plan",
            "map_source=%s"
            % (
                "world_sdf_boxes+task_boxes+static_scene_cloud"
                if safety["static_cloud_enabled"]
                else "world_sdf_boxes+task_boxes"
            ),
            "world_boxes=%d task_boxes=%d total_boxes=%d"
            % (safety["world_boxes"], safety["task_boxes"], safety["total_boxes"]),
            "static_cloud_enabled=%s ready=%s voxels=%d"
            % (
                str(safety["static_cloud_enabled"]),
                str(safety["static_cloud_ready"]),
                safety["static_cloud_voxels"],
            ),
            "precondition=%s raw_start_z=%.3f planning_start_z=%.3f min_planning_z=%.3f"
            % (
                self.last_precondition,
                self.last_raw_start_z,
                self.last_planning_start_z,
                self.min_planning_z,
            ),
            "transit_mode=%s" % transit_mode,
            "selected_candidate=%d candidate_count=%d"
            % (self.last_selected_candidate, len(self._candidate_route_options())),
            "uav_waypoints=%d" % len(path),
            "arm_waypoints=%d" % len(arm_waypoints),
            "standoff_goal=[%.3f, %.3f, %.3f]" % goal_xyz,
            "reach_theta_phi=[%.2f, %.2f]" % (self.reach_theta_phi[0], self.reach_theta_phi[1]),
            "tip_error=%.3f" % tip_error,
            "min_body_margin=%.3f obstacle=%s point=[%.3f, %.3f, %.3f]"
            % (
                safety["body_margin"],
                safety["body_obstacle"],
                safety["body_point"][0],
                safety["body_point"][1],
                safety["body_point"][2],
            ),
            "min_arm_margin=%.3f obstacle=%s point=[%.3f, %.3f, %.3f]"
            % (
                safety["arm_margin"],
                safety["arm_obstacle"],
                safety["arm_point"][0],
                safety["arm_point"][1],
                safety["arm_point"][2],
            ),
            "stages=COMPACT_ARM -> COMPACT_TRANSIT -> HOLD_STANDOFF -> REACH_IN -> VERIFY_TIP",
        ]
        return "\n".join(lines)

    def _safety_text(self, status, transit_mode, safety):
        return (
            "status=%s transit=%s world_boxes=%d task_boxes=%d total_boxes=%d static_cloud=%s voxels=%d "
            "min_body_margin=%s body_obstacle=%s min_arm_margin=%s arm_obstacle=%s"
            % (
                status,
                transit_mode,
                safety["world_boxes"],
                safety["task_boxes"],
                safety["total_boxes"],
                str(safety["static_cloud_ready"]) if safety["static_cloud_enabled"] else "disabled",
                safety["static_cloud_voxels"],
                self._format_margin(safety["body_margin"]),
                safety["body_obstacle"],
                self._format_margin(safety["arm_margin"]),
                safety["arm_obstacle"],
            )
        )

    def _make_markers(self, path, arm_waypoints, status, transit_mode):
        stamp = rospy.Time.now()
        arr = MarkerArray()

        path_marker = Marker()
        marker_header(path_marker, self.frame_id, stamp, "joint_uav_path", 0, Marker.LINE_STRIP)
        path_marker.scale.x = 0.035
        path_marker.color = color(0.0, 0.9, 1.0, 1.0 if "BLOCKED" not in transit_mode else 0.35)
        path_marker.points = [Point(x=p.position.x, y=p.position.y, z=p.position.z) for p in path]
        arr.markers.append(path_marker)

        for idx, pose in enumerate(path[:: max(1, len(path) // 12)]):
            marker = Marker()
            marker_header(marker, self.frame_id, stamp, "joint_uav_waypoints", idx, Marker.SPHERE)
            marker.pose = pose
            marker.scale.x = 0.055
            marker.scale.y = 0.055
            marker.scale.z = 0.055
            marker.color = color(0.0, 0.9, 1.0, 0.65)
            arr.markers.append(marker)

        if self.standoff_goal is not None:
            compact_pts = world_arm_points(self.standoff_goal, self.compact_theta, self.compact_phi, self._compact_arm_params())
            compact = Marker()
            marker_header(compact, self.frame_id, stamp, "joint_compact_arm_at_standoff", 100, Marker.LINE_STRIP)
            compact.scale.x = self.arm_params["radius"] * 1.5
            compact.color = color(1.0, 0.55, 0.0, 1.0)
            compact.points = [point_msg(*p) for p in compact_pts]
            arr.markers.append(compact)

            reach_theta, reach_phi = arm_waypoints[-1][1], arm_waypoints[-1][2]
            reach_pts = world_arm_points(self.standoff_goal, reach_theta, reach_phi, self.arm_params)
            reach = Marker()
            marker_header(reach, self.frame_id, stamp, "joint_reach_in_arm", 101, Marker.LINE_STRIP)
            reach.scale.x = self.arm_params["radius"] * 2.5
            reach.color = color(1.0, 1.0, 0.0, 1.0)
            reach.points = [point_msg(*p) for p in reach_pts]
            arr.markers.append(reach)

        text = Marker()
        marker_header(text, self.frame_id, stamp, "joint_plan_status", 200, Marker.TEXT_VIEW_FACING)
        text.pose.position.x = self.target["pose"][0]
        text.pose.position.y = self.target["pose"][1] - 0.8
        text.pose.position.z = self.target["pose"][2] + 0.9
        text.pose.orientation.w = 1.0
        text.scale.z = 0.14
        text.color = color(1.0, 1.0, 1.0, 1.0)
        text.text = "%s\n%s\ncompact transit + reach-in" % (status, transit_mode)
        arr.markers.append(text)
        return arr

    def _publish_waiting(self):
        missing = []
        if self.current_pose is None:
            missing.append("current_uav_pose")
        if not self.reach_candidates and (self.standoff_goal is None or self.reach_theta_phi is None):
            missing.append("planner_reach_candidates")
        if self.use_static_cloud and self.require_static_cloud and not self.static_occupancy.ready():
            missing.append("static_scene_cloud")
        msg = "WAITING_FOR_INPUT " + ",".join(missing)
        self.status_pub.publish(String(data=msg))

    def _plan_once(self):
        if (
            self.current_pose is None
            or (not self.reach_candidates and (self.standoff_goal is None or self.reach_theta_phi is None))
            or (self.use_static_cloud and self.require_static_cloud and not self.static_occupancy.ready())
        ):
            self._publish_waiting()
            if self.use_static_cloud:
                self.occupancy_pub.publish(String(data=self.static_occupancy.report()))
            return

        path, transit_mode = self._build_uav_route()
        arm_waypoints = self._build_arm_waypoints()
        transit_ok = transit_mode not in ["BLOCKED_COMPACT_TRANSIT", "BLOCKED_NO_REACH_CANDIDATE"]
        reach_ok = self._reach_sequence_safe(arm_waypoints)
        tip_error = self._tip_error()
        safety = self._safety_report(path, arm_waypoints)

        if transit_ok and reach_ok:
            status = "JOINT_PLAN_READY"
        elif not transit_ok:
            status = "JOINT_PLAN_BLOCKED_TRANSIT"
        else:
            status = "JOINT_PLAN_BLOCKED_REACH_SEQUENCE"

        rospy.loginfo_throttle(
            3.0,
            "joint_planner_node: %s transit=%s tip_error=%.3f body_margin=%s arm_margin=%s",
            status,
            transit_mode,
            tip_error,
            self._format_margin(safety["body_margin"]),
            self._format_margin(safety["arm_margin"]),
        )

        self.status_pub.publish(
            String(
                data="%s transit=%s precondition=%s tip_error=%.3f min_body_margin=%s min_arm_margin=%s"
                % (
                    status,
                    transit_mode,
                    self.last_precondition,
                    tip_error,
                    self._format_margin(safety["body_margin"]),
                    self._format_margin(safety["arm_margin"]),
                )
            )
        )
        self.safety_pub.publish(String(data=self._safety_text(status, transit_mode, safety)))
        self.plan_pub.publish(String(data=self._plan_text(status, transit_mode, path, arm_waypoints, tip_error, safety)))
        self.path_pub.publish(self._path_msg(path))
        self.arm_waypoints_pub.publish(self._arm_msg(arm_waypoints))
        self.marker_pub.publish(self._make_markers(path, arm_waypoints, status, transit_mode))

    def spin(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            self._plan_once()
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("joint_planner_node")
    try:
        JointPlannerNode().spin()
    except rospy.ROSInterruptException:
        pass
