#!/usr/bin/env python3
import copy
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import rospy
import tf.transformations as tft
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, String

try:
    from traj_utils.msg import Bspline
except ImportError:
    Bspline = None

from uav_continuum_opportunity_planner.common import point_distance, point_msg


def param_bool(name, default=False):
    value = rospy.get_param(name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ["1", "true", "yes", "on"]
    return bool(value)


class EgoReferenceBridgeNode:
    def __init__(self):
        self.frame_id = rospy.get_param("/frame_id", "world")
        self.scenario_name = rospy.get_param("/scenario/name", "unknown")
        self.mode = rospy.get_param("~mode", "prefer_real_ego")
        self.spacing = float(rospy.get_param("~reference_spacing", 0.25))
        self.max_points = int(rospy.get_param("~max_reference_points", 120))
        self.publish_period = float(rospy.get_param("~publish_period", 1.0))
        self.goal_change_distance = float(rospy.get_param("~goal_change_distance", 0.15))
        self.goal_keepalive_period = float(rospy.get_param("~goal_keepalive_period", 0.0))
        self.maximum_ego_start_gap = float(rospy.get_param("~maximum_ego_start_gap", 0.0))
        self.ego_bspline_topic = rospy.get_param("~ego_bspline_topic", "/xtdrone/iris_0/planning/bspline")
        self.ego_odom_topic = rospy.get_param("~ego_odom_topic", "/vins_estimator/odometry")
        self.ego_goal_topic = rospy.get_param("~ego_goal_topic", "/move_base_simple/goal")
        self.ego_goal_source = rospy.get_param("~ego_goal_source", "global_path_local")
        self.local_goal_lookahead = float(rospy.get_param("~local_goal_lookahead", 3.0))
        self.local_goal_min_lookahead = float(rospy.get_param("~local_goal_min_lookahead", 1.0))
        self.enable_rolling_goal = param_bool("~enable_rolling_goal", True)
        self.rolling_goal_stale_age = float(rospy.get_param("~rolling_goal_stale_age", 6.0))
        self.rolling_goal_advance = float(rospy.get_param("~rolling_goal_advance", 2.0))
        self.rolling_goal_max_ahead = float(rospy.get_param("~rolling_goal_max_ahead", 6.0))
        self.min_reference_z = float(rospy.get_param("~min_reference_z", rospy.get_param("/global_path/z_min", 0.0)))
        self.align_reference_z_to_global_path = param_bool("~align_reference_z_to_global_path", True)
        self.global_path_z_margin = float(rospy.get_param("~global_path_z_margin", 0.0))
        self.append_global_tail = param_bool("~append_global_tail", True)
        self.allow_global_path_fallback = param_bool("~allow_global_path_fallback", False)
        self.apply_ego_odom_offset = param_bool("~apply_ego_odom_offset", True)
        self.publish_selected_goal_to_ego = param_bool("~publish_selected_goal_to_ego", True)
        self.allow_uav_goal_fallback_to_ego = param_bool("~allow_uav_goal_fallback_to_ego", False)
        self.ego_timeout = float(rospy.get_param("~ego_timeout", 1.5))
        self.reuse_stale_ego_reference = param_bool("~reuse_stale_ego_reference", True)
        self.ego_reference_valid_grace = float(rospy.get_param("~ego_reference_valid_grace", 2.0))
        self.stale_ego_reference_timeout = float(rospy.get_param("~stale_ego_reference_timeout", 120.0))
        self.fallback_after = float(rospy.get_param("~fallback_after", 3.0))
        self.global_path = None
        self.selected_goal = None
        self.uav_goal = None
        self.last_goal_source = "none"
        self.last_goal_world = None
        self.last_goal_path_idx = -1
        self.last_goal_nearest_idx = -1
        self.last_goal_distance = 0.0
        self.last_goal_publish_time = rospy.Time(0)
        self.current_pose = None
        self.ego_odom_pose = None
        self.last_bspline = None
        self.last_bspline_time = rospy.Time(0)
        self.start_time = rospy.Time.now()
        self.last_publish_time = rospy.Time(0)
        self.last_ref_msg = None
        self.reference_republish_period = float(rospy.get_param("~reference_republish_period", 0.5))

        uav_pose_topic = rospy.get_param("~uav_pose_topic", "/iris_0/world_pose")
        rospy.Subscriber("/planner/global_path", Path, self._path_cb, queue_size=1)
        rospy.Subscriber("/planner/selected_reach_goal", PoseStamped, self._selected_goal_cb, queue_size=1)
        rospy.Subscriber("/planner/uav_goal", PoseStamped, self._uav_goal_cb, queue_size=1)
        rospy.Subscriber(uav_pose_topic, PoseStamped, self._pose_cb, queue_size=1)
        rospy.Subscriber(self.ego_odom_topic, Odometry, self._ego_odom_cb, queue_size=1)
        if self.mode in ["real_ego", "prefer_real_ego"] and Bspline is not None:
            rospy.Subscriber(self.ego_bspline_topic, Bspline, self._bspline_cb, queue_size=1)

        self.ref_pub = rospy.Publisher("/planner/ego_reference_trajectory", Path, queue_size=1, latch=True)
        self.ref_array_pub = rospy.Publisher("/planner/ego_reference_trajectory_array", Float32MultiArray, queue_size=1, latch=True)
        self.report_pub = rospy.Publisher("/planner/ego_reference_report", String, queue_size=1, latch=True)
        self.ego_goal_pub = rospy.Publisher(self.ego_goal_topic, PoseStamped, queue_size=1, latch=True)
        rospy.Timer(rospy.Duration(self.reference_republish_period), self._republish_reference_cb)
        rospy.loginfo(
            "ego_reference_bridge_node: mode=%s ego_bspline_topic=%s fallback=%s",
            self.mode,
            self.ego_bspline_topic,
            str(self.allow_global_path_fallback),
        )

    def _path_cb(self, msg):
        self.global_path = msg

    def _selected_goal_cb(self, msg):
        self.selected_goal = msg
        self._publish_selected_goal_to_ego()

    def _uav_goal_cb(self, msg):
        self.uav_goal = msg
        self._publish_selected_goal_to_ego()

    def _pose_cb(self, msg):
        self.current_pose = msg.pose

    def _ego_odom_cb(self, msg):
        self.ego_odom_pose = msg.pose.pose

    def _bspline_cb(self, msg):
        self.last_bspline = msg
        self.last_bspline_time = rospy.Time.now()

    def _yaw_between(self, a, b):
        return math.atan2(b[1] - a[1], b[0] - a[0])

    def _make_pose_stamped(self, xyz, yaw, stamp):
        msg = PoseStamped()
        msg.header.frame_id = self.frame_id
        msg.header.stamp = stamp
        msg.pose.position = point_msg(*xyz)
        q = tft.quaternion_from_euler(0.0, 0.0, yaw)
        msg.pose.orientation.x = q[0]
        msg.pose.orientation.y = q[1]
        msg.pose.orientation.z = q[2]
        msg.pose.orientation.w = q[3]
        return msg

    def _points_from_path(self, path):
        return [(p.pose.position.x, p.pose.position.y, p.pose.position.z) for p in path.poses]

    def _apply_reference_floor(self, points):
        if self.min_reference_z <= 0.0:
            return points
        return [(p[0], p[1], max(p[2], self.min_reference_z)) for p in points]

    def _apply_global_path_z_floor(self, points):
        if not self.align_reference_z_to_global_path or self.global_path is None or not self.global_path.poses:
            return points, 0
        global_points = self._points_from_path(self.global_path)
        adjusted = []
        adjusted_count = 0
        for point in points:
            best = None
            best_xy = float("inf")
            for gp in global_points:
                dx = point[0] - gp[0]
                dy = point[1] - gp[1]
                dist_xy = dx * dx + dy * dy
                if dist_xy < best_xy:
                    best_xy = dist_xy
                    best = gp
            floor_z = best[2] + self.global_path_z_margin if best is not None else point[2]
            z = max(point[2], floor_z)
            if z > point[2] + 1e-6:
                adjusted_count += 1
            adjusted.append((point[0], point[1], z))
        return adjusted, adjusted_count

    def _append_global_path_tail(self, ref_points):
        if not self.append_global_tail or self.global_path is None or not self.global_path.poses or not ref_points:
            return ref_points, False
        global_points = self._points_from_path(self.global_path)
        if len(global_points) < 2:
            return ref_points, False
        anchor = ref_points[-1]
        best_idx = 0
        best_dist = float("inf")
        for idx, point in enumerate(global_points):
            dist = point_distance(anchor, point)
            if dist < best_dist:
                best_idx = idx
                best_dist = dist
        tail = global_points[best_idx + 1 :]
        if not tail:
            return ref_points, False
        return self._resample(ref_points + tail), True

    def _ego_to_world_offset(self):
        if not self.apply_ego_odom_offset:
            return (0.0, 0.0, 0.0)
        if self.current_pose is None or self.ego_odom_pose is None:
            return None
        return (
            self.current_pose.position.x - self.ego_odom_pose.position.x,
            self.current_pose.position.y - self.ego_odom_pose.position.y,
            self.current_pose.position.z - self.ego_odom_pose.position.z,
        )

    def _pose_from_world_point(self, xyz, yaw=0.0):
        pose = PoseStamped()
        pose.header.frame_id = self.frame_id
        pose.pose.position = point_msg(*xyz)
        q = tft.quaternion_from_euler(0.0, 0.0, yaw)
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]
        return pose

    def _select_global_path_local_goal(self):
        if self.global_path is None or len(self.global_path.poses) < 2 or self.current_pose is None:
            return None, "global_path_local_unavailable"
        points = self._points_from_path(self.global_path)
        current = (self.current_pose.position.x, self.current_pose.position.y, self.current_pose.position.z)
        nearest_idx = 0
        nearest_dist = float("inf")
        for idx, point in enumerate(points):
            dist = point_distance(current, point)
            if dist < nearest_dist:
                nearest_idx = idx
                nearest_dist = dist
        target_idx = nearest_idx
        distance = 0.0
        min_lookahead = max(self.local_goal_min_lookahead, 0.0)
        lookahead = max(self.local_goal_lookahead, min_lookahead)
        while target_idx < len(points) - 1 and distance < lookahead:
            step = point_distance(points[target_idx], points[target_idx + 1])
            distance += step
            target_idx += 1
        if distance < min_lookahead and target_idx < len(points) - 1:
            target_idx += 1
        rolling_used = False
        # A stale EGO spline is not evidence that the vehicle progressed.  The
        # old implementation advanced the goal while the vehicle was held at
        # the start, eventually pushing it outside EGO's useful local region.
        # Only apply the optional extra advance after measured path progress.
        if (self.enable_rolling_goal and self.last_goal_path_idx >= 0 and
                self.last_goal_nearest_idx >= 0 and nearest_idx > self.last_goal_nearest_idx and
                self.last_bspline is not None):
            bspline_age = (rospy.Time.now() - self.last_bspline_time).to_sec()
            if bspline_age >= self.rolling_goal_stale_age:
                advance_idx = self.last_goal_path_idx
                advance_dist = 0.0
                while advance_idx < len(points) - 1 and advance_dist < max(self.rolling_goal_advance, 0.0):
                    step = point_distance(points[advance_idx], points[advance_idx + 1])
                    advance_dist += step
                    advance_idx += 1
                max_idx = nearest_idx
                max_dist = 0.0
                max_ahead = max(self.rolling_goal_max_ahead, lookahead)
                while max_idx < len(points) - 1 and max_dist < max_ahead:
                    step = point_distance(points[max_idx], points[max_idx + 1])
                    max_dist += step
                    max_idx += 1
                rolled_idx = min(max(target_idx, advance_idx), max_idx)
                if rolled_idx > target_idx:
                    target_idx = rolled_idx
                    distance = 0.0
                    for idx in range(nearest_idx, target_idx):
                        distance += point_distance(points[idx], points[idx + 1])
                    rolling_used = True
        point = points[target_idx]
        if target_idx < len(points) - 1:
            yaw = self._yaw_between(point, points[target_idx + 1])
        elif target_idx > 0:
            yaw = self._yaw_between(points[target_idx - 1], point)
        else:
            yaw = 0.0
        goal = self._pose_from_world_point(point, yaw)
        self.last_goal_path_idx = target_idx
        self.last_goal_nearest_idx = nearest_idx
        self.last_goal_distance = distance
        tag = "rolling" if rolling_used else "local"
        return goal, "global_path_%s_idx_%d_nearest_%d_dist_%.2f" % (tag, target_idx, nearest_idx, distance)

    def _select_ego_goal_world(self):
        if self.ego_goal_source == "global_path_local":
            goal, source = self._select_global_path_local_goal()
            if goal is not None:
                return goal, source
        if self.selected_goal is not None:
            return self.selected_goal, "selected_reach_goal"
        if self.allow_uav_goal_fallback_to_ego and self.uav_goal is not None:
            return self.uav_goal, "planner_uav_goal"
        return None, "none"

    def _publish_selected_goal_to_ego(self, force=False):
        if not self.publish_selected_goal_to_ego:
            return False
        now = rospy.Time.now()
        if not force and (now - self.last_goal_publish_time).to_sec() < self.publish_period:
            return False
        offset = self._ego_to_world_offset()
        if offset is None:
            return False
        selected_goal, goal_source = self._select_ego_goal_world()
        if selected_goal is None:
            self.last_goal_source = goal_source
            return False
        selected_world = (
            selected_goal.pose.position.x,
            selected_goal.pose.position.y,
            selected_goal.pose.position.z,
        )
        if not force and self.last_goal_world is not None:
            change = point_distance(selected_world, self.last_goal_world)
            keepalive_due = (
                self.goal_keepalive_period > 0.0 and
                (now - self.last_goal_publish_time).to_sec() >= self.goal_keepalive_period
            )
            if change < self.goal_change_distance and not keepalive_due:
                return False
        goal = PoseStamped()
        goal.header.stamp = now
        goal.header.frame_id = self.frame_id
        goal.pose = copy.deepcopy(selected_goal.pose)
        goal.pose.position.x = selected_goal.pose.position.x - offset[0]
        goal.pose.position.y = selected_goal.pose.position.y - offset[1]
        goal.pose.position.z = selected_goal.pose.position.z - offset[2]
        self.ego_goal_pub.publish(goal)
        self.last_goal_source = goal_source
        self.last_goal_world = selected_world
        self.last_goal_publish_time = now
        return True

    def _points_from_bspline(self, msg, offset):
        if msg is None or msg.order < 1 or len(msg.pos_pts) < msg.order + 1:
            return []
        knots = list(msg.knots)
        points = [(p.x + offset[0], p.y + offset[1], p.z + offset[2]) for p in msg.pos_pts]
        order = int(msg.order)
        if len(knots) < len(points) + order + 1:
            return points[: self.max_points]
        start_t = knots[order]
        end_t = knots[len(points)]
        if end_t <= start_t:
            return points[: self.max_points]
        sample_count = max(2, min(self.max_points * 3, int(math.ceil((end_t - start_t) / 0.05)) + 1))
        sampled = []
        for idx in range(sample_count):
            ratio = float(idx) / float(sample_count - 1)
            t = start_t + (end_t - start_t) * ratio
            sampled.append(self._de_boor(order, knots, points, t))
        return self._resample(sampled)

    def _bspline_duration(self, msg):
        if msg is None or msg.order < 1 or not msg.pos_pts:
            return 0.0
        knots = list(msg.knots)
        order = int(msg.order)
        if len(knots) < len(msg.pos_pts) + order + 1:
            return 0.0
        start_t = float(knots[order])
        end_t = float(knots[len(msg.pos_pts)])
        return max(0.0, end_t - start_t)

    def _bspline_valid_age_limit(self, msg):
        duration = self._bspline_duration(msg)
        if duration <= 1e-6:
            return self.ego_timeout
        return min(max(self.ego_timeout, duration + self.ego_reference_valid_grace), self.stale_ego_reference_timeout)

    def _de_boor(self, order, knots, points, t):
        n = len(points) - 1
        if t >= knots[n + 1]:
            return points[n]
        span = order
        for idx in range(order, n + 1):
            if knots[idx] <= t < knots[idx + 1]:
                span = idx
                break
        d = [list(points[span - order + j]) for j in range(order + 1)]
        for r in range(1, order + 1):
            for j in range(order, r - 1, -1):
                left = knots[span - order + j]
                right = knots[span + j - r + 1]
                alpha = 0.0 if abs(right - left) < 1e-9 else (t - left) / (right - left)
                d[j][0] = (1.0 - alpha) * d[j - 1][0] + alpha * d[j][0]
                d[j][1] = (1.0 - alpha) * d[j - 1][1] + alpha * d[j][1]
                d[j][2] = (1.0 - alpha) * d[j - 1][2] + alpha * d[j][2]
        return tuple(d[order])

    def _nearest_start_index(self, points):
        if self.current_pose is None or not points:
            return 0
        current = (self.current_pose.position.x, self.current_pose.position.y, self.current_pose.position.z)
        best_idx = 0
        best_dist = float("inf")
        for idx, point in enumerate(points):
            dist = point_distance(current, point)
            if dist < best_dist:
                best_idx = idx
                best_dist = dist
        return best_idx

    def _resample(self, points):
        if len(points) < 2:
            return points
        start_idx = self._nearest_start_index(points)
        points = points[start_idx:]
        if len(points) < 2:
            return points
        out = [points[0]]
        carry = 0.0
        last = points[0]
        for point in points[1:]:
            seg_len = point_distance(last, point)
            if seg_len < 1e-6:
                continue
            direction = ((point[0] - last[0]) / seg_len, (point[1] - last[1]) / seg_len, (point[2] - last[2]) / seg_len)
            dist_along = self.spacing - carry
            while dist_along <= seg_len + 1e-9 and len(out) < self.max_points:
                out.append((last[0] + direction[0] * dist_along, last[1] + direction[1] * dist_along, last[2] + direction[2] * dist_along))
                dist_along += self.spacing
            carry = max(0.0, seg_len - (dist_along - self.spacing))
            last = point
            if len(out) >= self.max_points:
                break
        if len(out) < self.max_points and point_distance(out[-1], points[-1]) > 1e-3:
            out.append(points[-1])
        return out

    def _publish_report(self, status, **kwargs):
        lines = ["status=%s" % status, "scenario=%s" % self.scenario_name, "mode=%s" % self.mode]
        for key in sorted(kwargs):
            lines.append("%s=%s" % (key, kwargs[key]))
        self.report_pub.publish(String(data="\n".join(lines)))

    def _path_from_points(self, ref_points):
        stamp = rospy.Time.now()
        msg = Path()
        msg.header.frame_id = self.frame_id
        msg.header.stamp = stamp
        for idx, point in enumerate(ref_points):
            if idx < len(ref_points) - 1:
                yaw = self._yaw_between(point, ref_points[idx + 1])
            else:
                yaw = self._yaw_between(ref_points[idx - 1], point)
            msg.poses.append(self._make_pose_stamped(point, yaw, stamp))
        return msg

    def _publish_reference_msg(self, msg):
        self.last_ref_msg = msg
        self.ref_pub.publish(msg)
        array_msg = self._path_array_msg(msg)
        self.ref_array_pub.publish(array_msg)
        try:
            rospy.set_param(
                "/planner/latest_ego_reference_array_cache",
                {
                    "stamp": rospy.Time.now().to_sec(),
                    "frame_id": msg.header.frame_id,
                    "count": len(msg.poses),
                    "data": list(array_msg.data),
                },
            )
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "ego_reference_bridge_node: failed to cache reference param: %s", exc)

    def _invalidate_reference(self):
        """Replace a latched stale path with an explicit empty reference."""
        self.last_ref_msg = None
        msg = Path()
        msg.header.frame_id = self.frame_id
        msg.header.stamp = rospy.Time.now()
        self.ref_pub.publish(msg)
        self.ref_array_pub.publish(self._path_array_msg(msg))
        try:
            rospy.set_param(
                "/planner/latest_ego_reference_array_cache",
                {"stamp": msg.header.stamp.to_sec(), "frame_id": self.frame_id, "count": 0, "data": []},
            )
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "ego_reference_bridge_node: failed to invalidate reference cache: %s", exc)

    def _republish_reference_cb(self, _event):
        # Copy the object reference once.  Invalidation may run in another
        # rospy callback between the old None check and publication.
        path_msg = self.last_ref_msg
        if path_msg is None:
            return
        path_msg.header.stamp = rospy.Time.now()
        for pose_stamped in path_msg.poses:
            pose_stamped.header.stamp = path_msg.header.stamp
        self.ref_pub.publish(path_msg)
        self.ref_array_pub.publish(self._path_array_msg(path_msg))

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

    def _reference_length(self, points):
        length = 0.0
        for a, b in zip(points, points[1:]):
            length += point_distance(a, b)
        return length

    def _publish_global_path_fallback(self):
        if self.global_path is None or not self.global_path.poses:
            self._publish_report("WAITING_GLOBAL_PATH")
            return
        points = self._points_from_path(self.global_path)
        ref_points = self._resample(points)
        if len(ref_points) < 2:
            self._publish_report("REFERENCE_TOO_SHORT", source="global_path_fallback", source_waypoints=len(points), output_waypoints=len(ref_points))
            return
        self._publish_reference_msg(self._path_from_points(ref_points))
        length = self._reference_length(ref_points)
        self._publish_report(
            "EGO_REFERENCE_READY",
            source="global_path_fallback",
            source_waypoints=len(points),
            output_waypoints=len(ref_points),
            reference_length="%.3f" % length,
            spacing="%.3f" % self.spacing,
        )

    def _publish_real_ego_reference(self):
        if Bspline is None:
            self._publish_report("WAITING_EGO_BSPLINE_TYPE", source="real_ego", topic=self.ego_bspline_topic)
            return False
        now = rospy.Time.now()
        if self.last_bspline is None:
            self._publish_report("WAITING_EGO_BSPLINE", source="real_ego", topic=self.ego_bspline_topic)
            return False
        age = (now - self.last_bspline_time).to_sec()
        duration = self._bspline_duration(self.last_bspline)
        valid_age_limit = self._bspline_valid_age_limit(self.last_bspline)
        reused = age > self.ego_timeout
        expired = age > valid_age_limit
        if reused and (not self.reuse_stale_ego_reference or expired):
            self._invalidate_reference()
            self._publish_report(
                "WAITING_FRESH_EGO_BSPLINE",
                source="real_ego",
                topic=self.ego_bspline_topic,
                age="%.2f" % age,
                duration="%.2f" % duration,
                valid_age_limit="%.2f" % valid_age_limit,
            )
            return False
        offset = self._ego_to_world_offset()
        if offset is None:
            self._publish_report("WAITING_EGO_WORLD_ALIGNMENT", source="real_ego", topic=self.ego_odom_topic)
            return False
        ref_points = self._points_from_bspline(self.last_bspline, offset)
        ref_points = self._apply_reference_floor(ref_points)
        ref_points, z_adjusted_count = self._apply_global_path_z_floor(ref_points)
        ref_points, added_tail = self._append_global_path_tail(ref_points)
        if len(ref_points) < 2:
            self._publish_report(
                "REFERENCE_TOO_SHORT",
                source="real_ego",
                control_points=len(self.last_bspline.pos_pts),
                output_waypoints=len(ref_points),
                traj_id=self.last_bspline.traj_id,
            )
            return False
        if self.maximum_ego_start_gap > 0.0 and self.current_pose is not None:
            current = (self.current_pose.position.x, self.current_pose.position.y, self.current_pose.position.z)
            start_gap = point_distance(current, ref_points[0])
            if start_gap > self.maximum_ego_start_gap:
                self._invalidate_reference()
                self._publish_report(
                    "EGO_REFERENCE_DISCONTINUOUS",
                    source="real_ego",
                    topic=self.ego_bspline_topic,
                    start_gap="%.3f" % start_gap,
                    maximum_start_gap="%.3f" % self.maximum_ego_start_gap,
                    fallback_allowed=str(self.allow_global_path_fallback),
                )
                return False
        self._publish_reference_msg(self._path_from_points(ref_points))
        length = self._reference_length(ref_points)
        self._publish_report(
            "EGO_REFERENCE_READY_REUSED" if reused else "EGO_REFERENCE_READY",
            source="real_ego",
            topic=self.ego_bspline_topic,
            traj_id=self.last_bspline.traj_id,
            order=self.last_bspline.order,
            control_points=len(self.last_bspline.pos_pts),
            output_waypoints=len(ref_points),
            reference_length="%.3f" % length,
            spacing="%.3f" % self.spacing,
            age="%.2f" % age,
            duration="%.2f" % duration,
            reused=str(reused),
            valid_age_limit="%.2f" % valid_age_limit,
            world_offset="[%.3f, %.3f, %.3f]" % offset,
            min_reference_z="%.3f" % self.min_reference_z,
            global_path_z_floor=str(self.align_reference_z_to_global_path),
            global_path_z_adjusted=z_adjusted_count,
            global_tail=str(added_tail),
            selected_goal_to_ego=str(self.publish_selected_goal_to_ego),
            ego_goal_source=self.last_goal_source,
            rolling_goal=str(self.enable_rolling_goal),
            rolling_goal_idx=self.last_goal_path_idx,
            rolling_goal_nearest_idx=self.last_goal_nearest_idx,
            rolling_goal_distance="%.2f" % self.last_goal_distance,
            ego_goal_world=(
                "[%.3f, %.3f, %.3f]" % self.last_goal_world
                if self.last_goal_world is not None
                else "none"
            ),
        )
        return True

    def _publish_reference(self):
        if self.mode == "from_global_path":
            self._publish_global_path_fallback()
            return
        if self.mode == "real_ego":
            self._publish_real_ego_reference()
            return
        if self.mode != "prefer_real_ego":
            self._publish_report("UNSUPPORTED_MODE")
            return
        if self._publish_real_ego_reference():
            return
        waited = (rospy.Time.now() - self.start_time).to_sec()
        if self.allow_global_path_fallback and waited >= self.fallback_after:
            self._publish_global_path_fallback()

    def spin(self):
        rate = rospy.Rate(float(rospy.get_param("~rate_hz", 5.0)))
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            if (now - self.last_publish_time).to_sec() >= self.publish_period:
                self.last_publish_time = now
                self._publish_selected_goal_to_ego()
                self._publish_reference()
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("ego_reference_bridge_node")
    try:
        EgoReferenceBridgeNode().spin()
    except rospy.ROSInterruptException:
        pass
