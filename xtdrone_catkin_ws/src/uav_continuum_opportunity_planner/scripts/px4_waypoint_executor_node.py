#!/usr/bin/env python3
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from nav_msgs.msg import Path
from std_msgs.msg import Float32MultiArray, String


class Px4WaypointExecutorNode:
    def __init__(self):
        self.pose_topic = rospy.get_param("~pose_topic", "/iris_0/mavros/local_position/pose")
        self.state_topic = rospy.get_param("~state_topic", "/iris_0/mavros/state")
        self.setpoint_topic = rospy.get_param("~setpoint_topic", "/iris_0/mavros/setpoint_position/local")
        self.arm_cmd_topic = rospy.get_param("~arm_cmd_topic", "/arm/theta_phi_cmd")
        self.set_mode_service = rospy.get_param("~set_mode_service", "/iris_0/mavros/set_mode")
        self.arming_service = rospy.get_param("~arming_service", "/iris_0/mavros/cmd/arming")

        self.rate_hz = float(rospy.get_param("~rate_hz", 30.0))
        self.acceptance_radius = float(rospy.get_param("~acceptance_radius", 0.28))
        self.acceptance_z = float(rospy.get_param("~acceptance_z", 0.22))
        self.max_step_wait_sec = float(rospy.get_param("~max_step_wait_sec", 18.0))
        self.goal_hold_sec = float(rospy.get_param("~goal_hold_sec", 1.0))
        self.min_flight_z = float(rospy.get_param("~min_flight_z", 0.75))
        self.direct_to_standoff = bool(rospy.get_param("~direct_to_standoff", True))
        self.require_offboard_armed = bool(rospy.get_param("~require_offboard_armed", True))
        self.require_plan_ready = bool(rospy.get_param("~require_plan_ready", True))
        self.auto_offboard = bool(rospy.get_param("~auto_offboard", False))
        self.auto_arm = bool(rospy.get_param("~auto_arm", False))
        self.execute_arm = bool(rospy.get_param("~execute_arm", True))
        self.path_stride = max(1, int(rospy.get_param("~path_stride", 8)))
        self.direct_to_standoff = bool(rospy.get_param("~direct_to_standoff", self.direct_to_standoff))

        self.path = None
        self.arm_waypoints = None
        self.pose = None
        self.state = State()
        self.command = "IDLE"
        self.running = False
        self.last_result = "NONE"
        self.active_setpoint = None
        self.exec_path = None
        self.exec_arm_waypoints = None
        self.plan_status = "UNKNOWN"

        rospy.Subscriber("/planner/uav_waypoints", Path, self._path_cb, queue_size=1)
        rospy.Subscriber("/planner/arm_waypoints", Float32MultiArray, self._arm_cb, queue_size=1)
        rospy.Subscriber("/planner/joint_status", String, self._plan_status_cb, queue_size=1)
        rospy.Subscriber("/planner/task_plan", String, self._plan_status_cb, queue_size=1)
        rospy.Subscriber("/planner/px4_executor_cmd", String, self._cmd_cb, queue_size=10)
        rospy.Subscriber(self.pose_topic, PoseStamped, self._pose_cb, queue_size=1)
        rospy.Subscriber(self.state_topic, State, self._state_cb, queue_size=1)

        self.setpoint_pub = rospy.Publisher(self.setpoint_topic, PoseStamped, queue_size=10)
        self.arm_pub = rospy.Publisher(self.arm_cmd_topic, Float32MultiArray, queue_size=10)
        self.status_pub = rospy.Publisher("/planner/px4_executor_status", String, queue_size=10)

        self.set_mode = rospy.ServiceProxy(self.set_mode_service, SetMode)
        self.arming = rospy.ServiceProxy(self.arming_service, CommandBool)

        rospy.loginfo(
            "px4_waypoint_executor_node: setpoint=%s pose=%s auto_offboard=%s auto_arm=%s",
            self.setpoint_topic,
            self.pose_topic,
            self.auto_offboard,
            self.auto_arm,
        )

    def _path_cb(self, msg):
        filtered = Path()
        filtered.header = msg.header
        if msg.poses:
            if self.direct_to_standoff:
                filtered.poses = [msg.poses[-1]]
            else:
                filtered.poses = list(msg.poses[:: self.path_stride])
                if filtered.poses[-1] is not msg.poses[-1]:
                    filtered.poses.append(msg.poses[-1])
        self.path = filtered

    def _arm_cb(self, msg):
        data = list(msg.data)
        self.arm_waypoints = [(data[i], data[i + 1]) for i in range(0, len(data) - 1, 2)]

    def _plan_status_cb(self, msg):
        self.plan_status = msg.data.strip()

    def _plan_ready(self):
        return self.plan_status.startswith("JOINT_PLAN_READY") or "status=WHOLE_BODY_PLAN_READY" in self.plan_status

    def _pose_cb(self, msg):
        self.pose = msg

    def _state_cb(self, msg):
        self.state = msg

    def _cmd_cb(self, msg):
        cmd = msg.data.strip().upper()
        if cmd in ["START", "STOP", "RESET"]:
            self.command = cmd

    def _ready(self):
        return (
            self.path is not None
            and bool(self.path.poses)
            and self.arm_waypoints is not None
            and bool(self.arm_waypoints)
            and self.pose is not None
            and self.state.connected
        )

    def _publish_active_setpoint(self):
        if self.active_setpoint is None:
            return
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.pose.header.frame_id if self.pose is not None else "map"
        msg.pose = self.active_setpoint.pose
        self.setpoint_pub.publish(msg)

    def _copy_waypoint(self, waypoint, min_z=None):
        out = PoseStamped()
        out.header = waypoint.header
        out.pose = waypoint.pose
        if min_z is not None and out.pose.position.z < min_z:
            out.pose.position.z = min_z
        return out

    def _publish_arm(self, theta, phi):
        if not self.execute_arm:
            return
        msg = Float32MultiArray()
        msg.data = [float(theta), float(phi)]
        self.arm_pub.publish(msg)

    def _dist_to(self, waypoint):
        if self.pose is None:
            return float("inf"), float("inf")
        p = self.pose.pose.position
        g = waypoint.pose.position
        xy = math.sqrt((p.x - g.x) ** 2 + (p.y - g.y) ** 2)
        dz = abs(p.z - g.z)
        return xy, dz

    def _ensure_mode_and_arm(self):
        if self.auto_offboard and self.state.mode != "OFFBOARD":
            try:
                self.set_mode(0, "OFFBOARD")
            except rospy.ServiceException as exc:
                self.status_pub.publish(String(data="SET_OFFBOARD_FAILED %s" % exc))
                return False
        if self.auto_arm and not self.state.armed:
            try:
                self.arming(True)
            except rospy.ServiceException as exc:
                self.status_pub.publish(String(data="ARM_FAILED %s" % exc))
                return False
        if self.require_offboard_armed and (self.state.mode != "OFFBOARD" or not self.state.armed):
            self.status_pub.publish(
                String(data="WAIT_OFFBOARD_ARMED connected=%s armed=%s mode=%s" % (self.state.connected, self.state.armed, self.state.mode))
            )
            return False
        return True

    def _prime_setpoints(self, waypoint):
        self.active_setpoint = waypoint
        rate = rospy.Rate(self.rate_hz)
        for _ in range(int(max(self.rate_hz, 1.0))):
            if rospy.is_shutdown() or self.command == "STOP":
                return False
            self._publish_active_setpoint()
            rate.sleep()
        return True

    def _run_compact_arm(self):
        arm_waypoints = self.exec_arm_waypoints if self.exec_arm_waypoints is not None else self.arm_waypoints
        if not arm_waypoints:
            return False
        theta, phi = arm_waypoints[1] if len(arm_waypoints) > 1 else arm_waypoints[0]
        self.status_pub.publish(String(data="COMPACT_ARM theta=%.2f phi=%.2f" % (theta, phi)))
        for _ in range(int(max(self.rate_hz * 0.5, 2))):
            if rospy.is_shutdown() or self.command == "STOP":
                return False
            self._publish_arm(theta, phi)
            self._publish_active_setpoint()
            rospy.sleep(1.0 / self.rate_hz)
        return True

    def _run_path(self):
        rate = rospy.Rate(self.rate_hz)
        poses = self.exec_path if self.exec_path is not None else self.path.poses
        for idx, raw_waypoint in enumerate(poses):
            if rospy.is_shutdown() or self.command == "STOP":
                return False
            waypoint = self._copy_waypoint(raw_waypoint, self.min_flight_z)
            if idx == len(poses) - 1:
                waypoint = self._copy_waypoint(raw_waypoint, None)
            self.active_setpoint = waypoint
            started = rospy.Time.now()
            while not rospy.is_shutdown():
                if self.command == "STOP":
                    return False
                self._publish_active_setpoint()
                xy, dz = self._dist_to(waypoint)
                self.status_pub.publish(
                    String(data="FLY_WAYPOINT %d/%d xy=%.3f z=%.3f mode=%s armed=%s" % (
                        idx + 1,
                        len(poses),
                        xy,
                        dz,
                        self.state.mode,
                        self.state.armed,
                    ))
                )
                if xy <= self.acceptance_radius and dz <= self.acceptance_z:
                    break
                if (rospy.Time.now() - started).to_sec() > self.max_step_wait_sec:
                    self.last_result = "FAILED_WAYPOINT_TIMEOUT_%d" % (idx + 1)
                    self.status_pub.publish(String(data=self.last_result))
                    return False
                rate.sleep()
        return True

    def _run_reach_in(self):
        rate = rospy.Rate(max(4.0, self.rate_hz * 0.25))
        arm_waypoints = self.exec_arm_waypoints if self.exec_arm_waypoints is not None else self.arm_waypoints
        for idx, (theta, phi) in enumerate(arm_waypoints):
            if rospy.is_shutdown() or self.command == "STOP":
                return False
            self.status_pub.publish(String(data="REACH_IN %d/%d theta=%.2f phi=%.2f" % (idx + 1, len(arm_waypoints), theta, phi)))
            for _ in range(4):
                self._publish_arm(theta, phi)
                self._publish_active_setpoint()
                rate.sleep()
        return True

    def _execute_once(self):
        if not self._ready():
            self.status_pub.publish(String(data="WAITING_FOR_PLAN_OR_MAVROS"))
            return False
        if self.require_plan_ready and not self._plan_ready():
            self.last_result = "BLOCKED_PLAN_NOT_READY"
            self.status_pub.publish(String(data="%s status=%s" % (self.last_result, self.plan_status)))
            return False
        self.running = True
        self.last_result = "RUNNING"
        self.exec_path = [self._copy_waypoint(p, None) for p in self.path.poses]
        self.exec_arm_waypoints = list(self.arm_waypoints)
        if self.pose is not None and self.exec_path:
            takeoff = self._copy_waypoint(self.exec_path[0], self.min_flight_z)
            takeoff.pose.position.x = self.pose.pose.position.x
            takeoff.pose.position.y = self.pose.pose.position.y
            self.exec_path = [takeoff] + self.exec_path
        self.active_setpoint = self._copy_waypoint(self.exec_path[0], self.min_flight_z)
        if not self._prime_setpoints(self.active_setpoint):
            self.running = False
            self.exec_path = None
            self.exec_arm_waypoints = None
            return False
        if not self._ensure_mode_and_arm():
            self.running = False
            self.last_result = "WAIT_OFFBOARD_ARMED"
            self.exec_path = None
            self.exec_arm_waypoints = None
            return False
        if not self._run_compact_arm():
            self.running = False
            self.last_result = "STOPPED_AT_COMPACT_ARM"
            self.exec_path = None
            self.exec_arm_waypoints = None
            return False
        if not self._run_path():
            self.running = False
            if self.last_result == "RUNNING":
                self.last_result = "STOPPED_AT_PATH"
            self.exec_path = None
            self.exec_arm_waypoints = None
            return False
        self.status_pub.publish(String(data="HOLD_STANDOFF"))
        hold_until = rospy.Time.now() + rospy.Duration(self.goal_hold_sec)
        rate = rospy.Rate(self.rate_hz)
        while rospy.Time.now() < hold_until and not rospy.is_shutdown():
            self._publish_active_setpoint()
            rate.sleep()
        if not self._run_reach_in():
            self.running = False
            self.last_result = "STOPPED_AT_REACH_IN"
            self.exec_path = None
            self.exec_arm_waypoints = None
            return False
        self.last_result = "DONE VERIFY_TIP"
        self.status_pub.publish(String(data=self.last_result))
        self.running = False
        self.exec_path = None
        self.exec_arm_waypoints = None
        return True

    def spin(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            if self.command == "RESET":
                self.command = "IDLE"
                self.last_result = "NONE"
                self.running = False
            elif self.command == "START" and not self.running:
                self._execute_once()
                self.command = "IDLE"
            elif not self.running:
                if self._ready():
                    state = "READY_IDLE last=%s plan_status=%s" % (self.last_result, self.plan_status)
                else:
                    state = "WAITING_FOR_PLAN_OR_MAVROS"
                self.status_pub.publish(String(data=state))
            self._publish_active_setpoint()
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("px4_waypoint_executor_node")
    try:
        Px4WaypointExecutorNode().spin()
    except rospy.ROSInterruptException:
        pass
