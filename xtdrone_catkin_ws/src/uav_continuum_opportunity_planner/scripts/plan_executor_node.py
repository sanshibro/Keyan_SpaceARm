#!/usr/bin/env python3
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import DeleteModel, SetModelState, SpawnModel
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Float32MultiArray, String
from visualization_msgs.msg import Marker


class PlanExecutorNode:
    def __init__(self):
        self.model_name = rospy.get_param("~model_name", "uav_plan_preview")
        self.reference_frame = rospy.get_param("~reference_frame", "world")
        self.execute_gazebo = bool(rospy.get_param("~execute_gazebo", False))
        self.allow_real_model = bool(rospy.get_param("~allow_real_model", False))
        self.execute_arm = bool(rospy.get_param("~execute_arm", True))
        self.publish_continuum_control = bool(rospy.get_param("~publish_continuum_control", True))
        self.auto_start = bool(rospy.get_param("~auto_start", False))
        self.path_rate_hz = float(rospy.get_param("~path_rate_hz", 8.0))
        self.arm_rate_hz = float(rospy.get_param("~arm_rate_hz", 4.0))
        self.goal_hold_sec = float(rospy.get_param("~goal_hold_sec", 1.0))
        self.loop = bool(rospy.get_param("~loop", False))

        self.path = None
        self.arm_waypoints = None
        self.task_plan = ""
        self.command = "START" if self.auto_start else "IDLE"
        self.running = False
        self.last_result = "NONE"
        self.set_model_state = None
        self.spawn_model = None
        self.delete_model = None
        self.preview_spawned = False

        rospy.Subscriber("/planner/uav_waypoints", Path, self._path_cb, queue_size=1)
        rospy.Subscriber("/planner/arm_waypoints", Float32MultiArray, self._arm_cb, queue_size=1)
        rospy.Subscriber("/planner/task_plan", String, self._task_plan_cb, queue_size=1)
        rospy.Subscriber("/planner/executor_cmd", String, self._cmd_cb, queue_size=10)

        self.status_pub = rospy.Publisher("/planner/executor_status", String, queue_size=10)
        self.pose_pub = rospy.Publisher("/planner/executor_pose_preview", PoseStamped, queue_size=10)
        self.arm_cmd_pub = rospy.Publisher("/arm/theta_phi_cmd", Float32MultiArray, queue_size=10)
        self.continuum_arm_pub = rospy.Publisher("/continuum_arm/control_angles", Float32MultiArray, queue_size=10)
        self.stage_pub = rospy.Publisher("/planner/executor_stage_marker", Marker, queue_size=10)

        rospy.loginfo(
            "plan_executor_node: execute_gazebo=%s model=%s execute_arm=%s auto_start=%s",
            self.execute_gazebo,
            self.model_name,
            self.execute_arm,
            self.auto_start,
        )

    def _path_cb(self, msg):
        self.path = msg

    def _arm_cb(self, msg):
        data = list(msg.data)
        self.arm_waypoints = [(data[i], data[i + 1]) for i in range(0, len(data) - 1, 2)]

    def _task_plan_cb(self, msg):
        self.task_plan = msg.data

    def _cmd_cb(self, msg):
        cmd = msg.data.strip().upper()
        if cmd in ["START", "STOP", "RESET"]:
            self.command = cmd

    def _ready(self):
        return self.path is not None and self.path.poses and self.arm_waypoints is not None and self.arm_waypoints

    def _connect_gazebo(self):
        if self.set_model_state is not None:
            return True
        try:
            rospy.wait_for_service("/gazebo/set_model_state", timeout=0.5)
            rospy.wait_for_service("/gazebo/spawn_sdf_model", timeout=0.5)
            rospy.wait_for_service("/gazebo/delete_model", timeout=0.5)
            self.set_model_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
            self.spawn_model = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
            self.delete_model = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
            return True
        except rospy.ROSException:
            return False

    def _preview_sdf(self):
        return """<sdf version='1.6'>
  <model name='uav_plan_preview'>
    <static>true</static>
    <link name='body'>
      <visual name='body_visual'>
        <geometry><box><size>0.42 0.42 0.08</size></box></geometry>
        <material>
          <ambient>0.0 0.8 1.0 0.55</ambient>
          <diffuse>0.0 0.8 1.0 0.55</diffuse>
        </material>
      </visual>
      <visual name='front_visual'>
        <pose>0.28 0 0 0 1.5708 0</pose>
        <geometry><cylinder><radius>0.035</radius><length>0.18</length></cylinder></geometry>
        <material>
          <ambient>1.0 1.0 0.0 0.8</ambient>
          <diffuse>1.0 1.0 0.0 0.8</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""

    def _ensure_preview_model(self, pose):
        if self.preview_spawned:
            return True
        if not self._connect_gazebo():
            return False
        try:
            self.delete_model(self.model_name)
        except Exception:
            pass
        try:
            self.spawn_model(self.model_name, self._preview_sdf(), "", pose, self.reference_frame)
            self.preview_spawned = True
            return True
        except rospy.ServiceException as exc:
            rospy.logwarn_throttle(1.0, "plan_executor_node: spawn preview model failed: %s", exc)
            return False

    def _set_gazebo_pose(self, pose):
        if not self.execute_gazebo:
            return True
        if self.model_name in ["iris_0", "iris"] and not self.allow_real_model:
            self.status_pub.publish(
                String(data="REFUSED_REAL_MODEL_EXECUTION model=%s set allow_real_model:=true only if intentional" % self.model_name)
            )
            return False
        if not self._connect_gazebo():
            return False
        if not self.allow_real_model and not self._ensure_preview_model(pose):
            return False
        state = ModelState()
        state.model_name = self.model_name
        state.pose = pose
        state.reference_frame = self.reference_frame
        try:
            return bool(self.set_model_state(state).success)
        except rospy.ServiceException as exc:
            rospy.logwarn_throttle(1.0, "plan_executor_node: set_model_state failed: %s", exc)
            self.set_model_state = None
            return False

    def _publish_pose_preview(self, pose_stamped):
        out = PoseStamped()
        out.header = pose_stamped.header
        out.header.stamp = rospy.Time.now()
        out.pose = pose_stamped.pose
        self.pose_pub.publish(out)

    def _publish_arm(self, theta, phi):
        if not self.execute_arm:
            return
        msg = Float32MultiArray()
        msg.data = [float(theta), float(phi)]
        self.arm_cmd_pub.publish(msg)
        if self.publish_continuum_control:
            self.continuum_arm_pub.publish(msg)

    def _publish_stage_marker(self, stage):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "plan_executor_stage"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = 0.0
        marker.pose.position.y = 5.2
        marker.pose.position.z = 1.8
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.16
        marker.color.r = 0.4
        marker.color.g = 1.0
        marker.color.b = 0.4
        marker.color.a = 1.0
        marker.text = "EXECUTOR: %s" % stage
        self.stage_pub.publish(marker)

    def _sleep_rate(self, hz):
        rospy.sleep(1.0 / max(hz, 1e-3))

    def _run_arm_to(self, target_idx, stage):
        if self.arm_waypoints is None:
            return False
        target_idx = max(0, min(target_idx, len(self.arm_waypoints) - 1))
        theta, phi = self.arm_waypoints[target_idx]
        self.status_pub.publish(String(data="%s theta=%.2f phi=%.2f" % (stage, theta, phi)))
        self._publish_stage_marker(stage)
        for _ in range(max(2, int(self.arm_rate_hz * 0.5))):
            if rospy.is_shutdown() or self.command == "STOP":
                return False
            self._publish_arm(theta, phi)
            self._sleep_rate(self.arm_rate_hz)
        return True

    def _run_arm_sequence(self):
        for idx, (theta, phi) in enumerate(self.arm_waypoints):
            if rospy.is_shutdown() or self.command == "STOP":
                return False
            stage = "REACH_IN %d/%d theta=%.2f phi=%.2f" % (idx + 1, len(self.arm_waypoints), theta, phi)
            self.status_pub.publish(String(data=stage))
            self._publish_stage_marker(stage)
            self._publish_arm(theta, phi)
            self._sleep_rate(self.arm_rate_hz)
        return True

    def _run_uav_path(self):
        n = len(self.path.poses)
        for idx, pose_stamped in enumerate(self.path.poses):
            if rospy.is_shutdown() or self.command == "STOP":
                return False
            ok = self._set_gazebo_pose(pose_stamped.pose)
            self._publish_pose_preview(pose_stamped)
            stage = "COMPACT_TRANSIT %d/%d" % (idx + 1, n)
            if self.execute_gazebo and not ok:
                stage += " GAZEBO_SET_FAILED"
            self.status_pub.publish(String(data=stage))
            self._publish_stage_marker(stage)
            self._sleep_rate(self.path_rate_hz)
        return True

    def _execute_once(self):
        if not self._ready():
            self.status_pub.publish(String(data="WAITING_FOR_PLAN"))
            return False

        self.running = True
        self.status_pub.publish(String(data="START_PLAN_EXECUTION execute_gazebo=%s" % self.execute_gazebo))

        if not self._run_arm_to(1 if len(self.arm_waypoints) > 1 else 0, "COMPACT_ARM"):
            self.status_pub.publish(String(data="STOPPED_AT_COMPACT_ARM"))
            self.running = False
            self.last_result = "STOPPED_AT_COMPACT_ARM"
            return False

        if not self._run_uav_path():
            self.status_pub.publish(String(data="STOPPED_AT_COMPACT_TRANSIT"))
            self.running = False
            self.last_result = "STOPPED_AT_COMPACT_TRANSIT"
            return False

        self.status_pub.publish(String(data="HOLD_STANDOFF"))
        self._publish_stage_marker("HOLD_STANDOFF")
        rospy.sleep(self.goal_hold_sec)

        if not self._run_arm_sequence():
            self.status_pub.publish(String(data="STOPPED_AT_REACH_IN"))
            self.running = False
            self.last_result = "STOPPED_AT_REACH_IN"
            return False

        self.status_pub.publish(String(data="DONE VERIFY_TIP"))
        self._publish_stage_marker("DONE VERIFY_TIP")
        self.running = False
        self.last_result = "DONE VERIFY_TIP"
        return True

    def spin(self):
        rate = rospy.Rate(2.0)
        while not rospy.is_shutdown():
            if self.command == "RESET":
                self.command = "IDLE"
                self.running = False
                self.last_result = "NONE"
                self.status_pub.publish(String(data="RESET_TO_IDLE"))
            elif self.command == "START" and not self.running:
                done = self._execute_once()
                if done and self.loop:
                    self.command = "START"
                else:
                    self.command = "IDLE"
            elif not self.running:
                state = "READY_IDLE last=%s" % self.last_result if self._ready() else "WAITING_FOR_PLAN"
                self.status_pub.publish(String(data=state))
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("plan_executor_node")
    try:
        PlanExecutorNode().spin()
    except rospy.ROSInterruptException:
        pass
