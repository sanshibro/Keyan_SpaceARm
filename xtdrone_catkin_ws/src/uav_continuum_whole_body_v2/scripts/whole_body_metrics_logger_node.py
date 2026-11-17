#!/usr/bin/env python3
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

PACKAGE = Path(__file__).resolve().parents[1]
REPOSITORY = PACKAGE.parents[2]
sys.path.insert(0, str(PACKAGE / "src"))

import numpy as np
import rospy
import yaml
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import ActuatorControl, AttitudeTarget
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray, String

from uav_continuum_whole_body_v2.msg import WholeBodyTrajectory
from uav_continuum_whole_body_v2.execution import yaw_from_quaternion
from uav_continuum_whole_body_v2.mapping import MapSnapshot


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class WholeBodyMetricsLoggerNode:
    def __init__(self):
        root = Path(rospy.get_param(
            "~output_dir", str(REPOSITORY / "xtdrone_plan" / "experiments" / "v2_runs")
        ))
        requested_name = str(rospy.get_param("~run_name", "")).strip()
        if requested_name:
            run_name = requested_name
        else:
            run_name = "run_%s_%d" % (time.strftime("%Y%m%d_%H%M%S"), os.getpid())
        self.output = root / run_name
        suffix = 1
        while self.output.exists():
            self.output = root / (run_name + "_%02d" % suffix)
            suffix += 1
        run_name = self.output.name
        self.output.mkdir(parents=True, exist_ok=False)
        self.events = self.output / "events.jsonl"
        self.tracking = self.output / "tracking.jsonl"
        self.commands = self.output / "commands.jsonl"
        self.command_kinematics = self.output / "command_kinematics.jsonl"
        self.flight_dynamics = self.output / "flight_dynamics.jsonl"
        self.pose = None
        self.arm = None
        self.arm_command = None
        self.last_tracking_write = rospy.Time(0)
        self.last_imu_write = rospy.Time(0)
        config = Path(rospy.get_param(
            "~planner_config", str(PACKAGE / "config" / "planner_v2.yaml")
        ))
        scenario = Path(rospy.get_param(
            "~task_file", str(PACKAGE / "config" / "scenario_3c_unified.yaml")
        ))
        baseline_registry = PACKAGE / "config" / "baseline_registry.yaml"
        with scenario.open("r", encoding="utf-8") as handle:
            task_data = yaml.safe_load(handle) or {}
        map_mode = str(rospy.get_param("~map_mode", "regression_exact"))
        map_snapshot_file = str(rospy.get_param("~map_snapshot_file", ""))
        oracle_file = str(rospy.get_param("~oracle_file", "")).strip()
        map_hash = None
        map_source = None
        map_metadata = None
        if map_mode == "frozen_slam" and Path(map_snapshot_file).is_file():
            snapshot = MapSnapshot.load(map_snapshot_file)
            map_hash = snapshot.map_hash
            map_source = snapshot.source
            map_metadata = snapshot.metadata
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(REPOSITORY), text=True
            ).strip()
            tracked_dirty = subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=normal", "--",
                 "xtdrone_catkin_ws/src/uav_continuum_whole_body_v2",
                 "xtdrone_catkin_ws/src/xtdrone_ego_tools",
                 "xtdrone_catkin_ws/src/uav_continuum_opportunity_planner/scripts/arm_shape_sim_node.py",
                 "xtdrone_catkin_ws/src/uav_continuum_opportunity_planner/scripts/arm_kinematics_node.py"],
                cwd=str(REPOSITORY), text=True,
            ).splitlines()
        except (OSError, subprocess.CalledProcessError):
            commit, tracked_dirty = "unknown", ["git_state_unavailable"]
        manifest = {
            "run_name": run_name, "created": rospy.Time.now().to_sec(), "created_wall_time": time.time(),
            "planner_config": str(config), "planner_config_sha256": sha256(config),
            "scenario": str(scenario), "scenario_sha256": sha256(scenario),
            "scenario_name": str(task_data.get("name", scenario.stem)),
            "scenario_family": task_data.get("family"),
            "scenario_difficulty": task_data.get("difficulty"),
            "tip_tolerance_m": float(task_data.get("tip_tolerance_m", 0.08)),
            "random_seed": task_data.get("seed"),
            "baseline_registry": str(baseline_registry),
            "baseline_registry_sha256": sha256(baseline_registry),
            "method": str(rospy.get_param("~method", "rolling_wb_v2")),
            "map_mode": map_mode, "map_snapshot_file": map_snapshot_file,
            "map_hash": map_hash, "map_source": map_source,
            "map_metadata": map_metadata,
            "oracle_file": oracle_file or None,
            "oracle_sha256": (
                sha256(Path(oracle_file)) if oracle_file and Path(oracle_file).is_file()
                else None
            ),
            "commit": commit, "tracked_dirty": tracked_dirty,
            "reproducibility_scope_dirty": tracked_dirty,
            "experiment_class": str(rospy.get_param(
                "~experiment_class", "engineering_regression"
            )),
        }
        (self.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        rospy.Subscriber("/planner/v2/whole_body_trajectory", WholeBodyTrajectory, self._trajectory_cb, queue_size=1)
        rospy.Subscriber("/planner/v2/status", String, self._event_cb, queue_size=20)
        rospy.Subscriber("/planner/v2/executor_status", String, self._event_cb, queue_size=20)
        rospy.Subscriber("/planner/v2/execution_feedback", String, self._event_cb, queue_size=20)
        rospy.Subscriber("/planner/v2/commanded_pose_world", PoseStamped, self._command_cb, queue_size=20)
        rospy.Subscriber(
            "/planner/v2/commanded_twist_world", TwistStamped,
            self._command_twist_cb, queue_size=20,
        )
        rospy.Subscriber(rospy.get_param("~pose_topic", "/iris_0/world_pose"), PoseStamped, self._pose_cb, queue_size=10)
        rospy.Subscriber(rospy.get_param("~arm_state_topic", "/arm/current_theta_phi"), Float32MultiArray, self._arm_cb, queue_size=10)
        rospy.Subscriber(rospy.get_param("~arm_cmd_topic", "/arm/theta_phi_cmd"), Float32MultiArray, self._arm_command_cb, queue_size=10)
        rospy.Subscriber(
            rospy.get_param("~imu_topic", "/iris_0/mavros/imu/data"),
            Imu, self._imu_cb, queue_size=50,
        )
        rospy.Subscriber(
            rospy.get_param(
                "~attitude_target_topic",
                "/iris_0/mavros/setpoint_raw/target_attitude",
            ),
            AttitudeTarget, self._attitude_target_cb, queue_size=50,
        )
        rospy.Subscriber(
            rospy.get_param("~actuator_topic", "/iris_0/mavros/actuator_control"),
            ActuatorControl, self._actuator_cb, queue_size=50,
        )
        rospy.Subscriber(
            rospy.get_param(
                "~target_actuator_topic",
                "/iris_0/mavros/target_actuator_control",
            ),
            ActuatorControl, self._target_actuator_cb, queue_size=50,
        )
        rospy.loginfo("whole_body_metrics_logger_node: output=%s", self.output)

    @staticmethod
    def _append(path, payload):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")

    def _event_cb(self, message):
        try:
            payload = json.loads(message.data)
        except ValueError:
            payload = {"raw": message.data}
        payload["logger_stamp"] = rospy.Time.now().to_sec()
        self._append(self.events, payload)

    def _trajectory_cb(self, message):
        identifier = int(message.trajectory_id)
        states = np.asarray([
            [pose.position.x, pose.position.y, pose.position.z,
             yaw_from_quaternion(pose.orientation),
             message.theta_deg[index], message.phi_deg[index]]
            for index, pose in enumerate(message.uav_poses)
        ], dtype=float)
        velocities = np.asarray([
            [value.x, value.y, value.z, message.yaw_rates[index],
             message.theta_rate_deg_s[index], message.phi_rate_deg_s[index]]
            for index, value in enumerate(message.linear_velocities)
        ], dtype=float)
        np.savez_compressed(
            str(self.output / ("trajectory_%06d.npz" % identifier)),
            time=np.asarray(message.time_from_start), states=states, velocities=velocities,
        )
        metadata = {
            "trajectory_id": identifier, "method": message.method, "validator_status": message.validator_status,
            "planning_state_stamp": message.planning_state_stamp.to_sec(),
            "execution_start_time": message.execution_start_time.to_sec(), "duration": message.duration,
            "samples": len(message.time_from_start),
            "state_columns": ["x_m", "y_m", "z_m", "yaw_rad", "theta_deg", "phi_deg"],
            "velocity_columns": [
                "vx_m_s", "vy_m_s", "vz_m_s", "yaw_rate_rad_s",
                "theta_rate_deg_s", "phi_rate_deg_s",
            ],
            "trajectory_log_schema_version": 2,
        }
        (self.output / ("trajectory_%06d.json" % identifier)).write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _pose_cb(self, message):
        self.pose = message
        now = rospy.Time.now()
        if (now - self.last_tracking_write).to_sec() < 0.05:
            return
        p = message.pose.position
        q = message.pose.orientation
        self._append(self.tracking, {
            "stamp": now.to_sec(), "source_stamp": message.header.stamp.to_sec(),
            "position": [p.x, p.y, p.z], "orientation": [q.x, q.y, q.z, q.w], "arm_deg": self.arm,
        })
        self.last_tracking_write = now

    def _command_cb(self, message):
        p = message.pose.position
        q = message.pose.orientation
        self._append(self.commands, {
            "stamp": rospy.Time.now().to_sec(), "source_stamp": message.header.stamp.to_sec(),
            "position": [p.x, p.y, p.z], "orientation": [q.x, q.y, q.z, q.w],
            "arm_deg": self.arm_command,
        })

    def _command_twist_cb(self, message):
        linear = message.twist.linear
        angular = message.twist.angular
        self._append(self.command_kinematics, {
            "stamp": rospy.Time.now().to_sec(),
            "source_stamp": message.header.stamp.to_sec(),
            "linear_velocity": [linear.x, linear.y, linear.z],
            "yaw_rate": angular.z,
        })

    def _arm_cb(self, message):
        if len(message.data) >= 2:
            self.arm = [float(message.data[0]), float(message.data[1])]

    def _arm_command_cb(self, message):
        if len(message.data) >= 2:
            self.arm_command = [float(message.data[0]), float(message.data[1])]

    @staticmethod
    def _roll_pitch(quaternion):
        x, y, z, w = (
            quaternion.x, quaternion.y, quaternion.z, quaternion.w
        )
        roll = math.atan2(
            2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)
        )
        pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
        return roll, pitch

    def _imu_cb(self, message):
        now = rospy.Time.now()
        if (now - self.last_imu_write).to_sec() < 0.02:
            return
        roll, pitch = self._roll_pitch(message.orientation)
        self._append(self.flight_dynamics, {
            "kind": "imu", "stamp": now.to_sec(),
            "source_stamp": message.header.stamp.to_sec(),
            "roll_rad": roll, "pitch_rad": pitch,
            "angular_velocity_rad_s": [
                message.angular_velocity.x, message.angular_velocity.y,
                message.angular_velocity.z,
            ],
            "linear_acceleration_m_s2": [
                message.linear_acceleration.x, message.linear_acceleration.y,
                message.linear_acceleration.z,
            ],
        })
        self.last_imu_write = now

    def _attitude_target_cb(self, message):
        roll, pitch = self._roll_pitch(message.orientation)
        self._append(self.flight_dynamics, {
            "kind": "attitude_target", "stamp": rospy.Time.now().to_sec(),
            "source_stamp": message.header.stamp.to_sec(),
            "type_mask": int(message.type_mask),
            "roll_rad": roll, "pitch_rad": pitch,
            "body_rate_rad_s": [
                message.body_rate.x, message.body_rate.y, message.body_rate.z,
            ],
            "normalized_thrust": float(message.thrust),
        })

    def _actuator_cb(self, message):
        self._write_actuator(message, "actuator_control")

    def _target_actuator_cb(self, message):
        self._write_actuator(message, "target_actuator_control")

    def _write_actuator(self, message, kind):
        controls = [float(value) for value in message.controls]
        self._append(self.flight_dynamics, {
            "kind": kind, "stamp": rospy.Time.now().to_sec(),
            "source_stamp": message.header.stamp.to_sec(),
            "group_mix": int(message.group_mix), "controls": controls,
            "maximum_absolute_control": max((abs(value) for value in controls), default=0.0),
        })


if __name__ == "__main__":
    rospy.init_node("whole_body_metrics_logger_node")
    WholeBodyMetricsLoggerNode()
    rospy.spin()
