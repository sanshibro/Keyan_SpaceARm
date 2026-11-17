#!/usr/bin/env python3
import argparse
import json
import math
import sys
import time

import rosnode
import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from nav_msgs.msg import Path
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String

from uav_continuum_whole_body_v2.msg import WholeBodyTrajectory


class LiveAcceptance:
    def __init__(self):
        self.mavros = None
        self.pose = None
        self.cloud = None
        self.global_path = None
        self.ego_path = None
        self.ego_report = ""
        self.last_ready_ego_report = ""
        self.status = {}
        self.safe_status = None
        self.trajectory = None
        self.visualization = {}
        rospy.Subscriber("/iris_0/mavros/state", State, self._set("mavros"), queue_size=1)
        rospy.Subscriber("/iris_0/world_pose", PoseStamped, self._set("pose"), queue_size=1)
        rospy.Subscriber("/world/static_scene_cloud", PointCloud2, self._set("cloud"), queue_size=1)
        rospy.Subscriber("/planner/global_path", Path, self._set("global_path"), queue_size=1)
        rospy.Subscriber("/planner/ego_reference_trajectory", Path, self._ego_path_cb, queue_size=1)
        rospy.Subscriber("/planner/ego_reference_report", String, self._ego_report_cb, queue_size=5)
        rospy.Subscriber("/planner/v2/status", String, self._status_cb, queue_size=20)
        rospy.Subscriber("/planner/v2/accepted_status", String, self._accepted_status_cb, queue_size=1)
        rospy.Subscriber("/planner/v2/whole_body_trajectory", WholeBodyTrajectory, self._set("trajectory"), queue_size=1)
        rospy.Subscriber("/planner/v2/visualization_status", String, self._visualization_cb, queue_size=5)

    def _set(self, name):
        def callback(message):
            setattr(self, name, message)
        return callback

    def _ego_path_cb(self, message):
        self.ego_path = message

    def _ego_report_cb(self, message):
        self.ego_report = message.data
        if "status=EGO_REFERENCE_READY" in message.data:
            self.last_ready_ego_report = message.data

    def _status_cb(self, message):
        try:
            self.status = json.loads(message.data)
        except ValueError:
            self.status = {"state": "INVALID_STATUS_JSON", "raw": message.data}
        if self.status.get("state") == "SAFE_TRAJECTORY":
            self.safe_status = dict(self.status)

    def _accepted_status_cb(self, message):
        try:
            accepted = json.loads(message.data)
        except ValueError:
            return
        if accepted.get("state") == "SAFE_TRAJECTORY":
            self.safe_status = accepted

    def _visualization_cb(self, message):
        try:
            self.visualization = json.loads(message.data)
        except ValueError:
            self.visualization = {"state": "INVALID_STATUS_JSON"}

    @staticmethod
    def _distance_pose(a, b):
        pa, pb = a.position, b.position
        return math.sqrt((pa.x - pb.x) ** 2 + (pa.y - pb.y) ** 2 + (pa.z - pb.z) ** 2)

    def evaluate(self):
        failures = []
        details = {}
        if self.mavros is None or not self.mavros.connected:
            failures.append("mavros_not_connected")
        else:
            details["mavros"] = {"armed": self.mavros.armed, "mode": self.mavros.mode}
            if not self.mavros.armed or self.mavros.mode != "OFFBOARD":
                failures.append("mavros_not_hovering_offboard")
        if self.pose is None:
            failures.append("missing_world_pose")
        else:
            p = self.pose.pose.position
            details["world_pose"] = [p.x, p.y, p.z]
            if not (0.75 <= p.z <= 2.20):
                failures.append("world_pose_altitude_out_of_bounds")
        if self.cloud is None:
            failures.append("missing_static_cloud")
        if self.global_path is None or len(self.global_path.poses) < 2:
            failures.append("missing_global_path")
        else:
            details["global_path_points"] = len(self.global_path.poses)
        if self.safe_status is None:
            failures.append("no_safe_trajectory_status")
            if not self.last_ready_ego_report:
                failures.append("no_fresh_ego_reference_report")
            if self.ego_path is None or len(self.ego_path.poses) < 2:
                failures.append("missing_ego_reference_path")
        else:
            metrics = self.safe_status.get("metrics", {})
            details["safe_status"] = self.safe_status
            reference_gap = self.safe_status.get("reference_start_gap", math.inf)
            if reference_gap > 0.75:
                failures.append("accepted_reference_discontinuous")
            body_margin = float(rospy.get_param("/planner_v2/limits/body_margin", 0.14))
            arm_margin = float(rospy.get_param("/planner_v2/limits/arm_margin", 0.06))
            velocity_max = float(rospy.get_param("/planner_v2/limits/velocity_max", 0.18))
            limits = (
                (metrics.get("body_clearance_min", -math.inf) >= body_margin, "body_clearance"),
                (metrics.get("arm_clearance_min", -math.inf) >= arm_margin, "arm_clearance"),
                (metrics.get("com_offset_max", math.inf) <= 0.08, "com_offset"),
                (metrics.get("gravity_torque_max", math.inf) <= 2.25 * 9.81 * 0.08, "gravity_torque"),
                (metrics.get("velocity_max", math.inf) <= velocity_max, "velocity"),
                (metrics.get("acceleration_max", math.inf) <= 1.00, "acceleration"),
                (metrics.get("jerk_max", math.inf) <= 3.00, "jerk"),
                (metrics.get("yaw_rate_max", math.inf) <= math.radians(45.0), "yaw_rate"),
                (metrics.get("theta_rate_max", math.inf) <= math.radians(50.0), "theta_rate"),
                (metrics.get("phi_rate_max", math.inf) <= math.radians(60.0), "phi_rate"),
                (metrics.get("tilt_max", math.inf) <= math.radians(15.0), "tilt"),
                (metrics.get("thrust_ratio_min", -math.inf) >= 0.50, "thrust_min"),
                (metrics.get("thrust_ratio_max", math.inf) <= 1.60, "thrust_max"),
            )
            failures.extend("validator_" + name for passed, name in limits if not passed)
        if self.trajectory is None:
            failures.append("missing_whole_body_trajectory")
        else:
            message = self.trajectory
            lengths = [
                len(message.time_from_start), len(message.uav_poses), len(message.linear_velocities),
                len(message.yaw_rates), len(message.theta_deg), len(message.phi_deg),
                len(message.theta_rate_deg_s), len(message.phi_rate_deg_s),
            ]
            details["trajectory"] = {
                "id": int(message.trajectory_id), "validator_status": message.validator_status,
                "duration": message.duration, "array_lengths": lengths,
            }
            if message.validator_status != "safe" or min(lengths) < 2 or len(set(lengths)) != 1:
                failures.append("invalid_whole_body_trajectory")
        if self.visualization.get("state") != "READY":
            failures.append("whole_body_visualization_not_ready")
        else:
            details["visualization"] = self.visualization
        try:
            nodes = rosnode.get_node_names()
        except Exception:
            nodes = []
        if "/whole_body_metrics_logger_v2" not in nodes:
            failures.append("metrics_logger_not_running")
        return not failures, sorted(set(failures)), details


def main():
    parser = argparse.ArgumentParser(description="Validate one live V2 planning-only integration run.")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args(rospy.myargv()[1:])
    rospy.init_node("whole_body_v2_live_acceptance", anonymous=True)
    acceptance = LiveAcceptance()
    deadline = time.monotonic() + max(1.0, args.timeout)
    last_print = 0.0
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        passed, failures, details = acceptance.evaluate()
        if passed:
            print(json.dumps({"passed": True, "failures": [], "details": details}, indent=2, sort_keys=True))
            return 0
        if time.monotonic() - last_print >= 10.0:
            print("V2 acceptance waiting:", ", ".join(failures))
            last_print = time.monotonic()
        rospy.sleep(0.2)
    passed, failures, details = acceptance.evaluate()
    print(json.dumps({"passed": passed, "failures": failures, "details": details}, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
