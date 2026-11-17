#!/usr/bin/env python3
"""Fail-closed live gate for the common SLAM method-comparison interface."""

import argparse
import json
import time

import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from nav_msgs.msg import Path
from std_msgs.msg import String

from uav_continuum_whole_body_v2.msg import WholeBodyTrajectory


class Gate:
    def __init__(self):
        self.mavros = None
        self.pose = None
        self.reference = None
        self.reference_status = {}
        self.planner_status = {}
        self.accepted = None
        self.trajectory = None
        self.visualization = {}
        rospy.Subscriber("/iris_0/mavros/state", State, self._set("mavros"), queue_size=1)
        rospy.Subscriber("/iris_0/world_pose", PoseStamped, self._set("pose"), queue_size=1)
        rospy.Subscriber("/planner/experiment/reference", Path, self._set("reference"), queue_size=1)
        rospy.Subscriber("/planner/experiment/reference_status", String,
                         self._json("reference_status"), queue_size=1)
        rospy.Subscriber("/planner/v2/status", String,
                         self._json("planner_status"), queue_size=20)
        rospy.Subscriber("/planner/v2/accepted_status", String,
                         self._accepted_cb, queue_size=1)
        rospy.Subscriber("/planner/v2/whole_body_trajectory", WholeBodyTrajectory,
                         self._set("trajectory"), queue_size=1)
        rospy.Subscriber("/planner/v2/visualization_status", String,
                         self._json("visualization"), queue_size=5)

    def _set(self, name):
        def callback(message):
            setattr(self, name, message)
        return callback

    def _json(self, name):
        def callback(message):
            try:
                setattr(self, name, json.loads(message.data))
            except ValueError:
                setattr(self, name, {"state": "INVALID_JSON"})
        return callback

    def _accepted_cb(self, message):
        try:
            value = json.loads(message.data)
        except ValueError:
            return
        if value.get("state") == "SAFE_TRAJECTORY":
            self.accepted = value

    def evaluate(self):
        failures = []
        if self.mavros is None or not self.mavros.connected:
            failures.append("mavros_not_connected")
        elif not self.mavros.armed or self.mavros.mode != "OFFBOARD":
            failures.append("mavros_not_armed_offboard")
        if self.pose is None:
            failures.append("missing_world_pose")
        if self.reference is None or len(self.reference.poses) < 2:
            failures.append("missing_common_slam_reference")
        if self.reference_status.get("state") != "REFERENCE_READY":
            failures.append("common_reference_not_ready")
        if self.accepted is None:
            failures.append("no_safe_trajectory")
        elif self.accepted.get(
                "truth_geometry_access",
                self.accepted.get("diagnostics", {}).get("truth_geometry_access"),
        ) is not False:
            failures.append("planner_truth_access_not_false")
        if self.trajectory is None or self.trajectory.validator_status != "safe":
            failures.append("missing_validated_whole_body_trajectory")
        if self.visualization.get("state") != "READY":
            failures.append("visualization_not_ready")
        details = {
            "reference": self.reference_status,
            "planner": self.accepted,
            "trajectory_id": None if self.trajectory is None
            else int(self.trajectory.trajectory_id),
            "method": None if self.accepted is None else self.accepted.get(
                "method", None if self.trajectory is None else self.trajectory.method
            ),
            "map_hash": None if self.accepted is None else self.accepted.get(
                "map_hash", self.accepted.get("diagnostics", {}).get("map_hash")
            ),
        }
        return not failures, sorted(set(failures)), details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=240.0)
    args = parser.parse_args(rospy.myargv()[1:])
    rospy.init_node("method_comparison_live_gate", anonymous=True)
    gate = Gate()
    deadline = time.monotonic() + args.timeout
    last = 0.0
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        passed, failures, details = gate.evaluate()
        if passed:
            print(json.dumps({"passed": True, "failures": [], "details": details},
                             indent=2, sort_keys=True))
            return 0
        if gate.planner_status.get("state") in ("METHOD_FAILED", "METHOD_EXCEPTION"):
            failures.append("terminal_planner_failure")
            print(json.dumps({
                "passed": False, "failures": sorted(set(failures)),
                "details": details, "terminal_status": gate.planner_status,
            }, indent=2, sort_keys=True))
            return 2
        if time.monotonic() - last >= 10.0:
            print("comparison live gate waiting: " + ", ".join(failures))
            last = time.monotonic()
        rospy.sleep(0.2)
    passed, failures, details = gate.evaluate()
    print(json.dumps({"passed": passed, "failures": failures, "details": details},
                     indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
