#!/usr/bin/env python3
"""Compare PX4 position/feed-forward interfaces on one repeatable open-space path."""

import argparse
import json
import math
from pathlib import Path
import statistics
import time

import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import PositionTarget, State
from mavros_msgs.srv import SetMode


MODES = ("position", "xy_feedforward", "xyz_feedforward")


def percentile(values, fraction):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return float("nan")
    index = max(0, min(len(ordered) - 1, int(math.ceil(fraction * len(ordered))) - 1))
    return ordered[index]


def metrics(values):
    values = [abs(float(value)) for value in values]
    return {
        "rmse": math.sqrt(sum(value * value for value in values) / len(values)),
        "p95": percentile(values, 0.95),
        "max": max(values),
        "mean_abs": statistics.fmean(values),
    }


class TrackingCalibration:
    def __init__(self, output, duration, settle, modes):
        self.output = Path(output)
        self.duration = max(12.0, float(duration))
        self.settle = max(3.0, float(settle))
        self.modes = tuple(modes)
        self.pose = None
        self.state = None
        self.samples = {mode: [] for mode in self.modes}
        self.pose_pub = rospy.Publisher(
            "/iris_0/mavros/setpoint_position/local", PoseStamped, queue_size=20
        )
        self.raw_pub = rospy.Publisher(
            "/iris_0/mavros/setpoint_raw/local", PositionTarget, queue_size=20
        )
        rospy.Subscriber(
            "/iris_0/mavros/local_position/pose", PoseStamped, self._pose_cb, queue_size=20
        )
        rospy.Subscriber("/iris_0/mavros/state", State, self._state_cb, queue_size=10)
        self.set_mode = rospy.ServiceProxy("/iris_0/mavros/set_mode", SetMode)

    def _pose_cb(self, message):
        self.pose = message

    def _state_cb(self, message):
        self.state = message

    @staticmethod
    def _profile(base, elapsed, duration):
        # Starts and ends at base with continuous position and velocity.  The
        # amplitudes stay inside the obstacle-free spawn region while testing
        # horizontal tracking and the same 12 cm vertical variation used by
        # the narrow-passage trajectory.
        omega = 2.0 * math.pi / duration
        x = base[0] + 0.16 * math.sin(omega * elapsed)
        y = base[1] - 0.20 * (1.0 - math.cos(omega * elapsed))
        z = base[2] + 0.12 * math.sin(2.0 * omega * elapsed)
        vx = 0.16 * omega * math.cos(omega * elapsed)
        vy = -0.20 * omega * math.sin(omega * elapsed)
        vz = 0.24 * omega * math.cos(2.0 * omega * elapsed)
        return (x, y, z), (vx, vy, vz)

    def _publish(self, mode, position, velocity):
        if mode == "position":
            message = PoseStamped()
            message.header.stamp = rospy.Time.now()
            message.header.frame_id = self.pose.header.frame_id or "map"
            message.pose.position.x, message.pose.position.y, message.pose.position.z = position
            message.pose.orientation.w = 1.0
            self.pose_pub.publish(message)
            return
        target = PositionTarget()
        target.header.stamp = rospy.Time.now()
        target.header.frame_id = self.pose.header.frame_id or "map"
        target.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        target.type_mask = (
            PositionTarget.IGNORE_AFX
            | PositionTarget.IGNORE_AFY
            | PositionTarget.IGNORE_AFZ
            | PositionTarget.IGNORE_YAW_RATE
        )
        if mode == "xy_feedforward":
            target.type_mask |= PositionTarget.IGNORE_VZ
        target.position.x, target.position.y, target.position.z = position
        target.velocity.x, target.velocity.y, target.velocity.z = velocity
        target.yaw = 0.0
        self.raw_pub.publish(target)

    def _ready(self):
        return (
            self.pose is not None and self.state is not None and self.state.connected
            and self.state.armed and self.state.mode == "OFFBOARD"
        )

    def _hold(self, base, seconds):
        deadline = time.monotonic() + seconds
        rate = rospy.Rate(40)
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            self._publish("position", base, (0.0, 0.0, 0.0))
            rate.sleep()

    def run(self):
        deadline = time.monotonic() + 30.0
        while not rospy.is_shutdown() and not self._ready() and time.monotonic() < deadline:
            rospy.sleep(0.1)
        if not self._ready():
            raise RuntimeError("PX4 must be connected, armed and in OFFBOARD before calibration")
        base = (
            float(self.pose.pose.position.x), float(self.pose.pose.position.y),
            float(self.pose.pose.position.z),
        )
        if base[2] < 0.45:
            raise RuntimeError("calibration requires a stable airborne start")
        rate = rospy.Rate(40)
        for mode in self.modes:
            rospy.loginfo("PX4 tracking calibration mode=%s settling", mode)
            self._hold(base, self.settle)
            start = time.monotonic()
            while not rospy.is_shutdown():
                elapsed = time.monotonic() - start
                if elapsed >= self.duration:
                    break
                if not self._ready():
                    raise RuntimeError("PX4 readiness lost during {}".format(mode))
                desired, velocity = self._profile(base, elapsed, self.duration)
                self._publish(mode, desired, velocity)
                actual = self.pose.pose.position
                errors = (
                    actual.x - desired[0], actual.y - desired[1], actual.z - desired[2]
                )
                if math.sqrt(sum(value * value for value in errors)) > 0.45:
                    raise RuntimeError("tracking runaway during {}: {}".format(mode, errors))
                self.samples[mode].append({
                    "t": elapsed, "desired": list(desired),
                    "actual": [actual.x, actual.y, actual.z], "error": list(errors),
                })
                rate.sleep()
            self._hold(base, self.settle)
        report = {"base": list(base), "duration_per_mode": self.duration, "modes": {}}
        for mode, samples in self.samples.items():
            axis = list(zip(*(sample["error"] for sample in samples)))
            norms = [math.sqrt(sum(value * value for value in sample["error"])) for sample in samples]
            report["modes"][mode] = {
                "samples": len(samples), "x": metrics(axis[0]), "y": metrics(axis[1]),
                "z": metrics(axis[2]), "position": metrics(norms),
            }
        # Passage safety is dominated by vertical error; use z P95 first and
        # total-position P95 second as a deterministic tie breaker.
        report["selected_mode"] = min(
            self.modes,
            key=lambda mode: (
                report["modes"][mode]["z"]["p95"],
                report["modes"][mode]["position"]["p95"],
            ),
        )
        self.output.parent.mkdir(parents=True, exist_ok=True)
        trace_path = self.output.with_suffix(".trace.jsonl")
        with trace_path.open("w") as stream:
            for mode, samples in self.samples.items():
                for sample in samples:
                    sample = dict(sample)
                    sample["mode"] = mode
                    stream.write(json.dumps(sample, sort_keys=True) + "\n")
        report["trace"] = str(trace_path)
        self.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
        self._hold(base, 2.0)
        try:
            self.set_mode(base_mode=0, custom_mode="AUTO.LAND")
        except rospy.ServiceException as exc:
            rospy.logerr("AUTO.LAND request failed: %s", exc)
        return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration", type=float, default=24.0)
    parser.add_argument("--settle", type=float, default=5.0)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    arguments, _ = parser.parse_known_args()
    rospy.init_node("calibrate_px4_tracking")
    TrackingCalibration(
        arguments.output, arguments.duration, arguments.settle, arguments.modes
    ).run()


if __name__ == "__main__":
    main()
