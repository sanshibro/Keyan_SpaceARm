#!/usr/bin/env python3
"""Measure PX4 post-command stopping excursion in obstacle-free SITL space.

The whole-body executor can freeze trajectory phase when measured clearance
contracts.  A continuous tracking RMSE does not characterize what happens at
that instant, so this test approaches a fixed point at the planner's nominal
speed and then removes velocity feed-forward.  Positive directional excursion
beyond the frozen point is the quantity that consumes collision clearance.
"""

import argparse
import json
import math
from pathlib import Path
import time

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import ParamValue, PositionTarget, State
from mavros_msgs.srv import ParamSet, SetMode


PROFILES = {
    "current": {"MPC_Z_P": 1.5, "MPC_Z_VEL_D_ACC": 0.0,
                "MPC_XY_VEL_D_ACC": 0.2},
    "z_d_020": {"MPC_Z_P": 1.5, "MPC_Z_VEL_D_ACC": 0.2,
                "MPC_XY_VEL_D_ACC": 0.2},
    "z_d_040": {"MPC_Z_P": 1.5, "MPC_Z_VEL_D_ACC": 0.4,
                "MPC_XY_VEL_D_ACC": 0.2},
    "z_d_060": {"MPC_Z_P": 1.5, "MPC_Z_VEL_D_ACC": 0.6,
                "MPC_XY_VEL_D_ACC": 0.2},
    "z_d_040_xy_d_030": {"MPC_Z_P": 1.5, "MPC_Z_VEL_D_ACC": 0.4,
                         "MPC_XY_VEL_D_ACC": 0.3},
    "z_d_040_xy_d_040": {"MPC_Z_P": 1.5, "MPC_Z_VEL_D_ACC": 0.4,
                         "MPC_XY_VEL_D_ACC": 0.4},
    "xy_d_030": {"MPC_Z_P": 1.5, "MPC_Z_VEL_D_ACC": 0.0,
                 "MPC_XY_VEL_D_ACC": 0.3},
    "xy_d_040": {"MPC_Z_P": 1.5, "MPC_Z_VEL_D_ACC": 0.0,
                 "MPC_XY_VEL_D_ACC": 0.4},
    "xy_d_050": {"MPC_Z_P": 1.5, "MPC_Z_VEL_D_ACC": 0.0,
                 "MPC_XY_VEL_D_ACC": 0.5},
    "xy_p_220": {"MPC_Z_P": 1.5, "MPC_Z_VEL_D_ACC": 0.0,
                 "MPC_XY_VEL_P_ACC": 2.2, "MPC_XY_VEL_D_ACC": 0.2},
    "xy_p_260": {"MPC_Z_P": 1.5, "MPC_Z_VEL_D_ACC": 0.0,
                 "MPC_XY_VEL_P_ACC": 2.6, "MPC_XY_VEL_D_ACC": 0.2},
    "xy_p_300": {"MPC_Z_P": 1.5, "MPC_Z_VEL_D_ACC": 0.0,
                 "MPC_XY_VEL_P_ACC": 3.0, "MPC_XY_VEL_D_ACC": 0.2},
}


class StopCalibration:
    def __init__(self, arguments):
        self.arguments = arguments
        self.pose = None
        self.velocity = None
        self.state = None
        self.pose_pub = rospy.Publisher(
            "/iris_0/mavros/setpoint_position/local", PoseStamped, queue_size=20
        )
        self.raw_pub = rospy.Publisher(
            "/iris_0/mavros/setpoint_raw/local", PositionTarget, queue_size=20
        )
        rospy.Subscriber(
            "/iris_0/mavros/local_position/pose", PoseStamped, self._pose_cb, queue_size=30
        )
        rospy.Subscriber(
            "/iris_0/mavros/local_position/velocity_local", TwistStamped,
            self._velocity_cb, queue_size=30,
        )
        rospy.Subscriber("/iris_0/mavros/state", State, self._state_cb, queue_size=10)
        self.param_set = rospy.ServiceProxy("/iris_0/mavros/param/set", ParamSet)
        self.set_mode = rospy.ServiceProxy("/iris_0/mavros/set_mode", SetMode)
        self.trace = []

    def _pose_cb(self, message):
        self.pose = message

    def _velocity_cb(self, message):
        self.velocity = message

    def _state_cb(self, message):
        self.state = message

    def _ready(self):
        return (self.pose is not None and self.state is not None
                and self.state.connected and self.state.armed
                and self.state.mode == "OFFBOARD")

    def _set_profile(self, name):
        rospy.wait_for_service("/iris_0/mavros/param/set", timeout=10.0)
        # Explicitly set the unchanged gains as well so every trial starts
        # from a recorded, repeatable controller rather than saved EEPROM.
        values = {
            "MPC_XY_P": 0.95,
            "MPC_XY_VEL_P_ACC": 1.8,
            "MPC_XY_VEL_I_ACC": 0.4,
            "MPC_Z_VEL_P_ACC": 4.0,
            "MPC_Z_VEL_I_ACC": 2.0,
        }
        values.update(PROFILES[name])
        for param_id, real_value in values.items():
            response = self.param_set(
                param_id=param_id,
                value=ParamValue(integer=0, real=float(real_value)),
            )
            if not response.success:
                raise RuntimeError("failed to set {} for {}".format(param_id, name))
        rospy.loginfo("stopping calibration profile=%s values=%s", name, values)
        return values

    def _publish(self, position, velocity=(0.0, 0.0, 0.0)):
        target = PositionTarget()
        target.header.stamp = rospy.Time.now()
        target.header.frame_id = self.pose.header.frame_id or "map"
        target.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        target.type_mask = (PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY
                            | PositionTarget.IGNORE_AFZ
                            | PositionTarget.IGNORE_YAW_RATE)
        target.position.x, target.position.y, target.position.z = position
        target.velocity.x, target.velocity.y, target.velocity.z = velocity
        target.yaw = 0.0
        self.raw_pub.publish(target)

    def _hold(self, target, seconds):
        rate = rospy.Rate(40)
        deadline = time.monotonic() + seconds
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            self._publish(target)
            rate.sleep()

    def _measured_speed(self):
        if self.velocity is None:
            return float("inf")
        linear = self.velocity.twist.linear
        return math.sqrt(linear.x ** 2 + linear.y ** 2 + linear.z ** 2)

    def _wait_stable(self, target, timeout=12.0):
        """Require a settled initial condition, not merely a fixed delay."""
        rate = rospy.Rate(40)
        begin = time.monotonic()
        stable_since = None
        while not rospy.is_shutdown() and time.monotonic() - begin < timeout:
            self._publish(target)
            actual = self.pose.pose.position
            error = math.sqrt(
                (actual.x - target[0]) ** 2 + (actual.y - target[1]) ** 2
                + (actual.z - target[2]) ** 2
            )
            speed = self._measured_speed()
            if error <= 0.025 and speed <= 0.025:
                stable_since = stable_since or time.monotonic()
                if time.monotonic() - stable_since >= 0.75:
                    return {"position_error": error, "speed": speed,
                            "wait_time": time.monotonic() - begin}
            else:
                stable_since = None
            rate.sleep()
        raise RuntimeError(
            "controller did not settle at {} within {:.1f}s".format(target, timeout)
        )

    def _smooth_move(self, start, target, seconds):
        rate = rospy.Rate(40)
        begin = time.monotonic()
        while not rospy.is_shutdown():
            elapsed = time.monotonic() - begin
            if elapsed >= seconds:
                break
            u = max(0.0, min(1.0, elapsed / seconds))
            blend = u * u * (3.0 - 2.0 * u)
            blend_rate = 6.0 * u * (1.0 - u) / seconds
            position = tuple(start[i] + blend * (target[i] - start[i]) for i in range(3))
            velocity = tuple(blend_rate * (target[i] - start[i]) for i in range(3))
            self._publish(position, velocity)
            rate.sleep()
        self._wait_stable(target)

    @staticmethod
    def _axis_index(axis):
        return {"x": 0, "y": 1, "z": 2}[axis]

    def _trial(self, profile, center, axis, direction, repeat):
        index = self._axis_index(axis)
        distance = self.arguments.distance
        start = list(center)
        end = list(center)
        start[index] -= direction * distance / 2.0
        end[index] += direction * distance / 2.0
        current = self.pose.pose.position
        self._smooth_move((current.x, current.y, current.z), start,
                          self.arguments.reposition_time)

        move_time = distance / self.arguments.speed
        move_start = time.monotonic()
        velocity = [0.0, 0.0, 0.0]
        velocity[index] = direction * self.arguments.speed
        rate = rospy.Rate(40)
        while not rospy.is_shutdown():
            elapsed = time.monotonic() - move_start
            if elapsed >= move_time:
                break
            position = list(start)
            position[index] += velocity[index] * elapsed
            self._publish(position, velocity)
            rate.sleep()

        hold_start = time.monotonic()
        samples = []
        while not rospy.is_shutdown():
            elapsed = time.monotonic() - hold_start
            if elapsed >= self.arguments.hold_time:
                break
            if not self._ready():
                raise RuntimeError("PX4 readiness lost during stop trial")
            self._publish(end)
            actual = self.pose.pose.position
            actual_xyz = (actual.x, actual.y, actual.z)
            error = tuple(actual_xyz[i] - end[i] for i in range(3))
            speed = None
            if self.velocity is not None:
                linear = self.velocity.twist.linear
                speed = math.sqrt(linear.x ** 2 + linear.y ** 2 + linear.z ** 2)
            sample = {
                "profile": profile, "axis": axis, "direction": direction,
                "repeat": repeat,
                "t": elapsed, "target": list(end), "actual": list(actual_xyz),
                "error": list(error), "speed": speed,
            }
            samples.append(sample)
            self.trace.append(sample)
            rate.sleep()

        directional = [max(0.0, direction * sample["error"][index]) for sample in samples]
        absolute = [abs(sample["error"][index]) for sample in samples]
        position_error = [math.sqrt(sum(value * value for value in sample["error"]))
                          for sample in samples]
        tolerance = self.arguments.settle_tolerance
        last_outside = max((sample["t"] for sample in samples
                            if abs(sample["error"][index]) > tolerance), default=0.0)
        settled = abs(samples[-1]["error"][index]) <= tolerance
        result = {
            "axis": axis, "direction": direction, "repeat": repeat,
            "directional_overshoot": max(directional),
            "axis_error_max": max(absolute),
            "position_error_max": max(position_error),
            "final_axis_error": samples[-1]["error"][index],
            "settling_time": last_outside if settled else None,
            "peak_speed_after_freeze": max(
                sample["speed"] for sample in samples if sample["speed"] is not None
            ) if any(sample["speed"] is not None for sample in samples) else None,
            "samples": len(samples),
        }
        rospy.loginfo("stop result profile=%s axis=%s dir=%+d overshoot=%.4f "
                      "max_error=%.4f settle=%s", profile, axis, direction,
                      result["directional_overshoot"], result["axis_error_max"],
                      result["settling_time"])
        return result

    def run(self):
        deadline = time.monotonic() + 30.0
        while not rospy.is_shutdown() and not self._ready() and time.monotonic() < deadline:
            rospy.sleep(0.1)
        if not self._ready():
            raise RuntimeError("PX4 must be connected, armed and in OFFBOARD")
        pose = self.pose.pose.position
        center = (float(pose.x), float(pose.y), float(pose.z))
        if center[2] < 0.50:
            raise RuntimeError("calibration requires stable altitude >= 0.50 m")

        report = {"center": list(center), "speed": self.arguments.speed,
                  "distance": self.arguments.distance, "profiles": {}}
        for profile in self.arguments.profiles:
            parameters = self._set_profile(profile)
            current = self.pose.pose.position
            self._smooth_move((current.x, current.y, current.z), center,
                              self.arguments.reposition_time)
            self._hold(center, self.arguments.profile_settle)
            self._wait_stable(center)
            trials = []
            for repeat in range(self.arguments.repeats):
                for axis in self.arguments.axes:
                    for direction in (-1, 1):
                        trials.append(self._trial(
                            profile, center, axis, direction, repeat
                        ))
            worst_overshoot = max(item["directional_overshoot"] for item in trials)
            worst_error = max(item["axis_error_max"] for item in trials)
            unsettled = sum(item["settling_time"] is None for item in trials)
            report["profiles"][profile] = {
                "parameters": parameters, "trials": trials,
                "worst_directional_overshoot": worst_overshoot,
                "worst_axis_error": worst_error, "unsettled_trials": unsettled,
                "passes": (unsettled == 0
                           and worst_overshoot <= self.arguments.max_overshoot
                           and worst_error <= self.arguments.max_axis_error),
            }

        report["selected_profile"] = min(
            self.arguments.profiles,
            key=lambda name: (
                not report["profiles"][name]["passes"],
                report["profiles"][name]["unsettled_trials"],
                report["profiles"][name]["worst_axis_error"],
                report["profiles"][name]["worst_directional_overshoot"],
            ),
        )
        report["acceptance_max_overshoot"] = self.arguments.max_overshoot
        report["acceptance_max_axis_error"] = self.arguments.max_axis_error
        report["selected_passes"] = report["profiles"][report["selected_profile"]]["passes"]
        self._set_profile(report["selected_profile"])
        self._hold(center, 2.0)

        output = Path(self.arguments.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        trace_path = output.with_suffix(".trace.jsonl")
        with trace_path.open("w") as stream:
            for sample in self.trace:
                stream.write(json.dumps(sample, sort_keys=True) + "\n")
        report["trace"] = str(trace_path)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
        if self.arguments.land:
            try:
                self.set_mode(base_mode=0, custom_mode="AUTO.LAND")
            except rospy.ServiceException as exc:
                rospy.logerr("AUTO.LAND request failed: %s", exc)
        return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--profiles", nargs="+", choices=sorted(PROFILES),
                        default=list(PROFILES))
    parser.add_argument("--axes", nargs="+", choices=("x", "y", "z"), default=("y", "z"))
    parser.add_argument("--speed", type=float, default=0.08)
    parser.add_argument("--distance", type=float, default=0.12)
    parser.add_argument("--reposition-time", type=float, default=2.0)
    parser.add_argument("--hold-time", type=float, default=4.0)
    parser.add_argument("--profile-settle", type=float, default=3.0)
    parser.add_argument("--settle-tolerance", type=float, default=0.02)
    parser.add_argument("--max-overshoot", type=float, default=0.025)
    parser.add_argument("--max-axis-error", type=float, default=0.040)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--land", action="store_true")
    arguments, _ = parser.parse_known_args()
    if arguments.speed <= 0.0 or arguments.distance <= 0.0:
        parser.error("speed and distance must be positive")
    if arguments.repeats < 1:
        parser.error("repeats must be at least one")
    rospy.init_node("calibrate_px4_stopping")
    StopCalibration(arguments).run()


if __name__ == "__main__":
    main()
