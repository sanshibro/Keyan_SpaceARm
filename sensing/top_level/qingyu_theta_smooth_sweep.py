#!/usr/bin/env python3
import argparse
import math
import signal
import sys
import threading
import time

import rospy
from std_msgs.msg import Float32MultiArray


running = True
start_event = threading.Event()


def handle_signal(_signum, _frame):
    global running
    running = False
    start_event.set()


def wait_for_enter():
    try:
        input()
    except EOFError:
        pass
    start_event.set()


def triangle_wave(t, value_min, value_max, period):
    if period <= 0.0:
        return value_min
    phase = (t % period) / period
    if phase < 0.5:
        ratio = 2.0 * phase
    else:
        ratio = 2.0 - 2.0 * phase
    return value_min + ratio * (value_max - value_min)


def sine_wave(t, value_min, value_max, period):
    if period <= 0.0:
        return value_min
    center = 0.5 * (value_min + value_max)
    amplitude = 0.5 * (value_max - value_min)
    # Starts at value_min, reaches value_max at period/2, then returns.
    return center - amplitude * math.cos(2.0 * math.pi * (t % period) / period)


def publish_pose(pub, theta, fai):
    msg = Float32MultiArray()
    msg.data = [float(theta), float(fai)]
    pub.publish(msg)


def smooth_move(pub, rate, start_theta, start_fai, end_theta, end_fai, seconds, hz):
    steps = max(int(max(seconds, 0.0) * hz), 1)
    for step in range(steps + 1):
        ratio = step / steps
        blend = 0.5 - 0.5 * math.cos(math.pi * ratio)
        theta = start_theta + blend * (end_theta - start_theta)
        fai = start_fai + blend * (end_fai - start_fai)
        publish_pose(pub, theta, fai)
        rate.sleep()


def main():
    parser = argparse.ArgumentParser(
        description="Sweep Qingyu A01 theta between two limits while holding fai fixed."
    )
    parser.add_argument("--theta-min", type=float, default=0.0)
    parser.add_argument("--theta-max", type=float, default=70.0)
    parser.add_argument("--fai", type=float, default=0.0)
    parser.add_argument(
        "--period",
        type=float,
        default=8.0,
        help="Seconds for one full theta-min -> theta-max -> theta-min cycle.",
    )
    parser.add_argument("--hz", type=float, default=100.0)
    parser.add_argument(
        "--profile",
        choices=("sine", "triangle"),
        default="sine",
        help="sine is smoother; triangle gives constant-speed ramps with sharp turnarounds.",
    )
    parser.add_argument(
        "--start-seconds",
        type=float,
        default=4.0,
        help="Smooth transition time from the pre-start pose to theta-min.",
    )
    parser.add_argument(
        "--cycles",
        type=float,
        default=0.0,
        help="Number of cycles to run. 0 means run until Ctrl+C.",
    )
    parser.add_argument(
        "--wait-enter",
        action="store_true",
        help="Hold pre-start pose until Enter is pressed.",
    )
    parser.add_argument(
        "--pre-theta",
        type=float,
        default=0.0,
        help="Theta while waiting for Enter.",
    )
    parser.add_argument(
        "--pre-fai",
        type=float,
        default=None,
        help="Fai while waiting for Enter. Defaults to --fai.",
    )
    parser.add_argument(
        "--topic",
        default="/continuum_arm/control_angles",
        help="Angle command topic used by qingyu_api_bridge.",
    )
    parser.add_argument(
        "--zero-on-exit",
        action="store_true",
        help="Move smoothly to theta=0 fai=0 before exit.",
    )
    parser.add_argument("--zero-seconds", type=float, default=5.0)
    args = parser.parse_args()

    if args.theta_max < args.theta_min:
        print("--theta-max must be >= --theta-min", file=sys.stderr)
        return 2
    if args.hz <= 0.0:
        print("--hz must be > 0", file=sys.stderr)
        return 2

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    rospy.init_node("qingyu_theta_smooth_sweep", anonymous=True)
    pub = rospy.Publisher(args.topic, Float32MultiArray, queue_size=10)
    rate = rospy.Rate(args.hz)
    pre_fai = args.fai if args.pre_fai is None else args.pre_fai

    rospy.loginfo(
        "Publishing theta sweep: theta=[%.2f, %.2f] fai=%.2f period=%.2fs profile=%s hz=%.1f topic=%s",
        args.theta_min,
        args.theta_max,
        args.fai,
        args.period,
        args.profile,
        args.hz,
        args.topic,
    )

    if args.wait_enter:
        rospy.loginfo(
            "Holding pre-start pose: theta=%.2f fai=%.2f. Press Enter to start sweep.",
            args.pre_theta,
            pre_fai,
        )
        threading.Thread(target=wait_for_enter, daemon=True).start()
        while running and not rospy.is_shutdown() and not start_event.is_set():
            publish_pose(pub, args.pre_theta, pre_fai)
            rospy.loginfo_throttle(
                0.5,
                "holding theta=%.2f fai=%.2f, press Enter to start",
                args.pre_theta,
                pre_fai,
            )
            rate.sleep()

        if running and not rospy.is_shutdown():
            smooth_move(
                pub, rate, args.pre_theta, pre_fai,
                args.theta_min, args.fai, args.start_seconds, args.hz,
            )

    wave = sine_wave if args.profile == "sine" else triangle_wave
    start = time.monotonic()
    duration = math.inf if args.cycles <= 0.0 else args.cycles * args.period
    theta = args.pre_theta if args.wait_enter else args.theta_min
    fai = pre_fai if args.wait_enter else args.fai

    while running and not rospy.is_shutdown():
        elapsed = time.monotonic() - start
        if elapsed >= duration:
            break
        theta = wave(elapsed, args.theta_min, args.theta_max, args.period)
        publish_pose(pub, theta, args.fai)
        rospy.loginfo_throttle(0.5, "theta=%.2f fai=%.2f", theta, args.fai)
        rate.sleep()

    if args.zero_on_exit and not rospy.is_shutdown():
        smooth_move(pub, rate, theta, fai, 0.0, 0.0, args.zero_seconds, args.hz)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
