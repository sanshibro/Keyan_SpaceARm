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


def triangle_fai(t, fai_min, fai_max, period):
    if period <= 0.0:
        return fai_min
    phase = (t % period) / period
    if phase < 0.5:
        ratio = phase * 2.0
    else:
        ratio = 2.0 - phase * 2.0
    return fai_min + ratio * (fai_max - fai_min)


def sine_fai(t, fai_min, fai_max, period):
    if period <= 0.0:
        return fai_min
    center = 0.5 * (fai_min + fai_max)
    amplitude = 0.5 * (fai_max - fai_min)
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
        description="Continuously publish Qingyu A01 theta/fai commands."
    )
    parser.add_argument("--theta", type=float, default=30.0)
    parser.add_argument("--theta-min", type=float, default=20.0)
    parser.add_argument("--theta-max", type=float, default=70.0)
    parser.add_argument("--fai", type=float, default=0.0)
    parser.add_argument("--fai-min", type=float, default=-90.0)
    parser.add_argument("--fai-max", type=float, default=90.0)
    parser.add_argument(
        "--sweep-axis",
        choices=("fai", "theta"),
        default="fai",
        help="Sweep fai with fixed theta, or sweep theta with fixed fai.",
    )
    parser.add_argument(
        "--period",
        type=float,
        default=4.0,
        help="Seconds for one full -90 -> +90 -> -90 cycle.",
    )
    parser.add_argument(
        "--profile",
        choices=("sine", "triangle"),
        default="sine",
        help="sine gives smooth turnarounds; triangle gives constant-speed ramps.",
    )
    parser.add_argument(
        "--start-seconds",
        type=float,
        default=4.0,
        help="Smooth transition time from the pre-start pose to the sweep start.",
    )
    parser.add_argument("--hz", type=float, default=100.0)
    parser.add_argument(
        "--wait-enter",
        action="store_true",
        help="Hold theta/pre-fai until Enter is pressed, then start sweeping.",
    )
    parser.add_argument(
        "--pre-fai",
        type=float,
        default=0.0,
        help="Initial fai command while waiting for Enter.",
    )
    parser.add_argument(
        "--pre-theta",
        type=float,
        default=None,
        help="Initial theta command while waiting for Enter. Defaults to --theta.",
    )
    parser.add_argument(
        "--cycles",
        type=float,
        default=0.0,
        help="Number of cycles to run. 0 means run until Ctrl+C.",
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

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    rospy.init_node("qingyu_fai_smooth_sweep", anonymous=True)
    pub = rospy.Publisher(args.topic, Float32MultiArray, queue_size=10)
    rate = rospy.Rate(args.hz)

    pre_theta = args.theta if args.pre_theta is None else args.pre_theta

    if args.sweep_axis == "theta":
        rospy.loginfo(
            "Publishing smooth theta sweep: theta=[%.2f, %.2f] fai=%.2f period=%.2fs hz=%.1f topic=%s",
            args.theta_min,
            args.theta_max,
            args.fai,
            args.period,
            args.hz,
            args.topic,
        )
    else:
        rospy.loginfo(
            "Publishing smooth fai sweep: theta=%.2f fai=[%.2f, %.2f] period=%.2fs hz=%.1f topic=%s",
            args.theta,
            args.fai_min,
            args.fai_max,
            args.period,
            args.hz,
            args.topic,
        )

    if args.wait_enter:
        rospy.loginfo(
            "Holding pre-start pose: theta=%.2f fai=%.2f. Press Enter to start sweep.",
            pre_theta,
            args.pre_fai,
        )
        threading.Thread(target=wait_for_enter, daemon=True).start()
        hold_msg = Float32MultiArray()
        hold_msg.data = [float(pre_theta), float(args.pre_fai)]
        while running and not rospy.is_shutdown() and not start_event.is_set():
            pub.publish(hold_msg)
            rospy.loginfo_throttle(
                0.5,
                "holding theta=%.2f fai=%.2f, press Enter to start",
                pre_theta,
                args.pre_fai,
            )
            rate.sleep()

        if running and not rospy.is_shutdown():
            start_theta = args.theta_min if args.sweep_axis == "theta" else args.theta
            start_fai = args.fai if args.sweep_axis == "theta" else args.fai_min
            smooth_move(
                pub, rate, pre_theta, args.pre_fai,
                start_theta, start_fai, args.start_seconds, args.hz,
            )

    wave = sine_fai if args.profile == "sine" else triangle_fai
    start = time.monotonic()
    duration = math.inf if args.cycles <= 0.0 else args.cycles * args.period
    theta = pre_theta
    fai = args.pre_fai

    while running and not rospy.is_shutdown():
        elapsed = time.monotonic() - start
        if elapsed >= duration:
            break
        if args.sweep_axis == "theta":
            theta = wave(elapsed, args.theta_min, args.theta_max, args.period)
            fai = args.fai
        else:
            theta = args.theta
            fai = wave(elapsed, args.fai_min, args.fai_max, args.period)
        publish_pose(pub, theta, fai)
        rospy.loginfo_throttle(0.5, "theta=%.2f fai=%.2f", theta, fai)
        rate.sleep()

    if args.zero_on_exit and not rospy.is_shutdown():
        smooth_move(pub, rate, theta, fai, 0.0, 0.0, args.zero_seconds, args.hz)

    return 0


if __name__ == "__main__":
    sys.exit(main())
