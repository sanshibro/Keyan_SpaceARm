#!/usr/bin/env python3
import argparse
import math
import threading
import time

import rospy
from std_msgs.msg import Float32MultiArray


class QingyuInteractiveControl:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Interactive Qingyu A01 angle control.")
        parser.add_argument("--topic", default="/continuum_arm/control_angles")
        parser.add_argument("--hz", type=float, default=100.0)
        parser.add_argument("--theta", type=float, default=0.0)
        parser.add_argument("--fai", type=float, default=0.0)
        parser.add_argument(
            "--transition-seconds",
            type=float,
            default=5.0,
            help="Seconds used for each smooth move to a new target.",
        )
        args = parser.parse_args()

        rospy.init_node("qingyu_interactive_control", anonymous=True)
        self.pub = rospy.Publisher(args.topic, Float32MultiArray, queue_size=10)
        self.hz = max(args.hz, 1.0)
        self.current_theta = float(args.theta)
        self.current_fai = float(args.fai)
        self.start_theta = self.current_theta
        self.start_fai = self.current_fai
        self.target_theta = self.current_theta
        self.target_fai = self.current_fai
        self.transition_seconds = max(float(args.transition_seconds), 0.0)
        self.move_started = time.monotonic()
        self.running = True
        self.lock = threading.Lock()

        self.thread = threading.Thread(target=self.publish_loop, daemon=True)
        self.thread.start()

        print("")
        print("Qingyu interactive control started.")
        print("Input format:")
        print("  theta fai       example: 30 0")
        print("  fai             example: -90   (theta keeps current value)")
        print("  z               zero: theta=0 fai=0")
        print("  q               quit")
        print("")
        print(f"Transition time: {self.transition_seconds:.2f}s")
        print(f"Current target: theta={self.target_theta:.2f}, fai={self.target_fai:.2f}")

    def update_current_locked(self, now):
        if self.transition_seconds <= 0.0:
            ratio = 1.0
        else:
            ratio = min(max((now - self.move_started) / self.transition_seconds, 0.0), 1.0)
        blend = 0.5 - 0.5 * math.cos(math.pi * ratio)
        self.current_theta = self.start_theta + blend * (self.target_theta - self.start_theta)
        self.current_fai = self.start_fai + blend * (self.target_fai - self.start_fai)

    def publish_loop(self):
        rate = rospy.Rate(self.hz)
        while self.running and not rospy.is_shutdown():
            with self.lock:
                self.update_current_locked(time.monotonic())
                theta = self.current_theta
                fai = self.current_fai
            msg = Float32MultiArray()
            msg.data = [theta, fai]
            self.pub.publish(msg)
            rate.sleep()

    def set_target(self, theta, fai):
        with self.lock:
            now = time.monotonic()
            self.update_current_locked(now)
            self.start_theta = self.current_theta
            self.start_fai = self.current_fai
            self.target_theta = float(theta)
            self.target_fai = float(fai)
            self.move_started = now
        print(
            f"Moving smoothly to theta={self.target_theta:.2f}, "
            f"fai={self.target_fai:.2f} in {self.transition_seconds:.2f}s"
        )

    def run(self):
        while self.running and not rospy.is_shutdown():
            try:
                text = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("")
                break

            if not text:
                continue
            if text.lower() in ("q", "quit", "exit"):
                break
            if text.lower() in ("z", "zero", "0"):
                self.set_target(0.0, 0.0)
                continue

            parts = text.replace(",", " ").split()
            try:
                if len(parts) == 1:
                    fai = float(parts[0])
                    with self.lock:
                        self.update_current_locked(time.monotonic())
                        theta = self.target_theta
                    self.set_target(theta, fai)
                elif len(parts) == 2:
                    theta = float(parts[0])
                    fai = float(parts[1])
                    self.set_target(theta, fai)
                else:
                    print("Invalid input. Use: theta fai, or fai, or z, or q")
            except ValueError:
                print("Invalid number. Use: theta fai, or fai, or z, or q")

        self.running = False
        time.sleep(0.1)


if __name__ == "__main__":
    QingyuInteractiveControl().run()
