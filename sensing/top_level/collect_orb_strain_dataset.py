#!/usr/bin/env python3
import argparse
import csv
import os
import re
import sys
import threading
import time
from collections import deque
from datetime import datetime

import matplotlib
import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray, Int32


def parse_frame(line):
    text = line.strip()
    if not text:
        return None
    if text.endswith(";"):
        text = text[:-1]
    parts = [p.strip() for p in text.split(",")]
    if len(parts) < 6:
        return None
    values = []
    for part in parts:
        if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", part):
            return None
        values.append(float(part))
    return values


def read_frame(ser, terminator):
    if terminator == "semicolon":
        return ser.read_until(b";")
    if terminator == "newline":
        return ser.readline()
    raise ValueError(f"unsupported terminator: {terminator}")


def read_latest_frame(ser, terminator):
    if terminator != "semicolon":
        return read_frame(ser, terminator)
    first = ser.read_until(b";")
    if not first:
        return first
    data = first
    waiting = getattr(ser, "in_waiting", 0)
    if waiting:
        data += ser.read(waiting)
    parts = data.split(b";")
    complete = [part.strip() for part in parts[:-1] if part.strip()]
    if complete:
        return complete[-1] + b";"
    return first


def make_run_dir(base_dir):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, stamp)
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


def open_serial(port, baudrate, timeout):
    import serial

    return serial.Serial(port=port, baudrate=baudrate, timeout=timeout)


class LatestOrbAngle:
    def __init__(self):
        self.lock = threading.Lock()
        self.theta = None
        self.phi = None
        self.valid = 0
        self.tracking_state = -1
        self.wall_time = None

    def angle_cb(self, msg):
        if len(msg.data) < 2:
            return
        now = time.time()
        with self.lock:
            self.theta = float(msg.data[0])
            self.phi = float(msg.data[1])
            self.wall_time = now

    def valid_cb(self, msg):
        with self.lock:
            self.valid = int(msg.data)

    def state_cb(self, msg):
        with self.lock:
            self.tracking_state = int(msg.data)

    def snapshot(self, now):
        with self.lock:
            if self.wall_time is None:
                return None, None, None, self.valid, self.tracking_state
            return (
                self.theta,
                self.phi,
                now - self.wall_time,
                self.valid,
                self.tracking_state,
            )


def configure_live_plot(enable):
    if not enable:
        matplotlib.use("Agg", force=True)
        return None
    if not os.environ.get("DISPLAY"):
        print("Live plot disabled: DISPLAY is not set.", file=sys.stderr)
        matplotlib.use("Agg", force=True)
        return None
    try:
        matplotlib.use("TkAgg", force=True)
        import matplotlib.pyplot as plt

        plt.ion()
        fig, (ax_strain, ax_theta) = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
        strain_line, = ax_strain.plot([], [], "tab:blue", label="strain ch")
        theta_line, = ax_theta.plot([], [], "tab:orange", label="ORB theta")
        ax_strain.set_ylabel("strain value")
        ax_theta.set_ylabel("theta (deg)")
        ax_theta.set_xlabel("time (s)")
        ax_strain.grid(True, alpha=0.3)
        ax_theta.grid(True, alpha=0.3)
        ax_strain.legend(loc="upper left")
        ax_theta.legend(loc="upper left")
        fig.tight_layout()
        return plt, fig, ax_strain, ax_theta, strain_line, theta_line
    except Exception as exc:
        print(f"Live plot disabled: {exc}", file=sys.stderr)
        matplotlib.use("Agg", force=True)
        return None


def update_live_plot(live, t, strain, theta, channel):
    if live is None or not t:
        return live
    plt, fig, ax_strain, ax_theta, strain_line, theta_line = live
    try:
        if not plt.fignum_exists(fig.number):
            return None
        strain_line.set_data(t, strain)
        theta_line.set_data(t, theta)
        ax_strain.set_ylabel(f"strain ch{channel}")
        for ax in (ax_strain, ax_theta):
            ax.relim()
            ax.autoscale_view()
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.001)
    except Exception as exc:
        print(f"\nLive plot disabled: {exc}", file=sys.stderr)
        return None
    return live


def moving_average(values, window):
    if window <= 1 or len(values) < window:
        return None
    kernel = [1.0 / window] * window
    return np.convolve(np.asarray(values, dtype=float), np.asarray(kernel), mode="same")


def save_plot(path, t, strain, theta, channel, smooth_window):
    if not t:
        return
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, (ax_strain, ax_theta) = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
    ax_strain.plot(t, strain, "tab:blue", linewidth=0.9, alpha=0.55, label="raw")
    strain_smooth = moving_average(strain, smooth_window)
    if strain_smooth is not None:
        ax_strain.plot(t, strain_smooth, "tab:blue", linewidth=1.6, label=f"moving avg {smooth_window}")
        ax_strain.legend(loc="upper left")
    ax_theta.plot(t, theta, "tab:orange", linewidth=1.2)
    ax_strain.set_ylabel(f"strain ch{channel}")
    ax_theta.set_ylabel("ORB theta (deg)")
    ax_theta.set_xlabel("time (s)")
    ax_strain.grid(True, alpha=0.3)
    ax_theta.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def collect(args):
    rospy.init_node("orb_strain_dataset_collector", anonymous=True)
    latest = LatestOrbAngle()
    rospy.Subscriber(args.angle_topic, Float32MultiArray, latest.angle_cb, queue_size=100)
    rospy.Subscriber(args.valid_topic, Int32, latest.valid_cb, queue_size=20)
    rospy.Subscriber(args.state_topic, Int32, latest.state_cb, queue_size=20)

    ser = open_serial(args.port, args.baudrate, args.timeout)
    run_dir = make_run_dir(args.output_dir)
    csv_path = os.path.join(run_dir, "orb_strain_sync.csv")
    plot_path = os.path.join(run_dir, "orb_strain_sync.png")
    meta_path = os.path.join(run_dir, "meta.txt")
    print(f"Saving synchronized CSV to: {csv_path}")
    print(f"Saving plot to: {plot_path}")

    with open(meta_path, "w", encoding="utf-8") as meta:
        for key, value in sorted(vars(args).items()):
            meta.write(f"{key}={value}\n")
        meta.write(f"start_time={datetime.now().isoformat(timespec='seconds')}\n")

    start_wall = time.time()
    rows = 0
    bad_lines = 0
    valid_pairs = 0
    last_idle_report = start_wall
    last_rx_wall = start_wall
    last_plot_wall = 0.0
    t_plot = deque(maxlen=args.window)
    strain_plot = deque(maxlen=args.window)
    theta_plot = deque(maxlen=args.window)
    t_all = []
    strain_all = []
    theta_all = []
    live = configure_live_plot(not args.no_live)

    header = [
        "wall_time",
        "rel_time",
        "ch1",
        "ch2",
        "ch3",
        "ch4",
        "ch5",
        "ch6",
        "strain_channel",
        "strain_value",
        "orb_theta_deg",
        "orb_phi_deg",
        "theta_age",
        "angle_valid",
        "tracking_state",
        "usable",
        "raw_line",
    ]

    interrupted = False
    with ser:
        ser.reset_input_buffer()
        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(header)
            csv_file.flush()
            print("Collecting strain + ORB theta. Press Ctrl+C to stop.")
            try:
                while not rospy.is_shutdown():
                    now_loop = time.time()
                    if args.duration is not None and now_loop - start_wall >= args.duration:
                        break
                    if args.samples is not None and rows >= args.samples:
                        break

                    raw = read_latest_frame(ser, args.terminator) if args.latest_only else read_frame(ser, args.terminator)
                    if not raw:
                        now = time.time()
                        if now - last_idle_report >= args.idle_report_every:
                            print(
                                f"\nNo strain serial bytes received for {now - last_rx_wall:.1f}s.",
                                file=sys.stderr,
                            )
                            last_idle_report = now
                        continue
                    last_rx_wall = time.time()
                    line = raw.decode(args.encoding, errors="replace").strip()
                    values = parse_frame(line)
                    if values is None:
                        bad_lines += 1
                        if args.print_bad:
                            print(f"Bad strain line: {line}", file=sys.stderr)
                        continue

                    now = time.time()
                    rel = now - start_wall
                    channels = values[:6]
                    strain_value = channels[args.channel - 1]
                    theta, phi, theta_age, angle_valid, tracking_state = latest.snapshot(now)
                    has_fresh_theta = theta is not None and theta_age is not None and theta_age <= args.max_theta_age
                    usable = bool(has_fresh_theta and (not args.require_valid or angle_valid == 1))
                    if usable:
                        valid_pairs += 1

                    writer.writerow(
                        [
                            f"{now:.6f}",
                            f"{rel:.6f}",
                            *[f"{v:.6f}" for v in channels],
                            args.channel,
                            f"{strain_value:.6f}",
                            "" if theta is None else f"{theta:.6f}",
                            "" if phi is None else f"{phi:.6f}",
                            "" if theta_age is None else f"{theta_age:.6f}",
                            angle_valid,
                            tracking_state,
                            int(usable),
                            line,
                        ]
                    )
                    rows += 1
                    if rows % args.flush_every == 0:
                        csv_file.flush()

                    t_plot.append(rel)
                    strain_plot.append(strain_value)
                    theta_plot.append(float("nan") if theta is None else theta)
                    t_all.append(rel)
                    strain_all.append(strain_value)
                    theta_all.append(float("nan") if theta is None else theta)

                    plot_period = 1.0 / args.plot_hz if args.plot_hz > 0 else 0.0
                    if rows % args.plot_every == 0 and now - last_plot_wall >= plot_period:
                        live = update_live_plot(
                            live,
                            list(t_plot),
                            list(strain_plot),
                            list(theta_plot),
                            args.channel,
                        )
                        last_plot_wall = now

                    if rows % args.status_every == 0:
                        hz = rows / rel if rel > 0 else 0.0
                        age_text = "none" if theta_age is None else f"{theta_age:.3f}s"
                        print(
                            f"\rrows={rows} usable={valid_pairs} bad={bad_lines} "
                            f"hz={hz:.1f} ch{args.channel}={strain_value:.3f} "
                            f"theta={theta if theta is not None else 'none'} age={age_text}",
                            end="",
                            flush=True,
                        )
            except KeyboardInterrupt:
                interrupted = True
                print("\nStopping and saving captured data...", file=sys.stderr)
            finally:
                csv_file.flush()

    save_plot(plot_path, t_all, strain_all, theta_all, args.channel, args.smooth_window)
    print(f"\nDone. rows={rows}, usable={valid_pairs}, bad_lines={bad_lines}, run_dir={run_dir}")
    return 130 if interrupted else (0 if valid_pairs else 2)


def main():
    parser = argparse.ArgumentParser(description="Synchronize Elastreme strain data with ORB theta.")
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--channel", type=int, default=1, choices=range(1, 7))
    parser.add_argument("--angle-topic", default="/continuum_angles")
    parser.add_argument("--valid-topic", default="/continuum_angles_valid")
    parser.add_argument("--state-topic", default="/orb_slam3/tracking_state")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--output-dir", default="orb_strain_runs")
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument("--encoding", default="ascii")
    parser.add_argument("--terminator", choices=["semicolon", "newline"], default="semicolon")
    parser.add_argument(
        "--latest-only",
        action="store_true",
        default=True,
        help="drop queued serial frames and process only the newest complete frame",
    )
    parser.add_argument(
        "--keep-all-frames",
        dest="latest_only",
        action="store_false",
        help="process every serial frame in the OS buffer; not recommended for ORB synchronization",
    )
    parser.add_argument("--max-theta-age", type=float, default=0.2)
    parser.add_argument("--require-valid", action="store_true", help="only mark rows usable when /continuum_angles_valid == 1")
    parser.add_argument("--no-live", action="store_true")
    parser.add_argument("--window", type=int, default=1200)
    parser.add_argument("--smooth-window", type=int, default=11, help="moving-average window used only in the saved plot")
    parser.add_argument("--plot-every", type=int, default=1)
    parser.add_argument("--plot-hz", type=float, default=10.0, help="maximum live plot refresh rate")
    parser.add_argument("--flush-every", type=int, default=10)
    parser.add_argument("--status-every", type=int, default=10)
    parser.add_argument("--idle-report-every", type=float, default=3.0)
    parser.add_argument("--print-bad", action="store_true")
    parser.add_argument("--fast", action="store_true", help="disable live plot and reduce terminal/file flush overhead")
    args = parser.parse_args()
    if args.fast:
        args.no_live = True
        args.flush_every = max(args.flush_every, 100)
        args.status_every = max(args.status_every, 100)
        args.plot_hz = 0.0

    try:
        return collect(args)
    except Exception as exc:
        if exc.__class__.__name__ == "SerialException":
            print(f"Could not open serial port: {exc}", file=sys.stderr)
            return 1
        raise


if __name__ == "__main__":
    raise SystemExit(main())
