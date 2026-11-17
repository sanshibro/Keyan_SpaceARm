#!/usr/bin/env python3
import argparse
import csv
import json
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
    text = line.strip().rstrip(";")
    parts = [part.strip() for part in text.split(",")]
    if len(parts) < 6:
        return None
    if any(not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", part) for part in parts):
        return None
    return [float(part) for part in parts]


def read_latest_frame(ser):
    first = ser.read_until(b";")
    if not first:
        return first
    data = first
    if ser.in_waiting:
        data += ser.read(ser.in_waiting)
    complete = [part.strip() for part in data.split(b";")[:-1] if part.strip()]
    return complete[-1] + b";" if complete else first


def polynomial_features(x1, x2, degree):
    values = []
    for total in range(degree + 1):
        for p1 in range(total, -1, -1):
            p2 = total - p1
            values.append((x1 ** p1) * (x2 ** p2))
    return np.asarray(values, dtype=float)


class MovingAverage:
    def __init__(self, window):
        self.values = deque(maxlen=max(int(window), 1))

    def update(self, value):
        self.values.append(float(value))
        return float(np.mean(self.values))


class LatestOrb:
    def __init__(self):
        self.lock = threading.Lock()
        self.theta = None
        self.fai = None
        self.wall_time = None
        self.valid = 0
        self.state = -1

    def angle_cb(self, msg):
        if len(msg.data) < 2:
            return
        with self.lock:
            self.theta = float(msg.data[0])
            self.fai = float(msg.data[1])
            self.wall_time = time.time()

    def valid_cb(self, msg):
        with self.lock:
            self.valid = int(msg.data)

    def state_cb(self, msg):
        with self.lock:
            self.state = int(msg.data)

    def snapshot(self, now):
        with self.lock:
            age = None if self.wall_time is None else now - self.wall_time
            return self.theta, self.fai, age, self.valid, self.state


def configure_plot(enabled):
    if not enabled or not os.environ.get("DISPLAY"):
        matplotlib.use("Agg", force=True)
        return None
    try:
        matplotlib.use("TkAgg", force=True)
        import matplotlib.pyplot as plt

        plt.ion()
        fig, (ax_theta, ax_fai, ax_err) = plt.subplots(3, 1, sharex=True, figsize=(10, 9))
        lines = {
            "orb_theta": ax_theta.plot([], [], "k.", markersize=2, label="ORB theta")[0],
            "pred_theta": ax_theta.plot([], [], "tab:red", linewidth=1.2, label="pred theta")[0],
            "orb_fai": ax_fai.plot([], [], "k.", markersize=2, label="ORB fai")[0],
            "pred_fai": ax_fai.plot([], [], "tab:blue", linewidth=1.2, label="pred fai")[0],
            "theta_err": ax_err.plot([], [], "tab:red", linewidth=0.9, label="theta error")[0],
            "fai_err": ax_err.plot([], [], "tab:blue", linewidth=0.9, label="fai error")[0],
        }
        ax_theta.set_ylabel("theta (deg)")
        ax_fai.set_ylabel("fai (deg)")
        ax_err.set_ylabel("error (deg)")
        ax_err.set_xlabel("time (s)")
        for ax in (ax_theta, ax_fai, ax_err):
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper left")
        fig.tight_layout()
        return plt, fig, (ax_theta, ax_fai, ax_err), lines
    except Exception as exc:
        print(f"Live plot disabled: {exc}", file=sys.stderr)
        matplotlib.use("Agg", force=True)
        return None


def update_plot(live, t, pred_theta, orb_theta, pred_fai, orb_fai):
    if live is None or not t:
        return live
    plt, fig, axes, lines = live
    try:
        if not plt.fignum_exists(fig.number):
            return None
        theta_err = np.asarray(pred_theta) - np.asarray(orb_theta)
        fai_err = np.asarray(pred_fai) - np.asarray(orb_fai)
        lines["orb_theta"].set_data(t, orb_theta)
        lines["pred_theta"].set_data(t, pred_theta)
        lines["orb_fai"].set_data(t, orb_fai)
        lines["pred_fai"].set_data(t, pred_fai)
        lines["theta_err"].set_data(t, theta_err)
        lines["fai_err"].set_data(t, fai_err)
        for ax in axes:
            ax.relim()
            ax.autoscale_view()
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.001)
    except Exception as exc:
        print(f"\nLive plot disabled: {exc}", file=sys.stderr)
        return None
    return live


def save_plot(path, t, pred_theta, orb_theta, pred_fai, orb_fai):
    if not t:
        return
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    t = np.asarray(t)
    pt = np.asarray(pred_theta)
    ot = np.asarray(orb_theta)
    pf = np.asarray(pred_fai)
    of = np.asarray(orb_fai)
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 9), constrained_layout=True)
    axes[0].plot(t, ot, "k.", markersize=2, label="ORB theta")
    axes[0].plot(t, pt, "tab:red", linewidth=1.2, label="pred theta")
    axes[1].plot(t, of, "k.", markersize=2, label="ORB fai")
    axes[1].plot(t, pf, "tab:blue", linewidth=1.2, label="pred fai")
    axes[2].plot(t, pt - ot, "tab:red", linewidth=0.9, label="theta error")
    axes[2].plot(t, pf - of, "tab:blue", linewidth=0.9, label="fai error")
    axes[0].set_ylabel("theta (deg)")
    axes[1].set_ylabel("fai (deg)")
    axes[2].set_ylabel("error (deg)")
    axes[2].set_xlabel("time (s)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Live dual-strain theta/fai prediction compared with ORB.")
    parser.add_argument("model_json")
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument("--baseline-seconds", type=float, default=20.0)
    parser.add_argument("--input-smooth-window", type=int, default=3)
    parser.add_argument("--output-smooth-window", type=int, default=1)
    parser.add_argument("--plot-hz", type=float, default=5.0)
    parser.add_argument("--window", type=int, default=1500)
    parser.add_argument("--max-theta-age", type=float, default=0.2)
    parser.add_argument("--no-live", action="store_true")
    parser.add_argument("--output-dir", default="live_dual_strain_theta_fai_runs")
    args = parser.parse_args()

    with open(args.model_json, encoding="utf-8") as f:
        model = json.load(f)
    degree = int(model["degree"])
    weights = np.asarray(model["weights_bx_by"], dtype=float)

    rospy.init_node("live_dual_strain_theta_fai_compare", anonymous=True)
    orb = LatestOrb()
    rospy.Subscriber("/continuum_angles", Float32MultiArray, orb.angle_cb, queue_size=100)
    rospy.Subscriber("/continuum_angles_valid", Int32, orb.valid_cb, queue_size=20)
    rospy.Subscriber("/orb_slam3/tracking_state", Int32, orb.state_cb, queue_size=20)

    import serial
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, stamp)
    os.makedirs(run_dir)
    csv_path = os.path.join(run_dir, "live_dual_compare.csv")
    plot_path = os.path.join(run_dir, "live_dual_compare.png")
    meta_path = os.path.join(run_dir, "meta.txt")
    print(f"Saving CSV to: {csv_path}")
    print(f"Saving plot to: {plot_path}")

    ch1_ma = MovingAverage(args.input_smooth_window)
    ch2_ma = MovingAverage(args.input_smooth_window)
    theta_ma = MovingAverage(args.output_smooth_window)
    fai_ma = MovingAverage(args.output_smooth_window)
    baseline_samples = []
    baseline = None
    start = time.time()
    rows = usable = bad = 0
    last_plot = 0.0
    live = configure_plot(not args.no_live)
    t_plot = deque(maxlen=args.window)
    pt_plot = deque(maxlen=args.window)
    ot_plot = deque(maxlen=args.window)
    pf_plot = deque(maxlen=args.window)
    of_plot = deque(maxlen=args.window)
    t_all, pt_all, ot_all, pf_all, of_all = [], [], [], [], []

    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"model_json={os.path.abspath(args.model_json)}\n")
        for key, value in sorted(vars(args).items()):
            f.write(f"{key}={value}\n")

    header = [
        "wall_time", "rel_time", "ch1", "ch2", "smoothed_ch1", "smoothed_ch2",
        "baseline_ch1", "baseline_ch2", "pred_theta_deg", "pred_fai_deg",
        "orb_theta_deg", "orb_fai_deg", "theta_error_deg", "fai_error_deg",
        "theta_age", "angle_valid", "tracking_state", "baseline_locked", "raw_line",
    ]

    try:
        with serial.Serial(args.port, args.baudrate, timeout=args.timeout) as ser:
            ser.reset_input_buffer()
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                print("Hold theta=0, fai=0 during baseline calibration.")
                print("Running live dual prediction. Press Ctrl+C to stop.")
                while not rospy.is_shutdown():
                    raw = read_latest_frame(ser)
                    if not raw:
                        continue
                    line = raw.decode("ascii", errors="replace").strip()
                    values = parse_frame(line)
                    if values is None:
                        bad += 1
                        continue
                    now = time.time()
                    rel = now - start
                    ch1 = values[0]
                    ch2 = values[1]
                    s1 = ch1_ma.update(ch1)
                    s2 = ch2_ma.update(ch2)
                    theta_orb, fai_orb, age, valid, state = orb.snapshot(now)

                    if baseline is None:
                        baseline_samples.append((s1, s2))
                        if rel >= args.baseline_seconds:
                            baseline = np.median(np.asarray(baseline_samples), axis=0)
                            theta_ma = MovingAverage(args.output_smooth_window)
                            fai_ma = MovingAverage(args.output_smooth_window)
                            print(f"\nBaseline locked: ch1={baseline[0]:.3f}, ch2={baseline[1]:.3f}")

                    pred_theta = pred_fai = ""
                    theta_err = fai_err = ""
                    fresh = (
                        theta_orb is not None
                        and age is not None
                        and age <= args.max_theta_age
                        and valid == 1
                        and state == 2
                    )
                    if baseline is not None:
                        features = polynomial_features(s1 - baseline[0], s2 - baseline[1], degree)
                        bx, by = features @ weights
                        pred_theta = theta_ma.update(float(np.hypot(bx, by)))
                        pred_fai = fai_ma.update(float(np.clip(np.degrees(np.arctan2(by, bx)), 0.0, 90.0)))
                        if fresh:
                            theta_err = pred_theta - theta_orb
                            fai_err = pred_fai - fai_orb
                            usable += 1
                            t_plot.append(rel)
                            pt_plot.append(pred_theta)
                            ot_plot.append(theta_orb)
                            pf_plot.append(pred_fai)
                            of_plot.append(fai_orb)
                            t_all.append(rel)
                            pt_all.append(pred_theta)
                            ot_all.append(theta_orb)
                            pf_all.append(pred_fai)
                            of_all.append(fai_orb)

                    writer.writerow(
                        [
                            f"{now:.6f}", f"{rel:.6f}", f"{ch1:.6f}", f"{ch2:.6f}",
                            f"{s1:.6f}", f"{s2:.6f}",
                            "" if baseline is None else f"{baseline[0]:.6f}",
                            "" if baseline is None else f"{baseline[1]:.6f}",
                            "" if pred_theta == "" else f"{pred_theta:.6f}",
                            "" if pred_fai == "" else f"{pred_fai:.6f}",
                            "" if theta_orb is None else f"{theta_orb:.6f}",
                            "" if fai_orb is None else f"{fai_orb:.6f}",
                            "" if theta_err == "" else f"{theta_err:.6f}",
                            "" if fai_err == "" else f"{fai_err:.6f}",
                            "" if age is None else f"{age:.6f}", valid, state,
                            int(baseline is not None), line,
                        ]
                    )
                    rows += 1
                    if rows % 20 == 0:
                        f.flush()
                        print(
                            f"\rrows={rows} usable={usable} bad={bad} ch1={ch1:.1f} ch2={ch2:.1f} "
                            f"pred_theta={pred_theta if pred_theta != '' else 'waiting'} "
                            f"pred_fai={pred_fai if pred_fai != '' else 'waiting'}",
                            end="", flush=True,
                        )
                    if baseline is not None and now - last_plot >= 1.0 / max(args.plot_hz, 0.1):
                        live = update_plot(
                            live, list(t_plot), list(pt_plot), list(ot_plot),
                            list(pf_plot), list(of_plot),
                        )
                        last_plot = now
    except KeyboardInterrupt:
        print("\nStopping and saving...", file=sys.stderr)

    save_plot(plot_path, t_all, pt_all, ot_all, pf_all, of_all)
    print(f"\nDone. rows={rows}, usable={usable}, bad_lines={bad}, run_dir={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
