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
            return self.theta, self.phi, now - self.wall_time, self.valid, self.tracking_state


class CausalMovingAverage:
    def __init__(self, window):
        self.window = max(int(window), 1)
        self.values = deque(maxlen=self.window)

    def update(self, value):
        self.values.append(float(value))
        return float(np.mean(self.values))


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
        fig, (ax_theta, ax_err) = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
        orb_line, = ax_theta.plot([], [], "k.", markersize=2, label="ORB theta")
        pred_line, = ax_theta.plot([], [], "tab:red", linewidth=1.3, label="strain predicted theta")
        err_line, = ax_err.plot([], [], "tab:purple", linewidth=1.0, label="pred - ORB")
        ax_theta.set_ylabel("theta (deg)")
        ax_err.set_ylabel("error (deg)")
        ax_err.set_xlabel("time (s)")
        for ax in (ax_theta, ax_err):
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper left")
        fig.tight_layout()
        return plt, fig, ax_theta, ax_err, orb_line, pred_line, err_line
    except Exception as exc:
        print(f"Live plot disabled: {exc}", file=sys.stderr)
        matplotlib.use("Agg", force=True)
        return None


def update_live_plot(live, t, pred, orb):
    if live is None or not t:
        return live
    plt, fig, ax_theta, ax_err, orb_line, pred_line, err_line = live
    try:
        if not plt.fignum_exists(fig.number):
            return None
        orb_arr = np.asarray(orb, dtype=float)
        pred_arr = np.asarray(pred, dtype=float)
        err = pred_arr - orb_arr
        orb_line.set_data(t, orb)
        pred_line.set_data(t, pred)
        err_line.set_data(t, err)
        for ax in (ax_theta, ax_err):
            ax.relim()
            ax.autoscale_view()
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.001)
    except Exception as exc:
        print(f"\nLive plot disabled: {exc}", file=sys.stderr)
        return None
    return live


def save_plot(path, t, pred, orb):
    if not t:
        return
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    t = np.asarray(t, dtype=float)
    pred = np.asarray(pred, dtype=float)
    orb = np.asarray(orb, dtype=float)
    err = pred - orb
    fig, (ax_theta, ax_err) = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
    ax_theta.plot(t, orb, "k.", markersize=2, label="ORB theta")
    ax_theta.plot(t, pred, "tab:red", linewidth=1.2, label="strain predicted theta")
    ax_err.plot(t, err, "tab:purple", linewidth=1.0, label="pred - ORB")
    ax_theta.set_ylabel("theta (deg)")
    ax_err.set_ylabel("error (deg)")
    ax_err.set_xlabel("time (s)")
    for ax in (ax_theta, ax_err):
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load_model(path):
    with open(path, encoding="utf-8") as f:
        model = json.load(f)
    return {
        "channel": int(model.get("channel", 1)),
        "smooth_window": int(model.get("smooth_window", 1)),
        "baseline": float(model["baseline"]),
        "coeffs": np.asarray(model["coefficients_high_to_low"], dtype=float),
    }


def predict_theta(coeffs, smoothed_strain, baseline):
    return float(np.polyval(coeffs, smoothed_strain - baseline))


def run(args):
    model = load_model(args.model_json)
    channel = args.channel if args.channel is not None else model["channel"]
    input_smooth_window = args.input_smooth_window or model["smooth_window"]
    rospy.init_node("live_strain_theta_predict_compare", anonymous=True)
    latest = LatestOrbAngle()
    rospy.Subscriber(args.angle_topic, Float32MultiArray, latest.angle_cb, queue_size=100)
    rospy.Subscriber(args.valid_topic, Int32, latest.valid_cb, queue_size=20)
    rospy.Subscriber(args.state_topic, Int32, latest.state_cb, queue_size=20)

    ser = open_serial(args.port, args.baudrate, args.timeout)
    run_dir = make_run_dir(args.output_dir)
    csv_path = os.path.join(run_dir, "live_strain_theta_compare.csv")
    plot_path = os.path.join(run_dir, "live_strain_theta_compare.png")
    meta_path = os.path.join(run_dir, "meta.txt")
    print(f"Saving CSV to: {csv_path}")
    print(f"Saving plot to: {plot_path}")

    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"model_json={os.path.abspath(args.model_json)}\n")
        for key, value in sorted(vars(args).items()):
            f.write(f"{key}={value}\n")
        f.write(f"model_smooth_window={model['smooth_window']}\n")
        f.write(f"effective_input_smooth_window={input_smooth_window}\n")
        f.write(f"model_baseline={model['baseline']}\n")
        f.write(f"start_time={datetime.now().isoformat(timespec='seconds')}\n")

    ma = CausalMovingAverage(input_smooth_window)
    pred_output_ma = CausalMovingAverage(args.output_smooth_window)
    baseline_samples = []
    baseline_orb_samples = []
    baseline = model["baseline"]
    theta_bias = 0.0
    baseline_locked = not args.auto_baseline
    start_wall = time.time()
    rows = 0
    usable = 0
    bad = 0
    interrupted = False
    last_plot_wall = 0.0
    live = configure_live_plot(not args.no_live)
    t_plot = deque(maxlen=args.window)
    pred_plot = deque(maxlen=args.window)
    orb_plot = deque(maxlen=args.window)
    t_all, pred_all, orb_all = [], [], []

    header = [
        "wall_time",
        "rel_time",
        "strain_value",
        "smoothed_strain",
        "baseline",
        "pred_theta_deg",
        "pred_raw_theta_deg",
        "pred_scale",
        "pred_offset",
        "theta_bias",
        "orb_theta_deg",
        "orb_phi_deg",
        "theta_age",
        "angle_valid",
        "tracking_state",
        "error_deg",
        "baseline_locked",
        "raw_line",
    ]

    with ser:
        ser.reset_input_buffer()
        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(header)
            csv_file.flush()
            print("Running live prediction. Press Ctrl+C to stop.")
            if args.auto_baseline:
                print(f"Hold zero pose for the first {args.baseline_seconds:.1f}s to estimate baseline.")
            try:
                while not rospy.is_shutdown():
                    if args.duration is not None and time.time() - start_wall >= args.duration:
                        break
                    raw = read_latest_frame(ser, args.terminator) if args.latest_only else read_frame(ser, args.terminator)
                    if not raw:
                        continue
                    line = raw.decode(args.encoding, errors="replace").strip()
                    values = parse_frame(line)
                    if values is None:
                        bad += 1
                        continue

                    now = time.time()
                    rel = now - start_wall
                    strain = values[channel - 1]
                    smoothed = ma.update(strain)
                    theta, phi, theta_age, angle_valid, tracking_state = latest.snapshot(now)
                    if args.auto_baseline and not baseline_locked:
                        baseline_samples.append(smoothed)
                        if theta is not None and theta_age is not None and theta_age <= args.max_theta_age:
                            baseline_orb_samples.append(theta)
                        if rel >= args.baseline_seconds:
                            baseline = float(np.median(baseline_samples))
                            baseline_pred = predict_theta(model["coeffs"], baseline, baseline)
                            if baseline_orb_samples:
                                theta_zero = float(np.median(baseline_orb_samples))
                                if args.auto_theta_bias:
                                    theta_bias = baseline_pred - theta_zero
                                    print(
                                        f"\nTheta bias locked: {theta_bias:.3f} deg "
                                        f"(pred_zero={baseline_pred:.3f}, orb_zero={theta_zero:.3f})"
                                    )
                                if abs(theta_zero) > args.baseline_orb_warn_deg:
                                    print(
                                        f"\nWARNING: ORB theta during baseline was {theta_zero:.3f} deg. "
                                        "Keep the arm straight and start orb_angle_calculator at zero pose.",
                                        file=sys.stderr,
                                    )
                            baseline_locked = True
                            pred_output_ma = CausalMovingAverage(args.output_smooth_window)
                            print(f"\nBaseline locked: {baseline:.3f}")

                    pred_raw = predict_theta(model["coeffs"], smoothed, baseline) - theta_bias
                    pred_calibrated = pred_raw * args.pred_scale + args.pred_offset
                    pred_theta = pred_output_ma.update(pred_calibrated)
                    fresh = theta is not None and theta_age is not None and theta_age <= args.max_theta_age
                    row_usable = bool(fresh and (not args.require_valid or angle_valid == 1))
                    err = "" if theta is None else pred_theta - theta
                    rows += 1
                    usable += int(row_usable)

                    writer.writerow(
                        [
                            f"{now:.6f}",
                            f"{rel:.6f}",
                            f"{strain:.6f}",
                            f"{smoothed:.6f}",
                            f"{baseline:.6f}",
                            f"{pred_theta:.6f}",
                            f"{pred_raw:.6f}",
                            f"{args.pred_scale:.9f}",
                            f"{args.pred_offset:.6f}",
                            f"{theta_bias:.6f}",
                            "" if theta is None else f"{theta:.6f}",
                            "" if phi is None else f"{phi:.6f}",
                            "" if theta_age is None else f"{theta_age:.6f}",
                            angle_valid,
                            tracking_state,
                            "" if err == "" else f"{err:.6f}",
                            int(baseline_locked),
                            line,
                        ]
                    )
                    if rows % args.flush_every == 0:
                        csv_file.flush()

                    if row_usable and baseline_locked:
                        t_plot.append(rel)
                        pred_plot.append(pred_theta)
                        orb_plot.append(theta)
                        t_all.append(rel)
                        pred_all.append(pred_theta)
                        orb_all.append(theta)

                    plot_period = 1.0 / args.plot_hz if args.plot_hz > 0 else 0.0
                    if rows % args.plot_every == 0 and baseline_locked and now - last_plot_wall >= plot_period:
                        live = update_live_plot(live, list(t_plot), list(pred_plot), list(orb_plot))
                        last_plot_wall = now

                    if rows % args.status_every == 0:
                        hz = rows / rel if rel > 0 else 0.0
                        if err == "":
                            err_text = "none"
                        else:
                            err_text = f"{err:.2f}"
                        print(
                            f"\rrows={rows} usable={usable} bad={bad} hz={hz:.1f} "
                            f"strain={strain:.1f} pred={pred_theta:.2f} "
                            f"orb={theta if theta is not None else 'none'} err={err_text} baseline={baseline:.1f}",
                            end="",
                            flush=True,
                        )
            except KeyboardInterrupt:
                interrupted = True
                print("\nStopping and saving captured data...", file=sys.stderr)
            finally:
                csv_file.flush()

    save_plot(plot_path, t_all, pred_all, orb_all)
    print(f"\nDone. rows={rows}, usable={usable}, bad_lines={bad}, baseline={baseline:.3f}, run_dir={run_dir}")
    return 130 if interrupted else 0


def main():
    parser = argparse.ArgumentParser(description="Live strain theta prediction compared with ORB theta.")
    parser.add_argument("model_json")
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--channel", type=int, default=None)
    parser.add_argument("--angle-topic", default="/continuum_angles")
    parser.add_argument("--valid-topic", default="/continuum_angles_valid")
    parser.add_argument("--state-topic", default="/orb_slam3/tracking_state")
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument("--encoding", default="ascii")
    parser.add_argument("--terminator", choices=["semicolon", "newline"], default="semicolon")
    parser.add_argument("--latest-only", action="store_true", default=True, help="drop queued serial frames and process only the newest complete frame")
    parser.add_argument("--keep-all-frames", dest="latest_only", action="store_false", help="process every serial frame, which is for logging not low-latency live display")
    parser.add_argument("--max-theta-age", type=float, default=0.2)
    parser.add_argument("--require-valid", action="store_true")
    parser.add_argument("--auto-baseline", action="store_true")
    parser.add_argument("--baseline-seconds", type=float, default=20.0)
    parser.add_argument("--auto-theta-bias", action="store_true", help="subtract zero-pose pred-ORB bias measured during baseline")
    parser.add_argument("--baseline-orb-warn-deg", type=float, default=5.0)
    parser.add_argument("--input-smooth-window", type=int, default=None, help="override the model strain smoothing window")
    parser.add_argument("--output-smooth-window", type=int, default=1, help="smooth predicted theta with a causal moving average")
    parser.add_argument("--pred-scale", type=float, default=1.0, help="multiply predicted theta after bias correction")
    parser.add_argument("--pred-offset", type=float, default=0.0, help="add degrees to predicted theta after scaling")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--output-dir", default="live_strain_theta_runs")
    parser.add_argument("--no-live", action="store_true")
    parser.add_argument("--window", type=int, default=1500)
    parser.add_argument("--plot-every", type=int, default=1)
    parser.add_argument("--plot-hz", type=float, default=5.0)
    parser.add_argument("--flush-every", type=int, default=20)
    parser.add_argument("--status-every", type=int, default=20)
    args = parser.parse_args()

    try:
        return run(args)
    except Exception as exc:
        if exc.__class__.__name__ == "SerialException":
            print(f"Could not open serial port: {exc}", file=sys.stderr)
            return 1
        raise


if __name__ == "__main__":
    raise SystemExit(main())
