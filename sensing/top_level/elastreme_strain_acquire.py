#!/usr/bin/env python3
import argparse
import csv
import os
import re
import sys
import time
from datetime import datetime

import matplotlib


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


def make_run_dir(base_dir):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, stamp)
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


def save_plot(path, times, channel_values, channel, title):
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.plot(times, channel_values, linewidth=1.4)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(f"channel {channel} value")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=150)
    plt.close(fig)


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
        fig, ax = plt.subplots(figsize=(10, 5))
        (line,) = ax.plot([], [], linewidth=1.4)
        ax.set_xlabel("time (s)")
        ax.grid(True, alpha=0.3)
        return plt, fig, ax, line
    except Exception as exc:
        print(f"Live plot disabled: {exc}", file=sys.stderr)
        matplotlib.use("Agg", force=True)
        return None


def update_live_plot(live, times, values, channel):
    if live is None or not times:
        return
    plt, fig, ax, line = live
    line.set_data(times, values)
    ax.set_ylabel(f"channel {channel} value")
    ax.set_title(f"Elastreme Sense channel {channel}")
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.001)


def print_status(samples, bad_lines, rel_time, channel, value):
    hz = samples / rel_time if rel_time > 0 else 0.0
    text = (
        f"\rsamples={samples} bad={bad_lines} "
        f"t={rel_time:.2f}s hz={hz:.1f} ch{channel}={value:.3f}"
    )
    print(text, end="", flush=True)


def open_serial(port, baudrate, timeout):
    import serial

    return serial.Serial(port=port, baudrate=baudrate, timeout=timeout)


def sniff_raw(args):
    end_time = time.time() + args.raw_seconds
    count = 0
    with open_serial(args.port, args.baudrate, args.timeout) as ser:
        ser.reset_input_buffer()
        print(f"Listening for raw bytes on {args.port} for {args.raw_seconds:.1f}s...")
        while time.time() < end_time:
            chunk = ser.read(max(1, args.raw_chunk))
            if not chunk:
                continue
            count += len(chunk)
            text = chunk.decode(args.encoding, errors="replace")
            print(f"{len(chunk)} bytes: {chunk!r} | {text!r}")
    print(f"Raw bytes received: {count}")
    return 0 if count else 2


def acquire(args):
    ser = open_serial(args.port, args.baudrate, args.timeout)

    run_dir = make_run_dir(args.output_dir)
    csv_path = os.path.join(run_dir, "strain_channels.csv")
    plot_path = os.path.join(run_dir, f"channel_{args.channel}.png")
    meta_path = os.path.join(run_dir, "meta.txt")
    print(f"Saving CSV to: {csv_path}")
    print(f"Saving plot to: {plot_path}")

    start_wall = time.time()
    times = []
    channel_values = []
    good_frames = 0
    bad_lines = 0
    interrupted = False
    last_rx_wall = start_wall
    last_idle_report = start_wall
    live = configure_live_plot(not args.no_live)

    with open(meta_path, "w", encoding="utf-8") as meta:
        meta.write(f"port={args.port}\n")
        meta.write(f"baudrate={args.baudrate}\n")
        meta.write(f"channel={args.channel}\n")
        meta.write(f"start_time={datetime.now().isoformat(timespec='seconds')}\n")

    with ser:
        ser.reset_input_buffer()
        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "wall_time",
                    "rel_time",
                    "ch1",
                    "ch2",
                    "ch3",
                    "ch4",
                    "ch5",
                    "ch6",
                    "raw_line",
                ]
            )
            csv_file.flush()

            print("Acquiring. Press Ctrl+C to stop.")
            try:
                while True:
                    if args.duration is not None and time.time() - start_wall >= args.duration:
                        break
                    if args.samples is not None and good_frames >= args.samples:
                        break

                    raw = read_frame(ser, args.terminator)
                    if not raw:
                        now = time.time()
                        if now - last_idle_report >= args.idle_report_every:
                            idle = now - last_rx_wall
                            print(
                                f"\nNo serial bytes received for {idle:.1f}s "
                                f"on {args.port}.",
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
                            print(f"Bad line: {line}", file=sys.stderr)
                        continue

                    now = time.time()
                    rel = now - start_wall
                    channels = values[:6]
                    selected = channels[args.channel - 1]
                    times.append(rel)
                    channel_values.append(selected)
                    good_frames += 1

                    writer.writerow(
                        [
                            f"{now:.6f}",
                            f"{rel:.6f}",
                            *[f"{v:.6f}" for v in channels],
                            line,
                        ]
                    )
                    if good_frames % args.flush_every == 0:
                        csv_file.flush()

                    if good_frames % args.plot_every == 0:
                        update_live_plot(live, times, channel_values, args.channel)

                    if good_frames % args.status_every == 0:
                        print_status(good_frames, bad_lines, rel, args.channel, selected)
            except KeyboardInterrupt:
                interrupted = True
                print("\nStopping and saving captured data...", file=sys.stderr)
            finally:
                csv_file.flush()

    if times:
        save_plot(
            plot_path,
            times,
            channel_values,
            args.channel,
            f"Elastreme Sense channel {args.channel}",
        )
    else:
        print("No valid frames were captured; plot was not generated.", file=sys.stderr)

    print(f"Done. valid_samples={good_frames}, bad_lines={bad_lines}, run_dir={run_dir}")
    if interrupted:
        return 130
    return 0 if good_frames else 2


def main():
    parser = argparse.ArgumentParser(
        description="Acquire and plot Elastreme Sense 6-channel serial data."
    )
    parser.add_argument("--port", default="/dev/ttyUSB0", help="serial device")
    parser.add_argument("--baudrate", type=int, default=115200, help="serial baudrate")
    parser.add_argument("--channel", type=int, default=1, choices=range(1, 7), help="channel to plot")
    parser.add_argument("--duration", type=float, default=None, help="seconds to acquire")
    parser.add_argument("--samples", type=int, default=None, help="number of valid frames to acquire")
    parser.add_argument("--output-dir", default="elastreme_runs", help="directory for saved runs")
    parser.add_argument("--timeout", type=float, default=1.0, help="serial read timeout in seconds")
    parser.add_argument("--encoding", default="ascii", help="serial text encoding")
    parser.add_argument(
        "--terminator",
        choices=["semicolon", "newline"],
        default="semicolon",
        help="frame terminator used by the serial stream",
    )
    parser.add_argument(
        "--idle-report-every",
        type=float,
        default=3.0,
        help="seconds between no-data status messages",
    )
    parser.add_argument("--no-live", action="store_true", help="disable live plot window")
    parser.add_argument("--plot-every", type=int, default=1, help="refresh live plot every N valid frames")
    parser.add_argument("--flush-every", type=int, default=10, help="flush CSV every N valid frames")
    parser.add_argument("--status-every", type=int, default=1, help="print status every N valid frames")
    parser.add_argument("--print-bad", action="store_true", help="print unparsable serial lines")
    parser.add_argument("--raw-sniff", action="store_true", help="print raw serial bytes and exit")
    parser.add_argument("--raw-seconds", type=float, default=5.0, help="seconds for --raw-sniff")
    parser.add_argument("--raw-chunk", type=int, default=64, help="read size for --raw-sniff")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        if args.raw_sniff:
            return sniff_raw(args)
        return acquire(args)
    except FileNotFoundError as exc:
        print(f"Serial port not found: {exc}", file=sys.stderr)
        return 1
    except PermissionError as exc:
        print(f"No permission to access serial port: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        if exc.__class__.__name__ == "SerialException":
            print(f"Could not open serial port: {exc}", file=sys.stderr)
            return 1
        raise


if __name__ == "__main__":
    raise SystemExit(main())
