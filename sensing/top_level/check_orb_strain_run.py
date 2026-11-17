#!/usr/bin/env python3
import argparse
import csv
import math
import os
import sys

import numpy as np


def resolve_csv(path):
    if os.path.isdir(path):
        candidate = os.path.join(path, "orb_strain_sync.csv")
        if os.path.exists(candidate):
            return candidate
    return path


def read_rows(csv_path, channel, max_theta_age):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if int(row.get("strain_channel", channel)) != channel:
                    continue
                if int(float(row.get("usable", "0") or 0)) != 1:
                    continue
                theta_age = row.get("theta_age", "")
                if theta_age and float(theta_age) > max_theta_age:
                    continue
                rows.append(
                    (
                        float(row["rel_time"]),
                        float(row["strain_value"]),
                        float(row["orb_theta_deg"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def read_meta(csv_path):
    meta_path = os.path.join(os.path.dirname(csv_path), "meta.txt")
    meta = {}
    if not os.path.exists(meta_path):
        return meta
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            if "=" not in line:
                continue
            key, value = line.rstrip("\n").split("=", 1)
            meta[key] = value
    return meta


def estimate_lag(t, strain, theta, dt, max_lag):
    if len(t) < 20 or t[-1] <= t[0]:
        return float("nan"), float("nan")
    grid = np.arange(t[0], t[-1], dt)
    if len(grid) < 20:
        return float("nan"), float("nan")
    s = np.interp(grid, t, strain)
    y = np.interp(grid, t, theta)
    ds = np.gradient(s)
    dy = np.gradient(y)
    ds = (ds - np.mean(ds)) / (np.std(ds) + 1e-9)
    dy = (dy - np.mean(dy)) / (np.std(dy) + 1e-9)
    max_steps = int(round(max_lag / dt))
    best_corr = -float("inf")
    best_lag = 0.0
    for step in range(-max_steps, max_steps + 1):
        if step < 0:
            a = ds[-step:]
            b = dy[: len(a)]
        elif step > 0:
            a = ds[:-step]
            b = dy[step:]
        else:
            a = ds
            b = dy
        if len(a) < 20:
            continue
        corr = float(np.mean(a * b))
        if corr > best_corr:
            best_corr = corr
            best_lag = step * dt
    return best_lag, best_corr


def main():
    parser = argparse.ArgumentParser(description="Check synchronized ORB theta + strain run quality.")
    parser.add_argument("csv_or_run_dir")
    parser.add_argument("--channel", type=int, default=1)
    parser.add_argument("--max-theta-age", type=float, default=0.2)
    parser.add_argument("--min-corr", type=float, default=0.85)
    parser.add_argument("--max-abs-lag", type=float, default=0.3)
    args = parser.parse_args()

    csv_path = resolve_csv(args.csv_or_run_dir)
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 1
    meta = read_meta(csv_path)

    rows = read_rows(csv_path, args.channel, args.max_theta_age)
    if len(rows) < 20:
        print(f"usable_samples={len(rows)}")
        print("quality=BAD reason=not_enough_usable_samples")
        return 2

    arr = np.asarray(rows, dtype=float)
    t = arr[:, 0]
    strain = arr[:, 1]
    theta = arr[:, 2]
    duration = float(t[-1] - t[0])
    hz = float(len(rows) / duration) if duration > 0 else float("nan")
    corr = float(np.corrcoef(strain, theta)[0, 1]) if np.std(strain) > 1e-9 and np.std(theta) > 1e-9 else float("nan")
    lag, lag_corr = estimate_lag(t, strain, theta, dt=0.05, max_lag=2.0)
    strain_range = float(np.max(strain) - np.min(strain))
    theta_range = float(np.max(theta) - np.min(theta))
    theta_step_max = float(np.max(np.abs(np.diff(theta)))) if len(theta) > 1 else 0.0

    ok = (
        math.isfinite(corr)
        and corr >= args.min_corr
        and math.isfinite(lag)
        and abs(lag) <= args.max_abs_lag
        and strain_range > 50.0
        and theta_range > 40.0
    )

    print(f"csv={csv_path}")
    if meta:
        print(f"latest_only={meta.get('latest_only', 'unknown')}")
    print(f"usable_samples={len(rows)} duration={duration:.3f}s hz={hz:.2f}")
    print(f"strain_range={strain_range:.3f} min={float(np.min(strain)):.3f} max={float(np.max(strain)):.3f}")
    print(f"theta_range={theta_range:.3f} min={float(np.min(theta)):.3f} max={float(np.max(theta)):.3f}")
    print(f"corr_strain_theta={corr:.3f}")
    print(f"best_derivative_lag={lag:.3f}s lag_corr={lag_corr:.3f}")
    print(f"theta_step_max={theta_step_max:.3f}deg")
    print("quality=GOOD" if ok else "quality=BAD")
    if meta.get("latest_only") == "False":
        print("warning=latest_only_false_can_pair_old_serial_frames_with_current_orb_theta")
    if not ok:
        print("recommendation=do_not_use_for_training")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
