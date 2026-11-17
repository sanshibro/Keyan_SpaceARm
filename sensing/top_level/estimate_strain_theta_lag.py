#!/usr/bin/env python3
import argparse
import csv
import os
import sys

import numpy as np


def resolve_csv(path):
    if os.path.isdir(path):
        candidate = os.path.join(path, "orb_strain_sync.csv")
        if os.path.exists(candidate):
            return candidate
    return path


def read_rows(csv_path, channel, require_usable):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if int(row.get("strain_channel", channel)) != channel:
                    continue
                if require_usable and int(float(row.get("usable", "0") or 0)) != 1:
                    continue
                rows.append(
                    (
                        float(row["rel_time"]),
                        float(row["strain_value"]),
                        float(row["orb_theta_deg"]),
                        float(row.get("theta_age", "0") or 0.0),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    return np.asarray(rows, dtype=float)


def normalize(values):
    values = np.asarray(values, dtype=float)
    return (values - np.mean(values)) / (np.std(values) + 1e-12)


def estimate_lag(t, strain, theta, max_lag, dt):
    grid = np.arange(float(t[0]), float(t[-1]), dt)
    if len(grid) < 20:
        raise ValueError("not enough time span for lag estimation")
    strain_i = np.interp(grid, t, strain)
    theta_i = np.interp(grid, t, theta)

    # Derivatives make the estimate less sensitive to offset and slow drift.
    strain_d = normalize(np.diff(strain_i))
    theta_d = normalize(np.diff(theta_i))
    g = grid[1:]

    best = None
    lag_values = np.arange(-max_lag, max_lag + 0.5 * dt, dt)
    for lag in lag_values:
        shifted_theta = np.interp(g + lag, g, theta_d, left=np.nan, right=np.nan)
        mask = ~np.isnan(shifted_theta)
        if int(np.sum(mask)) < 20:
            continue
        corr = float(np.mean(strain_d[mask] * shifted_theta[mask]))
        if best is None or corr > best[1]:
            best = (float(lag), corr, int(np.sum(mask)))
    if best is None:
        raise ValueError("could not estimate lag")
    return best


def main():
    parser = argparse.ArgumentParser(description="Estimate time lag between strain and ORB theta.")
    parser.add_argument("csv_or_run_dir")
    parser.add_argument("--channel", type=int, default=1)
    parser.add_argument("--max-lag", type=float, default=3.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--allow-unusable", action="store_true")
    args = parser.parse_args()

    csv_path = resolve_csv(args.csv_or_run_dir)
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 1

    rows = read_rows(csv_path, args.channel, require_usable=not args.allow_unusable)
    if len(rows) < 20:
        print(f"Not enough samples: {len(rows)}", file=sys.stderr)
        return 2

    t, strain, theta, theta_age = rows.T
    hz = 1.0 / float(np.mean(np.diff(t))) if len(t) > 1 else 0.0
    lag, corr, used = estimate_lag(t, strain, theta, args.max_lag, args.dt)

    print(f"csv={csv_path}")
    print(f"samples={len(rows)} duration={t[-1] - t[0]:.3f}s mean_hz={hz:.2f}")
    print(f"theta_age_mean={np.mean(theta_age):.4f}s theta_age_max={np.max(theta_age):.4f}s")
    print(f"best_lag={lag:.3f}s corr={corr:.3f} used={used}")
    if lag < 0:
        print("Interpretation: theta changes earlier than strain; strain appears delayed by about %.3fs." % abs(lag))
    elif lag > 0:
        print("Interpretation: strain changes earlier than theta by about %.3fs." % lag)
    else:
        print("Interpretation: no measurable lag at this resolution.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
