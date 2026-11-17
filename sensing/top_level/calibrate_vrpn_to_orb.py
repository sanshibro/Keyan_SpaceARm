#!/usr/bin/env python3
import argparse
import csv
import math
import os
import sys

import numpy as np


def normalize_signed(angle):
    return (angle + 180.0) % 360.0 - 180.0


def circular_mean_deg(values):
    angles = np.radians(np.array(values, dtype=float))
    return normalize_signed(math.degrees(math.atan2(float(np.mean(np.sin(angles))), float(np.mean(np.cos(angles))))))


def circular_abs_error(values):
    return float(np.median(np.abs([normalize_signed(v) for v in values])))


def read_angles(csv_path):
    orb = []
    vrpn = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            item = (
                float(row["rel_time"]),
                float(row["theta_deg"]),
                float(row["phi_deg"]),
                int(float(row.get("valid", 1))),
            )
            if row["source"] == "orb":
                orb.append(item)
            elif row["source"] == "vrpn":
                vrpn.append(item)
    return orb, vrpn


def pair_by_time(orb, vrpn, max_dt):
    pairs = []
    j = 0
    for o in orb:
        while j + 1 < len(vrpn) and abs(vrpn[j + 1][0] - o[0]) < abs(vrpn[j][0] - o[0]):
            j += 1
        if vrpn and abs(vrpn[j][0] - o[0]) <= max_dt:
            pairs.append((o, vrpn[j]))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Estimate display calibration from VRPN angles to ORB angles.")
    parser.add_argument("run_dir", help="angle_runs/YYYYMMDD_HHMMSS directory or angles.csv path")
    parser.add_argument("--theta-min", type=float, default=5.0, help="Use samples with both theta above this value.")
    parser.add_argument("--max-dt", type=float, default=0.08, help="Max pairing time difference in seconds.")
    args = parser.parse_args()

    csv_path = args.run_dir
    if os.path.isdir(csv_path):
        csv_path = os.path.join(csv_path, "angles.csv")
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 2

    orb, vrpn = read_angles(csv_path)
    pairs = pair_by_time(orb, vrpn, args.max_dt)
    pairs = [
        (o, v)
        for o, v in pairs
        if o[3] and v[3] and o[1] >= args.theta_min and v[1] >= args.theta_min
    ]
    if len(pairs) < 20:
        print(f"Not enough paired samples after filtering: {len(pairs)}", file=sys.stderr)
        print("Use a normal-vision run with several bends and clear theta > 5 deg.", file=sys.stderr)
        return 3

    orb_theta = np.array([o[1] for o, _ in pairs], dtype=float)
    vrpn_theta = np.array([v[1] for _, v in pairs], dtype=float)
    A = np.vstack([vrpn_theta, np.ones_like(vrpn_theta)]).T
    theta_scale, theta_bias = np.linalg.lstsq(A, orb_theta, rcond=None)[0]

    orb_phi = np.array([o[2] for o, _ in pairs], dtype=float)
    vrpn_phi = np.array([v[2] for _, v in pairs], dtype=float)

    best = None
    for sign in (1.0, -1.0):
        offsets = [normalize_signed(o - sign * v) for o, v in zip(orb_phi, vrpn_phi)]
        offset = circular_mean_deg(offsets)
        errors = [normalize_signed(o - (sign * v + offset)) for o, v in zip(orb_phi, vrpn_phi)]
        err = circular_abs_error(errors)
        candidate = (err, sign, offset)
        if best is None or candidate[0] < best[0]:
            best = candidate

    phi_err, phi_sign, phi_offset = best
    theta_pred = theta_scale * vrpn_theta + theta_bias
    theta_err = float(np.median(np.abs(theta_pred - orb_theta)))

    print(f"paired_samples={len(pairs)}")
    print(f"theta_scale={theta_scale:.6f}")
    print(f"theta_bias={theta_bias:.6f}")
    print(f"median_theta_error_after_fit={theta_err:.3f} deg")
    print(f"vrpn_phi_sign={phi_sign:.0f}")
    print(f"vrpn_phi_offset_deg={phi_offset:.6f}")
    print(f"median_phi_error_after_fit={phi_err:.3f} deg")
    print()
    print("Launch command with display calibration:")
    print(
        "/home/uav/桌面/Test/run_orb_vrpn_angle_compare.sh "
        f"vrpn_theta_scale:={theta_scale:.6f} "
        f"vrpn_theta_bias:={theta_bias:.6f} "
        f"vrpn_phi_sign:={phi_sign:.0f} "
        f"vrpn_phi_offset_deg:={phi_offset:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
