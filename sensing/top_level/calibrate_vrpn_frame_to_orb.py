#!/usr/bin/env python3
import argparse
import math
import os
import sys

import numpy as np

try:
    import rosbag
except ImportError:
    rosbag = None


def quat_to_rotation_xyzw(q):
    x, y, z, w = q
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n < 1e-12:
        raise ValueError("zero quaternion")
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
        [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
        [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)],
    ])


def average_rotation(rotations):
    mean = np.mean(np.stack(rotations), axis=0)
    u, _, vt = np.linalg.svd(mean)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def rotation_angle_deg(r):
    c = (np.trace(r) - 1.0) * 0.5
    return math.degrees(math.acos(float(np.clip(c, -1.0, 1.0))))


def nearest_pairs(orb, vrpn, max_dt):
    pairs = []
    j = 0
    for to, ro in orb:
        while j + 1 < len(vrpn) and abs(vrpn[j + 1][0] - to) < abs(vrpn[j][0] - to):
            j += 1
        if vrpn and abs(vrpn[j][0] - to) <= max_dt:
            pairs.append((to, ro, vrpn[j][0], vrpn[j][1]))
    return pairs


def read_bag(path, orb_topic, vrpn_topic):
    orb = []
    vrpn = []
    with rosbag.Bag(path) as bag:
        for topic, msg, t in bag.read_messages(topics=[orb_topic, vrpn_topic]):
            stamp = msg.header.stamp.to_sec() if getattr(msg, "header", None) else t.to_sec()
            if topic == orb_topic:
                q = msg.pose.pose.orientation
                orb.append((stamp, quat_to_rotation_xyzw((q.x, q.y, q.z, q.w))))
            elif topic == vrpn_topic:
                q = msg.pose.orientation
                vrpn.append((stamp, quat_to_rotation_xyzw((q.x, q.y, q.z, q.w))))
    orb.sort(key=lambda x: x[0])
    vrpn.sort(key=lambda x: x[0])
    return orb, vrpn


def estimate_frame_rotation(pairs, zero_samples, min_angle_deg):
    if len(pairs) <= zero_samples + 20:
        raise ValueError(f"not enough paired samples: {len(pairs)}")

    orb0 = average_rotation([p[1] for p in pairs[:zero_samples]])
    vrpn0 = average_rotation([p[3] for p in pairs[:zero_samples]])

    # Estimate X such that A ~= X B X.T, where
    # A = orb0.T @ orb_i, B = vrpn0.T @ vrpn_i.
    src = []
    dst = []
    axes = np.eye(3)
    for _, ro, _, rv in pairs[zero_samples:]:
        a = orb0.T @ ro
        b = vrpn0.T @ rv
        if rotation_angle_deg(a) < min_angle_deg or rotation_angle_deg(b) < min_angle_deg:
            continue
        for k in range(3):
            src.append(b @ axes[:, k])
            dst.append(a @ axes[:, k])

    if len(src) < 30:
        raise ValueError("not enough moving samples; do several left/right bends after zero pose")

    src = np.stack(src)
    dst = np.stack(dst)
    h = src.T @ dst
    u, _, vt = np.linalg.svd(h)
    x = vt.T @ u.T
    if np.linalg.det(x) < 0:
        vt[-1, :] *= -1
        x = vt.T @ u.T

    errors = []
    for _, ro, _, rv in pairs[zero_samples:]:
        a = orb0.T @ ro
        b = vrpn0.T @ rv
        if rotation_angle_deg(a) < min_angle_deg or rotation_angle_deg(b) < min_angle_deg:
            continue
        pred = x @ b @ x.T
        errors.append(rotation_angle_deg(pred.T @ a))
    return x, errors


def main():
    parser = argparse.ArgumentParser(description="Calibrate VRPN rigid-body orientation frame to ORB orientation frame.")
    parser.add_argument("bag_or_run_dir", help="fai/angle run directory or .bag path")
    parser.add_argument("--orb-topic", default="/orb_slam3/odometry")
    parser.add_argument("--vrpn-topic", default="/vrpn_client_node/Rigid2/pose")
    parser.add_argument("--max-dt", type=float, default=0.03)
    parser.add_argument("--zero-samples", type=int, default=80)
    parser.add_argument("--min-angle-deg", type=float, default=5.0)
    args = parser.parse_args()

    if rosbag is None:
        print("Cannot import rosbag. Run this inside ROS Noetic Python environment.", file=sys.stderr)
        return 2

    path = args.bag_or_run_dir
    if os.path.isdir(path):
        candidates = [
            os.path.join(path, "fai_compare.bag"),
            os.path.join(path, "angles.bag"),
            os.path.join(path, "orb_fai.bag"),
        ]
        path = next((p for p in candidates if os.path.exists(p)), "")
    if not path or not os.path.exists(path):
        print("bag not found", file=sys.stderr)
        return 2

    orb, vrpn = read_bag(path, args.orb_topic, args.vrpn_topic)
    pairs = nearest_pairs(orb, vrpn, args.max_dt)
    print(f"bag={path}")
    print(f"orb_samples={len(orb)} vrpn_samples={len(vrpn)} paired={len(pairs)}")

    x, errors = estimate_frame_rotation(pairs, args.zero_samples, args.min_angle_deg)
    print("vrpn_to_orb_rotation_matrix:")
    for row in x:
        print("  [" + ", ".join(f"{v:.9f}" for v in row) + "]")
    print(f"median_rotation_error_deg={float(np.median(errors)):.3f}")
    print(f"mean_rotation_error_deg={float(np.mean(errors)):.3f}")
    print()
    flat = ",".join(f"{v:.9f}" for v in x.reshape(-1))
    print("Use this matrix as VRPN frame alignment:")
    print(flat)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
