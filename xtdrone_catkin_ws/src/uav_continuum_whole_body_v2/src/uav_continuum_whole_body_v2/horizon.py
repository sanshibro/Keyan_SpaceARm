import math

import numpy as np

from .trajectory import cut_path_by_distance
from .types import PHI, THETA, YAW


def yaw_from_quaternion(quaternion):
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
    )


def path_to_states(path, default_shape=(0.0, 0.0)):
    states = []
    for item in path.poses:
        pose = item.pose
        states.append([
            pose.position.x, pose.position.y, pose.position.z,
            yaw_from_quaternion(pose.orientation), default_shape[0], default_shape[1],
        ])
    states = np.asarray(states, dtype=float)
    if len(states) > 1:
        for index in range(len(states) - 1):
            delta = states[index + 1, :2] - states[index, :2]
            if np.linalg.norm(delta) > 1e-5:
                states[index, YAW] = math.atan2(delta[1], delta[0])
        states[-1, YAW] = states[-2, YAW]
    return states


def local_horizon(reference_states, current_state, distance):
    reference = np.asarray(reference_states, dtype=float)
    current = np.asarray(current_state, dtype=float)
    if len(reference) < 2:
        raise ValueError("reference must contain at least two states")
    nearest = int(np.argmin(np.linalg.norm(reference[:, :3] - current[:3], axis=1)))
    tail = reference[nearest:].copy()
    if len(tail) < 2:
        tail = reference[-2:].copy()
    combined = np.vstack((current, tail))
    filtered = [combined[0]]
    for state in combined[1:]:
        if np.linalg.norm(state[:3] - filtered[-1][:3]) > 1e-4:
            filtered.append(state)
    if len(filtered) < 2:
        raise ValueError("reference horizon has no forward motion")
    result = cut_path_by_distance(np.asarray(filtered), distance)
    result[:, YAW] = np.unwrap(result[:, YAW])
    result[:, PHI] = np.unwrap(result[:, PHI])
    result[0, [THETA, PHI]] = current[[THETA, PHI]]
    return result, nearest
