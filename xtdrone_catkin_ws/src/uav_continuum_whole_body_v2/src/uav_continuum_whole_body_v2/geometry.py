import math

import numpy as np


def rotation_matrix_rpy(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz.dot(ry).dot(rx)


def wrap_angle(value):
    return math.atan2(math.sin(value), math.cos(value))


def unwrap_angles(values):
    return np.unwrap(np.asarray(values, dtype=float))


def local_arm_sections(theta, phi, model):
    """Return local base, flexible and tool points before mount transform."""
    base_count = max(2, int(math.ceil(model.base_offset_z / max(model.arc_length / model.segment_count, 1e-3))))
    base = np.array([
        [0.0, 0.0, model.base_offset_z * i / float(base_count)]
        for i in range(base_count + 1)
    ], dtype=float)

    flexible = []
    for i in range(model.segment_count + 1):
        s = model.arc_length * i / float(model.segment_count)
        if abs(theta) < 1e-8:
            point = [0.0, 0.0, model.base_offset_z + s]
        else:
            kappa = theta / model.arc_length
            radius = 1.0 / kappa
            radial = radius * (1.0 - math.cos(kappa * s))
            point = [
                radial * math.cos(phi),
                radial * math.sin(phi),
                model.base_offset_z + radius * math.sin(kappa * s),
            ]
        flexible.append(point)
    flexible = np.asarray(flexible, dtype=float)

    if abs(theta) < 1e-8:
        tangent = np.array([0.0, 0.0, 1.0])
    else:
        tangent = np.array([
            math.sin(theta) * math.cos(phi),
            math.sin(theta) * math.sin(phi),
            math.cos(theta),
        ])
    tool_count = max(2, int(math.ceil(model.gripper_length / max(model.arc_length / model.segment_count, 1e-3))))
    tip = flexible[-1]
    tool = np.array([
        tip + tangent * (model.gripper_length * i / float(tool_count))
        for i in range(1, tool_count + 1)
    ], dtype=float)
    return base, flexible, tool


def world_arm_sections(state, model):
    position = np.asarray(state[:3], dtype=float)
    yaw = float(state[3])
    theta = float(state[4])
    phi = float(state[5])
    base, flexible, tool = local_arm_sections(theta, phi, model)
    mount_r = rotation_matrix_rpy(*model.mount_rpy)
    yaw_r = rotation_matrix_rpy(0.0, 0.0, yaw)
    rotation = yaw_r.dot(mount_r)
    mount = position + yaw_r.dot(np.asarray(model.mount_xyz, dtype=float))

    def transform(points):
        return points.dot(rotation.T) + mount

    return transform(base), transform(flexible), transform(tool)


def world_arm_points(state, model):
    sections = world_arm_sections(state, model)
    points = np.vstack(sections)
    if len(getattr(model, 'extra_tool_points', ())):
        # Existing terminal objectives use [-1] as the PCC tool tip.
        extra = world_extra_tool_points_batch(np.asarray(state).reshape(1, 6), model)[0]
        points = np.vstack((sections[0], sections[1], extra, sections[2]))
    return points


def world_arm_sections_batch(states, model):
    """Vectorized equivalent of :func:`world_arm_sections`.

    The comparison planners evaluate thousands of swept-volume states at a
    time.  Keeping the exact PCC geometry while batching the trigonometry
    avoids making Python object creation part of a method's planning budget.
    """
    states = np.asarray(states, dtype=float).reshape((-1, 6))
    count = len(states)
    theta = states[:, 4]
    phi = states[:, 5]

    spacing = max(model.arc_length / model.segment_count, 1e-3)
    base_count = max(2, int(math.ceil(model.base_offset_z / spacing)))
    base_z = np.linspace(0.0, model.base_offset_z, base_count + 1)
    base = np.zeros((count, len(base_z), 3), dtype=float)
    base[:, :, 2] = base_z[None, :]

    s = np.linspace(0.0, model.arc_length, model.segment_count + 1)
    angle = theta[:, None] * s[None, :] / model.arc_length
    straight = np.abs(theta) < 1e-8
    safe_theta = np.where(straight, 1.0, theta)
    radius = model.arc_length / safe_theta
    radial = radius[:, None] * (1.0 - np.cos(angle))
    height = model.base_offset_z + radius[:, None] * np.sin(angle)
    radial[straight, :] = 0.0
    height[straight, :] = model.base_offset_z + s[None, :]
    flexible = np.empty((count, len(s), 3), dtype=float)
    flexible[:, :, 0] = radial * np.cos(phi)[:, None]
    flexible[:, :, 1] = radial * np.sin(phi)[:, None]
    flexible[:, :, 2] = height

    tangent = np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])
    tool_count = max(2, int(math.ceil(model.gripper_length / spacing)))
    tool_s = model.gripper_length * np.arange(1, tool_count + 1) / float(tool_count)
    tool = flexible[:, -1:, :] + tangent[:, None, :] * tool_s[None, :, None]

    mount_r = rotation_matrix_rpy(*model.mount_rpy)
    yaw = states[:, 3]
    cosine, sine = np.cos(yaw), np.sin(yaw)
    mount_local = np.asarray(model.mount_xyz, dtype=float)
    mount = states[:, :3].copy()
    mount[:, 0] += cosine * mount_local[0] - sine * mount_local[1]
    mount[:, 1] += sine * mount_local[0] + cosine * mount_local[1]
    mount[:, 2] += mount_local[2]

    def transform(points):
        mounted = np.einsum("nij,kj->nik", points, mount_r)
        transformed = mounted.copy()
        transformed[:, :, 0] = (
            cosine[:, None] * mounted[:, :, 0]
            - sine[:, None] * mounted[:, :, 1]
        )
        transformed[:, :, 1] = (
            sine[:, None] * mounted[:, :, 0]
            + cosine[:, None] * mounted[:, :, 1]
        )
        return transformed + mount[:, None, :]

    return transform(base), transform(flexible), transform(tool)


def world_extra_tool_points_batch(states, model):
    states = np.asarray(states, dtype=float).reshape((-1, 6))
    local = np.asarray(model.extra_tool_points, dtype=float).reshape((-1, 3))
    roots = world_arm_sections_batch(states, model)[1][:, -1, :]
    output = []
    for state, root in zip(states, roots):
        theta, phi = state[4:6]
        axis = np.array([-np.sin(phi), np.cos(phi), 0.])
        x, y, z = axis
        cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
        bend = np.eye(3) + np.sin(theta)*cross + (1.-np.cos(theta))*(cross@cross)
        rotation = rotation_matrix_rpy(0., 0., state[3]) @ rotation_matrix_rpy(*model.mount_rpy) @ bend
        output.append(local @ rotation.T + root)
    return np.asarray(output)


def world_arm_points_batch(states, model):
    sections = world_arm_sections_batch(states, model)
    points = np.concatenate(sections, axis=1)
    if len(getattr(model, 'extra_tool_points', ())):
        points = np.concatenate((sections[0], sections[1], world_extra_tool_points_batch(states, model), sections[2]), axis=1)
    return points


def world_body_points(state, limits, azimuth_samples=16, vertical_samples=3):
    """Sample the cylindrical UAV envelope without treating its height as its radius."""
    center = np.asarray(state[:3], dtype=float)
    yaw = float(state[3])
    half_height = 0.5 * limits.body_height
    points = [center, center + [0.0, 0.0, half_height], center - [0.0, 0.0, half_height]]
    for z in np.linspace(-half_height, half_height, max(2, int(vertical_samples))):
        for index in range(azimuth_samples):
            angle = yaw + 2.0 * math.pi * index / float(azimuth_samples)
            points.append(center + [limits.body_radius * math.cos(angle), limits.body_radius * math.sin(angle), z])
    return np.asarray(points, dtype=float)


def world_body_points_batch(states, limits, azimuth_samples=16, vertical_samples=3):
    """Vectorized equivalent of :func:`world_body_points`."""
    states = np.asarray(states, dtype=float).reshape((-1, 6))
    count = len(states)
    center = states[:, :3]
    half_height = 0.5 * limits.body_height
    core = np.repeat(center[:, None, :], 3, axis=1)
    core[:, 1, 2] += half_height
    core[:, 2, 2] -= half_height

    z_values = np.linspace(-half_height, half_height, max(2, int(vertical_samples)))
    azimuth = 2.0 * math.pi * np.arange(azimuth_samples) / float(azimuth_samples)
    angles = states[:, 3, None] + azimuth[None, :]
    ring_xy = limits.body_radius * np.stack([np.cos(angles), np.sin(angles)], axis=2)
    rings = np.repeat(ring_xy[:, None, :, :], len(z_values), axis=1)
    ring_points = np.empty((count, len(z_values), azimuth_samples, 3), dtype=float)
    ring_points[:, :, :, :2] = rings + center[:, None, None, :2]
    ring_points[:, :, :, 2] = center[:, None, None, 2] + z_values[None, :, None]
    return np.concatenate([core, ring_points.reshape((count, -1, 3))], axis=1)


def coupling_metrics(state, arm_model, mass_model):
    base, flexible, tool = world_arm_sections(state, arm_model)
    uav = np.asarray(state[:3], dtype=float)
    base_com = np.mean(base, axis=0)
    flexible_com = np.mean(flexible, axis=0)
    tool_com = np.mean(tool, axis=0) if len(tool) else flexible[-1]
    system = (
        mass_model.uav * uav
        + mass_model.arm_base * base_com
        + mass_model.flexible * flexible_com
        + mass_model.tool * tool_com
    ) / mass_model.total
    offset = float(np.linalg.norm((system - uav)[:2]))
    torque = mass_model.total * 9.81 * offset
    return {
        "system_com": system,
        "com_offset": offset,
        "gravity_torque": torque,
        "base_com": base_com,
        "flexible_com": flexible_com,
        "tool_com": tool_com,
    }


def coupling_metrics_batch(states, arm_model, mass_model):
    """Return the COM offsets for a batch using the exact scalar mass model."""
    states = np.asarray(states, dtype=float).reshape((-1, 6))
    base, flexible, tool = world_arm_sections_batch(states, arm_model)
    uav = states[:, :3]
    base_com = np.mean(base, axis=1)
    flexible_com = np.mean(flexible, axis=1)
    tool_com = np.mean(tool, axis=1) if tool.shape[1] else flexible[:, -1, :]
    system = (
        mass_model.uav * uav
        + mass_model.arm_base * base_com
        + mass_model.flexible * flexible_com
        + mass_model.tool * tool_com
    ) / mass_model.total
    return np.linalg.norm((system - uav)[:, :2], axis=1)


def interpolate_state(a, b, ratio):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = (1.0 - ratio) * a + ratio * b
    out[3] = a[3] + wrap_angle(b[3] - a[3]) * ratio
    out[5] = a[5] + wrap_angle(b[5] - a[5]) * ratio
    return out
