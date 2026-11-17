import math

import numpy as np
import rospy
import tf.transformations as tft
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker


def get_nested_param(path, default=None):
    if rospy.has_param(path):
        return rospy.get_param(path)
    return default


def quaternion_msg(q):
    return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])


def point_msg(x, y, z):
    return Point(x=float(x), y=float(y), z=float(z))


def color(r, g, b, a=1.0):
    return ColorRGBA(float(r), float(g), float(b), float(a))


def pose_to_matrix(pose):
    q = [
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ]
    mat = tft.quaternion_matrix(q)
    mat[0, 3] = pose.position.x
    mat[1, 3] = pose.position.y
    mat[2, 3] = pose.position.z
    return mat


def transform_point(mat, xyz):
    x, y, z = xyz
    out = mat.dot([x, y, z, 1.0])
    return (float(out[0]), float(out[1]), float(out[2]))


def mount_matrix(mount_xyz, mount_rpy):
    mat = tft.euler_matrix(mount_rpy[0], mount_rpy[1], mount_rpy[2])
    mat[0, 3] = mount_xyz[0]
    mat[1, 3] = mount_xyz[1]
    mat[2, 3] = mount_xyz[2]
    return mat


def arm_points(theta_rad, phi_rad, length, segment_count, base_offset_z, gripper_length=0.0):
    pts = []
    n = max(int(segment_count), 2)
    base = max(float(base_offset_z), 0.0)
    if base > 1e-6:
        base_segments = max(2, int(math.ceil(base / max(length / float(n), 1e-3))))
        for i in range(base_segments + 1):
            z = base * float(i) / float(base_segments)
            pts.append((0.0, 0.0, z))
    for i in range(n + 1):
        s = length * float(i) / float(n)
        if abs(theta_rad) < 1e-6:
            x = 0.0
            y = 0.0
            z = base + s
        else:
            k = theta_rad / length
            radius = 1.0 / k
            radial = radius * (1.0 - math.cos(k * s))
            x = radial * math.cos(phi_rad)
            y = radial * math.sin(phi_rad)
            z = base + radius * math.sin(k * s)
        if pts and point_distance(pts[-1], (x, y, z)) < 1e-9:
            continue
        pts.append((x, y, z))
    grip = max(float(gripper_length), 0.0)
    if grip > 1e-6:
        if abs(theta_rad) < 1e-6:
            tangent = (0.0, 0.0, 1.0)
        else:
            tangent = (
                math.sin(theta_rad) * math.cos(phi_rad),
                math.sin(theta_rad) * math.sin(phi_rad),
                math.cos(theta_rad),
            )
        tip = pts[-1]
        grip_segments = max(2, int(math.ceil(grip / max(length / float(n), 1e-3))))
        for i in range(1, grip_segments + 1):
            d = grip * float(i) / float(grip_segments)
            pts.append((
                tip[0] + tangent[0] * d,
                tip[1] + tangent[1] * d,
                tip[2] + tangent[2] * d,
            ))
    return pts


def world_arm_points(uav_pose, theta_deg, phi_deg, arm_params):
    body = pose_to_matrix(uav_pose)
    mount = mount_matrix(arm_params["mount_xyz"], arm_params["mount_rpy"])
    base = body.dot(mount)
    local_pts = arm_points(
        math.radians(theta_deg),
        math.radians(phi_deg),
        arm_params["arc_length"],
        arm_params["segment_count"],
        arm_params["base_offset_z"],
        arm_params.get("gripper_length", 0.0),
    )
    return [transform_point(base, p) for p in local_pts]


def make_pose_stamped(frame_id, stamp, xyz, quat=None):
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.pose.position = point_msg(*xyz)
    msg.pose.orientation = quaternion_msg(quat or [0.0, 0.0, 0.0, 1.0])
    return msg


def marker_header(marker, frame_id, stamp, ns, marker_id, marker_type):
    marker.header.frame_id = frame_id
    marker.header.stamp = stamp
    marker.ns = ns
    marker.id = marker_id
    marker.type = marker_type
    marker.action = Marker.ADD


def point_distance(a, b):
    return math.sqrt(
        (a[0] - b[0]) * (a[0] - b[0])
        + (a[1] - b[1]) * (a[1] - b[1])
        + (a[2] - b[2]) * (a[2] - b[2])
    )


class BoxObstacle:
    def __init__(self, name, center, size):
        self.name = name
        self.center = tuple(float(v) for v in center)
        self.size = tuple(float(v) for v in size)
        self.half = tuple(v * 0.5 for v in self.size)

    def signed_distance(self, point):
        dx = abs(point[0] - self.center[0]) - self.half[0]
        dy = abs(point[1] - self.center[1]) - self.half[1]
        dz = abs(point[2] - self.center[2]) - self.half[2]
        ox = max(dx, 0.0)
        oy = max(dy, 0.0)
        oz = max(dz, 0.0)
        outside = math.sqrt(ox * ox + oy * oy + oz * oz)
        inside = min(max(dx, max(dy, dz)), 0.0)
        return outside + inside

    def marker(self, frame_id, stamp, marker_id, rgba):
        marker = Marker()
        marker_header(marker, frame_id, stamp, "known_obstacles", marker_id, Marker.CUBE)
        marker.pose.position = point_msg(*self.center)
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.size[0]
        marker.scale.y = self.size[1]
        marker.scale.z = self.size[2]
        marker.color = rgba
        return marker


class VoxelOccupancy:
    def __init__(
        self,
        resolution,
        inflation_radius=0.0,
        max_points=250000,
        build_distance_field=False,
        distance_field_padding=0.75,
        max_distance_field_cells=8000000,
        distance_field_min=None,
        distance_field_max=None,
    ):
        self.resolution = max(float(resolution), 1e-3)
        self.inflation_radius = max(float(inflation_radius), 0.0)
        self.max_points = max(int(max_points), 1)
        self.build_distance_field = bool(build_distance_field)
        self.distance_field_padding = max(float(distance_field_padding), 0.0)
        self.max_distance_field_cells = max(int(max_distance_field_cells), 1)
        self.distance_field_min = tuple(float(v) for v in distance_field_min) if distance_field_min is not None else None
        self.distance_field_max = tuple(float(v) for v in distance_field_max) if distance_field_max is not None else None
        self.occupied = set()
        self.distance_cache = {}
        self.distance_field = None
        self.distance_field_origin = None
        self.distance_field_shape = None
        self.distance_field_status = "disabled"
        self.source_points = 0
        self.raw_source_points = 0
        self.frame_id = ""
        self.stamp = rospy.Time(0)

    def _key(self, point):
        return (
            int(round(float(point[0]) / self.resolution)),
            int(round(float(point[1]) / self.resolution)),
            int(round(float(point[2]) / self.resolution)),
        )

    def _point_from_key(self, key):
        return (
            key[0] * self.resolution,
            key[1] * self.resolution,
            key[2] * self.resolution,
        )

    def update(self, points, frame_id="", stamp=None):
        base = []
        seen = set()
        raw_count = 0
        for point in points:
            raw_count += 1
            if self.distance_field_min is not None and self.distance_field_max is not None:
                if (
                    point[0] < self.distance_field_min[0]
                    or point[0] > self.distance_field_max[0]
                    or point[1] < self.distance_field_min[1]
                    or point[1] > self.distance_field_max[1]
                    or point[2] < self.distance_field_min[2]
                    or point[2] > self.distance_field_max[2]
                ):
                    continue
            key = self._key(point)
            if key in seen:
                continue
            seen.add(key)
            base.append(key)
            if len(base) >= self.max_points:
                break

        inflate_steps = int(math.ceil(self.inflation_radius / self.resolution))
        occupied = set(base)
        if inflate_steps > 0:
            max_dist2 = (self.inflation_radius + self.resolution * 0.5) ** 2
            for key in base:
                for dx in range(-inflate_steps, inflate_steps + 1):
                    for dy in range(-inflate_steps, inflate_steps + 1):
                        for dz in range(-inflate_steps, inflate_steps + 1):
                            dist2 = (dx * self.resolution) ** 2 + (dy * self.resolution) ** 2 + (dz * self.resolution) ** 2
                            if dist2 <= max_dist2:
                                occupied.add((key[0] + dx, key[1] + dy, key[2] + dz))

        self.occupied = occupied
        self.distance_cache = {}
        self._build_esdf()
        self.source_points = len(base)
        self.raw_source_points = raw_count
        self.frame_id = frame_id
        if stamp is not None:
            self.stamp = stamp
        else:
            try:
                self.stamp = rospy.Time.now()
            except rospy.exceptions.ROSInitException:
                self.stamp = rospy.Time(0)

    def ready(self):
        return bool(self.occupied)

    def _edt_1d(self, values):
        n = int(values.shape[0])
        out = np.empty(n, dtype=np.float64)
        v = np.zeros(n, dtype=np.int32)
        z = np.empty(n + 1, dtype=np.float64)
        k = 0
        v[0] = 0
        z[0] = -float("inf")
        z[1] = float("inf")
        for q in range(1, n):
            while True:
                p = int(v[k])
                denom = 2.0 * float(q - p)
                if abs(denom) < 1e-12:
                    s = float("inf")
                else:
                    s = ((float(values[q]) + q * q) - (float(values[p]) + p * p)) / denom
                if s > z[k]:
                    break
                k -= 1
                if k < 0:
                    k = 0
                    break
            k += 1
            v[k] = q
            z[k] = s
            z[k + 1] = float("inf")
        k = 0
        for q in range(n):
            while z[k + 1] < q:
                k += 1
            p = int(v[k])
            out[q] = (q - p) * (q - p) + values[p]
        return out

    def _edt_axis(self, arr, axis):
        moved = np.moveaxis(arr, axis, 0)
        out = np.empty_like(moved, dtype=np.float64)
        for idx in np.ndindex(moved.shape[1:]):
            out[(slice(None),) + idx] = self._edt_1d(moved[(slice(None),) + idx])
        return np.moveaxis(out, 0, axis)

    def _build_esdf(self):
        self.distance_field = None
        self.distance_field_origin = None
        self.distance_field_shape = None
        if not self.build_distance_field:
            self.distance_field_status = "disabled"
            return
        if not self.occupied:
            self.distance_field_status = "empty"
            return
        padding = int(math.ceil(self.distance_field_padding / self.resolution))
        if self.distance_field_min is not None and self.distance_field_max is not None:
            min_key = tuple(int(math.floor(self.distance_field_min[i] / self.resolution)) - padding for i in range(3))
            max_key = tuple(int(math.ceil(self.distance_field_max[i] / self.resolution)) + padding for i in range(3))
            keys = [
                key for key in self.occupied
                if min_key[0] <= key[0] <= max_key[0]
                and min_key[1] <= key[1] <= max_key[1]
                and min_key[2] <= key[2] <= max_key[2]
            ]
        else:
            keys = list(self.occupied)
            min_key = tuple(min(key[i] for key in keys) - padding for i in range(3))
            max_key = tuple(max(key[i] for key in keys) + padding for i in range(3))
        if not keys:
            self.distance_field_status = "empty_in_bounds"
            return
        shape = tuple(max_key[i] - min_key[i] + 1 for i in range(3))
        cells = int(shape[0] * shape[1] * shape[2])
        if cells > self.max_distance_field_cells:
            self.distance_field_status = "too_large:%d" % cells
            return

        occ = np.zeros(shape, dtype=bool)
        for key in keys:
            occ[key[0] - min_key[0], key[1] - min_key[1], key[2] - min_key[2]] = True
        inf = float(shape[0] * shape[0] + shape[1] * shape[1] + shape[2] * shape[2] + 1)
        dist2 = np.where(occ, 0.0, inf).astype(np.float64)
        dist2 = self._edt_axis(dist2, 0)
        dist2 = self._edt_axis(dist2, 1)
        dist2 = self._edt_axis(dist2, 2)
        self.distance_field = np.sqrt(dist2).astype(np.float32) * self.resolution
        self.distance_field_origin = min_key
        self.distance_field_shape = shape
        self.distance_field_status = "ready:%dx%dx%d" % shape

    def signed_distance(self, point, search_radius):
        if not self.occupied:
            return float("inf"), "static_cloud:none", point
        key = self._key(point)
        if self.distance_field is not None and self.distance_field_origin is not None:
            ix = key[0] - self.distance_field_origin[0]
            iy = key[1] - self.distance_field_origin[1]
            iz = key[2] - self.distance_field_origin[2]
            shape = self.distance_field_shape
            if 0 <= ix < shape[0] and 0 <= iy < shape[1] and 0 <= iz < shape[2]:
                dist = float(self.distance_field[ix, iy, iz]) - self.resolution * 0.5
                return dist, "static_scene_esdf", point
        search_steps = max(1, int(math.ceil(float(search_radius) / self.resolution)))
        cache_key = (key, search_steps)
        cached = self.distance_cache.get(cache_key)
        if cached is not None:
            return cached
        best_dist = float("inf")
        best_key = None
        for dx in range(-search_steps, search_steps + 1):
            for dy in range(-search_steps, search_steps + 1):
                for dz in range(-search_steps, search_steps + 1):
                    candidate = (key[0] + dx, key[1] + dy, key[2] + dz)
                    if candidate not in self.occupied:
                        continue
                    center = self._point_from_key(candidate)
                    dist = point_distance(point, center) - self.resolution * 0.5
                    if dist < best_dist:
                        best_dist = dist
                        best_key = candidate
        if best_key is None:
            result = (float("inf"), "static_cloud:none", point)
        else:
            result = (best_dist, "static_cloud", self._point_from_key(best_key))
        if len(self.distance_cache) < 200000:
            self.distance_cache[cache_key] = result
        return result

    def report(self):
        age = float("inf")
        if self.stamp and self.stamp != rospy.Time(0):
            age = max(0.0, (rospy.Time.now() - self.stamp).to_sec())
        bounds = ""
        if self.distance_field_min is not None and self.distance_field_max is not None:
            bounds = " bounds=[%.2f,%.2f,%.2f]->[%.2f,%.2f,%.2f]" % (
                self.distance_field_min[0],
                self.distance_field_min[1],
                self.distance_field_min[2],
                self.distance_field_max[0],
                self.distance_field_max[1],
                self.distance_field_max[2],
            )
        return (
            "ready=%s frame=%s resolution=%.3f inflation=%.3f raw_points=%d source_voxels=%d occupied_voxels=%d age=%.3f"
            % (
                str(self.ready()),
                self.frame_id or "unknown",
                self.resolution,
                self.inflation_radius,
                self.raw_source_points,
                self.source_points,
                len(self.occupied),
                age,
            )
        ) + " esdf=%s%s" % (self.distance_field_status, bounds)


def load_arm_params():
    arm = get_nested_param("/arm", {})
    return {
        "arc_length": float(arm.get("arc_length", 0.28)),
        "radius": float(arm.get("radius", 0.025)),
        "segment_count": int(arm.get("segment_count", 18)),
        "base_offset_z": float(arm.get("base_offset_z", 0.0704)),
        "gripper_length": float(arm.get("gripper_length", 0.05)),
        "mount_xyz": [float(v) for v in arm.get("mount_xyz", [0.0, 0.0, -0.03])],
        "mount_rpy": [float(v) for v in arm.get("mount_rpy", [0.0, math.pi, 0.0])],
        "theta_min_deg": float(arm.get("theta_min_deg", 0.0)),
        "theta_max_deg": float(arm.get("theta_max_deg", 90.0)),
        "theta_step_deg": float(arm.get("theta_step_deg", 5.0)),
        "phi_min_deg": float(arm.get("phi_min_deg", -180.0)),
        "phi_max_deg": float(arm.get("phi_max_deg", 180.0)),
        "phi_step_deg": float(arm.get("phi_step_deg", 10.0)),
    }


def load_boxes():
    items = load_world_boxes_raw() + load_task_boxes_raw()
    return [
        BoxObstacle(item["name"], item["center"], item["size"])
        for item in items
    ]


def load_world_boxes_raw():
    return list(get_nested_param("/world_boxes/boxes", []))


def load_task_boxes_raw():
    return list(get_nested_param("/boxes", []))


def boxes_from_items(items):
    return [
        BoxObstacle(item["name"], item["center"], item["size"])
        for item in items
    ]


def load_body_keepout_boxes():
    return [
        BoxObstacle(item["name"], item["center"], item["size"])
        for item in get_nested_param("/body_keepout_boxes", [])
    ]


def load_target():
    target = get_nested_param("/target", {})
    pose = target.get("pose", [0.0, 0.0, 1.0, 0.0])
    return {
        "name": target.get("name", "target"),
        "pose": [float(v) for v in pose],
    }


def frange(start, stop, step):
    vals = []
    if step <= 0.0:
        return vals
    value = start
    while value <= stop + 1e-9:
        vals.append(value)
        value += step
    return vals
