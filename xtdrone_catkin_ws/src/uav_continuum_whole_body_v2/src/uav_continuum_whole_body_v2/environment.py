from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import yaml

from .dependencies import import_scipy


scipy = import_scipy()
from scipy.spatial import cKDTree  # noqa: E402


@dataclass(frozen=True)
class BoxObstacle:
    name: str
    center: tuple
    size: tuple
    yaw: float = 0.0

    @property
    def half(self):
        return np.asarray(self.size, dtype=float) * 0.5

    def signed_distance(self, point):
        point = np.asarray(point, dtype=float)
        local = self._local_points(point.reshape((1, 3)))[0]
        q = np.abs(local) - self.half
        outside = float(np.linalg.norm(np.maximum(q, 0.0)))
        inside = float(min(max(q[0], q[1], q[2]), 0.0))
        return outside + inside

    def signed_distances(self, points):
        points = np.asarray(points, dtype=float)
        q = np.abs(self._local_points(points)) - self.half
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(np.max(q, axis=1), 0.0)
        return outside + inside

    def _local_points(self, points):
        points = np.asarray(points, dtype=float).reshape((-1, 3))
        delta = points - np.asarray(self.center, dtype=float)
        cosine, sine = np.cos(self.yaw), np.sin(self.yaw)
        local = delta.copy()
        local[:, 0] = cosine * delta[:, 0] + sine * delta[:, 1]
        local[:, 1] = -sine * delta[:, 0] + cosine * delta[:, 1]
        return local

    @property
    def world_aabb_half(self):
        cosine, sine = abs(np.cos(self.yaw)), abs(np.sin(self.yaw))
        half = self.half
        return np.asarray([
            cosine * half[0] + sine * half[1],
            sine * half[0] + cosine * half[1],
            half[2],
        ])


class BoxWorld:
    def __init__(self, boxes: Iterable[BoxObstacle] = ()):
        self.boxes: List[BoxObstacle] = list(boxes)
        self._centers = np.asarray([box.center for box in self.boxes], dtype=float).reshape((-1, 3))
        self._halves = np.asarray([box.half for box in self.boxes], dtype=float).reshape((-1, 3))
        self._yaws = np.asarray([box.yaw for box in self.boxes], dtype=float)
        self._names = [box.name for box in self.boxes]

    def _distance_matrix(self, points):
        """Exact box SDF for every (box, point), evaluated in one NumPy call."""
        points = np.asarray(points, dtype=float).reshape((-1, 3))
        if not self.boxes or len(points) == 0:
            return np.empty((len(self.boxes), len(points)), dtype=float)
        delta = points[None, :, :] - self._centers[:, None, :]
        cosine = np.cos(self._yaws)[:, None]
        sine = np.sin(self._yaws)[:, None]
        local = delta.copy()
        local[:, :, 0] = cosine * delta[:, :, 0] + sine * delta[:, :, 1]
        local[:, :, 1] = -sine * delta[:, :, 0] + cosine * delta[:, :, 1]
        q = np.abs(local) - self._halves[:, None, :]
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=2)
        inside = np.minimum(np.max(q, axis=2), 0.0)
        return outside + inside

    def signed_distance(self, point):
        if not self.boxes:
            return float("inf"), "none"
        values = self._distance_matrix(np.asarray(point, dtype=float).reshape((1, 3)))[:, 0]
        index = int(np.argmin(values))
        return float(values[index]), self._names[index]

    def signed_distances(self, points):
        points = np.asarray(points, dtype=float).reshape((-1, 3))
        if not self.boxes:
            return np.full(len(points), np.inf)
        return np.min(self._distance_matrix(points), axis=0)

    def minimum_distance(self, points):
        points = np.asarray(points, dtype=float)
        if not self.boxes or len(points) == 0:
            return float("inf"), "none", None
        values = self._distance_matrix(points)
        flat = int(np.argmin(values))
        box_index, point_index = np.unravel_index(flat, values.shape)
        return float(values[box_index, point_index]), self._names[box_index], points[point_index].copy()

    def minimum_distances_grouped(self, point_groups):
        """Return one exact minimum and obstacle name for each fixed-size point group."""
        groups = np.asarray(point_groups, dtype=float)
        if groups.ndim != 3 or groups.shape[2] != 3:
            raise ValueError("point_groups must have shape (G, N, 3)")
        count = len(groups)
        if not self.boxes or groups.shape[1] == 0:
            return np.full(count, np.inf), ["none"] * count
        values = self._distance_matrix(groups.reshape((-1, 3)))
        values = values.reshape((len(self.boxes), count, groups.shape[1]))
        per_box = np.min(values, axis=2)
        box_indices = np.argmin(per_box, axis=0)
        group_indices = np.arange(count)
        return per_box[box_indices, group_indices], [self._names[index] for index in box_indices]

    @classmethod
    def from_dicts(cls, items):
        boxes = []
        for index, item in enumerate(items or []):
            boxes.append(BoxObstacle(
                str(item.get("name", "box_%d" % index)),
                tuple(float(v) for v in item["center"]),
                tuple(float(v) for v in item["size"]),
                float(item.get("yaw", np.deg2rad(item.get("yaw_deg", 0.0)))),
            ))
        return cls(boxes)

    @classmethod
    def from_yaml_files(cls, paths):
        items = []
        for raw in paths:
            path = Path(raw)
            with path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
            if "world_boxes" in data:
                data = data["world_boxes"]
            items.extend(data.get("boxes", []))
        return cls.from_dicts(items)

    def build_esdf(self, minimum, maximum, resolution):
        minimum = np.asarray(minimum, dtype=float)
        maximum = np.asarray(maximum, dtype=float)
        axes = [
            np.arange(minimum[i], maximum[i] + resolution * 0.5, resolution)
            for i in range(3)
        ]
        grid = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
        points = np.column_stack([axis.ravel() for axis in grid])
        distances = np.full(len(points), np.inf, dtype=float)
        for box in self.boxes:
            distances = np.minimum(distances, box.signed_distances(points))
        values = distances.reshape((len(axes[0]), len(axes[1]), len(axes[2]))).astype(np.float32)
        return EsdfGrid(axes[0], axes[1], axes[2], values)

    def filtered(self, minimum, maximum, padding=0.0):
        minimum = np.asarray(minimum, dtype=float) - padding
        maximum = np.asarray(maximum, dtype=float) + padding
        selected = []
        for box in self.boxes:
            center = np.asarray(box.center, dtype=float)
            half = box.world_aabb_half
            if np.all(center + half >= minimum) and np.all(center - half <= maximum):
                selected.append(box)
        return BoxWorld(selected)


class EsdfGrid:
    def __init__(self, x, y, z, values):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.z = np.asarray(z, dtype=float)
        self.values = np.asarray(values, dtype=float)
        expected = (len(self.x), len(self.y), len(self.z))
        if self.values.shape != expected:
            raise ValueError("ESDF shape %r does not match axes %r" % (self.values.shape, expected))

    @property
    def resolution(self):
        return float(min(np.min(np.diff(self.x)), np.min(np.diff(self.y)), np.min(np.diff(self.z))))

    def _cell(self, axis, value):
        if value <= axis[0]:
            return 0, 0.0
        if value >= axis[-1]:
            return len(axis) - 2, 1.0
        idx = int(np.searchsorted(axis, value) - 1)
        ratio = (value - axis[idx]) / (axis[idx + 1] - axis[idx])
        return idx, float(ratio)

    def signed_distance(self, point):
        ix, rx = self._cell(self.x, float(point[0]))
        iy, ry = self._cell(self.y, float(point[1]))
        iz, rz = self._cell(self.z, float(point[2]))
        value = 0.0
        for dx in (0, 1):
            wx = rx if dx else 1.0 - rx
            for dy in (0, 1):
                wy = ry if dy else 1.0 - ry
                for dz in (0, 1):
                    wz = rz if dz else 1.0 - rz
                    value += wx * wy * wz * self.values[ix + dx, iy + dy, iz + dz]
        return float(value), "esdf"

    def signed_distances(self, points):
        points = np.asarray(points, dtype=float)
        indices = []
        ratios = []
        for axis_index, axis in enumerate((self.x, self.y, self.z)):
            values = np.clip(points[:, axis_index], axis[0], axis[-1])
            idx = np.searchsorted(axis, values, side="right") - 1
            idx = np.clip(idx, 0, len(axis) - 2)
            ratio = (values - axis[idx]) / (axis[idx + 1] - axis[idx])
            indices.append(idx)
            ratios.append(ratio)
        result = np.zeros(len(points), dtype=float)
        for dx in (0, 1):
            wx = ratios[0] if dx else 1.0 - ratios[0]
            for dy in (0, 1):
                wy = ratios[1] if dy else 1.0 - ratios[1]
                for dz in (0, 1):
                    wz = ratios[2] if dz else 1.0 - ratios[2]
                    result += wx * wy * wz * self.values[indices[0] + dx, indices[1] + dy, indices[2] + dz]
        return result

    def gradient(self, point):
        h = self.resolution * 0.5
        point = np.asarray(point, dtype=float)
        grad = []
        for axis in range(3):
            a = point.copy()
            b = point.copy()
            a[axis] -= h
            b[axis] += h
            grad.append((self.signed_distance(b)[0] - self.signed_distance(a)[0]) / (2.0 * h))
        return np.asarray(grad)

    def save(self, path, metadata=None):
        np.savez_compressed(
            str(path), x=self.x, y=self.y, z=self.z, values=self.values,
            metadata=np.asarray([metadata or {}], dtype=object),
        )

    @classmethod
    def load(cls, path):
        data = np.load(str(path), allow_pickle=True)
        return cls(data["x"], data["y"], data["z"], data["values"])


class PointCloudWorld:
    """Nearest-surface distance model for online point-cloud obstacles."""

    def __init__(self, points, inflation=0.02):
        self.points = np.asarray(points, dtype=float).reshape((-1, 3))
        self.inflation = float(inflation)
        self.tree = cKDTree(self.points) if len(self.points) else None

    def signed_distance(self, point):
        if self.tree is None:
            return float("inf"), "point_cloud"
        distance, _ = self.tree.query(np.asarray(point, dtype=float), k=1)
        return float(distance) - self.inflation, "point_cloud"

    def minimum_distance(self, points):
        points = np.asarray(points, dtype=float)
        if self.tree is None or len(points) == 0:
            return float("inf"), "point_cloud", None
        distances, _ = self.tree.query(points, k=1)
        index = int(np.argmin(distances))
        return float(distances[index]) - self.inflation, "point_cloud", points[index].copy()

    def filtered(self, minimum, maximum, padding=0.0):
        """Build a local tree from points inside a padded axis-aligned ROI.

        Callers must choose padding larger than the maximum query-geometry
        reach plus the clearance of interest.  Then an excluded point cannot
        be the nearest safety-critical obstacle for any query in the ROI.
        """
        minimum = np.asarray(minimum, dtype=float) - float(padding)
        maximum = np.asarray(maximum, dtype=float) + float(padding)
        if not len(self.points):
            return PointCloudWorld([], self.inflation)
        mask = np.all((self.points >= minimum) & (self.points <= maximum), axis=1)
        return PointCloudWorld(self.points[mask], self.inflation)


class HybridDistanceWorld:
    """Use the conservative minimum of exact primitives and an ESDF."""
    def __init__(self, box_world=None, esdf=None, point_world=None):
        self.box_world = box_world
        self.esdf = esdf
        self.point_world = point_world

    def signed_distance(self, point):
        candidates = []
        if self.box_world is not None:
            candidates.append(self.box_world.signed_distance(point))
        if self.esdf is not None:
            candidates.append(self.esdf.signed_distance(point))
        if self.point_world is not None:
            candidates.append(self.point_world.signed_distance(point))
        if not candidates:
            return float("inf"), "none"
        return min(candidates, key=lambda item: item[0])

    def minimum_distance(self, points):
        points = np.asarray(points, dtype=float)
        best = (float("inf"), "none", None)
        if self.box_world is not None:
            best = self.box_world.minimum_distance(points)
        if self.esdf is not None:
            values = self.esdf.signed_distances(points)
            index = int(np.argmin(values))
            if values[index] < best[0]:
                best = (float(values[index]), "esdf", points[index].copy())
        if self.point_world is not None:
            candidate = self.point_world.minimum_distance(points)
            if candidate[0] < best[0]:
                best = candidate
        return best

    def minimum_distances_grouped(self, point_groups):
        """Vectorized grouped minimum used by the independent dense validator."""
        groups = np.asarray(point_groups, dtype=float)
        if groups.ndim != 3 or groups.shape[2] != 3:
            raise ValueError("point_groups must have shape (G, N, 3)")
        count, points_per_group = groups.shape[:2]
        best = np.full(count, np.inf, dtype=float)
        names = np.asarray(["none"] * count, dtype=object)
        if self.box_world is not None:
            values, sources = self.box_world.minimum_distances_grouped(groups)
            mask = values < best
            best[mask] = values[mask]
            names[mask] = np.asarray(sources, dtype=object)[mask]
        flat = groups.reshape((-1, 3))
        if self.esdf is not None and len(flat):
            values = self.esdf.signed_distances(flat).reshape((count, points_per_group)).min(axis=1)
            mask = values < best
            best[mask] = values[mask]
            names[mask] = "esdf"
        if self.point_world is not None and self.point_world.tree is not None and len(flat):
            # Large dense swept-volume batches are independent nearest-neighbour
            # queries.  Use SciPy's native worker pool on the target CPU; this
            # changes neither points nor distances.  Retain compatibility with
            # older SciPy builds that predate the workers argument.
            try:
                values, _ = self.point_world.tree.query(
                    flat, k=1, workers=-1 if len(flat) >= 4096 else 1
                )
            except TypeError:
                try:
                    values, _ = self.point_world.tree.query(
                        flat, k=1, n_jobs=-1 if len(flat) >= 4096 else 1
                    )
                except TypeError:
                    values, _ = self.point_world.tree.query(flat, k=1)
            values = values.reshape((count, points_per_group)).min(axis=1) - self.point_world.inflation
            mask = values < best
            best[mask] = values[mask]
            names[mask] = "point_cloud"
        return best, names.tolist()
