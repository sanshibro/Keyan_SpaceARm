"""SLAM-map snapshots and collision queries shared by every experiment method.

The planning stack deliberately does not know about Gazebo primitives here.
Analytic truth geometry belongs to the offline oracle evaluator only.
"""

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Dict, Iterable, Optional

import numpy as np

from .dependencies import import_scipy


scipy = import_scipy()
from scipy.spatial import cKDTree  # noqa: E402


MAP_MODES = ("regression_exact", "frozen_slam", "online_slam")


def validate_map_mode(value):
    value = str(value).strip().lower()
    if value not in MAP_MODES:
        raise ValueError("map_mode must be one of %s, got %r" % (MAP_MODES, value))
    return value


def _points(values):
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return np.empty((0, 3), dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
        raise ValueError("map points must be a finite array with shape (N, 3)")
    return array


def voxel_indices(points, resolution):
    return np.floor(_points(points) / float(resolution) + 1e-9).astype(np.int64)


def voxel_centers(indices, resolution):
    indices = np.asarray(indices, dtype=np.int64).reshape((-1, 3))
    return (indices.astype(float) + 0.5) * float(resolution)


def transform_points(points, translation, quaternion_xyzw):
    """Transform sensor-frame points with a camera-to-map pose.

    The helper is ROS-independent so registration math is unit tested instead
    of being hidden inside a topic bridge.
    """
    points = _points(points)
    translation = np.asarray(translation, dtype=float).reshape((3,))
    quaternion = np.asarray(quaternion_xyzw, dtype=float).reshape((4,))
    if not np.all(np.isfinite(translation)) or not np.all(np.isfinite(quaternion)):
        raise ValueError("pose must be finite")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-9:
        raise ValueError("pose quaternion must be non-zero")
    x, y, z, w = quaternion / norm
    rotation = np.asarray([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),
         2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z),
         2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w),
         1.0 - 2.0 * (x * x + y * y)],
    ])
    return points.dot(rotation.T) + translation


def _unique_rows(values):
    values = np.asarray(values, dtype=np.int64).reshape((-1, 3))
    if not len(values):
        return values
    return np.unique(values, axis=0)


def _voxel_keys(indices):
    """Pack practical map indices into sortable signed 63-bit keys."""
    indices = np.asarray(indices, dtype=np.int64).reshape((-1, 3))
    offset = np.int64(1 << 20)
    shifted = indices + offset
    if len(shifted) and (np.any(shifted < 0) or np.any(shifted >= (1 << 21))):
        raise ValueError("voxel index outside supported +/-1048576 range")
    return ((shifted[:, 0] << np.int64(42))
            | (shifted[:, 1] << np.int64(21)) | shifted[:, 2])


class VoxelIndexMembership:
    """Vectorized, memory-bounded membership for millions of voxel indices."""

    def __init__(self, indices):
        self.keys = np.unique(_voxel_keys(indices))

    def contains(self, indices):
        query = _voxel_keys(indices)
        if not len(self.keys):
            return np.zeros(len(query), dtype=bool)
        positions = np.searchsorted(self.keys, query)
        valid = positions < len(self.keys)
        result = np.zeros(len(query), dtype=bool)
        result[valid] = self.keys[positions[valid]] == query[valid]
        return result

    def contains_one(self, index):
        return bool(self.contains(np.asarray(index, dtype=np.int64).reshape((1, 3)))[0])


def _canonical_metadata(metadata):
    ignored = {"map_hash", "stamp", "created_wall_time"}
    return {str(key): value for key, value in (metadata or {}).items() if key not in ignored}


@dataclass
class MapSnapshot:
    resolution: float
    occupied_indices: np.ndarray
    observed_indices: np.ndarray
    frame_id: str = "world"
    source: str = "unknown"
    stamp: float = 0.0
    metadata: Dict = field(default_factory=dict)
    traversed_positions: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=float)
    )

    def __post_init__(self):
        self.resolution = float(self.resolution)
        if not np.isfinite(self.resolution) or self.resolution <= 0.0:
            raise ValueError("map resolution must be positive")
        self.occupied_indices = _unique_rows(self.occupied_indices)
        self.observed_indices = _unique_rows(self.observed_indices)
        self.frame_id = str(self.frame_id or "world")
        self.source = str(self.source or "unknown")
        self.traversed_positions = _points(self.traversed_positions)
        occupied = {tuple(row) for row in self.occupied_indices.tolist()}
        observed = {tuple(row) for row in self.observed_indices.tolist()}
        if not occupied.issubset(observed):
            raise ValueError("every occupied voxel must also be observed")

    @property
    def occupied_points(self):
        return voxel_centers(self.occupied_indices, self.resolution)

    @property
    def observed_points(self):
        return voxel_centers(self.observed_indices, self.resolution)

    @property
    def map_hash(self):
        digest = hashlib.sha256()
        digest.update(("resolution=%.12g\n" % self.resolution).encode("ascii"))
        digest.update(("frame=%s\nsource=%s\n" % (self.frame_id, self.source)).encode("utf-8"))
        digest.update(self.occupied_indices.astype("<i8", copy=False).tobytes(order="C"))
        digest.update(b"\nobserved\n")
        digest.update(self.observed_indices.astype("<i8", copy=False).tobytes(order="C"))
        if len(self.traversed_positions):
            digest.update(b"\ntraversed_positions\n")
            digest.update(
                self.traversed_positions.astype("<f8", copy=False).tobytes(order="C")
            )
        digest.update(json.dumps(
            _canonical_metadata(self.metadata), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8"))
        return digest.hexdigest()

    def world(self, inflation=0.0, unknown_is_occupied=True):
        return VoxelMapWorld(self, inflation=inflation, unknown_is_occupied=unknown_is_occupied)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(self.metadata)
        metadata["map_hash"] = self.map_hash
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=path.name + ".", suffix=".npz", dir=str(path.parent)
        )
        os.close(descriptor)
        try:
            np.savez_compressed(
                temporary_name,
                resolution=np.asarray([self.resolution], dtype=float),
                occupied_indices=self.occupied_indices.astype(np.int64),
                observed_indices=self.observed_indices.astype(np.int64),
                frame_id=np.asarray([self.frame_id]),
                source=np.asarray([self.source]),
                stamp=np.asarray([self.stamp], dtype=float),
                metadata_json=np.asarray([
                    json.dumps(metadata, sort_keys=True, ensure_ascii=False)
                ]),
                traversed_positions=self.traversed_positions.astype(float),
            )
            # Force the complete ZIP to storage before publishing its name.
            with open(temporary_name, "rb") as handle:
                os.fsync(handle.fileno())
            with np.load(temporary_name, allow_pickle=False) as verification:
                required = {
                    "resolution", "occupied_indices", "observed_indices",
                    "frame_id", "source", "stamp", "metadata_json",
                    "traversed_positions",
                }
                if not required.issubset(verification.files):
                    raise ValueError("incomplete map snapshot archive")
            os.replace(temporary_name, str(path))
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    @classmethod
    def load(cls, path):
        path = Path(path)
        with np.load(str(path), allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"][0]))
            expected_hash = metadata.pop("map_hash", None)
            snapshot = cls(
                resolution=float(data["resolution"][0]),
                occupied_indices=data["occupied_indices"],
                observed_indices=data["observed_indices"],
                frame_id=str(data["frame_id"][0]),
                source=str(data["source"][0]),
                stamp=float(data["stamp"][0]),
                metadata=metadata,
                traversed_positions=(
                    data["traversed_positions"]
                    if "traversed_positions" in data.files
                    else np.empty((0, 3), dtype=float)
                ),
            )
        if expected_hash is not None and snapshot.map_hash != expected_hash:
            raise ValueError("map snapshot hash mismatch for %s" % path)
        return snapshot

    @classmethod
    def from_points(cls, occupied_points, observed_points, resolution=0.05,
                    frame_id="world", source="registered_cloud", stamp=0.0,
                    metadata=None, traversed_positions=()):
        occupied = voxel_indices(occupied_points, resolution)
        observed = voxel_indices(observed_points, resolution)
        if len(occupied):
            observed = np.vstack((observed, occupied))
        return cls(
            resolution, occupied, observed, frame_id=frame_id, source=source,
            stamp=stamp, metadata=dict(metadata or {}),
            traversed_positions=traversed_positions,
        )


def fuse_map_snapshots(snapshots, reference_index=0, sensor_suite=(),
                       observation_support_voxels=0,
                       corridor_xy_padding=None, corridor_z_bounds=None,
                       comparison_independent_survey=False,
                       survey_protocol="unspecified", survey_manifest_sha256=None):
    """Conservatively fuse independent sensor-map acquisition sessions.

    An occupied return from any session stays occupied. Registered RGB-D rays
    may use one Chebyshev-connected voxel of support to represent finite pixel
    area and depth quantization instead of an unphysical zero-width ray. No Oracle geometry,
    interpolation, occupied-point deletion, or support larger than one voxel
    is allowed. A designated input supplies the measured traversed centerline.
    """
    snapshots = list(snapshots)
    if len(snapshots) < 2:
        raise ValueError("at least two map snapshots are required for fusion")
    reference_index = int(reference_index)
    if reference_index < 0 or reference_index >= len(snapshots):
        raise ValueError("reference_index is outside the snapshot list")
    observation_support_voxels = int(observation_support_voxels)
    if observation_support_voxels not in (0, 1):
        raise ValueError("observation_support_voxels must be 0 or 1")
    comparison_independent_survey = bool(comparison_independent_survey)
    survey_protocol = str(survey_protocol).strip()
    if comparison_independent_survey and survey_protocol in ("", "unspecified"):
        raise ValueError(
            "an independent map requires a named survey_protocol"
        )
    if comparison_independent_survey and not survey_manifest_sha256:
        raise ValueError(
            "an independent map requires a survey protocol manifest hash"
        )
    if comparison_independent_survey:
        for snapshot in snapshots:
            if snapshot.metadata.get("comparison_independent_survey") is not True:
                raise ValueError("all input maps require measured independent-survey certification")
            if snapshot.metadata.get("survey_protocol_manifest_sha256") != survey_manifest_sha256:
                raise ValueError("input map survey manifest hash does not match")
            if snapshot.metadata.get("survey_protocol") != survey_protocol:
                raise ValueError("input map survey protocol does not match")

    first = snapshots[0]
    localization_sources = []
    for index, snapshot in enumerate(snapshots):
        if not np.isclose(snapshot.resolution, first.resolution, atol=1e-12):
            raise ValueError("map resolution mismatch at input %d" % index)
        if snapshot.frame_id != first.frame_id:
            raise ValueError("map frame mismatch at input %d" % index)
        if snapshot.metadata.get("truth_derived") is not False:
            raise ValueError("input %d is not explicitly non-truth-derived" % index)
        if snapshot.metadata.get("unknown_is_occupied") is not True:
            raise ValueError("input %d does not use conservative unknown space" % index)
        localization = str(snapshot.metadata.get(
            "localization_source", "unspecified"
        )).strip().lower()
        if localization in ("", "unspecified", "gazebo_truth", "ground_truth",
                            "synthetic_fixture"):
            raise ValueError("input %d has an unapproved localization source" % index)
        localization_sources.append(localization)
    if len(set(localization_sources)) != 1:
        raise ValueError("localization source mismatch across map inputs")

    occupied = _unique_rows(np.vstack([
        snapshot.occupied_indices for snapshot in snapshots
    ]))
    observed = _unique_rows(np.vstack([
        snapshot.observed_indices for snapshot in snapshots
    ] + [occupied]))
    if observation_support_voxels == 1:
        offsets = np.asarray([
            [dx, dy, dz]
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
        ], dtype=np.int64)
        observed = _unique_rows(np.vstack([
            observed + offset for offset in offsets
        ] + [occupied]))
    reference = snapshots[reference_index]
    if not len(reference.traversed_positions):
        raise ValueError("reference map has no measured traversed centerline")

    crop_bounds = None
    if corridor_xy_padding is not None:
        corridor_xy_padding = float(corridor_xy_padding)
        if not np.isfinite(corridor_xy_padding) or corridor_xy_padding <= 0.0:
            raise ValueError("corridor_xy_padding must be positive")
        if corridor_z_bounds is None or len(corridor_z_bounds) != 2:
            raise ValueError("corridor_z_bounds must contain z_min and z_max")
        z_min, z_max = [float(value) for value in corridor_z_bounds]
        if not np.isfinite(z_min) or not np.isfinite(z_max) or z_min >= z_max:
            raise ValueError("corridor z bounds must be finite and increasing")
        minimum = np.min(reference.traversed_positions, axis=0)
        maximum = np.max(reference.traversed_positions, axis=0)
        minimum[:2] -= corridor_xy_padding
        maximum[:2] += corridor_xy_padding
        minimum[2], maximum[2] = z_min, z_max

        def inside(indices):
            centers = voxel_centers(indices, first.resolution)
            return np.all((centers >= minimum) & (centers <= maximum), axis=1)

        occupied = occupied[inside(occupied)]
        observed = observed[inside(observed)]
        observed = _unique_rows(np.vstack((observed, occupied)))
        crop_bounds = {
            "minimum": minimum.tolist(),
            "maximum": maximum.tolist(),
            "xy_padding_m": corridor_xy_padding,
            "source": "onboard_estimator_reference_session",
        }

    metadata = {
        "acquisition_mode": "conservative_multi_session_sensor_fusion",
        "formal_experiment_allowed": False,
        "comparison_independent_survey": comparison_independent_survey,
        "input_map_hashes": [snapshot.map_hash for snapshot in snapshots],
        "input_map_sources": [snapshot.source for snapshot in snapshots],
        "input_session_count": len(snapshots),
        "localization_source": localization_sources[0],
        "observation_support_model": (
            "rgbd_finite_pixel_chebyshev_one_voxel"
            if observation_support_voxels else "zero_width_voxel_rays"
        ),
        "observation_support_voxels": observation_support_voxels,
        "reference_session_index": reference_index,
        "sensor_suite": [str(value) for value in sensor_suite],
        "survey_protocol": survey_protocol,
        "survey_protocol_manifest_sha256": survey_manifest_sha256,
        "task_corridor_crop": crop_bounds,
        "traversed_centerline_source": "onboard_estimator_reference_session",
        "truth_derived": False,
        "unknown_is_occupied": True,
    }
    return MapSnapshot(
        resolution=first.resolution,
        occupied_indices=occupied,
        observed_indices=observed,
        frame_id=first.frame_id,
        source="fused_sensor_slam",
        stamp=max(snapshot.stamp for snapshot in snapshots),
        metadata=metadata,
        traversed_positions=reference.traversed_positions,
    )


class VoxelMapWorld:
    """Conservative occupied-voxel distance field with explicit unknown space.

    Distances are measured to occupied voxel centres and reduced by the voxel
    circumscribed radius plus sensor/map inflation. Unknown voxels return a
    negative distance so an unobserved corridor cannot be certified.
    """

    def __init__(self, snapshot, inflation=0.0, unknown_is_occupied=True,
                 occupied_indices=None):
        self.snapshot = snapshot
        self.resolution = snapshot.resolution
        self.inflation = float(inflation)
        self.unknown_is_occupied = bool(unknown_is_occupied)
        self._occupied_indices = (
            snapshot.occupied_indices if occupied_indices is None
            else _unique_rows(occupied_indices)
        )
        self.points = voxel_centers(self._occupied_indices, self.resolution)
        self.tree = cKDTree(self.points) if len(self.points) else None
        self._observed = VoxelIndexMembership(snapshot.observed_indices)
        self._voxel_radius = np.sqrt(3.0) * 0.5 * self.resolution

    def _known_mask(self, points):
        if not self.unknown_is_occupied:
            return np.ones(len(points), dtype=bool)
        indices = voxel_indices(points, self.resolution)
        return self._observed.contains(indices)

    def signed_distances(self, points):
        points = _points(points)
        if self.tree is None:
            values = np.full(len(points), np.inf, dtype=float)
        else:
            # Reference preparation queries every observed voxel. Use the
            # native cKDTree worker pool for these independent exact nearest-
            # neighbour queries; this changes no geometry or threshold and
            # keeps large formal maps inside the common 15 s front-end budget.
            try:
                values, _ = self.tree.query(
                    points, k=1, workers=-1 if len(points) >= 4096 else 1
                )
            except TypeError:
                try:
                    values, _ = self.tree.query(
                        points, k=1, n_jobs=-1 if len(points) >= 4096 else 1
                    )
                except TypeError:
                    values, _ = self.tree.query(points, k=1)
            values = np.asarray(values, dtype=float) - self._voxel_radius - self.inflation
        known = self._known_mask(points)
        values[~known] = -self.resolution
        return values

    def signed_distance(self, point):
        point = np.asarray(point, dtype=float).reshape((1, 3))
        value = float(self.signed_distances(point)[0])
        name = "slam_occupied" if value >= -self.resolution + 1e-12 else "slam_unknown"
        if self.unknown_is_occupied and not self._known_mask(point)[0]:
            name = "slam_unknown"
        return value, name

    def minimum_distance(self, points):
        points = _points(points)
        if not len(points):
            return float("inf"), "none", None
        values = self.signed_distances(points)
        index = int(np.argmin(values))
        name = "slam_unknown" if not self._known_mask(points[[index]])[0] else "slam_occupied"
        return float(values[index]), name, points[index].copy()

    def minimum_distances_grouped(self, point_groups):
        groups = np.asarray(point_groups, dtype=float)
        if groups.ndim != 3 or groups.shape[2] != 3:
            raise ValueError("point_groups must have shape (G, N, 3)")
        flat = groups.reshape((-1, 3))
        values = self.signed_distances(flat).reshape(groups.shape[:2])
        known = self._known_mask(flat).reshape(groups.shape[:2])
        indices = np.argmin(values, axis=1)
        rows = np.arange(len(groups))
        names = np.where(known[rows, indices], "slam_occupied", "slam_unknown")
        return values[rows, indices], names.tolist()

    def filtered(self, minimum, maximum, padding=0.0):
        minimum = np.asarray(minimum, dtype=float) - float(padding)
        maximum = np.asarray(maximum, dtype=float) + float(padding)
        if not len(self.points):
            selected = np.empty((0, 3), dtype=np.int64)
        else:
            mask = np.all((self.points >= minimum) & (self.points <= maximum), axis=1)
            selected = self._occupied_indices[mask]
        return VoxelMapWorld(
            self.snapshot, inflation=self.inflation,
            unknown_is_occupied=self.unknown_is_occupied,
            occupied_indices=selected,
        )


class IncrementalVoxelMapper:
    """Accumulate a registered point cloud and ray-observed free voxels."""

    def __init__(self, resolution=0.05, maximum_range=8.0, ray_step_ratio=0.5):
        self.resolution = float(resolution)
        self.maximum_range = float(maximum_range)
        self.ray_step = max(0.1, float(ray_step_ratio)) * self.resolution
        self.occupied = set()
        self.observed = set()

    def integrate(self, points, sensor_origin, stride=1):
        points = _points(points)[::max(1, int(stride))]
        origin = np.asarray(sensor_origin, dtype=float).reshape((3,))
        if not np.all(np.isfinite(origin)):
            raise ValueError("sensor origin must be finite")
        accepted = 0
        for endpoint in points:
            delta = endpoint - origin
            distance = float(np.linalg.norm(delta))
            if distance <= 1e-9 or distance > self.maximum_range:
                continue
            samples = max(2, int(np.ceil(distance / self.ray_step)) + 1)
            ray = origin[None, :] + np.linspace(0.0, 1.0, samples)[:, None] * delta[None, :]
            indices = voxel_indices(ray, self.resolution)
            for item in indices[:-1]:
                self.observed.add(tuple(item))
            occupied = tuple(indices[-1])
            self.occupied.add(occupied)
            self.observed.add(occupied)
            accepted += 1
        return accepted

    def integrate_free_rays(self, endpoints, sensor_origin, stride=1):
        """Mark no-return rays observed free without erasing occupied hits."""
        endpoints = _points(endpoints)[::max(1, int(stride))]
        origin = np.asarray(sensor_origin, dtype=float).reshape((3,))
        if not np.all(np.isfinite(origin)):
            raise ValueError("sensor origin must be finite")
        accepted = 0
        for endpoint in endpoints:
            delta = endpoint - origin
            distance = float(np.linalg.norm(delta))
            if distance <= 1e-9 or distance > self.maximum_range + 1e-6:
                continue
            samples = max(2, int(np.ceil(distance / self.ray_step)) + 1)
            ray = origin[None, :] + np.linspace(0.0, 1.0, samples)[:, None] * delta[None, :]
            for item in voxel_indices(ray, self.resolution):
                self.observed.add(tuple(item))
            accepted += 1
        return accepted

    def mark_robot_footprint_free(self, center, radius_xy, half_height):
        """Clear only the volume physically occupied by the surveying robot.

        Mobile mapping stacks routinely clear the current robot footprint so
        a forward-facing sensor does not leave the platform itself in unknown
        space.  The caller must not include planning clearance margins here.
        """
        center = np.asarray(center, dtype=float).reshape((3,))
        radius_xy = float(radius_xy)
        half_height = float(half_height)
        if (not np.all(np.isfinite(center)) or radius_xy <= 0.0
                or half_height <= 0.0):
            raise ValueError("robot footprint parameters must be finite and positive")
        # Mark every voxel cell intersecting the physical cylinder.  Testing
        # only voxel centres would erode a narrow measured free tube by up to
        # half a voxel diagonal and can destroy connectivity at 5 cm grids.
        radial_cell_allowance = math.sqrt(2.0) * 0.5 * self.resolution
        vertical_cell_allowance = 0.5 * self.resolution
        voxel_radius = radius_xy + radial_cell_allowance
        voxel_half_height = half_height + vertical_cell_allowance
        lower = np.floor(
            (center - [voxel_radius, voxel_radius, voxel_half_height]) / self.resolution
        ).astype(np.int64)
        upper = np.floor(
            (center + [voxel_radius, voxel_radius, voxel_half_height]) / self.resolution
        ).astype(np.int64)
        cleared = 0
        for ix in range(int(lower[0]), int(upper[0]) + 1):
            for iy in range(int(lower[1]), int(upper[1]) + 1):
                voxel_xy = (np.asarray([ix, iy], dtype=float) + 0.5) * self.resolution
                if float(np.linalg.norm(voxel_xy - center[:2])) > voxel_radius:
                    continue
                for iz in range(int(lower[2]), int(upper[2]) + 1):
                    voxel_z = (float(iz) + 0.5) * self.resolution
                    if abs(voxel_z - center[2]) > voxel_half_height:
                        continue
                    item = (ix, iy, iz)
                    self.observed.add(item)
                    self.occupied.discard(item)
                    cleared += 1
        return cleared

    def snapshot(self, frame_id="world", source="online_slam", stamp=0.0,
                 metadata: Optional[Dict] = None, traversed_positions=()):
        occupied = np.asarray(sorted(self.occupied), dtype=np.int64).reshape((-1, 3))
        observed = np.asarray(sorted(self.observed), dtype=np.int64).reshape((-1, 3))
        return MapSnapshot(
            self.resolution, occupied, observed, frame_id=frame_id,
            source=source, stamp=stamp, metadata=dict(metadata or {}),
            traversed_positions=traversed_positions,
        )


class OnlineSnapshotAssembler:
    """Assemble independently transported occupied/observed map components."""

    def __init__(self):
        self.occupied_points = None
        self.observed_points = None
        self.frame_id = None
        self.stamp = 0.0
        self.metadata = {}

    def update_metadata(self, metadata):
        metadata = dict(metadata or {})
        if metadata.get("truth_geometry_access") is not False:
            raise ValueError("online planner map must explicitly deny truth geometry access")
        self.metadata = metadata
        return self.snapshot()

    def update_occupied(self, points, frame_id="world", stamp=0.0):
        self._check_frame(frame_id)
        self.occupied_points = _points(points)
        self.stamp = max(self.stamp, float(stamp))
        return self.snapshot()

    def update_observed(self, points, frame_id="world", stamp=0.0):
        self._check_frame(frame_id)
        self.observed_points = _points(points)
        self.stamp = max(self.stamp, float(stamp))
        return self.snapshot()

    def _check_frame(self, frame_id):
        frame_id = str(frame_id or "world")
        if self.frame_id is not None and frame_id != self.frame_id:
            raise ValueError("online map frame changed from %s to %s" % (
                self.frame_id, frame_id,
            ))
        self.frame_id = frame_id

    def snapshot(self):
        if (self.occupied_points is None or self.observed_points is None
                or not self.metadata):
            return None
        snapshot_metadata = dict(self.metadata.get("snapshot_metadata", {}))
        snapshot_metadata.setdefault(
            "localization_source", self.metadata.get("localization_source", "unspecified")
        )
        snapshot_metadata["unknown_is_occupied"] = True
        snapshot = MapSnapshot.from_points(
            self.occupied_points, self.observed_points,
            resolution=float(self.metadata.get("resolution", 0.05)),
            frame_id=self.frame_id or str(self.metadata.get("frame_id", "world")),
            source=str(self.metadata.get("source", "online_slam")),
            stamp=self.stamp, metadata=snapshot_metadata,
        )
        expected = self.metadata.get("map_hash")
        if expected and expected != snapshot.map_hash:
            return None
        return snapshot


def observed_box_points(minimum: Iterable[float], maximum: Iterable[float], resolution):
    """Create voxel centres for an explicitly declared observed test volume.

    This helper is only for frozen-map fixtures or a pre-mapped simulation.
    It must not be used to claim online SLAM visibility.
    """
    minimum = np.asarray(minimum, dtype=float)
    maximum = np.asarray(maximum, dtype=float)
    lower = np.floor(minimum / resolution).astype(np.int64)
    upper = np.floor(maximum / resolution).astype(np.int64)
    axes = [np.arange(lower[i], upper[i] + 1, dtype=np.int64) for i in range(3)]
    grid = np.meshgrid(*axes, indexing="ij")
    indices = np.column_stack([item.ravel() for item in grid])
    return voxel_centers(indices, resolution)
