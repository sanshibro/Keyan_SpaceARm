"""Common SLAM-map global reference used by every comparison method."""

import heapq
import math
import time

import numpy as np

from .dependencies import import_scipy
from .geometry import world_body_points
from .mapping import VoxelIndexMembership, voxel_centers, voxel_indices
from .types import Limits


scipy = import_scipy()
from scipy.spatial import cKDTree  # noqa: E402


class ReferencePlanningError(RuntimeError):
    pass


class VoxelAStarReferencePlanner:
    def __init__(self, snapshot, center_clearance=0.38, body_radius=None,
                 body_height=0.25, body_margin=0.10, z_min=0.75,
                 z_max=2.20, clearance_weight=0.04,
                 heuristic_weight=1.5,
                 maximum_expansions=300000, time_budget=10.0, body_center_z=0.):
        preparation_started = time.monotonic()
        self.snapshot = snapshot
        self.world = snapshot.world(inflation=0.0, unknown_is_occupied=True)
        self.center_clearance = float(center_clearance)
        self.body_radius = None if body_radius is None else float(body_radius)
        self.body_height = float(body_height)
        self.body_margin = float(body_margin)
        self.body_template = None
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.clearance_weight = float(clearance_weight)
        self.heuristic_weight = float(heuristic_weight)
        if self.heuristic_weight < 1.0:
            raise ValueError("heuristic_weight must be at least 1.0")
        self.maximum_expansions = int(maximum_expansions)
        self.time_budget = float(time_budget)
        self.occupied = VoxelIndexMembership(snapshot.occupied_indices)
        self.observed_indices = snapshot.observed_indices
        self.observed_points = snapshot.observed_points
        if len(self.observed_points):
            clearances = self.world.signed_distances(self.observed_points)
            if time.monotonic() - preparation_started > self.time_budget:
                raise ReferencePlanningError("reference map preparation timeout")
            z = self.observed_points[:, 2]
            occupied_mask = self.occupied.contains(self.observed_indices)
            if self.body_radius is None:
                geometry_feasible = clearances >= self.center_clearance
            else:
                body_limits = Limits(
                    body_radius=self.body_radius, body_height=self.body_height, body_center_z=body_center_z,
                )
                body_template = world_body_points(
                    np.zeros(6), body_limits,
                    azimuth_samples=16, vertical_samples=3,
                )
                self.body_template = body_template
                geometry_feasible = np.zeros(len(self.observed_points), dtype=bool)
                circumscribed = math.hypot(
                    self.body_radius, 0.5 * self.body_height + abs(body_center_z)
                ) + self.body_margin
                for start in range(0, len(self.observed_points), 2000):
                    if time.monotonic() - preparation_started > self.time_budget:
                        raise ReferencePlanningError(
                            "reference map preparation timeout"
                        )
                    stop = min(len(self.observed_points), start + 2000)
                    groups = (
                        self.observed_points[start:stop, None, :] + body_template[None, :, :]
                    )
                    # Most free-space centres are farther than the body's
                    # circumscribed bound and need no occupied-tree query.
                    # We still require every sampled envelope voxel to be
                    # observed, preserving fail-closed unknown-space semantics.
                    known = self.world._known_mask(
                        groups.reshape((-1, 3))
                    ).reshape(groups.shape[:2]).all(axis=1)
                    guaranteed = clearances[start:stop] >= circumscribed
                    feasible = known & guaranteed
                    uncertain = np.flatnonzero(known & ~guaranteed)
                    if len(uncertain):
                        values, _ = self.world.minimum_distances_grouped(
                            groups[uncertain]
                        )
                        feasible[uncertain] = values >= self.body_margin
                    geometry_feasible[start:stop] = feasible
            feasible_mask = (
                ~occupied_mask & (z >= self.z_min) & (z <= self.z_max)
                & geometry_feasible
            )
            self.feasible_indices = self.observed_indices[feasible_mask]
            self.feasible_points = self.observed_points[feasible_mask]
            self.feasible = VoxelIndexMembership(self.feasible_indices)
            # Cache only expanded A* states.  Materializing a Python dict for
            # every 0.05 m free voxel dominated reference startup time.
            self._clearance_cache = {}
            self.observed_tree = (
                cKDTree(self.feasible_points) if len(self.feasible_points) else None
            )
        else:
            self.feasible_indices = np.empty((0, 3), dtype=np.int64)
            self.feasible_points = np.empty((0, 3), dtype=float)
            self.feasible = VoxelIndexMembership(self.feasible_indices)
            self._clearance_cache = {}
            self.observed_tree = None
        self.neighbours = [
            (dx, dy, dz) for dx in (-1, 0, 1)
            for dy in (-1, 0, 1) for dz in (-1, 0, 1)
            if (dx, dy, dz) != (0, 0, 0)
        ]
        self.preparation_elapsed = time.monotonic() - preparation_started

    def _center(self, index):
        return voxel_centers(np.asarray([index], dtype=np.int64), self.snapshot.resolution)[0]

    def _clearance(self, index):
        index = tuple(index)
        if index not in self._clearance_cache:
            self._clearance_cache[index] = float(
                self.world.signed_distance(self._center(index))[0]
            )
        return self._clearance_cache[index]

    def _valid(self, index):
        return self.feasible.contains_one(index)

    def _continuous_body_valid(self, point):
        """Validate a continuous body pose without snapping it to a voxel centre."""
        point = np.asarray(point, dtype=float).reshape((3,))
        if point[2] < self.z_min or point[2] > self.z_max:
            return False
        if self.body_template is None:
            if not self.world._known_mask(point.reshape((1, 3)))[0]:
                return False
            return self.world.signed_distance(point)[0] >= self.center_clearance
        groups = point[None, :] + self.body_template
        if not self.world._known_mask(groups).all():
            return False
        values, _sources = self.world.minimum_distances_grouped(groups[None, :, :])
        return bool(values[0] >= self.body_margin)

    def _continuous_bridge_is_valid(self, first, second):
        first = np.asarray(first, dtype=float)
        second = np.asarray(second, dtype=float)
        distance = float(np.linalg.norm(second - first))
        count = max(2, int(np.ceil(distance / (0.5 * self.snapshot.resolution))) + 1)
        return all(
            self._continuous_body_valid(point)
            for point in np.linspace(first, second, count)
        )

    def _survey_centerline_reference(self, start, goal, started):
        """Recover a common reference from independently validated SLAM keyframes."""
        trajectory = np.asarray(
            self.snapshot.traversed_positions, dtype=float
        ).reshape((-1, 3))
        if len(trajectory) < 2:
            return None
        start_index = int(np.argmin(np.linalg.norm(trajectory - start, axis=1)))
        goal_index = int(np.argmin(np.linalg.norm(trajectory - goal, axis=1)))
        if goal_index >= start_index:
            segment = trajectory[start_index:goal_index + 1]
        else:
            segment = trajectory[goal_index:start_index + 1][::-1]
        maximum_start_snap = 0.10
        start_gap = float(np.linalg.norm(segment[0] - start))
        if start_gap > maximum_start_snap + 1e-9:
            return None
        prefix = [np.asarray(start, dtype=float)] if (
            self._continuous_body_valid(start)
            and self._continuous_bridge_is_valid(start, segment[0])
        ) else []
        points = np.vstack(tuple(prefix) + (segment, np.asarray(goal, dtype=float)[None, :]))
        # Drop near-duplicates without altering the measured polyline order.
        compact = [points[0]]
        for point in points[1:]:
            if float(np.linalg.norm(point - compact[-1])) >= 1e-4:
                compact.append(point)
        points = np.asarray(compact, dtype=float)
        if len(points) < 2:
            return None
        if not all(
                self._continuous_bridge_is_valid(first, second)
                for first, second in zip(points[:-1], points[1:])):
            return None
        # Preserve curvature, but merge short runs only when the direct full-
        # body bridge is independently valid in the frozen map.
        simplified = [points[0]]
        anchor = 0
        while anchor < len(points) - 1:
            candidate = min(len(points) - 1, anchor + 12)
            while candidate > anchor + 1 and not self._continuous_bridge_is_valid(
                    points[anchor], points[candidate]):
                candidate -= 1
            simplified.append(points[candidate])
            anchor = candidate
        result = np.asarray(simplified, dtype=float)
        return result, {
            "expanded": 0,
            "elapsed": time.monotonic() - started,
            "preparation_elapsed": self.preparation_elapsed,
            "total_elapsed": self.preparation_elapsed + time.monotonic() - started,
            "path_points_raw": int(len(points)),
            "path_points_simplified": int(len(result)),
            "map_hash": self.snapshot.map_hash,
            "terminal_reached": True,
            "continuous_goal_bridge": True,
            "reference_kind": "terminal",
            "search": "validated_slam_survey_centerline",
            "heuristic_weight": self.heuristic_weight,
            "requested_goal_distance": 0.0,
            "requested_start_distance": float(np.linalg.norm(result[0] - start)),
            "truth_geometry_access": False,
        }

    def _nearest_valid(self, point):
        if self.observed_tree is None:
            raise ReferencePlanningError("map contains no observed voxels")
        count = min(len(self.feasible_points), 2048)
        distances, indices = self.observed_tree.query(
            np.asarray(point, dtype=float), k=count
        )
        for item in np.atleast_1d(indices):
            candidate = tuple(self.feasible_indices[int(item)])
            if self._valid(candidate):
                return candidate
        raise ReferencePlanningError("no body-clear observed voxel near task endpoint")

    @staticmethod
    def _heuristic(first, second, resolution):
        return float(np.linalg.norm(np.asarray(first) - np.asarray(second))) * resolution

    def _edge_cost(self, first, second, clearance=None):
        length = self._heuristic(first, second, self.snapshot.resolution)
        clearance = max(
            self._clearance(second) if clearance is None else float(clearance), 1e-3
        )
        return length + self.clearance_weight * length / clearance

    def _line_is_valid(self, first, second):
        first = np.asarray(first, dtype=float)
        second = np.asarray(second, dtype=float)
        distance = float(np.linalg.norm(second - first))
        count = max(2, int(np.ceil(distance / (0.5 * self.snapshot.resolution))) + 1)
        indices = voxel_indices(np.linspace(first, second, count), self.snapshot.resolution)
        return bool(np.all(self.feasible.contains(indices)))

    def _simplify(self, points):
        if len(points) <= 2:
            return points
        result = [points[0]]
        anchor = 0
        while anchor < len(points) - 1:
            candidate = len(points) - 1
            while candidate > anchor + 1:
                if self._line_is_valid(points[anchor], points[candidate]):
                    break
                candidate -= 1
            result.append(points[candidate])
            anchor = candidate
        return np.asarray(result, dtype=float)

    def plan(self, start, goal, allow_partial=False):
        started = time.monotonic()
        start_index = self._nearest_valid(start)
        requested_goal_index = tuple(voxel_indices([goal], self.snapshot.resolution)[0])
        terminal_reached = self._valid(requested_goal_index)
        continuous_goal_bridge = False
        if terminal_reached:
            goal_index = requested_goal_index
        elif self._continuous_body_valid(goal):
            candidate = self._nearest_valid(goal)
            candidate_center = self._center(candidate)
            maximum_snap = math.sqrt(3.0) * self.snapshot.resolution
            if (float(np.linalg.norm(candidate_center - goal)) <= maximum_snap + 1e-9
                    and self._continuous_bridge_is_valid(candidate_center, goal)):
                # A continuous task endpoint need not coincide with the centre
                # of its voxel. Plan to the closest certified grid state, then
                # retain the exact endpoint through an independently checked
                # sub-voxel bridge shared by every method.
                goal_index = candidate
                terminal_reached = True
                continuous_goal_bridge = True
            elif allow_partial:
                goal_index = candidate
            else:
                raise ReferencePlanningError(
                    "continuous task goal has no certified sub-voxel bridge"
                )
        elif allow_partial:
            # Unknown space is still forbidden.  Select the observed,
            # body-clear frontier voxel closest to the task goal and let the
            # online mapper reveal the next window before replanning.
            goal_index = self._nearest_valid(goal)
            if goal_index == start_index:
                raise ReferencePlanningError(
                    "observed frontier provides no safe forward progress"
                )
        else:
            raise ReferencePlanningError(
                "task goal is not inside observed body-clear space"
            )
        queue = [(
            self.heuristic_weight * self._heuristic(
                start_index, goal_index, self.snapshot.resolution
            ),
            0.0, start_index,
        )]
        parent = {start_index: None}
        cost = {start_index: 0.0}
        expanded = 0
        while queue:
            if time.monotonic() - started > self.time_budget:
                raise ReferencePlanningError("reference planning timeout")
            _priority, current_cost, current = heapq.heappop(queue)
            if current_cost > cost.get(current, float("inf")) + 1e-12:
                continue
            if current == goal_index:
                break
            expanded += 1
            if expanded > self.maximum_expansions:
                raise ReferencePlanningError("reference expansion budget exceeded")
            candidate_indices = np.asarray([
                [current[i] + delta[i] for i in range(3)]
                for delta in self.neighbours
            ], dtype=np.int64)
            valid_indices = candidate_indices[
                self.feasible.contains(candidate_indices)
            ]
            candidate_clearances = self.world.signed_distances(
                voxel_centers(valid_indices, self.snapshot.resolution)
            )
            for neighbour_array, clearance in zip(
                    valid_indices, candidate_clearances):
                neighbour = tuple(neighbour_array)
                candidate_cost = current_cost + self._edge_cost(
                    current, neighbour, clearance=clearance
                )
                if candidate_cost + 1e-12 >= cost.get(neighbour, float("inf")):
                    continue
                cost[neighbour] = candidate_cost
                parent[neighbour] = current
                estimate = candidate_cost + self.heuristic_weight * self._heuristic(
                    neighbour, goal_index, self.snapshot.resolution
                )
                heapq.heappush(queue, (estimate, candidate_cost, neighbour))
        if goal_index not in parent:
            if terminal_reached:
                fallback = self._survey_centerline_reference(start, goal, started)
                if fallback is not None:
                    return fallback
            raise ReferencePlanningError("no path in observed body-clear space")
        indices = []
        current = goal_index
        while current is not None:
            indices.append(current)
            current = parent[current]
        indices.reverse()
        points = np.asarray([self._center(index) for index in indices])
        if (self._continuous_body_valid(start)
                and self._continuous_bridge_is_valid(start, points[0])):
            points[0] = np.asarray(start, dtype=float)
        points = self._simplify(points)
        if terminal_reached:
            if not self._continuous_bridge_is_valid(points[-1], goal):
                raise ReferencePlanningError("terminal continuous bridge became invalid")
            if float(np.linalg.norm(points[-1] - goal)) > 1e-9:
                points = np.vstack((points, np.asarray(goal, dtype=float)))
            else:
                points[-1] = np.asarray(goal, dtype=float)
        return points, {
            "expanded": expanded,
            "elapsed": time.monotonic() - started,
            "preparation_elapsed": self.preparation_elapsed,
            "total_elapsed": self.preparation_elapsed + time.monotonic() - started,
            "path_points_raw": len(indices),
            "path_points_simplified": len(points),
            "map_hash": self.snapshot.map_hash,
            "terminal_reached": bool(terminal_reached),
            "continuous_goal_bridge": bool(continuous_goal_bridge),
            "reference_kind": "terminal" if terminal_reached else "observed_frontier",
            "search": "weighted_astar",
            "heuristic_weight": self.heuristic_weight,
            "requested_goal_distance": float(np.linalg.norm(points[-1] - goal)),
            "requested_start_distance": float(np.linalg.norm(points[0] - start)),
        }


def path_states(points, default_shape=(0.0, 0.0)):
    points = np.asarray(points, dtype=float).reshape((-1, 3))
    if len(points) < 2:
        raise ValueError("reference path needs at least two points")
    states = np.zeros((len(points), 6), dtype=float)
    states[:, :3] = points
    states[:, 4:] = np.asarray(default_shape, dtype=float)
    for index in range(len(points)):
        if index + 1 < len(points):
            delta = points[index + 1] - points[index]
        else:
            delta = points[index] - points[index - 1]
        if np.linalg.norm(delta[:2]) > 1e-9:
            states[index, 3] = math.atan2(delta[1], delta[0])
        elif index:
            states[index, 3] = states[index - 1, 3]
    return states


def densify_path(points, maximum_step=0.20):
    points = np.asarray(points, dtype=float).reshape((-1, 3))
    if len(points) < 2:
        raise ValueError("path needs at least two points")
    output = [points[0]]
    for first, second in zip(points[:-1], points[1:]):
        distance = float(np.linalg.norm(second - first))
        count = max(1, int(np.ceil(distance / max(float(maximum_step), 1e-3))))
        output.extend(
            (first + ratio * (second - first)).tolist()
            for ratio in np.linspace(0.0, 1.0, count + 1)[1:]
        )
    return np.asarray(output, dtype=float)
