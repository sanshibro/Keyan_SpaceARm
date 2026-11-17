#!/usr/bin/env python3
from dataclasses import replace
import math
from pathlib import Path
import unittest

import numpy as np

from uav_continuum_whole_body_v2.configuration import load_planner_configuration
from uav_continuum_whole_body_v2.comparison_methods import (
    AmPlannerAdaptedMethod, ComparisonProblem, METHOD_NAMES, RrtStarUamMethod,
    create_method,
)
from uav_continuum_whole_body_v2.environment import (
    BoxObstacle, BoxWorld, HybridDistanceWorld, PointCloudWorld,
)
from uav_continuum_whole_body_v2.experiment_scenarios import (
    box_surface_points, generate_scenario, synthetic_development_snapshot,
)
from uav_continuum_whole_body_v2.execution import (
    bending_state_from_vector, bending_vector, bending_vector_rate,
    bounded_switch_match,
    bounded_whole_body_switch_match,
    clearance_progress_scale, clearance_tracking_budget,
    continuous_switch_phase_rate,
    evaluate_quintic_bridge, interpolation_bracket, largest_feasible_ratio,
    lerp_angle, quintic_bridge_coefficients,
    projected_phase_elapsed, reference_converged,
    stationary_endpoint_feedforward, tracking_progress_scale,
    trajectory_translation_progress, whole_body_switch_deltas,
    whole_body_switch_gate, wrap_angle,
)
from uav_continuum_whole_body_v2.experiment_metrics import (
    executed_translation_jerk_energy, executed_translation_jerk_profile,
    partition_executed_translation_jerk,
)
from uav_continuum_whole_body_v2.geometry import coupling_metrics, world_arm_points, world_body_points
from uav_continuum_whole_body_v2.mapping import (
    IncrementalVoxelMapper, MapSnapshot, OnlineSnapshotAssembler,
    fuse_map_snapshots,
    observed_box_points, transform_points, validate_map_mode, voxel_indices,
)
from uav_continuum_whole_body_v2.map_audit import (
    _planner_known_collision, audit_snapshot,
)
from uav_continuum_whole_body_v2.optimizer import WholeBodyOptimizer
from uav_continuum_whole_body_v2.reference_planner import (
    ReferencePlanningError, VoxelAStarReferencePlanner,
)
from uav_continuum_whole_body_v2.seed_planner import WholeBodySeedPlanner
from uav_continuum_whole_body_v2.survey_protocol import certify_survey_snapshot
from uav_continuum_whole_body_v2.trajectory import SplineTrajectory
from uav_continuum_whole_body_v2.types import (
    ArmModel, Limits, MassModel, PHI, PlannerConfig, THETA,
)
from uav_continuum_whole_body_v2.validator import SweptVolumeValidator


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.arm = ArmModel()
        self.mass = MassModel()
        self.limits = Limits()
        self.config = PlannerConfig(optimizer_time_budget=0.2)

    def test_deployment_configuration_loader(self):
        package = Path(__file__).resolve().parents[1]
        raw, config, limits, mass = load_planner_configuration(package / "config" / "planner_v2.yaml")
        self.assertAlmostEqual(config.nominal_speed, 0.08)
        self.assertAlmostEqual(config.dt_max, 12.0)
        self.assertAlmostEqual(limits.velocity_max, 0.12)
        self.assertAlmostEqual(limits.body_margin, 0.10)
        self.assertAlmostEqual(limits.arm_margin, 0.070)
        self.assertFalse(raw.get("enable_recovery_windows"))
        self.assertEqual(config.repair_attempts, 3)
        self.assertAlmostEqual(mass.total, 2.25)

    def test_independent_survey_requires_measured_ordered_waypoints(self):
        snapshot = MapSnapshot(
            frame_id="map", stamp=1.0, resolution=0.1,
            occupied_indices=np.asarray([[0, 0, 0]]),
            observed_indices=np.asarray([[0, 0, 0], [1, 0, 0]]),
            source="record_slam",
            metadata={"truth_derived": False, "localization_source": "px4_ekf"},
            traversed_positions=np.asarray([[0, 0, 1], [0, -1, 1], [0, -2, 1]]),
        )
        manifest = {
            "protocol_id": "unit_fixed", "waypoints": [
                {"pose": [0, 0, 1, 0]}, {"pose": [0, -2, 1, 0]},
            ],
        }
        certified, report = certify_survey_snapshot(snapshot, manifest, "abc", 0.1)
        self.assertTrue(report["passed"])
        self.assertTrue(certified.metadata["comparison_independent_survey"])
        reversed_manifest = {
            "protocol_id": "unit_fixed", "waypoints": [
                {"pose": [0, -2, 1, 0]}, {"pose": [0, 0, 1, 0]},
            ],
        }
        with self.assertRaises(ValueError):
            certify_survey_snapshot(snapshot, reversed_manifest, "abc", 0.1)

    def test_executed_jerk_metric_is_stable_under_timestamp_jitter(self):
        # v_x = 0.5 t^2 has constant jerk 1 m/s^3 and energy T.
        rng = np.random.RandomState(7)
        nominal = np.arange(0.0, 8.0 + 1e-9, 0.02)
        jittered = nominal + rng.uniform(-0.004, 0.004, size=len(nominal))
        velocities = np.column_stack((0.5 * jittered ** 2, 0.0 * jittered, 0.0 * jittered))
        energy = executed_translation_jerk_energy(jittered, velocities)
        self.assertIsNotNone(energy)
        self.assertAlmostEqual(energy, 8.0, delta=0.15)

    def test_jerk_profile_preserves_scalar_metric(self):
        times = np.arange(0.0, 10.0 + 1e-9, 0.02)
        velocities = np.column_stack((0.5 * times ** 2, 0.0 * times, 0.0 * times))
        profile = executed_translation_jerk_profile(times, velocities)
        self.assertIsNotNone(profile)
        self.assertAlmostEqual(
            profile["energy"], executed_translation_jerk_energy(times, velocities), places=12,
        )

    def test_jerk_partition_is_disjoint_and_conservative(self):
        times = np.arange(0.0, 10.0 + 1e-9, 0.02)
        velocities = np.column_stack((0.5 * times ** 2, 0.0 * times, 0.0 * times))
        report = partition_executed_translation_jerk(
            times, velocities, start_stamp=1.0, stop_stamp=9.0,
            switch_stamps=(4.0, 6.0), endpoint_radius_s=0.5,
            switch_radius_s=0.25,
        )
        self.assertIsNotNone(report)
        self.assertAlmostEqual(
            sum(report["components"].values()), report["total"], places=12,
        )
        self.assertAlmostEqual(
            sum(report["fractions"].values()), 1.0, places=12,
        )
        self.assertEqual(len(report["per_switch"]), 2)
        self.assertAlmostEqual(
            sum(item["energy"] for item in report["per_switch"]),
            report["components"]["switch"], places=12,
        )

    def test_jerk_partition_rejects_invalid_event_window(self):
        times = np.arange(0.0, 2.0 + 1e-9, 0.02)
        velocities = np.column_stack((times, 0.0 * times, 0.0 * times))
        with self.assertRaises(ValueError):
            partition_executed_translation_jerk(
                times, velocities, start_stamp=2.0, stop_stamp=1.0,
            )

    def test_bending_vector_removes_phi_singularity_at_straight_shape(self):
        first = bending_vector(0.0, 0.0)
        second = bending_vector(0.0, math.pi)
        np.testing.assert_allclose(first, second, atol=1e-12)

    def test_bending_vector_rate_matches_finite_difference(self):
        theta, phi = 0.8, -0.4
        theta_rate, phi_rate = 0.2, -0.3
        dt = 1e-6
        numerical = (
            bending_vector(theta + theta_rate * dt, phi + phi_rate * dt)
            - bending_vector(theta, phi)
        ) / dt
        np.testing.assert_allclose(
            bending_vector_rate(theta, phi, theta_rate, phi_rate),
            numerical, atol=1e-6,
        )

    def test_bending_vector_inverse_recovers_state_and_rate(self):
        theta, phi, theta_rate, phi_rate = 0.8, -0.4, 0.2, -0.3
        recovered = bending_state_from_vector(
            bending_vector(theta, phi),
            bending_vector_rate(theta, phi, theta_rate, phi_rate),
        )
        np.testing.assert_allclose(
            recovered, [theta, phi, theta_rate, phi_rate], atol=1e-12,
        )

    def test_quintic_bridge_matches_c2_boundaries(self):
        start = np.asarray([0.0, -1.0])
        finish = np.asarray([2.0, 0.5])
        start_rate = np.asarray([0.2, 0.1])
        finish_rate = np.asarray([-0.1, 0.3])
        start_acceleration = np.asarray([0.05, -0.02])
        finish_acceleration = np.asarray([0.0, 0.04])
        duration = 1.3
        coefficients = quintic_bridge_coefficients(
            start, start_rate, start_acceleration,
            finish, finish_rate, finish_acceleration, duration,
        )
        for derivative, expected_start, expected_finish in (
                (0, start, finish),
                (1, start_rate, finish_rate),
                (2, start_acceleration, finish_acceleration)):
            values = evaluate_quintic_bridge(
                coefficients, [0.0, duration], derivative=derivative,
            )
            np.testing.assert_allclose(values[0], expected_start, atol=1e-10)
            np.testing.assert_allclose(values[1], expected_finish, atol=1e-10)

    def test_whole_body_switch_deltas_wrap_yaw_and_use_bending_vector(self):
        old_state = [0.0, 0.0, 1.0, math.pi - 0.01, 0.0, 0.0]
        new_state = [0.03, 0.04, 1.0, -math.pi + 0.01, 0.0, math.pi]
        old_rate = [0.1, 0.0, 0.0, 0.1, 0.0, 1.0]
        new_rate = [0.0, 0.0, 0.0, 0.2, 0.0, -2.0]
        report = whole_body_switch_deltas(old_state, old_rate, new_state, new_rate)
        self.assertAlmostEqual(report["position_gap_m"], 0.05)
        self.assertAlmostEqual(report["yaw_gap_rad"], 0.02)
        self.assertAlmostEqual(report["bending_gap_rad"], 0.0)
        self.assertAlmostEqual(report["bending_rate_gap_rad_s"], 0.0)

    def test_whole_body_switch_match_prefers_continuous_tangent(self):
        times = [0.0, 0.2, 0.4]
        states = np.asarray([
            [0.00, 0.00, 1.0, 0.0, 0.0, 0.0],
            [0.02, 0.00, 1.0, 0.0, 0.0, 0.0],
            [0.04, 0.00, 1.0, 0.0, 0.0, 0.0],
        ])
        rates = np.asarray([
            [0.00, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.10, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.12, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])
        result = bounded_whole_body_switch_match(
            times, states, rates,
            old_state=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            old_rate=[0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            measured_position=[0.0, 0.0, 1.0], maximum_advance=0.04,
        )
        self.assertEqual(result[1], 1)
        self.assertAlmostEqual(result[3], 0.02)
        self.assertAlmostEqual(result[5], 0.2)

    def test_whole_body_switch_gate_reports_only_exceeded_c0_c1_limits(self):
        names = (
            "position_gap_m", "yaw_gap_rad", "bending_gap_rad",
            "linear_velocity_gap_m_s", "yaw_rate_gap_rad_s",
            "bending_rate_gap_rad_s",
        )
        limits = {name: 0.1 for name in names}
        metrics = {name: 0.05 for name in names}
        self.assertEqual(whole_body_switch_gate(metrics, limits), {})
        metrics["linear_velocity_gap_m_s"] = 0.11
        violations = whole_body_switch_gate(metrics, limits)
        self.assertEqual(set(violations), {"linear_velocity_gap_m_s"})
        self.assertAlmostEqual(
            violations["linear_velocity_gap_m_s"]["limit"], 0.1,
        )

    def test_whole_body_switch_gate_rejects_invalid_limits(self):
        names = (
            "position_gap_m", "yaw_gap_rad", "bending_gap_rad",
            "linear_velocity_gap_m_s", "yaw_rate_gap_rad_s",
            "bending_rate_gap_rad_s",
        )
        metrics = {name: 0.0 for name in names}
        limits = {name: 0.1 for name in names}
        limits["position_gap_m"] = 0.0
        with self.assertRaises(ValueError):
            whole_body_switch_gate(metrics, limits)

    def test_whole_body_switch_match_respects_advance_bound(self):
        states = np.asarray([
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.2, 0.0, 1.0, 0.0, 0.0, 0.0],
        ])
        rates = np.asarray([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])
        result = bounded_whole_body_switch_match(
            [0.0, 0.2], states, rates, states[0], rates[1], states[0, :3],
            maximum_advance=0.1,
        )
        self.assertEqual(result[1], 0)

    def test_whole_body_switch_match_uses_acceleration_when_available(self):
        times = [0.0, 0.2]
        states = np.asarray([
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.01, 0.0, 1.0, 0.0, 0.0, 0.0],
        ])
        rates = np.asarray([
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])
        accelerations = np.asarray([
            [0.4, 0.0, 0.0], [0.0, 0.0, 0.0],
        ])
        result = bounded_whole_body_switch_match(
            times, states, rates, states[0], rates[0], states[0, :3],
            maximum_advance=0.02, accelerations=accelerations,
            old_acceleration=[0.0, 0.0, 0.0],
        )
        self.assertEqual(result[1], 1)
        self.assertAlmostEqual(
            result[-1]["linear_acceleration_gap_m_s2"], 0.0,
        )

    def test_map_mode_is_explicit_and_fail_closed(self):
        self.assertEqual(validate_map_mode("frozen_slam"), "frozen_slam")
        with self.assertRaises(ValueError):
            validate_map_mode("fallback_to_truth")

    def test_fast_point_cloud_parser_preserves_organized_xyz_layout(self):
        # The production node is ROS-facing and is exercised by launch tests;
        # this fixture documents the structured-buffer layout used by its
        # vectorized parser, including row padding.
        dtype = np.dtype({
            "names": ["x", "y", "z"],
            "formats": ["<f4", "<f4", "<f4"],
            "offsets": [0, 4, 8], "itemsize": 16,
        })
        raw = bytearray(2 * 40)
        organized = np.ndarray(
            shape=(2, 2), dtype=dtype, buffer=raw, strides=(40, 16)
        )
        organized["x"] = [[1.0, 2.0], [3.0, 4.0]]
        organized["y"] = [[5.0, 6.0], [7.0, 8.0]]
        organized["z"] = [[9.0, 10.0], [11.0, 12.0]]
        points = np.column_stack((
            organized["x"].ravel(), organized["y"].ravel(),
            organized["z"].ravel(),
        ))
        np.testing.assert_allclose(points, [
            [1.0, 5.0, 9.0], [2.0, 6.0, 10.0],
            [3.0, 7.0, 11.0], [4.0, 8.0, 12.0],
        ])

    def test_organized_depth_grid_sampling_has_two_dimensional_coverage(self):
        height, width, maximum = 120, 160, 800
        step = int(np.ceil(np.sqrt(float(height * width) / maximum)))
        rows = np.arange(0, height, step, dtype=int)
        columns = np.arange(0, width, step, dtype=int)
        vv, uu = np.meshgrid(rows, columns, indexing="ij")
        self.assertEqual(step, 5)
        self.assertEqual(len(vv.ravel()), 768)
        self.assertEqual(len(np.unique(vv)), 24)
        self.assertEqual(len(np.unique(uu)), 32)

    def test_map_snapshot_hash_and_replay_are_deterministic(self):
        observed = observed_box_points([-0.2, -0.2, 0.8], [0.2, 0.2, 1.2], 0.1)
        snapshot = MapSnapshot.from_points(
            [[0.0, 0.0, 1.0]], observed, resolution=0.1,
            source="unit_test", metadata={"scenario": "fixture"},
        )
        temporary = Path("/tmp/v2_map_snapshot_test.npz")
        snapshot.save(temporary)
        replay = MapSnapshot.load(temporary)
        self.assertEqual(replay.map_hash, snapshot.map_hash)
        np.testing.assert_array_equal(replay.occupied_indices, snapshot.occupied_indices)
        temporary.unlink()

    def test_sensor_map_fusion_is_union_and_rejects_truth_inputs(self):
        metadata = {
            "truth_derived": False,
            "unknown_is_occupied": True,
            "localization_source": "px4_ekf_sensor_fusion",
        }
        first = MapSnapshot.from_points(
            [[0.05, 0.05, 1.05]], [[0.05, 0.05, 1.05], [0.15, 0.05, 1.05]],
            resolution=0.1, metadata=metadata,
            traversed_positions=[[0.0, 0.0, 1.0], [0.0, -1.0, 1.0]],
        )
        second = MapSnapshot.from_points(
            [[0.25, 0.05, 1.05]], [[0.15, 0.05, 1.05], [0.25, 0.05, 1.05]],
            resolution=0.1, metadata=metadata,
            traversed_positions=[[0.0, 0.0, 1.0]],
        )
        fused = fuse_map_snapshots(
            [first, second], sensor_suite=["forward_rgbd", "downward_rgbd"]
        )
        self.assertEqual(len(fused.occupied_indices), 2)
        self.assertEqual(len(fused.observed_indices), 3)
        np.testing.assert_array_equal(fused.traversed_positions, first.traversed_positions)
        self.assertFalse(fused.metadata["formal_experiment_allowed"])
        self.assertFalse(fused.metadata["comparison_independent_survey"])
        self.assertEqual(fused.metadata["input_map_hashes"], [
            first.map_hash, second.map_hash,
        ])
        truth = MapSnapshot.from_points(
            [], [[0.0, 0.0, 1.0]], resolution=0.1,
            metadata=dict(metadata, truth_derived=True),
            traversed_positions=[[0.0, 0.0, 1.0]],
        )
        with self.assertRaises(ValueError):
            fuse_map_snapshots([first, truth])
        supported = fuse_map_snapshots(
            [first, second], observation_support_voxels=1
        )
        self.assertGreater(len(supported.observed_indices), len(fused.observed_indices))
        self.assertEqual(len(supported.occupied_indices), len(fused.occupied_indices))
        with self.assertRaises(ValueError):
            fuse_map_snapshots([first, second], observation_support_voxels=2)
        with self.assertRaises(ValueError):
            fuse_map_snapshots(
                [first, second], comparison_independent_survey=True,
                survey_protocol="fixed_endpoint_scan",
            )
        with self.assertRaises(ValueError):
            fuse_map_snapshots(
                [first, second], comparison_independent_survey=True,
                survey_protocol="fixed_endpoint_scan",
                survey_manifest_sha256="abc",
            )
        cropped = fuse_map_snapshots(
            [first, second], corridor_xy_padding=0.11,
            corridor_z_bounds=(0.9, 1.2),
        )
        self.assertIsNotNone(cropped.metadata["task_corridor_crop"])
        self.assertLessEqual(len(cropped.observed_indices), len(fused.observed_indices))

    def test_slam_world_rejects_unknown_and_detects_occupied_voxels(self):
        observed = observed_box_points([-0.5, -0.5, 0.5], [0.5, 0.5, 1.5], 0.1)
        snapshot = MapSnapshot.from_points(
            [[0.0, 0.0, 1.0]], observed, resolution=0.1,
            source="unit_test",
        )
        world = snapshot.world(inflation=0.0, unknown_is_occupied=True)
        self.assertLess(world.signed_distance([0.05, 0.05, 1.05])[0], 0.0)
        self.assertGreater(world.signed_distance([0.45, 0.45, 1.45])[0], 0.0)
        unknown, source = world.signed_distance([4.0, 4.0, 4.0])
        self.assertLess(unknown, 0.0)
        self.assertEqual(source, "slam_unknown")

    def test_mapper_clears_only_measured_robot_footprint(self):
        mapper = IncrementalVoxelMapper(resolution=0.1)
        mapper.occupied.update({(0, 0, 10), (5, 0, 10)})
        mapper.observed.update(mapper.occupied)
        cleared = mapper.mark_robot_footprint_free(
            [0.05, 0.05, 1.05], radius_xy=0.21, half_height=0.11
        )
        self.assertGreater(cleared, 0)
        self.assertNotIn((0, 0, 10), mapper.occupied)
        self.assertIn((0, 0, 10), mapper.observed)
        self.assertIn((5, 0, 10), mapper.occupied)

    def test_incremental_mapper_marks_rays_observed_but_not_occupied(self):
        mapper = IncrementalVoxelMapper(resolution=0.1, maximum_range=3.0)
        accepted = mapper.integrate([[1.0, 0.0, 1.0]], [0.0, 0.0, 1.0])
        self.assertEqual(accepted, 1)
        snapshot = mapper.snapshot(source="unit_test")
        self.assertEqual(len(snapshot.occupied_indices), 1)
        self.assertGreater(len(snapshot.observed_indices), len(snapshot.occupied_indices))
        world = snapshot.world()
        self.assertGreater(world.signed_distance([0.45, 0.05, 1.05])[0], 0.0)
        self.assertLess(world.signed_distance([1.05, 0.05, 1.05])[0], 0.0)

    def test_no_return_ray_marks_free_without_erasing_occupied(self):
        mapper = IncrementalVoxelMapper(resolution=0.1, maximum_range=3.0)
        mapper.integrate([[1.0, 0.0, 1.0]], [0.0, 0.0, 1.0])
        occupied = next(iter(mapper.occupied))
        accepted = mapper.integrate_free_rays(
            [[2.0, 0.0, 1.0]], [0.0, 0.0, 1.0]
        )
        self.assertEqual(accepted, 1)
        self.assertIn(occupied, mapper.occupied)
        self.assertIn((15, 0, 10), mapper.observed)

    def test_snapshot_round_trips_estimator_centerline(self):
        centerline = np.asarray([[0.0, 0.0, 1.0], [0.0, -0.1, 1.0]])
        snapshot = MapSnapshot.from_points(
            [], [[0.0, 0.0, 1.0]], resolution=0.1,
            traversed_positions=centerline,
        )
        path = Path("/tmp/v2_map_centerline_roundtrip.npz")
        snapshot.save(path)
        replay = MapSnapshot.load(path)
        np.testing.assert_allclose(replay.traversed_positions, centerline)
        self.assertEqual(replay.map_hash, snapshot.map_hash)
        path.unlink()

    def test_registered_cloud_transform_uses_camera_pose_orientation(self):
        # +90 deg about world Z maps camera +X onto world +Y.
        half = np.sqrt(0.5)
        result = transform_points(
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            [3.0, 4.0, 5.0], [0.0, 0.0, half, half],
        )
        np.testing.assert_allclose(
            result, [[3.0, 5.0, 5.0], [1.0, 4.0, 5.0]], atol=1e-8,
        )

    def test_online_map_assembler_requires_audited_metadata_and_matching_hash(self):
        observed = observed_box_points([-0.2, -0.2, 0.8], [0.2, 0.2, 1.2], 0.1)
        source = MapSnapshot.from_points(
            [[0.0, 0.0, 1.0]], observed, resolution=0.1,
            source="online_slam", metadata={
                "localization_source": "vio", "unknown_is_occupied": True,
            },
        )
        assembler = OnlineSnapshotAssembler()
        self.assertIsNone(assembler.update_occupied(source.occupied_points, "world", 1.0))
        self.assertIsNone(assembler.update_observed(source.observed_points, "world", 1.0))
        with self.assertRaises(ValueError):
            assembler.update_metadata({"truth_geometry_access": True})
        replay = assembler.update_metadata({
            "truth_geometry_access": False,
            "resolution": 0.1,
            "source": "online_slam",
            "map_hash": source.map_hash,
            "snapshot_metadata": source.metadata,
        })
        self.assertIsNotNone(replay)
        self.assertEqual(replay.map_hash, source.map_hash)

    def test_common_slam_reference_routes_through_observed_gap(self):
        resolution = 0.2
        observed = observed_box_points([-1.0, -2.0, 0.8], [1.0, 2.0, 1.2], resolution)
        wall = []
        for x in np.arange(-1.0, 1.01, resolution):
            if abs(x) <= 0.35:
                continue
            for z in np.arange(0.8, 1.21, resolution):
                wall.append([x, 0.0, z])
        snapshot = MapSnapshot.from_points(
            wall, observed, resolution=resolution, source="unit_test"
        )
        path, diagnostics = VoxelAStarReferencePlanner(
            snapshot, center_clearance=0.08, z_min=0.8, z_max=1.3,
            time_budget=1.0,
        ).plan([0.1, -1.5, 1.1], [0.1, 1.5, 1.1])
        self.assertGreaterEqual(len(path), 2)
        self.assertGreater(diagnostics["expanded"], 0)
        self.assertEqual(diagnostics["search"], "weighted_astar")
        self.assertAlmostEqual(diagnostics["heuristic_weight"], 1.5)
        self.assertGreaterEqual(diagnostics["preparation_elapsed"], 0.0)
        self.assertGreaterEqual(
            diagnostics["total_elapsed"], diagnostics["elapsed"]
        )
        self.assertGreater(diagnostics["path_points_raw"], 2)
        self.assertLess(np.max(np.abs(path[:, 0])), 0.35)

    def test_online_reference_stops_at_observed_frontier_without_truth_fallback(self):
        resolution = 0.2
        observed = observed_box_points(
            [-0.4, -2.0, 1.0], [0.4, 0.0, 1.4], resolution
        )
        snapshot = MapSnapshot.from_points(
            [], observed, resolution=resolution, source="online_unit_test"
        )
        planner = VoxelAStarReferencePlanner(
            snapshot, center_clearance=0.01, z_min=1.0, z_max=1.5,
            time_budget=1.0,
        )
        with self.assertRaises(ReferencePlanningError):
            planner.plan([0.0, -0.2, 1.2], [0.0, -5.0, 1.2])
        path, diagnostics = planner.plan(
            [0.0, -0.2, 1.2], [0.0, -5.0, 1.2], allow_partial=True
        )
        self.assertFalse(diagnostics["terminal_reached"])
        self.assertEqual(diagnostics["reference_kind"], "observed_frontier")
        self.assertGreater(path[-1, 1], -2.1)
        self.assertLess(path[-1, 1], -1.5)

    def test_all_comparison_methods_share_one_problem_contract(self):
        observed = observed_box_points([-1.0, -1.0, 0.2], [1.0, 2.0, 2.0], 0.1)
        snapshot = MapSnapshot.from_points(
            [], observed, resolution=0.1, source="unit_test"
        )
        world = snapshot.world()
        reference = np.zeros((10, 6))
        reference[:, 1] = np.linspace(0.0, 1.0, len(reference))
        reference[:, 2] = 1.3
        problem = ComparisonProblem(
            reference_states=reference,
            current_state=reference[0],
            terminal_state=reference[-1],
            world=world, arm=self.arm, mass=self.mass, limits=self.limits,
            config=replace(
                self.config, optimizer_time_budget=0.08,
                nominal_speed=0.3, dt_max=3.0, repair_attempts=0,
            ),
            method_time_budget=15.0, random_seed=7,
        )
        for name in METHOD_NAMES:
            result = create_method(name).plan(problem)
            self.assertIsNotNone(result.validation, name)
            self.assertTrue(result.success, "%s: %s" % (
                name, result.validation.failure_reason,
            ))
            self.assertEqual(result.diagnostics["method"], name)
            if name == "informed_rrtstar_wb":
                self.assertEqual(result.diagnostics["random_seed"], 7)
                self.assertGreater(result.diagnostics["improvement_samples_run"], 0)

    def test_schema_v5_method_aliases_resolve_to_honest_schema_v6_names(self):
        aliases = {
            "aet_minjerk": "aet_sequential_minjerk",
            "rrtstar_uam": "informed_rrtstar_wb",
            "am_planner_adapted": "global_batch_wb",
        }
        for old_name, current_name in aliases.items():
            self.assertEqual(create_method(old_name).name, current_name)

    def test_comparison_wall_clock_budget_is_fail_closed(self):
        observed = observed_box_points([-1.0, -1.0, 0.2], [1.0, 2.0, 2.0], 0.1)
        snapshot = MapSnapshot.from_points([], observed, resolution=0.1, source="unit_test")
        reference = np.zeros((8, 6))
        reference[:, 1] = np.linspace(0.0, 0.5, len(reference))
        reference[:, 2] = 1.3
        problem = ComparisonProblem(
            reference_states=reference, current_state=reference[0],
            terminal_state=reference[-1], world=snapshot.world(), arm=self.arm,
            mass=self.mass, limits=self.limits, config=self.config,
            method_time_budget=1e-9,
        )
        result = create_method("fixed_morphology").plan(problem)
        self.assertFalse(result.success)
        self.assertEqual(result.validation.failure_reason, "plan_timeout")

    def test_batched_corridor_edges_match_scalar_feasibility(self):
        observed = observed_box_points([-1.0, -1.0, 0.2], [1.0, 2.0, 2.0], 0.1)
        snapshot = MapSnapshot.from_points(
            [[0.65, 0.2, 1.1]], observed, resolution=0.1, source="unit_test"
        )
        reference = np.zeros((8, 6))
        reference[:, 1] = np.linspace(0.0, 0.5, len(reference))
        reference[:, 2] = 1.3
        problem = ComparisonProblem(
            reference_states=reference, current_state=reference[0],
            world=snapshot.world(), arm=self.arm, mass=self.mass,
            limits=self.limits, config=self.config,
        )
        first = np.repeat(reference[:1], 6, axis=0)
        second = first.copy()
        second[:, 1] = np.linspace(0.05, 0.45, len(second))
        second[:, THETA] = np.deg2rad(np.linspace(0.0, 90.0, len(second)))
        second[:, PHI] = np.deg2rad(np.linspace(-170.0, 170.0, len(second)))
        batched = RrtStarUamMethod._edges_feasible(problem, first, second)
        scalar = np.asarray([
            RrtStarUamMethod._edge_feasible(problem, a, b)
            for a, b in zip(first, second)
        ])
        np.testing.assert_array_equal(batched, scalar)

    def test_batched_whole_body_geometry_matches_scalar_geometry(self):
        from uav_continuum_whole_body_v2.geometry import (
            coupling_metrics, coupling_metrics_batch, world_arm_points,
            world_arm_points_batch, world_body_points, world_body_points_batch,
        )
        states = np.asarray([
            [0.1, -0.2, 1.3, 0.0, 0.0, 0.0],
            [-0.3, 0.4, 1.6, 0.7, np.deg2rad(55.0), -1.2],
            [0.2, 0.8, 1.1, -2.0, np.deg2rad(118.0), 2.7],
        ])
        np.testing.assert_allclose(
            world_body_points_batch(states, self.limits),
            np.asarray([world_body_points(state, self.limits) for state in states]),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            world_arm_points_batch(states, self.arm),
            np.asarray([world_arm_points(state, self.arm) for state in states]),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            coupling_metrics_batch(states, self.arm, self.mass),
            np.asarray([
                coupling_metrics(state, self.arm, self.mass)["com_offset"]
                for state in states
            ]),
            atol=1e-12,
        )

    def test_configuration_corridor_honours_expired_deadline(self):
        observed = observed_box_points([-1.0, -1.0, 0.2], [1.0, 2.0, 2.0], 0.1)
        snapshot = MapSnapshot.from_points(
            [], observed, resolution=0.1, source="unit_test"
        )
        reference = np.zeros((8, 6))
        reference[:, 1] = np.linspace(0.0, 0.5, len(reference))
        reference[:, 2] = 1.3
        problem = ComparisonProblem(
            reference_states=reference, current_state=reference[0],
            terminal_state=reference[-1], world=snapshot.world(),
            arm=self.arm, mass=self.mass, limits=self.limits, config=self.config,
        )
        corridor, diagnostics = AmPlannerAdaptedMethod._configuration_corridor(
            problem, reference, deadline=0.0
        )
        self.assertIsNone(corridor)
        self.assertEqual(diagnostics["failure"], "plan_timeout")
        self.assertEqual(diagnostics["stage"], "corridor_layers")

    def test_comparison_factory_rejects_unregistered_method(self):
        with self.assertRaises(ValueError):
            create_method("ours_but_weaken_everyone_else")

    def test_parameterized_scenarios_are_paired_and_development_maps_are_audited(self):
        first = generate_scenario("multi_passage", "hard", seed=17)
        replay = generate_scenario("multi_passage", "hard", seed=17)
        other = generate_scenario("multi_passage", "hard", seed=18)
        self.assertEqual(first.boxes, replay.boxes)
        self.assertNotEqual(first.boxes, other.boxes)
        self.assertEqual(first.task["tip_tolerance_m"], 0.08)
        self.assertEqual(
            generate_scenario("clutter_reach", "medium", seed=17).task[
                "tip_tolerance_m"
            ],
            0.03,
        )
        snapshot = synthetic_development_snapshot(first, resolution=0.2)
        self.assertTrue(snapshot.metadata["truth_derived"])
        self.assertFalse(snapshot.metadata["formal_experiment_allowed"])
        self.assertEqual(snapshot.metadata["scenario"], first.name)

    def test_map_audit_promotes_only_nontruth_accurate_snapshots(self):
        spec = generate_scenario("passage_sweep", "easy", seed=4)
        synthetic = synthetic_development_snapshot(spec, resolution=0.2)
        candidate = MapSnapshot(
            resolution=synthetic.resolution,
            occupied_indices=synthetic.occupied_indices,
            observed_indices=synthetic.observed_indices,
            source="recorded_slam",
            metadata={
                "localization_source": "vins_fusion",
                "truth_derived": False,
                "formal_experiment_allowed": False,
                "comparison_independent_survey": True,
                "survey_protocol": "unit_test_fixed_protocol",
                "survey_protocol_manifest_sha256": "fixture",
            },
            traversed_positions=np.linspace(
                np.asarray(spec.task["start_state"][:3], dtype=float),
                np.asarray(spec.task["preferred_standoff"][:3], dtype=float),
                30,
            ),
        )
        passed = audit_snapshot(candidate, spec.oracle, spec.task, thresholds={
            "occupied_surface_recall_min": 0.75,
            "task_corridor_observed_coverage_min": 0.50,
        })
        self.assertTrue(passed["passed"], passed)
        truth_claim = MapSnapshot(
            resolution=candidate.resolution,
            occupied_indices=candidate.occupied_indices,
            observed_indices=candidate.observed_indices,
            source=candidate.source,
            metadata={"localization_source": "gazebo_truth", "truth_derived": True},
        )
        rejected = audit_snapshot(truth_claim, spec.oracle, spec.task)
        self.assertFalse(rejected["passed"])
        self.assertIn("truth_derived_map", rejected["failures"])

    def test_map_audit_rejects_terminal_unknown_hole_hidden_by_route_average(self):
        spec = generate_scenario("passage_sweep", "easy", seed=4)
        synthetic = synthetic_development_snapshot(spec, resolution=0.2)
        terminal = np.r_[
            np.asarray(spec.task["preferred_standoff"][:4], dtype=float),
            np.radians(spec.task["terminal_shape_deg"]),
        ]
        terminal_indices = voxel_indices(
            world_arm_points(terminal, self.arm), synthetic.resolution
        )
        occupied = {tuple(row) for row in synthetic.occupied_indices.tolist()}
        removable = next(
            tuple(row) for row in terminal_indices.tolist()
            if tuple(row) not in occupied
        )
        observed = np.asarray([
            row for row in synthetic.observed_indices.tolist()
            if tuple(row) != removable
        ], dtype=np.int64)
        candidate = MapSnapshot(
            resolution=synthetic.resolution,
            occupied_indices=synthetic.occupied_indices,
            observed_indices=observed,
            source="recorded_slam",
            metadata={
                "localization_source": "vins_fusion",
                "truth_derived": False,
                "formal_experiment_allowed": False,
                "comparison_independent_survey": True,
                "survey_protocol": "unit_test_fixed_protocol",
                "survey_protocol_manifest_sha256": "fixture",
            },
            traversed_positions=np.linspace(
                np.asarray(spec.task["start_state"][:3], dtype=float),
                np.asarray(spec.task["preferred_standoff"][:3], dtype=float),
                30,
            ),
        )
        report = audit_snapshot(candidate, spec.oracle, spec.task)
        self.assertFalse(report["passed"])
        self.assertIn("terminal_whole_body_observed_coverage", report["failures"])

    def test_map_audit_does_not_require_observation_through_known_obstacle(self):
        snapshot = MapSnapshot.from_points(
            occupied_points=[[0.0, 0.0, 1.0]],
            observed_points=[[0.0, 0.0, 1.0], [0.0, 0.0, 1.1]],
            resolution=0.1,
            source="unit_test",
        )
        # Once the leading surface sample is occupied, an unknown sample
        # behind it belongs to an already infeasible full-body state. The map
        # audit must not demand a depth camera to observe through the solid.
        self.assertTrue(_planner_known_collision(
            snapshot, [[0.0, 0.0, 1.0], [0.0, 0.0, 0.9]]
        ))
        # Unknown without any measured occupied intersection remains subject
        # to the strict observability gate and is not classified as collision.
        self.assertFalse(_planner_known_collision(
            snapshot, [[0.5, 0.0, 1.0], [0.5, 0.0, 0.9]]
        ))

    def test_realtime_profile_has_interior_dynamic_reserve(self):
        package = Path(__file__).resolve().parents[1]
        raw, config, limits, _ = load_planner_configuration(
            package / "config" / "planner_v2_realtime.yaml"
        )
        self.assertAlmostEqual(config.design_velocity_ratio, 0.94)
        self.assertAlmostEqual(np.rad2deg(config.theta_design_margin), 2.0)
        self.assertGreater(config.dynamic_repair_reserve, 1.0)
        self.assertGreater(config.dt_max, 5.5)
        self.assertEqual(raw.get("transit_reference_shape_deg"), [0.0, 0.0])
        self.assertLess(
            config.design_velocity_ratio * limits.velocity_max,
            limits.velocity_max,
        )

    def test_largest_feasible_ratio_finds_guard_boundary(self):
        ratio = largest_feasible_ratio(lambda value: 0.4 * value <= 0.1, iterations=16)
        self.assertAlmostEqual(ratio, 0.25, places=4)
        self.assertEqual(largest_feasible_ratio(lambda value: True), 1.0)
        self.assertEqual(largest_feasible_ratio(lambda value: value <= 0.0), 0.0)

    def test_tracking_progress_scale_has_safe_continuous_boundaries(self):
        self.assertEqual(tracking_progress_scale(0.005, 0.008, 0.020), 1.0)
        self.assertEqual(tracking_progress_scale(0.020, 0.008, 0.020), 0.0)
        self.assertAlmostEqual(tracking_progress_scale(0.014, 0.008, 0.020), 0.5)
        with self.assertRaises(ValueError):
            tracking_progress_scale(0.0, 0.02, 0.02)

    def test_clearance_tracking_budget_preserves_hard_line_and_caps_open_space(self):
        self.assertAlmostEqual(clearance_tracking_budget(0.10, 0.06, 0.01, 0.10), 0.03)
        self.assertAlmostEqual(clearance_tracking_budget(0.08, 0.02, 0.02, 0.10), 0.04)
        self.assertAlmostEqual(clearance_tracking_budget(0.50, 0.02, 0.02, 0.10), 0.10)
        self.assertEqual(clearance_tracking_budget(0.02, 0.02, 0.01, 0.10), 0.0)

    def test_clearance_progress_scale_does_not_deadlock_small_positive_budget(self):
        self.assertEqual(clearance_progress_scale(0.0, 0.01, 0.005), 1.0)
        self.assertGreater(clearance_progress_scale(0.002, 0.01, 0.005), 0.0)
        self.assertEqual(clearance_progress_scale(0.005, 0.01, 0.005), 0.0)
        self.assertEqual(clearance_progress_scale(0.0, 0.01, 0.0), 0.0)
        self.assertAlmostEqual(
            clearance_progress_scale(0.0, 0.01, 0.012, 0.060), 0.2
        )

    def test_rolling_prediction_uses_guarded_phase_clock(self):
        self.assertAlmostEqual(projected_phase_elapsed(11.14, 10.0, 0.0, 60.0), 11.14)
        self.assertAlmostEqual(projected_phase_elapsed(11.14, 10.0, 0.4, 60.0), 15.14)
        self.assertAlmostEqual(projected_phase_elapsed(58.0, 10.0, 1.0, 60.0), 60.0)
        self.assertAlmostEqual(projected_phase_elapsed(4.0, -2.0, 0.5, 60.0), 4.0)

    def test_commit_progress_uses_xyz_arc_length_at_executor_phase(self):
        # Production TrajectoryData stores time as ndarray; keep this test on
        # the exact runtime container type.
        times = np.asarray([0.0, 1.0, 2.0, 3.0])
        states = np.asarray([
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.3, 0.0, 1.0, 0.5, 0.2, 0.0],
            [0.3, 0.4, 1.0, 1.0, 0.4, 0.0],
            [0.3, 0.4, 1.5, 1.5, 0.6, 0.0],
        ])
        self.assertAlmostEqual(
            trajectory_translation_progress(times, states, 0.5), 0.15
        )
        self.assertAlmostEqual(
            trajectory_translation_progress(times, states, 1.5), 0.50
        )
        self.assertAlmostEqual(
            trajectory_translation_progress(times, states, 9.0), 1.20
        )
        # Pure yaw/arm changes do not falsely unlock a metre-based commit.
        shape_only = states.copy()
        shape_only[:, :3] = shape_only[0, :3]
        self.assertAlmostEqual(
            trajectory_translation_progress(times, shape_only, 3.0), 0.0
        )

    def test_rolling_switch_matches_a_bounded_certified_suffix(self):
        times = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
        positions = np.asarray([
            [0.00, 0.0, 1.0], [0.10, 0.0, 1.0], [0.20, 0.0, 1.0],
            [0.30, 0.0, 1.0], [0.40, 0.0, 1.0],
        ])
        elapsed, index, advance, gaps = bounded_switch_match(
            times, positions,
            targets=([0.24, 0.0, 1.0], [0.26, 0.0, 1.0]),
            maximum_advance=0.35,
        )
        self.assertEqual(index, 2)
        self.assertAlmostEqual(elapsed, 2.0)
        self.assertAlmostEqual(advance, 0.20)
        self.assertAlmostEqual(max(gaps), 0.06)
        # The closer 0.4 m self-intersection/shortcut is outside the certified
        # splice allowance and cannot be selected.
        elapsed, index, advance, _ = bounded_switch_match(
            times, positions, targets=([0.40, 0.0, 1.0],), maximum_advance=0.25
        )
        self.assertEqual(index, 2)
        self.assertAlmostEqual(elapsed, 2.0)
        self.assertAlmostEqual(advance, 0.20)

    def test_rolling_switch_match_rejects_malformed_inputs(self):
        with self.assertRaises(ValueError):
            bounded_switch_match([], [], ([0.0, 0.0, 0.0],), 0.3)
        with self.assertRaises(ValueError):
            bounded_switch_match([0.0], [[0.0, 0.0, 0.0]], (), 0.3)

    def test_terminal_completion_does_not_require_nonexistent_phase_progress(self):
        self.assertFalse(reference_converged(False, 0.42, 0.06, 0.18))
        self.assertTrue(reference_converged(True, 0.42, 0.06, 0.18))
        self.assertFalse(reference_converged(True, 1.0, 0.19, 0.18))
        self.assertTrue(reference_converged(False, 1.0, 0.06, 0.18))

    def test_rolling_switch_preserves_governed_phase_rate(self):
        self.assertAlmostEqual(continuous_switch_phase_rate(0.37), 0.37)
        self.assertAlmostEqual(continuous_switch_phase_rate(1.2), 1.0)
        self.assertAlmostEqual(continuous_switch_phase_rate(-0.2), 0.0)
        self.assertAlmostEqual(
            continuous_switch_phase_rate(0.37, scheduled_restart=True), 0.0
        )

    def test_terminal_endpoint_has_zero_feedforward(self):
        velocity, yaw_rate = stationary_endpoint_feedforward(
            True, [0.1, -0.2, 0.3], 0.4
        )
        self.assertEqual(velocity, (0.0, 0.0, 0.0))
        self.assertEqual(yaw_rate, 0.0)
        velocity, yaw_rate = stationary_endpoint_feedforward(
            False, [0.1, -0.2, 0.3], 0.4
        )
        self.assertEqual(velocity, (0.1, -0.2, 0.3))
        self.assertEqual(yaw_rate, 0.4)

    def test_clamped_spline_is_rest_to_rest_by_construction(self):
        controls = np.zeros((6, 6), dtype=float)
        controls[:, 0] = np.linspace(0.0, 1.0, len(controls))
        spline = SplineTrajectory(controls, np.ones(3, dtype=float))
        endpoints = spline.evaluate([0.0, spline.duration], derivative=0)
        endpoint_velocity = spline.evaluate(
            [0.0, spline.duration], derivative=1
        )
        np.testing.assert_allclose(endpoints[:, 0], [0.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(endpoint_velocity, 0.0, atol=1e-12)
        self.assertGreater(spline.duration, np.sum(spline.span_durations))

    def test_rolling_spline_inherits_initial_rate_and_stops_at_end(self):
        controls = np.zeros((6, 6), dtype=float)
        controls[:, 0] = np.linspace(0.0, 1.0, len(controls))
        durations = np.ones(3, dtype=float)
        desired_rate = np.asarray([0.12, -0.03, 0.01, 0.02, 0.04, -0.01])
        controls[1] = controls[0] + desired_rate * durations[0] / 3.0
        spline = SplineTrajectory(
            controls, durations, stationary_start=False, stationary_end=True
        )
        endpoint_velocity = spline.evaluate(
            [0.0, spline.duration], derivative=1
        )
        np.testing.assert_allclose(endpoint_velocity[0], desired_rate, atol=1e-12)
        np.testing.assert_allclose(endpoint_velocity[1], 0.0, atol=1e-12)
        self.assertAlmostEqual(
            spline.duration, np.sum(durations) + 0.5 * spline.finish_ramp_duration
        )
        geometric_knots = np.r_[0.0, np.cumsum(durations)]
        mapped, _rate, _acceleration, _jerk = spline._phase(spline.span_boundaries)
        np.testing.assert_allclose(mapped, geometric_knots, atol=1e-10)

    def test_rest_to_rest_phase_chain_rule_and_geometry(self):
        controls = np.zeros((7, 6), dtype=float)
        controls[:, 0] = np.linspace(0.0, 1.5, len(controls))
        controls[:, 2] = 1.0 + 0.1 * np.sin(np.linspace(0.0, np.pi, len(controls)))
        spline = SplineTrajectory(controls, np.full(4, 1.5, dtype=float))
        times = np.linspace(0.2, spline.duration - 0.2, 31)
        phase, rate, _acceleration, _jerk = spline._phase(times)
        self.assertTrue(np.all(np.diff(phase) > 0.0))
        self.assertGreaterEqual(float(np.min(rate)), 0.0)
        self.assertLessEqual(float(np.max(rate)), 1.0)
        step = 1e-5
        numerical_velocity = (
            spline.evaluate(times + step, 0) - spline.evaluate(times - step, 0)
        ) / (2.0 * step)
        np.testing.assert_allclose(
            spline.evaluate(times, 1), numerical_velocity, rtol=2e-5, atol=2e-6
        )
        geometric_knots = np.r_[0.0, np.cumsum(spline.span_durations)]
        execution_knots = spline.span_boundaries
        mapped, _rate, _acceleration, _jerk = spline._phase(execution_knots)
        np.testing.assert_allclose(mapped, geometric_knots, atol=1e-10)

    def test_optimizer_basis_preserves_measured_initial_state(self):
        optimizer = WholeBodyOptimizer(
            HybridDistanceWorld(BoxWorld([])), self.arm, self.mass,
            self.limits, self.config, validator=None,
        )
        basis = optimizer._make_basis(self.config.knot_count)
        np.testing.assert_allclose(basis[0], 0.0)

    def test_vector_box_distance_matches_scalar(self):
        box = BoxObstacle("box", (0.0, 0.0, 0.0), (2.0, 2.0, 2.0))
        points = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
        vector = box.signed_distances(points)
        scalar = np.asarray([box.signed_distance(point) for point in points])
        np.testing.assert_allclose(vector, scalar)
        self.assertLess(vector[0], 0.0)

    def test_oriented_box_distance_and_surface_fixture_are_exact(self):
        box = BoxObstacle(
            "rotated", (0.4, -0.2, 1.0), (2.0, 0.4, 1.2),
            yaw=np.deg2rad(31.0),
        )
        rng = np.random.RandomState(31)
        points = rng.uniform([-1.0, -1.0, 0.0], [2.0, 1.0, 2.0], (80, 3))
        np.testing.assert_allclose(
            box.signed_distances(points),
            [box.signed_distance(point) for point in points], atol=1e-12,
        )
        fixture = {
            "center": list(box.center), "size": list(box.size), "yaw_deg": 31.0,
        }
        surface = box_surface_points(fixture, resolution=0.1)
        self.assertLess(float(np.max(np.abs(box.signed_distances(surface)))), 1e-12)

    def test_grouped_distance_matches_independent_scalar_queries(self):
        boxes = BoxWorld.from_dicts([
            {"name": "left", "center": [-0.6, 0.0, 1.0], "size": [0.4, 1.2, 1.0]},
            {"name": "right", "center": [0.8, -0.4, 1.4], "size": [0.5, 0.6, 1.2]},
        ])
        cloud = PointCloudWorld([
            [-0.2, -0.3, 0.9], [0.2, -0.1, 1.1], [1.4, 0.2, 1.5],
        ], inflation=0.03)
        world = HybridDistanceWorld(boxes, point_world=cloud)
        rng = np.random.RandomState(19)
        groups = rng.uniform([-1.2, -1.0, 0.5], [1.5, 0.8, 1.9], size=(17, 23, 3))
        grouped_distance, grouped_name = world.minimum_distances_grouped(groups)
        scalar = [world.minimum_distance(group) for group in groups]
        np.testing.assert_allclose(grouped_distance, [item[0] for item in scalar], atol=1e-12)
        self.assertEqual(grouped_name, [item[1] for item in scalar])

    def test_local_point_cloud_crop_preserves_padded_roi_queries(self):
        rng = np.random.RandomState(23)
        points = rng.uniform([-5.0, -5.0, -2.0], [5.0, 5.0, 4.0], size=(4000, 3))
        world = PointCloudWorld(points, inflation=0.02)
        minimum = np.asarray([-1.0, -1.5, 0.8])
        maximum = np.asarray([1.2, 0.7, 1.8])
        padding = 1.0
        local = world.filtered(minimum, maximum, padding=padding)
        queries = rng.uniform(minimum - 0.3, maximum + 0.3, size=(700, 3))
        full_distance = world.tree.query(queries, k=1)[0] - world.inflation
        local_distance = local.tree.query(queries, k=1)[0] - local.inflation
        # Queries are at least 0.7 m inside the crop boundary.  Any excluded
        # point is farther than that, while this dense fixture has much nearer
        # neighbours, so the exact nearest result must be identical.
        self.assertLess(float(np.max(full_distance)), 0.7)
        np.testing.assert_allclose(local_distance, full_distance, atol=1e-12)

    def test_optimizer_batched_clearance_matches_scalar_geometry(self):
        boxes = BoxWorld.from_dicts([
            {"name": "wall", "center": [0.4, -0.4, 1.2], "size": [0.3, 1.0, 1.4]},
        ])
        world = HybridDistanceWorld(
            boxes, point_world=PointCloudWorld([[0.0, -0.8, 0.9], [0.8, 0.1, 1.5]])
        )
        optimizer = WholeBodyOptimizer(
            world, self.arm, self.mass, self.limits, self.config, validator=None
        )
        states = np.zeros((11, 6))
        states[:, 0] = np.linspace(-0.5, 0.8, len(states))
        states[:, 1] = np.linspace(0.3, -0.9, len(states))
        states[:, 2] = 1.3
        states[:, 3] = np.linspace(0.0, -0.7, len(states))
        states[:, 4] = np.linspace(0.0, np.deg2rad(80.0), len(states))
        batched = optimizer._clearance_samples(states)
        for current, (body, arm, coupling) in zip(states, batched):
            scalar_body = min(
                10.0, world.minimum_distance(world_body_points(current, self.limits))[0]
            )
            scalar_arm = min(
                10.0, world.minimum_distance(world_arm_points(current, self.arm))[0]
            ) - self.arm.radius
            scalar_coupling = coupling_metrics(current, self.arm, self.mass)
            self.assertAlmostEqual(body, scalar_body)
            self.assertAlmostEqual(arm, scalar_arm)
            self.assertAlmostEqual(coupling["com_offset"], scalar_coupling["com_offset"])

    def test_optimizer_seed_edge_uses_configured_limits_without_hidden_guard(self):
        limits = Limits(body_margin=0.14, arm_margin=0.06)
        self.assertFalse(WholeBodyOptimizer._seed_edge_violates_limits(0.062, 0.145, limits))
        self.assertTrue(WholeBodyOptimizer._seed_edge_violates_limits(0.0599, 0.145, limits))
        self.assertTrue(WholeBodyOptimizer._seed_edge_violates_limits(0.062, 0.1399, limits))

    def test_collision_translation_repair_is_ranked_and_preserves_boundaries(self):
        class ProxyOptimizer(WholeBodyOptimizer):
            def _collision_proxy(self, controls, span_durations, risk_spans):
                # A synthetic obstacle whose body and arm clearance both
                # increase along +x.  The test isolates deterministic repair
                # ranking from a particular scene geometry.
                slack = float(np.mean(controls[2:6, 0])) - 0.02
                return slack, slack + 0.01

        optimizer = ProxyOptimizer(
            HybridDistanceWorld(BoxWorld([])), self.arm, self.mass,
            self.limits, self.config, validator=None,
        )
        controls = np.zeros((8, 6))
        controls[:, 2] = 1.3
        durations = np.ones(5)
        candidates = optimizer._translation_repair_candidates(
            controls, durations, risk_spans=[2], preserve_terminal=True
        )
        self.assertTrue(candidates)
        label, repaired, repaired_durations = candidates[0]
        self.assertIn("whole_body_shift_", label)
        np.testing.assert_allclose(repaired[0], controls[0])
        np.testing.assert_allclose(repaired[-1], controls[-1])
        self.assertGreater(float(np.mean(repaired[2:6, 0])), 0.0)
        self.assertTrue(np.all(repaired_durations >= durations))
        body_slack, arm_slack = optimizer._collision_proxy(
            repaired, repaired_durations, [2]
        )
        self.assertGreaterEqual(min(body_slack, arm_slack), 0.004)

    def test_risk_shape_candidates_preserve_non_grid_task_boundary(self):
        planner = WholeBodySeedPlanner(
            HybridDistanceWorld(BoxWorld([])), self.arm, self.mass,
            self.limits, self.config,
        )
        reference = np.zeros(6)
        reference[4] = np.deg2rad(90.0)
        reference[5] = -np.pi
        candidates = planner._shape_candidates(reference, True, (np.deg2rad(75.0), 0.0))
        self.assertTrue(any(
            abs(theta - reference[4]) < 1e-9 and abs(wrap_angle(phi - reference[5])) < 1e-9
            for theta, phi in candidates
        ))

    def test_adaptive_phi_locks_redundant_bending_plane(self):
        wall = BoxWorld.from_dicts([
            {"name": "lower", "center": [0, 0, 0.49], "size": [4, 0.34, 0.98]},
            {"name": "upper", "center": [0, 0, 3.235], "size": [4, 0.34, 3.27]},
        ])
        reference = np.zeros((24, 6))
        reference[:, 1] = np.linspace(0.6, -1.0, len(reference))
        reference[:, 2] = 1.30
        config = replace(
            self.config, adaptive_phi=True,
            adaptive_phi_clearance_reserve=0.0,
            locked_phi_candidate_limit=6,
        )
        seed = WholeBodySeedPlanner(
            HybridDistanceWorld(wall), self.arm, self.mass, self.limits, config
        ).plan(reference, reference[0])
        self.assertEqual(seed.diagnostics["shape_mode"], "theta_only_fixed_phi")
        np.testing.assert_allclose(seed.states[:, PHI], seed.states[0, PHI], atol=1e-12)
        self.assertGreater(np.max(seed.states[:, THETA]), np.deg2rad(20.0))

    def test_terminal_task_always_unlocks_phi(self):
        reference = np.zeros((12, 6))
        reference[:, 1] = np.linspace(0.0, -1.0, len(reference))
        reference[:, 2] = 1.30
        terminal = reference[-1].copy()
        terminal[THETA] = np.deg2rad(90.0)
        terminal[PHI] = -np.pi
        config = replace(self.config, adaptive_phi=True)
        seed = WholeBodySeedPlanner(
            HybridDistanceWorld(BoxWorld([])), self.arm, self.mass, self.limits, config
        ).plan(reference, reference[0], terminal_state=terminal)
        self.assertEqual(seed.diagnostics["shape_mode"], "theta_phi")
        self.assertAlmostEqual(wrap_angle(seed.states[-1, PHI] - terminal[PHI]), 0.0)

    def test_optimizer_removes_locked_phi_coordinates(self):
        optimizer = WholeBodyOptimizer(
            HybridDistanceWorld(BoxWorld([])), self.arm, self.mass,
            self.limits, self.config, validator=None,
        )
        optimizer._phi_locked = True
        bounds = optimizer._bounds(self.config.knot_count - 3)
        for basis in range(self.config.optimization_basis_count):
            self.assertEqual(bounds[6 * basis + PHI], (0.0, 0.0))

    def test_cylindrical_body_uses_independent_height(self):
        state = np.asarray([0.0, 0.0, 1.30, 0.0, 0.0, 0.0])
        points = world_body_points(state, self.limits)
        self.assertAlmostEqual(np.max(points[:, 2]) - np.min(points[:, 2]), self.limits.body_height)
        self.assertAlmostEqual(np.max(points[:, 0]) - np.min(points[:, 0]), 2.0 * self.limits.body_radius)

    def test_center_of_mass_coupling_is_shape_dependent(self):
        straight = coupling_metrics([0, 0, 1.3, 0, 0, 0], self.arm, self.mass)
        bent = coupling_metrics([0, 0, 1.3, 0, np.deg2rad(90), 0], self.arm, self.mass)
        self.assertLess(straight["com_offset"], 1e-8)
        self.assertGreater(bent["com_offset"], straight["com_offset"] + 0.01)
        self.assertAlmostEqual(bent["gravity_torque"], self.mass.total * 9.81 * bent["com_offset"])

    def test_validator_rejects_and_accepts_independently(self):
        controls = np.zeros((8, 6))
        controls[:, 0] = np.linspace(-1.0, 1.0, 8)
        controls[:, 2] = 1.3
        spline = SplineTrajectory(controls, np.full(5, 2.0))
        open_validator = SweptVolumeValidator(HybridDistanceWorld(BoxWorld([])), self.arm, self.mass, self.limits, self.config)
        self.assertTrue(open_validator.validate(spline).feasible)
        wall = BoxWorld.from_dicts([{"name": "wall", "center": [0, 0, 1.3], "size": [0.1, 3, 3]}])
        blocked = SweptVolumeValidator(HybridDistanceWorld(wall), self.arm, self.mass, self.limits, self.config).validate(spline)
        self.assertFalse(blocked.feasible)
        self.assertIn(blocked.failure_reason, ("body_collision", "arm_collision"))

    def test_center_keepout_is_not_double_inflated_as_physical_geometry(self):
        keepout = BoxWorld.from_dicts([
            {"name": "task_center_keepout", "center": [0.0, -1.0, 1.3], "size": [0.6, 0.22, 0.78]}
        ])
        controls = np.zeros((8, 6))
        controls[:, 0] = np.linspace(-0.1, 0.1, 8)
        controls[:, 1] = -0.63  # center is outside, although the body envelope overlaps the abstract box
        controls[:, 2] = 1.3
        validator = SweptVolumeValidator(
            HybridDistanceWorld(BoxWorld([])), self.arm, self.mass, self.limits, self.config,
            center_keepout_world=keepout,
        )
        self.assertTrue(validator.validate(SplineTrajectory(controls, np.full(5, 2.0))).feasible)
        controls[:, 1] = -0.95
        result = validator.validate(SplineTrajectory(controls, np.full(5, 2.0)))
        self.assertFalse(result.feasible)
        self.assertEqual(result.failure_reason, "body_collision")
        self.assertTrue(result.metrics["body_obstacle_min"].startswith("center_keepout:"))

    def test_terminal_reach_is_certified_by_independent_validator(self):
        controls = np.zeros((8, 6))
        controls[:, 2] = 1.3
        spline = SplineTrajectory(controls, np.full(5, 2.0))
        target = world_arm_points(controls[-1], self.arm)[-1]
        safe = SweptVolumeValidator(
            HybridDistanceWorld(BoxWorld([])), self.arm, self.mass, self.limits, self.config,
            terminal_target=target, terminal_tolerance=0.05,
        ).validate(spline)
        self.assertTrue(safe.feasible)
        self.assertLess(safe.metrics["terminal_tip_error"], 1e-9)
        unsafe = SweptVolumeValidator(
            HybridDistanceWorld(BoxWorld([])), self.arm, self.mass, self.limits, self.config,
            terminal_target=target + np.asarray([0.2, 0.0, 0.0]), terminal_tolerance=0.05,
        ).validate(spline)
        self.assertFalse(unsafe.feasible)
        self.assertEqual(unsafe.failure_reason, "terminal_tip_error")

    def test_terminal_seed_and_unified_timing_reach_selected_candidate(self):
        world = HybridDistanceWorld(BoxWorld([]))
        terminal = np.asarray([
            2.52257657, -11.95891094, 1.08, np.deg2rad(-104.0),
            np.deg2rad(90.0), -np.pi,
        ])
        start = terminal.copy()
        start[1] += 0.60
        start[4:] = 0.0
        reference = np.asarray([
            (1.0 - ratio) * start + ratio * terminal for ratio in np.linspace(0.0, 1.0, 8)
        ])
        reference[:, 4] = np.linspace(0.0, terminal[4], 8)
        reference[:, 5] = np.linspace(0.0, terminal[5], 8)
        config = PlannerConfig(optimizer_time_budget=0.2, nominal_speed=0.30, dt_max=3.0)
        seed = WholeBodySeedPlanner(
            world, self.arm, self.mass, self.limits, config
        ).plan(reference, start, terminal_state=terminal)
        np.testing.assert_allclose(seed.states[-1], terminal, atol=1e-9)
        seed.diagnostics["terminal_shape"] = terminal[4:].tolist()
        target = world_arm_points(terminal, self.arm)[-1]
        validator = SweptVolumeValidator(
            world, self.arm, self.mass, self.limits, config,
            terminal_target=target, terminal_tolerance=0.05,
        )
        result = WholeBodyOptimizer(
            world, self.arm, self.mass, self.limits, config, validator
        ).optimize(seed, reference)
        self.assertTrue(result.success, result.validation.failure_reason)
        self.assertLessEqual(result.validation.metrics["terminal_tip_error"], 0.05)
        self.assertLessEqual(
            result.validation.metrics["theta_rate_max"], self.limits.theta_rate_max + 1e-6
        )
        self.assertLessEqual(
            result.validation.metrics["phi_rate_max"], self.limits.phi_rate_max + 1e-6
        )

    def test_optimizer_decode_hard_projects_bounded_spline_coordinates(self):
        world = HybridDistanceWorld(BoxWorld([]))
        optimizer = WholeBodyOptimizer(
            world, self.arm, self.mass, self.limits, self.config, validator=None
        )
        optimizer._seed_base = np.zeros((8, 6))
        optimizer._seed_base[:, 2] = self.limits.z_max
        optimizer._seed_base[:, 4] = self.arm.theta_max
        optimizer._basis = optimizer._make_basis(8)
        corrections = np.zeros((self.config.optimization_basis_count, 6))
        corrections[:, 2] = 1.0
        corrections[:, 4] = np.deg2rad(70.0)
        vector = np.concatenate((corrections.ravel(), np.full(5, 1.0)))
        controls, durations = optimizer._decode(vector)
        self.assertLessEqual(np.max(controls[:, 2]), self.limits.z_max)
        self.assertLessEqual(np.max(controls[:, 4]), self.arm.theta_max)
        self.assertAlmostEqual(controls[0, 4], self.arm.theta_max)
        self.assertLessEqual(
            np.max(controls[1:, 4]),
            self.arm.theta_max - self.config.theta_design_margin + 1e-12,
        )
        sampled = SplineTrajectory(controls, durations).evaluate(np.linspace(0.0, 5.0, 201))
        self.assertLessEqual(np.max(sampled[:, 2]), self.limits.z_max + 1e-9)
        self.assertLessEqual(np.max(sampled[:, 4]), self.arm.theta_max + 1e-9)

    def test_risk_is_dilated_for_spline_transition(self):
        wall = BoxWorld.from_dicts([
            {"name": "lower", "center": [0, 0, 0.49], "size": [4, 0.34, 0.98]},
            {"name": "upper", "center": [0, 0, 3.235], "size": [4, 0.34, 3.27]},
        ])
        reference = np.zeros((16, 6))
        reference[:, 1] = np.linspace(1.0, -1.0, len(reference))
        reference[:, 2] = 1.30
        seed = WholeBodySeedPlanner(HybridDistanceWorld(wall), self.arm, self.mass, self.limits, self.config).plan(reference, reference[0])
        raw = seed.diagnostics["raw_risk_knots"]
        self.assertTrue(raw)
        self.assertGreaterEqual(len(seed.risk_knots), len(raw))
        self.assertTrue(any(index not in raw for index in seed.risk_knots))

    def test_seed_conditions_large_initial_yaw_change(self):
        reference = np.zeros((16, 6))
        reference[:, 1] = np.linspace(0.0, -3.0, len(reference))
        reference[:, 2] = 1.3
        reference[1:, 3] = -0.5 * np.pi
        planner = WholeBodySeedPlanner(
            HybridDistanceWorld(BoxWorld([])), self.arm, self.mass, self.limits, self.config
        )
        seed = planner.plan(reference, reference[0])
        spline = SplineTrajectory(seed.states, np.full(5, self.config.dt_max))
        yaw_rate = np.max(np.abs(spline.evaluate(np.linspace(0.0, spline.duration, 200), 1)[:, 3]))
        self.assertLessEqual(yaw_rate, self.limits.yaw_rate_max + 1e-6)

    def test_shape_rate_conditioning_anticipates_bend(self):
        planner = WholeBodySeedPlanner(
            HybridDistanceWorld(BoxWorld([])), self.arm, self.mass, self.limits, self.config
        )
        states = np.zeros((8, 6))
        states[4:, 4] = np.deg2rad(120.0)
        conditioned = planner._condition_shape_rates(states)
        self.assertAlmostEqual(conditioned[0, 4], 0.0)
        self.assertGreater(conditioned[1, 4], 0.0)
        maximum_delta = 0.8 * self.limits.theta_rate_max * self.config.dt_max / 3.0
        self.assertLessEqual(np.max(np.abs(np.diff(conditioned[:, 4]))), maximum_delta + 1e-9)

    def test_executor_interpolates_instead_of_jumping_samples(self):
        self.assertEqual(interpolation_bracket([0.0, 1.0, 2.0], -1.0), (0, 0, 0.0))
        self.assertEqual(interpolation_bracket([0.0, 1.0, 2.0], 3.0), (2, 2, 0.0))
        lower, upper, ratio = interpolation_bracket([0.0, 1.0, 2.0], 0.25)
        self.assertEqual((lower, upper), (0, 1))
        self.assertAlmostEqual(ratio, 0.25)

    def test_executor_yaw_interpolation_uses_short_arc(self):
        value = lerp_angle(np.deg2rad(179.0), np.deg2rad(-179.0), 0.5)
        self.assertAlmostEqual(abs(wrap_angle(value)), np.pi, places=6)


if __name__ == "__main__":
    unittest.main()
