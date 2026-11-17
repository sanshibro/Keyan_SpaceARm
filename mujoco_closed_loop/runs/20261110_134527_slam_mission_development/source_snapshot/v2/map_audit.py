"""Offline map-quality audit; Oracle geometry never enters planning."""

import hashlib

import numpy as np

from .environment import BoxWorld
from .experiment_scenarios import box_surface_points
from .geometry import world_arm_points, world_body_points
from .mapping import VoxelIndexMembership, voxel_indices
from .reference_planner import (
    ReferencePlanningError,
    VoxelAStarReferencePlanner,
    densify_path,
    path_states,
)
from .types import ArmModel, Limits


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values, q):
    return None if not len(values) else float(np.percentile(values, q))


def _known_coverage(snapshot, points):
    points = np.asarray(points, dtype=float).reshape((-1, 3))
    if not len(points):
        return 0.0
    membership = VoxelIndexMembership(snapshot.observed_indices)
    return float(np.mean(membership.contains(voxel_indices(
        points, snapshot.resolution
    ))))


def _planner_known_collision(snapshot, points, membership=None):
    """Return whether a sampled body intersects a planner-visible obstacle.

    Unknown voxels remain unsafe and are handled by the coverage gate.  This
    predicate only identifies states that are already infeasible because at
    least one sampled body/arm point lies in a measured occupied voxel.
    Requiring sensor observability behind that occupied surface would demand
    seeing inside a solid obstacle and would make the audit physically
    impossible in narrow passages.
    """
    points = np.asarray(points, dtype=float).reshape((-1, 3))
    if not len(points):
        return False
    if membership is None:
        membership = VoxelIndexMembership(snapshot.occupied_indices)
    return bool(np.any(membership.contains(voxel_indices(
        points, snapshot.resolution
    ))))


def _reference_and_coverage(snapshot, task, limits, arm, time_budget):
    start = np.asarray(task["start_state"], dtype=float)
    preferred = np.asarray(task["preferred_standoff"][:4], dtype=float)
    terminal_shape = np.radians(np.asarray(
        task.get("terminal_shape_deg", [90.0, -180.0]), dtype=float
    ))
    terminal = np.r_[preferred, terminal_shape]
    planner = VoxelAStarReferencePlanner(
        snapshot,
        center_clearance=limits.body_radius + limits.body_margin,
        body_radius=limits.body_radius,
        body_height=limits.body_height,
        body_margin=limits.body_margin,
        z_min=limits.z_min,
        z_max=limits.z_max,
        time_budget=time_budget,
    )
    points, diagnostics = planner.plan(start[:3], terminal[:3])
    points = densify_path(points, maximum_step=max(0.05, snapshot.resolution))

    # Cover the body and representative continuum envelopes shared by every
    # method: initial, terminal, straight transit, and a compact transit pose.
    shape_candidates = np.asarray([
        start[4:6], terminal_shape, [0.0, 0.0], np.radians([75.0, 0.0]),
    ], dtype=float)
    shape_candidates = np.unique(np.round(shape_candidates, 12), axis=0)
    body_sets = []
    arm_sets = []
    state_coverages = []
    occupied_membership = VoxelIndexMembership(snapshot.occupied_indices)
    candidate_states_total = 0
    candidate_states_known_collision = 0
    eligible_shapes_per_path_state = np.zeros(len(points), dtype=int)
    terminal_body = world_body_points(terminal, limits)
    terminal_arm = world_arm_points(terminal, arm)
    for shape in shape_candidates:
        states = path_states(points, default_shape=shape)
        for path_index, state in enumerate(states):
            candidate_states_total += 1
            body = world_body_points(state, limits)
            arm_points = world_arm_points(state, arm)
            whole_body = np.vstack((body, arm_points))
            # Candidate shapes that visibly intersect an occupied surface are
            # not executable states. Exclude them from the observability
            # denominator rather than requiring depth sensing through the
            # surface into the solid interior. No Oracle geometry is used.
            if _planner_known_collision(
                    snapshot, whole_body, membership=occupied_membership):
                candidate_states_known_collision += 1
                continue
            eligible_shapes_per_path_state[path_index] += 1
            body_sets.append(body)
            arm_sets.append(arm_points)
            # An average route coverage can hide a single unknown voxel at a
            # critical passage or at the grasp endpoint.  The online planner
            # treats every unknown voxel as occupied, so the map admission
            # test must also report and gate the worst sampled full-body
            # state, not just the route-wide mean.
            state_coverages.append(_known_coverage(
                snapshot, whole_body
            ))
    body_points = (
        np.vstack(body_sets) if body_sets else np.empty((0, 3), dtype=float)
    )
    arm_points = (
        np.vstack(arm_sets) if arm_sets else np.empty((0, 3), dtype=float)
    )
    survey = np.asarray(snapshot.traversed_positions, dtype=float).reshape((-1, 3))
    # The survey log also contains ground spawn/takeoff poses below the
    # planner's certified altitude band. They are not valid execution states
    # and must not dilute task-corridor coverage.
    survey = survey[
        (survey[:, 2] >= limits.z_min) & (survey[:, 2] <= limits.z_max)
    ]
    if len(survey):
        stride = max(1, int(np.ceil(len(survey) / 300.0)))
        survey_states = np.zeros((len(survey[::stride]), 6), dtype=float)
        survey_states[:, :3] = survey[::stride]
        survey_body = np.vstack([
            world_body_points(state, limits, azimuth_samples=32, vertical_samples=5)
            for state in survey_states
        ])
        survey_body_coverage = _known_coverage(snapshot, survey_body)
    else:
        survey_body_coverage = 0.0
    return diagnostics, {
        "reference_body_observed_coverage": _known_coverage(
            snapshot, body_points
        ),
        "reference_arm_observed_coverage": _known_coverage(
            snapshot, arm_points
        ),
        "reference_whole_body_observed_coverage": _known_coverage(
            snapshot, np.vstack((body_points, arm_points))
        ),
        "reference_state_observed_coverage_min": (
            float(np.min(state_coverages)) if state_coverages else 0.0
        ),
        "terminal_body_observed_coverage": _known_coverage(
            snapshot, terminal_body
        ),
        "terminal_arm_observed_coverage": _known_coverage(
            snapshot, terminal_arm
        ),
        "terminal_whole_body_observed_coverage": _known_coverage(
            snapshot, np.vstack((terminal_body, terminal_arm))
        ),
        "survey_body_observed_coverage": survey_body_coverage,
        "survey_pose_samples": int(len(survey)),
        "reference_shape_samples": int(len(shape_candidates)),
        "reference_path_samples": int(len(points)),
        "reference_candidate_states_total": int(candidate_states_total),
        "reference_candidate_states_known_collision": int(
            candidate_states_known_collision
        ),
        "reference_candidate_states_observability_gated": int(
            len(state_coverages)
        ),
        "reference_feasible_shapes_per_path_state_min": (
            int(np.min(eligible_shapes_per_path_state))
            if len(eligible_shapes_per_path_state) else 0
        ),
    }


def audit_snapshot(snapshot, oracle, task, thresholds=None, arm=None, limits=None):
    """Audit a frozen sensor map using schema-v5 task-relevant gates.

    Planning and coverage use only ``snapshot``. Exact boxes are consulted
    afterwards solely to score occupied-return accuracy and false positives.
    Hidden/back box surfaces remain a descriptive recall metric because a
    local sensor cannot be required to observe physically occluded geometry.
    """
    thresholds = dict(thresholds or {})
    arm = arm or ArmModel()
    limits = limits or Limits()
    resolution = snapshot.resolution
    surface_spacing = float(thresholds.get("surface_spacing_m", resolution))
    match_radius = float(thresholds.get(
        "surface_match_radius_m", max(0.10, 2.0 * resolution)
    ))
    oracle_world = BoxWorld.from_dicts(oracle.get("boxes", []))
    surface = np.vstack([
        box_surface_points(box, surface_spacing)
        for box in oracle.get("boxes", [])
    ]) if oracle.get("boxes") else np.empty((0, 3))
    occupied = snapshot.occupied_points
    evaluated_occupied = occupied[occupied[:, 2] >= 0.20]

    # Planner-visible returns -> nearest exact obstacle surface is the
    # deployable map-accuracy direction. It does not penalize hidden backs.
    oracle_signed = oracle_world.signed_distances(evaluated_occupied)
    occupied_surface_errors = np.abs(oracle_signed)
    false_positive = (
        float(np.mean(oracle_signed > match_radius))
        if len(oracle_signed) else 0.0
    )

    # Retain old Oracle-surface recall only for transparent reporting.
    if len(surface) and len(occupied):
        from .dependencies import import_scipy
        scipy = import_scipy()
        tree = scipy.spatial.cKDTree(occupied)
        truth_surface_errors = tree.query(surface, k=1)[0]
    else:
        truth_surface_errors = np.full(len(surface), np.inf)
    truth_surface_recall = (
        float(np.mean(truth_surface_errors <= match_radius))
        if len(truth_surface_errors) else 0.0
    )

    reference_error = None
    reference_diagnostics = {}
    coverage_metrics = {
        "reference_body_observed_coverage": 0.0,
        "reference_arm_observed_coverage": 0.0,
        "reference_whole_body_observed_coverage": 0.0,
        "reference_state_observed_coverage_min": 0.0,
        "terminal_body_observed_coverage": 0.0,
        "terminal_arm_observed_coverage": 0.0,
        "terminal_whole_body_observed_coverage": 0.0,
        "survey_body_observed_coverage": 0.0,
        "survey_pose_samples": 0,
        "reference_shape_samples": 0,
        "reference_path_samples": 0,
        "reference_candidate_states_total": 0,
        "reference_candidate_states_known_collision": 0,
        "reference_candidate_states_observability_gated": 0,
        "reference_feasible_shapes_per_path_state_min": 0,
    }
    reference_time_budget = float(thresholds.get(
        "reference_total_time_max_s", 15.0
    ))
    try:
        reference_diagnostics, coverage_metrics = _reference_and_coverage(
            snapshot, task, limits, arm, reference_time_budget
        )
    except (ReferencePlanningError, KeyError, ValueError) as error:
        reference_error = str(error)

    localization_source = str(snapshot.metadata.get(
        "localization_source", "unspecified"
    )).strip().lower()
    truth_localization = localization_source in (
        "", "unspecified", "gazebo_truth", "ground_truth", "synthetic_fixture"
    )
    metrics = {
        "surface_samples_descriptive": int(len(surface)),
        "occupied_voxels_evaluated": int(len(evaluated_occupied)),
        "planner_occupied_surface_error_median_m": _percentile(
            occupied_surface_errors, 50
        ),
        "planner_occupied_surface_error_p95_m": _percentile(
            occupied_surface_errors, 95
        ),
        "planner_occupied_surface_error_max_m": (
            float(np.max(occupied_surface_errors))
            if len(occupied_surface_errors) else None
        ),
        "occupied_false_positive_rate": false_positive,
        "truth_surface_recall_descriptive": truth_surface_recall,
        "truth_surface_error_p95_descriptive_m": _percentile(
            truth_surface_errors, 95
        ),
        "localization_source": localization_source,
        "truth_localization": truth_localization,
        "resolution_m": resolution,
        "surface_match_radius_m": match_radius,
        "reference_ready": reference_error is None,
        "reference_error": reference_error,
        "reference_preparation_time_s": reference_diagnostics.get(
            "preparation_elapsed"
        ),
        "reference_search_time_s": reference_diagnostics.get("elapsed"),
        "reference_total_time_s": reference_diagnostics.get("total_elapsed"),
        **coverage_metrics,
    }
    limits_report = {
        "planner_occupied_surface_error_p95_max_m": float(thresholds.get(
            "planner_occupied_surface_error_p95_max_m",
            thresholds.get("surface_error_p95_max_m", max(0.12, 0.90 * resolution)),
        )),
        "occupied_false_positive_rate_max": float(thresholds.get(
            "occupied_false_positive_rate_max", 0.10
        )),
        "reference_body_observed_coverage_min": float(thresholds.get(
            "reference_body_observed_coverage_min", 0.99
        )),
        "reference_whole_body_observed_coverage_min": float(thresholds.get(
            "reference_whole_body_observed_coverage_min", 0.98
        )),
        "reference_state_observed_coverage_min": float(thresholds.get(
            "reference_state_observed_coverage_min", 1.0
        )),
        "terminal_whole_body_observed_coverage_min": float(thresholds.get(
            "terminal_whole_body_observed_coverage_min", 1.0
        )),
        "survey_body_observed_coverage_min": float(thresholds.get(
            "survey_body_observed_coverage_min", 0.99
        )),
        "reference_total_time_max_s": reference_time_budget,
    }
    failures = []
    if truth_localization:
        failures.append("localization_source_not_slam_vio_lio")
    if snapshot.metadata.get("truth_derived") is not False:
        failures.append("truth_derived_map")
    if snapshot.metadata.get("comparison_independent_survey") is not True:
        failures.append("comparison_method_dependent_survey")
    if not snapshot.metadata.get("survey_protocol_manifest_sha256"):
        failures.append("missing_survey_protocol_manifest")
    if not len(evaluated_occupied) or not oracle.get("boxes"):
        failures.append("empty_oracle_or_map")
    p95 = metrics["planner_occupied_surface_error_p95_m"]
    if (p95 is None
            or p95 > limits_report["planner_occupied_surface_error_p95_max_m"]):
        failures.append("planner_occupied_surface_error_p95")
    if false_positive > limits_report["occupied_false_positive_rate_max"]:
        failures.append("occupied_false_positive_rate")
    if reference_error is not None:
        failures.append("common_reference_not_ready")
    if (metrics["reference_body_observed_coverage"]
            < limits_report["reference_body_observed_coverage_min"]):
        failures.append("reference_body_observed_coverage")
    if (metrics["reference_whole_body_observed_coverage"]
            < limits_report["reference_whole_body_observed_coverage_min"]):
        failures.append("reference_whole_body_observed_coverage")
    if (metrics["reference_state_observed_coverage_min"]
            < limits_report["reference_state_observed_coverage_min"]):
        failures.append("reference_state_observed_coverage_min")
    if (metrics["terminal_whole_body_observed_coverage"]
            < limits_report["terminal_whole_body_observed_coverage_min"]):
        failures.append("terminal_whole_body_observed_coverage")
    if (metrics["survey_body_observed_coverage"]
            < limits_report["survey_body_observed_coverage_min"]):
        failures.append("survey_body_observed_coverage")
    total_time = metrics["reference_total_time_s"]
    if total_time is None or total_time > limits_report["reference_total_time_max_s"]:
        failures.append("reference_total_time")
    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "metrics": metrics,
        "limits": limits_report,
        "reference_diagnostics": reference_diagnostics,
    }
