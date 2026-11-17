"""Validation and evidence checks for comparison-independent map surveys."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .mapping import MapSnapshot


def load_survey_manifest(path):
    path = Path(path)
    raw = path.read_bytes()
    manifest = json.loads(raw.decode("utf-8"))
    if int(manifest.get("schema_version", 0)) != 1:
        raise ValueError("survey manifest schema_version must be 1")
    if manifest.get("protocol_type") != "method_independent_waypoint_survey":
        raise ValueError("unsupported survey protocol_type")
    if manifest.get("compared_planner_access") is not False:
        raise ValueError("survey must explicitly disable compared planner access")
    if manifest.get("oracle_geometry_access") is not False:
        raise ValueError("survey must explicitly disable oracle geometry access")
    if not str(manifest.get("protocol_id", "")).strip():
        raise ValueError("survey protocol_id is required")
    waypoints = manifest.get("waypoints") or []
    if len(waypoints) < 2:
        raise ValueError("survey requires at least two waypoints")
    for index, waypoint in enumerate(waypoints):
        pose = waypoint.get("pose") if isinstance(waypoint, dict) else None
        if not isinstance(pose, list) or len(pose) != 4:
            raise ValueError("waypoint %d pose must be [x,y,z,yaw]" % index)
        if not np.all(np.isfinite(np.asarray(pose, dtype=float))):
            raise ValueError("waypoint %d contains non-finite values" % index)
    digest = hashlib.sha256(raw).hexdigest()
    return manifest, digest


def certify_survey_snapshot(snapshot, manifest, manifest_sha256,
                            waypoint_tolerance_m=0.25):
    """Certify that a sensor snapshot followed a frozen non-planner route."""
    if snapshot.metadata.get("truth_derived") is not False:
        raise ValueError("truth-derived snapshots cannot be survey certified")
    localization = str(snapshot.metadata.get("localization_source", "")).lower()
    if not localization or any(token in localization for token in ("truth", "gazebo")):
        raise ValueError("survey localization must be onboard SLAM/VIO/LIO/EKF")
    traversed = np.asarray(snapshot.traversed_positions, dtype=float).reshape((-1, 3))
    if len(traversed) < 2:
        raise ValueError("snapshot has no measured survey trajectory")
    waypoints = np.asarray([item["pose"][:3] for item in manifest["waypoints"]], dtype=float)
    cursor = 0
    errors = []
    matched_indices = []
    for waypoint in waypoints:
        distances = np.linalg.norm(traversed[cursor:] - waypoint, axis=1)
        if not len(distances):
            raise ValueError("measured survey ended before all waypoints")
        entries = np.flatnonzero(distances <= float(waypoint_tolerance_m))
        if not len(entries):
            error = float(np.min(distances))
            raise ValueError(
                "measured survey missed waypoint by %.3f m (limit %.3f m)"
                % (error, waypoint_tolerance_m)
            )
        # Use the first temporal entry, not the global nearest sample. Routes
        # may deliberately return to a center after an endpoint scan.
        relative = int(entries[0])
        error = float(distances[relative])
        cursor += relative + 1
        errors.append(error)
        matched_indices.append(cursor - 1)
    metadata = dict(snapshot.metadata)
    metadata.update({
        "comparison_independent_survey": True,
        "formal_experiment_allowed": False,
        "survey_controller": "manifest_pose_setpoint_v1",
        "survey_protocol": str(manifest["protocol_id"]),
        "survey_protocol_manifest_sha256": str(manifest_sha256),
        "survey_waypoint_count": int(len(waypoints)),
        "survey_waypoint_error_max_m": float(max(errors)),
        "survey_waypoint_tolerance_m": float(waypoint_tolerance_m),
    })
    certified = MapSnapshot(
        frame_id=snapshot.frame_id, stamp=snapshot.stamp,
        resolution=snapshot.resolution,
        occupied_indices=snapshot.occupied_indices,
        observed_indices=snapshot.observed_indices,
        source="certified_independent_sensor_survey",
        metadata=metadata, traversed_positions=snapshot.traversed_positions,
    )
    report = {
        "passed": True,
        "manifest_sha256": str(manifest_sha256),
        "protocol_id": str(manifest["protocol_id"]),
        "snapshot_input_hash": snapshot.map_hash,
        "snapshot_output_hash": certified.map_hash,
        "waypoint_errors_m": errors,
        "waypoint_matched_indices": matched_indices,
        "waypoint_tolerance_m": float(waypoint_tolerance_m),
    }
    return certified, report
