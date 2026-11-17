#!/usr/bin/env python3
"""Memory-bounded, semantics-equivalent schema-v5 survey-map fusion.

The frozen implementation expands every observed voxel by all 27 one-voxel
offsets before applying the task-corridor crop.  For unusually dense surveys
that creates a multi-gigabyte temporary array.  This implementation first
removes source voxels farther than one support voxel from the final crop, then
performs the identical 27-offset union and final crop.  A source voxel outside
that one-voxel halo cannot contribute to the final map, so the result is
mathematically identical.  The script exists only as an infrastructure/OOM
recovery tool and is validated byte-for-byte by map hash on completed cases.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


SCRIPT = Path(__file__).resolve()
REPO = next(parent for parent in (SCRIPT.parent,) + tuple(SCRIPT.parents)
            if (parent / ".git").exists())
PACKAGE = REPO / "xtdrone_catkin_ws/src/uav_continuum_whole_body_v2"
sys.path.insert(0, str(PACKAGE / "src"))
from uav_continuum_whole_body_v2.mapping import (  # noqa: E402
    MapSnapshot, _unique_rows, voxel_centers,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("maps", nargs=2, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--survey-manifest", type=Path, required=True)
    parser.add_argument("--sensor", action="append", default=[])
    parser.add_argument("--reference-index", type=int, default=0)
    parser.add_argument("--observation-support-voxels", type=int, default=1)
    parser.add_argument("--corridor-xy-padding", type=float, required=True)
    parser.add_argument("--corridor-z-min", type=float, default=0.2)
    parser.add_argument("--corridor-z-max", type=float, default=2.5)
    parser.add_argument("--comparison-independent-survey", action="store_true")
    parser.add_argument("--survey-protocol", required=True)
    args = parser.parse_args()
    if args.observation_support_voxels != 1:
        raise SystemExit("this recovery tool is frozen to one-voxel support")
    snapshots = [MapSnapshot.load(path) for path in args.maps]
    first = snapshots[0]
    reference = snapshots[args.reference_index]
    manifest_hash = hashlib.sha256(args.survey_manifest.read_bytes()).hexdigest()
    localization_sources = []
    for index, snapshot in enumerate(snapshots):
        if not np.isclose(snapshot.resolution, first.resolution, atol=1e-12):
            raise ValueError("map resolution mismatch")
        if snapshot.frame_id != first.frame_id:
            raise ValueError("map frame mismatch")
        if snapshot.metadata.get("truth_derived") is not False:
            raise ValueError("truth-derived map is forbidden")
        if snapshot.metadata.get("unknown_is_occupied") is not True:
            raise ValueError("conservative unknown-space flag is required")
        if snapshot.metadata.get("comparison_independent_survey") is not True:
            raise ValueError("independent-survey certification is required")
        if snapshot.metadata.get("survey_protocol_manifest_sha256") != manifest_hash:
            raise ValueError("survey manifest hash mismatch")
        if snapshot.metadata.get("survey_protocol") != args.survey_protocol:
            raise ValueError("survey protocol mismatch")
        localization = str(snapshot.metadata.get("localization_source", "")).lower()
        if localization in ("", "unspecified", "gazebo_truth", "ground_truth",
                            "synthetic_fixture"):
            raise ValueError("unapproved localization source")
        localization_sources.append(localization)
    if len(set(localization_sources)) != 1:
        raise ValueError("localization source mismatch")
    if not len(reference.traversed_positions):
        raise ValueError("reference map has no measured traversed centerline")

    minimum = np.min(reference.traversed_positions, axis=0)
    maximum = np.max(reference.traversed_positions, axis=0)
    minimum[:2] -= args.corridor_xy_padding
    maximum[:2] += args.corridor_xy_padding
    minimum[2], maximum[2] = args.corridor_z_min, args.corridor_z_max

    occupied = _unique_rows(np.vstack([
        snapshot.occupied_indices for snapshot in snapshots
    ]))
    observed_base = _unique_rows(np.vstack([
        snapshot.observed_indices for snapshot in snapshots
    ] + [occupied]))

    # Exact pre-crop: with Chebyshev support radius one, only source centres
    # inside the final bounds enlarged by one resolution can enter the result.
    base_centers = voxel_centers(observed_base, first.resolution)
    halo_mask = np.all(
        (base_centers >= minimum - first.resolution)
        & (base_centers <= maximum + first.resolution), axis=1,
    )
    observed_base = observed_base[halo_mask]
    del base_centers, halo_mask
    offsets = np.asarray([
        [dx, dy, dz]
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    ], dtype=np.int64)
    observed = _unique_rows(np.vstack(
        [observed_base + offset for offset in offsets] + [occupied]
    ))

    def inside(indices):
        centers = voxel_centers(indices, first.resolution)
        return np.all((centers >= minimum) & (centers <= maximum), axis=1)

    occupied = occupied[inside(occupied)]
    observed = observed[inside(observed)]
    observed = _unique_rows(np.vstack((observed, occupied)))
    crop_bounds = {
        "minimum": minimum.tolist(), "maximum": maximum.tolist(),
        "xy_padding_m": args.corridor_xy_padding,
        "source": "onboard_estimator_reference_session",
    }
    metadata = {
        "acquisition_mode": "conservative_multi_session_sensor_fusion",
        "formal_experiment_allowed": False,
        "comparison_independent_survey": args.comparison_independent_survey,
        "input_map_hashes": [snapshot.map_hash for snapshot in snapshots],
        "input_map_sources": [snapshot.source for snapshot in snapshots],
        "input_session_count": len(snapshots),
        "localization_source": localization_sources[0],
        "observation_support_model": "rgbd_finite_pixel_chebyshev_one_voxel",
        "observation_support_voxels": 1,
        "reference_session_index": args.reference_index,
        "sensor_suite": [str(value) for value in args.sensor],
        "survey_protocol": args.survey_protocol,
        "survey_protocol_manifest_sha256": manifest_hash,
        "task_corridor_crop": crop_bounds,
        "traversed_centerline_source": "onboard_estimator_reference_session",
        "truth_derived": False,
        "unknown_is_occupied": True,
    }
    fused = MapSnapshot(
        resolution=first.resolution,
        occupied_indices=occupied,
        observed_indices=observed,
        frame_id=first.frame_id,
        source="fused_sensor_slam",
        stamp=max(snapshot.stamp for snapshot in snapshots),
        metadata=metadata,
        traversed_positions=reference.traversed_positions,
    )
    fused.save(args.output)
    report = {
        "fusion": "occupied_union_observed_union_no_interpolation_or_deletion",
        "input_maps": [str(path.resolve()) for path in args.maps],
        "input_map_hashes": [snapshot.map_hash for snapshot in snapshots],
        "occupied_voxels": int(len(fused.occupied_indices)),
        "observed_voxels": int(len(fused.observed_indices)),
        "output_map": str(args.output.resolve()),
        "output_map_hash": fused.map_hash,
        "observation_support_voxels": 1,
        "task_corridor_crop": crop_bounds,
        "reference_index": args.reference_index,
        "sensor_suite": list(args.sensor),
        "comparison_independent_survey": args.comparison_independent_survey,
        "survey_protocol": args.survey_protocol,
        "survey_protocol_manifest": str(args.survey_manifest.resolve()),
        "survey_protocol_manifest_sha256": manifest_hash,
        "truth_geometry_access": False,
        "memory_bounded_equivalence": (
            "pre-crop source halo by one support voxel before identical 27-offset union"
        ),
    }
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
