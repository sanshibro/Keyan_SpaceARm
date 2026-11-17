#!/usr/bin/env python3
"""Create a traceable, conservative union of sensor-derived map sessions."""

import argparse
import hashlib
import json
from pathlib import Path
import sys


PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "src"))

from uav_continuum_whole_body_v2.mapping import (  # noqa: E402
    MapSnapshot,
    fuse_map_snapshots,
)


def main():
    parser = argparse.ArgumentParser(
        description="Fuse two or more non-truth sensor maps without interpolation"
    )
    parser.add_argument("maps", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference-index", type=int, default=0,
        help="zero-based input supplying the measured reference centerline",
    )
    parser.add_argument(
        "--sensor", action="append", default=[],
        help="sensor-suite label; repeat for each sensor modality",
    )
    parser.add_argument(
        "--observation-support-voxels", type=int, choices=(0, 1), default=0,
        help="finite RGB-D pixel support; only audited values 0 or 1 are allowed",
    )
    parser.add_argument(
        "--corridor-xy-padding", type=float,
        help="crop to measured survey bounds plus this XY padding",
    )
    parser.add_argument("--corridor-z-min", type=float, default=0.20)
    parser.add_argument("--corridor-z-max", type=float, default=2.50)
    parser.add_argument(
        "--comparison-independent-survey", action="store_true",
        help="assert that acquisition did not use any compared planner",
    )
    parser.add_argument("--survey-protocol", default="unspecified")
    parser.add_argument(
        "--survey-manifest", type=Path,
        help="frozen protocol/waypoint manifest required for independent surveys",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    snapshots = [MapSnapshot.load(path) for path in args.maps]
    survey_manifest_sha256 = None
    if args.survey_manifest:
        survey_manifest_sha256 = hashlib.sha256(
            args.survey_manifest.read_bytes()
        ).hexdigest()
    fused = fuse_map_snapshots(
        snapshots,
        reference_index=args.reference_index,
        sensor_suite=args.sensor,
        observation_support_voxels=args.observation_support_voxels,
        corridor_xy_padding=args.corridor_xy_padding,
        corridor_z_bounds=(args.corridor_z_min, args.corridor_z_max),
        comparison_independent_survey=args.comparison_independent_survey,
        survey_protocol=args.survey_protocol,
        survey_manifest_sha256=survey_manifest_sha256,
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
        "observation_support_voxels": args.observation_support_voxels,
        "task_corridor_crop": fused.metadata.get("task_corridor_crop"),
        "reference_index": args.reference_index,
        "sensor_suite": list(args.sensor),
        "comparison_independent_survey": args.comparison_independent_survey,
        "survey_protocol": args.survey_protocol,
        "survey_protocol_manifest": (
            None if args.survey_manifest is None
            else str(args.survey_manifest.resolve())
        ),
        "survey_protocol_manifest_sha256": survey_manifest_sha256,
        "truth_geometry_access": False,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
