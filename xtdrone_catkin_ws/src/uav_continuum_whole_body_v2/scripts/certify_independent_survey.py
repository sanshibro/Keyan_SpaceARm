#!/usr/bin/env python3
"""Attach independent-survey evidence only after measured route verification."""

import argparse
import json
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "src"))

from uav_continuum_whole_body_v2.mapping import MapSnapshot  # noqa: E402
from uav_continuum_whole_body_v2.survey_protocol import (  # noqa: E402
    certify_survey_snapshot, load_survey_manifest,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("map", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    manifest, digest = load_survey_manifest(args.manifest)
    certified, report = certify_survey_snapshot(
        MapSnapshot.load(args.map), manifest, digest,
        waypoint_tolerance_m=float(manifest.get("waypoint_tolerance_m", 0.25)),
    )
    certified.save(args.output)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
