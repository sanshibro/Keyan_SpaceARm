#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parents[1]
REPOSITORY = PACKAGE.parents[2]
sys.path.insert(0, str(PACKAGE / "src"))

from uav_continuum_whole_body_v2.experiment_scenarios import (  # noqa: E402
    DIFFICULTIES, SCENARIO_FAMILIES, generate_scenario, write_scenario,
)


def main():
    parser = argparse.ArgumentParser(description="Generate paired comparison scenario fixtures")
    parser.add_argument("--output", type=Path, default=(
        REPOSITORY / "xtdrone_plan" / "experiments" / "method_comparison" / "scenarios"
    ))
    parser.add_argument("--families", nargs="+", choices=SCENARIO_FAMILIES,
                        default=list(SCENARIO_FAMILIES))
    parser.add_argument("--difficulties", nargs="+", choices=DIFFICULTIES,
                        default=list(DIFFICULTIES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--resolution", type=float, default=0.10)
    parser.add_argument("--no-development-map", action="store_true")
    args = parser.parse_args()
    count = 0
    for family in args.families:
        for difficulty in args.difficulties:
            for seed in args.seeds:
                spec = generate_scenario(family, difficulty, seed)
                write_scenario(
                    spec, args.output / spec.name,
                    development_snapshot=not args.no_development_map,
                    resolution=args.resolution,
                )
                count += 1
    print("GENERATED_SCENARIOS=%d" % count)
    print("OUTPUT=%s" % args.output)
    if not args.no_development_map:
        print("MAP_CLASS=synthetic_perfect_map_not_for_formal_results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
