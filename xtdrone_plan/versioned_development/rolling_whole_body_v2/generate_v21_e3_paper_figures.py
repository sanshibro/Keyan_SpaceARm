#!/usr/bin/env python3
"""Generate the E3 paper figure package from closed-loop simulated evidence.

This adapter reuses the already audited schema-6 plotting implementation, but
binds it to the V2.1 E3 aggregate and automatically selects a simulated hard
case with rich rolling-handoff evidence.  Hidden Oracle geometry is used only
for post-hoc visualization/clearance, never as planner input.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).resolve()
REPO = next(parent for parent in (SCRIPT.parent,) + tuple(SCRIPT.parents)
            if (parent / ".git").exists())
DEFAULT_ENGINE = (
    REPO / "xtdrone_plan/experiments/method_comparison/"
    "schema6_formal_combined_commit4f80214/final_450/"
    "paper_artifacts/generate_sci_figures.py"
)
OURS = "rolling_wb_v2"
SUPPLEMENTARY = SCRIPT.parent / "generate_v21_e3_supplementary_evidence.py"
DEFAULT_DYNAMICS = (
    REPO / "xtdrone_plan/experiments/method_comparison/"
    "v21_dynamics_evidence_commitd9e01a2"
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def select_representative(root):
    candidates = []
    for path in root.rglob("acceptance_report.json"):
        try:
            audit = load_json(path)
            manifest = load_json(path.parent / "manifest.json")
        except (OSError, json.JSONDecodeError):
            continue
        if audit.get("method") != OURS or not audit.get("passed"):
            continue
        continuity = audit.get("switch_continuity", {})
        modes = continuity.get("handoff_mode_counts", {})
        switches = int(continuity.get("switch_count", 0) or 0)
        c2 = int(modes.get("validated_c2_bridge", 0) or 0)
        difficulty = str(manifest.get("scenario_difficulty", ""))
        family = str(manifest.get("scenario_family", ""))
        score = (
            int(difficulty == "hard"),
            int(family == "multi_passage"),
            int(c2 > 0),
            switches,
            c2,
        )
        candidates.append((score, path.parent, audit, manifest))
    if not candidates:
        raise RuntimeError("no successful simulated V2.1 acceptance report found")
    return max(candidates, key=lambda item: item[0])


def verify_aggregate(aggregate):
    with (aggregate / "paired_trials.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    keys = {(row["scenario"], row["method"]) for row in rows}
    scenarios = {row["scenario"] for row in rows}
    if len(rows) != 450 or len(keys) != 450 or len(scenarios) != 90:
        raise RuntimeError(
            "refusing paper figures: expected 450 unique rows/90 scenarios, got %d/%d/%d"
            % (len(rows), len(keys), len(scenarios))
        )
    return rows


def write_package_readme(output, representative, engine_hash):
    run, audit, manifest = representative
    text = """# Rolling-WB-V2.1 E3 SCI figure package

This package is generated only after the pre-registered 90-pair/450-trial
integrity gate passes.  PNG files are 600 dpi and PDF files retain vector text.
All curves use recorded PX4/Gazebo closed-loop state or the frozen aggregate.
Hidden Oracle geometry appears only in post-hoc evaluation overlays.

## Figures

1. `fig01_success_and_failure_overview`: Wilson-interval overall success,
   difficulty/family breakdown, and retained failure taxonomy.
2. `fig02_execution_quality_distributions`: continuous metrics on the common
   five-method success subset; it does not compare differently selected easy
   cases.
3. `fig03_paired_outcome_matrix`: all 90 paired binary outcomes.
4. `fig04_measured_whole_body_trajectory`: closed-loop simulated UAV/tool paths
   and PCC arm snapshots with post-hoc obstacle overlays and a straight-arm
   counterfactual.
5. `fig05_rolling_horizon_updates`: accepted rolling windows versus the
   closed-loop simulated execution.
6. `fig06_representative_execution_time_history`: simulated progress, bending
   vector, clearance, tracking/COM, 15-s deadline and certified-buffer margin.
   In panel (f), orange bars are planning plus validation, blue diamonds are
   certified-buffer time remaining when the handoff is requested, and the red
   dashed line is the common planning deadline.
7. `fig07_handoff_continuity`: issued position, velocity, acceleration, jerk,
   bending vector and bending rate around a validated handoff using a C2 bridge.
8. `fig08_dynamics_proxy_validation`: planned thrust/tilt proxies versus PX4
   target and simulated IMU response in three representative tasks.

## Representative closed-loop simulated run

- scenario: `{scenario}`
- family/difficulty/seed: `{family}` / `{difficulty}` / `{seed}`
- switches: `{switches}`
- direct / validated with C2 bridge: `{c1}` / `{c2}`
- run directory: `{run}`

The source plotting-engine SHA-256 is `{engine_hash}`.  The adapter selected the
representative case by a frozen evidence rule: prefer hard multi-passage cases,
then a case containing a validated handoff using a C2 bridge, then the largest switch count.
""".format(
        scenario=manifest.get("scenario_name"),
        family=manifest.get("scenario_family"),
        difficulty=manifest.get("scenario_difficulty"),
        seed=manifest.get("random_seed"),
        switches=audit.get("switch_continuity", {}).get("switch_count"),
        c1=audit.get("switch_continuity", {}).get("handoff_mode_counts", {}).get("direct_c1"),
        c2=audit.get("switch_continuity", {}).get("handoff_mode_counts", {}).get("validated_c2_bridge"),
        run=run,
        engine_hash=engine_hash,
    )
    (output / "README.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--aggregate", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--dynamics-root", type=Path, default=DEFAULT_DYNAMICS)
    args = parser.parse_args()
    aggregate = (args.aggregate or args.result_root / "aggregate").resolve()
    output = (args.output or args.result_root / "paper_artifacts").resolve()
    engine = args.engine.resolve()
    if not engine.is_file():
        raise SystemExit("missing audited plotting engine: %s" % engine)
    verify_aggregate(aggregate)
    selected = select_representative(args.result_root.resolve())
    _, run, audit, manifest = selected
    task = Path(manifest["scenario"])
    oracle = Path(manifest["oracle_file"])
    map_file = Path(manifest["map_snapshot_file"])
    for path in (run, task, oracle, map_file):
        if not path.exists():
            raise RuntimeError("representative evidence missing: %s" % path)

    spec = importlib.util.spec_from_file_location("v21_e3_plot_engine", str(engine))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.FINAL = aggregate
    module.OUT = output
    module.FIGURES = output / "figures"
    module.TABLES = output / "tables"
    module.SOURCE_DATA = output / "source_data"
    for directory in (module.FIGURES, module.TABLES, module.SOURCE_DATA):
        directory.mkdir(parents=True, exist_ok=True)
    module.LABELS[OURS] = "Rolling-WB-V2.1"
    module.representative_paths = lambda: (run, oracle, task, map_file)
    module.main()
    result = subprocess.run(
        [sys.executable, str(SUPPLEMENTARY), str(args.result_root.resolve()),
         "--output", str(output), "--dynamics-root",
         str(args.dynamics_root.resolve())],
        cwd=str(REPO), check=False,
    )
    if result.returncode:
        raise RuntimeError("supplementary evidence generation failed")
    write_package_readme(output, (run, audit, manifest), sha256(engine))
    generation = load_json(output / "generation_report.json")
    generation.update({
        "e3_result_root": str(args.result_root.resolve()),
        "e3_aggregate": str(aggregate),
        "representative_scenario": manifest.get("scenario_name"),
        "plotting_engine": str(engine),
        "plotting_engine_sha256": sha256(engine),
        "adapter": str(SCRIPT),
    })
    (output / "generation_report.json").write_text(
        json.dumps(generation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "result": "PASS", "output": str(output),
        "representative": manifest.get("scenario_name"),
        "figures": len(list((output / "figures").glob("*.png"))),
    }, indent=2))


if __name__ == "__main__":
    main()
