# Keyan_SpaceARm — Morphology-Aware Rolling Whole-Body Planning

This export accompanies the supplied manuscript **Morphology-Aware Rolling Whole-Body Planning for Aerial Continuum Manipulation in Confined Spaces**. It preserves the manuscript/layout package, the frozen PX4/Gazebo planner implementation, the formal paired results, raw records, figure sources, and available sensing code/data.

此目录只对应论文使用的历史实验。2026年11月的MuJoCo开发工作没有混入。

## Contents and versions

| Location | Contents |
|---|---|
| `paper/` | Exact uploaded Overleaf ZIP, extracted manuscript, supplement, figures, layout/assembly scripts and supplied PDFs |
| `xtdrone_catkin_ws/src/` | Three ROS packages exported from the formal algorithm commit `e62b6f2` |
| `evidence_tools/` | Later ROS experiment/reporting scripts exported separately from `d5682f9`; these do not replace frozen algorithm code |
| `xtdrone_plan/versioned_development/rolling_whole_body_v2/` | Frozen plans, protocols, postprocessing and figure-generation code |
| `xtdrone_plan/experiments/method_comparison/` | Directly browsable aggregates, source-data CSVs, figure PDFs/PNGs, tables, scenes and audits |
| `datasets/` | Lossless raw datasets split into at most 40 MiB per Git file, with member/part SHA-256 manifests |
| `sensing/` | Exact sensing result images and available standalone acquisition/angle-estimation scripts; other code/models/data are in the sensing dataset |
| `provenance/` | Source commits, file hashes, paper-asset matches, and independent trial recount |

The supplied manuscript ZIP is preserved unchanged. Historical sources within it retain their original claims/comments. Inclusion is an archive of the supplied material, not independent verification of every manuscript statement.

## Frozen planning results

90 paired tasks × 5 methods = 450 evaluated runs. The retained system-error record is also archived and distinguished from these runs.

| Method | Success | Task median (s) | Planning P95 (s) | Jerk energy / path length |
|---|---:|---:|---:|---:|
| Fixed-Morphology | 55/90 | 130.716 | 2.936 | 0.086739 |
| AET-Sequential | 61/90 | 98.060 | 10.439 | 0.273351 |
| WB-RRT* | 68/90 | 168.552 | 13.017 | 0.061861 |
| Global-WB | 59/90 | 99.684 | 10.386 | 0.257431 |
| Rolling-WB-V2.1 | 89/90 | 103.616 | 3.697 | 0.126304 |

Continuous main-table metrics describe each method's successful trials. Matched continuous tests use the common-success subset. The 15-task paired handoff comparison has median per-pair jerk reduction **72.7838%**, which is not the ratio of the two marginal medians.

## Start here

```bash
python3 scripts/verify_publication.py
python3 scripts/restore_datasets.py
```

Restoration writes only into `.reproduction/restored/` by default and checks all archived file hashes. For one dataset:

```bash
python3 scripts/restore_datasets.py handoff_15_pairs_raw
```

Available names: `formal_450_raw`, `handoff_15_pairs_raw`, `dynamics_proxy_raw`, `sensing_code_models_data`. Split parts form one gzip/tar stream in manifest order; individual parts are not standalone archives.

The original raw formal records restore under:

```text
.reproduction/restored/xtdrone_plan/experiments/method_comparison/v21_e3_final_450_commite62b6f2/
```

Original logs/manifests retain historical absolute paths to preserve hashes. See `REPRODUCIBILITY.md` for relocation and the distinction between re-analysis, artwork assembly, and rerunning SITL.

## Paper and figure layout

Open `paper/main.pdf` and `paper/supplementary.pdf`. The supplied Overleaf project uses pdfLaTeX; `paper/original_overleaf_project.zip` can be imported directly. Source figures and their revised compositions are both included. IEEEtran source headers/licenses are retained.

## Evidence boundaries

- Planning uses frozen sensor-derived maps and a second-order arm response, rather than a material-level continuum dynamics model. UAV state feedback in the formal simulator includes a truth bridge.
- Baselines are project adaptations; do not describe them as unmodified author implementations.
- Sensing figures exactly match the two PNG files in the supplied `Test/Result` directory. Available historical sensing code includes online visual teaching and three-strain experiments. This archive does not establish an independent four-channel mocap training/test chain.
- The exact data-to-pixel regeneration chain for those two sensing PNGs is not established merely by archiving nearby code. See `provenance/` and `REPRODUCIBILITY.md`.
- No new flight trial or MuJoCo validation is implied by this publication export.

External ROS/PX4/XTDrone/ORB-SLAM dependencies are not redistributed as proprietary binaries. Preserve individual third-party licenses. No new blanket software license is asserted for this mixed-origin archive.
