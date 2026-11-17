# Reproduction and provenance

This export was prepared for `sanshibro/Keyan_SpaceARm` from the supplied November 17 Overleaf archive. It includes the original paper assets, frozen experimental code, existing results, and all raw files in the selected evidence datasets. Preparing or publishing the archive does not rerun the flights.

## Directory layout

| Path | Contents |
|---|---|
| `paper/` | Exact original ZIP, extracted manuscript, supplement, figures and layout scripts |
| `xtdrone_catkin_ws/src/` | Three project packages exported from algorithm commit `e62b6f2` |
| `evidence_tools/` | Later reporting scripts exported separately from `d5682f9` |
| `xtdrone_plan/versioned_development/rolling_whole_body_v2/` | Frozen plans, protocols, audits, evidence and figure generators |
| `xtdrone_plan/experiments/method_comparison/.../paper_artifacts/` | Existing plotting source tables, figures and statistical outputs |
| `datasets/` | Lossless split archives, part hashes and hashes of every raw member |
| `sensing/Result/` | Supplied sensing images and explanatory documents |
| `sensing/top_level/` | Available historical collection, prediction, ORB and control scripts |
| `provenance/` | Exported-source hashes, manuscript-asset mapping and independently recounted trials |

The formal raw dataset preserves all 451 result files, including the retained SYSTEM_ERROR. The statistical set is 450 method trials over 90 complete five-method pairs. All statistical-set manifests record `e62b6f2`; methods within each scene use the same map hash.

## Figure-to-evidence mapping

| Manuscript asset | Primary source |
|---|---|
| Framework | `paper/figure_source/` and the existing composed framework assets; conceptual diagram |
| `outcomes_revised.pdf` | Original Fig. 1, 450 paired outcomes and statistical tables |
| `quality_revised.pdf` | Original Fig. 2; common-success task subset for matched distributions |
| `passage_restored.pdf` | Original Fig. 4, representative hard Multi-Passage log and post-hoc geometry |
| `rolling_clear.pdf` | Original Fig. 5; displayed selection of recorded planning windows, full source retained |
| `sensing_restored.png` | The two exact supplied sensing PNGs, recomposed by `paper/scripts/restore_evidence_figures.py` |
| Supplementary dynamics/handoff | Original Fig. 7/8 and the archived dynamics/handoff records |
| 72.8% paired jerk claim | `v21_e2_15scene_commitd9e01a2/audit/paired_runs.csv`, with both old and new raw runs archived |

Do not substitute old V2 `90/90` results for the V2.1 `89/90` table. Existing figure compositions are included exactly; rendering metadata or cropping can make a revised PDF differ in bytes from its experimental source. `provenance/paper_asset_sources.json` reports exact byte matches without claiming all crops are identical files.

## 1. Verify hashes and numerical claims

```bash
python3 scripts/verify_publication.py
```

This standard-library check verifies the payload, split-part hashes, the five success counts and the median of the 15 per-pair jerk reductions. It does not need ROS, SciPy or MuJoCo.

## 2. Restore raw files

```bash
python3 scripts/restore_datasets.py
```

Default output: `.reproduction/restored/`. The script checks all raw-member hashes, rejects traversal/link entries and refuses to replace an existing file with different content. The formal data include registered-map records, map snapshots, manifests, commands, state logs, candidate trajectories, acceptance reports and execution events. The sensing archive includes the available training/collection scripts, model files and data directories, excluding unrelated patent documents and build caches.

## 3. Analyze the original planning logs

The original analysis code expects NumPy, SciPy, PyYAML and Matplotlib. The ROS experiment environment was Ubuntu 20.04 / ROS Noetic / Python 3.8. The code uses APIs from older SciPy, including `scipy.stats.binom_test`; use compatible dependencies or a separately documented compatibility environment. Do not overwrite frozen code to hide dependency differences.

From the publication repository, after restoring the formal dataset:

```bash
python3 evidence_tools/xtdrone_catkin_ws/src/uav_continuum_whole_body_v2/scripts/aggregate_method_comparison.py \
  .reproduction/restored/xtdrone_plan/experiments/method_comparison/v21_e3_final_450_commite62b6f2 \
  --output .reproduction/recomputed_aggregate
```

Original manifests retain absolute acquisition paths. Aggregation can use `scenario_name`; exact figure regeneration additionally needs relocation of the selected task, oracle and map paths. Work on a copied manifest tree: map the historical `/home/uav/桌面/uav_arm_1/` prefix to the restored-data root, and use the exported frozen scenario YAMLs where appropriate. Keep raw archived manifests untouched and record the relocation mapping. Existing frozen aggregate and figure source-data files remain directly inspectable without this step.

The original E3 figure entry point is:

```text
xtdrone_plan/versioned_development/rolling_whole_body_v2/generate_v21_e3_paper_figures.py
```

It takes `result_root`, `--aggregate`, `--output`, `--engine` and `--dynamics-root`. The matching plotting engine is included at its historical relative path. It expects a Git checkout to locate the repository root. Use a separate `.reproduction/` output rather than writing over the supplied figures.

## 4. Rebuild manuscript artwork and PDFs

The manuscript ZIP already contains generated assets. Artwork assembly additionally requires a compatible modern PyMuPDF (`fitz`), Pillow, Matplotlib and ReportLab; consult the source scripts, since current PyMuPDF redaction arguments differ from old versions.

In `paper/`, the documented artwork order is:

```bash
python3 scripts/build_figures.py
python3 scripts/refine_layout_figures.py
python3 scripts/restore_evidence_figures.py
```

Rebuild in a copy if exact supplied figure hashes need to be retained. Use pdfLaTeX/BibTeX or the supplied Overleaf project for `main.tex` and `supplementary.tex`. The original PDFs are retained for preview. This publication export does not assert that the present host has all TeX/PyMuPDF dependencies installed.

## 5. Rerun PX4/Gazebo separately

The archived launcher is `xtdrone_catkin_ws/src/uav_continuum_whole_body_v2/scripts/run_method_comparison_closed_loop.sh`. It requires ROS Noetic, a built catkin workspace, PX4 SITL, Gazebo, MAVROS and XTDrone assets. Some original paths are machine-specific. Inspect and configure an isolated execution environment before running. The paper package does not redistribute all external PX4/XTDrone sources or proprietary SDK binaries.

A cold-start flight reproduction is a new experiment. Save new timestamps, dependencies, configuration and outcomes; retain original acceptance reports unchanged. Static publication checks do not equal a fresh simulator PASS.

## Sensing limitations that remain visible in the archive

The manuscript's two original sensing images exactly match `sensing/Result/1280X1280.PNG` and `1280X1280 (1).PNG`. The supplied scripts/data are historical evidence, not proof that every source CSV can regenerate those exact pixels. In particular, the acquisition history includes online visual teaching and three active strain channels. The source archive does not independently substantiate the manuscript's four-channel mocap-supervised acquisition narrative. Do not rename labels, alter datasets or fabricate missing mocap records to make that narrative appear verified.

## Publication scope and access

External services require valid repository access. Authentication failure means the files remain locally prepared, not uploaded. A publication-status record must report the actual remote commit only after a successful push and remote verification.
