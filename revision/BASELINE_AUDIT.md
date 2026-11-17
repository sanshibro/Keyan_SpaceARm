# Baseline reproduction audit — checkpoint 2026-11-17

The archived 450-run table remains historical. Changed implementations require a new paired evaluation.

- RRT: corrected shortest-angle distance for yaw/phi and added yaw-dependent scalar/batched swept-edge samples. A pure-yaw interior-collision regression and periodic-angle tests pass. All 82 core tests pass.
- AET-Seq: the local fixed-UAV configuration corridor does not reproduce the source paper's end-effector minimum-jerk whole-body decomposition. Preserve this historical comparator as an explicitly self-built sequential method; implement and evaluate a separately identified source-aligned adaptation before attributing results to AET.
- Global-WB: among 90 archived runs, 60 produced a safe trajectory; 46 of these report skipped_to_preserve_deadline and 14 record the projected-coordinate optimization backend. These counts are planning events, not task-success counts. The fixed-UAV corridor prerequisite and refinement budget split need correction and retesting. This local method is not the authors' AM-Planner implementation.
- Fixed: development-set morphology and equal reference/map/budget contracts still need a full manifest-level audit.

Primary sources inspected: https://github.com/arclab-hku/AET and https://github.com/SYSU-HILAB/am-planner . Source-paper PDFs and the local registry are reviewed alongside the code; names alone do not establish algorithmic equivalence.

Next: complete source-equivalence checks; record each adaptation; test corrected planners on common frozen tasks before large-scale closed-loop evaluation. Do not reuse old success numbers under corrected method identities.
