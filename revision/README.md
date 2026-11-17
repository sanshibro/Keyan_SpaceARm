# Paper mechanism and rolling-execution revision

Branch: `feat/paper-mechanism-rolling-20261117`.

The active manuscript is `paper/main.tex`, imported from the latest Windows hardware/architecture revision. The original publication and the Linux development workspace remain preserved. Historical evaluation uses the frozen formal geometry, not the modified working tree or the newer CAD dimensions.

## Milestones

1. Import and fingerprint the latest paper; publish the isolated branch.
2. Audit the three-gate run, reconstruct issued/actual histories and replace the geometry and rolling figures.
3. Compile, visually inspect and publish the manuscript and focused page preview.
4. Qualify the new MuJoCo geometry, then implement known-map gate and rolling-handoff experiments before extending perception complexity.

The live monitor is served on NUC loopback port 8767 and forwarded to Windows loopback port 18767. It shows recorded-data replay separately from new physics experiments. Each completed milestone receives a descriptive commit and push; unfinished results are explicitly labelled.
