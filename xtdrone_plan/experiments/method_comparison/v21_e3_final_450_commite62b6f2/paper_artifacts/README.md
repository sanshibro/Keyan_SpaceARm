# Rolling-WB-V2.1 E3 SCI figure package

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

- scenario: `multi_passage_hard_s011`
- family/difficulty/seed: `multi_passage` / `hard` / `11`
- switches: `6`
- direct / validated with C2 bridge: `4` / `2`
- run directory: `/home/uav/桌面/uav_arm_1/xtdrone_plan/experiments/method_comparison/v21_e3_final_450_commite62b6f2/multi_passage_hard_s011__8068b3486507/closed_loop_runs/multi_passage_hard_s011__rolling_wb_v2__20260803_175558_1958222`

The source plotting-engine SHA-256 is `4400e4ce3784fc1be6e317b7f64291fbf0c9979dcfad76feab7ae20ffbdfbc7a`.  The adapter selected the
representative case by a frozen evidence rule: prefer hard multi-passage cases,
then a case containing a validated handoff using a C2 bridge, then the largest switch count.
