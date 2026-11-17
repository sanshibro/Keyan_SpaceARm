# Rolling-WB-V2.1 E3 最终完成与论文证据报告

日期：2026-08-04

冻结算法提交：`e62b6f22a20a9c2ba18d4041124427e1cfe5f59c`

预注册标签：`v2.1-e3-preregistered-20260801`
正式结果目录：`xtdrone_plan/experiments/method_comparison/v21_e3_final_450_commite62b6f2`

## 1. 完整性结论

E3 按预注册协议完成 3 个场景族、3 档难度、10 个未参与调参的随机种，共 90 个任务。每个任务均执行 5 种方法，共形成 450 条有效正式结果。独立终审结果为 `PASS`：

- 90 个五方法完整配对，无重复、无缺失；
- 450 份有效 manifest 全部指向冻结提交，同一场景五方法地图哈希一致；
- 官方配对表与独立重算配对表逐项相同；
- `paper_eligibility=true` 且 `statistical_claim_ready=true`；
- 45 个“每 10 次”不可变阶段快照全部通过完整性检查；
- 保留 1 份后续已恢复的 `SYSTEM_ERROR` 原始 trial 及 27 条基础设施事件，它们不进入方法成功率，也未替换方法失败；
- 18 次地图融合 OOM 恢复均发生在方法启动前，且通过逐元素等价哈希与 schema-v5 独立地图审计。

## 2. 主要结果

| 方法 | 成功率 | 任务时间中位数 (s) | 规划 P95 (s) | 机体/臂净空 P05 (m) | 跟踪 RMSE 中位数 (m) | Jerk 中位数 | 末端误差中位数 (m) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed-Morphology | 55/90 (61.1%) | 130.72 | 2.94 | 0.2324/0.1490 | 0.0214 | 0.0867 | 0.0176 |
| AET-Sequential-MinJerk | 61/90 (67.8%) | 98.06 | 10.44 | 0.2295/0.1436 | 0.0285 | 0.2734 | 0.0140 |
| Informed-RRT*-WB | 68/90 (75.6%) | 168.55 | 13.02 | 0.1793/0.1390 | 0.0195 | 0.0619 | 0.0155 |
| Global-Batch-WB | 59/90 (65.6%) | 99.68 | 10.39 | 0.2286/0.1434 | 0.0270 | 0.2574 | 0.0144 |
| **Rolling-WB-V2.1** | **89/90 (98.9%)** | 103.62 | 3.70 | 0.1827/0.1442 | 0.0240 | 0.1263 | **0.0129** |

Rolling-WB-V2.1 对四个 baseline 的配对 McNemar 检验分别为
`1.16e-10`、`7.45e-09`、`9.54e-07` 和 `1.86e-09`。这支持“任务成功率显著更高”的主要结论，但不支持“所有连续指标均最优”。V2.1 的机体净空不及 Fixed/AET/Global，Jerk 不及 Fixed 和 RRT*，论文必须同时报告这些边界。

## 3. V2.1 闭环证据

90 份 V2.1 验收报告中 89 份通过，共记录 494 次规划尝试、391 次滚动接管（C1 直接接管 339 次，C2 经验证桥接 52 次），无接管诊断缺失，无规划截止期违约。所有规划尝试的时间中位数/P95/最大值为 2.256/3.697/8.779 s，持续 hold 的 P95/最大值为 2.077/4.576 s。

唯一未通过的 `clutter_reach_hard_s018` 已按预注册语义保留为 `repeated_planning_rejection`：任务实际完成且无碰撞、无 unsafe/abort，但连续候选拒绝达 5 次，超过冻结上限 2，因而严格计为方法失败，没有事后放宽门槛。

## 4. 论文产物与口径

`paper_artifacts` 中已生成 6 张 PNG、6 张同源 PDF 和 3 张 LaTeX 表。`final_evidence` 中包含成功率、失败分类、McNemar、Friedman、Wilcoxon-Holm、共同成功集及 V2.1 接管审计。主表 `Plan P95` 统一定义为所有实测规划尝试的 P95；“每个成功任务内部 P95”只用于补充分布，不与主表混用。

Jerk、净空、跟踪与耗时必须同时报告各方法自身成功集和五方法共同成功集；COM 偏移与重力力矩仅称为动力学/飞行负担代理，不宣称为实测电机推力或力矩。

## 5. 最终验证

- `catkin_make run_tests_uav_continuum_whole_body_v2`：80/80 PASS；
- 终审工具独立重算 450 条结果：PASS；
- 43 个核心表图/证据产物 SHA-256 复核：全部一致；
- 论文图表数量、非空文件和 LaTeX 表结构检查：PASS。

最终审计入口为 `final_completion_audit.json` 和 `FINAL_COMPLETION_AUDIT_CN.md`。此后若修改算法、baseline、场景、地图、失败定义或主指标口径，不得继续沿用本 E3 冻结结果。

可使用下列命令在原始结果上重建统计、论文图表和哈希审计（不重跑 Gazebo）：

```bash
source xtdrone_catkin_ws/src/uav_continuum_whole_body_v2/env_v2_deps.sh
python3 xtdrone_plan/versioned_development/rolling_whole_body_v2/finalize_v21_e3.py \
  xtdrone_plan/experiments/method_comparison/v21_e3_final_450_commite62b6f2
```
