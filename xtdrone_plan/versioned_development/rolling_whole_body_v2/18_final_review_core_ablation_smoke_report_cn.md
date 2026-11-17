# V2.1 最终审稿修改与核心消融冒烟报告

## 1. 版本与范围

- 开发分支：`experiment/v2.1-final-review-fixes`
- 消融接口提交：`03a734f`
- 冻结标签：`v2.1-final-review-ablation-ready-20260804`
- 冒烟场景：`passage_sweep_hard_s011`
- 输入：与 E3 一致的冻结 SLAM 栅格及其独立隐藏几何；四种方法共用地图、PX4 参数、15 s 规划截止期和闭环验收器。

本轮没有覆盖或重算既有 450 条 E3 主实验记录。图题、图例、C2 表述、推力代理误差以及统计解释等无需重跑的修改，见文档 17。下表仅用于验证消融实现和诊断链路，不作为论文统计结论。

## 2. 单场景冒烟结果

| 变体 | 结果 | 规划 P50/P95 (s) | 验证拒绝 | 任务时间 (s) | Body/Arm 净空 (m) | 跟踪 RMSE (m) | Jerk/m | 接管段 Jerk 占比 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full | PASS | 2.414/2.589 | 0 | 107.212 | 0.2109/0.1348 | 0.0213 | 0.2134 | 1.22% |
| w/o Seed | FAIL | 6.284/6.495 | 8 | — | — | — | — | — |
| Fixed-phi | TIMEOUT | 1.853/2.922 | 50 | — | — | — | — | — |
| Position-only Switch | PASS | 2.364/2.998 | 0 | 93.100 | 0.2168/0.1406 | 0.0233 | 0.2085 | 7.47% |

一次人为中断的 Fixed-phi 运行被记录为 `SYSTEM_ERROR` 并由统计脚本排除；它不计入上述算法试验。

## 3. 诊断结论

1. **Full 回归通过。** 默认规划与全状态接管接口没有被消融代码破坏，任务完成、无 deadline miss、无 unsafe/abort，末端误差为 0.0065 m。
2. **w/o Seed 的失败机制可审计。** 日志明确记录 `reference_only_ablation`、零风险结点和零离散扩展；原始参考经过连续细化后仍发生全臂碰撞，候选被独立密集验证器拒绝，而不是仿真基础设施故障。
3. **Fixed-phi 的失败机制可审计。** 日志明确记录 `theta_only_fixed_phi` 和固定 `phi=0`；前两段可行，但末端窗口无法同时满足工具端约束，连续出现 `terminal_tip_error` 后任务超时。这证明接口确实锁定了弯曲平面，但单场景结果不能替代正式统计。
4. **Position-only 的对照有效。** 日志明确记录 `position_only_ablation`，未授权 C2 桥接。其总 Jerk/m 在该次运行中没有恶化，但接管邻域贡献由 Full 的 1.22% 增至 7.47%，约为 6.1 倍；因此论文应比较“接管局部跳变”，不能只看全程 Jerk 总量。

## 4. 验收判断与下一批次

四种变体均生成了独立方法标签、manifest、事件日志和 trial result；Full 通过，三个目标诊断标记均已出现，系统错误可被独立排除。因此核心消融接口达到“可进入资格测试”的工程验收条件。

尚不能把 1 个场景写成消融结论。下一步必须依次执行：

1. 三场景族 × 三难度的 9 场景资格测试；
2. 资格测试通过后，按同一 90 个冻结场景运行四变体，共 360 条正式闭环记录；
3. 使用 `analyze_core_ablation.py` 统一生成成功率、Wilson 95% CI、分组成功率、规划时间、验证拒绝、净空、COM、倾角/推力代理、Jerk/m 与接管局部指标；
4. 压力扫描与高动态推力验证单独成组，不与核心消融主表混合。

复核命令：

```bash
python3 xtdrone_catkin_ws/src/uav_continuum_whole_body_v2/scripts/analyze_core_ablation.py \
  xtdrone_plan/experiments/method_comparison/v21_final_review_ablation_smoke_commit03a734f
```

机器可读结果位于该实验根目录的 `ablation_audit/summary.json`，逐运行表位于 `ablation_audit/runs.csv`。
