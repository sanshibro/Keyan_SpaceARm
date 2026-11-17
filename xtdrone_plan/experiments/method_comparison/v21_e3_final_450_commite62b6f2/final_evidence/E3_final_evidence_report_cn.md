# Rolling-WB-V2.1 E3 预注册实验最终证据报告

## 1. 完整性与可用性

- 设计完整性：`PASS`（450 个 trial，90 个完整配对任务）。
- 随机种：[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]；每场景族 30 对，每难度 30 对，每“场景族×难度”单元 10 对。
- 原始失败和系统重试均保留，本报告未删除 baseline 失败。

## 2. 主要结果

| Method | Success [Wilson 95% CI] | Task median [IQR] (s) | Plan P95 (s) | Body/Arm P05 (m) | Track RMSE med. (m) | Jerk med. | Tip med. (m) |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_morphology | 55/90, 61.1% [50.8, 70.5] | 130.7 [89.86, 142.1] | 2.936 | 0.2324/0.149 | 0.02139 | 0.08674 | 0.01759 |
| aet_sequential_minjerk | 61/90, 67.8% [57.6, 76.5] | 98.06 [83.82, 108.6] | 10.44 | 0.2295/0.1436 | 0.02847 | 0.2734 | 0.01405 |
| informed_rrtstar_wb | 68/90, 75.6% [65.8, 83.3] | 168.6 [139.1, 207.1] | 13.02 | 0.1793/0.139 | 0.01953 | 0.06186 | 0.01546 |
| global_batch_wb | 59/90, 65.6% [55.3, 74.6] | 99.68 [92.09, 107.2] | 10.39 | 0.2286/0.1434 | 0.02698 | 0.2574 | 0.01438 |
| rolling_wb_v2 | 89/90, 98.9% [94.0, 99.8] | 103.6 [83.1, 122.5] | 3.697 | 0.1827/0.1442 | 0.02402 | 0.1263 | 0.01293 |

## 3. 配对统计

- 五种方法共同成功子集：54/90。
- McNemar 以所有 90 个配对任务为样本；连续指标只在五方法共同成功集合上作 Friedman 及 Wilcoxon–Holm，并对配对中位差给出 10000 次配对 bootstrap 95% CI，避免难度选择偏差。

| Baseline | Ours-only success | Baseline-only success | Exact McNemar p |
|---|---:|---:|---:|
| fixed_morphology | 34 | 0 | 1.16e-10 |
| aet_sequential_minjerk | 28 | 0 | 7.45e-09 |
| informed_rrtstar_wb | 21 | 0 | 9.54e-07 |
| global_batch_wb | 30 | 0 | 1.86e-09 |

## 4. V2.1 闭环、接管与动力学代理审计

- V2.1 验收报告：90；通过：89；规划尝试：494；deadline miss：0。
- 滚动接管：391 次（直接接管 339，使用 C2 桥接的验证接管 52，缺失诊断 0）。
- 桥接前候选对齐间隙 median/P95/max：位置 0.0156/0.05493/0.1276 m，速度 0.007708/0.01932/0.03196 m/s，加速度 0.01955/0.03924/0.08707 m/s²。
- 上述是可选 C2 桥接之前的候选间隙，不当作实际下发指令的不连续量，也不单独构成严格解析 C2 连续性的证明；实际命令连续性见论文补充图表。
- 规划时间 median/P95/max：2.256/3.697/8.779 s；最长持续 hold P95/max：2.077/4.576 s。
- 最小实测机体/臂净空：0.1686/0.122 m；跟踪 RMSE P95：0.02636 m。
- COM 偏移 P95：0.02757 m；重力力矩代理 P95：0.6085 N·m。这两项只作保守动力学代理，不宣称为实测电机推力/力矩。
- COM 差异只解释为与基线处于可比量级；本方法较小但满足阈值的机体净空解释为空间利用，而不宣称获得最大安全裕度。

## 5. 失败分类

```json
{
  "aet_sequential_minjerk": {
    "NONE": 61,
    "PLANNING_OR_LIVE_GATE_FAILURE": 28,
    "VALIDATION_REJECT": 1
  },
  "fixed_morphology": {
    "EXECUTION_HANDOFF_ABORT": 1,
    "NONE": 55,
    "VALIDATION_REJECT": 34
  },
  "global_batch_wb": {
    "EXECUTION_HANDOFF_ABORT": 1,
    "NONE": 59,
    "PLANNING_OR_LIVE_GATE_FAILURE": 28,
    "PLAN_TIMEOUT": 2
  },
  "informed_rrtstar_wb": {
    "NONE": 68,
    "NO_PATH": 18,
    "PLANNING_OR_LIVE_GATE_FAILURE": 1,
    "VALIDATION_REJECT": 3
  },
  "rolling_wb_v2": {
    "NONE": 89,
    "repeated_planning_rejection": 1
  }
}
```

## 6. 结论使用边界

只有当本报告第 1 节为 PASS，且 `paper_eligibility.json` 同时给出 `eligible=true` 和 `statistical_claim_ready=true`，才可将这批数据称为预注册 E3 正式结果。对于 jerk、净空和耗时，论文必须同时报告自身成功集与共同成功集，不得只选有利子集。不得宣称全局最优、完整刚柔耦合动力学保证或解析连续碰撞证明。
