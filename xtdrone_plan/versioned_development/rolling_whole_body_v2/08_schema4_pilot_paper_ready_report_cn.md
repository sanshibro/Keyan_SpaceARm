# 基于独立 RGB-D 地图的滚动全身规划三场景仿真 Pilot

> 用途：可直接改写进论文“实验设置/初步结果”部分。当前证据等级为正式协议
> pilot，不得将其中的描述性优势表述为统计显著性优势。

## 实验设置

实验在 PX4 SITL、Gazebo 与 MAVROS 闭环中进行，比较 Fixed-Morphology、
AET-MinJerk、RRT*-UAM、AM-Planner-Adapted 和本文 Rolling-WB-V2。前四者是依据
相应方法思想在同一动力学、碰撞检测、任务完成与计算预算接口下实现的适配版，
不是对外部作者代码“逐行复现”的宣称。所有方法共享同一冻结任务、同一冻结地图、
同一初始状态与 15 s 单次规划预算，方法失败仍计入成功率分母。

场景由 Passage-Sweep、Multi-Passage 和 Clutter-Reach 三个族构成，本轮各取
medium 难度 seed 0。地图由与被比较方法相互独立的前视/下视 RGB-D 测绘飞行
生成，点云使用 PX4 EKF 传感器融合位姿配准；Gazebo 几何仅在执行后作为隐藏
oracle 计算真值净空，不进入在线规划。地图采用 unknown-as-occupied，并须通过
schema-v4 准入：占据表面 P95 误差不大于 0.12 m、共同 reference 计算不超过
15 s、路线逐状态全身可观测覆盖和终端 body/arm/whole-body 覆盖均为 100%。

主要指标为任务成功率、执行后 oracle 无碰撞率、任务完成时间、规划时间 P95、
deadline miss、最大任意/非终端停留、机体与机械臂最小净空、轨迹跟踪 RMSE、
最大质心偏移、单位路径 jerk 能量和末端误差。成功要求完成任务、无碰撞、无
unsafe/abort 事件、跟踪与末端误差不超过预注册阈值；所有判定均由脚本从原始
ROS 日志重算，而不是由 RViz 目测。

## Pilot 结果

三幅地图均通过 schema-v4，且 15 个方法 trial 构成 3 个完整 pair，无缺失和
重复。结果如下（P05/P95 表示跨三个场景的描述性分位数）：

| 方法 | 成功率 | 任务时间中位数 /s | 规划 P95 /s | Deadline miss | 任意/非终端 Hold P95 /s | Body/Arm P05 /m | RMSE 中位数 /m | COM P95 /m | Jerk/m 中位数 | Tip 中位数 /m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed-Morphology | 3/3 | 148.424 | 7.159 | 0 | 2.014/0 | 0.270/0.161 | 0.0203 | 0.0258 | 0.091 | 0.0145 |
| AET-MinJerk | 0/3 | - | 22.305 | 0.333 | - | - | - | - | - | - |
| RRT*-UAM | 0/3 | - | 17.646 | 0.333 | - | - | - | - | - | - |
| AM-Planner-Adapted | 0/3 | - | 23.062 | 1.000 | - | - | - | - | - | - |
| Rolling-WB-V2 | **3/3** | **99.996** | **4.155** | **0** | **1.006/0** | 0.211/**0.172** | 0.0235 | 0.0259 | 0.408 | **0.0138** |

Rolling-WB-V2 与 Fixed-Morphology 均完成全部 pilot；相较后者，V2 的任务时间
中位数减少 32.6%，规划 P95 减少 42.0%，且未发生非终端停留、deadline miss、
buffer underrun、abort 或 unsafe event。V2 的臂净空和末端精度略高，但机体
净空、跟踪 RMSE 与 jerk 指标不及 Fixed，表明当前优势主要是可变形协调带来的
窄空间通过效率，而不是所有轨迹质量指标上的全面占优。三个更强约束的适配
baseline 未完成任务，其碰撞拒绝、无路径或超预算原因均保存在逐 trial 结果中。

## 可复核证据与结论边界

统一输出目录为：

```text
xtdrone_plan/experiments/method_comparison/schema4_medium_three_family_pilot_aggregate/
```

其中 `main_table.csv` 是主表，`paired_trials.csv` 含 15 个逐 trial 数据，
`completeness.json` 验证完整性，`paper_eligibility.json` 验证地图与日志来源，
`paired_statistics.json` 管理统计结论门槛。当前
`artifact_eligible=true`、`statistical_claim_ready=false`；因此本节只能证明
实现、输入协议、闭环安全和跨场景可运行性，不能声称显著优于 baseline。
此外，这三个 pilot 产生于平衡顺序调度器冻结之前，五种方法采用相同固定顺序；
该顺序因素是其不能进入最终主表的第二个原因，而不是事后调整数值的理由。

最终论文主实验的验收条件是每个场景族至少 30 个预注册完整 pair（建议
easy/medium/hard 各 10 个），即 450 个闭环 trial；三族全部达到门槛后再报告
Wilson 95% CI、成对 McNemar、成对 Wilcoxon-Holm 与效应量。任何基础设施失败
应自动补跑，任何方法失败必须保留，测试集结果产生后不得逐场景调参。
