# Rolling Whole-Body Planner V2

> **当前验收状态（2026-07-21 20:28 CST）**：基线和自适应实时研究
> profile 都保留。实时 profile 的干净冷启动三墙记录
> `run_20260721_202321_713638` 已通过：`TASK_COMPLETE`、8 次滚动切换、
> 0 次 unsafe/abort，实测 body/arm 最小净空 0.1123/0.0736 m，
> 跟踪 RMSE 0.0367 m，工具端终点误差 0.0682 m，最大连续等待
> 2.61 s。可行时 `phi=0`，只有独立验证证明固定弯曲平面不安全才解锁
> `phi`。这是单次 SITL 闭环证据，
> 不等于多 seed 论文统计或实机实时通过。

这是从 V1 冻结基线独立开发的论文研究版本。V1 对应 commit
`5dd7bdff3d72fde7890de4d1f128719b6a40b8f8`；V2 不修改该冻结基线的算法语义。

V2 对每个 3 m 滚动窗内的连续状态

```text
[x, y, z, yaw, theta, phi]
```

及非均匀 B-spline 时间联合优化。风险段只增加形态候选、配点密度和修复预算，
并不意味着只有风险点参与优化。实现采用 reduced-space 控制点修正场，不是把每个
50 ms 输出样本作为相互独立的变量；整窗曲线上的所有状态仍随优化变量共同变化，
最终由独立高密度扫掠验证器决定是否可执行。

滚动窗的首个状态严格承接实测形态，未来 transit reference 使用直臂
`theta=phi=0`作为软偏好；这不是把柔性臂强制伸直。优化器在障碍净空、COM、
重力矩、形态速度/加速度和时间代价下决定 `theta`的连续变化；固定
`phi=0` 候选只在独立扫掠验证报告臂部碰撞时才解锁为 `theta+phi`。

## 方法闭环

```text
global/EGO 局部几何参考（禁止拼接 global tail）
  -> 截取完整 3 m horizon
  -> 风险检测与邻域膨胀
  -> whole-body beam/DP seed
  -> UAV/yaw/arm/time 连续联合优化
  -> body/arm/COM/重力矩/运动学/动力学约束
  -> 独立自适应高密度 swept-volume 验证
  -> safe trajectory 双缓冲
  -> 在已认证后继轨迹内做有界相位匹配，消除规划延迟回跳
  -> UAV 与 theta/phi 同一时间轴同步执行
```

环境采用同一场景清单生成的精确盒体、ESDF 和共享静态点云。在线规划只查询当前
horizon 邻域的点云 KD-tree，最终验证仍使用完整局部精确几何和最新点云。执行器
保护已认证 active/pending 轨迹，不允许后来的候选覆盖尚未切换的安全后继轨迹。

## 构建与离线测试

```bash
cd /home/uav/桌面/uav_arm_1
./xtdrone_catkin_ws/src/uav_continuum_whole_body_v2/scripts/bootstrap_local_scipy.sh  # 仅首次
source /opt/ros/noetic/setup.bash
source xtdrone_catkin_ws/src/uav_continuum_whole_body_v2/env_v2_deps.sh
cd xtdrone_catkin_ws
catkin_make --pkg uav_continuum_whole_body_v2
source devel/setup.bash

rosrun uav_continuum_whole_body_v2 generate_unified_scenario.py --build-esdf
rosrun uav_continuum_whole_body_v2 run_offline_demo.py --time-budget 6
catkin_make run_tests_uav_continuum_whole_body_v2
catkin_test_results --verbose build/test_results/uav_continuum_whole_body_v2
```

基线约束以 `config/planner_v2.yaml` 为准，自适应实时 profile 以
`config/planner_v2_realtime.yaml` 为准，实测中止阈值由 executor launch
明确给出，两者不能混为一组数字：

- 计划 body/arm clearance >= 0.10/0.07 m；
- 实测 body/arm 硬中止线 >= 0.06/0.02 m；
- COM offset <= 0.08 m；
- baseline/realtime velocity <= 0.12/0.25 m/s，acceleration <= 1.00 m/s^2，
  jerk <= 3.00 m/s^3；
- z in [0.75, 2.20] m，tilt <= 15 deg，thrust ratio in [0.50, 1.60]；
- validator status 必须是 `safe`，否则不得覆盖已认证缓存。

## 原三墙场景：经过联调的启动顺序

若只需一次可重复、可离线审计的完整仿真，在确认所有旧 ROS/Gazebo/PX4
终端已关闭且根分区至少有 20 GiB 空间后，可直接执行：

```bash
/home/uav/桌面/uav_arm_1/xtdrone_catkin_ws/src/uav_continuum_whole_body_v2/scripts/run_scenario_3c_acceptance.sh
```

上述命令运行可复现基线。要运行已通过的自适应实时研究 profile，
只需一条命令：

```bash
/home/uav/桌面/uav_arm_1/xtdrone_catkin_ws/src/uav_continuum_whole_body_v2/scripts/run_scenario_3c_acceptance.sh --realtime
```

该命令默认打开 RViz，自动完成冷启动、起飞、规划门控、执行、实测审计和
进程清理。若还需观察 Gazebo GUI，在命令后加 `--gazebo-gui`；两个三维
界面同时运行会增加 CPU 负载。

默认打开 RViz 但不打开 Gazebo GUI，以避免远程桌面下两个三维界面抢占 CPU。
如需同时观察 Gazebo：

```bash
/home/uav/桌面/uav_arm_1/xtdrone_catkin_ws/src/uav_continuum_whole_body_v2/scripts/run_scenario_3c_acceptance.sh --gazebo-gui
```

该脚本会自动执行下文的全部安全顺序，只有 `check_v2_live.py` 通过才会启用
executor；完成后自动运行 `analyze_v2_run.py`，打印 `V2_ONE_CLICK_RESULT`、
`RUN_DIR` 和 `ACCEPTANCE_REPORT`，然后关闭仿真进程。任何失败都不会自动放宽阈值或
继续飞行。

`scenario_3c_v2_full.launch` 会启动 V1 的 global/reachability/EGO-reference 前端、
V2 planner/executor/logger/visualizer 和专用 RViz，但不会启动 Gazebo、PX4、MAVROS、
共享静态点云或 EGO planner 本体。

每个终端先执行：

```bash
source /opt/ros/noetic/setup.bash
source /home/uav/桌面/uav_arm_1/xtdrone_catkin_ws/src/uav_continuum_whole_body_v2/env_v2_deps.sh
source /home/uav/桌面/uav_arm_1/xtdrone_catkin_ws/devel/setup.bash
export XTDRONE_DIR=/home/uav/桌面/XTDrone_test/XTDrone
export ROS_PACKAGE_PATH=/home/uav/PX4-Autopilot:/home/uav/PX4-Autopilot/Tools/sitl_gazebo:$XTDRONE_DIR/sitl_config:$ROS_PACKAGE_PATH
```

按下列顺序启动。顺序是验收条件的一部分。

终端 1，启动 Gazebo/PX4/MAVROS/深度相机：

```bash
roslaunch xtdrone_ego_tools indoor1_depth_xtdrone.launch x:=0 y:=0 z:=1 gui:=true
```

该 launch 默认给 PX4 使用每次唯一的干净 SITL 工作目录，防止上一次碰撞后保存的
EEPROM/IMU 标定污染下一次运行。不要为复现实验手工传入固定的 `px4_workdir`。

终端 2，发布共享静态点云：

```bash
roslaunch uav_continuum_opportunity_planner static_scene_cloud.launch
```

终端 3，启动 EGO；必须关闭它自己的轨迹执行器，且不需要在 RViz 手工点目标：

```bash
roslaunch xtdrone_ego_tools single_uav_rviz_goal.launch enable_traj_server:=false
```

终端 4，启动三墙前端、V2 和 V2 专用 RViz。首次保持禁止执行：

```bash
roslaunch uav_continuum_whole_body_v2 scenario_3c_v2_full.launch auto_execute:=false
```

等待终端出现 `spawned 16 task boxes` 后再起飞。若在障碍生成期间解锁，Gazebo
暂停可能使 PX4 退出 OFFBOARD。

终端 5，使用单一、安全限速的起飞悬停发布器：

```bash
roslaunch uav_continuum_whole_body_v2 safe_takeoff_hold.launch x:=0 y:=0 z:=0.65
```

终端 6，加载 SITL 参数，并在 setpoint 已持续发布至少 2 s 后切换和解锁：

```bash
roslaunch xtdrone_ego_tools px4_sitl_param_fix.launch profile:=takeoff
rosservice call /iris_0/mavros/set_mode "{base_mode: 0, custom_mode: 'OFFBOARD'}"
rosservice call /iris_0/mavros/cmd/arming "{value: true}"
rostopic echo -n 1 /iris_0/mavros/state
rostopic echo -n 1 /iris_0/mavros/local_position/pose
```

参数脚本除关闭无 RC 的 SITL failsafe 外，还把 PX4 水平速度、升降速度、倾角和加速度
限制到低速实验包络；不得跳过，也不得用旧终端中遗留的参数替代。

确认 `connected: True`、`armed: True`、`mode: OFFBOARD`，局部位置接近
`x=0, y=0, z=0.65`。这里不再使用无反馈阶跃式 `rostopic pub -r 30` 起飞。
飞机稳定后、启动真值桥之前，切换到跟踪参数：

```bash
roslaunch xtdrone_ego_tools px4_sitl_param_fix.launch profile:=tracking
```

若起飞后任一时刻 `abs(x)>0.30`、`abs(y)>0.30`，或机身明显倾斜，立即执行
`AUTO.LAND`，不得继续启动真值桥或运行 V2 验收。`check_v2_live.py` 出现
`world_pose_altitude_out_of_bounds` 时同样不得启用 executor。

`safe_takeoff_hold_v2` 还会独立监视 MAVROS 局部位姿：水平偏离超过 0.50 m 或倾角
超过 35 deg 时自动请求 `AUTO.LAND`，超过 60 deg 时同时请求紧急解除武装。可用
下列命令查看其状态；出现 `EMERGENCY_ABORT` 后必须完整重启仿真：

```bash
rostopic echo /planner/v2/takeoff_hold_status/data
```

终端 7，飞机稳定后最后启动真值桥：

```bash
roslaunch xtdrone_ego_tools gazebo_truth_for_ego.launch
```

真值桥必须最后启动，否则 global/EGO/V2 可能在飞机落地位置上提前生成过期参考。
等待以下命令看到 `SAFE_TRAJECTORY`：

```bash
rostopic echo /planner/v2/status/data
```

规划期的自动验收可运行：

```bash
rosrun uav_continuum_whole_body_v2 check_v2_live.py --timeout 180
```

## 安全交接与执行

只有 `check_v2_live.py` 通过且 RViz 轨迹合理时，才把唯一 setpoint 发布权从起飞
悬停节点交给 V2。推荐在终端 6 执行以下原子式交接：

```bash
rostopic pub -1 /planner/v2/executor_enable std_msgs/Bool "data: true" &
sleep 0.2
rosnode kill /safe_takeoff_hold_v2
wait
```

不得让两个 setpoint 发布器长期并行，也不得先停悬停节点再启用 V2。执行中监视：

```bash
rostopic echo /planner/v2/status/data
rostopic echo /planner/v2/executor_status/data
rostopic echo /planner/v2/actual_safety_status/data
```

成功标准是：

- 规划器最终报告 `TASK_COMPLETE`；
- executor 应出现 `SWITCHED_ROLLING`；当前关闭短准备窗，后继必须是同一
  validator 认证的完整 whole-body 窗，不允许未认证的 global 段拼接；
- 无 `ABORT_*`、无越界、无 unsafe trajectory 被接受；
- 实际工具端连续进入 0.08 m 终点球并保持至少 1 s；
- 任务终点位于后部货架内，不得原地 `AUTO.LAND`；验收后应结束仿真，
  或先规划并执行安全退出轨迹，再在开阔区降落。

紧急停止时：

```bash
rostopic pub -1 /planner/v2/executor_enable std_msgs/Bool "data: false"
rosservice call /iris_0/mavros/set_mode "{base_mode: 0, custom_mode: 'AUTO.LAND'}"
```

## RViz 验收内容

`scenario_3c_v2_full.launch` 默认自动打开专用 RViz，Fixed Frame 为 `world`。它包括：

- global topology（仅参考，不执行）；
- EGO local reference；
- V2 当前已接受的局部 UAV path；
- V2 规划 UAV+柔性臂 ghost shapes；
- 实际执行 UAV history 与柔性臂 shape history；
- 当前柔性臂 backbone、精确障碍 marker 和共享静态点云。

`/planner/v2/uav_path` 只表示当前已认证的 3 m 局部窗，不是一条预先固定的三墙
全局执行轨迹。完整穿墙过程通过持续更新的局部窗和
`/planner/v2/executed_path` 显示。规划柔性臂 ghost 来自同一条
`/planner/v2/whole_body_trajectory` 中的 theta/phi，不是只显示当前姿态。

现有 `qingyu_continuum_description` Gazebo 链路主要是运动学可视化：通过
`set_link_state` 更新无重力、kinematic links，当前没有可用于接触验证的
collision/dynamics/controller 链。因此 RViz ghost 可用于检查几何和角度连续性，
但不能将该 Gazebo 模型宣称为柔性体动力学或接触证据。

## 已通过的冷启动记录与当前边界

当前自适应实时 profile 的最新完整冷启动闭环记录：

```text
xtdrone_plan/experiments/v2_runs/run_20260721_202321_713638
TASK_COMPLETE
measured samples: 2343
SWITCHED_ROLLING: 8
abort / unsafe events: 0 / 0
accepted-window planning: median 7.009 s, P95 9.464 s, max 10.319 s
all attempts (including rejected): median 7.435 s, P95 20.109 s, max 22.647 s
maximum continuous HOLD_FINAL: 2.612 s; maximum rejection streak: 1
measured theta: 0--118 deg, std 40.67 deg; phi: 0 deg in feasible transit windows,
  unlocked to -180 deg for the third-passage/terminal task
measured body/arm clearance min: 0.1123 / 0.0736 m
tracking error: RMSE 0.0367 m, P95 0.0607 m, max 0.0894 m
terminal actual tool-tip error: 0.0682 m
one-click wall time: about 347 s; enabled execution: about 235 s
analyze_v2_run.py: passed=true
```

飞行后必须用实测位姿重算全程包络；仅 `SAFE_TRAJECTORY` 和 RViz 可视化不算
闭环验收：

```bash
rosrun uav_continuum_whole_body_v2 analyze_v2_run.py \
  /home/uav/桌面/uav_arm_1/xtdrone_plan/experiments/v2_runs/<run_id>
```

这说明当前版本已是通过单次严格审计的自适应滚动仿真闭环实现，而不是实时飞控或实机完成版本。
本次已接受窗最大规划时间 10.32 s；一次 active-set 失败尝试可达
22.65 s，但执行器在已认证双缓冲内继续飞行，未形成墙前长时停留。因此该
profile 是“有限计算延迟可被滚动缓冲覆盖”的研究实现，不应宣称为硬实时。
实机部署仍需规划器性能优化、感知时延标定、
控制器鲁棒性验证和硬件在环测试。单次成功运行不能替代论文统计结论，正式论文仍须执行
`04_paper_experiment_protocol_cn.md` 中的多 seed、baseline、消融和鲁棒性实验。

运行数据写入 `xtdrone_plan/experiments/v2_runs/<run_id>/`。默认禁止自动执行；验证
失败时不会覆盖当前 active/pending 安全缓存。

## 方法对比开发入口

正式对比分支提供五种统一方法名：

```text
fixed_morphology
aet_sequential_minjerk
informed_rrtstar_wb
global_batch_wb
rolling_wb_v2
```

其中 AET-Sequential 是固定 UAV 参考后的形态分解，Informed-RRT*-WB 是
公共 SLAM 参考引导的六维 anytime 采样搜索，Global-Batch-WB 是带已验证
incumbent 的非滚动全时域联合优化。这三者都是公开记录适配项的结构基线，
不得在论文中声称为原作者代码的精确复现。主表统一方法预算为 15 s；
Informed-RRT*-WB 和 Global-Batch-WB 另以 60 s 预算作附录强基线检查，
两个预算组的结果不混合。

Schema-v6 的基线几何查询必须使用与逐状态 PCC/机体采样数学等价的批量实现；
该优化只消除 Python 循环，不改变采样点、SLAM 距离场、净空或动力学阈值。
方法日志中的 `stage_times_s` 分别记录形态走廊、采样改进、独立扫掠验证和
轨迹采样时间。形态走廊按统一墙钟截止时间 fail-closed，禁止在超时后把候选
伪装成实时成功；已经在截止时间内独立验证的 incumbent 才允许安全回退。
Global-Batch-WB 只有在独立验证后仍保留至少 8 s 净精修预算时才启动不可中断的
全时域候选评估；否则15 s实时组返回已验证全局走廊。60 s补充组仍执行完整批优化，
两组均在日志中明确记录 `optimizer_status`，不得把跳过精修写成优化收敛。

先生成参数化开发场景并运行一个离线 paired pilot：

```bash
rosrun uav_continuum_whole_body_v2 generate_comparison_scenarios.py \
  --families passage_sweep multi_passage clutter_reach \
  --difficulties easy medium hard --seeds 0 1 2 --resolution 0.05

rosrun uav_continuum_whole_body_v2 run_method_comparison_pilot.py \
  /home/uav/桌面/uav_arm_1/xtdrone_plan/experiments/method_comparison/scenarios \
  /home/uav/桌面/uav_arm_1/xtdrone_plan/experiments/method_comparison/pilot \
  --time-budget 10.0

rosrun uav_continuum_whole_body_v2 aggregate_method_comparison.py \
  /home/uav/桌面/uav_arm_1/xtdrone_plan/experiments/method_comparison/pilot
```

以上命令前必须依次加载 ROS、V2 隔离依赖和工作空间；否则系统 SciPy 可能找不到
隔离目录中的动态库：

```bash
source /opt/ros/noetic/setup.bash
source /home/uav/桌面/uav_arm_1/xtdrone_catkin_ws/src/uav_continuum_whole_body_v2/env_v2_deps.sh
source /home/uav/桌面/uav_arm_1/xtdrone_catkin_ws/devel/setup.bash
```

不同场景族保存在独立根目录时，可在同一次汇总中列出多个根目录；此时必须显式
指定输出目录。汇总器按文件绝对路径去重，并在出现相同 `scenario-method` 时拒绝
生成伪完整结果：

```bash
rosrun uav_continuum_whole_body_v2 aggregate_method_comparison.py \
  /absolute/run_root_a /absolute/run_root_b /absolute/run_root_c \
  --output /absolute/paper_aggregate
```

`paper_eligibility.json` 将“日志/地图是否具备论文资格”和“样本量能否支持统计
结论”分开报告。只有三个场景族各完成至少 30 个预注册 paired trials 时，
`statistical_claim_ready` 才会为 `true`；在此之前 `paired_statistics.json` 会保留
描述统计但主动隐藏 p 值。

生成的 `development_map.npz` 明确标记为 `synthetic_perfect_map` 和
`truth_derived=true`，只允许用于代码开发；`--formal` 会主动拒绝它。正式实验必须把
传感器/SLAM 录制的地图放入场景目录，并满足
`formal_experiment_allowed=true`、无真值访问和干净 commit。

录制传感器地图时，先将深度相机点云按相机位姿配准，再生成
`MapSnapshot`：

```bash
roslaunch uav_continuum_whole_body_v2 register_depth_cloud.launch \
  depth_cloud_topic:=/camera/depth/points \
  camera_pose_topic:=/iris_0/camera_pose

roslaunch uav_continuum_whole_body_v2 slam_map_snapshot.launch \
  mode:=record snapshot_file:=/absolute/path/to/frozen_map.npz \
  registered_cloud_topic:=/slam/registered_cloud \
  sensor_pose_topic:=/slam/sensor_pose \
  localization_source:=gazebo_truth formal_experiment_allowed:=false
```

上述 `gazebo_truth` 位姿只能作为“真值定位隔离误差的传感器建图”开发实验，
不能写成 SLAM 主结果。当 `camera_pose_topic` 替换为已审计的 VIO/LIO/SLAM
相机位姿后，录制结果仍不能自行声明论文可用；必须先执行：

```bash
rosrun uav_continuum_whole_body_v2 audit_slam_map.py \
  /absolute/path/to/recorded_map.npz \
  /absolute/path/to/oracle_truth.yaml /absolute/path/to/task.yaml \
  --report /absolute/path/to/map_audit.json \
  --audited-map-output /absolute/path/to/audited_map.npz
```

### 前视+下视冻结地图（方法对比）

完整 formal pilot 可直接用一条可恢复命令运行：

```bash
rosrun uav_continuum_whole_body_v2 run_formal_comparison_pipeline.sh \
  --scenario-dir /absolute/path/to/scenario \
  --output-dir /absolute/path/to/formal_pilot \
  --method-time-budget 15 --preflight-timeout 60 --task-timeout 600
```

命令依次完成独立前/下视测绘、schema-v5 审计、五方法闭环与表格聚合。
中断后重跑会复用已认证地图和已留痕 trial；某方法无解/超时是实验数据，
不会中断整批。

最终多场景主实验用预注册批处理器启动。先用 `--dry-run` 冻结并核对任务总数；
同一输出目录之后拒绝改变 commit、任务哈希、预算、方法或任务集合：

```bash
rosrun uav_continuum_whole_body_v2 run_formal_comparison_batch.py \
  /absolute/path/to/scenarios /absolute/path/to/formal_batch \
  --families passage_sweep multi_passage clutter_reach \
  --difficulties easy medium hard --seeds 0 1 2 3 4 5 6 7 8 9 \
  --method-time-budget 15 --preflight-timeout 60 --task-timeout 600 \
  --require-pairs-per-family 30 --require-statistical-ready --dry-run

# 核对 preregistered_batch_manifest.json 后，删除末尾 --dry-run 原样重跑。
```

该命令逐场景调用上述 schema-v5 流水线，掉电/中断后可原命令续跑；每个任务按
冻结的场景族、难度和 seed 对五种方法作平衡 Latin 循环轮换并写入
`method_order.json`。每个场景族内五种起始位置各出现 6 次，避免固定方法顺序
与仿真器热状态混杂。最终只有 90 个完整 pair、450 个有审计证据的 trial、
三族各 30 个样本以及统计门全部满足时才打印 `FORMAL_COMPARISON_BATCH=PASS`。
可在启动耗时仿真前单独核对某任务的实际顺序：

```bash
rosrun uav_continuum_whole_body_v2 run_formal_comparison_pipeline.sh \
  --scenario-dir /absolute/path/to/scenario \
  --output-dir /absolute/path/to/scenario_output --preflight-only
```

正式测绘不在飞行过程中压缩百万体素中间快照，而是在 25/25 航点和终端扫描
完成后通过 `/slam_map_snapshot/save_snapshot` 串行、显式保存；服务必须返回
成功和地图哈希后才能认证。这样最终体素集合不变，同时避免压缩阻塞 ROS 服务。

正式对比不得直接混用解析障碍和点云。先分别用前视、下视 RGB-D 完成两次测绘，
两次都必须执行同一份不读取障碍、Oracle 或任何对比规划器的冻结航点清单：

```bash
rosrun uav_continuum_whole_body_v2 generate_independent_survey_manifest.py \
  /path/to/scenario/task.yaml /path/to/frozen_survey_manifest.json

rosrun uav_continuum_whole_body_v2 run_independent_map_survey.sh \
  --scenario-dir /path/to/scenario \
  --manifest /path/to/frozen_survey_manifest.json \
  --output-dir /path/to/front_session --sensor-mode front

rosrun uav_continuum_whole_body_v2 run_independent_map_survey.sh \
  --scenario-dir /path/to/scenario \
  --manifest /path/to/frozen_survey_manifest.json \
  --output-dir /path/to/down_session --sensor-mode down
```

`certified_sensor_survey.npz` 只在实测轨迹按顺序到达清单航点后生成。
再生成所有方法共享的只读地图：

```bash
rosrun uav_continuum_whole_body_v2 fuse_slam_maps.py \
  /path/to/front_session/certified_sensor_survey.npz \
  /path/to/down_session/certified_sensor_survey.npz \
  --output /path/to/fused_candidate.npz \
  --report /path/to/fusion_report.json \
  --sensor forward_rgbd --sensor downward_rgbd \
  --comparison-independent-survey \
  --survey-protocol task_endpoint_scan_survey_v6_explicit_flush \
  --survey-manifest /path/to/frozen_survey_manifest.json \
  --observation-support-voxels 1 \
  --corridor-xy-padding 1.2 --corridor-z-min 0.2 --corridor-z-max 2.5

rosrun uav_continuum_whole_body_v2 audit_slam_map.py \
  /path/to/fused_candidate.npz \
  /path/to/scenario/oracle_truth.yaml /path/to/scenario/task.yaml \
  --report /path/to/map_audit_report.json \
  --audited-map-output /path/to/audited_frozen_map.npz
```

融合只做占据/已观测并集，不删除占据点；一体素有限像素支撑和 1.2 m 走廊裁剪
对所有方法固定。审计中的 Oracle 只在规划完成后计算误差，不发布给规划器。
融合器会逐个核对输入地图的测绘认证和清单哈希，不允许靠命令行自行声明。
只有 schema-v5 报告 `passed=true` 且输出地图元数据同时含
`formal_experiment_allowed=true`、`comparison_independent_survey=true` 和清单哈希时，
才能给 `--formal` 实验使用。

审计同时检查规划器占据点到隐藏真值表面的 P95 误差、误报率、共同
reference 可达性/总时间、代表性全身包络的平均覆盖、逐状态最差覆盖以及
终端抓取包络的 100% 覆盖；不可见墙背面召回只作描述。
schema-v5 只对规划地图中未与已测占据表面碰撞的代表状态施加 100% 逐状态
可观测要求；已知碰撞状态单独计数并排除，因为深度传感器不可能观察实体表面
之后的墙体内部。没有已知占据交点的未知体素仍按占据处理，终端全身包络仍须
无条件达到 100%，不能借此放宽未知空间。
只有 PASS 才生成 `formal_experiment_allowed=true` 的新地图与新哈希。
正式离线对比通过
`run_offline_method_comparison.py --map /absolute/path/to/audited_map.npz --formal`
显式选择地图，不再依赖固定文件名。

五种方法的文献来源、保留特征、适配项和禁止借用的 V2 特性固化在
`config/baseline_registry.yaml`。其中 AET、RRT*-UAM 和 AM-Planner 均是针对本平台的
`adapted_reimplementation`；论文中必须用“Adapted”名称，不得写成运行了作者原始二进制。

五种方法还共享同一条 SLAM 地图参考前端。该前端使用固定启发权重 1.5 的
Weighted-A* 快速求可行参考，并对每个体素继续执行已观测空间和机体包络检查；
它不是论文对比对象，也不宣称参考路径全局最优。启发权重必须对全部方法、场景固定。

0.05 m 是主对比分辨率；0.10 m 会因体素外接球与地图膨胀明显缩小窄缝，
只用于地图分辨率鲁棒性实验。`1 s` 预算只用于软件流水线冒烟，不能作为
RRT*、AM-Planner 和 V2 的正式公平预算。初始 pilot 使用 10 s，并同时报告实际墙钟时间。
预算从方法入口开始，包含方法特有的形态走廊/采样/种子、优化和独立验证；
任一环节使总墙钟超过预算都记为 `PLAN_TIMEOUT`。
有向 easy 闭环 10 s pilot 表明 AET/AM 分别因 11.37 s 总耗时和验证余量不足
而超时；在完全相同的 15 s 候选预算下补测，两者分别以 11.215/14.390 s
完成规划并闭环 PASS。因此 15 s 作为下一个公共候选预算；它仍须在全部方法和
开发集上验证后一次性冻结，不允许按方法或场景分别放宽。

场景生成器为 `parameterized_v3_oriented_walls`：除门洞宽高、墙厚、杂物和随机种子外，
各通道墙还含绕 z 轴的角度变化。合成点云、Gazebo 隐藏真值和 Oracle
使用同一有向几何。

使用真实冻结 SLAM 地图启动某种闭环方法：

```bash
roslaunch uav_continuum_whole_body_v2 method_comparison_core.launch \
  method:=fixed_morphology map_mode:=frozen_slam \
  map_snapshot_file:=/absolute/path/to/audited_slam_map.npz \
  task_file:=/absolute/path/to/task.yaml auto_execute:=false
```

`method_comparison_core.launch` 不会启动旧的已知障碍前端；统一 reference、baseline、
V2 和运行时安全检查均读取同一个 SLAM MapSnapshot。Gazebo/SDF 真值仅由离线 Oracle
评测器读取。

单个场景/方法的完整闭环试验只需一条命令：

```bash
rosrun uav_continuum_whole_body_v2 run_method_comparison_closed_loop.sh \
  --method rolling_wb_v2 \
  --scenario-dir /absolute/path/to/passage_sweep_medium_s000 \
  --map /absolute/path/to/passage_sweep_medium_s000/audited_map.npz \
  --method-time-budget 10 --task-timeout 900 --formal
```

它会拒绝已存在的 ROS/PX4/Gazebo 会话，并依次完成：受控空世界启动、
隐藏真值场景生成、共享地图规划、安全起飞、live gate、唯一 setpoint 交接、
任务执行和 Oracle 实测审计。失败 trial 也会保留 `trial_result.json`，不会被批处理静默丢弃。

完整闭环 paired pilot 使用一条可中断续跑的批处理命令：

```bash
rosrun uav_continuum_whole_body_v2 run_closed_loop_method_comparison.py \
  /absolute/path/to/scenarios /absolute/path/to/closed_loop_runs \
  --families passage_sweep multi_passage clutter_reach \
  --difficulties easy medium --seeds 0 1 2 \
  --map-name audited_map.npz --method-time-budget 10 --task-timeout 900 --formal
```

批处理按场景哈希循环轮换五种方法的运行顺序，降低仿真器热状态/时间漂移的
顺序偏差；再次执行时跳过已有 `trial_result.json` 的方法-场景对。算法无解、超时、
跟踪中止和 Oracle 不通过都保留为失败数据，只有没有形成审计结果的基础设施故障
才使批处理返回失败。禁止删除失败目录后只重跑失败方法。

任务精度不是统一放宽的：`Passage-Sweep/Multi-Passage` 是穿越任务，
`tip_tolerance_m=0.08`；`Clutter-Reach` 是末端操作任务，
`tip_tolerance_m=0.03`。该字段同时约束五种规划器、完成状态和独立 Oracle 审计。

已通过的新框架开发级闭环 smoke（合成真值地图，不可进入论文表）：

```text
scenario: Passage-Sweep medium, resolution 0.05 m
method: Rolling-WB-V2
task time: 83.648 s
planning attempts: 4, median 2.267 s, P95 2.505 s
non-terminal hold / deadline miss: 0.0 s / 0
Oracle body/arm clearance: 0.2121 / 0.1841 m
tracking RMSE/P95/max: 0.0244 / 0.0393 / 0.1111 m
COM offset max: 0.02584 m
terminal tip error: 0.06993 m
abort / unsafe: 0 / 0
```

Fixed-Morphology 也已在 `Passage-Sweep easy / 0.05 m` 完成同一 PX4/Gazebo
闭环 smoke：任务 79.168 s，规划 2.000 s，非终端停留与 deadline miss 均为 0，
Oracle 机体/臂净空 0.2405/0.3723 m，跟踪 RMSE 0.0232 m，末端误差 0.0250 m。
该结果仅验证 baseline 接口真实可执行，不能与上面的 medium 场景数值横向比较。

`aggregate_method_comparison.py` 同时纳入 PASS 和失败 `trial_result.json`，并拒绝
同一场景-方法的重复样本；随后生成主表、Wilson 95% CI、成对 McNemar、
Wilcoxon-Holm、完整性和 `paper_eligibility.json`。只有闭环、审计非真值地图、
相关代码干净三项同时满足时，数据才会标记为论文可用。

`Jerk energy` 不对每条候选/重叠滚动窗求和。executor 额外发布实际执行的
世界系速度命令。评测器只在任务执行区间统一重采样到 20 Hz，使用 0.5 s、三阶
Savitzky–Golay 微分器计算速度的二阶导数，再积分平移 jerk 平方并除以实际命令
路径长度。这样既抑制 ROS 回调时间戳抖动，又保证 baseline 和滚动方法均只计
“真正执行了什么”；滤波参数禁止按方法调整。
