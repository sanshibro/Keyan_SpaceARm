# FAST-LIO 接入状态（2026-11-09）

## 已验证

- MuJoCo 10 Hz 首表面激光扫描和 200 Hz IMU 经 ROS1 输入官方 FAST-LIO；发布字段为 `x,y,z,intensity,time,ring`，每束射线按同一仿真时刻采集。
- `run_fast_lio_pilot.sh` 的 15 秒轨迹收到 149 帧雷达、3000 条 IMU、144 条里程计和 144 帧世界坐标点云；对齐起点后的位移 RMS 44.3 mm，终点 15.8 mm。运行证据：`runs/20261107_fast_lio_online_map/`。
- 配准点云、RGB-D 深度和已观测自由射线生成 `MapSnapshot`；未获传感器证据的区域仍阻塞规划。已增加最多两轮、每轮一格的自由空间插值，候选须有至少两个相邻自由体素支持且距任何实测障碍大于 0.25 m；原始传感器地图另存为 `*_sensor_raw.npz`，推断数量写入元数据。直接入口 `run.sh mission-slam 180 --three-gaps --dashboard-video --gui` 可在桌面实时显示，无二维码、无飞控连接。
- FAST-LIO 里程计包含速度，状态估计器按原始时间戳融合位置、姿态、速度，并继续传播 IMU 到当前时刻。当前在线任务使用 16 线雷达近似、机身前向 D435 与末端 D435；机腹第三相机仅保留为可选试验。机身/机械臂的无接触 CAD 体积只标记其真实占据位置，实测障碍体素不会被擦除。
- 在线任务已恢复此前确认的 Stage 1 SUPER 原始机架、参考 D435 CAD、较短末端支架和 Qingyu RViz 黑色 15 点柔性臂装配；机身 D435 指向航线，末端 D435 保持原版朝向。对应自遮挡、工具包络和质心均使用同一装配。`runs/20261110_130215_slam_mission_development/` 的 12 秒自由飞行检查位置 RMS 14.8 mm、最大倾角 0.58°、0 次碰撞和电机饱和。当前仍是 MuJoCo 原型飞行动力学。
- 前沿搜索、地图快照、V2.1 规划和实时地图复核已移到独立进程或后台线程；物理在规划期间持续推进。三缝场景含三个交错窄门、低门槛、障碍与末端目标；直臂末端低于门槛，过门需要弯臂。`runs/20261109_195030_slam_mission_development/` 的 30 秒在线运行完成前两段短航迹，最大导航误差 17.4 mm、0 次碰撞。`runs/20261109_195713_slam_mission_development/` 的 65 秒在线运行连续完成三段 0.25 m 航迹，第四段正在执行，到第一道门前约 0.32 m；0 次碰撞，最大导航误差 55.4 mm（横向 FAST-LIO 漂移）。尚未穿过第一道门。
- 在线仿真新增 1280×720 主视角 `mission_view.mp4`，右侧实时显示控制实际消费的两台 D435 画面与地图；技术八宫格保留为 `online_dashboard.mp4`。三缝视频样例在 `runs/20261109_192530_slam_mission_development/`。
- 单门预扫描地图回归 `runs/20261108_140204_mission_development/` 已完成入口、接触抓取和两段带载退出，在仿真 222 秒进入 `completed_mission_hold`。整体 `passed=false`，因为统计窗口中的最小物体抬升量只有 33 mm，低于 50 mm 门槛。此模式使用冻结预扫描地图及 ArUco 定位，不能作为在线 SLAM 验收。

## 当前卡点

当前卡点是三道门的完整在线穿越尚未跑通。雷达和 D435 自由空间体素在近场有单格空洞，曾使航段缩短到 20–65 mm。受限插值后连续三段 0.25 m 已完成。当前代码把最大前沿扩展至 0.5 m；在 65 秒运行的地图上离线复核，0.5 m 前沿验证约 1.6 秒、V2.1 规划约 2.7 秒且轨迹终点准确。靠近第一道低门槛时，离线前沿选择先原地弯臂 95°，其 V2.1 轨迹验证通过；在线穿门仍待验证。前沿搜索已增加 25 mm 间距的快速路径清除检查。16 线雷达是当前稳定配置。

航段终点悬停曾因姿态估计小偏差发生厘米级往复摆动；当前用低噪声陀螺仪保持短时姿态，FAST-LIO 继续校正位置，在线 30 秒回归完成两段。该处理依赖已知初始姿态和仿真低漂移陀螺仪，尚不能代表真实飞控。目标检测、末端位姿、抓取和带载退出有开发模块，但在三缝在线模式未完成端到端验证；PX4 SITL 尚未连接。

## 依赖和复现

- FAST-LIO: `hku-mars/FAST_LIO` commit `7cc4175de6f8ba2edf34bab02a42195b141027e9`，许可证在 `vendor/FAST_LIO/LICENSE`；本地 CMake 仅为 ROS 消息依赖和低内存编译修改。
- ikd-Tree 子模块: commit `e2e3f4e9d3b95a9e66b1ba83dc98d4a05ed8a3c4`。
- Livox ROS driver: commit `3d240d5666129e1a3052e78ee8487a04b08fdda3`，许可证在 `vendor/livox_ros_driver/LICENSE.txt`。
- Livox SDK: commit `9306596a2bf15c1343bc023b497465ed0a32909d`，许可证在其 `LICENSE.txt`。
- 编译：`source /opt/ros/noetic/setup.bash && cd mujoco_closed_loop/ros_ws && catkin_make -j1`。机器 7.4 GiB RAM，`-j2` + 上游 Debug/O3 曾导致编译器被系统终止。
- 纯 SLAM 联调：`bash mujoco_closed_loop/run_fast_lio_pilot.sh mujoco_closed_loop/runs/fast_lio_pilot 15`。
- 在线三缝任务入口：`bash mujoco_closed_loop/run.sh mission-slam 180 --three-gaps --dashboard-video`。此入口仍可能因地图前沿不足而安全暂停，不能作为完整闭环验收。

当前控制仍为 MuJoCo 内部旋翼控制器；没有 PX4 SITL 或实体飞控连接。
