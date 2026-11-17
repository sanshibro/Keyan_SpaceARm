# MuJoCo 开发后端（未完成最终闭环）

实施与验收要求见 [PLAN.md](PLAN.md)，当前在线 SLAM 进度见 [SLAM_STATUS.md](SLAM_STATUS.md)。在线仿真现直接复用此前确认的 Stage 1 SUPER/Qingyu/D435 整机装配和短末端相机支架，配套碰撞、自遮挡及质心计算已同步。已接入 FAST-LIO 和异步 V2.1 航段规划。三缝模式已连续完成多个在线航段；完整穿越、抓取、带载退出与整体验收仍未通过。飞行控制目前是 MuJoCo 内部控制器，未接 PX4 SITL。

当前可复现入口（在仓库根目录执行）：

```bash
bash mujoco_closed_loop/run.sh mission-slam 180 --three-gaps --dashboard-video
bash mujoco_closed_loop/run.sh mission-slam 60 --dashboard-video  # 单门开发回归
```

输出保存在 `mujoco_closed_loop/runs/时间_slam_mission_development/`。`mission_view.mp4` 以可跟随的主仿真视角为主，右侧为控制实际消费的机身/末端 RGB 与在线地图；`online_dashboard.mp4` 保留多视角、深度和时间信息。窗口模式追加 `--gui`。两段录像均为在线物理运行的观察画面，不是离线轨迹拼接。

依赖隔离在根目录 `.mujoco_deps`，Python 3.8：

```bash
python3 -m pip install --target .mujoco_deps --index-url https://pypi.tuna.tsinghua.edu.cn/simple mujoco==3.2.3 scipy==1.10.1
./mujoco_closed_loop/run_scan.sh
python3 mujoco_closed_loop/check_frames.py
```

输出在 `runs/时间/`，包含 MJCF、16组扫描CSV和manifest。物理积分1kHz，无逐帧覆盖qpos；真值仅供阶段开发评估。扫描只检查端点，不能作为完整形态/夹爪/自由飞行验收。内部关节位置执行器是等效腱驱动假设；未标定刚度/摩擦。`SimArmBridge`只实现模拟指令转换，尚非ROS SDK桥，也不能连真实设备。

继续开发重点是传感器地图的自体遮挡/覆盖、V2.1接管和完整导航抓取串联。

当前默认尺寸已按用户要求切换为Qingyu配置：柔性长0.28m、基座0.0704m。基座外观复用SDK STL的确定性简化OBJ；原始文件和来源哈希保留在vendor。阶段扫描结果见 [STATUS.md](STATUS.md)。

`qingyu_model.py`校验归档SDK配置的SHA256并直接读取尺寸，模型、视觉PCC和新规划适配共用此来源。SDK Xacro只提供外观及固定连接，离散柔性动力学、质量、接触关节属于明确标记的仿真假设。夹爪外观来自SDK原始CAD拆分，径向滑动是运动近似。

扩展开发检查：

```bash
PYTHONPATH=.mujoco_deps python3 mujoco_closed_loop/shape_audit.py
PYTHONPATH=.mujoco_deps python3 mujoco_closed_loop/hover.py --mode both
MUJOCO_GL=egl PYTHONPATH=.mujoco_deps python3 mujoco_closed_loop/preview.py
```

悬停使用真值反馈和真值组合质心，仅为物理开发检查。全臂形状检查包含32组方向与幅度；所有结果保存在独立run目录，失败亦保留。

Qingyu SDK三指夹爪已拆分复用，可运行独立接触夹持台架：

```bash
PYTHONPATH=.mujoco_deps python3 mujoco_closed_loop/grasp_bench.py
MUJOCO_GL=egl PYTHONPATH=.mujoco_deps python3 mujoco_closed_loop/record_bench.py mujoco_closed_loop/runs/20261104_171956_grasp_bench
```

录像为已保存动力学的离线回放，标注FIXTURE BENCH，不提供在线感知或实时性能证据。原CAD外观复用、运动关节及指腹接触包络的近似范围详见STATUS。

传感器模块开发审计：

```bash
MUJOCO_GL=egl PYTHONPATH=.mujoco_deps python3 mujoco_closed_loop/perception_audit.py
```

已实现渲染RGB-D、ORB深度里程计、机体运动补偿形态、受控红色目标几何拟合，以及可见面雷达经估计位姿配准后复用V2体素地图。独立感知台架门槛已通过；普通纹理里程计的长期漂移仍未解决。抓取成功开发运行使用已测绘ArUco定位，不能当作无标记未知环境验收。

异步传感器反馈自由飞行开发检查（约30秒）：

```bash
MUJOCO_GL=egl PYTHONPATH=.mujoco_deps python3 mujoco_closed_loop/sensor_flight.py --seconds 30
```

RGB-D生成/估计独立进程，主进程持续积分及IMU控制；已知初始位姿，后续无真值反馈。默认富纹理受控环境，仅验证小幅平移/臂动作/保持及短时视觉遮挡，不是完整导航抓取演示。最近30秒测试结果与失败历史见STATUS。

原规划器适配检查：

```bash
PYTHONPATH=.mujoco_deps python3 mujoco_closed_loop/planner_audit.py --map mujoco_closed_loop/runs/20261104_173056_perception_audit/sensor_map.npz
PYTHONPATH=.mujoco_deps python3 mujoco_closed_loop/map_coverage_audit.py mujoco_closed_loop/runs/20261104_173056_perception_audit/sensor_map.npz
PYTHONPATH=.mujoco_deps python3 mujoco_closed_loop/async_planner_audit.py xtdrone_plan/experiments/method_comparison/v21_e3_final_450_commite62b6f2/multi_passage_medium_s011__1d65567df47b/audited_frozen_map.npz
```

当前在线地图覆盖不足会返回保持。最后一条仅是历史地图/固定基座异步基础设施回归，不是完整三墙闭环执行。各测试范围见STATUS。

下方观测开发检查：

```bash
MUJOCO_GL=egl PYTHONPATH=.mujoco_deps python3 mujoco_closed_loop/arm_survey_audit.py
PYTHONPATH=.mujoco_deps python3 mujoco_closed_loop/check_sensor_occlusion.py
```

腕部扫描仍未补齐全身地图。当前建图使用估计形态驱动的独立CAD深度投影，不消费外部场景标签；早期标签掩码仅作为开发对照。相机配准位置含PCC先验。最终传感器闭环不能据此宣称通过。30秒飞行结果见STATUS的210409记录。

视觉抓取开发入口（独立于导航规划）：

```bash
MUJOCO_GL=egl PYTHONPATH=.mujoco_deps python3 mujoco_closed_loop/visual_grasp_development.py
```

运行自动保存动力学、估计、真值评估、原始传感器样本及源代码快照。232227运行通过接近/接触/携物阶段，完整任务标记仍为false。生成视频的`record_mission.py`是记录轨迹的离线回放，不是在线GUI或在线传感器录像。

统一开发入口与在线面板：

```bash
./mujoco_closed_loop/run.sh visual-grasp --gui
./mujoco_closed_loop/run.sh visual-grasp --dashboard-video
./mujoco_closed_loop/run.sh survey --arm-sweep
./mujoco_closed_loop/run.sh navigation PATH_TO_SENSOR_MAP.npz
```

在线面板在独立进程渲染，观察视角使用实时物理快照，RGB-D 面板使用控制实际消费的帧，分别标注物理/传感器时间和延迟。显示不反向影响控制。旧单门、固定台架与纯抓取入口的历史范围见 STATUS；三缝在线闭环以 SLAM_STATUS 和本轮运行 manifest 为准。
