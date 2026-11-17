# ORB theta 与应变片同步采集训练流程

本文档用于采集青宇机械臂弯曲时的两类数据：

- ORB-SLAM3 计算得到的弯曲角 `theta`
- Elastreme 六通道采集盒的应变片数据

目标是生成同步数据集，并训练“应变数据 -> 弯曲角 theta”的预测模型。

## 1. 启动 ORB-SLAM3

新开终端 1：

```bash
source /opt/ros/noetic/setup.bash
source /home/uav/桌面/Test/catkin_ws/devel/setup.bash

roslaunch orb_slam3_ros d435_stereo.launch viewer:=true rviz:=true
```

如果相机已经由其他 launch 启动，改用：

```bash
roslaunch orb_slam3_ros d435_stereo.launch start_camera:=false viewer:=true rviz:=true
```

检查 ORB 输出：

```bash
rostopic echo /orb_slam3/tracking_state
rostopic echo /orb_slam3/odometry
```

`/orb_slam3/tracking_state = 2` 表示正常跟踪。

## 2. 启动 ORB 角度计算

新开终端 2。

启动前让机械臂保持直线零位不动，因为脚本会把启动时的 ORB 姿态作为零位。

```bash
source /opt/ros/noetic/setup.bash
source /home/uav/桌面/Test/catkin_ws/devel/setup.bash

python3 /home/uav/桌面/Test/orb_angle_calculator.py \
  _bend_axis:=z \
  _theta_deadband_deg:=0.2 \
  _phi_sign:=-1 \
  _phi_offset_deg:=270 \
  _phi_range:=signed
```

这个脚本会发布：

```text
/continuum_angles
/continuum_angles_valid
```

其中：

```text
/continuum_angles = [theta, fai]
theta: 弯曲角，单位 deg
fai: 已转换到青宇坐标系的弯曲方向角，单位 deg，范围 -180 至 180
```

检查角度输出：

```bash
rostopic echo /continuum_angles
```

可选：打开角度曲线窗口：

```bash
rosrun orb_slam3_ros angle_plotter.py
```

## 3. 启动青宇底层控制

新开终端 3：

```bash
source /opt/ros/noetic/setup.bash
source /opt/qingyu_api_ros1/setup.bash

roslaunch continuum_arm continuum_arm_visualize.launch use_mcu_packet_sender:=true
```

确认控制话题存在：

```bash
rostopic info /continuum_arm/control_angles
```

## 4. 启动同步采集

新开终端 4：

```bash
cd /home/uav/桌面/Test
source /opt/ros/noetic/setup.bash
source /home/uav/桌面/Test/catkin_ws/devel/setup.bash

./collect_orb_strain_dataset.py --port /dev/ttyUSB0 --channel 1 --latest-only --plot-hz 5
```

采集时会实时显示两条曲线：

- 上图：应变片通道 1
- 下图：ORB `theta`

如果需要尽量减少采集开销，不显示实时图：

```bash
./collect_orb_strain_dataset.py --port /dev/ttyUSB0 --channel 1 --latest-only --fast
```

采集结果会保存到：

```text
orb_strain_runs/YYYYMMDD_HHMMSS/orb_strain_sync.csv
orb_strain_runs/YYYYMMDD_HHMMSS/orb_strain_sync.png
orb_strain_runs/YYYYMMDD_HHMMSS/meta.txt
```

当前采集盒原始串口实测约为 `7.4 Hz`：

```bash
./measure_elastreme_serial_rate.py --port /dev/ttyUSB0 --duration 20
```

## 5. 启动 theta 扫动

新开终端 5：

```bash
cd /home/uav/桌面/Test
source /opt/ros/noetic/setup.bash
source /opt/qingyu_api_ros1/setup.bash

./qingyu_theta_smooth_sweep.py --period 10 --wait-enter
```

常用可直接复制命令：

```bash
cd /home/uav/桌面/Test
source /opt/ros/noetic/setup.bash
source /opt/qingyu_api_ros1/setup.bash
./qingyu_theta_smooth_sweep.py --period 10 --wait-enter --cycles 5 --zero-on-exit
```

如果想一直来回移动，直到按 `Ctrl+C`：

```bash
cd /home/uav/桌面/Test
source /opt/ros/noetic/setup.bash
source /opt/qingyu_api_ros1/setup.bash
./qingyu_theta_smooth_sweep.py --period 10 --wait-enter --zero-on-exit
```

默认参数：

```text
theta: 0 -> 70 -> 0 循环
fai: 0 固定
profile: sine
period: 8 秒一个完整来回
hz: 100 Hz 发布
```

建议训练采集时使用较慢周期，例如：

```bash
./qingyu_theta_smooth_sweep.py --period 10 --wait-enter --cycles 5 --zero-on-exit
```

操作顺序：

1. 先启动同步采集。
2. 确认采集窗口和终端都在更新。
3. 再到 theta 扫动终端按 Enter。
4. 扫动结束或数据足够后，停止同步采集。

## 6. 估计应变与 theta 时间偏移

采集完成后可以估计同步偏移：

```bash
cd /home/uav/桌面/Test

./estimate_strain_theta_lag.py orb_strain_runs/YYYYMMDD_HHMMSS --channel 1
```

如果输出：

```text
best_lag=0.000s
```

说明应变数据和 ORB theta 基本同步。

## 7. 训练应变到 theta 的模型

采集完成后训练：

```bash
cd /home/uav/桌面/Test

./train_strain_theta_model.py orb_strain_runs/YYYYMMDD_HHMMSS --channel 1 --degree 2
```

输出目录：

```text
orb_strain_runs/YYYYMMDD_HHMMSS/strain_theta_model/
```

包含：

```text
strain_theta_model.json
predictions.csv
fit.png
```

`strain_theta_model.json` 中保存了模型系数。

如果已经采集了多组数据，推荐用多组数据一起训练。这个脚本会对每一组数据单独估计零位 baseline，再把应变转换成相对零位的变化量进行训练：

```bash
cd /home/uav/桌面/Test
./train_multi_strain_theta_model.py \
  orb_strain_runs/20260614_143810 \
  orb_strain_runs/20260614_162702 \
  --channel 1 \
  --degree 3 \
  --smooth-window 3 \
  --output-dir orb_strain_runs/strain_theta_model_multi
```

输出模型：

```text
orb_strain_runs/strain_theta_model_multi/strain_theta_model.json
```

后续实时预测时直接把模型路径替换成这个多组模型路径即可。

## 8. 验证新采集数据

推荐直接复制一行命令，避免多行反斜杠后面误加空格。

用 `20260614_143810` 训练好的模型验证新数据 `20260614_144933`：

```bash
cd /home/uav/桌面/Test
./validate_strain_theta_model.py orb_strain_runs/20260614_143810/strain_theta_model/strain_theta_model.json orb_strain_runs/20260614_144933 --auto-baseline --baseline-seconds 20
```

验证任意新采集目录时，把最后一个目录替换成新的 run 目录：

```bash
cd /home/uav/桌面/Test
./validate_strain_theta_model.py orb_strain_runs/20260614_143810/strain_theta_model/strain_theta_model.json orb_strain_runs/YYYYMMDD_HHMMSS --auto-baseline --baseline-seconds 20
```

输出目录：

```text
orb_strain_runs/YYYYMMDD_HHMMSS/strain_theta_validation/
```

包含：

```text
validation_predictions.csv
validation.png
validation_report.json
```

## 9. 实时预测并对比 ORB theta

在 ORB-SLAM3、角度计算脚本、采集盒都已经正常运行后，可以实时调用训练好的模型，用应变片预测 theta，并和 ORB theta 对比。

启动前让机械臂保持直线零位，脚本会用前 20 秒自动估计当前应变零位 baseline。

推荐先使用稳定版模型：

```bash
cd /home/uav/桌面/Test
source /opt/ros/noetic/setup.bash
source /home/uav/桌面/Test/catkin_ws/devel/setup.bash
./live_strain_theta_predict_compare.py orb_strain_runs/20260614_162702/strain_theta_model_stable/strain_theta_model.json --port /dev/ttyUSB0 --auto-baseline --baseline-seconds 20 --auto-theta-bias --plot-hz 5 --output-smooth-window 5
```

如果发现实时预测有明显滞后，先把输出平滑关小。下面这个命令延迟更低：

```bash
cd /home/uav/桌面/Test
source /opt/ros/noetic/setup.bash
source /home/uav/桌面/Test/catkin_ws/devel/setup.bash
./live_strain_theta_predict_compare.py orb_strain_runs/20260614_162702/strain_theta_model_stable/strain_theta_model.json --port /dev/ttyUSB0 --auto-baseline --baseline-seconds 20 --auto-theta-bias --plot-hz 5 --output-smooth-window 1
```

如果最高角度段明显偏低，可以先加一个很小的在线校正：

```bash
cd /home/uav/桌面/Test
source /opt/ros/noetic/setup.bash
source /home/uav/桌面/Test/catkin_ws/devel/setup.bash
./live_strain_theta_predict_compare.py orb_strain_runs/20260614_162702/strain_theta_model_stable/strain_theta_model.json --port /dev/ttyUSB0 --auto-baseline --baseline-seconds 20 --auto-theta-bias --plot-hz 5 --output-smooth-window 5 --pred-scale 1.015 --pred-offset 1.2
```

如果只想先确认“应变曲线”和“ORB theta 曲线”是否同步，不调用模型，复制下面两个命令。

终端 A，启动同步采集：

```bash
cd /home/uav/桌面/Test
source /opt/ros/noetic/setup.bash
source /home/uav/桌面/Test/catkin_ws/devel/setup.bash
./collect_orb_strain_dataset.py --port /dev/ttyUSB0 --channel 1 --latest-only --plot-hz 5
```

终端 B，启动一个很慢的单周期来回运动：

```bash
cd /home/uav/桌面/Test
source /opt/ros/noetic/setup.bash
source /opt/qingyu_api_ros1/setup.bash
./qingyu_theta_smooth_sweep.py --period 20 --wait-enter --cycles 1 --zero-on-exit
```

操作顺序：

1. 先运行终端 A 的同步采集。
2. 再运行终端 B 的来回运动命令。
3. 终端 B 出现等待提示后，按 Enter 开始运动。
4. 观察同步采集窗口中应变曲线和 ORB theta 是否同时升降。

终端提示 baseline 锁定后再开始弯曲：

```text
Baseline locked: xxx.xxx
```

实时窗口会显示：

- ORB theta
- 应变模型预测 theta
- 预测误差 `pred - ORB`

实时预测脚本和同步采集脚本都应该处理串口缓冲区里的最新完整帧，避免串口数据积压导致应变数据落后 ORB。没有采集盒硬件时间戳时，不建议用 `--keep-all-frames` 做同步训练数据，因为旧应变帧会被配到当前 ORB theta。

结果保存到：

```text
live_strain_theta_runs/YYYYMMDD_HHMMSS/live_strain_theta_compare.csv
live_strain_theta_runs/YYYYMMDD_HHMMSS/live_strain_theta_compare.png
live_strain_theta_runs/YYYYMMDD_HHMMSS/meta.txt
```

如果不需要实时窗口，只保存数据：

```bash
cd /home/uav/桌面/Test
source /opt/ros/noetic/setup.bash
source /home/uav/桌面/Test/catkin_ws/devel/setup.bash
./live_strain_theta_predict_compare.py orb_strain_runs/20260614_143810/strain_theta_model/strain_theta_model.json --port /dev/ttyUSB0 --auto-baseline --baseline-seconds 20 --auto-theta-bias --no-live
```

## 10. 推荐实验方式

为了训练效果更好：

1. 每次采集前让机械臂保持直线零位。
2. ORB 角度计算脚本启动后再开始弯曲。
3. 使用较慢扫动周期，例如 `--period 10` 或 `--period 15`。
4. 采集多个周期，覆盖 `0-70 deg` 全范围。
5. 避免快速回弹和强烈振动。
6. 如果发现应变回程和去程不一致，后续模型需要加入历史窗口或变化率特征。

## 11. 常见检查命令

检查 ORB：

```bash
rostopic echo /orb_slam3/tracking_state
rostopic echo /orb_slam3/odometry
```

检查角度：

```bash
rostopic echo /continuum_angles
```

检查采集盒：

```bash
ls -l /dev/ttyUSB0
./measure_elastreme_serial_rate.py --port /dev/ttyUSB0 --duration 10
```

检查青宇控制话题：

```bash
rostopic info /continuum_arm/control_angles
```
