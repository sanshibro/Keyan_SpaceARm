# ORB-SLAM3 + D435

## 启动

新终端执行：

```bash
source /opt/ros/noetic/setup.bash
source /home/uav/桌面/Test/catkin_ws/devel/setup.bash
roslaunch orb_slam3_ros d435_stereo.launch viewer:=true rviz:=true
```

`viewer:=true` 打开 ORB-SLAM3 自带地图窗口，`rviz:=true` 打开 ROS 轨迹窗口。

如果相机已经由其他 launch 启动：

```bash
roslaunch orb_slam3_ros d435_stereo.launch start_camera:=false viewer:=true rviz:=true
```

不要同时运行 VINS-Fusion 和此 ORB-SLAM3 节点进行同一组实验。

## 输出话题

```text
/orb_slam3/odometry
/orb_slam3/pose
/orb_slam3/path
/orb_slam3/tracking_state
```

跟踪状态：

```text
0: 尚未收到图像
1: 尚未初始化
2: 正常跟踪
3: 最近丢失
4: 已丢失
```

检查状态和位姿：

```bash
rostopic echo /orb_slam3/tracking_state
rostopic echo /orb_slam3/odometry
```

## 查看相对角度

相机保持在零位姿，启动脚本后再弯曲：

```bash
python3 /home/uav/桌面/Test/orb_angle_calculator.py \
  _bend_axis:=z \
  _theta_deadband_deg:=0.2 \
  _phi_sign:=-1 \
  _phi_offset_deg:=270 \
  _phi_range:=signed
```

当前 `orb_angle_calculator.py` 默认不启用跟踪门控，会持续输出角度。
`/continuum_angles_valid=1` 表示当前角度有效。需要严格门控时加：

```bash
_use_tracking_gate:=true
```

启用严格门控后，只有 ORB 跟踪状态为 `2` 时才认为角度有效。连续出现
`Fail to track local map!` 时，角度有效标志会变为 0。

## ORB 与动补角度对比

如果动补话题 `/vrpn_client_node/Rigid2/pose` 已经有数据，可以一条命令同时启动：

- ORB 相对角度计算：输出 `/continuum_angles`
- 动补相对角度计算：输出 `/vrpn_angles`
- ORB/动补对比曲线窗口

```bash
source /opt/ros/noetic/setup.bash
source /home/uav/桌面/Test/catkin_ws/devel/setup.bash
roslaunch orb_slam3_ros angle_compare.launch
```

也可以直接用一条命令：

```bash
/home/uav/桌面/Test/run_orb_vrpn_angle_compare.sh
```

每次运行会自动新建一个实验文件夹：

```text
/home/uav/桌面/Test/angle_runs/YYYYMMDD_HHMMSS/
```

其中会保存：

```text
angles.csv            ORB/VRPN 的 theta、phi 数值记录
angles.bag            相关 ROS 话题数据包
angles_compare.png    退出时保存的曲线图
```

注意：`angles_compare.png` 会在你按 `Ctrl+C` 正常退出时保存。

默认角度换算使用当前已经校过的 `phi` 方向：

```text
bend_axis=z
theta_deadband_deg=0.2
phi_sign=-1
phi_offset_deg=270
phi_range=signed
```

动补角度默认只做启动零位姿态校准，不做滤波：

```text
vrpn_zero_samples=80
```

启动后先让柔性臂保持直线零位 3 到 5 秒，等终端显示：

```text
VRPN zero pose calibrated
```

再开始弯曲。当前 `/vrpn_angles` 是基于 VRPN 原始姿态计算出的相对角度，曲线不做平滑处理。

对比曲线窗口为了看起来更平滑，默认只对 VRPN 显示曲线做 5 点移动平均；原始 `/vrpn_angles` 话题不变。

如果想让图像更平滑：

```bash
/home/uav/桌面/Test/run_orb_vrpn_angle_compare.sh vrpn_smooth_window:=9
```

如果想完全关闭显示平滑：

```bash
/home/uav/桌面/Test/run_orb_vrpn_angle_compare.sh vrpn_smooth_window:=1
```

如果只想画图和保存 CSV，不想录 rosbag：

```bash
/home/uav/桌面/Test/run_orb_vrpn_angle_compare.sh record_bag:=false
```

## 只记录 ORB 的 fai

如果只把 ORB-SLAM3 的 `fai/phi` 当作参考值，不使用动补，先保证 ORB-SLAM3 已经在运行，然后执行：

```bash
/home/uav/桌面/Test/run_orb_fai_record.sh
```

每次运行会自动新建：

```text
/home/uav/桌面/Test/orb_fai_runs/YYYYMMDD_HHMMSS/
```

保存内容：

```text
orb_fai.csv      fai/phi 数值记录
orb_fai.png      退出时保存的 fai 曲线
orb_fai.bag      /continuum_angles、/orb_slam3/odometry 等相关话题
```

如果只想保存 CSV 和图片，不录 bag：

```bash
/home/uav/桌面/Test/run_orb_fai_record.sh record_bag:=false
```

## 只对比 ORB 与动补的 fai

如果只关心 `fai/phi`，同时记录 ORB 和动补，不画 theta：

```bash
/home/uav/桌面/Test/run_fai_compare.sh
```

每次运行会自动保存到：

```text
/home/uav/桌面/Test/fai_runs/YYYYMMDD_HHMMSS/
```

保存内容：

```text
fai_compare.csv      ORB/VRPN 的 fai 数据
fai_compare.png      ORB/VRPN 的 fai 曲线
fai_compare.bag      相关 ROS 话题
```

如果动捕 fai 的轴不对，优先试这些命令。启动后保持零位，左右旋转，观察 VRPN fai 是否在负到正之间变化，并且趋势是否和 ORB 一致：

```bash
/home/uav/桌面/Test/run_fai_compare.sh vrpn_bend_axis:=x
/home/uav/桌面/Test/run_fai_compare.sh vrpn_bend_axis:=y
/home/uav/桌面/Test/run_fai_compare.sh vrpn_bend_axis:=z
/home/uav/桌面/Test/run_fai_compare.sh vrpn_bend_axis:=-x
/home/uav/桌面/Test/run_fai_compare.sh vrpn_bend_axis:=-y
/home/uav/桌面/Test/run_fai_compare.sh vrpn_bend_axis:=-z
```

如果轴对了但正负反了：

```bash
/home/uav/桌面/Test/run_fai_compare.sh vrpn_bend_axis:=y vrpn_fai_sign:=-1
```

如果轴和正负都对，但零点偏移不对：

```bash
/home/uav/桌面/Test/run_fai_compare.sh vrpn_bend_axis:=y vrpn_fai_offset_deg:=90
```

## 只录原始 ORB 和动捕数据

如果后续自己离线处理，不希望在线脚本做任何坐标转换或 fai 处理，直接运行：

```bash
/home/uav/桌面/Test/run_raw_orb_vrpn_record.sh
```

每次运行会保存到：

```text
/home/uav/桌面/Test/raw_runs/YYYYMMDD_HHMMSS/
```

默认 bag：

```text
raw_orb_vrpn.bag
```

包含：

```text
/continuum_angles
/continuum_angles_valid
/orb_slam3/odometry
/orb_slam3/pose
/orb_slam3/path
/orb_slam3/tracking_state
/vrpn_client_node/Rigid2/pose
```

如果还想同时录红外图像：

```bash
/home/uav/桌面/Test/run_raw_orb_vrpn_record.sh record_images:=true
```

会额外保存：

```text
raw_images.bag
```

如果要改动补话题：

```bash
roslaunch orb_slam3_ros angle_compare.launch vrpn_pose_topic:=/vrpn_client_node/Rigid2/pose
```

注意：这个 launch 只启动角度计算和绘图，不启动 ORB-SLAM3 和相机。ORB-SLAM3、RealSense、VRPN 需要先启动。

## 当前配置

- 图像：D435 rectified infra1 + infra2，640x480@30 Hz
- 内参：`fx=fy=382.9065, cx=316.7031, cy=239.5040`
- 双目基线：`0.0501235 m`
- ORB 特征数：1600

配置文件：

```text
/home/uav/桌面/Test/catkin_ws/src/orb_slam3_ros/config/D435_stereo.yaml
```

ORB-SLAM3 是纯双目视觉方案。快速旋转、运动模糊、重复纹理、强光或两目遮挡都可能导致状态变为 3/4；重新看到已建图区域后可能重定位。

默认会关闭 RealSense 红外投射器：

```text
disable_emitter:=true
```

原因是 D435/D435i 中间的红外投射器会把相机自身产生的散斑投到场景中。ORB-SLAM3 可能跟踪这些散斑而不是场景本身，导致姿态和角度估计异常。关闭投射器等价于只使用真实环境纹理。

## D435i

当前 D435i 序列号为 `142122070607`，已单独读取并写入其双目内参。

```bash
source /opt/ros/noetic/setup.bash
source /home/uav/桌面/Test/catkin_ws/devel/setup.bash
roslaunch orb_slam3_ros d435i_stereo.launch viewer:=true rviz:=true
```

D435i 配置文件：

```text
/home/uav/桌面/Test/catkin_ws/src/orb_slam3_ros/config/D435i_stereo.yaml
```

该启动方式使用 D435i 的双目红外图像，不融合 IMU。这样可以与之前 D435 的纯双目 ORB-SLAM3 结果直接比较。D435i 的 IMU 已验证约为 200 Hz，可另行用于 Stereo-Inertial 模式。

## 故障注入与角度曲线

启动 D435 故障版 ORB-SLAM3：

```bash
source /opt/ros/noetic/setup.bash
source /home/uav/桌面/Test/catkin_ws/devel/setup.bash
roslaunch orb_slam3_ros d435_fault_stereo.launch viewer:=true rviz:=true
```

启动 D435i 故障版：

```bash
source /opt/ros/noetic/setup.bash
source /home/uav/桌面/Test/catkin_ws/devel/setup.bash
roslaunch orb_slam3_ros d435i_fault_stereo.launch viewer:=true rviz:=true
```

故障版会让 ORB 订阅：

```text
/fault/infra1/image_rect_raw
/fault/infra2/image_rect_raw
```

相机原始图像仍在：

```text
/camera/infra1/image_rect_raw
/camera/infra2/image_rect_raw
```

启动角度计算：

```bash
python3 /home/uav/桌面/Test/orb_angle_calculator.py \
  _bend_axis:=z \
  _theta_deadband_deg:=0.2 \
  _phi_sign:=-1 \
  _phi_offset_deg:=270 \
  _phi_range:=signed
```

启动角度曲线：

```bash
rosrun orb_slam3_ros angle_plotter.py
```

运行中切换过曝：

```bash
rosparam set /infra_fault_injector/mode overexpose
```

恢复正常图像：

```bash
rosparam set /infra_fault_injector/mode normal
```

可选故障模式：

```bash
rosparam set /infra_fault_injector/mode white
rosparam set /infra_fault_injector/mode dark
rosparam set /infra_fault_injector/mode blur
rosparam set /infra_fault_injector/mode low_feature
rosparam set /infra_fault_injector/mode noise
rosparam set /infra_fault_injector/mode dropout
rosparam set /infra_fault_injector/mode left_overexpose
rosparam set /infra_fault_injector/mode right_overexpose
```

建议实验顺序：

```text
normal -> 直线零位标定 -> 弯曲一次确认角度正常 -> overexpose -> 观察 theta/phi 曲线和 tracking_state
```
