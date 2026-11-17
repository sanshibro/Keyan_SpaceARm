# Qingyu CLI 控制：theta=30，fai 从 -90 到 +90

## 1. 启动底层节点

新开一个终端运行：

```bash
source /opt/ros/noetic/setup.bash
source /opt/qingyu_api_ros1/setup.bash
roslaunch continuum_arm continuum_arm_visualize.launch use_mcu_packet_sender:=true
```

## 2. fai 从 -90 到 +90 单向变化

每个目标角度以 50Hz 连续发送 50 次，也就是每个点大约保持 1 秒。

新开另一个终端运行：

```bash
source /opt/ros/noetic/setup.bash
source /opt/qingyu_api_ros1/setup.bash

for fai in -90 -75 -60 -45 -30 -15 0 15 30 45 60 75 90; do
  rosrun qingyu_api_sdk qingyu_api_sdk_cli joint 30 $fai 0 0 0 0 1 50 50
done
```

## 3. fai 从 -90 到 +90 再回到 -90

```bash
source /opt/ros/noetic/setup.bash
source /opt/qingyu_api_ros1/setup.bash

for fai in -90 -75 -60 -45 -30 -15 0 15 30 45 60 75 90 75 60 45 30 15 0 -15 -30 -45 -60 -75 -90; do
  rosrun qingyu_api_sdk qingyu_api_sdk_cli joint 30 $fai 0 0 0 0 1 50 50
done
```

## 4. 回零

```bash
source /opt/ros/noetic/setup.bash
source /opt/qingyu_api_ros1/setup.bash
rosrun qingyu_api_sdk qingyu_api_sdk_cli joint 0 0 0 0 0 0 1
```

如果回零命令不自动退出，用这个版本：

```bash
source /opt/ros/noetic/setup.bash
source /opt/qingyu_api_ros1/setup.bash
rosrun qingyu_api_sdk qingyu_api_sdk_cli joint 0 0 0 0 0 0 1 50 100
```

## 5. fai 在 -90 和 +90 之间直接切换

不经过中间点，只在两个目标之间来回切换。

每个目标保持约 1 秒：

```bash
source /opt/ros/noetic/setup.bash
source /opt/qingyu_api_ros1/setup.bash

while true; do
  rosrun qingyu_api_sdk qingyu_api_sdk_cli joint 30 -90 0 0 0 0 1 50 50
  rosrun qingyu_api_sdk qingyu_api_sdk_cli joint 30 90 0 0 0 0 1 50 50
done
```

每个目标保持约 0.4 秒：

```bash
source /opt/ros/noetic/setup.bash
source /opt/qingyu_api_ros1/setup.bash

while true; do
  rosrun qingyu_api_sdk qingyu_api_sdk_cli joint 30 -90 0 0 0 0 1 50 20
  rosrun qingyu_api_sdk qingyu_api_sdk_cli joint 30 90 0 0 0 0 1 50 20
done
```

停止循环用 `Ctrl+C`。

## 6. fai 在 -90 和 +90 之间平滑连续旋转

这个版本不是 CLI 点到点命令，而是用 ROS topic 以 100Hz 连续发布：

```text
theta = 30 deg
fai   = -90 -> +90 -> -90 连续变化
```

先启动底层：

```bash
source /opt/ros/noetic/setup.bash
source /opt/qingyu_api_ros1/setup.bash
roslaunch continuum_arm continuum_arm_visualize.launch use_mcu_packet_sender:=true
```

新开终端运行连续控制：

```bash
source /opt/ros/noetic/setup.bash
source /opt/qingyu_api_ros1/setup.bash
python3 /home/uav/桌面/Test/qingyu_fai_smooth_sweep.py --theta 30 --fai-min -90 --fai-max 90 --period 4 --hz 100
```

如果希望启动后先保持 `theta=30, fai=0`，按 Enter 后再开始旋转，用这个：

```bash
source /opt/ros/noetic/setup.bash
source /opt/qingyu_api_ros1/setup.bash
python3 /home/uav/桌面/Test/qingyu_fai_smooth_sweep.py --theta 30 --pre-fai 0 --fai-min -90 --fai-max 90 --period 4 --hz 100 --wait-enter
```

含义：

```text
--theta 30      固定弯曲角 theta=30 deg
--fai-min -90   fai 最小值 -90 deg
--fai-max 90    fai 最大值 +90 deg
--period 4      4 秒完成一次 -90 -> +90 -> -90
--hz 100        100Hz 连续发布
--wait-enter    先保持 theta=30 fai=0，按 Enter 后开始
```

如果想更快旋转，把 `--period` 改小：

```bash
python3 /home/uav/桌面/Test/qingyu_fai_smooth_sweep.py --theta 30 --fai-min -90 --fai-max 90 --period 2 --hz 100
```

如果只运行 3 个周期：

```bash
python3 /home/uav/桌面/Test/qingyu_fai_smooth_sweep.py --theta 30 --fai-min -90 --fai-max 90 --period 4 --hz 100 --cycles 3
```

停止用 `Ctrl+C`。

停止时自动回零：

```bash
python3 /home/uav/桌面/Test/qingyu_fai_smooth_sweep.py --theta 30 --fai-min -90 --fai-max 90 --period 4 --hz 100 --zero-on-exit
```

## 7. 命令含义

```text
joint theta1 phi1 theta2 phi2 theta3 phi3 enable
```

完整格式：

```text
joint theta1 phi1 theta2 phi2 theta3 phi3 enable hz count
```

A01 当前主要使用：

```text
theta1 = 弯曲角度
phi1   = 弯曲方向角
enable = 1
```

这里用的：

```text
hz    = 50
count = 50
```

表示每个角度连续发送 50 次，频率 50Hz，持续约 1 秒。

## 8. 交互式输入数字控制

先启动底层：

```bash
source /opt/ros/noetic/setup.bash
source /opt/qingyu_api_ros1/setup.bash
roslaunch continuum_arm continuum_arm_visualize.launch use_mcu_packet_sender:=true
```

新开终端启动交互控制：

```bash
source /opt/ros/noetic/setup.bash
source /opt/qingyu_api_ros1/setup.bash
python3 /home/uav/桌面/Test/qingyu_interactive_control.py
```

进入脚本后可以输入：

```text
30 0      设置 theta=30, fai=0
30 -90    设置 theta=30, fai=-90
30 90     设置 theta=30, fai=90
-45       只改变 fai=-45，theta 保持当前值
z         回零 theta=0, fai=0
q         退出
```

启动时给默认初始值：

```bash
python3 /home/uav/桌面/Test/qingyu_interactive_control.py --theta 30 --fai 0
```
