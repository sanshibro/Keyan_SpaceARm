# MuJoCo 开发版归档与运行说明

分支：`mujoco-development`。来源：原工作仓库 `d5682f9` 加截至2026-11-10的未提交MuJoCo开发代码，以及其实际使用的V2规划器工作树。上传没有修改原工作仓库，也没有宣称新增完整闭环验收。

## 1. 包含内容

- `mujoco_closed_loop/`：最新模型、控制、雷达/IMU桥、FAST-LIO、双RGB-D、在线地图、异步V2.1规划、夹爪、目标检测、多视角显示、审计与开发计划。
- `mujoco_closed_loop/vendor/`：运行所需的SUPER/Qingyu/D435模型与转换资产，以及FAST-LIO、Livox、SUPER的源代码和原许可证。排除嵌套Git数据库、编译产物和上游演示媒体。
- `xtdrone_catkin_ws/src/uav_continuum_whole_body_v2/`：当前实际复用的规划器，含geometry/types/validator的末端工具包络扩展。这不是论文主分支的冻结算法副本；主分支保持不变。
- `mujoco_closed_loop/runs/20261110_134527_slam_mission_development/`：最近完整保存的180秒开发运行，含原始传感器帧、动力学、地图、规划及执行记录、源码快照、主视角和面板视频。
- `runs/20261107_fast_lio_online_map/`：独立FAST-LIO联调记录。
- `runs/20261109_195713_slam_mission_development/`、`runs/20261108_140204_mission_development/`：代表性运行的顶层日志、地图和视频；不含其体积较大的原始传感器帧与源码快照。
- `mujoco_closed_loop/evidence/run_summaries/`：历史各运行的顶层JSON结果，包含失败记录，不是所有9.2GB历史原始数据的完整镜像。
- `provenance/mujoco_publication_manifest.json`：来源文件SHA-256、上游提交及未提交修改、导出范围和排除项。

`datasets/mujoco_super_original_cad.tar.gz`保留82MB原始SUPER装配CAD，可在根目录执行以下命令恢复。常规运行使用已转换模型，无需恢复该大文件：

```bash
tar -xzf datasets/mujoco_super_original_cad.tar.gz
```

## 2. 当前真实进度

已接入MuJoCo物理积分、旋翼内部控制器、Qingyu单段弯曲近似、夹爪接触、机身/末端RGB-D、雷达/IMU FAST-LIO、传感器地图、V2.1异步规划与多视角显示。

最近180秒运行：0次碰撞、0次电机饱和、位置RMS误差约54.2mm、6条在线入口航段；最终因时间限制结束，完整导航和抓取未完成，`passed=false`。这不是“完整自主抓取成功”证据。最新代码尚未接PX4 SITL或实体飞控。关节位置执行器、低漂移IMU、已知初始姿态和模拟应变回退均有建模假设；详情见原始README、PLAN、SLAM_STATUS和运行manifest。

## 3. 获取与基本依赖

```bash
git clone --branch mujoco-development git@github.com:sanshibro/Keyan_SpaceARm.git
cd Keyan_SpaceARm
```

已知开发环境为Ubuntu 20.04、Python 3.8、ROS Noetic，MuJoCo 3.2.3、NumPy 1.24.4、SciPy 1.10.1。Python依赖不上传安装目录。示例安装：

```bash
python3 -m pip install --target .mujoco_deps -r mujoco_closed_loop/requirements-publication.txt
# 原环境使用系统OpenCV 4.2.0（含aruco），不要同时混装多个opencv包。
sudo apt-get install python3-opencv python3-yaml python3-matplotlib ripgrep
```

无显示器运行需要可用的EGL/OpenGL驱动。`--gui`另需桌面显示环境。CAD重新转换才需要gmsh，现成运行资产不需要它。

## 4. ROS工作空间

原工作空间中的绝对软链接和build/devel未上传；用以下脚本创建可迁移的相对链接：

```bash
bash mujoco_closed_loop/setup_ros_workspace.sh
```

首次构建需先安装ROS Noetic、PCL/Eigen和Livox SDK。Livox SDK源代码已保留，可单独安装：

```bash
cmake -S mujoco_closed_loop/vendor/livox_ros_driver/livox_ros_driver/Livox-SDK -B /tmp/keyan_livox_build
cmake --build /tmp/keyan_livox_build -j1
sudo cmake --install /tmp/keyan_livox_build
source /opt/ros/noetic/setup.bash
rosdep install --from-paths mujoco_closed_loop/ros_ws/src --ignore-src -r -y
cd mujoco_closed_loop/ros_ws
catkin_make -j1
cd ../..
```

源码保留本地FAST-LIO低内存编译和消息依赖修订。不要复制旧devel目录，它包含原机器的绝对路径。

## 5. 查看和运行

先直接打开最近运行的`mission_view.mp4`和`online_dashboard.mp4`，或查看`model_preview.jpg`。

```bash
# 静态整机/场景预览（支持的参数见--help）
bash mujoco_closed_loop/run.sh stage1 --help
# 独立激光/IMU联调
bash mujoco_closed_loop/run_fast_lio_pilot.sh mujoco_closed_loop/runs/new_lio_pilot 15
# 三墙在线开发运行，不保证完成任务
bash mujoco_closed_loop/run.sh mission-slam 180 --three-gaps --dashboard-video
# 桌面显示可追加 --gui
```

`run.sh`以项目根目录定位依赖；不需要修改原开发机绝对路径。历史保存的MJCF/XML、manifest和日志保留原路径以保持证据，若离线回放报找不到mesh，应重定位运行副本，不能据此改写归档原件。

## 6. 上传验证的边界

本次验证记录见`mujoco_closed_loop/PUBLICATION_VALIDATION_CN.md`。上传检查不等于从新机器冷启动完成在线任务，也不等于重新进行论文450次试验。当前分支的SHA256SUMS对应该分支内容；原论文分支的文件与哈希不受影响。
