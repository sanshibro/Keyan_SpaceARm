# MuJoCo开发快照上传检查（2026-11-17）

验证在独立导出目录执行，复用原机器Python依赖；不是新机器冷启动验收。

|检查|结果|
|---|---|
|当前V2规划器核心测试|82/82 PASS，7.822秒|
|check_frames.py|PASS，机体运动补偿、角度恢复、过期时间戳拒绝|
|check_online_mapping.py|PASS，多次可见自由证据清除、遮挡后方未知保留|
|check_trajectory_identity.py|PASS，200个非网格时刻的位置/速度/加速度/jerk重建一致|
|check_tool_envelope.py|PASS，末端相机碰撞包络、末端索引、标量/批量一致|
|check_discretization.py|PASS，梯形离散最大端点误差约2.14mm，形态角误差约1.09度|
|check_rgbd_visibility.py|PASS，缺失深度、过近遮挡与明确超量程证据分开处理|
|三门aperture_qualification.py|PASS，在该脚本定义的高度及路径离散检查范围，直臂无物理无碰撞高度；110度弯臂示例0次碰撞，规划安全余量后残差约50mm|
|运行记录|最近180秒记录的passed=false原样保留，未改为成功|

首次直接调用核心测试因未设置规划器src搜索路径而导入失败；补齐PYTHONPATH后82项全部通过。测试未改动源算法。

归档针对移机做的非算法修改：增加依赖清单、工作空间相对链接初始化脚本、安装说明；ROS launch配置读取改为`$(dirname)/../config/...`，避免原工作空间绝对链接依赖。原工作目录未修改。

未重新运行完整三墙任务、未重新编译FAST-LIO、未重新进行论文450次实验；不将这些静态/单元检查冒充完整闭环验收。
