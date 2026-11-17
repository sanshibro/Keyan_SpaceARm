# Qingyu 模型复用范围

当前默认柔性臂直接使用归档 SDK 配置的 `arc_length=0.28m`、`visual_base_offset_z=0.0704m`。模型、视觉 PCC 和新规划适配共同通过 `qingyu_model.py` 读取原始 YAML，并校验归档 SHA256；旧 V2 的 0.22m/0.10m 配置保持独立。

|内容|实际来源与处理|
|---|---|
|基座外观|原始 A01Base.STL，毫米转米；派生 OBJ/LOD，保留原文件及转换误差记录|
|夹爪外观|原始 gripper.STEP.stl，按连通 CAD 壳拆成壳体和三指；姿态参考 SDK 配置的 Rz(π)Rx(π/2)，新安装平移单独记录|
|配置和 Xacro|原样归档于 vendor/qingyu_continuum_arm，来源为 /opt/qingyu_api_ros1/share/continuum_arm；文件哈希见 PROVENANCE.json|
|长度与坐标检查|按 SDK 新尺寸完成固定基座形态、端点、相机刚性外参及接触阶段检查；失败记录一并保存|

SDK Xacro 仅提供基座/夹爪外观和固定连接，没有可直接运行的连续体动力学、质量惯量或夹爪机构关节。因此，8 个数值弯曲单元、等效位置伺服、质量/刚度/摩擦、三指径向滑动及碰撞包络属于仿真假设。上层仍只有单段 θ/φ 两个形态变量，没有增加基座转台或独立可控多段。

基座 0.30kg、柔性体 0.25kg、工具 0.20kg 沿用明确标注的开发预算，不能称为 SDK 给出的实测参数。120° 极限附近，根部第二胶囊与相接基座有约 30–40 微米端帽重叠，按同一连续体接合边界处理；外部接触检测保留。四个弯曲方向的物理积分回归确认无该数值接触。

已核对安装包的 `Client::moveJoints/movePose` 和六项 θ/φ 字段；字段数量不代表实物有三段。现有 GUI 的 `continuum_arm/control_angles` 明确发布两个度数，`continuum_arm/grasp` 使用 Bool 开合。SDK Client 头文件没有夹爪方法。实机零点、符号、未用组策略和反馈来源仍需独立验证；当前 `SimArmBridge` 是仿真驱动转换器，尚非完整 ROS/实机桥。当前所有测试均未启动真实串口或 SDK 硬件连接。

原安装包 package.xml 声明 MIT，安装分发的上游提交未知，按实际来源记录。所有原始 CAD 保留，派生资产与实验代码有独立来源/版本快照。
