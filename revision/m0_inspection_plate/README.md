# 单门闭环的工业隔舱场景重渲

M0 复用研究分支 `951c512` 的单门记录。所有以秒为单位的标记都相对于飞行释放时刻。`review/gate_review_v2.pdf` 为两页审阅，`review/gate_main_v2.png` 为高清合版。`review/gate_multiview_v2.mp4` 使用同一场景的侧视、局部剖视以及当趟实际记录的腕部 RGB。

## 场景坐标

ENU 门洞中心位于 x = 1.00 m，墙厚 0.10 m，内宽 0.76 m。下槛和上梁内表面分别位于 z = 1.695 m 和 2.385 m。法兰口中心 [1.98, 0.25, 1.71] m 与目标平面重合。法兰外法向取注册工具轴的反向，半径 35 mm 的投影圆标记位置容差。视觉遮挡对应腕部曝光时钟上的 28–32 s。

## 离线生成

将 `M0_GATE_ARCHIVE` 指向包含 `evidence`、`formal`、`records` 和 `review/evidence-index.csv` 的 G2 归档。`gate_portable_model.py`、`gatereview_visuals.py`、`gatereview_pdf.py` 和 `gate_meeting_archive.py` 位于 G2 源码路径。`M0_OUTPUT` 可以指定单独的输出目录。场景与照片级主视图使用同一组已注册的 SUPER、Qingyu 和夹爪 CAD。下列程序仅从归档状态读取刚体变换。

```sh
python source/m0_audit.py
python source/m0_pathtrace.py
python source/m0_plates.py
python source/m0_vector.py
python source/m0_video.py
python source/m0_review.py
```

`m0_pathtrace.py` 使用 Mitsuba 3.9、CUDA 与 OptiX，`m0_render.py` 使用 OpenGL 驱动的视频绘制。矢量导出使用 VTracer 与 CairoSVG 将三维明暗转为色面，曲线、尺寸、最近点及轴向叠加保留原生矢量。高清 PNG 和审阅页使用连续色调渲染。运行依赖还包括 NumPy、SciPy、MuJoCo、trimesh、Pillow、Matplotlib、PyMuPDF、ReportLab 和 imageio-ffmpeg。

## 查证图中数据

`qa/archive-fingerprints.json` 记录 60 趟原包的校验。`qa/scene-clearance.json` 给出每一项新增几何对 42 趟成功飞行的保守分离下界。论文过门净空来自各趟已归档的原始评分。`qa/hero-projection.json` 记录四个整机姿态的屏幕包络，`qa/inset-projection.json` 记录每个特写对应的实际时刻，`qa/video-verification.json` 记录原始腕部画面与发布视频的逐帧比对。

`source/m0_archive.py build` 用既有组会站点生成器制作本轮静态页面及离线汇报包。公开页面按提交 SHA 校验远端大附件。M0 交付状态为视觉审阅。
