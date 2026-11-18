# 组会归档：一轮结束，一轮保存

固定栏目：`/archive/index.html`。每轮均有独立的 `/archive/<id>/index.html`，媒体和资料都在该轮的 `media/` 下。网页中的档案不轮询 GitHub，不读取 NUC，也不需要登录；离线 ZIP 解压后直接打开 `index.html`。

## 下轮交付顺序

1. 暂停仿真，把这次实际给评审的原图、曲线、MP4、数据和论文页放进一个单独的输入目录。只选可以公开的文件。让 MP4 使用 H.264；竖排的纸页用 PDF，曲线保留 PNG 和可放大的 PDF。没有视频就写清楚，不挪用其他轮次的录像。
2. 复制 `rounds/c2.json`，填写新的 `id`、北京时间、源数据的完整 Git 提交、观察结论及边界。原始数据和归档时的补绘分别注明。清单里的 `file` 是输入目录中的文件名；`cover` 必须指向该轮一张列出的图；视频可另列一张 `poster` 封面。不要复制一份旧清单后继续使用旧 ID。
3. 在网站源码根目录执行（Python 3.8+，标准库，无需启动仿真）：

   ```sh
   python3 tools/meeting_archive.py add --site dist --spec tools/rounds/你的新清单.json --source /path/to/this-round-approved-media
   python3 tools/meeting_archive.py verify --site dist
   python3 tools/verify_archive_links.py --site dist
   ```

   工具同时生成独立页面、可下载的离线资料包、各文件的 SHA-256 和归档总目录，并更新监视器首页的「最近一轮」。重复添加完全相同的材料是无操作；改过媒体、图注或结论时必须使用新 ID（例如 `...-r2`），并在新轮的说明中引用旧轮。同一个 ID 不会被覆盖。`separate: true` 的大附件在网页上可单独下载，不进入本轮简包。每个托管文件及简包限 24 MiB。
4. 将新轮材料以及清单备份到研究分支 `feat/paper-mechanism-rolling-20261117` 的 `revision/meeting_archive/`，提交阶段说明并 push；在 Sites 源码副本里执行同一套校验后发布同一站点，等待部署成功。不要改原工程或已有备份。阶段结束的交付条件是：点开总索引，图片可以放大，曲线可下载，视频可播放，PDF 可打开，旧轮仍可访问。向评审提供本轮固定链接后暂停。

在研究仓库中，`revision/meeting_archive/` 本身就是静态站根目录（用 `--site .`）；打开 `archive/index.html` 即可看已归档材料。`meeting_archive.py index --site dist` 可以在重建首页后恢复入口，不改变任何历史轮次。支持自定义首页标记 `<!-- meeting-archive-slot -->`。这套流程只在阶段交付时运行，无后台轮询或定时仿真。

## 此次补录的依据

| 轮次 | 数据与版本 | 归档处理 |
| --- | --- | --- |
| A | 5be3079 的三门记录，最终图片在 688e16a；论文图页取当轮本地预览 | 从 964 帧 `replay.json` 生成 4 倍速几何回放，未将其标作传感器原始录像 |
| B | 1b99084 的 16 版原生 CAD 与固定机身试验 | 保留原图、曲线、PDF 和数据；当轮未归档视频 |
| C1 | 1327b59 的 SUPER 初版（修正前） | 将原 MP4 转码为 H.264 并从当轮 CSV 补绘响应图；不重跑仿真 |
| C2 | 2dbbf2a 的修正积分；99a4c60 的原生相机装配对照 | 两路相机、SDK 夹爪、固定机身与悬停视频、曲线、CSV 及论文第 13–15 页 |
