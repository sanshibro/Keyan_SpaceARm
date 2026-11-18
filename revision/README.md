# Paper and simulation revision

Branch: `feat/paper-mechanism-rolling-20261117`. See `IMPLEMENTATION_PLAN.txt` for the current detailed plan and checkpoint criteria, `BASELINE_AUDIT.md` for comparator corrections.

## Current checkpoint

- Native 16-series CAD is packaged in `assets/cad16_native_meshes.zip` with relative paths, per-part hashes, metre units and source manifest identity. It is an appearance asset, not yet a calibrated dynamic model.
- Existing MuJoCo numerical PCC and actuator checks are in `data/`; the simulated actuator command/plan bound is 118 degrees, evaluation bound 120 degrees.
- `live/model_checkpoint.pdf` explains the native CAD preview and independent physics checks. `paper/main.tex` includes a scoped development-check appendix; historical task data are unchanged.
- SDK interface audit: `data/qingyu_interface_audit/`. The installed Qingyu API has not been actuated by this revision.
- Original workspace HEAD and pre-existing tracked changes were checked unchanged in `data/preservation_check.json`.

## Monitoring

Public stage snapshots: https://aerial-continuum-research-monitor.consideratepastry4.chatgpt.site . The public site retains a snapshot when the NUC is disconnected. Local runtime monitor: NUC loopback port 8767, Windows forwarded port 18767. CAD renderings, measured/integrated physics and recorded replay are labelled separately. Codex usage is a timestamped official-account snapshot, not third-party API credit.

## Next milestone

Reconcile the native CAD dimensions/frames/collision envelope, the SDK command/feedback contract and the current MuJoCo platform before a new free-flight experiment. Reuse the paper's actual rolling-buffer/gate/bridge implementation, then add PX4 SITL and sensor mapping in separately verified steps. Do not substitute per-leg stop-and-go execution for rolling handoff evidence.

## 组会逐轮归档

[公网归档目录](https://aerial-continuum-research-monitor.consideratepastry4.chatgpt.site/archive/index.html) · [离线归档入口](meeting_archive/archive/index.html) · [下一轮的归档与发布步骤](meeting_archive/README.md)。每轮结束时冻结当轮图、曲线、视频、数据与论文页，校验旧轮与离线包后发布并暂停。
