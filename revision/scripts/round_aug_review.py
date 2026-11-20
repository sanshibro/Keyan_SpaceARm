"""Two-page Chinese review of THIS round's exported scorer results.

No file is produced on import. The parent must run its PDF skill operation marker
before explicitly invoking this script. No old reference, training/test mocap,
estimator or publication endpoint is opened by this module.

Dependencies: numpy, Pillow, reportlab, round_aug_visuals.py. A supplied CJK
TrueType font is preferred (--font, optional --font-bold). Windows Microsoft
YaHei and Linux WenQuanYi are detected; STSong-Light CID is the portable fallback.

Usage: python round_aug_review.py --root revision/data/round_augmented_fusion
       --visual-dir revision/data/round_augmented_fusion/review --font /path/cjk.ttf
Required: scores.json, fusion_main.png, visual_manifest.json (same current round).
Output: review.pdf (exactly two pages), review_manifest.json, full_scores.csv.
All table values are scorer outputs. Drift pools common/differential/single by
the number of valid errors, never by taking an arithmetic mean of RMSE values.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from round_aug_visuals import (METHODS, CN_NAMES, COLORS, CONDITIONS, COND_CN,
    SOURCES, read_scores, score_entry, score_value, sha256)


def aggregate_drift(scores, condition, phase, method):
    supplied = score_entry(scores, "drift", condition, phase, method)
    if supplied is not None:
        return supplied
    rows = [score_entry(scores, s, condition, phase, method)
            for s in ("common", "differential", "single")]
    # Incomplete source pools are not silently renamed as the complete drift pool.
    if any(r is None for r in rows):
        return None
    n = sum(int(r["n"]) for r in rows)
    valid = sum(int(r["valid"]) for r in rows)
    phi_n = sum(int(r["phi_n"]) for r in rows)
    def pooled(key, count):
        den, ss = 0, 0.
        for r in rows:
            k = int(r[count]); v = score_value(r, key)
            if k and v is None:
                return None
            if k:
                den += k; ss += k * v * v
        return math.sqrt(ss/den) if den else None
    return {"n": n, "valid": valid, "coverage": valid/n if n else None,
            "theta": pooled("theta", "valid"), "phi": pooled("phi", "phi_n"),
            "phi_n": phi_n, "aggregation": "common+differential+single; pooled squared errors"}


def entry(scores, source, condition, phase, method):
    return aggregate_drift(scores, condition, phase, method) if source == "drift" else score_entry(scores, source, condition, phase, method)


def metric(value, digits=2):
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def cell(r):
    if r is None:
        return "— / — / —", "n: —"
    cov = score_value(r, "coverage")
    top = f"{metric(r.get('theta'))} / {metric(r.get('phi'))} / {metric(None if cov is None else 100*cov, 1)}"
    bottom = f"{int(r.get('valid', 0))}/{int(r.get('n', 0))}; φ {int(r.get('phi_n', 0))}"
    return top, bottom


def fonts(regular=None, bold=None):
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    candidates = ([Path(regular)] if regular else []) + [
        Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simhei.ttf"),
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        Path("/usr/share/fonts/truetype/arphic/uming.ttc")]
    used = None
    for path in candidates:
        if not path.is_file():
            continue
        try:
            pdfmetrics.registerFont(TTFont("AugRegular", str(path), subfontIndex=0))
            used = str(path); break
        except Exception:
            if regular and path == Path(regular):
                raise ValueError(f"Cannot embed supplied TrueType CJK font: {path}")
    if used is None:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        return "STSong-Light", "STSong-Light", "STSong-Light CID (not embedded)"
    bpath = Path(bold) if bold else Path("C:/Windows/Fonts/msyhbd.ttc")
    if bpath.is_file():
        try:
            pdfmetrics.registerFont(TTFont("AugBold", str(bpath), subfontIndex=0))
            return "AugRegular", "AugBold", used
        except Exception:
            if bold: raise
    return "AugRegular", "AugRegular", used


def export_table(scores, path):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "condition", "phase", "method", "n", "valid", "coverage", "theta_rmse_deg", "phi_rmse_deg", "phi_n"])
        for source in (*SOURCES, "drift"):
            for condition in CONDITIONS:
                for phase in ("all", "blind", "steady", "degraded", "recovered"):
                    for method in METHODS:
                        r = entry(scores, source, condition, phase, method)
                        if r is not None:
                            w.writerow([source, condition, phase, method, *[r.get(k) for k in ("n", "valid", "coverage", "theta", "phi", "phi_n")]])


def build(root, visual_dir=None, output=None, font=None, font_bold=None):
    from PIL import Image
    from reportlab.lib.colors import HexColor
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfgen.canvas import Canvas

    root = Path(root).resolve(); rd = Path(visual_dir or root / "review").resolve()
    output = Path(output or rd / "review.pdf").resolve()
    scores = read_scores(root)
    picture = rd / "fusion_main.png"
    vm_path = rd / "visual_manifest.json"
    vm = json.loads(vm_path.read_text(encoding="utf-8"))
    if vm.get("source") != "differential" or vm.get("condition") != "occluded":
        raise ValueError("The registered main plate is differential / occluded")
    if vm.get("case") != "26112012-left" or len(vm.get("epochs", [])) != 4:
        raise ValueError("The main plate must show the registered left flight and four epochs")
    # Bind imagery to the exact current scored trace, not merely its basename.
    expected = root / "scored" / vm["case"] / "differential-occluded.npz"
    trace_hash = sha256(expected)
    if trace_hash not in {x.get("sha256") for x in vm.get("inputs", [])}:
        raise ValueError("Main plate manifest does not match this round's scored trace")
    plate_hash = sha256(picture)
    if plate_hash not in {x.get("sha256") for x in vm.get("outputs", [])}:
        raise ValueError("Main plate bitmap does not match its visual manifest")
    regular, bold, font_source = fonts(font, font_bold)
    output.parent.mkdir(parents=True, exist_ok=True)
    W, H, L, R = 595.276, 841.89, 35., 560.276
    ink, muted, accent = "#263A40", "#617176", "#398777"
    c = Canvas(str(output), pagesize=(W, H), pageCompression=1)
    c.setTitle("面向视觉退化与应变迟滞的连续体臂形态融合：三趟留出飞行的配对评估")
    c.setAuthor("空中连续体操作研究组")

    def text(x, top, value, size=9, weight=False, color=ink, align="left"):
        name = bold if weight else regular
        c.setFont(name, size); c.setFillColor(HexColor(color))
        getattr(c, {"left": "drawString", "right": "drawRightString", "center": "drawCentredString"}[align])(x, H-top, str(value))

    def rule(top, color="#C7D3D4", width=.55, x0=L, x1=R):
        c.setStrokeColor(HexColor(color)); c.setLineWidth(width); c.line(x0, H-top, x1, H-top)

    def wrap(top, value, size=9, leading=14, color=ink, width=None):
        width = R-L if width is None else width
        line = ""
        for char in value:
            if char == "\n" or pdfmetrics.stringWidth(line+char, regular, size) > width:
                text(L, top, line, size, color=color); top += leading; line = "" if char == "\n" else char
            else: line += char
        if line: text(L, top, line, size, color=color); top += leading
        return top

    def header(page, title, subtitle):
        text(L, 28, "空中连续体操作研究组 / 形态感知评估", 8, color=muted)
        text(R, 28, "PX4 SITL · MuJoCo · RGB-D ORB-SLAM3", 8, color=muted, align="right")
        rule(40, width=.8); text(L, 72, title, 18, True)
        wrap(94, subtitle, 9, 14, muted)
        rule(803); text(L, 821, "三趟留出飞行 / 同轨配对条件 / 本轮冻结评分", 8, color=muted)
        text(R, 821, f"{page} / 2", 8, color=muted, align="right")

    header(1, "面向视觉退化与应变迟滞的连续体臂形态融合", "三趟 PX4 / MuJoCo 仿真飞行，腕部相机与臂中段四片预拉伸 SLIM M 同步观测。")
    with Image.open(picture) as im:
        im = im.convert("RGB"); height = (R-L)*im.height/im.width
        if height > 550:
            raise ValueError("Main figure is too tall for the two-page review template")
        c.drawImage(ImageReader(im), L, H-112-height, R-L, height)
    y = 112+height+15
    y = wrap(y, "图 1　左偏航迹中的差模偏移。上排按曝光时刻配对整机姿态与腕部图像。竖线标出各帧经过 0.20 s 延迟后的送达时刻。", 8.3, 12.8, muted)
    y += 9
    y = wrap(y, "连续逆模型从四路传感器读数重建弯曲向量。关键帧匹配与深度平面法向约束相机姿态。增广滤波器在曝光时刻更新臂形与二维应变偏置，再向前重放应变采样。失视期间沿用最近一次视觉校准的偏置。", 9, 14)
    y += 5
    normal = entry(scores, "nominal", "natural", "all", "proposed")
    blind = entry(scores, "differential", "occluded", "blind", "proposed")
    stats = (f"正常视野配合基础迟滞时，弯曲角 RMSE 为 {metric(None if normal is None else normal.get('theta'))}°。"
             f"差模漂移/遮挡窗 θ RMSE {metric(None if blind is None else blind.get('theta'))}°，"
             f"输出覆盖率 {metric(None if blind is None or blind.get('coverage') is None else 100*blind['coverage'], 1)}%。")
    y = wrap(y, stats, 9, 14, accent)
    rear = entry(scores, "nominal", "natural", "recovered", "vision")
    y = wrap(y+3, f"原地图上持续进行几何校验后，无遮挡后半程（36–46 s）的纯视觉弯曲角 RMSE 为 {metric(rear['theta'], 3)}°。", 9, 14)
    if y > 795:
        raise ValueError(f"Page one content overflow at {y:.1f} pt")
    c.showPage()

    header(2, "三种视觉条件下的配对结果", "每格依次为弯曲角 RMSE、弯曲方位 RMSE 和输出覆盖率（° / ° / %）。下行给出有效时刻数。")
    columns = [L, L+105, L+210, L+315, L+420, R]

    def table(top, title, rows, row_height=29):
        text(L, top, title, 10.3, True, accent)
        rule(top+8, width=.8)
        for j, name in enumerate(CN_NAMES):
            text((columns[j+1]+columns[j+2])/2, top+24, name, 8.6, True, COLORS[j], "center")
        text(L+2, top+24, "应变 / 视觉 / 区间", 8.2, color=muted)
        rule(top+31)
        y = top+46
        for label, source, condition, phase in rows:
            # Label uses two lines if explicitly separated by newline.
            for k, part in enumerate(label.split("\n")):
                text(L+2, y+k*10, part, 7.8, color=ink if k == 0 else muted)
            for j, method in enumerate(METHODS):
                r = entry(scores, source, condition, phase, method)
                value, counts = cell(r)
                x = (columns[j+1]+columns[j+2])/2
                text(x, y, value, 7.9, j == 3, COLORS[j] if j == 3 else ink, "center")
                text(x, y+10, counts, 6.5, color=muted, align="center")
            rule(y+17, color="#E6EBEB", width=.35)
            y += row_height
        return y-8

    rows = [(f"{slabel} / {COND_CN[cond]}\n全程 all", source, cond, "all")
            for source, slabel in (("nominal", "基础迟滞"), ("drift", "通道偏移")) for cond in CONDITIONS]
    y = table(124, "表 1  三视觉条件 × 基础迟滞 / 通道漂移", rows)
    y = wrap(y+4, "通道偏移合并共模、差模、单通道三项的逐时刻误差。各组均包含加载与卸载迟滞以及停驻松弛。每趟材料参数按预设范围变化，同一趟的各方法共享同一组参数。", 7.6, 11, muted)
    y = table(y+15, "表 2  差模偏移单独列示（全程）", [
        (f"差模 / {COND_CN[cond]}\n全程 all", "differential", cond, "all") for cond in CONDITIONS])
    y = table(y+15, "表 3  差模偏移下的分阶段评估", [
        ("正常视野\n稳态 steady", "differential", "natural", "steady"),
        ("渐进退化\n退化 degraded", "differential", "graded", "degraded"),
        ("完全遮挡\n遮挡 blind", "differential", "occluded", "blind"),
        ("遮挡后恢复\n恢复 recovered", "differential", "occluded", "recovered")], row_height=27)
    y = wrap(y+5, "全程为 14–46 s，清晰稳态为 24–46 s。完全遮挡的送达时段为 31.2–35.2 s，恢复评分为 36–46 s。参考与估计弯角均达 5° 时用圆周差计算方位误差。", 7.5, 11, muted)
    late = entry(scores,'late','occluded','blind','proposed')
    y = wrap(y+3, f"主试验在 18–22 s 施加偏移。附加试验将偏移移至 29–33 s，与遮挡重叠，此时本文方法的遮挡段弯曲角 RMSE 为 {metric(late['theta'])}°。", 7.7, 11, muted)
    checks = json.loads((root/"acceptance.json").read_text(encoding="utf-8"))["checks"]
    delays = [r["value"] for r in checks if r["metric"] == "differential first accepted s"]
    steps = [r["value"] for r in checks if r["metric"] == "differential post-lock step residual degrees"]
    if len(delays) != 3 or len(steps) != 3:
        raise ValueError("Expected three individual relock measurements")
    y = wrap(y+9, f"重锁时序　直行、左偏、右偏三趟从撤除遮挡到首次采用视觉的时间依次为 {'、'.join(metric(d,3) for d in delays)} s，均计入 0.20 s 图像延迟。差模工况重锁时最大超额单步角变化为 {metric(max(steps))}°。", 8.2, 12, accent)
    if y > 795:
        raise ValueError(f"Page two content overflow at {y:.1f} pt; reduce table spacing, never omit results")
    c.showPage(); c.save()
    export_table(scores, output.parent / "full_scores.csv")
    manifest = {"pages": 2, "pdf": str(output), "font": font_source,
        "inputs": [{"path": str(p), "sha256": sha256(p)} for p in (root/"scores.json", root/"acceptance.json", picture, vm_path)],
        "metric_units": {"theta": "degree RMSE", "phi": "circular degree RMSE", "coverage": "valid/n"},
        "drift_pool": ["common", "differential", "single"], "truth_reads": "none; uses scorer summaries"}
    (output.parent / "review_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True); p.add_argument("--visual-dir", type=Path)
    p.add_argument("--output", type=Path); p.add_argument("--font", type=Path); p.add_argument("--font-bold", type=Path)
    a = p.parse_args(); print(build(a.root, a.visual_dir, a.output, a.font, a.font_bold))


if __name__ == "__main__":
    main()
