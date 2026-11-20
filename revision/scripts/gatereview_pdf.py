"""Exactly two Chinese review pages from independent gate-flight metrics.

Explicit authoring only; no output on import. The parent runs the PDF skill marker
before generation. This script starts no simulation, publishes nothing, and does
not turn visualization thresholds into inferred success labels.

Dependencies: reportlab, Pillow, gatereview_visuals.py (numpy). Supply an embedded
CJK TrueType font with --font; Windows YaHei/Linux WenQuanYi are also discovered.
Inputs: protocol.json; decisions/metrics.json; review/gate_keyframes.png,
gate_four_metrics.png and visual_manifest.json from THIS round.

Metric format: {"cases": {"<case>": {canonical fields...}}}; a list under cases
or runs is also accepted. --metric-map JSON maps canonical fields to dot-separated
paths when the independent scorer uses different field names. Never guess values.
Canonical fields: success (bool), body_min_m, arm_tool_min_m, self_min_m, hold_s,
completion_s, tip_rmse_m, tip_peak_m, axis_rmse_deg, axis_peak_deg,
theta_rmse_deg, phi_rmse_deg, theta_blackout_rmse_deg, phi_blackout_rmse_deg.
Missing estimates/unaligned task windows remain absent, not zero.

Output: gate_review.pdf, gate_review_manifest.json, all_trials_metrics.csv.
The detailed CSV preserves all per-trial metrics; the table uses minima for
clearances and medians of per-trial diagnostics, with successes counted explicitly.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from gatereview_visuals import METHODS,CN,COLORS,readj,digest,protocol_cases

FIELDS=("success","body_min_m","arm_tool_min_m","self_min_m","hold_s","completion_s",
        "tip_rmse_m","tip_peak_m","axis_rmse_deg","axis_peak_deg","theta_rmse_deg",
        "phi_rmse_deg","theta_blackout_rmse_deg","phi_blackout_rmse_deg")


def lookup(obj,path):
    current=obj
    for part in path.split("."):
        if not isinstance(current,dict) or part not in current:return None
        current=current[part]
    return current


def read_metrics(path,field_map=None):
    raw=readj(path);rows=raw.get("cases",raw.get("runs",raw))
    if isinstance(rows,list):rows={str(r["case"]):r for r in rows}
    if not isinstance(rows,dict):raise ValueError("Metrics require a case-keyed mapping or list")
    mapping=field_map or {}
    result={}
    for case,r in rows.items():
        if not isinstance(r,dict):continue
        result[case]={k:lookup(r,mapping.get(k,k)) for k in FIELDS}
        v=result[case]["success"]
        if v is not None and not isinstance(v,bool):raise ValueError(f"{case}: success must be a scorer Boolean")
    return result,raw


def number(x):
    if x is None:return None
    y=float(x)
    return y if math.isfinite(y) else None


def aggregate(rows,key,how="median",scale=1.):
    values=[number(r.get(key)) for r in rows]
    values=[x*scale for x in values if x is not None]
    if not values:return None,0
    return float(np.min(values) if how=="min" else np.median(values)),len(values)


def show(x,digits=1):return "—" if x is None else f"{x:.{digits}f}"


def font_setup(font=None,bold=None):
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    choices=([Path(font)] if font else [])+[Path("C:/Windows/Fonts/msyh.ttc"),Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")]
    for path in choices:
        if not path.is_file():continue
        try:pdfmetrics.registerFont(TTFont("GateCJK",str(path),subfontIndex=0))
        except Exception:
            if font and path==Path(font):raise
            continue
        bp=Path(bold) if bold else Path("C:/Windows/Fonts/msyhbd.ttc")
        if bp.is_file():
            pdfmetrics.registerFont(TTFont("GateBold",str(bp),subfontIndex=0));return "GateCJK","GateBold",str(path)
        return "GateCJK","GateCJK",str(path)
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"));return "STSong-Light","STSong-Light","STSong-Light CID fallback"


def build(root,protocol=None,metrics=None,visual_dir=None,output=None,font=None,font_bold=None,metric_map=None,allow_incomplete=False):
    from PIL import Image
    from reportlab.lib.colors import HexColor
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfgen.canvas import Canvas

    root=Path(root).resolve();protocol=Path(protocol or root/"protocol.json")
    metrics=Path(metrics or root/"decisions/metrics.json");vd=Path(visual_dir or root/"review")
    output=Path(output or vd/"gate_review.pdf");rows=protocol_cases(protocol)
    scored,raw=read_metrics(metrics,readj(metric_map) if metric_map else None)
    if not allow_incomplete:
        missing=[r["case"] for r in rows if r["case"] not in scored or scored[r["case"]]["success"] is None]
        if missing:raise ValueError(f"Missing independent outcome scores for {len(missing)} registered flights")
        if len(rows)!=60:raise ValueError("Registered formal matrix must contain 60 cases; use --allow-incomplete only for a labeled development preview")
        paired_seed_sets=[]
        for material in ("nominal","differential"):
            for vision in ("natural","occluded"):
                for method in METHODS:
                    group=[r for r in rows if r["material"]==material and r["vision"]==vision and r["method"]==method]
                    seeds={str(r["seed"]) for r in group}
                    if len(group)!=3 or len(seeds)!=3:raise ValueError("Formal matrix requires three distinct paired seeds in every condition/method cell")
                    paired_seed_sets.append(seeds)
        if any(s!=paired_seed_sets[0] for s in paired_seed_sets):raise ValueError("Formal condition cells do not share the same paired seeds")
    vm_path=vd/"visual_manifest.json";vm=readj(vm_path)
    declared={x.get("sha256") for x in vm.get("outputs",[])}
    pictures=[vd/"gate_keyframes.png",vd/"gate_four_metrics.png"]
    for path in pictures:
        if digest(path) not in declared:raise ValueError(f"Figure is not bound to the current visual manifest: {path}")
    if not allow_incomplete and vm.get("split")!="formal":raise ValueError("Formal report cannot silently use development example figures")
    regular,bold,font_used=font_setup(font,font_bold)
    W,H,L,R=595.276,841.89,32.,563.276
    output.parent.mkdir(parents=True,exist_ok=True)
    c=Canvas(str(output),pagesize=(W,H),pageCompression=1)
    c.setTitle("冻结形态反馈的受限过门与双自由度作业闭环");c.setAuthor("空中连续体操作研究组")
    ink,muted,accent="#293B42","#65767A","#478B7A"
    def text(x,y,value,size=9,weight=False,color=ink,align="left"):
        c.setFillColor(HexColor(color));c.setFont(bold if weight else regular,size)
        getattr(c,{"left":"drawString","right":"drawRightString","center":"drawCentredString"}[align])(x,H-y,str(value))
    def rule(y,width=.5,color="#CFDADA"):
        c.setStrokeColor(HexColor(color));c.setLineWidth(width);c.line(L,H-y,R,H-y)
    def wrap(y,value,size=8.7,leading=13,color=ink):
        line=""
        for char in value:
            if char=="\n" or pdfmetrics.stringWidth(line+char,regular,size)>R-L:
                text(L,y,line,size,color=color);y+=leading;line="" if char=="\n" else char
            else:line+=char
        if line:text(L,y,line,size,color=color);y+=leading
        return y
    def header(page,title,subtitle):
        text(L,28,"空中连续体操作研究组 / 冻结感知闭环评估",8,color=muted)
        text(R,28,"PX4 SITL + MuJoCo",8,color=muted,align="right");rule(40,.8)
        text(L,71,title,18,True);wrap(93,subtitle,8.5,12,muted)
        rule(804);text(L,823,"SUPER 机架与 Qingyu 连续体臂　·　受限空间斜向检视",8,color=muted)
        text(R,823,f"{page} / 2",8,color=muted,align="right")
    def figure(path,top,max_height):
        with Image.open(path) as src:
            im=src.convert("RGB");height=(R-L)*im.height/im.width
            if height>max_height:raise ValueError(f"{path.name} exceeds allocated figure height; regenerate its layout")
            c.drawImage(ImageReader(im),L,H-top-height,R-L,height)
        return top+height

    header(1,"形态反馈驱动的过门与斜向对准","五种反馈源共用飞控参数、二维形态伺服与门框。每一列取自该方法独立飞出的轨迹。")
    y=figure(pictures[0],112,245)
    y=figure(pictures[1],y+8,355)
    representative=raw.get("cases",{}).get(vm.get("cases",{}).get("proposed",""),{})
    dyn=representative.get("dynamic_theta_range_deg");rot=representative.get("dynamic_phi_range_deg")
    vh=representative.get("vision_hold_blackout",{}).get("theta_rmse_deg")
    fus=representative.get("theta_blackout_rmse_deg")
    if all(v is not None for v in (dyn,rot,vh,fus)):
        y=wrap(y+12,f"图 1　同一趟融合飞行的 4 秒遮挡内，真实 θ 变化 {dyn:.1f}°、φ 变化 {rot:.1f}°。仅保留最后视觉帧时，θ 的 RMSE 为 {vh:.2f}°，融合为 {fus:.2f}°。",8.5,12)
    y=wrap(y+5,"上排连线连接臂爪与门框的三维最近点。下排比较实际工具轴与目标轴。过门后，位置误差须在 3.5 cm 内、轴向误差须在 3.0° 内，同时连续保持 1.0 s。",8.4,12,muted)
    if y>795:raise ValueError(f"Page 1 overflow ({y:.1f} pt)")
    c.showPage()

    scored_n=sum(r["case"] in scored and scored[r["case"]]["success"] is not None for r in rows)
    header(2,"配对种子上的任务结果与精度","表中值由独立评分器导入。成功按完整任务判据计数；净空报告组内最小值，其余连续指标为逐趟统计量的中位数。")
    text(L,121,f"注册 {len(rows)} 趟 / 已评 {scored_n} 趟；五方法 × 两材料条件 × 两视觉条件。",8.6,color=muted)
    widths=[54,33,48,33,33,36,62,60,57,115]
    # Reserve the remaining width for the blackout angle pair.
    widths[-1]=R-L-sum(widths[:-1])
    xs=[L]
    for w in widths:xs.append(xs[-1]+w)
    centers=[(a+b)/2 for a,b in zip(xs[:-1],xs[1:])]
    headers=[("方法",""),("成功","次/次"),("机身/臂爪","min cm"),("自距","min mm"),("保持","s"),("用时","s"),("末端位置","RMSE/峰 cm"),("轴向角","RMSE/峰 °"),("全程 θ/φ","RMSE °"),("遮挡 θ/φ","RMSE °")]
    rule(133,.8)
    for i,(a,b) in enumerate(headers):
        text(centers[i],148,a,7.8,True,align="center");text(centers[i],161,b,6.7,color=muted,align="center")
    rule(169)
    y=185;export_rows=[]
    for material,mlabel in (("nominal","基础物理迟滞"),("differential","物理迟滞 + 差模偏移")):
        for vision,vlabel in (("natural","清晰视野"),("occluded","动态变形期完全遮挡")):
            text(L+2,y,f"{mlabel} / {vlabel}",9.1,True,accent);y+=20
            for method in METHODS:
                selected=[r for r in rows if r["method"]==method and r["material"]==material and r["vision"]==vision]
                entries=[scored.get(r["case"],{}) for r in selected]
                known=sum(isinstance(r.get("success"),bool) for r in entries);passed=sum(r.get("success") is True for r in entries)
                success=f"{passed}/{len(selected)}" if known==len(selected) else f"{passed}/{len(selected)}*"
                def one(key,scale=1,how="median"):
                    val,n=aggregate(entries,key,how,scale);return show(val)+( "*" if n<len(selected) and n else "")
                def pair(a,b,scale=1,how="median"):
                    return one(a,scale,how)+"/"+one(b,scale,how)
                vals=[CN[method],success,pair("body_min_m","arm_tool_min_m",100,"min"),one("self_min_m",1000,"min"),one("hold_s"),one("completion_s"),pair("tip_rmse_m","tip_peak_m",100),pair("axis_rmse_deg","axis_peak_deg"),pair("theta_rmse_deg","phi_rmse_deg"),pair("theta_blackout_rmse_deg","phi_blackout_rmse_deg")]
                for i,value in enumerate(vals):text(centers[i],y,value,7.7,method=="proposed",COLORS[method] if i==0 else ink,"center")
                for r,e in zip(selected,entries):export_rows.append({**r,**{k:e.get(k) for k in FIELDS}})
                y+=18
            rule(y-10,.4);y+=9
    y=wrap(y+2,"三种子内取最小净空与最小自距，其余连续指标取逐趟中位数。用时计至安全通过且连续对准满 1 秒。星号表示有有效测量的试次数少于三趟，破折号表示无对应测量。",7.8,12,muted)
    y=wrap(y+2,"真值来自 MuJoCo 独立积分。φ 使用圆周误差，两弯角均达到 5° 时统计方位。图像按曝光后 0.20 秒交付，全黑期间有积压影像的到达时刻单独记录。",7.8,12,muted)
    if y>795:raise ValueError(f"Page 2 overflow ({y:.1f} pt)")
    c.showPage();c.save()
    csvpath=output.parent/"all_trials_metrics.csv"
    with csvpath.open("w",encoding="utf-8-sig",newline="") as f:
        cols=["case","method","material","vision","seed",*FIELDS]
        w=csv.DictWriter(f,fieldnames=cols,extrasaction="ignore");w.writeheader();w.writerows(export_rows)
    manifest={"pages":2,"registered_trials":len(rows),"scored_trials":scored_n,"font":font_used,
        "inputs":[{"path":str(p.resolve()),"sha256":digest(p)} for p in (protocol,metrics,vm_path,*pictures)],
        "metrics_mapping":readj(metric_map) if metric_map else {k:k for k in FIELDS},
        "output":str(output.resolve()),"no_outcome_recomputation":True}
    (output.parent/"gate_review_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("root","protocol","metrics","visual-dir","output","font","font-bold","metric-map"):
        p.add_argument("--"+name,type=Path,required=name=="root")
    p.add_argument("--allow-incomplete",action="store_true")
    a=p.parse_args();print(build(a.root,a.protocol,a.metrics,a.visual_dir,a.output,a.font,a.font_bold,a.metric_map,a.allow_incomplete))


if __name__=="__main__":main()
