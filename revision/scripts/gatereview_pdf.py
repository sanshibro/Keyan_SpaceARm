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
from io import BytesIO
import json
import math
import re
from pathlib import Path

import numpy as np

from gatereview_visuals import METHODS,CN,COLORS,readj,digest,protocol_cases,logical_name

MAIN_FIELDS=("success","body_min_m","arm_tool_min_m","self_min_m","hold_s","completion_s",
        "tip_rmse_m","tip_peak_m","axis_rmse_deg","axis_peak_deg","theta_rmse_deg",
        "phi_rmse_deg","theta_blackout_rmse_deg","phi_blackout_rmse_deg")
DETAIL_FIELDS={
    "safe":"safe","crossed":"crossed","dwell_reached":"dwell_reached","max_continuous_dwell_s":"max_dwell_s",
    "physical_contact_steps":"physical_contact_steps","inside_aperture":"passed_inside_aperture",
    "feedback_hold_s":"feedback_hold_s","clearance_hold_s":"clearance_hold_s",
    "whole_samples":"whole_angle.samples","whole_estimates":"whole_angle.available","whole_azimuth_samples":"whole_angle.azimuth_samples",
    "black_samples":"blackout_angle.samples","black_estimates":"blackout_angle.available","black_azimuth_samples":"blackout_angle.azimuth_samples",
    "dynamic_theta_deg":"dynamic_theta_range_deg","dynamic_phi_deg":"dynamic_phi_range_deg",
    "held_theta_rmse_deg":"vision_hold_blackout.theta_rmse_deg","held_phi_rmse_deg":"vision_hold_blackout.phi_rmse_deg",
    "held_theta_peak_deg":"vision_hold_blackout.theta_peak_deg","held_phi_peak_deg":"vision_hold_blackout.phi_peak_deg",
    "measured_max_bend_deg":"true_max_bend_deg","maximum_integrated_curvature_deg":"true_max_total_curvature_deg",
    "terminal_samples_after_gate":"terminal_after_gate_samples"}
FIELDS=MAIN_FIELDS+tuple(DETAIL_FIELDS)


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
    mapping={**DETAIL_FIELDS,**(field_map or {})}
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
    from pypdf import PdfReader, PdfWriter, Transformation
    from reportlab.lib.colors import HexColor
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfgen.canvas import Canvas

    root=Path(root).resolve();protocol=Path(protocol or root/"protocol.json")
    metrics=Path(metrics or root/"decisions/metrics.json");vd=Path(visual_dir or root/"review")
    output=Path(output or vd/"gate_review.pdf");rows=protocol_cases(protocol)
    scored,raw=read_metrics(metrics,readj(metric_map) if metric_map else None)
    approved=readj(protocol).get("approved","")[:7];registered=raw.get("design_commit","")[:7]
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
    pictures=[vd/"gate_keyframes.pdf",vd/"gate_four_metrics.pdf"]
    for path in pictures:
        if digest(path) not in declared:raise ValueError(f"Figure is not bound to the current visual manifest: {path}")
    if not allow_incomplete and vm.get("split")!="formal":raise ValueError("Formal report cannot silently use development example figures")
    regular,bold,font_used=font_setup(font,font_bold)
    W,H,L,R=595.276,841.89,32.,563.276
    output.parent.mkdir(parents=True,exist_ok=True)
    template=BytesIO();placements=[]
    c=Canvas(template,pagesize=(W,H),pageCompression=1)
    c.setTitle("冻结形态反馈的受限过门与双自由度作业闭环");c.setAuthor("空中连续体操作研究组")
    ink,muted,accent="#293B42","#65767A","#478B7A"
    def text(x,y,value,size=9,weight=False,color=ink,align="left"):
        c.setFillColor(HexColor(color));c.setFont(bold if weight else regular,size)
        getattr(c,{"left":"drawString","right":"drawRightString","center":"drawCentredString"}[align])(x,H-y,str(value))
    def rule(y,width=.5,color="#CFDADA"):
        c.setStrokeColor(HexColor(color));c.setLineWidth(width);c.line(L,H-y,R,H-y)
    def wrap(y,value,size=8.7,leading=13,color=ink):
        line=""
        tokens=re.findall(r"[0-9]+(?:[./–-][0-9]+)*(?:\s?(?:cm|mm|ms|s|°|%))?|[A-Za-z]+|\n|.",value)
        for token in tokens:
            if token=="\n" or (line and pdfmetrics.stringWidth(line+token,regular,size)>R-L):
                text(L,y,line.rstrip(),size,color=color);y+=leading;line="" if token=="\n" else token.lstrip()
            else:line+=token
        if line:text(L,y,line,size,color=color);y+=leading
        return y
    def header(page,title,subtitle):
        text(L,28,"空中连续体操作研究组 / 冻结感知闭环评估",8,color=muted)
        text(R,28,"PX4 SITL + MuJoCo",8,color=muted,align="right");rule(40,.8)
        text(L,71,title,18,True);wrap(93,subtitle,8.5,12,muted)
        rule(804);text(L,823,"SUPER 机架与 Qingyu 连续体臂　·　受限空间斜向检视",8,color=muted)
        if approved and registered:text(R-32,823,f"感知 {approved}　/　闭环 {registered}",7.2,color=muted,align="right")
        text(R,823,f"{page} / 2",8,color=muted,align="right")
    def figure(path,top,max_height):
        src=PdfReader(str(path));pa=src.pages[0];box=pa.mediabox
        scale=(R-L)/float(box.width);height=float(box.height)*scale
        if height>max_height:raise ValueError(f"{path.name} exceeds allocated figure height; regenerate its layout")
        transform=Transformation().translate(-float(box.left),-float(box.bottom)).scale(scale).translate(L,H-top-height)
        placements.append((pa,transform))
        return top+height

    header(1,"形态反馈驱动的过门与斜向对准","二维弯曲估计修正带载机械臂的驱动量，并更新全身包络。整机后缘越门后，机身与夹爪协同对准侧偏 25 cm 的目标。五列为同一种子下的独立飞行。")
    y=figure(pictures[0],112,245)
    y=figure(pictures[1],y+8,355)
    representative=raw.get("cases",{}).get(vm.get("cases",{}).get("proposed",""),{})
    dyn=representative.get("dynamic_theta_range_deg");rot=representative.get("dynamic_phi_range_deg")
    vh=representative.get("vision_hold_blackout",{}).get("theta_rmse_deg")
    fus=representative.get("theta_blackout_rmse_deg")
    vp=representative.get("vision_hold_blackout",{}).get("phi_rmse_deg")
    fp=representative.get("phi_blackout_rmse_deg")
    if all(v is not None for v in (dyn,rot,vh,fus,vp,fp)):
        y=wrap(y+12,f"图 1　同一趟融合飞行的 4 秒遮挡内，真实 θ 与 φ 分别变化 {dyn:.1f}° 和 {rot:.1f}°。沿该轨迹保留最后一次视觉观测，两个角的 RMSE 为 {vh:.2f}° 和 {vp:.2f}°，增广融合为 {fus:.2f}° 和 {fp:.2f}°。",8.5,12)
    y=wrap(y+5,"三行模型依次对应遮挡中的同一时刻、各趟最接近门框的过门时刻、门后同一时刻。红线连接臂爪与门框最近点。蓝绿实线与赭色虚线分别表示实际工具轴和目标轴。",8.4,12,muted)
    if y>795:raise ValueError(f"Page 1 overflow ({y:.1f} pt)")
    c.showPage()

    scored_n=sum(r["case"] in scored and scored[r["case"]]["success"] is not None for r in rows)
    header(2,"配对种子上的任务结果与精度","机身与臂爪净空均 ≥6.0 cm，自碰撞间距为正，整机无接触越门。门后位置 ≤3.5 cm 且轴向 ≤3.0°，连续保持 1.0 s。")
    text(L,121,f"五种反馈源 × 两种材料条件 × 两种视野 × 三个配对种子，共评分 {scored_n}/{len(rows)} 趟。",8.6,color=muted)
    widths=[47,31,57,34,46,34,59,59,58,106]
    # Reserve the remaining width for the blackout angle pair.
    widths[-1]=R-L-sum(widths[:-1])
    xs=[L]
    for w in widths:xs.append(xs[-1]+w)
    centers=[(a+b)/2 for a,b in zip(xs[:-1],xs[1:])]
    headers=[("方法",""),("成功","趟"),("机身/臂爪","最小 cm"),("自碰间距","最小 mm"),("停顿 s","总/断流"),("用时","s"),("末端位置","RMSE/峰 cm"),("轴向角","RMSE/峰 °"),("全程 θ/φ","RMSE °"),("遮挡 θ/φ","RMSE ° · 有效占比")]
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
                exposed=sum(r.get("black_samples") or 0 for r in entries)
                available=sum(r.get("black_estimates") or 0 for r in entries)
                black_text=(pair("theta_blackout_rmse_deg","phi_blackout_rmse_deg")+f" · {available/exposed:.0%}") if exposed else "—"
                vals=[CN[method],success,pair("body_min_m","arm_tool_min_m",100,"min"),one("self_min_m",1000,"min"),pair("hold_s","feedback_hold_s"),one("completion_s"),pair("tip_rmse_m","tip_peak_m",100),pair("axis_rmse_deg","axis_peak_deg"),pair("theta_rmse_deg","phi_rmse_deg"),black_text]
                for i,value in enumerate(vals):text(centers[i],y,value,7.7,method=="proposed",COLORS[method] if i==0 else ink,"center")
                for r,e in zip(selected,entries):export_rows.append({**r,**{k:e.get(k) for k in FIELDS}})
                y+=18
            rule(y-10,.4);y+=9
    note="净空与自碰撞间距取每组三趟最小值，误差、停顿和用时取逐趟中位数。末端误差统计窗为 49–55 s。用时计至安全通过并连续对准满 1 秒。"
    if scored_n<len(rows):note+="* 表示有效试次数少于三趟。"
    y=wrap(y+2,note+"空项用横线标示。",7.8,12,muted)
    y=wrap(y+2,"遮挡窗为曝光时钟的 28–32 s，有效占比按全部控制样本计算。曝光至交付延迟 0.20 s，使遮挡开始时仍可收到先前的图像。方位 RMSE 在真实与估计弯角均达到 5° 时按圆周差计算。",7.8,12,muted)
    focus={m:[v for v in raw["cases"].values() if v["case"]["material"]=="differential" and v["case"]["vision"]=="occluded" and v["case"]["method"]==m] for m in METHODS}
    if all(len(v)==3 for v in focus.values()):
        nominal=np.median([v["arm_tool_min_m"]*100 for v in focus["proposed"]])
        biased=np.median([v["arm_tool_min_m"]*100 for v in focus["strain"]])
        crossed=sum(v["arm_tool_min_m"]<.06 for v in focus["strain"])
        angles=np.median([v["axis_rmse_deg"] for v in focus["strain"]])
        kf=sum(v["success"] for v in focus["fixed"])
        proposed=sum(v["success"] for v in focus["proposed"])
        audit=readj(root/"decisions/case-evidence-audit.json")
        undershoot=np.median([audit[v["case"]["case"]]["arm_at_minimum"]["theta_des_minus_true_deg"] for v in focus["strain"]])
        gate_result="遮挡组三趟纯应变均守住 6.0 cm" if crossed==0 else f"遮挡组纯应变有 {crossed}/3 趟低于 6.0 cm"
        clear_strain=[v['arm_tool_min_m']*100 for v in raw['cases'].values() if
                     (v['case']['material'],v['case']['vision'],v['case']['method'])==('differential','natural','strain')]
        if len(clear_strain)==3:
            gate_result+=f"，清晰组最低 {min(clear_strain):.2f} cm，{sum(v<6. for v in clear_strain)}/3 趟低于红线"
        y=wrap(y+5,f"差模偏移与遮挡并存时，纯应变在臂爪最接近门框处的实际欠弯中位数为 {undershoot:.2f}°。融合与纯应变的最小净空中位数为 {nominal:.2f} 与 {biased:.2f} cm。{gate_result}。遮挡组纯应变门后轴向 RMSE 中位数为 {angles:.2f}°，固定 KF 和增广融合分别完成 {kf}/3 和 {proposed}/3 趟。",8.3,12,ink)
        clear=[v for v in raw['cases'].values() if (v['case']['material'],v['case']['vision'],v['case']['method'])==('differential','natural','vision')]
        if len(clear)==3:
            visual=focus['vision']
            clear_hold=np.median([v['feedback_hold_s'] for v in clear])
            dark_hold=np.median([v['feedback_hold_s'] for v in visual])
            dock_contact=sum(audit[v['case']['case']]['contact_and_holds']['first_phase']=='dock' for v in clear+visual)
            y=wrap(y+2,f"同为差模偏移时，清晰与遮挡条件下纯视觉因反馈断流停顿的中位数为 {clear_hold:.2f} 与 {dark_hold:.2f} s。六趟中有 {dock_contact} 趟在门后对准阶段首次触及自碰撞边界。",8.1,12,ink)
    if y>795:raise ValueError(f"Page 2 overflow ({y:.1f} pt)")
    c.showPage();c.save();template.seek(0)
    paper=PdfReader(template);writer=PdfWriter()
    writer.append(paper);writer.add_metadata(dict(paper.metadata))
    for page,transform in placements:
        writer.pages[0].merge_transformed_page(page,transform,over=True,expand=False)
    temp=output.with_suffix(".building")
    with temp.open("wb") as stream:writer.write(stream)
    temp.replace(output)
    csvpath=output.parent/"all_trials_metrics.csv"
    with csvpath.open("w",encoding="utf-8-sig",newline="") as f:
        cols=["case","method","material","vision","seed",*FIELDS]
        w=csv.DictWriter(f,fieldnames=cols,extrasaction="ignore");w.writeheader();w.writerows(export_rows)
    audited=root/"decisions/case-evidence-audit.json"
    used=[protocol,metrics,vm_path,*pictures]
    if audited.is_file():used.append(audited)
    manifest={"pages":2,"registered_trials":len(rows),"scored_trials":scored_n,"font":font_used,
        "inputs":[{"path":logical_name(p,root),"sha256":digest(p)} for p in used],
        "metrics_mapping":{**{k:k for k in MAIN_FIELDS},**DETAIL_FIELDS,**(readj(metric_map) if metric_map else {})},
        "output":logical_name(output,root),"output_sha256":digest(output),"vector_figures":True,"no_outcome_recomputation":True}
    (output.parent/"gate_review_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("root","protocol","metrics","visual-dir","output","font","font-bold","metric-map"):
        p.add_argument("--"+name,type=Path,required=name=="root")
    p.add_argument("--allow-incomplete",action="store_true")
    a=p.parse_args();print(build(a.root,a.protocol,a.metrics,a.visual_dir,a.output,a.font,a.font_bold,a.metric_map,a.allow_incomplete))


if __name__=="__main__":main()
