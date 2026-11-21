"""Two-page meeting proof with the recorded states, paired measurements and source identity."""
from io import BytesIO
from pathlib import Path
import sys
import numpy as np
import pymupdf
from PIL import Image
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas
from gatereview_pdf import font_setup
from m0_data import OUT,QA,selected,read_json,sha_file,write_json

W,H=A4
LEFT,RIGHT=31,W-31
TEAL='#116e64'; INK='#203a3e'; GRAY='#526a6b'; RULE='#c5d2cf'
FONT,BOLD,_=font_setup()

def text(c,x,y,txt,size=9,font=FONT,color=INK,right=False):
    c.setFillColor(HexColor(color));c.setFont(font,size)
    (c.drawRightString if right else c.drawString)(x,y,txt)

def rule(c,y,x=LEFT,end=RIGHT):
    c.setStrokeColor(HexColor(RULE));c.setLineWidth(.55);c.line(x,y,end,y)

def pic(c,stem,top,w,left=LEFT):
    im=Image.open(OUT/(stem+'.png')).convert('RGB')
    im.thumbnail((2100,1650),Image.Resampling.LANCZOS)
    temp=BytesIO();im.save(temp,'JPEG',quality=94,subsampling=0,optimize=True);temp.seek(0)
    h=w*im.height/im.width;c.drawImage(ImageReader(temp),left,top-h,w,h);return top-h

def head(c,title,sub,page):
    text(c,LEFT,H-25,'单门形态闭环   /   工业隔舱检修',8,color=TEAL)
    text(c,RIGHT,H-25,'M0    ·    2026.11.21',8,color=GRAY,right=True)
    text(c,LEFT,H-53,title,21,BOLD);text(c,LEFT,H-74,sub,9,color=GRAY)
    rule(c,H-85)
    rule(c,28);text(c,LEFT,15,'SUPER 机架与 Qingyu 连续体臂    ·    已归档闭环轨迹 951c512',7.3,color=GRAY)
    text(c,RIGHT,15,f'{page} / 2',8,color=TEAL,right=True)

def main():
    flights=selected();p=flights['proposed'];st=flights['strain'];audit=read_json(QA/'scene-clearance.json')
    name=OUT/'gate_review_v2.pdf';c=Canvas(str(name),pagesize=A4,pageCompression=1)
    c.setTitle('工业隔舱过门与斜向对准｜单门闭环视觉审阅');c.setAuthor('Aerial Continuum Manipulation')
    head(c,'穿过隔舱，再对准斜置接口','同一配对种子 26112131，进门前形成差模偏移。腕部相机在 28–32 s 完全遮挡。',1)
    bottom=pic(c,'gate_hero_v2',H-91,RIGHT-LEFT)
    rule(c,bottom-6)
    bottom=pic(c,'gate_insets_v2',bottom-13,RIGHT-LEFT)
    text(c,LEFT,44,'青线追踪机身质心，铜线追踪夹爪尖端。下排叠加目标轴、实际工具轴及半径 35 mm 的容差圆。',8.15,color=GRAY)
    assert bottom>=56,(bottom,H)
    c.showPage()
    head(c,'遮挡期间怎样维持净空和对准','四排曲线共用飞行释放后的时间轴。表中把相同工况下的五种反馈方法逐趟配对。',2)
    y=pic(c,'gate_four_metrics_v2',H-94,RIGHT-LEFT)
    rule(c,y-9);y-=33
    text(c,LEFT,y,'同一工况下的五趟独立飞行',14,BOLD);y-=21
    text(c,LEFT,y,'净空取各趟全程最小值，对准误差取 52.0 s 的同一时刻。',8.5,color=GRAY);y-=20
    cols=[LEFT+4,214,306,393,RIGHT-5]
    for j,s in enumerate(('反馈方法','臂爪净空 / cm','位置误差 / cm','轴向误差 / °','整趟验收')):
        text(c,cols[j],y,s,8.1,BOLD,color=GRAY,right=bool(j))
    rule(c,y-6);y-=23
    names={'vision':'纯视觉','strain':'纯应变','fixed':'固定权重 KF','proposed':'增广融合','oracle':'真值参考'}
    for k in ('vision','strain','fixed','proposed','oracle'):
        f=flights[k];i=f.row(52);z=f.z;score=f.metrics;after=bool(z['gate_exit'][i])
        row=(names[k],f"{100*score['arm_tool_min_m']:.2f}",
             f"{100*z['tip_error'][i]:.2f}" if after else '—',
             f"{z['axis_error_deg'][i]:.2f}" if after else '—',
             ('达标' if score['success'] else ('过门后轴向超差' if k=='strain' else '出现自接触')))
        if k=='proposed':
            c.setFillColor(HexColor('#edf5f2'));c.rect(LEFT,y-5,RIGHT-LEFT,18,fill=1,stroke=0)
        for j,s in enumerate(row):text(c,cols[j],y,s,8.6,BOLD if k=='proposed' else FONT,color=TEAL if k=='proposed' else INK,right=bool(j))
        y-=20
    y-=5;rule(c,y);y-=24;text(c,LEFT,y,'从估计误差到几何结果',13.5,BOLD);y-=21
    text(c,LEFT,y,f"遮挡期，保持最后视觉姿态的弯角峰值误差为 {p.metrics['vision_hold_blackout']['theta_peak_deg']:.1f}°。",9);y-=16
    text(c,LEFT,y,f"同窗弯角 RMSE 为增广融合 {p.metrics['blackout_angle']['theta_rmse_deg']:.2f}°、固定权重 KF {flights['fixed'].metrics['blackout_angle']['theta_rmse_deg']:.2f}°、纯应变 {st.metrics['blackout_angle']['theta_rmse_deg']:.2f}°。",9);y-=24
    a=audit['minimum_separating_bounds_m']['support'][0]
    text(c,LEFT,y,f"场景几何复核覆盖 {audit['successful_flights']} 趟成功轨迹、{audit['recorded_states']:,} 个记录姿态及相邻姿态间的扫掠。",8.1,color=GRAY);y-=14
    text(c,LEFT,y,f"新增法兰支架与机器人模型的保守分离下界为 {1000*a:.1f} mm。",8.1,color=GRAY)
    assert y>36,(y,H)
    c.showPage();c.save()
    doc=pymupdf.open(name);assert len(doc)==2
    for j,page in enumerate(doc):
        assert 'M0' in page.get_text() and len(page.get_images())==(2 if j==0 else 1)
    write_json(QA/'review-verification.json',{'pages':len(doc),'sha256':sha_file(name),'paper':'A4','source_revision':'951c512',
               'figure_order':[['gate_hero_v2','gate_insets_v2'],['gate_four_metrics_v2']]})
if __name__=='__main__':main()
