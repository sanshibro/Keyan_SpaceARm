"""Render the two-page F+ review from the archived audit, traces, and fixed-camera stills."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas

W,H=638,818
INK='#253a43'; GRAY='#4f6169'; LINE='#cad4d5'; LIGHT='#e3e9e8'
C={'straight':'#76848a','nominal':'#315f85','fastbend':'#a14d2e','slow':'#17746b'}
FONT_DIR=Path(os.environ.get('FP_REPORT_FONTS',r'C:\Windows\Fonts'))
for key,path in [('CN','msyh.ttc'),('CNB','msyhbd.ttc'),('Serif','times.ttf')]:
    pdfmetrics.registerFont(TTFont(key,str(FONT_DIR/path)))

def txt(c,x,y,s,size=10,color=INK,font='CN',align='l'):
    c.setFont(font,size);c.setFillColor(HexColor(color))
    {'l':c.drawString,'r':c.drawRightString,'c':c.drawCentredString}[align](x,H-y,s)

def rule(c,x,y,x2,y2,color=LINE,width=.55,dash=None):
    c.saveState();c.setStrokeColor(HexColor(color));c.setLineWidth(width)
    if dash:c.setDash(dash)
    c.line(x,H-y,x2,H-y2);c.restoreState()

def head(c,page,title,sub):
    txt(c,40,28,'SUPER  /  QINGYU     空场闭环飞行试验',9,GRAY)
    txt(c,598,28,'2026.11.19    ·    F+    /    %02d'%page,9,GRAY,align='r')
    rule(c,40,39,598,39,INK,.8)
    txt(c,40,79,title,25,INK,'CNB')
    txt(c,41,102,sub,10,GRAY)

def foot(c,n):
    rule(c,40,778,598,778)
    txt(c,40,797,'组会审阅    /    固定空场 · 标称空载 · 软件在环',8,GRAY)
    txt(c,598,797,'%02d   /   02'%n,8,GRAY,align='r')

def leg(c,x,y,entries):
    for key,label in entries:
        rule(c,x,y-3,x+12,y-3,C[key],1.8,[3,2] if key=='straight' else None)
        txt(c,x+16,y,label,8.2,C[key]); x+=66

def plot(c,box,traces,keys,zero,end,column,low,high,yt,xt,measure):
    x,top,w,h=box
    left=x+30;right=x+w-5;bt=top+h
    px=lambda t:left+(right-left)*t/end
    py=lambda v:bt-h*(v-low)/(high-low)
    for v in yt:
        yy=py(v);rule(c,left,yy,right,yy,LIGHT,.45)
        f=('%g'%v) if abs(v)>=1 or v==0 else ('%.2f'%v)
        txt(c,left-5,yy+2.5,f,8,GRAY,align='r')
    for t in xt:
        xx=px(t);rule(c,xx,bt,xx,bt+3,GRAY);txt(c,xx,bt+13,'%g'%t,8,GRAY,align='c')
    rule(c,left,bt,right,bt,GRAY,.65);rule(c,left,top,left,bt,GRAY,.65)
    if low<0:rule(c,left,py(0),right,py(0),'#abb9ba',.6)
    for key in keys:
        d=traces[key];sel=(d.s>=zero)&(d.s<=zero+end)
        t=d.s[sel].to_numpy()-zero;v=d[column][sel].to_numpy()
        c.saveState();c.setStrokeColor(HexColor(C[key]));c.setLineWidth(1.5 if key!='straight' else 1)
        if key=='straight':c.setDash([3,2])
        path=c.beginPath();path.moveTo(px(t[0]),H-py(v[0]))
        for ti,vi in zip(t[1:],v[1:]):path.lineTo(px(ti),H-py(vi))
        c.drawPath(path);c.restoreState()
    txt(c,left,top-7,measure,8,GRAY)
    txt(c,(left+right)/2,bt+27,'动作开始后的时间 (s)',8,GRAY,align='c')

def pair_image(c,root,x,y,key,variant1,variant2):
    im=Image.open(root/'review'/(key+'.png')).convert('RGB')
    for j,(col,title) in enumerate([(456,variant1),(889,variant2)]):
        crop=im.crop((col,105,col+416,521))
        c.drawImage(ImageReader(crop),x+10+j*129,H-y-125,width=125,height=125)
        txt(c,x+72+j*129,y+135,title,8,GRAY,align='c')

def table(c,y,headings,rows,widths):
    start=40;cent=[];left=start
    for w in widths:cent.append(left+w/2);left+=w
    rule(c,40,y-15,598,y-15,INK,.8)
    for x,v in zip(cent,headings):txt(c,x,y,v,8.7,INK,'CNB','c')
    rule(c,40,y+7,598,y+7)
    for i,row in enumerate(rows):
        pos=y+23+i*20
        if i==len(rows)-1:rule(c,40,pos-13,598,pos-13,LINE,.7)
        for x,v in zip(cent,row):txt(c,x,pos,v,9,INK,'CNB' if i==len(rows)-1 else 'CN','c')
    rule(c,40,pos+8,598,pos+8,INK,.65)

def main(root):
    m=pd.read_csv(root/'metrics.csv');a=json.loads((root/'audit.json').read_text(encoding='utf8'))
    assert a['all_integrity_checks'] and a['all_pairs_matched']
    med=m.groupby('variant').median(numeric_only=True);seed=241118
    tr={k:pd.read_csv(root/'traces'/('%s-%s.csv'%(seed,k))) for k in C}
    review=root/'review';c=Canvas(str(review/'review.pdf'),pagesize=(W,H),pageCompression=1)
    c.setTitle('机臂速度与定点稳定性 | F+ 闭环试验与真实性审查')
    head(c,1,'弯臂和收臂速度如何改变机身偏移','同一悬停定点、同一飞控和臂形路径。三组配对噪声，每组四次独立启动 PX4。')
    txt(c,40,138,'a   主动弯到 118°',15,INK,'CNB');txt(c,329,138,'b   从 118° 收直',15,INK,'CNB')
    txt(c,40,157,'动作开始后 4 秒 · 同一世界机位',8.7,GRAY)
    txt(c,329,157,'动作开始后 4 秒 · 同一世界机位',8.7,GRAY)
    pair_image(c,root,42,161,'bend','常速弯臂  /  6 s','快速弯臂  /  3 s')
    pair_image(c,root,330,161,'return','常速收臂  /  4 s','慢速收臂  /  8 s')
    rule(c,319,124,319,298,LIGHT)
    leg(c,52,325,[('straight','直臂'),('nominal','6 秒弯臂'),('fastbend','3 秒弯臂')])
    leg(c,341,325,[('straight','直臂'),('nominal','4 秒收臂'),('slow','8 秒收臂')])
    plot(c,(40,351,264,102),tr,['straight','nominal','fastbend'],16,8,'error_cm',0,30,[0,10,20,30],[0,2,4,6,8],'三维定点误差 (cm)')
    plot(c,(329,351,264,102),tr,['straight','nominal','slow'],24,11,'error_cm',0,30,[0,10,20,30],[0,2,4,6,8,10],'三维定点误差 (cm)')
    txt(c,40,504,'上方曲线来自预先选定的种子 241118。十字为设定位置，模型姿态取自物理引擎积分状态。',8.7,GRAY)
    txt(c,40,538,'各次动作的最大三维定点误差',14,INK,'CNB')
    txt(c,598,538,'单位  cm',9,GRAY,align='r')
    rows=[]
    for seed_id in sorted(m.seed.unique()):
        v=m[m.seed==seed_id].set_index('variant')
        vals=(v.loc['nominal','out_peak_cm'],v.loc['fastbend','out_peak_cm'],v.loc['nominal','ret_peak_cm'],v.loc['slow','ret_peak_cm'])
        rows.append([str(seed_id)]+['%.2f'%n for n in vals])
    vals=(med.loc['nominal','out_peak_cm'],med.loc['fastbend','out_peak_cm'],med.loc['nominal','ret_peak_cm'],med.loc['slow','ret_peak_cm'])
    rows.append(['中位数']+['%.2f'%n for n in vals])
    table(c,573,['配对种子','6 秒弯臂','3 秒弯臂','4 秒收臂','8 秒收臂'],rows,[106,113,113,113,113])
    txt(c,40,685,'统计窗',9.6,INK,'CNB')
    txt(c,93,685,'弯臂包含各自动作结束后的 2 秒，收臂包含各自动作结束后的 3 秒。',9.6)
    txt(c,40,709,'速度代价',9.6,INK,'CNB')
    txt(c,93,709,'慢收臂每次追加 4 秒。两个方向另用相同绝对观察窗重算，三个种子的效应方向一致。',9.6)
    txt(c,40,738,'直臂对照按常速收臂同一时窗评分，三次峰值中位数 %.2f cm。'%med.loc['straight','ret_peak_cm'],9,GRAY)
    foot(c,1);c.showPage()
    head(c,2,'机臂耦合与仿真可信度审查','俯仰方向。下图为臂组件施加给机身的安装座力矩，减去同姿态下臂组件的准静态重力矩。')
    txt(c,40,136,'a   弯臂过渡',13,INK,'CNB');txt(c,329,136,'b   收臂过渡',13,INK,'CNB')
    leg(c,50,158,[('nominal','6 秒弯臂'),('fastbend','3 秒弯臂')]);leg(c,337,158,[('nominal','4 秒收臂'),('slow','8 秒收臂')])
    plot(c,(40,184,264,88),tr,['nominal','fastbend'],16,8,'mount_residual_Nm',-.08,.08,[-.08,0,.08],[0,2,4,6,8],'安装座力矩残差 (N·m)')
    plot(c,(329,184,264,88),tr,['nominal','slow'],24,11,'mount_residual_Nm',-.08,.08,[-.08,0,.08],[0,2,4,6,8,10],'安装座力矩残差 (N·m)')
    txt(c,40,321,'虚拟力矩传感器由日志离线重建。已用 2 kg、0.10 m 的静载校核方向与幅值（1.962 N·m）。',9.2,GRAY)
    txt(c,40,339,'该残差包含机身和臂的加速度效应；各桨余量按标称 0–12 N 模型计算。',9.2,GRAY)
    txt(c,40,375,'哪些结论有证据，哪些参数还需实机核准',15,INK,'CNB')
    lineitems=[
       ('飞控及日志','原生 PX4 / EKF2 闭环运行。四路 ULog 输出与 HIL 下发及 MuJoCo 施加量逐项核对，',
        '12 次全部通过，原生 ULog 记录 0 个丢包事件。飞行阶段均保持解锁与 OFFBOARD。'),
       ('形态与载荷','SUPER 和 Qingyu 几何配准，机身自由运动，8 对弯曲铰积分。最大累计弯曲 %.2f°。'%m.curve_max_deg.max(),
        '标称总质量 2.25 kg，未加额外负载。臂刚度、阻尼、传动滞回与执行速率仍待台架测量。'),
       ('电机与环境','四路实际飞控指令施加到模型；两类动作中最小单桨余量 %.2f N，硬限幅持续 0 ms。'%min(m.out_margin_N.min(),m.ret_margin_N.min()),
        '推力按线性映射即时施加。电机响应、台架推力曲线、桨流和风扰尚待标定。'),
       ('传感与任务','合成惯导、气压、磁力计和理想化 GPS 进入估计器。该轮固定空场，在三个噪声种子上比较。',
        '已知地图单门、多门接管、相机与雷达在线建图、应变闭环及实飞仍在后续阶段。')]
    y=401
    for lab,l1,l2 in lineitems:
        rule(c,40,y-9,598,y-9,LINE)
        txt(c,40,y+9,lab,10,INK,'CNB');txt(c,117,y+9,l1,9);txt(c,117,y+26,l2,9,GRAY);y+=48
    rule(c,40,y-9,598,y-9,LINE)
    txt(c,40,626,'对 G1 单门实验的决定',15,INK,'CNB')
    txt(c,40,653,'保留当前飞控。用同一张已知地图、同一门后末端目标，比较原时序和受净空约束的平滑时序。',10)
    txt(c,40,673,'正式比较匹配总任务时间或报告时间代价。分别记录机身、臂和工具的实际最小净空与到位误差。',10)
    txt(c,40,704,'实物衔接',9.5,INK,'CNB')
    txt(c,99,704,'优先测臂形指令与独立观测、安装座力矩、电机推力与时延，随后重复同一空场动作。',9.2)
    txt(c,40,731,'可支持的结论限于标称模型中动作速度与过渡偏差的关系。实际门洞净空及实机速度限值另行验证。',9.1,GRAY)
    foot(c,2);c.save()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('round',type=Path);main(parser.parse_args().round)
