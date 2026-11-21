"""Paired descriptive statistics, evidence inventory and two-page review."""
import csv,hashlib,json,statistics
from io import BytesIO
from pathlib import Path
from PIL import Image
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas
from gatereview_pdf import font_setup
from m1_plates import ROOT,OUT,DATA,FILM,METHODS,read
NAMES=['直臂','固定形态','先机身后机械臂','联合几何（无耦合时序）','完整六维联合']
W,H=A4
L,R=33,W-33
TEAL='#126d64';INK='#233a38';GRAY='#526c67';RULE='#c7d0cb'
REG,BOLD,_=font_setup()
def write(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def median(xs):return float(statistics.median(xs))
def text(c,x,y,s,size=8.5,col=INK,bold=False,align='left'):
    c.setFont(BOLD if bold else REG,size);c.setFillColor(HexColor(col))
    {'left':c.drawString,'right':c.drawRightString,'centre':c.drawCentredString}[align](x,y,str(s))
def line(c,y):
    c.setStrokeColor(HexColor(RULE));c.setLineWidth(.55);c.line(L,y,R,y)
def fig(c,path,top,max_width,max_h,x=L):
    with Image.open(path) as im:
        im=im.convert('RGB');im.thumbnail((2100,1760))
        w=min(max_width,max_h*im.width/im.height);h=w*im.height/im.width
        data=BytesIO();im.save(data,'JPEG',quality=93,subsampling=0,optimize=True);data.seek(0)
    c.drawImage(ImageReader(data),x+(max_width-w)/2,top-h,w,h)
    return top-h
def head(c,p,title,sub):
    text(c,L,H-23,'M1    工业管廊检修    /    六维全身联合规划',8,TEAL,True)
    text(c,R,H-23,'SUPER  ·  Qingyu  ·  PX4 SITL',8,GRAY,align='right')
    text(c,L,H-55,title,21,INK,True);text(c,L,H-75,sub,9.0,GRAY);line(c,H-86)
    line(c,26);text(c,L,14,'配对闭环飞行   ·   2026 年 11 月 21 日',7.4,GRAY)
    text(c,R,14,f'{p} / 2',8,TEAL,align='right')
def collect():
    rows=[]
    for item in read(DATA/'matrix.json'):
        case=item['case'];plan=read(DATA/case['case']/'plan/decision.json')
        rows.append(dict(strategy=case['strategy'],seed=case['seed'],plan=plan['task_route'],solve_s=plan['elapsed_s'],
           success=item['task_success'],body_cm=100*item['clearance']['body']['m'],arm_cm=100*item['clearance']['arm_tool']['m'],
           self_mm=1000*item['self_clearance_50Hz_m'],tracking_cm=100*item['peak_tracking_m'],moment_Nm=item['peak_mount_reaction_Nm'],
           thrust_rms_N=item['rms_front_rear_N'],thrust_peak_N=item['peak_front_rear_N'],tilt_deg=item['peak_tilt_deg'],
           position_cm=100*item['final_pos_m'],axis_deg=item['final_axis_deg'],optical_cm=100*item['final_optical_m'],
           dock_s=item['optical_dwell_s'],contact_steps=item['collision_steps'],end_phase=item['end_phase'],
           max_theta=item['max_actual_bend_deg'],orb_valid=item['tracked_frames'],wrist_frames=item['matching_exposures']))
    assert len(rows)==15 and len({(r['strategy'],r['seed']) for r in rows})==15
    with (OUT/'m1_paired_flights.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    sums={}
    for m in METHODS:
        rr=sorted((r for r in rows if r['strategy']==m),key=lambda r:r['seed'])
        assert len(rr)==3
        sums[m]={k:median([r[k] for r in rr]) for k in ['solve_s','tracking_cm','moment_Nm','thrust_rms_N','thrust_peak_N','tilt_deg','position_cm','axis_deg','optical_cm','dock_s']}
        sums[m].update(success=sum(r['success'] for r in rr),min_body=min(r['body_cm'] for r in rr),min_arm=min(r['arm_cm'] for r in rr),
                       min_self=min(r['self_mm'] for r in rr),dock=sum(r['end_phase']=='inspect' for r in rr),
                       gain=sum(r['orb_valid'] for r in rr),frames=sum(r['wrist_frames'] for r in rr))
    write(OUT/'m1_methods.json',sums);return rows,sums
def main():
    rows,s=collect();c=Canvas(str(OUT/'M1_两页审阅.pdf'),pagesize=A4,pageCompression=1)
    c.setTitle('六维全身联合规划与受限管廊检修');c.setAuthor('Aerial Continuum Manipulation')
    head(c,1,'低梁、侧管与斜置法兰','同一工业隔舱内的十五趟配对飞行，分别约束机身高度、横向让位和末端检视姿态。')
    y=fig(c,OUT/'m1_hero.png',H-94,R-L,224);line(c,y-3)
    y=fig(c,OUT/'m1_sections.png',y-10,R-L,205);line(c,y-4);y-=27
    text(c,L,y,'从几何到观测',13,INK,True);y-=22
    text(c,L,y,'机身与工具轨迹取自同一趟飞行。低梁净高 67.0 cm，非对称支管进入臂的转向空间，法兰位于回旋区末端。',8.45);y-=15
    text(c,L,y,'各列近景固定相机和取景范围。基线飞到最后一段已验证样条时悬停，画面中的 x 是实际到达位置。',8.45);y-=22
    src=FILM/'flange-coupled-wrist.png';used=read(FILM/'sections.json')['flange']['coupled']
    fig(c,src,y,121,86,L)
    text(c,L+134,y-15,'腕部真实输入',11,TEAL,True)
    text(c,L+134,y-33,'曝光 %.2f s，交付 %.2f s；图像与全景共用场景、光轴和飞行姿态。'%(used['t'],used['arrival']),8.0)
    text(c,L+134,y-49,'每趟在 50 Hz 状态上核算工具位置、轴向和光学工作距，再核对相应深度图。',8)
    text(c,L+134,y-65,'配对试验均保留原生 ULog、ROS bag、腕部 RGB-D、物理状态及控制指令。',8)
    assert y-87>=31,y
    c.showPage();head(c,2,'机身让位与弯臂时序如何改变飞行','代表种子 26112161。实线和虚线对应独立飞行。曲线横轴从规划运动开始时刻计时。')
    y=fig(c,OUT/'m1_curves.png',H-90,R-L,348);line(c,y-2);y-=23
    text(c,L,y,'三个配对种子的实测结果',13,INK,True);y-=19
    xx=[L,L+151,L+201,L+247,L+291,L+365,L+440,R]
    labels=['策略','成功','求解 / s','机身净空','臂爪净空','机身偏差','末端位置','轴误差']
    for i,lab in enumerate(labels):text(c,xx[i],y,lab,7.9,GRAY,True,'right' if i else 'left')
    line(c,y-6);y-=22
    for k,n in zip(METHODS,NAMES):
        q=s[k];d=q['dock']>0
        if k=='coupled':
            c.setFillColor(HexColor('#e9f2ef'));c.rect(L-3,y-4,R-L+6,17,fill=1,stroke=0)
        cell=[n,f"{q['success']}/3",f"{q['solve_s']:.1f}",f"{q['min_body']:.2f}",f"{q['min_arm']:.2f}",
              f"{q['tracking_cm']:.2f}",f"{q['position_cm']:.2f}" if d else '—',f"{q['axis_deg']:.2f}°" if d else '—']
        for i,value in enumerate(cell):text(c,xx[i],y,value,8.25,TEAL if k=='coupled' else INK,k=='coupled','right' if i else 'left')
        y-=18
    y-=3;line(c,y);y-=15
    text(c,L,y,'距离单位 cm。净空取三趟全程最小值；求解、变形期跟踪峰值、末秒对准误差取三趟中位数。',8.1,GRAY);y-=14
    text(c,L,y,'横线表示飞行未进入法兰检视。整趟验收同时要求外障净空 ≥6 cm、自碰间隙为正、物理接触次数为零。',8.1,GRAY);y-=14
    text(c,L,y,'末端须在 ≤3.5 cm、≤3°、12–18 cm 光学工作距内持续 1 s，并由同步深度确认看到了法兰面。',8.1,GRAY);y-=23
    full=s['coupled'];abl=s['unregularized']
    text(c,L,y,'动态耦合的配对读数',11,INK,True);y-=16
    text(c,L,y,f"完整法与消融法的臂根水平反力矩峰值中位数为 {full['moment_Nm']:.2f} / {abl['moment_Nm']:.2f} N·m，前后桨推力差 RMS 为 {full['thrust_rms_N']:.2f} / {abl['thrust_rms_N']:.2f} N。",8.25)
    y-=15;text(c,L,y,'灰色虚线为时序消融；铜色点线为重力偏载矩；净空面板按 5 Hz 绘图，全程最小值按 50 Hz 搜索。',8.15,GRAY)
    assert y>31,y
    c.showPage();c.save()
if __name__=='__main__':main()
