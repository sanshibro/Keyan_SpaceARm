"""Paired descriptive statistics, evidence inventory and two-page review."""
import csv,hashlib,json,statistics
import numpy as np
import pymupdf
from io import BytesIO
from pathlib import Path
from PIL import Image
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas
from gatereview_pdf import font_setup
from m1_plates import ROOT,OUT,DATA,FILM,METHODS,read
from m1_mechanism import measure
from m1_execution_audit import inspect
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
        w=min(max_width,max_h*im.width/im.height);h=w*im.height/im.width
        left=x+(max_width-w)/2
        if path.with_suffix('.pdf').is_file():
            c.vector_plates.append((c.getPageNumber()-1,(left,H-top,left+w,H-top+h),path.with_suffix('.pdf')))
        else:
            im=im.convert('RGB');im.thumbnail((2100,1760))
            data=BytesIO();im.save(data,'JPEG',quality=93,subsampling=0,optimize=True);data.seek(0)
            c.drawImage(ImageReader(data),left,top-h,w,h)
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
        with np.load(DATA/case['case']/'plan/certified.npz') as tr:
            duration=float(tr['t'][-1]);spans=np.asarray(tr['dt'])
        with np.load(DATA/case['case']/'flight/control.npz') as ctrl:
            moving_window=(ctrl['phase']=='execute')&(ctrl['t']>=21)
            dt=np.diff(ctrl['t'],append=ctrl['t'][-1])
            holds={h:float(dt[moving_window&(ctrl['hold']==h)].sum()) for h in ('feedback','clearance','self','tracking')}
            last_spline=float(ctrl['spline_t'][-1])
        rows.append(dict(strategy=case['strategy'],seed=case['seed'],plan=plan['task_route'],solve_s=plan['elapsed_s'],
           planned_duration_s=duration,span_count=len(spans),shortest_span_s=float(spans.min()),longest_span_s=float(spans.max()),
           success=item['task_success'],body_cm=100*item['clearance']['body']['m'],arm_cm=100*item['clearance']['arm_tool']['m'],
           self_mm=1000*item['self_clearance_50Hz_m'],tracking_cm=100*item['peak_tracking_m'],moment_Nm=item['peak_mount_reaction_Nm'],
           thrust_rms_N=item['rms_front_rear_N'],thrust_peak_N=item['peak_front_rear_N'],tilt_deg=item['peak_tilt_deg'],
           position_cm=100*item['final_pos_m'],axis_deg=item['final_axis_deg'],optical_cm=100*item['final_optical_m'],
           dock_s=item['optical_dwell_s'],contact_steps=item['collision_steps'],end_phase=item['end_phase'],
           end_body_x_m=item['end_xyz'][0],final_spline_time_s=last_spline,body_min_pair=' / '.join(item['clearance']['body']['pair']),
           arm_min_pair=' / '.join(item['clearance']['arm_tool']['pair']),max_theta=item['max_actual_bend_deg'],
           orb_valid=item['tracked_frames'],wrist_frames=item['matching_exposures'],**{'hold_'+h+'_s':v for h,v in holds.items()}))
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
        docked=[r for r in rr if r['end_phase']=='inspect']
        sums[m]['terminal_inspection']={k:(median([r[k] for r in docked]) if docked else None) for k in ('position_cm','axis_deg','optical_cm','dock_s')}
    write(OUT/'m1_methods.json',sums);return rows,sums
def main():
    rows,s=collect();effects=measure(DATA,OUT);runs=inspect(DATA,OUT/'m1_execution_audit.json')
    paper=BytesIO();c=Canvas(paper,pagesize=A4,pageCompression=1);c.vector_plates=[]
    c.setTitle('六维全身联合规划与受限管廊检修');c.setAuthor('Aerial Continuum Manipulation')
    head(c,1,'低梁、侧管与斜置法兰','同一工业隔舱内的十五趟配对飞行，分别约束机身高度、横向让位和末端检视姿态。')
    y=fig(c,OUT/'m1_hero.png',H-94,R-L,224);line(c,y-3)
    y=fig(c,OUT/'m1_sections.png',y-10,R-L,251);line(c,y-3);y-=23
    text(c,L,y,'从外部轨迹到腕部原始曝光',12,INK,True);y-=16
    text(c,L,y,'上图各列固定机位。低梁净高 67.0 cm，侧管侵入臂的转向空间。短轨迹的终点是规划器验过净空的最后位置。',8.25);y-=10
    y=fig(c,FILM/'wrist-stations.png',y,R-L,123);y-=13
    text(c,L,y,'腕部依次看到检修窗下沿、非对称支管和法兰螺栓圈。三张图均与送入原生 ORB-SLAM3 的曝光逐帧核对。',8.15)
    y-=14;text(c,L,y,'图下分别标注曝光和交付时刻，相差 0.20 s。终端评分同时检查工具对准误差、光学工作距和法兰面深度。',8.15,GRAY)
    y-=17;text(c,L,y,'第一关的几何约束',9.0,TEAL,True)
    y-=14;text(c,L,y,'机身水平、臂下垂时，模型总高约 65.9 cm。67.0 cm 的开口只余 1.1 cm，上下两侧合计需 12.0 cm 净空。',8.15)
    assert y>=31,y
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
              f"{q['tracking_cm']:.2f}",f"{q['terminal_inspection']['position_cm']:.2f}" if d else '—',f"{q['terminal_inspection']['axis_deg']:.2f}°" if d else '—']
        for i,value in enumerate(cell):text(c,xx[i],y,value,8.25,TEAL if k=='coupled' else INK,k=='coupled','right' if i else 'left')
        y-=18
    y-=3;line(c,y);y-=15
    text(c,L,y,'距离单位 cm。净空取三趟中的最小值。求解耗时、变形期跟踪峰值取三趟中位数。末端误差取进入检视的飞行中位数。',7.95,GRAY);y-=14
    text(c,L,y,f"进入法兰检视的次数为完整法 {s['coupled']['dock']}/3、时序消融 {s['unregularized']['dock']}/3。整趟验收要求外障净空 ≥6 cm、自碰间隙为正、零接触。",8.05,GRAY);y-=14
    text(c,L,y,'末端须在 ≤3.5 cm、≤3°、12–18 cm 光学工作距内持续 1 s，并由同步深度确认看到了法兰面。',8.1,GRAY);y-=23
    full=s['coupled'];abl=s['unregularized']
    text(c,L,y,'动态耦合的配对读数',11,INK,True);y-=16
    text(c,L,y,f"完整法与消融法的臂根水平反力矩峰值中位数为 {full['moment_Nm']:.2f} / {abl['moment_Nm']:.2f} N·m，前后桨推力差 RMS 为 {full['thrust_rms_N']:.2f} / {abl['thrust_rms_N']:.2f} N。",8.25)
    y-=15;text(c,L,y,'a–d 的细点线为当前进度的认证样条，e 的铜色点线为重力偏载矩。净空按 5 Hz 绘图，全程按 50 Hz 搜索。',8.15,GRAY)
    y-=24;text(c,L,y,'按相同纵向位置比较',11,INK,True);y-=17
    j=effects[0]['strategies'];own=j['coupled'];ab=j['unregularized']
    text(c,L,y,f"代表种子中，完整法机身相对机身单独规划参考线的最大横向偏置为 {abs(own['max_signed_departure_cm']):.1f} cm，位于 x = {own['departure_x_m']:.2f} m。",8.25)
    y-=15;text(c,L,y,f"在 x = 0.80–1.92 m 的共同管段，完整法与时序消融的机身跟踪峰值为 {own['pipe_tracking_peak_cm']:.2f} / {ab['pipe_tracking_peak_cm']:.2f} cm。",8.25)
    y-=15;text(c,L,y,f"该管段完整法的弯曲幅值跨越 {own['theta_range_deg'][0]:.1f}°–{own['theta_range_deg'][1]:.1f}°，弯曲平面变化 {own['phi_range_deg']:.1f}°。",8.25)
    cuts=read(OUT/'m1_lateral_check.json')[0]['cuts']['side_riser']
    y-=15;text(c,L,y,f"恢复机身参考线的横向位置、保留其余实测姿态后，侧管段最小净空从 {100*cuts['minimum_observed_m']:.2f} 变为 {100*cuts['minimum_counterfactual_m']:.2f} cm。",8.15)
    stop=next(r for r in runs if r['case']['case']=='coupled-26112167')
    if not stop['success'] and 'self' in stop['hold']:
        y-=15;text(c,L,y,f"种子 26112167 的完整法停在 x = {stop['actual_end_x_m']:.2f} m。估计形态触发自碰守门共 {stop['hold']['self']['cumulative_s']:.1f} s，实测物理接触为零。",8.15)
    tight=[r for r in rows if r['seed']==26112173]
    cm={r['strategy']:r['body_cm'] for r in tight}
    y-=14;text(c,L,y,f"种子 26112173 的两种联合法均对准法兰。完整法与消融法通过低梁时的机身净空分别为 {cm['coupled']:.2f}、{cm['unregularized']:.2f} cm。",8.15)
    assert y>31,y
    c.showPage();c.save()
    with pymupdf.open(stream=paper.getvalue(),filetype='pdf') as doc:
        for page,rect,path in c.vector_plates:
            with pymupdf.open(path) as plate:doc[page].show_pdf_page(pymupdf.Rect(rect),plate,0)
        doc.save(OUT/'M1_两页审阅.pdf',garbage=4,deflate=True)
if __name__=='__main__':main()
