"""Read-only M2 scientific plates, auditable tables, offline review HTML and PDF.

Static evidence and actual flight evidence stay explicitly identified. All
curves use logged release time; planned, issued and independent actual shapes
are separate. No expected outcome or synthetic performance is encoded.
"""
import argparse,html,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from m2_review_data import METHODS,COLORS,load_manifest,read,sha
from m2_plant import STAGE

plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'axes.titlesize':10,'axes.labelsize':9,
 'axes.linewidth':.6,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':7,'savefig.dpi':220,
 'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
GROUPS={'A':['straight','fixed','sequential','unregularized','coupled'],'B':['stop','hard','coupled']}

def save(fig,out):
    fig.savefig(out.with_suffix('.png'),bbox_inches='tight',facecolor='white')
    fig.savefig(out.with_suffix('.svg'),bbox_inches='tight',facecolor='white');plt.close(fig)
def degrees(b):
    theta=np.rad2deg(np.linalg.norm(b,axis=1));phi=np.rad2deg(np.arctan2(b[:,1],b[:,0]));phi[theta<5]=np.nan
    jump=np.r_[False,abs(np.diff(phi))>180];phi[jump]=np.nan
    return theta,phi
def transitions(control):
    w=control['active_window'];return np.asarray(control['t'])[np.r_[False,w[1:]!=w[:-1]] & (w>=0)]

def four_curves(record,out):
    if record.series is None:raise ValueError('Independent audited series required for the four-panel plate')
    s,c=record.series,record.control;t=s['t'];tc=c['t'];fig,ax=plt.subplots(2,2,figsize=(11.6,6.4),constrained_layout=True)
    a=ax[0,0];actual=degrees(s['actual_b']);plan=c['plan_q'];planned_b=plan[:,4,None]*np.c_[np.cos(plan[:,5]),np.sin(plan[:,5])]
    desired=degrees(planned_b);issued=degrees(c['issued_b'])
    for j,(name,color) in enumerate((('theta','#276981'),('phi','#aa713f'))):
        a.plot(t,actual[j],color=color,lw=1.35,label=name+' actual')
        a.plot(tc,desired[j],color=color,lw=.7,ls='--',label=name+' planned')
        a.plot(tc,issued[j],color=color,lw=.65,ls=':',label=name+' issued')
    a.set(xlabel='Release time (s)',ylabel='Angle (deg)',title='a  Actual, planned and issued morphology');a.legend(ncol=3,loc='best');a.grid(alpha=.13)
    a=ax[0,1]
    for key,label,color in [('body','Body','#276981'),('arm','Arm / tool','#aa713f')]:a.plot(s['sample_t'],100*s['gap_'+key],label=label,color=color,lw=1.3)
    a.axhline(8.5,color='#747474',ls='--',lw=.8,label='Planning margin 8.5 cm');a.axhline(5.,color='#a24949',ls=':',lw=.9,label='Audit minimum 5.0 cm')
    if record.metrics:
        for key in ('body','arm'):
            cert=record.metrics.get('certificates',{}).get(key)
            if cert:a.scatter(cert['t_s'],100*cert['lower_m'],s=22,marker='v',c='#a24949',zorder=5)
    a.set(xlabel='Release time (s)',ylabel='External clearance (cm)',title='b  Independent native geometric clearance');a.legend(ncol=2);a.grid(alpha=.13)
    a=ax[1,0];a.plot(s['sample_t'],1000*s['gap_hazard'],color='#276981',label='Hazardous non-connection pairs')
    a.plot(s['sample_t'],1000*s['gap_fit'],color='#aa713f',label='Registered fit pairs')
    for y,ls,label in [(15,'--','Planning 15 mm'),(8,':','Projection 8 mm'),(3,'-.','Audit 3 mm')]:a.axhline(y,c='#777777',lw=.7,ls=ls,label=label)
    a.axhline(0,color='#a24949',lw=.7);a.set(xlabel='Release time (s)',ylabel='Self clearance (mm)',title='c  Hazard clearance and physical-fit separation');a.legend(ncol=2);a.grid(alpha=.13)
    a=ax[1,1];a.plot(t,100*s['tracking'],color='#276981',label='Body reference error')
    a.plot(t,100*s['terminal_position_error'],color='#aa713f',label='Tool target error')
    a.axhline(3.5,color='#777777',ls=':',lw=.8,label='Tool criterion 3.5 cm')
    b=a.twinx();b.spines['right'].set_visible(True);b.plot(t,s['axis_error'],color='#8b6585',lw=.8,label='Tool axis error');b.set_ylabel('Axis error (deg)',color='#8b6585')
    a.set(xlabel='Release time (s)',ylabel='Position error (cm)',title='d  Tracking and terminal alignment');a.legend(loc='upper left');b.legend(loc='upper right');a.grid(alpha=.13)
    for axes in ax.flat:
        for tick in transitions(c):axes.axvline(tick,c='#a8a8a8',lw=.5,alpha=.5)
    save(fig,out)

def rolling_plate(records,out):
    scored=[r for r in records if r.series is not None]
    if not scored:return None
    fig,axes=plt.subplots(4,1,figsize=(11.6,8),sharex=True,constrained_layout=True)
    for r in scored:
        s,c=r.series,r.control;t=s['t'];tc=c['t'];color=r.color
        speed=np.linalg.norm(np.gradient(s['qpos'][:,:3],t,axis=0),axis=1)
        axes[0].plot(t,speed,c=color,lw=1,label=r.label)
        # Executed local references are logged derivatives, not redifferentiated
        # target angles and not the independent truth trajectory.
        axes[1].plot(tc,np.linalg.norm(c['plan_ddq'][:,:3],axis=1),c=color,lw=.9,label=r.label)
        axes[2].plot(t,s['thrust_diff'],c=color,lw=.9,label=r.label)
        axes[3].plot(s['sample_t'],100*np.minimum(s['gap_body'],s['gap_arm']),c=color,lw=1,label=r.label)
        for tick in transitions(c):
            for a in axes:a.axvline(tick,c=color,lw=.55,alpha=.32)
    labels=[('a  Native translational motion','Speed (m/s)'),('b  Executed rolling reference','Reference acceleration (m/s2)'),
      ('c  Native rotor response','Thrust difference (N)'),('d  Independent full-body clearance','Clearance (cm)')]
    for a,(title,ylabel) in zip(axes,labels):a.set(title=title,ylabel=ylabel);a.grid(alpha=.13)
    axes[0].legend(ncol=min(3,len(scored)));axes[-1].axhline(5,color='#a24949',ls=':',lw=.8);axes[-1].set_xlabel('Release time (s); thin vertical lines = logged active-window changes')
    save(fig,out);return str(out.with_suffix('.png'))

def native_montage(record,out):
    from m2_video import Replay,contain
    from m2_scene import GATES
    from PIL import ImageDraw
    from m2_video import font
    replay=Replay([record]);x=record.physics['qpos'][:,0];moments=[]
    for gate in GATES:
        n=np.flatnonzero((x>=gate['x']) & (record.t>=record.control_summary['motion_start_s']))
        moments.append((gate['name'],float(record.t[n[0]]) if len(n) else None))
    phase=record.control['phase'];ix=np.flatnonzero(phase=='inspect')
    moments.append(('Inspection',float(record.control['t'][ix[0]]) if len(ix) else None))
    sheet=Image.new('RGB',(2560,820),'white');d=ImageDraw.Draw(sheet);evidence=[]
    for j,(label,t) in enumerate(moments):
        if t is None:
            d.text((j*640+20,340),label+': not reached in this record',font=font(21),fill='#666666');continue
        exp=record.exposure(t,require_tolerance=.026)
        if exp is None:
            d.text((j*640+20,340),'No synchronized exposure',font=font(21),fill='#666666');continue
        q=record.physics['qpos'][exp['physics_index']];sheet.paste(replay.native(q,'observer_oblique'),(j*640,50))
        with Image.open(exp['rgb_path']) as wrist:sheet.paste(contain(wrist.convert('RGB'),(640,360)),(j*640,450))
        d=ImageDraw.Draw(sheet);d.text((j*640+16,10),label+' | t = %.3f s'%exp['t'],font=font(23,True),fill=record.color)
        d.text((j*640+16,418),'Original wrist input | exposure frame %d'%exp['frame'],font=font(19),fill='#444444')
        evidence.append(dict(label=label,frame=exp['frame'],exposure_t=exp['t'],arrival_t=exp['arrival_t']))
    replay.close();sheet.save(out.with_suffix('.png'));out.with_suffix('.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')

def aggregate(records,out):
    result={}
    fields=('external_lower_m','hazard_lower_m','fit_lower_m','motion_to_inspect_s','terminal_error_m','terminal_axis_deg',
            'task_depth_dwell_s','thrust_diff_rms_N','peak_roll_pitch_reaction_Nm','position_tracking_peak_m')
    for group,methods in GROUPS.items():
        result[group]=[]
        for method in methods:
            rs=[r for r in records if r.method==method];valid=[r for r in rs if r.metrics is not None]
            summary=dict(method=method,label=METHODS[method],available_records=len(rs),audited_records=len(valid),
              successes=sum(r.metrics.get('success') is True for r in valid),seeds=[r.seed for r in valid],
              development=[bool(r.metrics.get('development')) for r in valid])
            for key in fields:
                vals=[r.metrics.get(key) for r in valid if r.metrics.get(key) is not None]
                summary[key]=dict(values=vals,median=float(np.median(vals)) if vals else None,min=float(min(vals)) if vals else None,max=float(max(vals)) if vals else None)
            result[group].append(summary)
    out.write_text(json.dumps(result,indent=2),encoding='utf-8');return result

def offline_page(out,certificate,records,plates):
    def esc(s):return html.escape(str(s))
    rows=''.join('<tr><td>'+esc(g['gate'])+'</td><td>'+('%.2f'%(1000*g['external_lower_m']))+'</td><td>'+('%.2f'%(1000*g['hazard_lower_m']))+'</td><td>'+('%.3f'%(1000*g['fit_lower_m']))+'</td><td>'+esc(str(g['theta_deg'])+' / '+str(g['phi_deg']))+'</td></tr>' for g in certificate['gate_witnesses'])
    tests=''.join('<tr><td>'+esc(r.label)+'</td><td>'+esc(r.seed)+'</td><td>'+esc(r.status())+'</td><td>'+esc('%.2f mm'%(1000*r.metrics['external_lower_m']) if r.metrics else '未完成独立核验')+'</td></tr>' for r in records)
    gallery=''.join('<figure><img src="'+esc(p.name)+'"><figcaption>'+esc(p.stem)+'</figcaption></figure>' for p in plates if p.exists())
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>M2 三关连续长廊审阅</title><style>
    :root{font-family:Georgia,"Microsoft YaHei",serif;color:#203039;background:#f6f7f7}body{margin:0}main{max-width:1220px;margin:auto;padding:36px 28px}h1{font-size:30px;margin-bottom:8px}h2{font-size:22px;margin-top:36px}p{line-height:1.7}.tag{color:#286a7a;font-size:15px}table{border-collapse:collapse;width:100%;background:white;font-variant-numeric:tabular-nums}th,td{text-align:left;border-bottom:1px solid #d2dadd;padding:12px}th{font-weight:600;color:#445a64}figure{margin:24px 0;background:white;padding:12px}img{max-width:100%;height:auto}figcaption{font-size:13px;color:#627078}code{font-size:11px;overflow-wrap:anywhere}footer{border-top:1px solid #bbc8cd;padding-top:18px;color:#657079;font-size:13px}</style><main>
    <div class="tag">M2 / 原生几何与原始日志 / 本地离线审阅</div><h1>三关连续受限长廊与滚动全身规划</h1>
    <p>上抬前折通过高位隔舱门，下潜侧弯绕过风管与错位立管，再经反向偏置门到达倾斜法兰。第一、第三人称共享同一工业几何和自然表面纹理。</p>
    <h2>静态可行见证</h2><p>下表为独立凸距离下界；单位 mm。危险非连接件与已注册物理配合对分别核验，支腿完整保留。</p>
    <table><thead><tr><th>关卡</th><th>外障净空</th><th>危险自间隙</th><th>配合间隙</th><th>θ / φ（°）</th></tr></thead><tbody>'''+rows+'''</tbody></table>
    <p>当前证据：静态外障预检 '''+('通过' if certificate['external_static_pass'] else '未通过')+'；静态自碰预检 '+('通过' if certificate['self_static_pass'] else '未通过')+'''。飞行成绩仅在独立核验文件形成后列入下表。</p>
    <h2>真实飞行记录</h2>'''+('<table><thead><tr><th>方法</th><th>种子</th><th>核验状态</th><th>全身外障下界</th></tr></thead><tbody>'+tests+'</tbody></table>' if records else '<p>本版仅包含静态几何和终端光学证据，尚未载入本轮飞行记录。</p>')+gallery+'''<footer>场景 SHA-256：<code>'''+esc(certificate['scene_sha256'])+'''</code><br>像素与独立物理姿态按曝光时刻校验；接收时刻为曝光后 0.20 s。原始文件与量化核验文件单独保存。</footer></main></html>'''
    (out/'index.html').write_text(page,encoding='utf-8')

def pdf_review(out,certificate,records,plates):
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4,landscape
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.lib.utils import ImageReader
    from reportlab.lib.colors import HexColor
    for path in ('C:/Windows/Fonts/simsun.ttc','/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'):
        if Path(path).exists():pdfmetrics.registerFont(TTFont('ReviewCN',path,subfontIndex=0));break
    else:pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'));pdfmetrics.registerFontFamily('ReviewCN',normal='STSong-Light')
    cn='ReviewCN' if 'ReviewCN' in pdfmetrics.getRegisteredFontNames() else 'STSong-Light'
    W,H=landscape(A4);c=canvas.Canvas(str(out/'M2_阶段审阅.pdf'),pagesize=(W,H));c.setTitle('M2 多窄口滚动全身规划阶段审阅')
    def text(x,y,s,size=10,color='#263941'):
        c.setFillColor(HexColor(color));c.setFont(cn,size);c.drawString(x,y,s)
    def frame(title,n):
        text(32,H-35,title,19);c.setStrokeColor(HexColor('#a8b7bc'));c.line(32,H-49,W-32,H-49);text(32,18,'M2 | 静态证据与独立核验日志 | '+certificate['scene_sha256'][:16],8);text(W-46,18,str(n),9)
    def image(path,x,y,w,h):
        if not Path(path).exists():return
        with Image.open(path) as im:iw,ih=im.size
        scale=min(w/iw,h/ih);ww,hh=iw*scale,ih*scale;c.drawImage(ImageReader(str(path)),x+(w-ww)/2,y+(h-hh)/2,ww,hh,mask='auto')
    frame('M2：三关工业长廊与同源光学场景',1)
    image(out/'three-gate-static-overview.png',32,205,540,320);image(out/'target-wrist.png',590,252,218,164)
    text(590,432,'终端腕部原生 RGB',12);opt=certificate.get('target_optics',{})
    text(590,225,'工作距离：%.1f mm'%(1000*opt.get('optical_depth_m',float('nan'))),11)
    text(590,205,'中心深度：%.1f mm'%(1000*opt.get('native_depth_at_face_m',float('nan'))),11)
    text(38,185,'三个独立静态姿态：G1 高位前折 → G2 下潜侧弯与支管避让 → G3 反向偏置门',11)
    text(38,157,'三关独立几何预检（有限搜索可行见证；距离为保守下界）',12)
    headers=['关卡','机身 z / m','θ / φ / °','外障 / mm','危险自碰 / mm','配合 / mm'];xs=[40,120,245,370,490,650]
    for x,h in zip(xs,headers):text(x,133,h,10)
    for j,g in enumerate(certificate['gate_witnesses']):
        vals=[g['gate'],'%.2f'%g['xyz'][2],'%g / %g'%(g['theta_deg'],g['phi_deg']),'%.2f'%(1e3*g['external_lower_m']),'%.2f'%(1e3*g['hazard_lower_m']),'%.3f'%(1e3*g['fit_lower_m'])]
        for x,v in zip(xs,vals):text(x,111-21*j,v,11)
    c.showPage();frame('M2：验收口径与本轮实证状态',2)
    text(38,H-78,'外障：规划 85 mm；静态见证 105 mm；先导 62 mm；正式 50 mm。',12)
    text(38,H-104,'危险非连接件：规划 15 mm；主动投影 8 mm；先导 5 mm；正式 3 mm。',12)
    text(38,H-130,'注册配合对：单列间隙并保持零穿透；夹指张开指令与全部支腿碰撞对保留。',11)
    text(38,H-156,'末端：12-18 cm 工作距，位置误差 ≤ 3.5 cm，轴误差 ≤ 3°，持续 ≥ 1 s。',11)
    if records:
        text(38,H-191,'真实记录状态（pass / fail 直接读取独立核验 metrics.json）',12)
        y=H-218
        for r in records:
            msg=r.label+' | seed '+str(r.seed)+' | '+r.status()
            if r.metrics:msg+=' | min external %.2f mm'%(1000*r.metrics['external_lower_m'])
            text(42,y,msg,10);y-=23
            if y<190:break
    else:
        text(38,H-193,'本版尚未载入飞行记录，当前完成项是三关静态可行性与终端光学预检。',12)
        text(38,H-217,'后续同页读取原始日志的独立核验结果；失败、未到达和未核验记录均保留。',11)
    launch=certificate.get('launch_preflight')
    if launch:
        text(38,335,'起飞准备：%d 个静态采样，零物理接触；平移路径外障下界 %.2f mm。'%(launch['sample_count'],1000*launch['interpolation_external_lower_m']),11)
        text(38,311,'低入口甲板 x ≤ 0.60 m，顶面 0.55 m；主甲板顶面 1.25 m，三关几何不变。',10)
    image(out/'gate1-closeup.png',38,44,370,200);image(out/'gate3-closeup.png',425,44,380,200)
    text(38,260,'同尺度近景：G1',10);text(425,260,'同尺度近景：G3',10)
    for n,p in enumerate([p for p in plates if p.name.endswith('-curves.png') or p.name=='rolling-comparison.png'],3):
        c.showPage();frame('M2：实际飞行曲线 / '+p.stem,n);image(p,28,45,W-56,H-110)
    c.save()

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path);p.add_argument('--output',type=Path,default=STAGE/'review');p.add_argument('--pdf',action='store_true');p.add_argument('--verify-pixels',action='store_true');v=p.parse_args()
    out=v.output;out.mkdir(parents=True,exist_ok=True);cert=read(STAGE/'qa/static/certified.json');records=load_manifest(v.manifest) if v.manifest else []
    plates=[]
    for name in ('three-gate-static-review.png','three-gate-static-overview.png','gate1-side.png','gate2-oblique.png','gate3-side.png','gate1-closeup.png','gate2-closeup.png','gate3-closeup.png','target-wrist.png','target-depth-mm.png','target-optics.json','certified.json'):
        src=STAGE/'qa/static'/name
        if src.exists():shutil.copy2(src,out/name)
    plates.extend(out/name for name in ('three-gate-static-review.png','target-wrist.png'))
    for r in records:
        stem=r.method+'-'+str(r.seed)
        if v.verify_pixels:(out/(stem+'-exposure-audit.json')).write_text(json.dumps(r.verify_all_exposures(),indent=2),encoding='utf-8')
        if r.series is not None:four_curves(r,out/(stem+'-curves'));plates.append(out/(stem+'-curves.png'))
        native_montage(r,out/(stem+'-native-montage'));plates.append(out/(stem+'-native-montage.png'))
    if records:aggregate(records,out/'group-tables.json');rolling_plate(records,out/'rolling-comparison');plates.append(out/'rolling-comparison.png')
    offline_page(out,cert,records,plates)
    if v.pdf:pdf_review(out,cert,records,plates)
    provenance=dict(scene_sha256=cert['scene_sha256'],registry_sha256=cert['registry_sha256'],flight_count=len(records),
      static_only=not bool(records),records=[dict(flight=str(r.root),audit=str(r.audit),status=r.status(),seed=r.seed,method=r.method) for r in records],
      authoring_sources={p.name:sha(p) for p in Path(__file__).parent.glob('m2_*') if p.name in ('m2_plates.py','m2_video.py','m2_review_data.py')})
    (out/'review-provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8');print(json.dumps(provenance))
if __name__=='__main__':main()
