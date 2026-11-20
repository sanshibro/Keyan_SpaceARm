"""Build and publish the group-meeting record for the completed single-gate cohort."""
import argparse
import csv
import hashlib
import html
import json
import re
import shutil
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw, ImageFont

TAG = "2026-11-20-gate-feedback"
PREVIOUS = "2026-11-20-augmented-fusion"
TITLE = "冻结形态反馈下的受限过门与斜向精细对准"
RAW = "https://raw.githubusercontent.com/sanshibro/Keyan_SpaceARm/"
CDN = "https://cdn.jsdelivr.net/gh/sanshibro/Keyan_SpaceARm@"
NAMES = {"vision":"纯视觉","strain":"纯应变","fixed":"固定噪声 KF","proposed":"增广融合","oracle":"真值参考"}
MAT = {"nominal":"基础迟滞","differential":"迟滞与差模偏移"}
VIEWS = {"natural":"清晰","occluded":"动态遮挡"}


def read(path):
    return json.loads(path.read_bytes())


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes((json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode())


def digest(path):
    with path.open('rb') as f:
        h=hashlib.sha256()
        for b in iter(lambda:f.read(1<<20),b''):
            h.update(b)
    return h.hexdigest()


def gen(stage,action,dist,*opts):
    subprocess.run([sys.executable,str(stage/'site-cache/tools/meeting_archive.py'),action,'--site',str(dist),*map(str,opts)],check=True)


def confirm(url,path):
    expected=digest(path)
    error=None
    for attempt in range(4):
        try:
            h=hashlib.sha256();n=0
            with urlopen(Request(url,headers={'User-Agent':'Aerial-Continuum-Group/1.0'}),timeout=40) as stream:
                for b in iter(lambda:stream.read(1<<20),b''):
                    h.update(b);n+=len(b)
            if expected!=h.hexdigest() or n!=path.stat().st_size:
                raise ValueError('Public revision does not match: '+path.name)
            return {'external_url':url,'bytes':n,'sha256':expected}
        except (OSError,ValueError) as e:
            error=e
            time.sleep(2+attempt)
    raise error


def summary_text(groups):
    v=groups['differential-occluded']
    p,s,f = (v[m] for m in ('proposed','strain','fixed'))
    return (f"进门前形成差模偏移，过门途中完全失视 4 秒。每种反馈源在相同的三个种子上各飞一趟。"
            f"增广融合完成 {p['passed']}/3 趟，纯应变完成 {s['passed']}/3 趟，固定噪声 KF 完成 {f['passed']}/3 趟。"
            f"三种子下，增广融合与纯应变的门后工具轴向 RMSE 中位数分别为 "
            f"{p['metrics']['axis_rmse_deg']['median']:.2f}° 和 {s['metrics']['axis_rmse_deg']['median']:.2f}°。")


def status_text(m):
    if m['success']:return '安全通过并对准'
    if not m['crossed']:return '门前保持'
    if not m['safe'] and not m['dwell_reached']:return '安全与对准未达标'
    if not m['safe']:return '已对准，安全未达标'
    return '已过门，对准未达标'


def table_rows(data,raw_url):
    rows=[]
    for m in sorted(data['cases'].values(),key=lambda v:(v['case']['material'],v['case']['vision'],v['case']['seed'],
                                                         list(NAMES).index(v['case']['method']))):
        c=m['case']; e=data['evidence'][c['case']]
        rows.append(dict(case=c['case'],seed=c['seed'],material=MAT[c['material']],vision=VIEWS[c['vision']],
                         method=NAMES[c['method']],result=status_text(m),body_cm=round(100*m['body_min_m'],3),
                         arm_cm=round(100*m['arm_tool_min_m'],3),self_mm=round(1000*m['self_min_m'],3),
                         hold_s=round(m['hold_s'],3),axis_rmse_deg=round(m['axis_rmse_deg'],3),
                         filename=e['file'],bytes=e['bytes'],sha256=e['sha256'],download=raw_url+e['file']))
    return rows


STYLES="""body{color:#233a43;background:white;font:15px/1.65 system-ui,'Microsoft YaHei',sans-serif;margin:24px auto 65px;max-width:1200px;padding:0 22px}
h1,h2{font-family:Georgia,'Songti SC','SimSun',serif;color:#1c3841}h1{font-size:clamp(24px,3.5vw,38px);margin:12px 0 7px}
a{color:#16615f;text-underline-offset:3px}.nav,.note{color:#5b6d70}details{margin:20px 0;border-top:1px solid #c7d5d5;padding-top:9px}summary{cursor:pointer;font-weight:600}
.scroll{overflow:auto}table{width:100%;font-size:13px;font-variant-numeric:tabular-nums;border-collapse:collapse}th,td{text-align:right;padding:5px 8px;white-space:nowrap;border-bottom:1px solid #dfe6e6}
th{color:#35585c;background:#f2f5f5}th:first-child,td:first-child,td:nth-child(4){text-align:left}
img,video{display:block;max-width:100%;height:auto;margin:12px auto}section{border-top:1px solid #b6c8c6;margin:24px 0;padding-top:4px}
code{font-size:11px}footer{font-size:13px;color:#5b6d70;margin-top:28px}"""


def flight_page(rows,local,lead):
    esc=html.escape
    nav=('<a href="review/gate_review.pdf">两页审阅 PDF</a>　·　<a href="review/gate_multiview_comparison.mp4">多机位视频</a>　·　'
         '<a href="review/all_trials_metrics.csv">逐趟指标</a>　·　<a href="组会离线包.zip">组会离线包</a>') if local else (
         '<a href="index.html">本轮实验图与视频</a>　·　<a href="../index.html">历次实验</a>')
    out=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
         '<title>单门闭环飞行原始记录</title><style>'+STYLES+'</style><main><nav class="nav">'+nav+'</nav>',
         '<h1>单门闭环飞行原始记录</h1><p>'+esc(lead)+'</p>',
         '<p>每趟使用相同的二维形态外环和安全守门逻辑。每份 ZIP 收录原生 PX4 ULog、ROS bag、MuJoCo 状态、视觉与应变观测、逐时刻净空以及成员文件指纹。</p>']
    if local:
        out+=['<section><h2>同视角过门与斜向对准</h2><a href="review/gate_keyframes.png"><img src="review/gate_keyframes.png" width="1050" alt="五种独立反馈条件下的物理关键帧"></a></section>',
              '<section><h2>角度、净空与末端误差</h2><a href="review/gate_four_metrics.png"><img src="review/gate_four_metrics.png" width="1000" alt="四条同步指标"></a></section>',
              '<section><h2>侧视、斜视与腕部 RGB</h2><video controls playsinline preload="metadata" poster="review/gate_keyframes.png" src="review/gate_multiview_comparison.mp4"></video></section>']
    for material in MAT.values():
        for vision in VIEWS.values():
            out.append('<details open><summary>'+esc(material+' · '+vision)+'</summary><div class="scroll"><table><thead><tr>'+
                       ''.join('<th>'+h+'</th>' for h in ('配对种子','方法','状态','机身 / 臂爪 cm','自距 mm','停顿 s','轴向 RMSE °','数据与校验'))+'</tr></thead><tbody>')
            for r in rows:
                if (r['material'],r['vision'])!=(material,vision):continue
                remote=esc(r['download'],quote=True)
                link=('<a href="evidence/'+esc(r['filename'],quote=True)+'">本地 ZIP</a> · ') if local else ''
                fields=(str(r['seed']),r['method'],r['result'],f"{r['body_cm']:.2f} / {r['arm_cm']:.2f}",f"{r['self_mm']:.2f}",f"{r['hold_s']:.2f}",f"{r['axis_rmse_deg']:.2f}")
                out.append('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in fields)+'<td>'+link+'<a href="'+remote+'">下载 ZIP</a> <code title="'+r['sha256']+'">'+r['sha256'][:8]+'</code></td></tr>')
            out.append('</tbody></table></div></details>')
    return ''.join(out)+'<footer>末端稳态窗为 49–55 s。通过要求机身和臂爪净空均达到 6.0 cm，工具位置与轴向误差同时保持在 3.5 cm 和 3.0° 以内满 1.0 s。</footer></main></html>'


def build(stage,evidence_commit):
    if not re.fullmatch('[0-9a-f]{40}',evidence_commit):raise ValueError('Full evidence revision required')
    root=stage/'canonical-release';src=root/'sources';src.mkdir(parents=True,exist_ok=True)
    rd=stage/'review';d=stage/'decisions';cohort=read(d/'cohort-summary.json')
    if cohort['trials']!=60 or cohort['cases_with_verified_raw_members']!=60:raise ValueError('Complete audited cohort required')
    native=read(d/'native-log-audit.json')
    if native['trials']!=60 or set(native['cases'])!=set(read(d/'metrics.json')['cases']):
        raise ValueError('The native motor/shape audit must cover the same sixty flights')
    if read(rd/'gate_review_manifest.json')['pages']!=2:raise ValueError('Expected two review pages')
    conditions=cohort['conditions'];lead=summary_text(conditions);raw=RAW+evidence_commit+'/revision/data/round_gate_feedback/evidence/'
    rows=table_rows(read(d/'metrics.json'),raw)
    with (rd/'evidence-index.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (root/'flights.html').write_bytes(flight_page(rows,False,lead).encode())
    (stage/'组会预览.html').write_bytes(flight_page(rows,True,lead).encode())
    for original,name,width in [('gate_keyframes','keyframes',2050),('gate_four_metrics','curves',2000),('gate_paired_outcomes','paired',1800),
                                ('gate_method_details','methods',2300),('gate_state_barcode','states',1600)]:
        with Image.open(rd/(original+'.png')) as im:
            im=im.convert('RGB');im.thumbnail((width,1600),Image.Resampling.LANCZOS);im.save(src/(name+'.webp'),quality=92,method=6)
    from matplotlib.font_manager import findfont
    overview=Image.new('RGB',(520,211),'white');pen=ImageDraw.Draw(overview)
    font=ImageFont.truetype(findfont('DejaVu Sans'),14)
    poster_frame=stage/'qa/final/video-33.50.png'
    if not poster_frame.is_file():
        import imageio_ffmpeg
        proof=read(rd/'video-verification.json')
        i=round((33.5-proof['displayed_start_s'])*proof['fps'])
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(rd/'gate_multiview_comparison.mp4'),
                        '-vf',f'select=eq(n\\,{i})','-frames:v','1','-y',str(poster_frame)],check=True)
    with Image.open(poster_frame) as frame:
        for i,(j,label,color) in enumerate(((1,'Strain only','#936e3c'),(3,'Augmented fusion','#267967'))):
            x=16+j*298+26
            overview.paste(frame.crop((x,301,x+240,481)),(8+i*260,25))
            pen.text((12+i*260,3),label,fill=color,font=font)
    overview.save(src/'overview.webp',quality=89,method=6)
    files=[]
    def add(kind,name,title,caption,source=None,**extra):
        if source is not None:shutil.copyfile(source,src/name)
        files.append(dict(kind=kind,file=name,title=title,caption=caption,**extra))
    add('image','overview.webp','进门后臂爪与门槛的距离','取自第 33.50 秒的同步斜视机位。左列为纯应变，右列为增广融合，红线连接各趟实际最近的表面。')
    add('image','keyframes.webp','同视角 3D 过门与门后对准','首行为同步遮挡关键帧，中行为各趟最接近门框的过门时刻，末行为同一时刻的夹爪与目标轴。')
    add('curve','curves.webp','弯曲、方位、三维净空与末端误差','五趟轨迹使用同一时间坐标。灰带覆盖腕部相机的四秒全遮挡。')
    add('curve','paired.webp','加上差模偏移后，净空和对准怎样变化','每段连接同一种子的两趟动态遮挡飞行。空心与实心符号分别对应基础迟滞和进门前已形成差模偏移的工况。纯视觉与真值参考两行给出独立重复飞行的波动。')
    add('curve','wrist-window.webp','真实腕部画面与保留末帧的动态误差','取自同一趟融合飞行，图像与角度按曝光和交付两种时间戳对齐。',rd/'gate_wrist_window.webp')
    add('curve','methods.webp','五方法独立飞行的完整时序','每列保留该趟自己的目标、真实形态和估计。查看整个飞行区间的净空、位置和工具轴向误差。')
    add('curve','states.webp','逐方法的状态与停顿','比较进入过门与对准阶段的时间，查看因反馈断流或包络保护而暂停的区间。')
    add('video','replay.mp4','五方法同步多机位回放','每列是一趟独立飞行，从两路系统模型视图切到当时送入跟踪器的腕部图像，下方同步显示臂爪净空和末端误差。',rd/'gate_multiview_comparison.mp4',poster='overview.webp')
    add('document','review.pdf','两页审阅 PDF','第一页解释形态反馈、净空与末端对位，第二页汇总三配对种子上的四种工况。',rd/'gate_review.pdf')
    add('data','all-trials.csv','60 趟闭环成绩','成功判据、停顿、工具误差、角度与方位精度，以及每项有效观测的样本数。',rd/'all_trials_metrics.csv')
    add('data','physical-audit.csv','过门最近点与视觉恢复时间','每趟臂爪最近点附近的弯角偏差、物理欠角、工具高度和曝光到交付的恢复时间。',d/'gate-physical-audit.csv')
    add('data','native-log-audit.json','飞控、ROS 与物理执行的逐趟核对',f"按时间戳配对 {native['exact_timestamp_motor_matches']:,} 条 PX4 电机输出与 HIL 指令。ROS 的二维参考、估计与驱动量各有 {native['matched_shape_samples_per_topic']:,} 个样本逐点核对。",d/'native-log-audit.json')
    add('data','evidence-index.csv','原生飞行日志索引','按配对种子和工况保存原始包的固定 Git 修订地址、字节数和完整 SHA-256。',rd/'evidence-index.csv')
    add('data','cohort-summary.json','配对工况汇总','逐种工况给出三趟的最小值、中位数、最大值、任务通过数和可用观测数。',d/'cohort-summary.json')
    add('data','protocol.json','注册的窄门与对准协议','带载机械臂、门框尺寸、目标位姿、许可阈值、遮挡时刻与配对矩阵。',stage/'protocol.json')
    add('data','perception-lock.json','获批感知模块指纹','六维增广形态融合与传感器参数逐文件锁定。',stage/'source/gate_feedback_perception_lock.json')
    specpath=root/'round.json';when=read(specpath)['when'] if specpath.exists() else datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    save(specpath,dict(schema=1,id=TAG,stage='G2 · 单门二维形态闭环',when=when,revision=evidence_commit,title=TITLE,summary=lead,
         cover='overview.webp',files=files,parent_round=PREVIOUS,
         guide=['用同一机位比较五列的物理臂形、最近表面连线和门后目标轴。',
                '把视频停在 28 至 32 秒，比较真实腕部黑屏、保留末帧与连续应变估计。',
                '查阅两页报告及逐趟评分表，点击原始日志索引可追溯至同一趟 ULog、ROS bag 和漂移记录。'],
         scope='SUPER 机架与 Qingyu 单段臂，末端载荷 45 g。每趟独立运行 PX4 SITL、MuJoCo 和原生 ORB-SLAM3。虚拟动捕提供真值参考和独立评分。',
         source_note=f"感知冻结于 {cohort['approved_perception'][:7]}，闭环协议冻结于 {cohort['design_commit'][:7]}。代表种子为 26112131。腕部曝光后 0.20 秒交付。"))
    dist=root/'dist';gen(stage,'add',dist,'--spec',specpath,'--source',src)
    target=dist/'archive'/TAG
    shutil.copyfile(root/'flights.html',target/'flights.html')
    page=(target/'index.html').read_text(encoding='utf8')
    if 'href="flights.html"' not in page:
        page=page.replace('</header>','<p><a href="flights.html">打开 60 趟独立飞行的成绩、日志与校验值</a></p></header>',1)
        (target/'index.html').write_bytes(page.encode())
        snap=read(target/'snapshot.json');snap['outputs']['page']['sha256']=digest(target/'index.html')
        save(target/'snapshot.json',snap)
    gen(stage,'verify',dist)
    shutil.copyfile(target/'meeting.zip',stage/'组会离线包.zip')
    with tarfile.open(root/'meeting-mirror.tar.gz','w:gz') as tar:
        for f in sorted(target.rglob('*')):
            if f.is_file():tar.add(f,arcname=f.relative_to(target).as_posix(),recursive=False)


def publish(stage,media_commit):
    if not re.fullmatch('[0-9a-f]{40}',media_commit):raise ValueError('Full media revision required')
    root=stage/'canonical-release';orig=root/'dist/archive'/TAG
    repo=stage/'site-cache';dist=repo/'dist';dest=dist/'archive'/TAG;media=dest/'media';media.mkdir(parents=True,exist_ok=True)
    previous={p:digest(p) for p in (dist/'archive').glob('*/snapshot.json') if p.parent!=dest}
    if (dist/'archive'/PREVIOUS/'snapshot.json') not in previous:raise ValueError('Previous reviewed stage missing')
    snap=read(orig/'snapshot.json');spec=read(root/'round.json')
    prefix='revision/meeting_archive/archive/'+TAG+'/'
    # The research Git keeps a byte-identical independent copy of the video. Host
    # the verified replay on the group site's own origin whenever it fits.
    remote=RAW+media_commit+'/'+prefix
    cdn=CDN+media_commit+'/'+prefix+'media/'
    ext={'replay.mp4':(cdn+'replay.mp4',orig/'media/replay.mp4'),'meeting.zip':(remote+'meeting.zip',orig/'meeting.zip')}
    with ThreadPoolExecutor(max_workers=len(ext)) as pool:
        tasks={n:pool.submit(confirm,*v) for n,v in ext.items()};proof={n:w.result() for n,w in tasks.items()}
    evidence=repo/'source-evidence'/TAG;save(evidence/'verified.json',proof)
    shutil.copyfile(orig/'meeting.zip',evidence/'meeting.zip')
    save(stage/'public-media-verification.json',proof)
    entries={row['file']:row for row in spec['files']}
    page=(orig/'index.html').read_text(encoding='utf-8')
    outside=sum(p.stat().st_size for p in dist.rglob('*') if p.is_file() and dest not in p.parents)
    native_size=outside+sum(r['bytes'] for r in snap['files'])
    native_video=native_size<255*1024**2
    for f in snap['files']:
        n=f['file']
        if n in proof and not (n=='replay.mp4' and native_video):
            if proof[n]['sha256']!=f['sha256']:raise ValueError(n)
            url=proof[n]['external_url'];f['external_url']=entries[n]['external_url']=url
            page=page.replace('"media/'+n+'"','"'+html.escape(url,quote=True)+'"')
        else:
            shutil.copyfile(orig/'media'/n,media/n)
    page=page.replace('href="meeting.zip"','href="'+proof['meeting.zip']['external_url']+'"')
    page=page.replace('打开对应飞行的原始日志与验收记录','上一轮获批的视觉与应变融合模块')
    (dest/'index.html').write_bytes(page.encode())
    shutil.copyfile(orig/'flights.html',dest/'flights.html')
    snap['outputs']['page']['sha256']=digest(dest/'index.html')
    snap['outputs']['pack']['external_url']=proof['meeting.zip']['external_url']
    save(dest/'snapshot.json',snap)
    save(repo/'tools/rounds/gate-feedback.json',spec)
    attr=repo/'.gitattributes';rules=attr.read_text(encoding='utf8')
    for rule in ('dist/archive/'+TAG+'/media/* -text','source-evidence/'+TAG+'/* -text'):
        if rule not in rules:rules+='\n'+rule+'\n'
    attr.write_bytes(rules.encode())
    gen(stage,'index',dist)
    route='archive/'+TAG+'/';home=dist/'index.html';page=home.read_text(encoding='utf-8')
    replay=route+'media/replay.mp4' if native_video else proof['replay.mp4']['external_url']
    video=(f'<section id="reviewVideo"><h2>{TITLE}</h2><video controls playsinline preload="metadata" poster="{route}media/overview.webp" '
           f'style="display:block;width:100%;max-height:600px;background:white"><source src="{replay}" type="video/mp4"></video>'
           f'<p><a href="{route}media/review.pdf">两页审阅</a> · <a href="{route}index.html">实验图、曲线与视频</a> · '
           f'<a href="{route}flights.html">逐趟原始日志</a></p></section>')
    page,n=re.subn(r'<section id="reviewVideo">.*?</section>',lambda _:video,page,count=1,flags=re.S)
    if n!=1:raise ValueError('Existing meeting video section missing')
    slot='<section><h2>阶段文件</h2><p>';link='<a href="'+route+'media/review.pdf" target="_blank">单门二维形态闭环审阅</a>　'
    if slot not in page:raise ValueError('Existing manuscript review section missing')
    if link not in page:page=page.replace(slot,slot+link,1)
    home.write_bytes(page.encode('utf-8'))
    state=read(dist/'status_public.json');cohort=read(stage/'decisions/cohort-summary.json')
    f=cohort['conditions']['differential-occluded'];diag=read(stage/'review/wrist-window.json')
    state.update(updated=spec['when'],stage=spec['stage'],status='awaiting_review',progress=100,
                 mode='60 趟独立 PX4 / MuJoCo 飞行 · 原生腕部 ORB-SLAM3',
                 detail=spec['summary'],geometry=route+'media/keyframes.webp',geometry_caption='五方法过门最近点和门后工具轴',
                 rolling=route+'media/curves.webp',rolling_caption='弯曲、方位、净空及工具误差',replay=None,
                 items=[f"差模与遮挡下的安全通过并对准：增广融合 {f['proposed']['passed']}/3，固定 KF {f['fixed']['passed']}/3",
                        f"代表趟遮挡期弯角变化 {diag['true_theta_range_deg']:.1f}°，弯曲方位变化 {diag['true_phi_range_deg']:.1f}°",
                        f"保留最后视觉角与连续融合的弯角 RMSE 为 {diag['vision_hold']['theta_rmse_deg']:.2f}° 和 {diag['blackout']['theta_rmse_deg']:.2f}°",
                        '60 份原生 ULog、ROS bag、物理与传感器记录已逐文件校验'],
                 commits=[cohort['approved_perception'][:7]+' · 冻结感知',cohort['design_commit'][:7]+' · 注册闭环',
                          spec['revision'][:7]+' · 60 趟原始记录',media_commit[:7]+' · 图表与视频'])
    # The historical homepage supports local and absolute resource paths.
    for key in ('geometry','rolling'):
        if state[key].startswith(route):
            state[key]='https://aerial-continuum-research-monitor.consideratepastry4.chatgpt.site/'+state[key]
    save(dist/'status_public.json',state)
    sys.path.insert(0,str(stage.parent/'D_整机几何与规划验收_20261118'))
    try:
        from quota_snapshot import read_windows
        quota=read_windows()
    except (ImportError,OSError,RuntimeError):
        pass
    else:save(dist/'usage_public.json',quota)
    if any(digest(p)!=sha for p,sha in previous.items()):raise ValueError('Historic round changed')
    gen(stage,'verify',dist)
    size=sum(p.stat().st_size for p in dist.rglob('*') if p.is_file())
    print('Public site size',size,'rounds',len(previous)+1,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('build','publish'));p.add_argument('stage',type=Path);p.add_argument('revision')
    a=p.parse_args();{'build':build,'publish':publish}[a.mode](a.stage.resolve(),a.revision)
