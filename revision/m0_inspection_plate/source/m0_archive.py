"""Assemble the meeting record, then publish hash-verified media beside earlier rounds."""
import argparse
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timedelta,timezone
from PIL import Image

from m0_data import ROOT,OUT,WORK,QA,read_json,write_json,sha_file
from gate_meeting_archive import confirm
ID='2026-11-21-m0-cutaway'
PARENT='2026-11-20-gate-feedback'
SOURCE='951c512fbeaf860c5158e18c44aed527dbf7a899'
TITLE='穿过隔舱，再对准斜置检修接口'
SUMMARY='同一工况下，纯应变、固定权重 KF、增广融合的臂爪最小净空依次为 6.21、7.38、8.90 cm。全景与同机位特写把这段差异定位到门槛，并在门后比较夹爪与法兰的轴向。'
CHECKOUT=Path(os.environ.get('M0_SITE_CHECKOUT',ROOT/'site-cache/.sites-checkout'))
CAN=ROOT/'meeting'
ORIGIN='https://aerial-continuum-research-monitor.consideratepastry4.chatgpt.site'
ROUTE='archive/'+ID+'/'
STEMS=('gate_hero_v2','gate_insets_v2','gate_four_metrics_v2','gate_main_v2')

def gen(action,dest,*args):
    subprocess.run([sys.executable,str(CHECKOUT/'tools/meeting_archive.py'),action,'--site',str(dest),*map(str,args)],check=True)

def jpgweb(orig,dest,w,h):
    with Image.open(orig) as im:
        im=im.convert('RGB');im.thumbnail((w,h),Image.Resampling.LANCZOS)
        im.save(dest,quality=91,method=6)

def styled(page):
    css="""#images .gallery{grid-template-columns:1fr}#images .media img{height:auto;max-height:680px}
#images .media{background:white;padding:0;border:0}.block h2,.read h2{font-family:Georgia,SimSun,serif}
#curves .gallery.solo .media img{height:auto;max-height:635px}
.video{background:#fff;max-height:690px}.gallery.solo{max-width:1040px;margin:auto}
.primary-link{font-weight:650}"""
    return page.replace('</style>',css+'</style>',1).replace('</header>',
        '<p class="quick"><a class="primary-link" href="media/review.pdf">打开两页对照</a>'
        '<a href="#videos">播放五方法对比</a><a href="media/print.tar.gz">下载高清图与矢量源图</a></p></header>',1)

def build():
    src=CAN/'sources';src.mkdir(parents=True,exist_ok=True);specfile=CAN/'round.json'
    files=[]
    def add(kind,file,title,caption,**extra):
        files.append(dict(kind=kind,file=file,title=title,caption=caption,**extra))
    for stem,local,desc,cap,kind in (
        (STEMS[0],'hero','全景残影与夹爪流线','按 24.0、30.0、35.5、52.0 秒读四个整机姿态。铜色曲线跟踪工具中心，青色曲线跟踪机身质心。','image'),
        (STEMS[1],'insets','下槛净空与法兰对准','上排是各趟独立的最近时刻，下排固定在 52.0 秒。每排共用视角、比例和裁切。','image'),
        (STEMS[2],'curves','弯曲幅值、方位、净空与对准误差','灰带内完全失视 4 秒。本文方法的弯角 RMSE 为 0.27°，保留遮挡前最后一帧的弯角峰值误差为 41.4°。','curve')):
        jpgweb(OUT/(stem+'.png'),src/(local+'.webp'),1900,1400)
        add(kind,local+'.webp',desc,cap)
    jpgweb(QA/'film-435.jpg',src/'poster.webp',1120,680)
    add('poster','poster.webp','视频封面','五组方法于 52.0 秒的同场景对照。')
    for fn,origin in (('replay.mp4',OUT/'gate_multiview_v2.mp4'),('review.pdf',OUT/'gate_review_v2.pdf'),
                      ('print.pdf',OUT/'gate_main_v2.pdf'),('geometry.json',QA/'scene-clearance.json'),
                      ('fingerprints.json',QA/'archive-fingerprints.json')):
        shutil.copyfile(origin,src/fn)
    add('video','replay.mp4','同机位闭环飞行回放','1920 × 1080。侧视、过门或对准近景、原始腕部 RGB 及读数同步回放，覆盖释放后 23–55 秒。',poster='poster.webp')
    add('document','review.pdf','两页审阅 PDF','第一页看任务空间与接触最近点，第二页读同步曲线与五方法配对表。')
    add('document','print.pdf','全景、近景和同步曲线合版','三组图的可放大矢量版。')
    papers=[OUT/(stem+ext) for stem in STEMS for ext in ('.png','.pdf','.svg')]
    packed=src/'print.tar.gz';rebuild=True
    if packed.is_file():
        import hashlib
        with tarfile.open(packed) as tar:
            rebuild=set(tar.getnames())!={p.name for p in papers} or any(
                hashlib.file_digest(tar.extractfile(p.name),'sha256').hexdigest()!=sha_file(p) for p in papers)
    if rebuild:
        with tarfile.open(packed,'w:gz',compresslevel=6) as tar:
            for p in papers:tar.add(p,arcname=p.name,recursive=False)
    add('data','print.tar.gz','四组高清图与矢量源图','含全景、同机位特写、四排曲线及合版。每组提供 PNG、PDF 和 SVG。',separate=True)
    add('data','geometry.json','场景几何校验','42 趟成功轨迹共 153,500 个记录姿态与 153,458 段相邻姿态扫掠的分离证书。')
    add('data','fingerprints.json','六十趟原始记录指纹','按趟核对原包、物理状态、控制记录和评分轨迹。')
    old=read_json(specfile)['when'] if specfile.exists() else datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    spec=dict(schema=1,id=ID,stage='M0 · 隔舱场景与主图',when=old,revision=SOURCE,title=TITLE,summary=SUMMARY,
        cover='hero.webp',files=files,parent_round=PARENT,
        guide=['从全景沿夹爪流线看预弯、失视进窗、越槛和斜向对准。','在同机位特写中读最小净空，再比较夹爪轴与法兰轴。','将视频停在 28–32 秒，结合两页审阅中的弯角与方位曲线讲解失视期间的反馈。'],
        scope='给获批的单门仿真数据增加可复用的工业隔舱和斜置法兰。半透明墙体、金属窗套与支架按原门洞和目标坐标配准。',
        source_note='60 趟闭环记录取自 G2 归档 951c512。代表种子 26112131，差模偏移及腕部全遮挡，视频保留原始腕部曝光和帧交付时刻。')
    write_json(specfile,spec);dist=CAN/'dist';gen('add',dist,'--spec',specfile,'--source',src)
    target=dist/'archive'/ID;page=target/'index.html'
    content=page.read_text('utf8')
    if 'primary-link' not in content:
        page.write_text(styled(content),encoding='utf8',newline='\n')
        snap=read_json(target/'snapshot.json');snap['outputs']['page']['sha256']=sha_file(page);write_json(target/'snapshot.json',snap)
    gen('verify',dist)
    shutil.copyfile(target/'meeting.zip',ROOT/'组会离线包.zip')
    local=page.read_text('utf8').replace('href="../index.html"','href="meeting/dist/archive/index.html"').replace(
       'href="../'+PARENT+'/index.html"','href="'+ORIGIN+'/archive/'+PARENT+'/index.html"').replace(
       '"media/','"meeting/dist/archive/'+ID+'/media/').replace('href="meeting.zip"','href="组会离线包.zip"').replace(
       'href="snapshot.json"','href="meeting/dist/archive/'+ID+'/snapshot.json"')
    (ROOT/'组会预览.html').write_text(local,encoding='utf8',newline='\n')
    print('Canonical archive:',target,flush=True)

def publish(rev):
    if not re.fullmatch(r'[a-f0-9]{40}',rev):raise ValueError('Full NUC media revision required')
    dist=CHECKOUT/'dist';src=CAN/'dist/archive'/ID;dest=dist/'archive'/ID;media=dest/'media';media.mkdir(parents=True,exist_ok=True)
    previous={p:sha_file(p) for p in (dist/'archive').glob('*/snapshot.json') if p.parent!=dest}
    if (dist/'archive'/PARENT/'snapshot.json') not in previous:raise ValueError('Expected G2 archive')
    spec=read_json(CAN/'round.json');snap=read_json(src/'snapshot.json');entries={f['file']:f for f in spec['files']}
    path='revision/meeting_archive/archive/'+ID+'/'
    raw='https://raw.githubusercontent.com/sanshibro/Keyan_SpaceARm/'+rev+'/'+path
    cdn='https://cdn.jsdelivr.net/gh/sanshibro/Keyan_SpaceARm@'+rev+'/'+path
    names={'replay.mp4':(cdn+'media/replay.mp4',src/'media/replay.mp4'),
           'print.tar.gz':(raw+'media/print.tar.gz',src/'media/print.tar.gz'),
           'print.pdf':(raw+'media/print.pdf',src/'media/print.pdf'),'meeting.zip':(raw+'meeting.zip',src/'meeting.zip')}
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks={n:pool.submit(confirm,url,file) for n,(url,file) in names.items()}
        proof={n:p.result() for n,p in tasks.items()}
    evidence=CHECKOUT/'source-evidence'/ID;write_json_mkdir(evidence/'verified.json',proof)
    shutil.copyfile(src/'meeting.zip',evidence/'meeting.zip')
    page=(src/'index.html').read_text('utf8')
    for f in snap['files']:
        n=f['file']
        if n in proof:
            if f['sha256']!=proof[n]['sha256']:raise ValueError(n)
            f['external_url']=entries[n]['external_url']=proof[n]['external_url']
            page=page.replace('"media/'+n+'"','"'+html.escape(f['external_url'],quote=True)+'"')
        else:shutil.copyfile(src/'media'/n,media/n)
    page=page.replace('href="meeting.zip"','href="'+proof['meeting.zip']['external_url']+'"')
    (dest/'index.html').write_text(page,encoding='utf8',newline='\n')
    snap['outputs']['page']['sha256']=sha_file(dest/'index.html');snap['outputs']['pack']['external_url']=proof['meeting.zip']['external_url']
    write_json(dest/'snapshot.json',snap);write_json(CHECKOUT/'tools/rounds/m0-cutaway.json',spec)
    attr=CHECKOUT/'.gitattributes';content=attr.read_text('utf8')
    for line in ('dist/archive/'+ID+'/media/* -text','source-evidence/'+ID+'/* -text'):
        if line not in content:content+='\n'+line+'\n'
    attr.write_text(content,encoding='utf8',newline='\n')
    gen('index',dist)
    home=dist/'index.html';content=home.read_text(encoding='utf8')
    film=(f'<section id="reviewVideo"><h2>{TITLE}</h2><video controls playsinline preload="metadata" poster="{ROUTE}media/poster.webp" '
          f'style="display:block;width:100%;max-height:600px;background:white"><source src="{proof["replay.mp4"]["external_url"]}" type="video/mp4"></video>'
          f'<p><a href="{ROUTE}media/review.pdf">两页审阅</a> · <a href="{ROUTE}index.html">场景、曲线与逐帧视频</a> · '
          f'<a href="archive/{PARENT}/flights.html">60 趟原始飞行记录</a></p></section>')
    content,count=re.subn(r'<section id="reviewVideo">.*?</section>',lambda _:film,content,count=1,flags=re.S)
    if count!=1:raise ValueError('Missing review panel')
    clause='<section><h2>阶段文件</h2><p>';link=f'<a href="{ROUTE}media/review.pdf">工业隔舱与斜置法兰审阅</a>　'
    if link not in content:content=content.replace(clause,clause+link,1)
    home.write_text(content,encoding='utf8',newline='\n')
    status=read_json(dist/'status_public.json')
    status.update(stage=spec['stage'],updated=spec['when'],mode='归档飞行离线三维重渲',status='awaiting_review',progress=100,
        detail=SUMMARY,geometry=ORIGIN+'/'+ROUTE+'media/hero.webp',geometry_caption='隔舱检修任务的四个记录姿态',
        rolling=ORIGIN+'/'+ROUTE+'media/curves.webp',rolling_caption='四排同步曲线',replay=None,
        items=['门洞内宽 760 mm、内高 690 mm','42 趟成功轨迹与新增窗套和法兰支架完成几何复核','五组原始腕部视图共核对 2,405 帧','M0 场景视觉验收'],
        commits=['881a799 · 冻结感知','951c512 · 单门闭环归档',rev[:7]+' · M0 场景与图表'])
    write_json(dist/'status_public.json',status)
    sys.path.insert(0,str(ROOT.parent/'D_整机几何与规划验收_20261118'))
    try:from quota_snapshot import read_windows
    except ImportError:pass
    else:
        try:q=read_windows()
        except (OSError,RuntimeError):q=None
        if q:write_json(dist/'usage_public.json',q)
    if any(sha_file(p)!=v for p,v in previous.items()):raise ValueError('Historical snapshot changed')
    gen('verify',dist)
    size=sum(p.stat().st_size for p in dist.rglob('*') if p.is_file())
    if size>=256*1024**2:raise ValueError('Hosting capacity: '+str(size))
    write_json(QA/'site-media-verification.json',dict(media=proof,previous_rounds=len(previous),dist_bytes=size,commit=rev))
    print('Verified public source:',len(previous)+1,'rounds,',size,'bytes',flush=True)

def write_json_mkdir(p,data):p.parent.mkdir(parents=True,exist_ok=True);write_json(p,data)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('build','publish'));ap.add_argument('revision',nargs='?')
    args=ap.parse_args()
    if args.action=='build':build()
    else:publish(args.revision)
