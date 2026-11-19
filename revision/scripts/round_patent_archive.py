"""Archive this completed round for the group. Existing round snapshots are immutable."""
import argparse,hashlib,json,re,shutil,subprocess,sys,tarfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone,timedelta
from pathlib import Path
from urllib.request import Request,urlopen
import time
TAG='2026-11-19-confidence'
TITLE='腕部视觉何时能纠正应变偏差？'
SUMMARY='四趟单门轨迹上，逐帧比较清晰、局部遮挡、短时与长时失视，以及应变零点偏移和材料回程。'
BASE='https://raw.githubusercontent.com/sanshibro/Keyan_SpaceARm/'
def read(p):return json.loads(p.read_bytes())
def write(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((json.dumps(d,ensure_ascii=False,indent=2)+'\n').encode())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def gen(stage,op,site,*opts):
    subprocess.run([sys.executable,str(stage/'site-cache/tools/meeting_archive.py'),op,'--site',str(site),*map(str,opts)],check=True)
def build(stage,rev):
    work=stage/'canonical';src=work/'sources';src.mkdir(parents=True,exist_ok=True)
    files=[]
    def add(kind,parent,name,title,caption,**extra):
        shutil.copyfile(parent/name,src/name);files.append(dict(kind=kind,file=name,title=title,caption=caption,**extra))
    r=stage/'review'
    add('image',r,'tracking.webp','腕部相机原帧、置信度与弯角误差','相机依次经历清晰、局部遮挡、全遮挡及恢复。横轴按估计器输出时钟对齐。')
    add('curve',r,'paired.webp','图像清晰和失视，各自适合怎样的权重？','四趟重放的弯角 RMSE，分别对照健康应变、零点偏移和材料迟滞。')
    add('video',r,'replay.mp4','腕部视角与动捕同步回放','17 秒中同时查看实际输入相机图像、视觉置信度、估计弯角与虚拟动捕。',poster='tracking.webp')
    add('document',r,'review.pdf','两页组会预览','第 1 页说明两路观测和修正条件，第 2 页给出五种方法的同轨结果与误差边界。')
    for name,title,cap in [
        ('paired-scores.json','五种方法的分阶段评分','弯角、可观弯曲方向、弯曲向量误差和有效帧覆盖，含共同有效时刻。'),
        ('case-scores.json','逐趟配对结果','四条已知轨迹逐一核对，区分光照与应变观测工况。'),
        ('teacher-and-recovery.json','视觉教学与重定位','逐趟记录修正次数、原生或几何配准恢复和输出时延。'),
        ('audit.json','因果性与失视检查','12 次截断输入、无视觉时冻结修正、两路均失时方差增长及弯曲方向跨界。'),
        ('support-scores.json','局部遮挡的图像覆盖检查','主试验后另做的同轨诊断。低维特征支撑会改变视觉相对权重。'),
        ('variance-bench.json','应变置信区间的留出检验','按物理工况分层检查实际误差落入预测区间的比例。'),
        ('visual-calibration.json','独立飞行上的视觉标定','800 帧估计弯角和弯曲方向的方差尺度。'),
        ('evidence-index.json','原始归档校验','两份数据包的大小、文件数和 SHA-256。')]:
        add('data',stage,name,title,cap)
    for name,title in [('confidence-inputs.tar.gz','传感输入与配准记录'),('confidence-results.tar.gz','估计轨迹与独立评分真值')]:
        add('data',stage/'evidence',name,title,'完整压缩数据，可单独下载并与校验表核对。',separate=True)
    spec=work/'round.json';when=read(spec)['when'] if spec.exists() else datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    write(spec,dict(schema=1,id=TAG,stage='G1 · 视觉与应变置信度',when=when,revision=rev,title=TITLE,summary=SUMMARY,
        cover='tracking.webp',parent_round='2026-11-19-slim',files=files,
        guide=['先看视频中腕部图像被遮挡时两种置信度的变化。','对照清晰与全遮挡的误差，再看在线修正在不同应变工况下的得失。','打开两页 PDF 或离线包准备汇报。'],
        scope='复用既有飞行轨迹检验估计器。地图恢复使用腕部 RGB-D 几何；阶段结论尚待新的轨迹与四通道实物采集复核。',
        source_note='425eabb 登记同轨主试验，587b89d 实现两路方差与延迟融合，本轮评分与校验随数据包保存。'))
    target=work/'dist';gen(stage,'add',target,'--spec',spec,'--source',src);gen(stage,'verify',target)
    folder=target/'archive'/TAG;shutil.copyfile(folder/'meeting.zip',stage/'组会离线包.zip')
    with tarfile.open(work/'meeting-mirror.tar.gz','w:gz') as tf:
        for path in sorted(folder.rglob('*')):
            if path.is_file() and not path.name.endswith('.tar.gz'):tf.add(path,path.relative_to(folder).as_posix())
def confirm(url,p):
    for i in range(4):
        try:
            with urlopen(Request(url,headers={'User-Agent':'RoboticsResearchArchive/1'}),timeout=28) as f:data=f.read()
            expect=digest(p)
            if hashlib.sha256(data).hexdigest()!=expect:raise ValueError('Remote bytes differ: '+p.name)
            return dict(bytes=len(data),sha256=expect,external_url=url)
        except Exception:
            if i==3:raise
            time.sleep(i+2)
def publish(stage,rev):
    site=stage/'site-cache';dist=site/'dist';orig=stage/'canonical/dist/archive'/TAG;dest=dist/'archive'/TAG
    previous={p:digest(p) for p in (dist/'archive').glob('*/snapshot.json') if p.parent!=dest}
    snap=read(orig/'snapshot.json');spec=read(stage/'canonical/round.json');assert len(previous)>=14
    route='archive/'+TAG+'/';remote=BASE+rev+'/revision/meeting_archive/'+route
    packs=read(stage/'evidence-index.json');other={}
    for x in snap['files']:
        n=x['file']
        if n in ('replay.mp4','review.pdf'):continue
        url=BASE+rev+'/revision/data/round_patent_confidence/evidence/'+n if n in packs else remote+'media/'+n
        other[n]=(url,orig/'media'/n)
    other['meeting.zip']=(remote+'meeting.zip',orig/'meeting.zip')
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs={n:pool.submit(confirm,*pair) for n,pair in other.items()};proof={k:v.result() for k,v in jobs.items()}
    cached=site/'source-evidence'/TAG;write(cached/'verified.json',proof);shutil.copyfile(orig/'meeting.zip',cached/'meeting.zip')
    write(stage/'public-raw-verification.json',proof)
    page=(orig/'index.html').read_text(encoding='utf8')
    (dest/'media').mkdir(parents=True,exist_ok=True)
    specs={f['file']:f for f in spec['files']}
    for x in snap['files']:
        n=x['file']
        if n in proof:
            assert x['sha256']==proof[n]['sha256'];url=proof[n]['external_url']
            x['external_url']=specs[n]['external_url']=url;page=page.replace('"media/'+n+'"','"'+url+'"')
        else:shutil.copyfile(orig/'media'/n,dest/'media'/n)
    snap['outputs']['pack']['external_url']=proof['meeting.zip']['external_url']
    page=page.replace('href="meeting.zip"','href="'+proof['meeting.zip']['external_url']+'"')
    (dest/'index.html').write_bytes(page.encode());snap['outputs']['page']['sha256']=digest(dest/'index.html')
    write(dest/'snapshot.json',snap);write(site/'tools/rounds/confidence.json',spec)
    attrs=site/'.gitattributes';data=attrs.read_text(encoding='utf8')
    for rule in ['dist/'+route+'media/* -text','source-evidence/'+TAG+'/* -text']:
        if rule not in data:data+='\n'+rule+'\n'
    attrs.write_bytes(data.encode());gen(stage,'index',dist)
    home=dist/'index.html';html=home.read_text(encoding='utf8');cover=proof['tracking.webp']['external_url']
    block=('<section id="reviewVideo"><h2>'+TITLE+'</h2><video controls playsinline preload="metadata" poster="'+cover+'" style="display:block;width:100%;max-height:600px;background:white"><source src="'+route+'media/replay.mp4" type="video/mp4"></video><p><a href="'+route+'index.html">本轮图像与原始数据</a> · <a href="'+proof['paired.webp']['external_url']+'">五种方法的对照</a> · <a href="'+route+'media/review.pdf">两页组会 PDF</a></p></section>')
    html,n=re.subn(r'<section id="reviewVideo">.*?</section>',lambda _:block,html,count=1,flags=re.S);assert n==1
    key='<section><h2>阶段文件</h2><p>';assert key in html
    item='<a href="'+route+'media/review.pdf" target="_blank">视觉与应变置信度审阅</a>　'
    if item not in html:html=html.replace(key,key+item,1)
    home.write_bytes(html.encode())
    state=read(dist/'status_public.json');state.update(updated=spec['when'],stage=spec['stage'],status='awaiting_review',progress=100,
        mode='同轨重放 · 腕部 RGB-D · 四路应变',detail=SUMMARY,geometry=cover,geometry_caption='腕部图像、置信度与弯角误差',
        rolling=proof['paired.webp']['external_url'],rolling_caption='健康、零点偏移和材料回程的配对结果',replay=None,
        items=['四趟相机重放每趟 850 帧','4 秒全遮挡期间各应变工况分别覆盖 800 / 800 时刻','原地图经 RGB-D 配准恢复约 0.48 s','下一步辨识应变漂移与回程，并在新轨迹上验证'],
        commits=['425eabb · 同轨试验协议','587b89d · 置信度与延迟融合','b72ee60 · 分阶段评分与数据归档'])
    write(dist/'status_public.json',state)
    sys.path.insert(0,str(stage.parent/'D_整机几何与规划验收_20261118'))
    from quota_snapshot import read_windows
    try:q=read_windows()
    except (RuntimeError,OSError):print('Previous timestamped quota snapshot retained')
    else:write(dist/'usage_public.json',q)
    assert all(digest(p)==h for p,h in previous.items());gen(stage,'verify',dist)
    total=sum(p.stat().st_size for p in dist.rglob('*') if p.is_file())
    print('Hosted bytes',total,'rounds',len(previous)+1,flush=True)
    if total>256*1024**2:raise ValueError('Hosting payload exceeds the static archive limit')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','publish']);p.add_argument('stage',type=Path);p.add_argument('commit')
    a=p.parse_args();(build if a.action=='build' else publish)(a.stage.resolve(),a.commit)
