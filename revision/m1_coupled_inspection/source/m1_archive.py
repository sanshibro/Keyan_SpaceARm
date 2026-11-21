"""Append the M1 archive to the research group's existing publication."""
import argparse,hashlib,html,json,re,shutil,subprocess,sys,tarfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone,timedelta
from pathlib import Path
from PIL import Image
from gate_meeting_archive import confirm
from m1_plates import ROOT,OUT,FILM,DATA,read
from m0_archive import styled,jpgweb
ID='2026-11-21-m1-coupled';PARENT='2026-11-21-m0-cutaway'
CAN=ROOT/'meeting';SITE=ROOT/'site-cache/.sites-checkout';ROUTE='archive/'+ID+'/'
ORIGIN='https://aerial-continuum-research-monitor.consideratepastry4.chatgpt.site'
TITLE='跨越低梁与侧管，斜向检视法兰'
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n','utf8')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def gen(what,site,*argv):
    subprocess.run([sys.executable,str(SITE/'tools/meeting_archive.py'),what,'--site',str(site),*map(str,argv)],check=True)
def build():
    s=read(OUT/'m1_methods.json');reg=read(DATA/'registered.json');src=CAN/'sources';src.mkdir(parents=True,exist_ok=True);items=[]
    def add(kind,name,title,caption,**kw):items.append(dict(kind=kind,file=name,title=title,caption=caption,**kw))
    for name,lab,desc,kind in [('m1_hero','穿越检修舱的四个姿态','两条流线分别给出机身与夹爪的物理轨迹。','image'),
            ('m1_sections','同机位近景与三个实测横截面','剖切真实连杆几何。每一列使用同一相机、比例和剖切平面。','image'),
            ('m1_curves','六维运动与机臂反作用','同一配对种子的机身位置、偏航、双弯曲变量、臂根力矩、前后桨推力差及三维净空。','curve')]:
        jpgweb(OUT/(name+'.png'),src/(name+'.webp'),1680,1250);add(kind,name+'.webp',lab,desc)
    jpgweb(FILM/'flange-coupled-wrist.png',src/'wrist.webp',640,480)
    add('image','wrist.webp','检视时的原始腕部 RGB','这一曝光来自送入原生 ORB-SLAM3 的同一帧。法兰和管廊与外部机位共用几何。')
    jpgweb(FILM/'flange-coupled.jpg',src/'poster.webp',920,520);add('poster','poster.webp','检视帧','录像中的实体法兰。')
    for n,p in [('replay.mp4',FILM/'m1_multiview.mp4'),('review.pdf',OUT/'M1_两页审阅.pdf'),('flights.csv',OUT/'m1_paired_flights.csv'),
            ('evidence.json',DATA/'evidence.json'),('registered.json',DATA/'registered.json')]:shutil.copyfile(p,src/n)
    add('video','replay.mp4','五方法配对飞行','1920 × 1080；每章在同一个释放时钟上比较两趟飞行，同时显示注册外部视角、原生腕部曝光和交付时刻。',poster='poster.webp')
    add('document','review.pdf','打开两页审阅','第一页阅读三关几何和腕部视野，第二页比较耦合时序与成绩表。')
    add('data','flights.csv','十五趟逐趟数据','每行一个预登记方法与种子，含求解时间、任务验收和实际动态响应。')
    add('data','registered.json','配对试验登记','场景摘要、源提交、固定种子、感知锁定和统一评分阈值。')
    add('data','evidence.json','原始飞行与传感数据指纹','每趟 ULog、ROS bag、物理与控制日志，以及全部 RGB、深度曝光的指纹。')
    with tarfile.open(src/'print.tar.gz','w:gz',compresslevel=5) as z:
        for f in ('m1_hero','m1_sections','m1_curves'):
            for ext in ('.pdf','.svg','.png'):z.add(OUT/(f+ext),arcname=f+ext)
    add('data','print.tar.gz','高清主图与矢量文件','三组图，各含 PNG、PDF 和 SVG。',separate=True)
    meta=CAN/'round.json';t=read(meta)['when'] if meta.exists() else datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    summary=f"三种规划约束、一个时序消融与完整联合规划各飞三趟。完整法通过 {s['coupled']['success']}/3，时序消融通过 {s['unregularized']['success']}/3。腕部与外部机位共用同一套工业几何。"
    spec=dict(schema=1,id=ID,parent_round=PARENT,stage='M1 · 六维全身联合规划',when=t,revision=reg['source_commit'],title=TITLE,
              summary=summary,cover='m1_hero.webp',files=items,guide=['沿两条轨迹区分机身侧移与柔性臂转向。','在同一剖面比较各方法实际达到的位置。','对照成对飞行的反力矩、推力和净空，再查看腕部检视画面。'],
              scope='在已测绘的三维管廊上独立规划并闭环飞行，比较五种空间及时序策略。',source_note='3 个配对种子，15 个原生 PX4 SITL / MuJoCo 飞行记录；冻结增广感知 881a799。')
    write(meta,spec);dist=CAN/'dist';gen('add',dist,'--spec',meta,'--source',src)
    target=dist/ROUTE;page=target/'index.html';ht=page.read_text('utf8')
    if 'primary-link' not in ht:
        ht=styled(ht).replace('打开对应飞行的原始日志与验收记录','打开上一轮单门场景归档')
        page.write_text(ht,'utf8',newline='\n');snap=read(target/'snapshot.json');snap['outputs']['page']['sha256']=sha(page);write(target/'snapshot.json',snap)
    gen('verify',dist);shutil.copyfile(target/'meeting.zip',ROOT/'M1_组会离线包.zip')
def publish(rev):
    if not re.fullmatch('[0-9a-f]{40}',rev):raise ValueError('Expected immutable media revision')
    top=SITE/'dist';source=CAN/'dist'/ROUTE;dest=top/ROUTE;media=dest/'media';media.mkdir(parents=True,exist_ok=True)
    prev={p:sha(p) for p in (top/'archive').glob('*/snapshot.json') if p.parent!=dest};assert top/'archive'/PARENT/'snapshot.json' in prev
    base='revision/meeting_archive/'+ROUTE
    raw='https://raw.githubusercontent.com/sanshibro/Keyan_SpaceARm/'+rev+'/'+base
    cdn='https://cdn.jsdelivr.net/gh/sanshibro/Keyan_SpaceARm@'+rev+'/'+base
    remotes={'replay.mp4':(cdn+'media/replay.mp4',source/'media/replay.mp4'),'print.tar.gz':(raw+'media/print.tar.gz',source/'media/print.tar.gz'),
             'review.pdf':(cdn+'media/review.pdf',source/'media/review.pdf'),'meeting.zip':(raw+'meeting.zip',source/'meeting.zip')}
    with ThreadPoolExecutor(max_workers=4) as pool:proof=dict(zip(remotes,pool.map(lambda x:confirm(*x),remotes.values())))
    evidence=SITE/'source-evidence'/ID;write(evidence/'verified.json',proof);shutil.copyfile(source/'meeting.zip',evidence/'meeting.zip')
    spec=read(CAN/'round.json');snap=read(source/'snapshot.json');h=(source/'index.html').read_text('utf8');entries={f['file']:f for f in spec['files']}
    for f in snap['files']:
        n=f['file']
        if n in proof:
            url=proof[n]['external_url'];assert proof[n]['sha256']==f['sha256']
            f['external_url']=entries[n]['external_url']=url;h=h.replace('"media/'+n+'"','"'+html.escape(url,quote=True)+'"')
        else:shutil.copyfile(source/'media'/n,media/n)
    h=h.replace('href="meeting.zip"','href="'+proof['meeting.zip']['external_url']+'"')
    (dest/'index.html').write_text(h,'utf8',newline='\n');snap['outputs']['page']['sha256']=sha(dest/'index.html');snap['outputs']['pack']['external_url']=proof['meeting.zip']['external_url']
    write(dest/'snapshot.json',snap);write(SITE/'tools/rounds/m1-coupled.json',spec);gen('index',top)
    attr=SITE/'.gitattributes';txt=attr.read_text('utf8')
    for path in ('dist/'+ROUTE+'media/* -text','source-evidence/'+ID+'/* -text'):
        if path not in txt:txt+='\n'+path+'\n'
    attr.write_text(txt,'utf8',newline='\n')
    h=(top/'index.html').read_text('utf8');panel=(f'<section id="reviewVideo"><h2>{TITLE}</h2><video controls playsinline preload="metadata" poster="{ROUTE}media/poster.webp" style="width:100%;max-height:610px"><source src="{proof["replay.mp4"]["external_url"]}" type="video/mp4"></video><p><a href="{ROUTE}index.html">轨迹、剖面、曲线与十五趟记录</a> · <a href="{proof["review.pdf"]["external_url"]}">两页审阅 PDF</a></p></section>')
    h,n=re.subn(r'<section id="reviewVideo">.*?</section>',lambda _:panel,h,count=1,flags=re.S);assert n==1
    (top/'index.html').write_text(h,'utf8',newline='\n')
    status=read(top/'status_public.json');status.update(stage=spec['stage'],updated=spec['when'],mode='原生 PX4 SITL + MuJoCo 闭环',status='awaiting_review',progress=100,detail=spec['summary'],
        geometry=ORIGIN+'/'+ROUTE+'media/m1_hero.webp',geometry_caption='三关工业检修舱与连续残影',
        rolling=ORIGIN+'/'+ROUTE+'media/m1_curves.webp',rolling_caption='机臂动力学与三维净空',replay=None,
        items=['5 种策略 × 3 个配对种子','原始腕部 RGB-D 与外部机位共用几何','12–18 cm 光学工作距','等待 M1 审阅'],
        commits=['881a799 · 感知锁定','951c512 · 单门闭环',rev[:7]+' · M1 结果'])
    write(top/'status_public.json',status)
    sys.path.insert(0,str(ROOT.parent/'D_整机几何与规划验收_20261118'))
    from quota_snapshot import read_windows
    try:q=read_windows()
    except (OSError,RuntimeError):q=None
    if q:write(top/'usage_public.json',q)
    assert all(sha(k)==v for k,v in prev.items());gen('verify',top)
    size=sum(p.stat().st_size for p in top.rglob('*') if p.is_file());assert size<256*1024**2
    write(ROOT/'qa/site-verified.json',{'branch_commit':rev,'unchanged_previous_rounds':len(prev),'bytes':size,'media':proof})
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=('build','publish'));a.add_argument('rev',nargs='?');v=a.parse_args()
    (build() if v.action=='build' else publish(v.rev))
