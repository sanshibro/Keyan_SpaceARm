"""Publish this single round using the existing archive schema and checksum verifier."""
import argparse,hashlib,json,re,shutil,subprocess,sys,tarfile,time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone,timedelta
from pathlib import Path
from urllib.request import Request,urlopen
TAG='2026-11-19-wrist-slam'
TITLE='视觉失效时，柔性臂怎样保持形态估计？'
SUMMARY='四趟单门飞行重放中，遮挡 4 秒时原地图视觉输出中断，融合每个时刻仍有估计，弯曲量 RMSE 为 0.50°。'
BASE='https://raw.githubusercontent.com/sanshibro/Keyan_SpaceARm/'
def read(p):return json.loads(p.read_bytes())
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(gen,op,site,*args):subprocess.run([sys.executable,str(gen),op,'--site',str(site),*map(str,args)],check=True)
def build(stage,revision):
    work=stage/'canonical';src=work/'sources';src.mkdir(parents=True,exist_ok=True)
    entries=[]
    def add(kind,p,name,title,caption,**kwargs):
        shutil.copyfile(p,src/name)
        entries.append(dict(kind=kind,file=name,title=title,caption=caption,**kwargs))
    for kind,name,title,caption in [
        ('image','mechanism.png','遮挡前、遮挡中与重新建图','同一趟运行的侧视剖面与末端相机原始输入。'),
        ('curve','gauges.png','四条表面应变路径怎样产生读数','独立台架测试动作。四通道 ADC 相对零点的变化、训练模型输出与虚拟动捕逐时对应。'),
        ('curve','curves.png','视觉失效前后的弯曲量、误差和可用性','同一时钟的虚拟动捕、应变、原地图视觉与融合。阴影为视觉遮挡对应的输出时段。'),
        ('document','review.pdf','两页组会审阅','训练与评分数据、四组对照、遮挡恢复和单门净空的定量衔接。')]:
        add(kind,stage/'review'/name,name,title,caption)
    add('video',stage/'review/comparison.mp4','comparison.mp4','32 秒同步录像','原始仿真飞行轨迹、实际进入 SLAM 的腕部图像和融合弯曲量，同步回放。',poster='mechanism.png')
    for name,title,caption in [('markerless-scores.json','四趟逐组误差与覆盖率','正常与遮挡分别评分，包含相同可见帧和弯曲平面可辨时刻。'),
            ('bench-scores.json','独立台架评分','8 段训练、2 段验证、3 段测试。'),('geometry.json','整机净空与判定代价','80 个门区时刻乘以 33 种假设门槛高度；沿用此前校准的经验余量。'),
            ('audit.json','时钟、前缀及坐标不变性复核','逐趟记录原地图恢复情况和角度范围，检查不使用未来观测。'),
            ('evidence-index.json','原始数据包校验值','四份证据包的字节数、成员数和 SHA-256。')]:
        add('data',stage/name,name,title,caption)
    add('data',stage/'source/round_wrist_slam_protocol.json','protocol.json','无标签对照协议','试验前冻结光学遮挡时段、训练划分、初始坐标系与评价范围。')
    for file,title,cap in [('bench-training.tar.gz','虚拟台架训练和测试数据','13 段原始读数、独立动捕点簇、模型和台架评估。'),
        ('flight-sensors.tar.gz','飞行传感器和动捕数据','四路应变采样、独立评分真值、六趟记录、标签诊断与净空。'),
        ('orb-slam-results.tar.gz','无标签 SLAM 跟踪与融合数据','原生逐帧跟踪状态、地图编号、重投影误差、估计结果、RGB-D 样本及完整图像指纹。'),
        ('tagged-reference.tar.gz','有标签定位诊断','保留用于核对相机外参和解角链的早期对照。')]:
        add('data',stage/'evidence'/file,file,title,cap,separate=True)
    spec=work/'wrist.json';when=read(spec)['when'] if spec.exists() else datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    write(spec,dict(schema=1,id=TAG,when=when,stage='G1 · 柔性臂本体感知',title=TITLE,summary=SUMMARY,
        revision=revision,parent_round='2026-11-19-g1-online',cover='mechanism.png',files=entries,
        guide=['从四路应变曲线查看独立动捕的训练标签。','按下同步录像，观察第 31 秒遮挡与第 35 秒恢复时的地图编号变化。','比较仅应变与相同动态滤波器的结果，区分感知冗余和滤波平滑的贡献。'],
        scope='仿真应变采集与动捕，四趟既有 PX4/MuJoCo 轨迹回放，带非编码纹理的已知单门。退出遮挡后追踪新地图，恢复原地图坐标前继续使用应变。',
        source_note='04d4df3 登记传感链，16b2796 登记无标签 SLAM 对照，8980baa 接入原生 ORB-SLAM3 与延迟量测，ac47047 保存审查与证据。'))
    gen=stage/'site-cache/tools/meeting_archive.py';dist=work/'dist'
    if not (dist/'archive'/TAG).is_dir():run(gen,'add',dist,'--spec',spec,'--source',src)
    run(gen,'verify',dist)
    folder=dist/'archive'/TAG;shutil.copyfile(folder/'meeting.zip',stage/'组会离线包.zip')
    with tarfile.open(work/'meeting-mirror.tar.gz','w:gz') as a:
        for f in sorted(folder.rglob('*')):
            if f.is_file() and not f.name.endswith('.tar.gz'):a.add(f,arcname=f.relative_to(folder).as_posix())
def verify_public(url,local):
    for i in range(3):
        try:
            with urlopen(Request(url,headers={'User-Agent':'ResearchArchive/1.0'}),timeout=35) as resp:
                h=hashlib.sha256();n=0
                while True:
                    b=resp.read(262144)
                    if not b:break
                    h.update(b);n+=len(b)
            assert h.hexdigest()==sha(local) and n==local.stat().st_size
            return dict(bytes=n,sha256=h.hexdigest(),external_url=url)
        except Exception:
            if i==2:raise
            time.sleep(2+i)
def publish(stage,revision):
    site=stage/'site-cache';dist=site/'dist';native=stage/'canonical/dist/archive'/TAG;dest=dist/'archive'/TAG
    previous={p:sha(p) for p in (dist/'archive').glob('*/snapshot.json') if p.parent!=dest}
    assert len(previous)==12
    snaps=read(native/'snapshot.json');spec=read(stage/'canonical/wrist.json')
    assets=read(stage/'evidence-index.json')
    with ThreadPoolExecutor(max_workers=4) as workers:
        future={n:workers.submit(verify_public,BASE+revision+'/revision/data/round_shape_clearance/evidence/'+n,stage/'evidence'/n) for n in assets}
        proof={n:v.result() for n,v in future.items()}
    cache=site/'source-evidence'/TAG;write(cache/'verified.json',proof);write(stage/'public-raw-verification.json',proof)
    dest.mkdir(exist_ok=True,parents=True);(dest/'media').mkdir(exist_ok=True)
    page=(native/'index.html').read_text(encoding='utf-8');files={i['file']:i for i in spec['files']}
    for f in snaps['files']:
        name=f['file']
        if name in proof:
            p=proof[name];assert f['sha256']==p['sha256'] and f['bytes']==p['bytes']
            f['external_url']=files[name]['external_url']=p['external_url']
            page=page.replace('href="media/'+name+'"','href="'+p['external_url']+'"')
        else:shutil.copyfile(native/'media'/name,dest/'media'/name)
    shutil.copyfile(native/'meeting.zip',dest/'meeting.zip')
    (dest/'index.html').write_bytes(page.encode());snaps['outputs']['page']['sha256']=sha(dest/'index.html')
    write(dest/'snapshot.json',snaps);write(site/'tools/rounds/wrist-slam.json',spec)
    attrs=site/'.gitattributes'
    with attrs.open('ab') as f:
        for line in ('dist/archive/'+TAG+'/media/* -text','source-evidence/'+TAG+'/* -text'):
            if line not in attrs.read_text(encoding='utf-8'):f.write(('\n'+line+'\n').encode())
    gen=site/'tools/meeting_archive.py';run(gen,'index',dist)
    link='archive/'+TAG+'/'
    home=dist/'index.html';html=home.read_text(encoding='utf-8')
    block=('<section id="reviewVideo"><h2>'+TITLE+'</h2><video controls playsinline preload="metadata" poster="'+link+'media/mechanism.png" style="display:block;width:100%;max-height:600px;background:white"><source src="'+link+'media/comparison.mp4" type="video/mp4"></video><p><a href="'+link+'index.html">本轮图像、数据与离线包</a> · <a href="'+link+'media/gauges.png">四路应变的实际读数</a> · <a href="'+link+'media/review.pdf">两页组会审阅</a></p></section>')
    html,n=re.subn(r'<section id="reviewVideo">.*?</section>',lambda _:block,html,count=1,flags=re.S);assert n==1
    prefix='<section><h2>阶段文件</h2><p>';assert prefix in html
    item='<a href="'+link+'media/review.pdf" target="_blank">腕部视觉与应变融合审阅</a>　'
    if item not in html:html=html.replace(prefix,prefix+item,1)
    home.write_bytes(html.encode());state=read(dist/'status_public.json')
    state.update(status='awaiting_review',stage='G1 · 本体感知对照完成',mode='真值训练应变 · 无标签 ORB-SLAM3 · 自适应融合',
        updated=spec['when'],progress=100,detail=SUMMARY,geometry=link+'media/gauges.png',geometry_caption='四条虚拟表面应变路径与留出动作的 ADC 读数',
        rolling=link+'media/curves.png',rolling_caption='真实算法对遮挡输入的响应',replay=None,
        items=['8 段训练、2 段验证、3 段独立台架测试','4 趟留出轨迹，各自运行正常和遮挡两种视觉输入','遮挡期间应变持续更新','下一轮审阅项为原地图重定位与视觉重新接入'],
        commits=['16b2796 · 无标签对照登记','8980baa · 原生 SLAM 和融合','ac47047 · 证据归档与校验'])
    write(dist/'status_public.json',state)
    sys.path.insert(0,str(stage.parent/'D_整机几何与规划验收_20261118'))
    from quota_snapshot import read_windows
    try:quota=read_windows()
    except (OSError,RuntimeError):print('Previous account snapshot retained with its timestamp')
    else:write(dist/'usage_public.json',quota)
    assert all(sha(p)==h for p,h in previous.items())
    run(gen,'verify',dist)
    size=sum(f.stat().st_size for f in dist.rglob('*') if f.is_file())
    assert size<=256*1024**2,size
    print('Public archive ready; existing snapshots verified:',len(previous),'total bytes:',size)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('build','publish'));p.add_argument('stage',type=Path);p.add_argument('revision')
    a=p.parse_args();(build if a.action=='build' else publish)(a.stage.resolve(),a.revision)
