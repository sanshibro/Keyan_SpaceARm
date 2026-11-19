"""Build immutable offline evidence, and publish the same round in the group meeting archive."""
import argparse,hashlib,json,re,shutil,subprocess,sys,tarfile,time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone,timedelta
from pathlib import Path
from urllib.request import Request,urlopen
TAG='2026-11-19-slim'
TITLE='中段预拉伸应变片怎样观测弯曲与回程？'
SUMMARY='把手册的电阻曲线、实物采集盒和同轨仿真对齐，比较四种安装跨度，并单独检验加载、卸载和定长保持。'
BASE='https://raw.githubusercontent.com/sanshibro/Keyan_SpaceARm/'
def read(path):return json.loads(path.read_bytes())
def write(path,x):path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(gen,op,site,*a):subprocess.run([sys.executable,str(gen),op,'--site',str(site),*map(str,a)],check=True)
def build(stage,revision):
    work=stage/'canonical';src=work/'sources';src.mkdir(parents=True,exist_ok=True);entries=[]
    def add(kind,p,name,title,caption,**kw):
        shutil.copyfile(p,src/name);entries.append(dict(kind=kind,file=name,title=title,caption=caption,**kw))
    for kind,name,title,caption in [
        ('image','mechanism.webp','从贴片安装到电阻变化','MuJoCo 臂体中段的四条表面路径、25 °C 产品曲线及 24 个弯曲平面的应变范围。'),
        ('curve','memory.webp','相同长度下的加载与卸载','受控往返的支路、定长保持，以及另一段真实采集记录的独立检验。'),
        ('curve','curves.webp','四路采集值怎样变成弯角','同一条 G1 轨迹上对齐虚拟动捕、采集盒、腕部视觉与零点改变后的融合输出。'),
        ('document','review.pdf','两页组会审阅','产品资料、四种安装、静态与动态传感器在遮挡中的结果，以及实机标定步骤。')]:
        add(kind,stage/'review'/name,name,title,caption)
    add('video',stage/'review/replay.mp4','replay.mp4','32 秒同步回放','同一时钟上的整机、腕部相机、四路模拟读数与弯曲量。此视频展示 30% 预拉伸的静态材料基线。',poster='mechanism.webp')
    for name,title,cap in [('audit.json','几何与采样独立检查','包含触底与松弛、刚体变换不变性和遮挡覆盖。'),
         ('bench-scores.json','台架角度预测与安装跨度','8 段训练、2 段验证、3 段整段测试。'),('flight-scores.json','四趟同轨对照与采集盒灵敏度','完整分阶段评分、各模式覆盖率与零点敏感度。'),
         ('memory-scores.json','迟滞、回落及真实采集盒迁移','参数仅由第一个历史文件拟合，第二个文件检验。含同向与反向配对。'),
         ('tracking-check.json','指令与真实运动轨迹的差异','已下发指令与虚拟动捕比较，和估计误差分别计算。'),
         ('sweep.json','120° 扫角结果','四种预拉伸的内、外侧范围与首次松弛角度。'),
         ('box-profile.json','柔致采集盒历史特性','串口采样、三个实物输入的有效增益与跨记录残差。'),
         ('datasheet-plateaus.json','厂家重复拉伸图的复读记录','第 2 页图上平台的像素坐标及读图精度。'),
         ('evidence-index.json','原始证据包校验','四个归档文件的成员数、字节数和 SHA-256。')]:
        add('data',stage/name,name,title,cap)
    for name,short,title in [('round_slim_protocol.json','protocol.json','安装与同轨基线协议'),('round_slim_memory_scope.json','memory-scope.json','迟滞、时间尺度与适用范围')]:
        add('data',stage/'source'/name,short,title,'区分设备规格、历史采集量与本次试验设定。')
    for name,title in [('slim-material-paths.tar.gz','原始表面轨迹'),('slim-bench.tar.gz','台架训练与留出测试'),('slim-flights.tar.gz','四次任务重放'),('slim-memory.tar.gz','动态传感器与留出检验')]:
        add('data',stage/'evidence'/name,name,title,'可复核数值与原始数组，单独下载。',separate=True)
    spec=work/'slim.json';when=read(spec)['when'] if spec.is_file() else datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    write(spec,dict(schema=1,id=TAG,stage='G1 · 电阻式应变本体感知',when=when,title=TITLE,summary=SUMMARY,revision=revision,parent_round='2026-11-19-wrist-slam',
        cover='mechanism.webp',files=entries,guide=['看中段安装图，比较 25、32.5 和 35 mm 的机械余量。','对照加载、卸载曲线及实际串口记录，区分材料回程和采集盒零点。','打开两页 PDF 对照视觉中断的覆盖率与角度误差。'],
        scope='四次已有单门飞行重放。迟滞与保持回落的幅度有条件假设，实物盒记录只有三个有效通道，四条新装贴片仍待定长标定。',
        source_note='c2f2828 登记主模型，5d4ccc9 接入采集盒，c691c34 保存原始数据，04787dc 检查往返历史与跟踪误差。'))
    gen=stage/'site-cache/tools/meeting_archive.py';dist=work/'dist'
    run(gen,'add',dist,'--spec',spec,'--source',src);run(gen,'verify',dist)
    folder=dist/'archive'/TAG;shutil.copyfile(folder/'meeting.zip',stage/'组会离线包.zip')
    with tarfile.open(work/'meeting-mirror.tar.gz','w:gz') as tar:
        for f in sorted(folder.rglob('*')):
            if f.is_file() and not f.name.endswith('.tar.gz'):tar.add(f,arcname=f.relative_to(folder).as_posix())
def confirm(url,local):
    for attempt in range(4):
        try:
            with urlopen(Request(url,headers={'User-Agent':'ResearchMeetingArchive/1.0'}),timeout=25) as r:data=r.read()
            if len(data)!=local.stat().st_size or hashlib.sha256(data).hexdigest()!=sha(local):raise ValueError('Remote checksum mismatch')
            return dict(bytes=len(data),sha256=sha(local),external_url=url)
        except Exception:
            if attempt==3:raise
            time.sleep(2+attempt)
def publish(stage,rev):
    site=stage/'site-cache';dist=site/'dist';native=stage/'canonical/dist/archive'/TAG;dest=dist/'archive'/TAG;link='archive/'+TAG+'/'
    prev={p:sha(p) for p in (dist/'archive').glob('*/snapshot.json') if p.parent!=dest}
    assert len(prev)==13,len(prev);snap=read(native/'snapshot.json');spec=read(stage/'canonical/slim.json')
    data=BASE+rev+'/revision/data/round_slim_sensor/';arch=BASE+rev+'/revision/meeting_archive/archive/'+TAG+'/'
    external={n:(data+'evidence/'+n,stage/'evidence'/n) for n in read(stage/'evidence-index.json')}
    external['flight-scores.json']=(data+'flight-scores.json',stage/'flight-scores.json');external['meeting.zip']=(arch+'meeting.zip',native/'meeting.zip')
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs={n:pool.submit(confirm,*args) for n,args in external.items()};proof={n:f.result() for n,f in jobs.items()}
    cached=site/'source-evidence'/TAG;write(cached/'verified.json',proof);shutil.copyfile(native/'meeting.zip',cached/'meeting.zip')
    write(stage/'public-raw-verification.json',proof)
    page=(native/'index.html').read_text(encoding='utf-8');files={f['file']:f for f in spec['files']}
    (dest/'media').mkdir(parents=True,exist_ok=True)
    for f in snap['files']:
        n=f['file']
        if n in external:
            report=proof[n];assert f['sha256']==report['sha256'] and f['bytes']==report['bytes']
            f['external_url']=files[n]['external_url']=report['external_url'];page=page.replace('href="media/'+n+'"','href="'+report['external_url']+'"')
        else:shutil.copyfile(native/'media'/n,dest/'media'/n)
    entry=snap['outputs']['pack'];entry['external_url']=proof['meeting.zip']['external_url']
    page=page.replace('href="meeting.zip"','href="'+entry['external_url']+'"')
    (dest/'index.html').write_bytes(page.encode());snap['outputs']['page']['sha256']=sha(dest/'index.html')
    write(dest/'snapshot.json',snap);write(site/'tools/rounds/slim-mount.json',spec)
    attrs=site/'.gitattributes'
    with attrs.open('ab') as fo:
        for row in ['dist/archive/'+TAG+'/media/* -text','source-evidence/'+TAG+'/* -text']:
            if row not in attrs.read_text(encoding='utf-8'):fo.write(('\n'+row+'\n').encode())
    gen=site/'tools/meeting_archive.py';run(gen,'index',dist)
    home=dist/'index.html';html=home.read_text(encoding='utf-8')
    block=('<section id="reviewVideo"><h2>'+TITLE+'</h2><video controls playsinline preload="metadata" poster="'+link+'media/mechanism.webp" style="display:block;width:100%;max-height:600px;background:white"><source src="'+link+'media/replay.mp4" type="video/mp4"></video><p><a href="'+link+'index.html">本轮图像与原始数据</a> · <a href="'+link+'media/memory.webp">加载与卸载曲线</a> · <a href="'+link+'media/review.pdf">两页组会 PDF</a></p></section>')
    html,n=re.subn(r'<section id="reviewVideo">.*?</section>',lambda _:block,html,count=1,flags=re.S);assert n==1
    key='<section><h2>阶段文件</h2><p>';assert key in html
    item='<a href="'+link+'media/review.pdf" target="_blank">中段预拉伸与迟滞审阅</a>　'
    if item not in html:html=html.replace(key,key+item,1)
    home.write_bytes(html.encode())
    state=read(dist/'status_public.json');state.update(status='awaiting_review',stage='G1 · 四路预拉伸应变与迟滞复核',mode='同轨重放 · 厂商曲线 · 采集盒独立验证',
        detail=SUMMARY,updated=spec['when'],progress=100,geometry=link+'media/mechanism.webp',geometry_caption='中段四条 32.5 mm 表面路径及内外侧应变范围',
        rolling=link+'media/memory.webp',rolling_caption='相同长度下的去程、回程与保持回落',replay=None,
        items=['25 mm 敏感段使用 30% 中立预拉伸','实际采集盒记录约 40.2 Hz，三个通道可交叉验证','四次同轨回放中将控制误差和估计误差分别计算','下一步采集四通道定长往返与定长保持记录'],
        commits=['5d4ccc9 · 中段贴片及采集盒','c691c34 · 扫角与同轨原始数据','04787dc · 迟滞及独立实物记录复核'])
    write(dist/'status_public.json',state)
    sys.path.insert(0,str(stage.parent/'D_整机几何与规划验收_20261118'));from quota_snapshot import read_windows
    try:quota=read_windows()
    except (OSError,RuntimeError):print('Previous timestamped account snapshot retained')
    else:write(dist/'usage_public.json',quota)
    assert all(sha(p)==h for p,h in prev.items());run(gen,'verify',dist)
    size=sum(f.stat().st_size for f in dist.rglob('*') if f.is_file());assert size<=256*1024**2,size
    print('Verified',len(prev)+1,'rounds;',size,'hosted bytes')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','publish']);p.add_argument('stage',type=Path);p.add_argument('revision')
    args=p.parse_args();(build if args.action=='build' else publish)(args.stage.resolve(),args.revision)
