"""Publish a paired native-flight handoff review with its self-contained offline copy."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
from urllib.parse import quote

from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
ID='2026-11-21-m2-handoff-response'
OLD=Path('qa/layered-pilot03/figure')
NEW=Path('qa/servo-pilot07/figure')
TRACE=OLD/'handoff-audit'
FILES=[
    ('video','original.mp4',OLD/'三关开发飞行_同源双视角与腕部.mp4','原版开发飞行','约在播放第 10.20 秒和 17.92 秒观察两次接管。画面下方同时标出曝光时刻和到达时刻。'),
    ('video','revised.mp4',NEW/'新版先导_三关接管实录.mp4','同种子新版先导飞行','四次接管使用 0.80 至 1.45 秒的过渡。原始腕部 RGB 与外部视角逐帧对时。'),
    ('curve','response.webp',TRACE/'handoff-response.png','机身和柔性臂怎样响应接管指令','浅蓝色为桥接时段，竖虚线为两张截图的曝光时刻。两列纵轴使用相同尺度。'),
    ('document','response.pdf',TRACE/'handoff-response.pdf','接管响应曲线 PDF','矢量曲线可放大核对指令加速度、机身倾角和弯曲幅值。'),
    ('data','frames.json',TRACE/'sources-and-measurements.json','两帧原始数据与切换事件','曝光时钟、视觉更新、桥接端点、实测姿态和四路电机推力。'),
    ('data','switches.json',TRACE/'paired-development-handoffs.json','两版全部接管的同口径统计','姿态响应窗口从接管开始至桥接结束后 0.65 秒，两趟均为种子 26112207。')]

def save(path, value):
    path.write_bytes((json.dumps(value,indent=2,ensure_ascii=False)+'\n').encode('utf8'))

def unpack_module(checkout):
    spec=importlib.util.spec_from_file_location('meeting_archive',checkout/'tools/meeting_archive.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def url(commit,path):
    if not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('Expected fixed Git commit')
    return 'https://raw.githubusercontent.com/sanshibro/Keyan_SpaceARm/'+commit+'/revision/m2_multipassage/'+quote(path.as_posix())

def prepare(checkout,commit):
    api=unpack_module(checkout);dest=ROOT/'meeting-handoffs';src=dest/'assets'
    src.mkdir(parents=True,exist_ok=True)
    for _,name,path,*rest in FILES:
        if name.endswith('.webp'):
            with Image.open(ROOT/path) as image:
                image.thumbnail((1460,1060));image.save(src/name,quality=92,method=6)
        else:shutil.copyfile(ROOT/path,src/name)
    local=dest/'offline-source';local.mkdir(exist_ok=True)
    mtime=(ROOT/FILES[1][2]).stat().st_mtime
    when=datetime.fromtimestamp(mtime,timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    spec=dict(schema=1,id=ID,title='两次滚动接管为何引起机身摆动',stage='M2 · 接管动态复核',revision=commit,when=when,
              summary='对齐 31.20 秒与 38.92 秒的腕部曝光、规划指令和物理响应。新版四次接管的指令加速度峰值为 0.19 至 0.29 m/s²，对应机身倾角峰值为 1.84 至 4.29°。',
              scope='同一开发种子的两个软件版本。新版通过三关，末端最小外障净空实测 4.42 cm，先导要求 6.20 cm。',
              guide=['先看第一段录像在两次换轨时飞机如何侧倾，再将画面时钟与曲线对齐。','第二次切换前有 0.82 秒等待。接管同时要求横移、爬升和收臂。','查看第二段录像及全部接管数据，最后单独汇报法兰净空。'],
              source_note='PX4 SITL、MuJoCo 与原生 ORB-SLAM3 开发飞行。视频以光学曝光时的真实仿真位姿生成。',cover='response.webp',
              files=[dict(kind=k,file=n,title=s,caption=c) for k,n,p,s,c in FILES])
    save(dest/'round.json',spec);api.add(local,dest/'round.json',src)
    folder=local/'archive'/ID
    shutil.copyfile(folder/'meeting.zip',dest/'meeting.zip')
    print(hashlib.sha256((dest/'meeting.zip').read_bytes()).hexdigest())

def publish(checkout,commit,pack_commit):
    api=unpack_module(checkout);dest=ROOT/'meeting-handoffs';dist=checkout/'dist'
    local=dest/'offline-source/archive'/ID;folder=dist/'archive'/ID
    receipt=api.read(local/'snapshot.json')
    if receipt['revision']!=commit:raise ValueError('Wrong evidence revision')
    if not folder.exists():shutil.copytree(local,folder)
    elif not (folder/'snapshot.json').is_file():raise ValueError('Unexpected existing archive')
    evidence=checkout/'source-evidence'/ID;evidence.mkdir(parents=True,exist_ok=True)
    map_path={n:p for _,n,p,_,_ in FILES}
    for row in receipt['files']:
        if row['kind'] not in ('video','document'):continue
        name=row['file'];row['external_url']=url(commit,map_path[name]);private=evidence/name
        shutil.copyfile(local/'media'/name,private)
        if api.sha(private.read_bytes())!=row['sha256']:raise ValueError('Media bytes changed')
        (folder/'media'/name).unlink(missing_ok=True)
    pack=evidence/'meeting.zip';shutil.copyfile(local/'meeting.zip',pack)
    if api.sha(pack.read_bytes())!=receipt['outputs']['pack']['sha256']:raise ValueError('Offline package bytes changed')
    pack_url=url(pack_commit,Path('meeting-handoffs/meeting.zip'))
    receipt['outputs']['pack']['external_url']=pack_url
    (folder/'meeting.zip').unlink(missing_ok=True)
    page=api.detail(receipt).replace('href="meeting.zip"','href="'+pack_url+'"')
    (folder/'index.html').write_bytes(page.encode('utf8'))
    receipt['outputs']['page']['sha256']=api.sha(page.encode('utf8'))
    save(folder/'snapshot.json',receipt);api.index(dist)
    api.all_rounds(dist,True)
    status=api.read(dist/'status_public.json')
    status.update(stage=receipt['stage'],updated=receipt['when'],mode='原生闭环 · 接管动态',status='development',
                  detail=receipt['summary'],rolling='archive/'+ID+'/media/response.webp',rolling_caption='两次换轨同步诊断')
    save(dist/'status_public.json',status)
    total=sum(f.stat().st_size for f in dist.rglob('*') if f.is_file())
    if total>256*1024**2:raise ValueError('Public archive exceeds site capacity')
    print(ID,total)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','publish'));p.add_argument('checkout',type=Path)
    p.add_argument('research_commit');p.add_argument('--pack-commit');a=p.parse_args()
    if a.mode=='prepare':prepare(a.checkout,a.research_commit)
    else:publish(a.checkout,a.research_commit,a.pack_commit)
