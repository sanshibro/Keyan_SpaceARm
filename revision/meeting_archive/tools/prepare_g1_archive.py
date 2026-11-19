"""Append a verified round to the existing public research monitor."""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime,timezone,timedelta
from pathlib import Path
import pandas as pd
import numpy as np


def write(path,data):
    path.write_bytes((json.dumps(data,ensure_ascii=False,indent=2)+'\n').encode('utf8'))


def main(stage,site):
    dist=site/'dist';source=stage/'source';review=stage/'review';media=stage/'meeting'
    media.mkdir(exist_ok=True)
    tag='2026-11-19-g1'
    revision='980a4e852e77fef8d77dea2b2130d44c8adf4253'
    manifest=site/'tools/rounds/g1.json'
    when=json.loads(manifest.read_bytes())['when'] if manifest.exists() else datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    files=[]
    def add(kind,path,name,title,caption,**kwargs):
        shutil.copyfile(path,media/name)
        files.append(dict(kind=kind,file=name,title=title,caption=caption,**kwargs))
    add('image',review/'closest.png','closest.png','直臂贴近门楣，夹爪最接近门槛','预先选定的噪声种子 241119。红线连接完整三维凸体几何的最近点，各画面取各自最近时刻。')
    add('image',review/'matched.png','matched.png','在同一时刻观察两种时序','飞行第 28 秒。相机固定在同一世界坐标，左右两图使用相同尺度。绿色为夹爪轨迹，十字为门后的目标。')
    add('curve',review/'curves.png','curves.png','净空、执行弯角与三组配对结果','前三幅显示种子 241119 的保守净空曲线和实际弯角。最后一幅连接各配对种子的精细凸体距离。')
    add('video',review/'comparison.mp4','comparison.mp4','同步观看过门及门后夹爪到位','1 倍速，从离地后第 14 秒到第 46 秒。中剖面省略侧柱，物理碰撞模型包含全部四面门框。',poster='matched.png')
    add('document',review/'review.pdf','review.pdf','两页组会结果及因果审查','第一页给出最近点和四种方案的结果，第二页解释为什么相同弯角能留下不同的夹爪净空。')
    for name,title,caption in [
        ('metrics.csv','十三趟飞行逐次评分','协议通过、接触、部件净空下界、截止时夹爪误差、姿态与到位时间。宽门链路检查单列。'),
        ('convex-distance.json','三维凸体间最近点','每个运行列出机身、臂与工具最近点的两端坐标、仿真时间与数值夹逼区间。'),
        ('pose-swap.json','交换机身与臂形的几何诊断','三组配对逐一重放。两个混合状态为离线重构，原始两组来自独立的 PX4 飞行。'),
        ('plans.json','飞行前冻结的路径与名义净空','直臂、固定弯臂和可变形方案均独立搜索机身高度；常规与调时共用机身参考和截止时间。'.replace('；','，')),
        ('audit.json','PX4、MuJoCo 与 ROS 日志核对','逐次记录四项 ROS 主题检查、ULog 电机对时、预登记通过条件和原始文件指纹。'),
        ('evidence-index.json','原始压缩包指纹','SHA-256 及文件大小，可用来核对逐趟下载的原始 ULog、ROS bag、HIL 与物理状态。')]:
        add('data',stage/name,name,title,caption)
    add('data',source/'round_g1.json','protocol.json','飞行前冻结的任务定义','包含统一的目标位姿、门框、阈值、噪声顺序和评分截止时间。')
    add('data',source/'round_g1_review.txt','notes.txt','详细解读、证据边界及复现命令','说明单门搜索的适用范围、离线反事实、后续论文安排和新目录复现步骤。')
    series={};columns=None
    for f in sorted((stage/'traces').glob('*.csv')):
        frame=pd.read_csv(f)
        if columns is None:columns=list(frame)
        assert columns==list(frame)
        series[f.stem]=frame.to_numpy()
    assert len(series)==13
    np.savez_compressed(stage/'all-traces.npz',columns=np.array(columns),**series)
    add('data',stage/'all-traces.npz','all-traces.npz','十三趟 50 Hz 曲线数据','每趟一张数组和共用列名，含三维机身与夹爪位置、弯角、三组部件净空、越门程度和目标误差。')
    names=dict(wide='宽门链路检查',straight='直臂',fixed='固定弯臂',variable_nominal='可变形常规时序',variable_timed='可变形提前调时')
    for f in sorted((stage/'evidence').glob('*.tar.gz')):
        seed,mode=f.name.split('.')[0].split('-')
        add('data',f,f.name,seed+' · '+names[mode]+'原始包','原生 ULog 和 bag、50 Hz 广义状态、250 Hz HIL、事件、版本与控制台记录。',separate=True)
    spec=dict(schema=1,id=tag,stage='G1 · 已知单门闭环',title='同一扇门与同一目标，机身和臂各留出多少空间？',
        when=when,summary='12 趟窄门均无接触并到达门后目标。固定 40° 已能通过。配对调时后，臂与爪的最小凸体净空增加 3.91–4.14 cm。',
        guide=['先看固定世界机位视频和最近点。红线连接实际注册的部件与门框。','对照净空曲线和两页审查。固定弯臂同样通过，直臂的差异在机身到门楣的余量。','看机身与臂形交换结果，区分物理闭环运行与离线几何归因。'],
        scope='1 趟宽门链路检查，加上 3 个噪声种子、4 种方案的窄门比较。标称空载与已知地图。整机运动源自实际运行的 PX4 软件在环；距离按 50 Hz 状态复核。'.replace('；','，'),
        revision=revision,source_note='980d9bc 冻结协议，2976b30 保存飞行原始证据，980a4e8 补充姿态交换、图与视频。',cover='closest.png',files=files)
    write(manifest,spec)
    attrs=site/'.gitattributes';rule='dist/archive/'+tag+'/media/* -text'
    a=attrs.read_text(encoding='utf8').splitlines()
    if rule not in a:attrs.write_bytes(('\n'.join(a+[rule])+'\n').encode())
    gen=site/'tools/meeting_archive.py'
    subprocess.run([sys.executable,str(gen),'add','--site',str(dist),'--spec',str(manifest),'--source',str(media)],check=True)
    subprocess.run([sys.executable,str(gen),'verify','--site',str(dist)],check=True)
    url='archive/'+tag+'/';home=dist/'index.html';h=home.read_text(encoding='utf8')
    block=('<section id="reviewVideo"><h2>单门穿越与夹爪到位</h2><p>同一世界机位、共同目标和 52 秒时限。对比常规与提前弯臂、延后收臂两种时序。</p>'
           '<video controls playsinline preload="metadata" poster="'+url+'media/matched.png" style="display:block;width:100%;max-height:600px;background:white"><source src="'+url+'media/comparison.mp4" type="video/mp4"></video>'
           '<p><a href="'+url+'index.html">打开本轮记录、曲线和原始日志</a> · <a href="'+url+'media/review.pdf">阅读两页审查</a></p></section>')
    h,n=re.subn(r'<section id="reviewVideo">.*?</section>',lambda _:block,h,count=1,flags=re.S);assert n==1
    prefix='<section><h2>阶段文件</h2><p>';assert prefix in h
    link='<a href="'+url+'media/review.pdf" target="_blank">G1 单门两页审查</a>　'
    if link not in h:h=h.replace(prefix,prefix+link,1)
    home.write_bytes(h.encode('utf8'))
    status=json.loads((dist/'status_public.json').read_bytes())
    status.update(status='awaiting_review',progress=100,stage='G1 完成 · 已知单门闭环',
        mode='SUPER 与 Qingyu · PX4 EKF2 软件在环 · MuJoCo 自由飞行',detail=spec['summary'],updated=when,
        geometry=url+'media/closest.png',geometry_caption='直臂与门楣、夹爪与门槛的三维最近点',
        rolling=url+'media/curves.png',rolling_caption='执行净空、实际弯角与三次配对结果',replay=None,
        items=['1 趟宽门链路检查和 12 趟窄门对照','相同 52 s 时限与门后夹爪目标','全部飞行接触积分步为 0','实际累计弯曲最大 72.71°'],
        commits=['980d9bc · 预登记与单门规划','2976b30 · 十三趟原始飞行','980a4e8 · 可视化与因果审查'])
    write(dist/'status_public.json',status)
    sys.path.insert(0,str(stage.parent/'D_整机几何与规划验收_20261118'))
    from quota_snapshot import read_windows
    try:write(dist/'usage_public.json',read_windows())
    except (OSError,RuntimeError) as err:print('Kept prior dated quota snapshot:',type(err).__name__)
    shutil.copyfile(dist/'archive'/tag/'meeting.zip',stage/'G1_组会离线包.zip')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',type=Path);p.add_argument('site',type=Path)
    a=p.parse_args();main(a.stage,a.site)
