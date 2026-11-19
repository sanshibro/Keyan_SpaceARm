"""Build the self-contained round once; every presentation asset refers to the scored runs."""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import numpy as np
import pandas as pd

TAG='2026-11-19-g1-online'
TITLE='机身在门前停顿，柔性臂何时收回？'
SUMMARY='21 趟单门飞行均在 52 秒内到位。停顿时按时钟收臂的机体后缘尚距出口 58.2 cm，状态触发时已越过 3.3 cm。'
MODES={'online_nominal':'连续 · 状态触发','clock_nominal':'连续 · 固定时钟','clock_pause':'停顿 · 固定时钟',
       'phase_pause':'停顿 · 指令进度','online_pause':'停顿 · 状态触发','phasebend_pause':'进度弯／在线收','onlinebend_pause':'在线弯／进度收'}


def write(p,obj):
    p.write_bytes((json.dumps(obj,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))


def main(stage,revision):
    stage=stage.resolve()
    checkout=stage/'site/.sites-checkout'
    canonical=stage/'canonical'
    sources=canonical/'sources'
    sources.mkdir(parents=True,exist_ok=True)
    spec=canonical/'g1-online.json'
    when=json.loads(spec.read_bytes())['when'] if spec.exists() else datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    entries=[]
    def add(kind,path,name,title,caption,**extras):
        target=sources/name
        if not target.exists():
            try:
                os.link(path,target)
            except OSError:
                shutil.copyfile(path,target)
        assert hashlib.sha256(target.read_bytes()).digest()==hashlib.sha256(path.read_bytes()).digest()
        entries.append(dict(kind=kind,file=name,title=title,caption=caption,**extras))

    review=stage/'review'
    for kind,name,title,caption in [
        ('image','release.png','三种方法分别在什么位置收臂','种子 241119。门槛和门楣给出同一中央截面，正值表示机体最后方部件已经越过出口面。'),
        ('image','matched.png','同一飞行时刻的三个机体与夹爪','固定世界相机、同一门框。对照三种方法的车辆位置和夹爪轨迹。'),
        ('curve','curves.png','整机后缘、实际弯角与臂爪净空','前三幅为种子 241119。末幅连接三组噪声种子在不同方法下的最小凸体净空。'),
        ('document','review.pdf','两页组会审阅','第一页比较收臂位置、净空与到位时间。第二页用交叉对照定位最近点发生的阶段。')]:
        add(kind,review/name,name,title,caption)
    add('video',review/'comparison.mp4','comparison.mp4','门前停顿与恢复前进的同步录像','三个独立 PX4 运行在相同仿真时钟回放。窗口覆盖飞行第 19 到 50 秒，按实际速度播放。',poster='matched.png')
    for name,title,caption in [
        ('all-metrics.csv','21 趟逐次评分','每行一趟，含保守与凸体净空、最大弯角、收臂位置、到位时间和计算耗时。'),
        ('factorial.json','弯臂与收臂的交叉对照','每个种子四个组合。两组复用主试验，另两组独立补飞。'),
        ('engineering-checks.json','触发前缀、几何及状态故障检查','检查最近点和收臂的先后顺序、前缀指令逐拍一致性以及过期状态或门框封堵时的行为。'),
        ('all-audit.json','PX4、MuJoCo、ROS 与决策一致性','核对原生 ULog、物理状态和控制输入，重放在线决策并核对同路线的机身指令。'),
        ('all-evidence-index.json','逐趟原始证据 SHA-256','全部 21 个压缩包按文件名给出原始字节数与指纹。')]:
        add('data',stage/name,name,title,caption)
    for origin,dest,title,caption in [
        ('round_g1_online.json','protocol.json','主试验协议','运行前冻结的 15 趟顺序、两种机身进度与 5 Hz 状态触发规则。'),
        ('round_g1_online_split.json','split-protocol.json','交叉试验补充协议','主试验之后登记的 6 趟，沿用同一门框和截止时间。'),
        ('round_g1_online_notes.txt','notes.txt','组会讲解与复现入口','说明配对、判据、证据范围和在新目录复现的方式。')]:
        add('data',stage/'source'/origin,dest,title,caption)
    frames={}
    columns=None
    for f in sorted((stage/'traces').glob('*.csv')):
        frame=pd.read_csv(f)
        if columns is None:
            columns=list(frame)
        assert list(frame)==columns
        frames[f.stem]=frame.to_numpy()
    assert len(frames)==21 and not any(v.dtype.kind=='O' for v in frames.values())
    trace=stage/'all-traces.npz'
    if not trace.is_file():
        np.savez_compressed(trace,columns=np.array(columns),**frames)
    with np.load(trace,allow_pickle=False) as packed:
        assert list(packed['columns'])==columns and all(np.array_equal(v,packed[k],equal_nan=True) for k,v in frames.items())
    add('data',trace,trace.name,'21 趟 50 Hz 状态与净空曲线','按种子和方法分数组，保留统一列名和原有单位。',separate=True)
    fingerprints=json.loads((stage/'all-evidence-index.json').read_bytes())
    for name,meta in sorted(fingerprints.items()):
        path=stage/'evidence'/name
        assert path.stat().st_size==meta['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==meta['sha256']
        seed,method=name.split('.')[0].split('-')
        add('data',path,name,seed+' · '+MODES[method],'原生飞行日志、机身和机械臂状态、时序决策与执行输入。',separate=True)
    body=dict(schema=1,id=TAG,stage='G1 · 单门在线调时',title=TITLE,summary=SUMMARY,when=when,revision=revision,
              parent_round='2026-11-19-g1',cover='release.png',files=entries,
              guide=['先看三种策略的同步录像，再看各自开始收臂的瞬间。','比较中位净空和到位时间，留意连续与停顿两种任务的结果方向。','用四组交叉对照定位最近点。最近点先于收臂，弯臂阶段留下的差异需单独解释。'],
              scope='3 个配对种子，15 趟主对照和 6 趟补充飞行。SUPER 与 Qingyu 空载标称模型，固定门框与机身路线，PX4 EKF2 驱动的 MuJoCo 软件在环。',
              source_note='bcfd992 登记主试验，72a7810 登记交叉补充，d38c40d 保存 21 趟原始日志与评分。')
    write(spec,body)
    dist=canonical/'dist'
    gen=checkout/'tools/meeting_archive.py'
    subprocess.run([sys.executable,str(gen),'add','--site',str(dist),'--spec',str(spec),'--source',str(sources)],check=True)
    subprocess.run([sys.executable,str(gen),'verify','--site',str(dist)],check=True)
    folder=dist/'archive'/TAG
    shutil.copyfile(folder/'meeting.zip',stage/'G1在线_组会离线包.zip')
    with tarfile.open(canonical/'mirror-input.tar.gz','w:gz') as archive:
        for path in sorted(p for p in folder.rglob('*') if p.is_file() and not p.name.endswith('.tar.gz')):
            archive.add(path,arcname=str(path.relative_to(folder)).replace(os.sep,'/'))
    print(len(entries),'verified assets;', (folder/'meeting.zip').stat().st_size,'offline bytes')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stage',type=Path)
    p.add_argument('revision')
    a=p.parse_args()
    main(a.stage,a.revision)
