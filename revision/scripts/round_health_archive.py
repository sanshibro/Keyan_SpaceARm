"""Append an immutable, locally portable review of the held-out strain-health experiment."""
import argparse, json, re, shutil, sys, tarfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone,timedelta
from pathlib import Path
TAG='2026-11-20-strain-health'
TITLE='四路应变能区分零点偏移和材料迟滞吗？'
SUMMARY='三趟新飞行仿真上比较同一腕部视觉配不同应变工况，测量偏移报警、失视弯角误差和误差区间的覆盖率。'
BASE='https://raw.githubusercontent.com/sanshibro/Keyan_SpaceARm/'
def common(stage):
    try:import round_patent_archive as util
    except ImportError:
        paths=list(stage.parent.glob('*/source/round_patent_archive.py'))
        if not paths:raise
        sys.path.insert(0,str(paths[0].parent));import round_patent_archive as util
    return util
def build(stage,revision):
    u=common(stage);work=stage/'canonical';src=work/'sources';src.mkdir(parents=True,exist_ok=True)
    files=[]
    def add(kind,dir,name,title,caption,**extra):
        shutil.copyfile(dir/name,src/name);files.append(dict(kind=kind,file=name,title=title,caption=caption,**extra))
    rd=stage/'review'
    add('image',rd,'mechanism.webp','零点偏移报警与腕部相机失视','同一趟左偏飞行中的机器人、腕部图像、偏移概率和弯角误差。灰带对应 4 秒黑帧。')
    add('curve',rd,'transfer.webp','五种应变工况下的弯角误差与覆盖率','三趟留出轨迹合计评分，包含健康基线胜出的工况和定向偏移反例。')
    add('video',rd,'replay.mp4','失视前后 18 秒同步回放','观察随实际推力指令渲染的旋翼、视野遮挡、独立虚拟动捕和估计误差范围。',poster='mechanism.webp')
    add('document',rd,'review.pdf','两页组会预览','第 1 页解释感知与训练，第 2 页列出定量结果、反例和下一步的实验。')
    datasets=[
      ('bench-scores.json','独立台架动作的测试结果','加载、保持、卸载与相同角度的回程读数。'),
      ('flight-scores.json','清晰与遮挡的总体评分','8 种估计方法的弯曲量、方向、弯曲向量、有效率及名义区间覆盖。'),
      ('case-scores.json','三趟飞行逐趟评分','可按轨迹、材料回程、采集盒偏移与视觉条件定位。'),
      ('timing-reports.json','偏移识别和报警延迟','先对每趟轨迹的灵敏度与特异度取均值，再对轨迹等权平均。'),
      ('audit.json','不可辨识反例与因果性检查','核对构造的 35° / 43° 反例、输入时间截断、失视窗口和原始飞行日志。'),
      ('optical.json','ORB-SLAM3 跟踪与配准恢复','各次运行的帧数、原地图姿态以及由 RGB-D 关键帧配准回原坐标的结果。'),
      ('freeze.json','训练和标定冻结记录','独立台架动作标识、代码与输入指纹、模型 SHA-256。'),
      ('evidence-index.json','原始数据包清单与指纹','可核对每个压缩包及其成员文件的 SHA-256。')]
    for n,t,c in datasets:add('data',stage,n,t,c)
    titles={'health-bench.tar.gz':'14 段台架动作与冻结模型','health-26112011-straight.tar.gz':'直行轨迹原始证据',
            'health-26112012-left.tar.gz':'左偏轨迹原始证据','health-26112013-right.tar.gz':'右偏轨迹原始证据'}
    for n,t in titles.items():
        cap='独立原始读数、虚拟动捕及冻结权重。' if n.startswith('health-bench') else '原始 ULog、ROS bag、物理状态、传感输入、估计轨迹、选取的原始相机帧及完整 RGB-D 序列指纹。'
        add('data',stage/'evidence',n,t,cap,separate=True)
    sp=work/'round.json';when=u.read(sp)['when'] if sp.exists() else datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    u.write(sp,dict(schema=1,id=TAG,stage='G1 · 应变零偏和回程辨识',when=when,revision=revision,title=TITLE,summary=SUMMARY,
        cover='mechanism.webp',files=files,guide=['先播放视频，看相机被遮挡后读数和真实弯角如何分离。','比较五种应变工况的误差及区间覆盖，特别注意定向偏移。','打开两页 PDF，使用随附校验的数据复核。'],
        scope='三趟新的 PX4 SITL / MuJoCo 飞行仿真。观测器使用各趟传感器记录回放，虚拟动捕独立评分。尚待实物四通道和反馈控制验证。',
        source_note='1fd2dee 登记方案，368a8d8 冻结台架模型，fd3d5c2 完成主评分，3f54116 记录报警延迟与原始日志。'))
    target=work/'dist';u.gen(stage,'add',target,'--spec',sp,'--source',src);u.gen(stage,'verify',target)
    folder=target/'archive'/TAG;shutil.copyfile(folder/'meeting.zip',stage/'组会离线包.zip')
    with tarfile.open(work/'meeting-mirror.tar.gz','w:gz') as tf:
        for p in sorted(folder.rglob('*')):
            if p.is_file() and not p.name.endswith('.tar.gz'):tf.add(p,p.relative_to(folder).as_posix())
def publish(stage,revision):
    u=common(stage);repo=stage/'site-cache';dist=repo/'dist'
    orig=stage/'canonical/dist/archive'/TAG;dest=dist/'archive'/TAG;snap=u.read(orig/'snapshot.json')
    spec=u.read(stage/'canonical/round.json');previous={p:u.digest(p) for p in (dist/'archive').glob('*/snapshot.json') if p.parent!=dest}
    if len(previous)<15:raise ValueError('Historical records missing')
    route='archive/'+TAG+'/';remote=BASE+revision+'/revision/meeting_archive/'+route
    packs=u.read(stage/'evidence-index.json');urls={}
    for row in snap['files']:
        name=row['file']
        if name in ('replay.mp4','review.pdf'):continue
        url=BASE+revision+'/revision/data/round_strain_health/evidence/'+name if name in packs else remote+'media/'+name
        urls[name]=(url,orig/'media'/name)
    urls['meeting.zip']=(remote+'meeting.zip',orig/'meeting.zip')
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs={k:pool.submit(u.confirm,*v) for k,v in urls.items()};proof={k:v.result() for k,v in jobs.items()}
    cache=repo/'source-evidence'/TAG;u.write(cache/'verified.json',proof);shutil.copyfile(orig/'meeting.zip',cache/'meeting.zip')
    u.write(stage/'public-raw-verification.json',proof)
    page=(orig/'index.html').read_text(encoding='utf8');(dest/'media').mkdir(parents=True,exist_ok=True)
    entries={v['file']:v for v in spec['files']}
    for row in snap['files']:
        name=row['file']
        if name in proof:
            assert row['sha256']==proof[name]['sha256'];url=proof[name]['external_url']
            row['external_url']=entries[name]['external_url']=url;page=page.replace('"media/'+name+'"','"'+url+'"')
        else:shutil.copyfile(orig/'media'/name,dest/'media'/name)
    snap['outputs']['pack']['external_url']=proof['meeting.zip']['external_url']
    page=page.replace('href="meeting.zip"','href="'+proof['meeting.zip']['external_url']+'"')
    prevroute='../2026-11-19-confidence/index.html'
    page=page.replace('</header>','<p><a href="'+prevroute+'">上一轮：腕部视觉何时能纠正应变偏差？</a></p></header>',1)
    (dest/'index.html').write_bytes(page.encode());snap['outputs']['page']['sha256']=u.digest(dest/'index.html')
    u.write(dest/'snapshot.json',snap);u.write(repo/'tools/rounds/strain-health.json',spec)
    attr=repo/'.gitattributes';rules=attr.read_text(encoding='utf8')
    for rule in ['dist/'+route+'media/* -text','source-evidence/'+TAG+'/* -text']:
        if rule not in rules:rules+='\n'+rule+'\n'
    attr.write_bytes(rules.encode());u.gen(stage,'index',dist)
    home=dist/'index.html';html=home.read_text(encoding='utf8');cover=proof['mechanism.webp']['external_url']
    vid=('<section id="reviewVideo"><h2>'+TITLE+'</h2><video controls playsinline preload="metadata" poster="'+cover+'" style="display:block;width:100%;max-height:600px;background:white"><source src="'+route+'media/replay.mp4" type="video/mp4"></video><p><a href="'+route+'index.html">本轮图片、曲线与飞行日志</a> · <a href="'+proof['transfer.webp']['external_url']+'">各工况定量结果</a> · <a href="'+route+'media/review.pdf">两页组会 PDF</a></p></section>')
    html,count=re.subn(r'<section id="reviewVideo">.*?</section>',lambda _:vid,html,count=1,flags=re.S);assert count==1
    key='<section><h2>阶段文件</h2><p>';assert key in html
    item='<a href="'+route+'media/review.pdf" target="_blank">应变零偏与材料回程审阅</a>　'
    if item not in html:html=html.replace(key,key+item,1)
    home.write_bytes(html.encode())
    state=u.read(dist/'status_public.json');state.update(updated=spec['when'],stage=spec['stage'],status='awaiting_review',progress=100,
        mode='三趟留出飞行 · 腕部 RGB-D · 四片 Slim M',detail=SUMMARY,geometry=cover,geometry_caption='偏移报警与失视弯角误差',
        rolling=proof['transfer.webp']['external_url'],rolling_caption='五种应变工况与区间覆盖',replay=None,
        items=['3 趟新 PX4 SITL / MuJoCo 仿真，已归档 ULog 和 ROS bag','常规零偏在 6/6 组重放中报警，延迟 0.62–2.73 s','逐趟等权的材料迟滞识别平衡准确率 60.5%','偏移与回程叠加时，失视 4 秒仅有 56/600 个时刻落在名义 95% 区间内'],
        commits=['1fd2dee · 登记协议','368a8d8 · 冻结台架形态模型','fd3d5c2 · 评分与可辨识性','3f54116 · 原始飞行数据与报警时序'])
    u.write(dist/'status_public.json',state)
    sys.path.insert(0,str(stage.parent/'D_整机几何与规划验收_20261118'));from quota_snapshot import read_windows
    try:q=read_windows()
    except (OSError,RuntimeError):print('Previous timestamped quota snapshot retained')
    else:u.write(dist/'usage_public.json',q)
    assert all(u.digest(p)==d for p,d in previous.items());u.gen(stage,'verify',dist)
    print('Hosted bytes',sum(p.stat().st_size for p in dist.rglob('*') if p.is_file()),'rounds',len(previous)+1,flush=True)
if __name__=='__main__':
    cmd,root,rev=sys.argv[1:];{'build':build,'publish':publish}[cmd](Path(root).resolve(),rev)
