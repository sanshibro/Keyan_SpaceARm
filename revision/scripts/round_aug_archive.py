"""Publish immutable, versioned media and keep a portable two-page review for the group."""
import json
import re
import shutil
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from pathlib import Path

from PIL import Image

TAG = '2026-11-20-augmented-fusion'
PREVIOUS = '2026-11-20-strain-health'
TITLE = '面向视觉退化与应变迟滞的连续体臂形态融合'
SUMMARY = '三趟 PX4 / MuJoCo 飞行中用可见阶段校准应变偏置。相对贴片反向偏移时，4 秒全遮挡内的弯曲角误差降至 0.40°，覆盖全部 600 个输出时刻。'
RAW = 'https://raw.githubusercontent.com/sanshibro/Keyan_SpaceARm/'
CDN = 'https://cdn.jsdelivr.net/gh/sanshibro/Keyan_SpaceARm@'

def utilities(stage):
    try:
        import round_patent_archive as u
    except ImportError:
        module = next(stage.parent.glob('*/source/round_patent_archive.py'))
        sys.path.insert(0, str(module.parent))
        import round_patent_archive as u
    return u

def make_images(rd):
    with Image.open(rd/'fusion_main.png') as im:
        small = im.convert('RGB').crop((0, int(.032*im.height), im.width, int(.323*im.height)))
        small.thumbnail((690,300),Image.Resampling.LANCZOS)
        small.save(rd/'overview.webp',quality=73,method=6)
        im.convert('RGB').save(rd/'fusion-main.webp',quality=90,method=6)
    with Image.open(rd/'fusion_matrix.png') as im:
        im.convert('RGB').save(rd/'comparison.webp',quality=90,method=6)

def build(stage, revision):
    u = utilities(stage)
    work, rd = stage/'canonical-release', stage/'review'
    src = work/'sources'
    src.mkdir(parents=True,exist_ok=True)
    make_images(rd)
    files = []
    def add(kind, path, name, title, caption, **options):
        shutil.copyfile(path,src/name)
        files.append(dict(kind=kind,file=name,title=title,caption=caption,**options))
    add('image',rd/'overview.webp','overview.webp','腕部相机的四个时刻','左偏航迹中同期的整机姿态与腕部原帧，依次为清晰、局部遮挡、完全遮挡和重锁。')
    add('curve',rd/'fusion-main.webp','fusion-main.webp','视觉、应变与弯曲误差的同步轨迹','上方按曝光时刻选帧。置信度、实际更新增益、两个偏置分量和四方法弯曲误差按估计输出时钟对齐。')
    add('curve',rd/'comparison.webp','comparison.webp','三种视觉条件的四方法比较','三趟合并的全程成绩与等时长关键窗口；虚拟动捕独立计分。')
    add('video',rd/'fusion_replay.mp4','replay.mp4','穿门中视野退化、全黑与恢复的同步回放','22–41 s。逐帧回看飞机和腕部图像，并联读弯曲角及误差、观测置信度和在线偏置。',poster='overview.webp')
    add('document',rd/'review.pdf','review.pdf','两页组会 PDF','第一页面向同一时刻解释估计机制；第二页给出四种方法的角度、方向、覆盖率、重锁时序和跨遮挡起漂成绩。')
    for name,title,caption in [
        ('scores.json','三趟总体及逐趟评分','按应变工况、视觉工况和时段读取角度、方位与覆盖；同时提供四方法共同有效时刻的成绩。'),
        ('acceptance.json','量化指标及逐趟复核','67 项硬性检查与逐趟明细；记录实际数值、阈值、方向与结果。'),
        ('frozen-settings.json','模型与参数冻结记录','记录输入、传感器模型及估计器的文件指纹。'),
        ('pilot-selection.json','独立标定航迹与基线选择','固定协方差基线与增广滤波器在独立标定航迹上的调参记录。'),
        ('release-lock.json','估计轨迹的封存与因果核验','各趟、工况的估计先于真值评分封存。'),
        ('optical-audit.json','原生视觉跟踪与几何校验','RGB-D、视觉标定及不同时段重锁统计，并保留原生二进制指纹。'),
        ('bench-validation.json','四段独立台架动作','独立动作上的连续应变逆模型精度、方差覆盖和截断复算。'),
        ('evidence-index.json','原始数据包及逐文件指纹','可按包名和成员路径核对 SHA-256。')]:
        add('data',stage/name,name,title,caption)
    add('data',rd/'full_scores.csv','full-scores.csv','可筛选的完整指标表','按工况、时段和算法比较有效样本数、覆盖率、弯曲角及方位角 RMSE。')
    names = {'calibration-and-audit.tar.gz':'独立台架、标定和冻结模型','gate-26112011-straight.tar.gz':'直行轨迹原始数据','gate-26112012-left.tar.gz':'左偏轨迹原始数据','gate-26112013-right.tar.gz':'右偏轨迹原始数据'}
    for name,heading in names.items():
        cap = '训练外台架动作、标定航迹、传感参数和估计输入。' if name.startswith('calibration') else '原生 ULog、ROS bag、MuJoCo 状态、RGB-D 关键时刻及输入指纹、传感器重放、四方法输出和独立评分真值。'
        add('data',stage/'evidence'/name,name,heading,cap,separate=True)
    spec = work/'round.json'
    when = u.read(spec)['when'] if spec.is_file() else datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    u.write(spec,dict(schema=1,id=TAG,stage='G1 · 视觉与应变增广融合',when=when,revision=revision,title=TITLE,summary=SUMMARY,cover='overview.webp',files=files,
                     parent_round=PREVIOUS,
                     guide=['先看视频中腕部画面转黑的时刻，并对照偏置在遮挡前的收敛过程。',
                            '比较完整四方法曲线，再打开 PDF 查看三趟同轨统计与重锁延迟。',
                            '逐趟原始日志和输入指纹可以单独下载；离线包收录图、视频、两页报告和指标。'],
                     scope='三趟留出的 PX4 SITL / MuJoCo 仿真飞行；连续逆模型与协方差在独立数据上标定。估计器按传感器到达顺序回放，虚拟动捕用于独立评分。',
                     source_note='主试验在 18–22 s 引入通道偏移。附加试验将同一差模偏移移至 29–33 s，与遮挡重叠，遮挡段误差为 7.06°。'))
    target = work/'dist'
    u.gen(stage,'add',target,'--spec',spec,'--source',src)
    u.gen(stage,'verify',target)
    folder = target/'archive'/TAG
    shutil.copyfile(folder/'meeting.zip',stage/'组会离线包.zip')
    with tarfile.open(work/'meeting-mirror.tar.gz','w:gz') as tf:
        for file in sorted(folder.rglob('*')):
            if file.is_file() and not file.name.endswith('.tar.gz'):
                tf.add(file,file.relative_to(folder).as_posix())

def publish(stage, revision):
    u = utilities(stage)
    repo = stage/'site-cache'
    dist = repo/'dist'
    orig = stage/'canonical-release/dist/archive'/TAG
    route = 'archive/'+TAG+'/'
    dest = dist/route
    previous = {p:u.digest(p) for p in (dist/'archive').glob('*/snapshot.json') if p.parent!=dest}
    assert (dist/'archive'/PREVIOUS/'snapshot.json') in previous
    snap,spec = u.read(orig/'snapshot.json'),u.read(stage/'canonical-release/round.json')
    raw = RAW+revision+'/revision/meeting_archive/'+route
    media = CDN+revision+'/revision/meeting_archive/'+route+'media/'
    urls = {}
    packs = u.read(stage/'evidence-index.json')
    for row in snap['files']:
        name = row['file']
        if name=='overview.webp':
            continue
        if name in packs:
            url = RAW+revision+'/revision/data/round_augmented_fusion/evidence/'+name
        elif row['kind'] in ('document','curve','video','image'):
            url = media+name
        else:
            url = raw+'media/'+name
        urls[name] = (url,orig/'media'/name)
    urls['meeting.zip'] = (raw+'meeting.zip',orig/'meeting.zip')
    with ThreadPoolExecutor(max_workers=6) as pool:
        tasks = {name:pool.submit(u.confirm,*pair) for name,pair in urls.items()}
        proof = {name:job.result() for name,job in tasks.items()}
    cache = repo/'source-evidence'/TAG
    u.write(cache/'verified.json',proof)
    shutil.copyfile(orig/'meeting.zip',cache/'meeting.zip')
    u.write(stage/'public-raw-verification.json',proof)
    page = (orig/'index.html').read_text(encoding='utf8')
    (dest/'media').mkdir(parents=True,exist_ok=True)
    entries = {v['file']:v for v in spec['files']}
    for row in snap['files']:
        name = row['file']
        if name in proof:
            assert row['sha256']==proof[name]['sha256']
            url = proof[name]['external_url']
            row['external_url'] = entries[name]['external_url'] = url
            page = page.replace('"media/'+name+'"','"'+url+'"')
        else:
            shutil.copyfile(orig/'media'/name,dest/'media'/name)
    snap['outputs']['pack']['external_url'] = proof['meeting.zip']['external_url']
    page = page.replace('href="meeting.zip"','href="'+proof['meeting.zip']['external_url']+'"')
    page = page.replace('</header>','<p><a href="../'+PREVIOUS+'/index.html">上一轮应变零点与材料回程实验</a></p></header>',1)
    (dest/'index.html').write_bytes(page.encode())
    snap['outputs']['page']['sha256'] = u.digest(dest/'index.html')
    u.write(dest/'snapshot.json',snap)
    u.write(repo/'tools/rounds/augmented-fusion.json',spec)
    attr = repo/'.gitattributes'
    rules = attr.read_text(encoding='utf8')
    for r in ('dist/'+route+'media/* -text','source-evidence/'+TAG+'/* -text'):
        if r not in rules:
            rules += '\n'+r+'\n'
    attr.write_bytes(rules.encode())
    u.gen(stage,'index',dist)
    href = lambda n:proof[n]['external_url']
    cover = href('fusion-main.webp')
    home=dist/'index.html'
    html=home.read_text(encoding='utf8')
    video=('<section id="reviewVideo"><h2>'+TITLE+'</h2><video controls playsinline preload="metadata" poster="'+route+'media/overview.webp" '
           'style="display:block;width:100%;max-height:600px;background:white"><source src="'+href('replay.mp4')+'" type="video/mp4"></video>'
           '<p><a href="'+route+'index.html">本轮图、曲线与飞行日志</a> · <a href="'+href('comparison.webp')+'">四方法比较</a> · '
           '<a href="'+href('review.pdf')+'">两页组会 PDF</a></p></section>')
    html,count=re.subn(r'<section id="reviewVideo">.*?</section>',lambda _:video,html,count=1,flags=re.S)
    assert count==1
    key='<section><h2>阶段文件</h2><p>'
    item='<a href="'+href('review.pdf')+'" target="_blank">视觉与应变增广融合审阅</a>　'
    assert key in html
    html=re.sub(r'<a href="[^"]+" target="_blank">视觉与应变增广融合审阅</a>　','',html)
    html=html.replace(key,key+item,1)
    html=html.replace("$(id).src=REMOTE+url+'?v='+s.updated","$(id).src=/^https:\\/\\//.test(url)?url:REMOTE+url+'?v='+s.updated")
    home.write_bytes(html.encode())
    state=u.read(dist/'status_public.json')
    state.update(updated=spec['when'],stage=spec['stage'],status='awaiting_review',progress=100,
                 mode='三趟 PX4 / MuJoCo 仿真 · 腕部 RGB-D · 四片 SLIM M',detail=SUMMARY,
                 geometry=cover,geometry_caption='同一轨迹上的置信度、应变偏置和弯曲误差',
                 rolling=href('comparison.webp'),rolling_caption='四方法在三种视觉条件下的配对比较',replay=None,
                 items=['无遮挡后半程纯视觉弯曲角 RMSE 0.257°','正常视野、基础迟滞下融合 RMSE 0.299°',
                        '差模偏移下完全遮挡 4 秒：纯视觉 0/600，融合 600/600',
                        '计入图像延迟，三趟重锁分别耗时 0.492、0.564、0.324 s'],
                 commits=[spec['revision'][:7]+' · 冻结估计、同轨对比与日志',revision[:7]+' · 组会图表与视频归档'])
    u.write(dist/'status_public.json',state)
    sys.path.insert(0,str(stage.parent/'D_整机几何与规划验收_20261118'))
    from quota_snapshot import read_windows
    try:
        quota=read_windows()
    except (OSError,RuntimeError):
        print('Historical timestamped quota snapshot retained',flush=True)
    else:
        u.write(dist/'usage_public.json',quota)
    assert all(u.digest(p)==h for p,h in previous.items())
    u.gen(stage,'verify',dist)
    print('Hosted bytes:',sum(p.stat().st_size for p in dist.rglob('*') if p.is_file()),'rounds:',len(previous)+1,flush=True)

if __name__=='__main__':
    cmd,root,rev=sys.argv[1:]
    {'build':build,'publish':publish}[cmd](Path(root).resolve(),rev)
