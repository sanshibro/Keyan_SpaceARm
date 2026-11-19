"""Publish a byte-verified mirror while keeping the earlier round snapshots unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

from round_g1_online_archive import TAG, TITLE, SUMMARY, write


def read(p):
    return json.loads(p.read_bytes())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(stage):
    stage=stage.resolve()
    site=stage/'site/.sites-checkout'
    dist=site/'dist'
    native=stage/'canonical/dist/archive'/TAG
    dest=dist/'archive'/TAG
    previous={p:sha(p) for p in (dist/'archive').glob('*/snapshot.json') if p.parent!=dest}
    assert len(previous)==11
    raw=read(stage/'public-raw-verification.json')
    mirrors=read(stage/'public-mirror-verification.json')
    assert len(raw)==21 and set(mirrors)=={'meeting.zip','all-traces.npz'}
    external={**raw,**mirrors}
    cache=site/'source-evidence'/TAG
    cache.mkdir(parents=True,exist_ok=True)
    write(cache/'verified.json',raw)
    shutil.copyfile(native/'meeting.zip',cache/'meeting.zip')
    shutil.copyfile(native/'media'/'all-traces.npz',cache/'all-traces.npz')
    snaps=read(native/'snapshot.json')
    spec=read(stage/'canonical/g1-online.json')
    by_name={f['file']:f for f in spec['files']}
    page=(native/'index.html').read_text(encoding='utf-8')
    outputs=snaps['outputs']
    assert external['meeting.zip']['sha256']==outputs['pack']['sha256']
    (dest/'media').mkdir(parents=True,exist_ok=True)
    for f in snaps['files']:
        name=f['file']
        if name in external:
            proof=external[name]
            assert (f['sha256'],f['bytes'])==(proof['sha256'],proof['bytes'])
            f['external_url']=by_name[name]['external_url']=proof['external_url']
            old='href="media/'+name+'"'
            assert old in page
            page=page.replace(old,'href="'+proof['external_url']+'"')
        else:
            shutil.copyfile(native/'media'/name,dest/'media'/name)
    old='href="meeting.zip"'
    assert old in page
    pack_url=external['meeting.zip']['external_url']
    page=page.replace(old,'href="'+pack_url+'"')
    outputs['pack']['external_url']=pack_url
    (dest/'index.html').write_bytes(page.encode())
    outputs['page']['sha256']=sha(dest/'index.html')
    write(dest/'snapshot.json',snaps)
    write(site/'tools/rounds/g1-online.json',spec)
    gen=site/'tools/meeting_archive.py'
    original=gen.read_text(encoding='utf-8')
    old_line="        b=p.read_bytes();req(len(b)==f['bytes'] and sha(b)==f['sha256'],'media hash differs: '+f['file'])"
    new_block="""        if f.get('external_url') and not p.is_file():
            proof=read(p.parent/'verified.json').get(f['file'],{})
            req(all(proof.get(key)==f[key] for key in ('external_url','bytes','sha256')),
                'missing or inconsistent verified remote asset: '+f['file'])
            continue
""" + old_line
    if new_block not in original:
        assert original.count(old_line)==1
        gen.write_bytes(original.replace(old_line,new_block).encode())
    attrs=site/'.gitattributes'
    rules=attrs.read_text(encoding='utf-8').splitlines()
    for rule in ['dist/archive/'+TAG+'/media/* -text','source-evidence/'+TAG+'/* -text']:
        if rule not in rules:
            rules.append(rule)
    attrs.write_bytes(('\n'.join(rules)+'\n').encode())
    redir=dist/'_redirects'
    redirects=redir.read_text(encoding='utf-8').splitlines()
    for name,proof in external.items():
        relative=('' if name=='meeting.zip' else 'media/')+name
        line='/archive/'+TAG+'/'+relative+' '+proof['external_url']+' 302'
        if line not in redirects:
            redirects.append(line)
    redir.write_bytes(('\n'.join(redirects)+'\n').encode())
    subprocess.run([sys.executable,str(gen),'index','--site',str(dist)],check=True)
    home=dist/'index.html'
    link='archive/'+TAG+'/'
    block=('<section id="reviewVideo"><h2>'+TITLE+'</h2><p>同一机身进度、同一门框、同一世界机位。'
           '三种时序在第 19 到 50 秒同步回放。</p><video controls playsinline preload="metadata" poster="'+link
           +'media/matched.png" style="display:block;width:100%;max-height:600px;background:white"><source src="'+link
           +'media/comparison.mp4" type="video/mp4"></video><p><a href="'+link+'index.html">本轮图片、曲线与逐趟日志</a> · '
           '<a href="'+link+'media/review.pdf">两页组会审阅</a></p></section>')
    html,n=re.subn(r'<section id="reviewVideo">.*?</section>',lambda _:block,home.read_text(encoding='utf-8'),count=1,flags=re.S)
    assert n==1
    prefix='<section><h2>阶段文件</h2><p>'
    assert prefix in html
    lnk='<a href="'+link+'media/review.pdf" target="_blank">G1 单门在线调时审阅</a>　'
    if lnk not in html:
        html=html.replace(prefix,prefix+lnk,1)
    home.write_bytes(html.encode())
    state=read(dist/'status_public.json')
    state.update(status='awaiting_review',stage='G1 · 单门在线调时完成',mode='PX4 EKF2 与 MuJoCo · 5 Hz 在线臂形调时',
                 detail=SUMMARY,updated=spec['when'],progress=100,geometry=link+'media/release.png',
                 geometry_caption='三种时序各自开始收臂时的实测后缘位置',rolling=link+'media/curves.png',
                 rolling_caption='后缘位置、净空、弯角与跨种子对照',replay=None,
                 items=['15 趟主对照和 6 趟交叉补充','21 趟全部零接触并在 52 秒前到位','实际最大弯角 72.70°',
                        '最近点出现在各组收臂之前'],
                 commits=['bcfd992 · 预登记','72a7810 · 主对照与补充协议','d38c40d · 21 趟飞行与独立核对','28337e2 · 两页审阅与复现'])
    write(dist/'status_public.json',state)
    sys.path.insert(0,str(stage.parent/'D_整机几何与规划验收_20261118'))
    from quota_snapshot import read_windows
    try:
        quota=read_windows()
    except (OSError,RuntimeError) as err:
        print('Kept timestamp of previously verified usage snapshot:',type(err).__name__)
    else:
        write(dist/'usage_public.json',quota)
    for path,checksum in previous.items():
        assert sha(path)==checksum
    subprocess.run([sys.executable,str(gen),'verify','--site',str(dist)],check=True)
    size=sum(p.stat().st_size for p in dist.rglob('*') if p.is_file())
    assert size<=256*1024**2,size
    print('Earlier snapshots verified:',len(previous),'expanded publish bytes:',size)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stage',type=Path)
    a=p.parse_args()
    main(a.stage)
