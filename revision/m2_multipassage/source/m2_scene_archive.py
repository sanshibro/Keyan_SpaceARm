"""Append the industrial M2 static verification to the existing meeting archive."""
import argparse
from datetime import datetime,timezone,timedelta
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
from PIL import Image

STAGE=Path(__file__).resolve().parents[1]
ID='2026-11-21-m2-industrial-geometry'
def write(path,d):
    path.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def read(path):return json.loads(path.read_text(encoding='utf8'))
def publish(checkout,revision):
    src=STAGE/'meeting-static'/'sources';src.mkdir(parents=True,exist_ok=True)
    qa=STAGE/'qa/static';dist=checkout/'dist';native=read(qa/'native-crosscheck.json')
    lock=read(qa/'collision-lock.json');optics=read(qa/'target-optics.json');cert=read(qa/'certified.json')
    if not(native['passed'] and native['scene_sha256']==cert['scene_sha256']==optics['scene_sha256']==lock['scene_sha256']):raise ValueError('static registration')
    for name,out,size in [('three-gate-static-review.png','corridor.webp',(1850,1420)),('target-wrist.png','wrist.webp',(480,360)),
                          ('flange/flange-oblique.png','flange.webp',(950,570))]:
        image=Image.open(qa/name).convert('RGB');image.thumbnail(size,Image.Resampling.LANCZOS);image.save(src/out,quality=89,method=6)
    for name in ('native-crosscheck.json','target-optics.json','collision-lock.json'):shutil.copyfile(qa/name,src/name)
    a,b,c=[x['external_lower_m']*100 for x in cert['gate_witnesses']]
    summary='三种静态构型的全身外障净空分别为 %.2f、%.2f、%.2f cm。末端光心距法兰基准面 %.2f cm。'%(a,b,c,optics['optical_depth_m']*100)
    add=lambda kind,name,title,caption:dict(kind=kind,file=name,title=title,caption=caption)
    files=[add('image','corridor.webp','三关静态剖面与局部配合','三处姿态分别对应高位隔舱门、风管窄缝和反向偏置门。'),
           add('image','wrist.webp','腕部 RGB 原始视角','中心定位环、铜环和八个螺栓来自可供两类相机共同观察的工业构件。'),
           add('image','flange.webp','末端外部机位','从通廊侧面观察夹爪与斜置法兰的相对位置。'),
           add('data','native-crosscheck.json','两台机器的静态几何核验','NUC 原生机模与本机同姿态的几何下界和启动检查。'),
           add('data','target-optics.json','光学工作距与深度','光心、法兰靶心像素与原始深度对照。'),
           add('data','collision-lock.json','碰撞体逐项比对','全部登记构件的形状、尺寸、位姿与场景指纹。')]
    moment=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    spec=dict(schema=1,id=ID,parent_round='2026-11-21-m1-coupled',title='三关工业检修廊的同源几何与腕部视野',stage='M2 · 工业场景静态复核',
              when=moment,revision=revision,summary=summary,cover='corridor.webp',files=files,guide=['从三张关口近景看机身与柔性臂的相对位置。','对照腕部与外部视角辨认同一个法兰及定位环。'],
              scope='冻结碰撞包络的静态几何与光学核验。',source_note='独立关口姿态；腕部使用原生 MuJoCo RGB-D 相机。')
    specpath=src.parent/'round.json';write(specpath,spec)
    tool=checkout/'tools/meeting_archive.py'
    subprocess.run([sys.executable,str(tool),'add','--site',str(dist),'--spec',str(specpath),'--source',str(src)],check=True)
    subprocess.run([sys.executable,str(tool),'verify','--site',str(dist)],check=True)
    status=read(dist/'status_public.json');status.update(branch='feat/m2-multipassage-rolling-20261121',stage=spec['stage'],updated=moment,
      status='development',detail=summary,geometry='archive/'+ID+'/media/corridor.webp',geometry_caption='三关剖面与工业细节',
      items=['45 个碰撞体锁定','75 个起飞静态采样','两平台几何复核','正式组以独立先导门禁放行'])
    status.pop('progress',None);write(dist/'status_public.json',status)
    page=dist/'index.html';doc=page.read_text(encoding='utf8')
    entry='<p><a href="archive/'+ID+'/index.html">M2 三关工业检修廊 · 净空与腕部视野</a></p>'
    if entry not in doc:
        if '<main' not in doc:raise ValueError('unexpected homepage')
        pos=doc.index('>',doc.index('<main'))+1;page.write_text(doc[:pos]+entry+doc[pos:],encoding='utf8')
    total=sum(p.stat().st_size for p in dist.rglob('*') if p.is_file())
    if total>=256*1024**2:raise ValueError('publication archive size '+str(total))
    print('rendered',ID,'total bytes',total)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkout',type=Path);p.add_argument('revision');a=p.parse_args()
    publish(a.checkout.resolve(),a.revision)
