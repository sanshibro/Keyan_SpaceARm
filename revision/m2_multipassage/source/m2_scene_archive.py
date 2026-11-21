"""Append the industrial M2 static verification to the existing meeting archive."""
import argparse
from datetime import datetime,timezone,timedelta
import hashlib,json,shutil,subprocess,sys,zipfile
from pathlib import Path
from PIL import Image

STAGE=Path(__file__).resolve().parents[1]
ID='2026-11-21-m2-industrial-geometry'
def write(path,d):
    path.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def read(path):return json.loads(path.read_text(encoding='utf8'))
def layout_round(folder):
    """Give the main scientific plate full width in both web and offline copies."""
    css='#file-corridor-webp{grid-column:1/-1}#file-corridor-webp .media img{height:auto;max-height:none;width:100%}'
    def arrange(raw):
        text=raw.decode('utf8')
        return (text if css in text else text.replace('</style>',css+'</style>',1)).encode('utf8')
    page=folder/'index.html';page.write_bytes(arrange(page.read_bytes()))
    pack=folder/'meeting.zip'
    with zipfile.ZipFile(pack) as z:entries=[(i,z.read(i.filename)) for i in z.infolist()]
    with zipfile.ZipFile(pack,'w') as z:
        for info,data in entries:z.writestr(info,arrange(data) if info.filename=='index.html' else data,compress_type=zipfile.ZIP_DEFLATED,compresslevel=6)
    snapshot=read(folder/'snapshot.json')
    for kind,path in [('page',page),('pack',pack)]:snapshot['outputs'][kind]['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    write(folder/'snapshot.json',snapshot)
def publish(checkout,revision,pdf_url=None):
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
    if pdf_url:
        if not pdf_url.startswith('https://raw.githubusercontent.com/sanshibro/Keyan_SpaceARm/'+revision+'/'):
            raise ValueError('PDF must use the fixed original research commit')
        shutil.copyfile(STAGE/'review/视觉阶段定稿_三关几何与同源光学_双页.pdf',src/'visual-stage.pdf')
        files.append(dict(kind='document',file='visual-stage.pdf',title='视觉阶段双页中文审阅',
                          caption='三关净空、制造细节和同源腕图；固定研究版本的原始 PDF 单独下载。',separate=True))
    moment=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    spec=dict(schema=1,id=ID,title='三关工业检修廊的同源几何与腕部视野',stage='M2 · 工业场景静态复核',
              when=moment,revision=revision,summary=summary,cover='corridor.webp',files=files,guide=['从三张关口近景看机身与柔性臂的相对位置。','对照腕部与外部视角辨认同一个法兰及定位环。'],
              scope='冻结碰撞包络的静态几何与光学核验。',source_note='独立关口姿态；腕部使用原生 MuJoCo RGB-D 相机。')
    specpath=src.parent/'round.json';write(specpath,spec)
    tool=checkout/'tools/meeting_archive.py'
    subprocess.run([sys.executable,str(tool),'add','--site',str(dist),'--spec',str(specpath),'--source',str(src)],check=True)
    if pdf_url:
        # Keep the hosted archive below its 256 MiB cap without removing history.
        # The offline pack already marks this original PDF as a separate download.
        folder=dist/'archive'/ID;snapshot=read(folder/'snapshot.json')
        document=next(x for x in snapshot['files'] if x['file']=='visual-stage.pdf')
        evidence=checkout/'source-evidence'/ID;evidence.mkdir(parents=True,exist_ok=True)
        pdf=folder/'media/visual-stage.pdf';payload=pdf.read_bytes()
        if hashlib.sha256(payload).hexdigest()!=document['sha256']:raise ValueError('PDF source changed')
        (evidence/'visual-stage.pdf').write_bytes(payload)
        document['external_url']=pdf_url
        page=folder/'index.html';page.write_text(page.read_text(encoding='utf8').replace('media/visual-stage.pdf',pdf_url),encoding='utf8')
        snapshot['outputs']['page']['sha256']=hashlib.sha256(page.read_bytes()).hexdigest()
        write(folder/'snapshot.json',snapshot);pdf.unlink()
        subprocess.run([sys.executable,str(tool),'index','--site',str(dist)],check=True)
    layout_round(dist/'archive'/ID)
    subprocess.run([sys.executable,str(tool),'verify','--site',str(dist)],check=True)
    status=read(dist/'status_public.json');status.update(branch='feat/m2-multipassage-rolling-20261121',stage=spec['stage'],updated=moment,
      status='development',mode='原生机模 · 静态几何与同源光学',detail=summary,
      geometry='archive/'+ID+'/media/corridor.webp',geometry_caption='三关剖面与工业细节',
      rolling='archive/'+ID+'/media/wrist.webp',rolling_caption='同源腕部 RGB · 静态法兰视野',
      items=['45 个碰撞体锁定','75 个起飞静态采样','两平台几何复核','正式组以独立先导门禁放行'])
    status['commits']=[x for x in status.get('commits',[]) if 'M2 静态' not in x]+[revision[:7]+' · M2 静态']
    status.pop('progress',None);write(dist/'status_public.json',status)
    # meeting_archive.index() updates its existing homepage slot without assuming
    # a specific legacy page element or duplicating the latest-round entry.
    total=sum(p.stat().st_size for p in dist.rglob('*') if p.is_file())
    if total>=256*1024**2:raise ValueError('publication archive size '+str(total))
    print('rendered',ID,'total bytes',total)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkout',type=Path);p.add_argument('revision');p.add_argument('--pdf-url');a=p.parse_args()
    publish(a.checkout.resolve(),a.revision,a.pdf_url)
