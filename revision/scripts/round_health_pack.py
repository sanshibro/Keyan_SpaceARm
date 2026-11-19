"""Archive native telemetry, observations and estimates and re-hash each stored member."""
import hashlib, io, json, tarfile
from pathlib import Path
from round_health_flights import OUT,SPEC
from round_wrist_fusion import savej
def h(data):return hashlib.sha256(data).hexdigest()
def archive(name,paths):
    dest=OUT/'evidence';dest.mkdir(exist_ok=True);file=dest/(name+'.tar.gz')
    checked={}
    with tarfile.open(file,'w:gz',compresslevel=6) as tar:
        for src in sorted(set(paths)):
            if not src.is_file() or src.is_symlink():raise ValueError(src)
            data=src.read_bytes();p=src.relative_to(OUT).as_posix()
            header=tarfile.TarInfo(p);header.size=len(data);header.mode=0o644
            tar.addfile(header,io.BytesIO(data));checked[p]=h(data)
    with tarfile.open(file,'r:gz') as tar:
        replay={mem.name:h(tar.extractfile(mem).read()) for mem in tar}
    assert replay==checked
    assert file.stat().st_size<24*1024**2
    return file.name,dict(bytes=file.stat().st_size,sha256=h(file.read_bytes()),files=len(checked),member_sha256=checked)
def files(p):return [f for f in p.rglob('*') if f.is_file() and not f.is_symlink()]
def run():
    packs={};packs.update([archive('health-bench',files(OUT/'bench')+files(OUT/'model')+[OUT/'freeze.json'])])
    for cfg in SPEC['flight']:
        case=cfg['case'];run=OUT/'runs'/case;frames=OUT/'inputs/textured'/case
        rgb=sorted(frames.glob('r*.jpg'));depth=sorted(frames.glob('d*.png'));assert len(rgb)==len(depth)==850
        sources={p.name:h(p.read_bytes()) for p in rgb+depth}
        folder=OUT/'source-manifests';folder.mkdir(exist_ok=True);m=folder/(case+'.json')
        savej(m,dict(case=case,rgb=850,depth=850,
              directory='inputs/textured/'+case,description='All encoded RGB/depth originals retained on the NUC. Sample original frames and all timestamps are included in this pack.',sha256=sources))
        keep=[run/f for f in ['physics.npz','hil.csv','flight.ulg','flight.bag','timing.jsonl','run.json','events.jsonl','provenance.json']]
        keep+=files(OUT/'records'/case)+files(OUT/'traces'/case)+files(OUT/'camera'/case)+files(OUT/'artifacts/slam/textured'/case)
        keep+=[m,frames/'frames.txt',frames/'settings.yaml',frames/'complete.json']
        for i in (0,150,300,450,550,650,849):keep.extend([rgb[i],depth[i]])
        packs.update([archive('health-'+case,keep)])
    savej(OUT/'evidence-index.json',packs)
    for name,p in packs.items():print(name,p['files'],p['bytes'],p['sha256'][:12],flush=True)
if __name__=='__main__':run()
