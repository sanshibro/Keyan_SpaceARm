"""Keep the exact RGB-D inputs on the NUC and publish bounded, reproducible evidence packs."""
import gzip,hashlib,io,json,os,shutil,tarfile
from pathlib import Path
from round_shape_pixels import OUTPUT,EVIDENCE,ROOT
from round_wrist_fusion import CASES,savej

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pack(dst,files):
    members=sorted(files.items())
    if dst.exists():
        with tarfile.open(dst,'r:gz') as a:
            if a.getnames()==[n for n,_ in members] and all(hashlib.sha256(a.extractfile(n).read()).hexdigest()==sha(p) for n,p in members):
                return
    with dst.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0,compresslevel=4) as gz,tarfile.open(fileobj=gz,mode='w') as archive:
        for n,p in members:
            info=tarfile.TarInfo(n);info.size=p.stat().st_size;info.mode=0o644
            with p.open('rb') as f:archive.addfile(info,f)
    with tarfile.open(dst,'r:gz') as a:
        assert a.getnames()==[n for n,_ in members]
        for n,p in members:assert hashlib.sha256(a.extractfile(n).read()).hexdigest()==sha(p)
    assert dst.stat().st_size<24*1024**2
def tree(where,*patterns):
    return {str(p.relative_to(OUTPUT)):p for pattern in patterns for p in where.rglob(pattern) if p.is_file()}
def main():
    evidence=OUTPUT/'evidence';evidence.mkdir(exist_ok=True)
    store=Path('/tmp/g1_wrist_orb_20261119');persistent=OUTPUT/'rgbd-inputs'
    files=[];inputs={};representative={}
    for f in sorted(store.rglob('*')):
        if not f.is_file() or f.name=='track':continue
        rel=f.relative_to(store);dest=persistent/rel;dest.parent.mkdir(exist_ok=True,parents=True)
        if not dest.exists():
            try:os.link(f,dest)
            except OSError:shutil.copyfile(f,dest)
        elif sha(f)!=sha(dest):raise ValueError('Archived input changed: '+str(rel))
        digest=sha(f);files.append((str(rel),f.stat().st_size,digest))
        if len(rel.parts)==3 and f.name in ['frames.txt','settings.yaml','complete.json','r0000.jpg','d0000.png',
            'r0400.jpg','d0400.png','r0525.jpg','d0525.png','r0650.jpg','d0650.png']:
            representative['rgbd-inputs/'+str(rel)]=dest
    index=OUTPUT/'slam/input-fingerprints.tsv.gz'
    with index.open('wb') as out,gzip.GzipFile(fileobj=out,mode='wb',mtime=0) as gz:
        gz.write(('relative_path\tbytes\tsha256\n'+''.join('%s\t%d\t%s\n'%row for row in files)).encode())
    selected=['241118-online_pause',*CASES]
    for case in selected+['241118-clock_pause']:
        d=EVIDENCE/'runs'/case
        inputs[case]={name:sha(d/name) for name in ['physics.npz','timing.jsonl','hil.csv','run.json']}
    savej(OUTPUT/'slam/input-lineage.json',dict(native_G1=inputs,rgbd_manifest_sha256=sha(index),
            full_frame_count=sum(n.endswith('.jpg') for n,_,_ in files),depth_count=sum(n.startswith(('plain/','textured/')) and n.split('/')[-1].startswith('d') and n.endswith('.png') for n,_,_ in files),
            local_full_input_path=str(persistent),time_window='12 through 46 seconds at 25 Hz, plain-scene control 6 seconds',material='Seeded uncoded surface texture; no feature locations exposed to tracker.'))
    groups={'bench-training':{**tree(OUTPUT/'bench','*.json','*.npz'),**tree(OUTPUT/'bench-vision','*.json','*.npz')},
            'flight-sensors':tree(OUTPUT/'runs','*.npz'),
            'tagged-reference':{**tree(OUTPUT/'frames','*.npz'),**{n:OUTPUT/n for n in ('scores.json','pre-observability-audit.json','frozen-pads.json')}},
            'orb-slam-results':{**tree(OUTPUT/'slam','*.json','*.csv','*.yaml','*-estimate.npz','*.tsv.gz'),**representative}}
    manifest={}
    for name,items in groups.items():
        f=evidence/(name+'.tar.gz');pack(f,items)
        manifest[f.name]=dict(bytes=f.stat().st_size,sha256=sha(f),members=len(items))
    savej(OUTPUT/'evidence-index.json',manifest)
    print({k:v['bytes'] for k,v in manifest.items()},flush=True)
if __name__=='__main__':main()
