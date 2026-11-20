"""Pack traceable evaluation inputs, independent labels and native logs; verify all archived bytes."""
import hashlib,io,json,tarfile
from pathlib import Path
from round_aug_runtime import ROOT,OUT,SPEC,digest,write

def pack(name,rows):
    target=OUT/'evidence';target.mkdir(exist_ok=True)
    file=target/(name+'.tar.gz');manifest={}
    with tarfile.open(file,'w:gz',compresslevel=5) as tf:
        for path,label in sorted(rows,key=lambda x:x[1]):
            if label in manifest:raise ValueError(label)
            data=path.read_bytes();h=hashlib.sha256(data).hexdigest()
            info=tarfile.TarInfo(label);info.mode=0o644;info.size=len(data);tf.addfile(info,io.BytesIO(data));manifest[label]=h
    with tarfile.open(file) as tf:
        assert {m.name:hashlib.sha256(tf.extractfile(m).read()).hexdigest() for m in tf}==manifest
    if file.stat().st_size>24*1024**2:raise ValueError('Archive exceeds the meeting-file size limit')
    return file.name,dict(sha256=digest(file),bytes=file.stat().st_size,files=len(manifest),member_sha256=manifest)
def collect(root):
    return [(p,p.relative_to(OUT).as_posix()) for p in root.rglob('*') if p.is_file()]
def main():
    old=ROOT/'revision/data/round_strain_health'
    manifest={}
    bench=collect(OUT/'strain-model')+collect(OUT/'strain-bench-independent')+collect(OUT/'geometry')+collect(OUT/'vision_sources')
    pilot=SPEC['pilot'];bench+=collect(OUT/'strain'/pilot)+collect(OUT/'vision'/pilot)
    for n in ('frozen-settings','pilot-selection','release-lock','acceptance','scores'):
        bench.append((OUT/(n+'.json'),n+'.json'))
    bench.append((OUT/'vision/audit.json','vision/audit.json'))
    manifest.update([pack('calibration-and-audit',bench)])
    for case in SPEC['cases']:
        files=collect(OUT/'strain'/case)+collect(OUT/'vision'/case)+collect(OUT/'released'/case)
        dest=OUT/'scoring-metadata/strain'/(case+'.json');files.append((dest,'scoring-metadata/strain/'+case+'.json'))
        files.append((old/'records'/case/'strain/reference.npz','independent-holdout/reference.npz'))
        for n in ['flight.ulg','flight.bag','physics.npz','timing.jsonl','run.json','provenance.json']:
            files.append((old/'runs'/case/n,'native-flight/'+n))
        for c in SPEC['conditions']:
            origin=OUT/'vision_inputs'/case/c;pix=origin/'pixels.json'
            for n in ['settings.yaml','frames.txt','pixels.json']:files.append((origin/n,'rgbd/'+c+'/'+n))
            rows=(origin/'frames.txt').read_text().splitlines()
            for sec in [14,22,26,30.6,33,35.6,38,42]:
                tt,r,d=min((row.split() for row in rows),key=lambda z:abs(float(z[0])-sec))
                files.extend([(origin/r,'rgbd/'+c+'/'+r),(origin/d,'rgbd/'+c+'/'+d)])
        manifest.update([pack('gate-'+case,files)])
    write(OUT/'evidence-index.json',manifest);print([(n,v['files'],v['bytes']) for n,v in manifest.items()],flush=True)
if __name__=='__main__':main()
