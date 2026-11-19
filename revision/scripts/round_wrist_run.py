"""Build a headless caller for a stock ORB-SLAM3 build, then replay the recorded RGB-D."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
from round_shape_pixels import OUTPUT

def build(orb,store):
    dest=store/'track';store.mkdir(parents=True,exist_ok=True)
    src=Path(__file__).with_name('round_wrist_orb.cc')
    if dest.exists() and dest.stat().st_mtime>=src.stat().st_mtime:return dest
    b=orb/'build';d=b/'CMakeFiles/rgbd_tum.dir';lines=(d/'flags.make').read_text().splitlines()
    flags=[]
    for key in ('CXX_FLAGS','CXX_DEFINES','CXX_INCLUDES'):
        flags+=shlex.split(next(s.partition('=')[2] for s in lines if s.startswith(key+' ')))
    link=shlex.split((d/'link.txt').read_text())
    libs=link[next(i for i,s in enumerate(link) if s.startswith('-Wl,-rpath')):]
    subprocess.run([link[0],*flags,'-w',str(src.resolve()),'-o',str(dest),*libs],cwd=b,check=True)
    return dest

def run(orb,store,case,finish,cond):
    data=store/finish/case
    count=json.loads((data/'complete.json').read_bytes())['n']
    out=OUTPUT/'slam'/finish/case
    out.mkdir(parents=True,exist_ok=True)
    csv=out/(cond+'.csv');manifest=out/(cond+'.json')
    if manifest.is_file() and csv.is_file() and json.loads(manifest.read_bytes())['processed']==count:return
    exe=build(orb,store)
    args=[exe,orb/'Vocabulary/ORBvoc.txt',data/'settings.yaml',data,cond,csv.resolve()]
    with (out/(cond+'.log')).open('w') as log:
        result=subprocess.run(list(map(str,args)),cwd=data,stdout=log,stderr=subprocess.STDOUT,timeout=420)
    actual=len(csv.read_text().splitlines())-1 if csv.is_file() else 0
    def sha(p):
        h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        return h.hexdigest()
    meta=dict(method='ORB-SLAM3::System::TrackRGBD, stock linked shared library',case=case,finish=finish,condition=cond,
              processed=actual,expected=count,exit=result.returncode,library_sha256=sha(orb/'lib/libORB_SLAM3.so'),
              vocabulary_sha256=sha(orb/'Vocabulary/ORBvoc.txt'),binary_sha256=sha(exe),
              frames_list_sha256=sha(data/'frames.txt'),settings_sha256=sha(data/'settings.yaml'))
    manifest.write_text(json.dumps(meta,indent=2))
    print(case,finish,cond,result.returncode,actual,'/',count,flush=True)
    if actual!=count or result.returncode:raise RuntimeError('Incomplete ORB-SLAM3 stream: '+str(manifest))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--orb',type=Path,default=Path(os.environ.get('ORB_SLAM3_ROOT',str(Path.home()/'桌面/Test/ORB_SLAM3'))))
    p.add_argument('--store',type=Path,default=Path('/tmp/g1_wrist_orb_20261119'))
    p.add_argument('--finish',choices=('plain','textured'),default='textured')
    p.add_argument('--conditions',nargs='+',choices=('natural','occluded'),default=['natural','occluded'])
    p.add_argument('--build',action='store_true');p.add_argument('cases',nargs='*');a=p.parse_args()
    if a.build:build(a.orb,a.store)
    for case in a.cases:
        for c in a.conditions:run(a.orb,a.store,case,a.finish,c)
