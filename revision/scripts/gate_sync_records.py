"""Mirror completed NUC flight evidence into the local group-meeting round."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile

SERVER='nuc-uav'
BASE='/home/uav/paper_revision_20261117/revision/data/round_gate_feedback'
SELECTED=26112131

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as fp:
        for b in iter(lambda:fp.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def fetch(remote,local):
    local.parent.mkdir(parents=True,exist_ok=True)
    staging=local.with_suffix(local.suffix+'.part')
    subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=18',SERVER+':'+remote,str(staging)],check=True,timeout=210)
    return staging

def save_json(path,d):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');temp.replace(path)

def sync(stage,known):
    protocol=json.loads((stage/'records/formal-design.json').read_text(encoding='utf8'))
    expected={v['case']:v for v in protocol['cases']}
    dec=stage/'decisions';dec.mkdir(exist_ok=True)
    metrics=dec/'metrics.json';stagefile=fetch(BASE+'/decisions/metrics.json',metrics)
    data=json.loads(stagefile.read_text(encoding='utf-8'));stagefile.replace(metrics)
    if data['design_commit']!=protocol['design_commit']:raise ValueError('Unexpected design commit')
    for k,item in data['evidence'].items():
        if k not in expected or item['file']!=k+'.zip':raise ValueError('Unknown case')
        out=stage/'evidence'/item['file']
        if k in known and out.is_file() and out.stat().st_size==item['bytes']:continue
        if not out.is_file() or sha(out)!=item['sha256']:
            tmp=fetch(BASE+'/evidence/'+item['file'],out)
            if tmp.stat().st_size!=item['bytes'] or sha(tmp)!=item['sha256']:raise IOError('Failed ZIP fingerprint: '+k)
            tmp.replace(out)
        meta=expected[k]
        if meta['seed']==SELECTED and meta['material']=='differential' and meta['vision']=='occluded':
            dest=stage/'formal'/k;dest.mkdir(parents=True,exist_ok=True)
            with zipfile.ZipFile(out) as z:
                digests=json.loads(z.read('evidence-hashes.json'))
                for name in ('physics.npz','score-trace.npz','control-summary.json','run.json','metrics.json','protocol.json','optics/index.jsonl','optics/complete.json'):
                    target=dest/name;target.parent.mkdir(parents=True,exist_ok=True)
                    b=z.read(name)
                    if hashlib.sha256(b).hexdigest()!=digests[name]:raise IOError(name)
                    target.write_bytes(b)
        known.add(k);print('received',len(known),k,flush=True)
    progress=stage/'records/current.json'
    try:
        temp=fetch(BASE+'/campaign-progress.json',progress);json.loads(temp.read_text());temp.replace(progress)
    except (OSError,subprocess.SubprocessError):pass
    save_json(stage/'records/verified-evidence.json',{'design':data['design_commit'],'count':len(known),'archives':sorted(known)})
    return len(known)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',type=Path);parser.add_argument('--once',action='store_true');args=parser.parse_args()
    known=set();start=time.monotonic()
    while True:
        try:
            n=sync(args.stage,known)
            if n==60 or args.once:break
        except (OSError,subprocess.SubprocessError,ValueError) as e:
            print('waiting for remote verification',e,flush=True)
            if args.once:raise
        time.sleep(45)
