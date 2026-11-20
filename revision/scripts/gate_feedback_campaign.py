"""Resume the registered, paired single-gate flights without replacing completed records."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

from gate_feedback_spec import ROOT,OUT,P,SPECFILE,digest,dump,verify_frozen

SCRIPTS=['gate_feedback_flight.py','gate_feedback_model.py','round_f_sitl.py','gate_feedback_scores.py',
         'gate_live_optics.py','gate_live_orb.cc','gate_live_shape.py','gate_distance_audit.py',
         'round_f_plant.py','gate_feedback_spec.py']
PROTOCOL=OUT/'formal-design.json'
DEC=OUT/'decisions'
EVID=OUT/'evidence'

def sources():
    return {str(p.relative_to(ROOT)):digest(p) for p in [SPECFILE]+[ROOT/'revision/scripts'/n for n in SCRIPTS]}

def design():
    matrix={(r['method'],r['material'],r['vision'],r['seed']) for r in P['cases']}
    if len(matrix)!=60 or len({r['case'] for r in P['cases']})!=60:raise ValueError('Incomplete or duplicate cases')
    observed=sources()
    if PROTOCOL.exists():
        locked=json.loads(PROTOCOL.read_text())
        if locked['inputs']!=observed or locked['perception_lock']!=verify_frozen():
            raise RuntimeError('Registered flight or scoring source differs from the formal design')
    else:
        if (OUT/'formal').exists() and any((OUT/'formal').iterdir()):raise RuntimeError('Preexisting formal trials without a lock')
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        for name,h in observed.items():
            blob=subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT)
            if hashlib.sha256(blob).hexdigest()!=h:raise RuntimeError('Commit the design before formal registration: '+name)
        dump(PROTOCOL,dict(inputs=observed,perception_lock=verify_frozen(),design_commit=commit,
                           cases=P['cases'],registered_utc=time.strftime('%FT%TZ',time.gmtime())))
    return json.loads(PROTOCOL.read_text())

def archive(folder):
    EVID.mkdir(parents=True,exist_ok=True)
    path=EVID/(folder.name+'.zip')
    core=['flight.ulg','flight.bag','physics.npz','control.npz','model.xml','protocol.json','score-trace.npz','metrics.json',
          'run.json','provenance.json','control-summary.json','events.jsonl','optics/index.jsonl','optics/complete.json',
          'optics/native.csv','optics/optical.jsonl','optics/observations.npz','optics/fixed-calibration.json',
          'sensors/strain_observations.npz','sensors/strain_generation.json','sensors/estimator_provenance.json']
    core+=sorted(str(p.relative_to(folder)) for p in (folder/'sensors').glob('estimate_*.npz'))
    hashes={n:digest(folder/n) for n in core}
    dump(folder/'evidence-hashes.json',hashes)
    tmp=path.with_suffix('.part')
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=2) as f:
        for name in core+['evidence-hashes.json']:f.write(folder/name,name)
    with zipfile.ZipFile(tmp) as f:
        if f.testzip() is not None:raise IOError('Corrupt archive for '+folder.name)
    tmp.replace(path)
    return dict(file=path.name,bytes=path.stat().st_size,sha256=digest(path))

def execute(argv,log):
    with log.open('ab') as f:
        out=subprocess.run([sys.executable,'-u',*argv],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    if out.returncode:raise RuntimeError('Process returned %d; inspect %s'%(out.returncode,log))

def acquire(c):
    folder=OUT/'formal'/c['case']
    log=OUT/'logs'/(c['case']+'.log');log.parent.mkdir(exist_ok=True)
    if not (folder/'run.json').exists() and not folder.exists():
        execute(['revision/scripts/gate_feedback_flight.py','--formal','--case',c['case']],log)
    if not (folder/'run.json').exists():raise RuntimeError('Incomplete acquisition; inspect '+str(folder))
    run=json.loads((folder/'run.json').read_text())
    if run['error']:raise RuntimeError('Flight requires adjudication: '+str(run['error']))
    if not (folder/'metrics.json').exists():execute(['revision/scripts/gate_feedback_scores.py',str(folder)],log)
    metric=json.loads((folder/'metrics.json').read_text())
    if metric['case']!=c:raise RuntimeError('Mismatched case identity')
    return folder,metric

def checkpoint(n):
    paths=[str(p.relative_to(ROOT)) for p in [PROTOCOL,DEC/'metrics.json',EVID]]
    subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True)
    subprocess.run(['git','commit','--only','-m','Archive paired gate flights %02d of 60'%n,'--',*paths],cwd=ROOT,check=True)
    push=['git','-c','core.sshCommand=ssh -o BatchMode=yes -o ConnectTimeout=12 -o HostKeyAlias=github.com -o HostName=ssh.github.com -p 443 -o StrictHostKeyChecking=yes','push','origin','HEAD']
    with (OUT/'push.log').open('ab') as f:
        for i in range(3):
            if subprocess.run(push,cwd=ROOT,stdout=f,stderr=f).returncode==0:return True
    return False

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--register',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'campaign.lock').open('a') as fh:
        fcntl.flock(fh,fcntl.LOCK_EX|fcntl.LOCK_NB)
        locked=design()
        if args.register:
            print('registered',locked['design_commit'],len(locked['cases']));return
        DEC.mkdir(exist_ok=True);EVID.mkdir(exist_ok=True)
        result_file=DEC/'metrics.json'
        result=json.loads(result_file.read_text()) if result_file.is_file() else dict(design_commit=locked['design_commit'],cases={},evidence={})
        for c in P['cases']:
            name=c['case']
            if name in result['cases']:
                item=result['evidence'][name]
                if digest(EVID/item['file'])!=item['sha256']:raise IOError('Evidence hash mismatch: '+name)
                continue
            design()
            dump(OUT/'campaign-progress.json',dict(completed=len(result['cases']),total=60,active=name,phase='fly / score'))
            folder,metric=acquire(c)
            item=archive(folder)
            result['cases'][name]=metric;result['evidence'][name]=item
            dump(result_file,result)
            n=len(result['cases']);print('%02d/60'%n,name,'success',metric['success'],flush=True)
            if n%5==0:
                ok=checkpoint(n)
                print('checkpoint',n,'pushed',ok,flush=True)
        dump(OUT/'campaign-progress.json',dict(completed=60,total=60,active=None,phase='complete'))

if __name__=='__main__':main()
