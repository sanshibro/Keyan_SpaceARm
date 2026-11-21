"""Run native M2 flights sequentially, with an immutable slot per attempt and a verified pilot receipt."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import median
from m2_gate import DEV,FORMAL,METHODS,authorize,certify,digest,identity

ROOT=Path(__file__).resolve().parents[1]
SCRIPTS=ROOT/'source'

def save(path,value):
    tmp=path.with_suffix('.new')
    tmp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    tmp.replace(path)

def cmd(program,args,log):
    with log.open('x',encoding='utf8') as dst:
        subprocess.run([sys.executable,'-u',str(SCRIPTS/program)]+[str(x) for x in args],stdout=dst,stderr=subprocess.STDOUT,check=True)

def lead_check(m):
    if not(m['success'] and m['native_logs_verified'] and m['synchronized_rgbd'] and m['gripper_open_command'] and
           m['task_depth_dwell_s']>=1.-1e-7 and m['hazard_lower_m']>=.005 and m['external_lower_m']>=.062 and
           m['motion_to_inspect_s'] is not None and 24.<=m['motion_to_inspect_s']<=34. and
           m['abnormal_stall_s']==m['planner_pause_s']==0 and m['async_coverage'] and m['rolling_verified'] and
           m['causal_map_verified'] and m['activated_replans']>=3 and len(m['solve_wall_s']) and
           max(m['solve_wall_s'])<=.95 and median(m['solve_wall_s'])<1.):
        raise RuntimeError('Coupled pilot failed; see the recorded metrics before further runs')

def run(mode,records,gate,instance,name):
    records=records.resolve();gate=gate.resolve()
    if not name or Path(name).name!=name:raise ValueError('attempt name')
    root=records/mode/name;root.mkdir(parents=True,exist_ok=True)
    status=root/'campaign.json';frozen=identity()
    if status.exists():
        state=json.loads(status.read_text())
        if state['identity']!=frozen or state['gate_digest']!=digest(gate):raise ValueError('changed configuration: start a fresh attempt')
    else:
        state=dict(identity=frozen,gate_digest=digest(gate),mode=mode,rows=[],active=None)
        save(status,state)
    matrix=([('coupled',s) for s in DEV]+[(k,DEV[0]) for k in METHODS if k!='coupled']) if mode=='development' else [
            (METHODS[(j+2*i)%len(METHODS)],s) for i,s in enumerate(FORMAL) for j in range(len(METHODS))]
    metrics=[]
    for k,s in matrix:
        authorize(k,s,mode=='development',gate)
        slot=root/('%s-%d'%(k,s));evidence=slot/'audit'/'metrics.json'
        prior=next((r for r in state['rows'] if r['strategy']==k and r['seed']==s),None)
        if prior:
            if not evidence.exists() or digest(evidence)!=prior['sha256']:raise ValueError('prior evidence mismatch')
        else:
            if slot.exists():raise FileExistsError('existing incomplete slot: '+str(slot))
            if identity()!=frozen:raise ValueError('runtime changed during the campaign')
            slot.mkdir();state['active']=dict(strategy=k,seed=s,flight=str(slot/'flight'),started_wall=time.time());save(status,state)
            args=[k,s,instance,slot/'flight','--gate',gate]
            if mode=='development':args.append('--development')
            cmd('m2_flight.py',args,slot/'native.log')
            cmd('m2_score.py',[slot/'flight',slot/'audit'],slot/'audit.log')
            m=json.loads(evidence.read_text());state['rows'].append(dict(strategy=k,seed=s,success=m['success'],path=str(evidence),sha256=digest(evidence)))
            state['active']=None;save(status,state)
        m=json.loads(evidence.read_text())
        if mode=='development' and k=='coupled':lead_check(m)
        metrics.append(evidence)
    if mode=='development':certify(gate,metrics,root/'qualification.json')
    state['complete']=True;save(status,state)
    save(root/'flight-manifest.json',dict(cases=[dict(flight=os.path.relpath(e.parent.parent/'flight',root),audit=os.path.relpath(e.parent,root)) for e in metrics]))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['development','formal']);p.add_argument('name')
    p.add_argument('records',type=Path);p.add_argument('gate',type=Path);p.add_argument('--instance',type=int,default=4)
    a=p.parse_args();a.records.mkdir(parents=True,exist_ok=True)
    with (a.records/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(a.mode,a.records,a.gate,a.instance,a.name)
