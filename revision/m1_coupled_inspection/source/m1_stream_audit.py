"""Score native flight logs on the workstation with a cross-platform replay check."""
import argparse,hashlib,json,os,subprocess,sys,tarfile,time
from pathlib import Path
import numpy as np
from m1_mirror import BASE,REMOTE
def read(p):return json.loads(p.read_text(encoding='utf8'))
def check(port):
    return compare(port/'audit',port/'windows-audit')
def compare(native,workstation):
    a=read(native/'metrics.json');b=read(workstation/'metrics.json');maximum=0.;differences={}
    keys=['task_success','collision_steps','task_dwell_s','optical_dwell_s','tracked_frames','matching_exposures',
          'final_pos_m','final_axis_deg','final_optical_m','peak_mount_reaction_Nm','peak_front_rear_N',
          'rms_front_rear_N','peak_tracking_m']
    for k in keys:
        delta=abs(float(a[k])-float(b[k]));maximum=max(delta,maximum);assert delta<2e-6,(k,delta)
        differences[k]=delta
    lx,ux=a['self_interval_50Hz_m'];ly,uy=b['self_interval_50Hz_m']
    assert lx>0 and ly>0 and max(lx,ly)<=min(ux,uy)+1e-8
    assert max(ux-lx,uy-ly)<=.00015
    delta=abs(a['self_clearance_50Hz_m']-b['self_clearance_50Hz_m'])
    differences['self_clearance_50Hz_m']=delta;maximum=max(maximum,delta);assert delta<=.00015
    for k in ('body','arm_tool'):
        x,y=a['clearance'][k],b['clearance'][k]
        assert max(x['lower'],y['lower'])<=min(x['upper'],y['upper'])+1e-8,(k,x,y)
        assert x['pair']==y['pair'] and abs(x['t']-y['t'])<.021
        delta=abs(x['m']-y['m']);maximum=max(maximum,delta);assert delta<=.00015,(k,delta)
        differences['minimum_'+k+'_m']=delta
    with np.load(native/'series.npz') as x,np.load(workstation/'series.npz') as y:
        for k in ('external_exact','self_exact','mount_reaction','rotors','tracking','uv','actual_b'):
            delta=np.max(abs(x[k]-y[k]));maximum=max(maximum,delta)
            differences[k]=float(delta)
            tolerance=.00015 if k in ('external_exact','self_exact') else 1e-6
            assert delta<=tolerance,(k,delta)
    return {'case':a['case'],'checks':keys,'max_abs_difference':maximum,'differences':differences,'signed_distance_tolerance_m':.00015,'linux_broad_phase':'component boxes and cylindrical axes',
            'workstation_broad_phase':'independent global AABBs','exact_evaluator':'same signed convex GJK/EPA and native MuJoCo 3.2.3 dynamics'}
def send(dest,case):
    a=dest/'audit.tmp.tar'
    with tarfile.open(a,'w') as tar:
        for p in sorted((dest/'audit').iterdir()):tar.add(p,arcname=p.name)
    remote=REMOTE+'/publication/'+case
    cmd=f"set -e; cd '{remote}'; test -s flight/control-summary.json; mkdir -p audit-verified-transfer; tar -xf - -C audit-verified-transfer; if test ! -d audit; then mv audit-verified-transfer audit; fi; test -s audit/metrics.json"
    with a.open('rb') as stream:subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','nuc-uav',cmd],stdin=stream,check=True,timeout=150)
    a.unlink();(dest/'audit-synced').write_text('registered\n')
def main(publish):
    proof=check(BASE/'work/portability');(BASE/'qa/dual-platform-replay.json').write_text(json.dumps(proof,indent=2))
    base=BASE/'records/formal';done=set()
    while len(done)<15:
        for marker in sorted(base.glob('*/mirror.json')):
            folder=marker.parent;n=folder.name
            if n in done:continue
            prefix=folder/'audit';m=prefix/'metrics.json'
            if not m.exists():
                print('audit',n,flush=True)
                with (folder/'workstation-audit.log').open('w') as log:
                    subprocess.run([sys.executable,'-u',str(BASE/'source/m1_score.py'),str(folder/'flight'),str(prefix)],stdout=log,stderr=subprocess.STDOUT,check=True)
                cert={'same_model':json.loads((BASE/'work/native-task.json').read_text())['mjb_sha256'],
                      'code_sha256':hashlib.sha256((BASE/'source/m1_score.py').read_bytes()).hexdigest(),'case':n,'cross_platform':proof}
                (prefix/'validation-provenance.json').write_text(json.dumps(cert,indent=2))
            if publish and not (folder/'audit-synced').exists():
                try:send(folder,n)
                except (OSError,subprocess.SubprocessError):continue
            done.add(n)
        if len(done)<15:time.sleep(10)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--publish',action='store_true');main(p.parse_args().publish)
