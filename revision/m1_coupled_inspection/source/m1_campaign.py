"""Registered paired closed-loop campaign. Each cell owns its solve, SITL, pixels and ULog."""
import argparse,fcntl,hashlib,json,os,subprocess,sys,time,threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from m1_layout import METHODS,SEEDS
from m1_plan import writej
from m1_plant import SCENE
from gate_feedback_model import verify_frozen

HERE=Path(__file__).resolve().parent
def cmd(script,args,log):
    with log.open('a') as stream:
        subprocess.run([sys.executable,'-u',str(HERE/script),*map(str,args)],stdout=stream,stderr=subprocess.STDOUT,check=True)
def run(folder):
    folder=folder.resolve();folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'campaign.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    order=[(method,seed) for i,seed in enumerate(SEEDS) for method in METHODS[i:]+METHODS[:i]]
    digest=json.loads((SCENE/'manifest.json').read_text())['scene_sha256']
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE).decode().strip()
    spec=dict(scene_sha256=digest,perception_lock=verify_frozen(),source_commit=commit,cells=order,shared_pose='surveyed hover; online PX4 and fused shape feedback',
          score='6 cm external; positive self gap; zero contact; concurrent 3.5 cm, 3 deg and optical standoff 12-18 cm for 1 s',
          planning='each cell independently solved, unchanged source, serial wall-clock timing')
    prereg=folder/'registered.json'
    if prereg.exists() and json.loads(prereg.read_text())['source_commit']!=commit:raise RuntimeError('Registered source changed')
    if not prereg.exists():writej(prereg,spec)
    cases=[]
    for method,seed in order:
        phase=folder/('%s-%d'%(method,seed));phase.mkdir(exist_ok=True);plan=phase/'plan'
        if not (plan/'decision.json').exists():
            print('solve',phase.name,flush=True);cmd('m1_plan.py',[method,plan],phase/'planner.log')
        d=json.loads((plan/'decision.json').read_text());assert d['map']==digest
        cases.append((method,seed,phase,plan))
    launch_lock=threading.Lock()
    def available_mib():
        with open('/proc/meminfo') as source:
            return int(next(line for line in source if line.startswith('MemAvailable:')).split()[1])//1024
    def launch_flight(method,seed,instance,plan,flight,log):
        with log.open('a') as stream:
            # Load only one native RGB-D vocabulary at a time. A nonempty exposure
            # index marks completed startup. Post-processing begins after flights.
            with launch_lock:
                while available_mib()<2700:time.sleep(5)
                print('release',flight.parent.name,flush=True)
                process=subprocess.Popen([sys.executable,'-u',str(HERE/'m1_flight.py'),method,str(seed),str(instance),str(plan),str(flight)],stdout=stream,stderr=subprocess.STDOUT)
                index=flight/'optics/index.jsonl'
                while process.poll() is None and not (index.exists() and index.stat().st_size>512):time.sleep(3)
            if process.wait()!=0:raise RuntimeError('Preserve native startup and flight logs: '+str(flight))
    def flights(items,instance):
        for method,seed,phase,plan in items:
            flight=phase/'flight'
            if not (flight/'control-summary.json').is_file():
                if flight.exists():
                    runfile=flight/'run.json'
                    if runfile.is_file() and json.loads(runfile.read_text()).get('error') is None:
                        raise RuntimeError('Closed flight has incomplete sensor export: '+str(flight))
                    raise RuntimeError('Technical interruption, inspect preserved attempt: '+str(flight))
                launch_flight(method,seed,instance,plan,flight,phase/'flight.log')
            assert json.loads((flight/'run.json').read_text())['error'] is None
            print('landed log',phase.name,flush=True)
    workers=max(1,min(2,int(os.environ.get('M1_FLIGHT_WORKERS','2'))))
    with ThreadPoolExecutor(workers) as pool:
        jobs=[pool.submit(flights,cases[i::workers],4+i) for i in range(workers)]
        for job in jobs:job.result()
    print('all flight logs complete',flush=True)
    def score(case):
        method,seed,phase,plan=case;audit=phase/'audit'
        if not (audit/'metrics.json').exists():cmd('m1_score.py',[phase/'flight',audit],phase/'audit.log')
        return json.loads((audit/'metrics.json').read_text())
    # Each independent convex-geometry audit owns a second complete native model.
    with ThreadPoolExecutor(1) as pool:metrics=list(pool.map(score,cases))
    writej(folder/'matrix.json',metrics);print('closed',len(metrics),'cells',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);run(p.parse_args().folder)
