"""Mirror closed experimental cells over SSH as immutable, per-flight tar streams."""
import argparse,hashlib,json,os,subprocess,tarfile,time
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];REMOTE='/home/uav/paper_revision_20261117/revision/data/round_m1_coupled'
def ssh(*args,**opts):return subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','nuc-uav',*args],check=True,**opts)
def stream(folder,target,items):
    target.mkdir(parents=True,exist_ok=True)
    temp=target/'.source-transfer.tar'
    with temp.open('wb') as w:ssh('tar','-C',folder,'-cf','-',*items,stdout=w,timeout=700)
    with tarfile.open(temp) as bundle:
        for f in bundle:
            dest=(target/f.name).resolve()
            if not dest.is_relative_to(target.resolve()) or f.issym() or f.islnk():raise ValueError('Unexpected archive member')
            bundle.extract(f,target,filter='data')
    temp.unlink()
def mirror():
    target=BASE/'records/formal';target.mkdir(parents=True,exist_ok=True)
    registered=target/'registered.json';done=set()
    while len(done)<15:
        try:
            cmd="find "+REMOTE+"/publication -mindepth 3 -maxdepth 3 -type f -name control-summary.json"
            present=ssh(cmd,stdout=subprocess.PIPE,text=True,timeout=15).stdout.splitlines()
            if not registered.exists() and present:stream(REMOTE+'/publication',target,['registered.json','plan-registration.json'])
            for item in sorted(present):
                n=Path(item).parts[-3]
                key=n.split('-')
                if len(key)!=2 or key[0] not in ('straight','fixed','sequential','coupled','unregularized') or key[1] not in ('26112161','26112167','26112173'):continue
                d=target/n;finish=d/'mirror.json'
                if not finish.exists():
                    print('loading',n,flush=True);stream(REMOTE+'/publication/'+n,d,['flight','plan'])
                    c=json.loads((d/'flight/control-summary.json').read_text());r=json.loads((d/'flight/run.json').read_text())
                    if c['case']['case']!=n or r['error'] or not (d/'flight/flight.ulg').stat().st_size:raise ValueError(n)
                    finish.write_text(json.dumps({'name':n,'source':item,'captures':c['captures']}),'utf8')
                done.add(n)
        except (OSError,subprocess.SubprocessError,tarfile.TarError) as e:print('transfer retry',type(e).__name__,flush=True)
        if len(done)<15:time.sleep(16)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--pilot',action='store_true');v=a.parse_args()
    if v.pilot:stream(REMOTE+'/pilot13',BASE/'work/portability',['flight','audit'])
    else:mirror()
