"""Release all estimates from three whitelisted sensor packet types before scoring."""
import argparse, hashlib, json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from augmented_bend_kf import Settings,estimate_clock,vision_release,strain_release

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'revision/data/round_augmented_fusion'
PROTOCOL=ROOT/'revision/protocols/round_aug_contract.json'
SPEC=json.loads(PROTOCOL.read_bytes())
METHODS=SPEC['methods']
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_bytes((json.dumps(data,ensure_ascii=False,allow_nan=False,indent=2)+'\n').encode());tmp.replace(p)
def packet(path,keys):
    with np.load(path,allow_pickle=False) as f:return {key:f[key] for key in keys}
def read_inputs(case,source,cond):
    base=OUT/'geometry'/(case+'.npz');s=OUT/'strain'/case/(source+'.npz');v=OUT/'vision'/case/(cond+'.npz')
    times=packet(base,('t',))['t']
    times=times[times<SPEC['phases']['all'][1]+.201]
    return times,packet(s,('t','b','R')),packet(v,('t','b','R','valid','quality','ready'))
def get_config():
    pack=json.loads((OUT/'frozen-settings.json').read_bytes())
    assert pack['protocol']==digest(PROTOCOL)
    for name,d in pack['software'].items():
        if digest(ROOT/name)!=d:raise ValueError('Estimator or input pipeline changed after freeze: '+name)
    return Settings(**pack['settings'])
def estimates(t,s,v,c):
    p=estimate_clock(t,s,v,c,True);fixed=estimate_clock(t,s,v,c,False)
    return dict(t=t,b=np.stack([vision_release(t,v,c.delay),strain_release(t,s),fixed['b'],p['b']]),bias=p['bias'],
                gain_v=p['gain_v'],gain_s=p['gain_s'],confidence_v=p['confidence_v'],confidence_s=p['confidence_s'],
                accepted=p['accepted'],nis=p['nis'],cov=p['P'],capture=v['t'],optical=v['b'],optical_valid=v['valid'])
def truncated(s,mask):return {k:v[mask] for k,v in s.items()}
def release(case,source,cond,c,do_prefix=False):
    t,s,v=read_inputs(case,source,cond)
    est=estimates(t,s,v,c);output=OUT/'released'/case/(source+'-'+cond+'.npz')
    output.parent.mkdir(exist_ok=True,parents=True);np.savez_compressed(output,**est)
    audit={}
    if do_prefix:
        for cut in (24.96,30.98,33.44,35.90,39.54):
            take=t<=cut;at=t[take][-1]
            ss=truncated(s,s['t']<=at+1e-9);vv=truncated(v,v['t']+c.delay<=at+1e-9)
            short=estimates(t[take],ss,vv,c)
            vals={}
            for key in ('b','bias','gain_v','gain_s','confidence_v','confidence_s','cov','accepted','nis'):
                comb=key=='b';a=est[key][:,take] if comb else est[key][take];b=short[key]
                if not np.array_equal(a,b,equal_nan=True):raise AssertionError((case,source,cond,cut,key))
                vals[key]=0.
            audit[str(cut)]=vals
    return output,audit
def run():
    import sys
    # Process-level evidence wall in addition to the member allow-list in read_inputs.
    # The process may open only the declared NPZ sensor/clock streams, or its own writable outputs.
    def guard(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)):return
        p=Path(args[0].decode() if isinstance(args[0],bytes) else args[0]).resolve()
        if p.suffix!='.npz' or OUT/'released' in p.parents:return
        if p.parent not in permitted:raise PermissionError('Input isolation: '+p.name)
    permitted={OUT/'geometry'}|{OUT/k/case for k in ('strain','vision') for case in SPEC['cases']}
    sys.addaudithook(guard)
    c=get_config();index={};audit={};bound={}
    for case in SPEC['cases']:
        for source in SPEC['sources']:
            for cond in SPEC['conditions']:
                p,pre=release(case,source,cond,c,source=='differential' and cond=='occluded')
                key=p.relative_to(OUT).as_posix();index[key]=digest(p)
                if pre:audit[key]=pre
        print('Sealed released estimates',case,flush=True)
    for case in SPEC['cases']:
        for kind,names in [('strain',SPEC['sources']),('vision',SPEC['conditions'])]:
            for name in names:
                p=OUT/kind/case/(name+'.npz');bound[p.relative_to(OUT).as_posix()]=digest(p)
    write(OUT/'release-lock.json',dict(frozen_settings=digest(OUT/'frozen-settings.json'),files=index,sensor_packets=bound,prefix=audit,
              scope='Inference reads clock, sensor samples, and optical validity/uncertainty only. A separate process joins these sealed results with evaluation labels.'))
if __name__=='__main__':run()
