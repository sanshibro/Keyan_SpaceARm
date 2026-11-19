"""Execute the registered split-policy checks and combine the native evidence manifests."""
import json
import time
import hashlib
import tarfile
from pathlib import Path
import numpy as np
import pandas as pd
from round_g1_model import Geometry,make_plant
from gate_shape_timing import GateEnvelope
from round_g1_distance import Solids
from round_g1_online_run import P,DEST
from round_g1_audit import save
from round_g1_online_audit import score_one
from round_g1_online_split import SPEC

def main():
    base=json.loads((DEST/'audit.json').read_text())['runs']
    m=pd.read_csv(DEST/'metrics.csv')
    geom=Geometry(make_plant());env=GateEnvelope(geom,P['controller']['angle_pad_per_degree_m']);solids=Solids(geom)
    all_runs=dict(base);rows=[]
    for s,variants in SPEC['order'].items():
        for v in variants:
            name=s+'-'+v;folder=DEST/'runs'/name
            limit=time.monotonic()+240
            while not (folder/'flight.ulg').is_file() and time.monotonic()<limit:time.sleep(1.)
            row,d=score_one(folder,env,solids,True);rows.append(row);all_runs[name]=d
            save(DEST/'per-run'/(name+'.json'),dict(row=row,detail=d))
            print(name,row['pass_all'],row['failed'],round(row['arm_convex_cm'],3),row['return_t'],flush=True)
    scores=pd.concat([m,pd.DataFrame(rows)],ignore_index=True)
    assert len(scores)==21
    scores.to_csv(DEST/'all-metrics.csv',index=False)
    # Check all 250 Hz translational commands per condition, including the two new interventions.
    pairs=[]
    for s in SPEC['order']:
        seed=int(s);sel=scores[(scores.seed==seed)&(scores.condition=='pause')].set_index('method')
        a,b,c,d=(sel.loc[k] for k in ('phase','phasebend','onlinebend','online'))
        one=[]
        for _,r in sel.iterrows():
            folder=DEST/'runs'/r['case']
            w=pd.read_csv(folder/'hil.csv');release=json.loads((folder/'run.json').read_text())['release_us']
            hit=(w.us>=release)&(w.us<=release+52e6)
            one.append(w.loc[hit,['rx','ry','rz']].to_numpy())
        for q in one[1:]:np.testing.assert_array_equal(q,one[0])
        value=lambda r:float(r.arm_convex_cm)
        cells={r['case']:dict(minimum_cm=value(r),critical_time_s=all_runs[r['case']]['convex']['arm_tool']['t'],return_s=float(r.return_t),
                               target_time_s=float(r.target_time_s)) for r in (a,b,c,d)}
        pairs.append(dict(seed=seed,phase_out_phase_back_cm=value(a),phase_out_online_back_cm=value(b),
            online_out_phase_back_cm=value(c),online_out_online_back_cm=value(d),
            outward_effect_phase_return_cm=value(c)-value(a),outward_effect_online_return_cm=value(d)-value(b),
            return_effect_phase_out_cm=value(b)-value(a),return_effect_online_out_cm=value(d)-value(c),
            interaction_cm=(value(d)-value(b))-(value(c)-value(a)),cells=cells))
    save(DEST/'factorial.json',dict(supplementary=True,condition='pause',pairs=pairs,
        interpretation='Repeated independent SITL starts, matching simulated noise streams and identical base commands. Small run-to-run differences include asynchronous native PX4 transport.'))
    save(DEST/'all-audit.json',dict(runs=all_runs,primary=15,supplementary=6,base_references_verified=True))
    dest=DEST/'evidence';dest.mkdir(exist_ok=True);manifest={}
    for name,d in all_runs.items():
        source=DEST/'runs'/name;target=dest/(name+'.tar.gz')
        if not target.exists():
            tmp=target.with_suffix('.partial')
            with tarfile.open(str(tmp),'w:gz',compresslevel=5) as tar:
                for n in sorted(list(d['sha256'])+['px4-console.txt']):
                    tar.add(source/n,arcname=name+'/'+n,recursive=False)
            tmp.replace(target)
        with tarfile.open(str(target),'r:gz') as tar:
            for n,h in d['sha256'].items():assert hashlib.sha256(tar.extractfile(name+'/'+n).read()).hexdigest()==h
        manifest[target.name]=dict(sha256=hashlib.sha256(target.read_bytes()).hexdigest(),bytes=target.stat().st_size)
    save(DEST/'all-evidence-index.json',manifest)
    for r in pairs:
        print(r['seed'],*[round(r[k],3) for k in ('phase_out_phase_back_cm','phase_out_online_back_cm','online_out_phase_back_cm','online_out_online_back_cm')],flush=True)


if __name__=='__main__':main()
