"""Post-hoc geometric attribution using two synchronized, recorded state histories per seed."""
import json
import numpy as np
import pandas as pd
from round_g1_model import DATA, P, Geometry, make_plant
from round_g1_distance import Solids
from round_g1_audit import save


def load(case):
    folder=DATA/'runs'/case
    r=json.loads((folder/'run.json').read_text())['release_us']
    with np.load(str(folder/'physics.npz')) as z:
        return r,z['us'].copy(),z['qpos'].copy()


def mixed(solid, base, shape):
    r,us,q=base;r2,us2,other=shape;times=(us-r)/1e6
    kin=q.copy()
    for j in range(7,q.shape[1]):
        kin[:,j]=np.interp(times,(us2-r2)/1e6,other[:,j])
    rows=[];g=solid.g
    for i in np.flatnonzero((times>=0)&(times<=P['deadline_s'])):
        g.from_record(kin[i]);d=g.distances(*g.bounds())
        rows.append([times[i],float(d[g.cat!='body'].min())])
    tr=pd.DataFrame(rows,columns=['t','arm_tool_m'])
    return solid.scan(dict(qpos=kin,us=us),tr,None,'arm_tool',r)


def main():
    exact=json.loads((DATA/'convex-distance.json').read_text())['runs'];rows=[]
    solid=Solids(Geometry(make_plant()))
    for seed in P['seed_order']:
        n=seed+'-variable_nominal';s=seed+'-variable_timed';nraw,sraw=load(n),load(s)
        nn=100*exact[n]['arm_tool']['upper_m'];ss=100*exact[s]['arm_tool']['upper_m']
        ns=mixed(solid,nraw,sraw);sn=mixed(solid,sraw,nraw)
        ns_cm=100*ns['upper_m'];sn_cm=100*sn['upper_m']
        rows.append(dict(seed=int(seed),nominal_actual_cm=nn,nominal_base_timed_arm_cm=ns_cm,
            timed_base_nominal_arm_cm=sn_cm,timed_actual_cm=ss,within_pair_change_cm=ss-nn,
            arm_swap_at_nominal_base_cm=ns_cm-nn,base_swap_at_nominal_arm_cm=sn_cm-nn,
            nominal_base_timed_arm=ns,timed_base_nominal_arm=sn))
    save(DATA/'pose-swap.json',dict(method='Offline geometry only. Change base pose or post-base joint coordinates between two synchronized simulated flights of the same seed. Shape coordinates linearly interpolated at each recorded base time. Recompute the 3-D convex clearance minimum over the complete flight. Hybrid histories are not dynamically integrated.',pairs=rows))
    for r in rows: print(r['seed'],*[round(r[k],2) for k in ('nominal_actual_cm','nominal_base_timed_arm_cm','timed_base_nominal_arm_cm','timed_actual_cm')])


if __name__=='__main__':main()
