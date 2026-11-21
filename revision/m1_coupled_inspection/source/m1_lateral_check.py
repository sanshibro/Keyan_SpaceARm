"""Counterfactual geometry at measured poses: replace only the aircraft's lateral position."""
from pathlib import Path
import argparse,json
import numpy as np,mujoco
from m1_replay import build,writej
from gate_distance_audit import DistanceAudit
def evaluate(data,out,seeds=(26112161,26112167,26112173)):
    p=build();audit=DistanceAudit(p.m,tolerance_m=.00015,strict=False,obstacle_prefix='COL_');result=[]
    for seed in seeds:
        case=data/('coupled-'+str(seed));r=np.load(case/'plan/body-reference.npy')
        z=np.load(case/'audit/series.npz');t=z['t'];q=z['qpos'];x=z['body'][:,0]
        cells={}
        for label,xlo,xhi in [('side_riser',1.36,1.94),('turning_bay',2.32,3.08)]:
            ids=np.flatnonzero((t>=21)&(z['phase']=='execute')&(x>=xlo)&(x<=xhi))[::10]
            both=[]
            for i in ids:
                q0=q[i].copy();q1=q0.copy();q1[1]=np.interp(q0[0],r[:,0],r[:,1])
                gaps=[];tags=[]
                for pose in (q0,q1):
                    audit.d.qpos[:]=pose;mujoco.mj_kinematics(p.m,audit.d);lo,hi=audit.bounds()
                    b=audit._scan('body',audit.external['body'],lo,hi);a=audit._scan('arm_tool',audit.external['arm_tool'],lo,hi)
                    f=b if b['upper']<a['upper'] else a
                    gaps.append([min(b['lower'],a['lower']),min(b['upper'],a['upper'])]);tags.append([f['name_a'],f['name_b']])
                both.append(dict(t=t[i],x=q0[0],y_displacement_m=q0[1]-q1[1],observed=gaps[0],substitute_reference_y=gaps[1],
                                 observed_pair=tags[0],substitute_pair=tags[1]))
            if both:
                witness=min(both,key=lambda v:v['substitute_reference_y'][1])
                cells[label]=dict(samples=len(both),paired_witness=witness,minimum_observed_m=min(z['observed'][0] for z in both),
                                minimum_counterfactual_m=min(z['substitute_reference_y'][0] for z in both),paired_series=both)
        result.append(dict(seed=seed,cuts=cells))
    writej(out,result)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('out',type=Path)
    p.add_argument('--seeds',type=int,nargs='+',default=(26112161,26112167,26112173));a=p.parse_args();evaluate(a.data,a.out,a.seeds)
