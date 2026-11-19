"""Estimator replay tests, convex geometry checks and strictly pre-retraction comparisons."""
import json
import math
import mujoco
import numpy as np
import pandas as pd
from gate_shape_timing import GateEnvelope,RollingGateTiming,enu_body_rotation
from round_f_plant import LAUNCH
from round_g1_audit import save
from round_g1_distance import Solids
from round_g1_model import Geometry,make_plant
from round_g1_online_run import DEST,P,Route,phase_time


def bounds_test():
    g=Geometry(make_plant());env=GateEnvelope(g,P['controller']['angle_pad_per_degree_m']);s=Solids(g)
    rng=np.random.default_rng(20261119);smallest=math.inf;largest=0.;count=0
    for j in range(36):
        theta=float(rng.uniform(0,118));p=np.array([rng.uniform(.58,1.48),rng.uniform(-.10,.10),rng.uniform(2.10,2.21)])
        rpy=np.array([rng.uniform(-.17,.17),rng.uniform(-.17,.17),np.pi/2+rng.uniform(-.23,.23)])
        R=enu_body_rotation(rpy)
        prediction=env.inspect(p[None,:],R,[theta])
        g.place(p,theta);quat=np.zeros(4);mujoco.mju_mat2Quat(quat,np.ascontiguousarray(R).reshape(-1))
        g.d.qpos[3:7]=quat;mujoco.mj_kinematics(g.m,g.d)
        lo,hi=g.bounds();ds=g.distances(lo,hi)
        for z,sel in enumerate([g.cat=='body',g.cat!='body']):
            broad=ds.copy();broad[~sel]=np.inf
            upper=math.inf;lower=math.inf
            for pair in np.argsort(broad,axis=None):
                lb=float(broad.flat[pair])
                if lb>=upper:break
                i,k=np.unravel_index(pair,broad.shape);l,u,_,_=s.pair(int(g.ids[i]),k,lb)
                lower=min(lower,l);upper=min(upper,u)
            predicted=float(prediction[z][0])
            assert predicted<=upper+1e-8, (j,z,predicted,lower,upper)
            smallest=min(smallest,min(lower,upper)-predicted)
            largest=max(largest,upper-predicted);count+=1
    return dict(poses=36,group_checks=count,minimum_lower_bound_difference_m=smallest,
                max_conservatism_m=largest,all_predicted_bounds_below_convex_upper=True)


def feedback_faults():
    g=GateEnvelope(Geometry(make_plant()),P['controller']['angle_pad_per_degree_m'])
    rt=Route('nominal');pr=RollingGateTiming(g,P)
    ref=rt.preview(20.,pr.times);xyz=ref[0]
    obs=(xyz,np.zeros(3),[0.,0.,np.pi/2],0.,.45,.02)
    assert pr.update(20.,None,ref)['reason']=='missing_feedback'
    assert pr.update(20.,obs,ref)['reason']=='stale_feedback' and pr.bend_start is None
    # Impossible future elevation intersects a physical lintel while the starting pose remains healthy.
    ref[12:,0]=1.;ref[12:,2]=2.50
    d=pr.update(20.,(xyz,np.zeros(3),[0.,0.,np.pi/2],0.,.02,.02),ref)
    assert pr.bend_start is None and not any(r['geometry_ok'] for r in d['candidates'])
    dt=.001;grid=np.arange(20.,32.,dt);phase=np.array([phase_time(v,'pause') for v in grid])
    rate=np.gradient(phase,dt)
    assert rate.min()>-1e-7 and rate.max()<1.000001 and abs(phase_time(52.,'pause')-46.)<1e-9
    return dict(missing_feedback_no_commit=True,stale_feedback_no_commit=True,
                blocked_lintel_no_commit=True,max_phase_rate=float(rate.max()),net_route_delay_s=6.)


def histories():
    data=json.loads((DEST/'all-audit.json').read_text())['runs']
    table=pd.read_csv(DEST/'all-metrics.csv').set_index('case')
    rv=[]
    for seed in P['seed_order']:
        for l,r in [('phase','phasebend'),('onlinebend','online')]:
            ids=[seed+'-'+i+'_pause' for i in (l,r)]
            streams=[]
            for key in ids:
                folder=DEST/'runs'/key;w=pd.read_csv(folder/'hil.csv')
                zero=json.loads((folder/'run.json').read_text())['release_us']
                tt=(w.us.to_numpy()-zero)/1e6
                streams.append((tt,w.request_deg.to_numpy()))
            earliest=float(min(table.loc[ids,'return_t']))
            prefix=[u[(t>=0)&(t<earliest)] for t,u in streams]
            assert len(prefix[0])==len(prefix[1])
            diff=float(abs(prefix[0]-prefix[1]).max())
            # Events are quantized to the first valid 5 Hz estimator tick, with the same release phase.
            # Small differences in a triggered onset are saved, not erased by time-warping the signals.
            critical=[float(data[k]['convex']['arm_tool']['t']) for k in ids]
            rv.append(dict(seed=int(seed),outward='phase' if l=='phase' else 'online',
                first_return_s=earliest,critical_s=critical,both_minima_before_either_return=max(critical)<earliest,
                max_pre_return_command_difference_deg=diff))
    return rv


if __name__=='__main__':
    out=dict(geometry=bounds_test(),faults=feedback_faults(),prefix=histories())
    save(DEST/'engineering-checks.json',out)
    print(json.dumps(out,indent=2))
