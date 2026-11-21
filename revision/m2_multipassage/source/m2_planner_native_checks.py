"""Independent static M2 geometry/live-model checks. No integrator or PX4."""
import json,time
from pathlib import Path
import mujoco,numpy as np
from gate_distance_audit import DistanceAudit
from gate_distance_certificate import CertifiedDistance
from m2_geometry import Geometry,RangeMap,NativeSelf
from m2_plant import build,task_pose,place,STAGE
from m2_scene import HOVER
from m2_self_registry import partition

def run(out):
    p=build();m=p.m;g=Geometry(p);sensor=RangeMap(p)
    audit=DistanceAudit(m,tolerance_m=1e-5,strict=True,obstacle_prefix='COL_')
    audit.__class__=CertifiedDistance;audit.records=[]
    hazard,fit=partition(m,audit.self_pairs)
    registered=json.loads((STAGE/'qa/static/certified.json').read_text())['gate_witnesses']
    task=task_pose(p)
    cases=[dict(name=r['gate'],xyz=r['xyz'],theta=r['theta_deg'],phi=r['phi_deg']) for r in registered]
    cases += [dict(name='hover',xyz=HOVER.tolist(),theta=0,phi=180),dict(name='flange',xyz=task['body'],theta=task['theta_deg'],phi=task['phi_deg'])]
    results=[]
    for c in cases:
        q=np.r_[c['xyz'],0,np.deg2rad(c['theta']),np.deg2rad(c['phi'])];g.set_snapshot(sensor.reveal(0,q))
        wall=time.monotonic();local=g.state(q,True);native_wall=time.monotonic()-wall
        place(m,audit.d,c['xyz'],c['theta'],c['phi']);lo,hi=audit.bounds()
        exact={key:audit._scan(key,pairs,lo,hi) for key,pairs in [('body',audit.external['body']),('arm',audit.external['arm_tool']),('hazard',hazard),('fit',fit)]}
        for key,pair in [('body','body'),('arm','arm')]:
            if local[key]>exact[pair]['upper']+2e-5:raise AssertionError('Optimistic environmental certificate '+c['name'])
        if local['self_contact']['hazardous']>exact['hazard']['upper']+2e-5:raise AssertionError('Optimistic self certificate')
        results.append(dict(**c,native_s=native_wall,planning=local,exact={k:dict(lower=v['lower'],upper=v['upper'],pair=[v['name_a'],v['name_b']]) for k,v in exact.items()}))
    rng=np.random.default_rng(26112189);timings=[];differences=[]
    for theta,phi in rng.uniform([.05,-np.pi],[1.65,np.pi],size=(24,2)):
        wall=time.monotonic();r=g.self_report(theta,phi);timings.append(time.monotonic()-wall)
        place(m,audit.d,[0,0,0],np.rad2deg(theta),np.rad2deg(phi));lo,hi=audit.bounds();exact=audit._scan('random_hazard',hazard,lo,hi)
        if r['hazardous']>exact['upper']+2e-5:raise AssertionError('Optimistic native support bound')
        if exact['lower']>0:differences.append(abs(r['hazardous']-exact['lower']))
    report=dict(scene_hash=p.scene_digest,model_source='live m2_plant.build',flight_executed=False,cases=results,
                native_self_median_s=float(np.median(timings)),native_self_max_s=max(timings),
                max_positive_lower_difference_m=max(differences),hazard_pairs=len(hazard),fit_pairs=len(fit),random_seed=26112189)
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2));return report

if __name__=='__main__':
    r=run(STAGE/'qa/planner/native-certificates.json')
    print(json.dumps({k:v for k,v in r.items() if k!='cases'},indent=2))
    for c in r['cases']:print(c['name'],c['planning']['reason'],c['exact'])
