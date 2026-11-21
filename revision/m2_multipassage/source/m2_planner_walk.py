"""Causal command-space window exercise: no physics integration, aircraft or flight.

Each next observation is taken only on the preceding accepted local spline.
This checks planner reachability/timing; it is not a tracking experiment.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from m2_plant import build,task_pose,STAGE
from m2_scene import HOVER
from m2_geometry import RangeMap
from m2_planner import Kernel
from m2_rolling import clean

def run(method='coupled',count=7,resume=False):
    p=build();sensor=RangeMap(p);goal=task_pose(p)
    try:
        from m2_inspection_ik import goal_family
        goal['candidates']=goal_family(p,goal)
    except ImportError:pass
    before=time.monotonic();kernel=Kernel(p,method,goal);prep=time.monotonic()-before
    q=np.r_[HOVER,0.,0.,np.pi];v=np.zeros(6);a=np.zeros(6);release=21.;records=[]
    folder=STAGE/'qa/planner'/('walk-'+method);folder.mkdir(parents=True,exist_ok=True)
    if resume:
        old=json.loads((folder/'summary.json').read_text());failed=old['windows'][-1];q=np.asarray(failed['state']);release=failed['map']['stamp']
        previous=old['windows'][-2];f=np.load(folder/('window-%02d.npz'%previous['window']))
        index=int(np.argmin(np.linalg.norm(f['q'][:,:3]-q[:3],axis=1)));v=f['dq'][index];a=f['ddq'][index]
        from uav_continuum_whole_body_v2.trajectory import SplineTrajectory
        kernel.last_curve=SplineTrajectory(f['control_states'],f['span_durations'],stationary_start=bool(f['stationary_start']) if 'stationary_start' in f else False,
             stationary_end=bool(f['stationary_end']) if 'stationary_end' in f else False,boundary_ramp_duration=float(f['boundary_ramp_duration']) if 'boundary_ramp_duration' in f else .65)
        kernel.last_opt=np.array(previous['timing']['optimizer']['best'])
        folder=STAGE/'qa/planner'/('walk-resume-'+method);folder.mkdir(parents=True,exist_ok=True)
    for k in range(count):
        snap=sensor.reveal(release,q);start=time.monotonic()
        try:
            result=kernel.solve(snap,q,v,a,k);curve=result['curve']
            record=dict(window=k,state=q.tolist(),map=snap.metadata(),accepted=True,timing=result['timing'],duration_s=curve.duration,
                        end=curve.evaluate([curve.duration],0)[0].tolist(),proof=clean(result['proof']),layers=clean(result['layers']))
            ts=np.linspace(0,curve.duration,2001);states=curve.evaluate(ts,0)
            advance=np.flatnonzero(states[:,0]>=q[0]+.78)
            if not len(advance):cut=curve.duration
            else:cut=float(ts[advance[0]])
            np.savez_compressed(folder/('window-%02d.npz'%k),t=ts,q=states,dq=curve.evaluate(ts,1),ddq=curve.evaluate(ts,2),
                                control_states=curve.control_states,span_durations=curve.span_durations,seed=result['route'],
                                stationary_start=curve.stationary_start,stationary_end=curve.stationary_end,boundary_ramp_duration=curve.boundary_ramp_duration)
            if result['final']:records.append(record);break
            nextq=curve.evaluate([cut],0)[0];v=curve.evaluate([cut],1)[0];a=curve.evaluate([cut],2)[0]
            if cut>=curve.duration-1e-6:v[:]=0;a[:]=0
            release+=cut
            if nextq[0]<=q[0]+.01:record['zero_progress']=True;records.append(record);break
            q=nextq
        except Exception as exc:
            record=dict(window=k,state=q.tolist(),map=snap.metadata(),accepted=False,error=str(exc),wall_s=time.monotonic()-start,
                        timing=getattr(exc,'timing',None),proof=clean(getattr(exc,'proof',None)),layers=getattr(exc,'layers',None))
            records.append(record);print('REJECT',k,record['error'],record['timing'],flush=True);break
        records.append(record);print('WINDOW',k,record['state'][:3],'->',record['end'][:3],record['duration_s'],record['timing'],flush=True)
    result=dict(kind='static command-space exercise only',flight_executed=False,scene_hash=p.scene_digest,method=method,preparation_s=prep,windows=records)
    (folder/'summary.json').write_text(json.dumps(clean(result),indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');p.add_argument('--count',type=int,default=7);args=p.parse_args()
    run(count=args.count,resume=args.resume)
