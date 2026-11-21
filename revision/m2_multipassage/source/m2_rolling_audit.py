"""Reconstruct rolling visibility, physical time and switching evidence from a native flight."""
import hashlib
import json
from pathlib import Path
import numpy as np
from super_handoff import coordinates,rates,accelerations
from uav_continuum_whole_body_v2.execution import evaluate_quintic_bridge
from uav_continuum_whole_body_v2.trajectory import SplineTrajectory

def load(path):
    return json.loads(path.read_text(encoding='utf8'))

def check(value,message):
    if not value:raise ValueError(message)

def cart(q,v,a):
    return np.stack((coordinates(q),rates(q,v),accelerations(q,v,a)))

def close_endpoint(a,b):
    delta=a-b
    delta[0,3]=(delta[0,3]+np.pi)%(2*np.pi)-np.pi
    return np.max(abs(delta))<1.e-6

def contiguous_time(t,mask,min_span):
    starts=np.flatnonzero(mask&~np.r_[False,mask[:-1]])
    ends=np.flatnonzero(mask&~np.r_[mask[1:],False])
    return float(sum(t[j]-t[i] for i,j in zip(starts,ends) if t[j]-t[i]>=min_span))

class RecordedWindow:
    def __init__(self,path):
        with np.load(path/'trajectory.npz',allow_pickle=False) as z:
            states=z['control_states'];dt=z['span_durations']
            self.offset=float(z['trajectory_offset_s']) if 'trajectory_offset_s' in z else 0.
            check(10<=len(states)<=12 and states.shape[1]==6 and (dt>0).all(),'spline topology')
            self.curve=SplineTrajectory(states,dt,stationary_start=bool(z['stationary_start']),stationary_end=bool(z['stationary_end']),
                                        boundary_ramp_duration=float(z['boundary_ramp_duration']) if 'boundary_ramp_duration' in z else .65)
            for i,name in enumerate(('q','dq','ddq')):
                check(np.allclose(self.curve.evaluate(z['t'],i),z[name],rtol=1e-7,atol=1.e-8),'reconstructed spline '+name)
        self.duration=0.;self.coef=None;self.handoff=load(path/'handoff.json')
        if self.handoff['kind']=='C2_quintic':
            with np.load(path/'bridge.npz') as z:
                self.coef=z['coefficients'].copy();self.duration=float(z['duration'])
            for name,t in [('before',0.),('after',self.duration)]:
                d=self.handoff[name]
                check(close_endpoint(self.bridge(t),cart(*(np.asarray(d[k]) for k in ('q','dq','ddq')))),'C2 boundary '+name)
        if self.handoff['kind']!='hard_switch':
            check(self.handoff['proof']['feasible'],'missing swept handoff geometry')
        self.activation=None
    def bridge(self,t):
        return np.array([evaluate_quintic_bridge(self.coef,[t],i)[0] for i in range(3)])
    def state(self,release):
        local=max(0.,release-self.activation)
        if local<self.duration:return self.bridge(local)
        local=local-self.duration+self.offset
        if local>=self.curve.duration:
            q=self.curve.evaluate([self.curve.duration],0)[0]
            return cart(q,np.zeros(6),np.zeros(6))
        return cart(*(self.curve.evaluate([local],i)[0] for i in range(3)))

def audit_rolling(f,c,t,xy,physical):
    folder=Path(f)/'rolling';events=[json.loads(line) for line in (folder/'events.jsonl').read_text(encoding='utf8').splitlines()]
    settings=load(folder/'settings.json');radius=float(settings['radius'])
    check(abs(radius-2.4)<1e-8,'unexpected view radius')
    req={};complete={};accepted={};rejections={};times=[];wall_latencies=[];switches=[];has_async=[]
    for e in events:
        typ=e['event'];i=e.get('id');stamp=e['release_s']
        if typ=='submit':
            check(i not in req,'duplicate planning request')
            req[i]=e;meta=e['map'];where=folder/'windows'/('%03d'%i)
            n=int(np.argmin(abs(t-stamp)))
            check(abs(t[n]-stamp)<1.e-6 and np.allclose(meta['origin'],c['px4_xyz'][n],atol=2e-7,rtol=0),'request must originate at its PX4 observation')
            check(meta['stamp']==stamp and meta['known_frontier']<=meta['origin'][0]+radius+1e-8,'causal map extent')
            with np.load(where/'snapshot.npz',allow_pickle=False) as snapshot:
                rows=np.ascontiguousarray(snapshot['obstacle_rows'],dtype=np.float64)
                origin=np.asarray(snapshot['origin'],dtype=np.float64)
                check(np.allclose(origin,meta['origin']) and len(rows)==meta['row_count'] and list(snapshot['ids'])==meta['ids'],'saved map mismatch')
                if len(rows):
                    furthest=np.linalg.norm(np.abs(rows[:,:3]-origin)+rows[:,3:6],axis=1)
                    check(np.max(furthest)<=radius+2e-8,'unseen geometry in local map')
                digest=hashlib.sha256(rows.tobytes()+origin.tobytes()+str(meta['version']).encode()).hexdigest()
                check(digest==meta['map_hash'] and load(where/'snapshot.json')['map_hash']==digest,'local map fingerprint')
        elif typ=='complete':
            check(i in req and i not in complete,'worker result without unique request')
            check(e['map_hash']==req[i]['map']['map_hash'],'worker returned another snapshot')
            elapsed=float(e['wall_completed_s']-e['wall_started_s'])
            check(elapsed>=0 and e['wall_started_s']>=req[i]['wall_submit_s']-1e-6,'worker wall clock')
            times.append(elapsed);wall_latencies.append(e['wall_completed_s']-req[i].get('wall_before_reveal_s',req[i]['wall_submit_s']))
            complete[i]=e
        elif typ=='reject':
            rejections[i]=e['reason']
            if i not in complete:
                archive=folder/'windows'/('%03d'%i)
                info=load(archive/'failure.json') or {}
                timing=e.get('timing') or {}
                submitted=load(archive/'request.json')
                start_wall=submitted.get('worker_started_s',req[i]['wall_submit_s'])
                end_wall=info.get('wall_completed_s',e['wall_monotonic_s'])
                wall_latencies.append(end_wall-req[i].get('wall_before_reveal_s',req[i]['wall_submit_s']))
                # A rejected request consumed a slot too. Include all worker
                # stages and use its recorded finish whenever available.
                val=submitted.get('worker_finished_s',end_wall)-start_wall
                times.append(float(max(val,timing.get('total_with_handoff_s',timing.get('total_s',0.)))))
        elif typ=='activate':
            check(i in complete and i not in accepted,'activated without successful solve')
            path=folder/'windows'/('%03d'%i); proof=load(path/'proof.json');g=proof['metrics']['geometry']
            check(proof['feasible'] and g['accepted'] and g['map_hash']==req[i]['map']['map_hash'],'receiving window geometric proof')
            check(g['body_min']>=.085-1e-9 and g['arm_min']>=.085-1e-9 and g['self_min']>=.015-1e-9,'nominal margins')
            w=RecordedWindow(path);w.activation=stamp;accepted[i]=w
            check(abs(w.offset-e.get('trajectory_offset_s',0.))<1.e-8,'receiving trajectory clock mismatch')
            before=e['source_id']
            if before>=0:
                check(before in accepted,'missing active source')
                old=accepted[before].state(stamp);new=w.state(stamp);jump=new-old
                if w.handoff['kind']=='C2_quintic':
                    check(close_endpoint(old,new),'actual source and bridge start disagree')
                sel=(t>=req[i]['submit_release_s']-1e-8)&(t<complete[i]['release_s']-1e-8)
                has_async.append(bool(np.any(sel&(c['active_window']==before)&(c['window_status']=='running'))))
                switches.append(dict(window=int(i),source=int(before),release_s=stamp,kind=w.handoff['kind'],bridge_s=w.duration,
                                     acceleration_jump_m_s2=float(np.linalg.norm(jump[2,:3])),bend_rate_jump_rad_s=float(np.linalg.norm(jump[1,4:]))))
    phase=c['phase'];start=load(Path(f)/'control-summary.json')['motion_start_s']
    active=(t>=start)&(phase!='inspect')
    pause=(phase=='rolling_hold')|(phase=='safe_prefix_end')
    planner_pause=float(np.sum(np.diff(t)*active[:-1]*pause[:-1]))
    # Measure sustained lack of progress while translating or bending is commanded.
    window=max(1,int(round(.5/np.median(np.diff(t)))));actual_motion=np.ones(len(t),dtype=bool)
    actual_b=np.column_stack([np.linalg.norm(xy[window:,:3]-xy[:-window,:3],axis=1),np.linalg.norm(c['est_b'][window:]-c['est_b'][:-window],axis=1)])
    actual_motion[window:]=(actual_b[:,0]>.004)|(actual_b[:,1]>np.deg2rad(1.))
    demand=(np.linalg.norm(c['plan_dq'][:,:3],axis=1)>.04)|(np.abs(c['plan_dq'][:,4])>np.deg2rad(4.))
    stall=contiguous_time(t,active&((phase=='feedback_hold')|((~actual_motion)&demand)),.8)
    return dict(solve_wall_s=times,request_wall_latency_s=wall_latencies,submitted_windows=len(req),activated_replans=len(switches),
                rolling_verified=bool(accepted),causal_map_verified=bool(req),async_coverage=bool(has_async and all(has_async)),
                switches=switches,planning_diagnosis=list(dict.fromkeys(rejections.values())),planner_pause_s=planner_pause,abnormal_stall_s=stall,
                handoff_accel_jump_m_s2=max((s['acceleration_jump_m_s2'] for s in switches),default=0.),
                handoff_brate_jump_rad_s=max((s['bend_rate_jump_rad_s'] for s in switches),default=0.))
