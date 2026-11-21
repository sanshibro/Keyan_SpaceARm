"""Nonblocking M2 trajectory buffer and one asynchronous local planner worker."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict,is_dataclass
from pathlib import Path
import json,time
import numpy as np
from m2_geometry import Geometry,RangeMap
from m2_planner import Kernel,Settings,METHODS,FIXED_THETA,FIXED_PHI
from m2_handoff import qualify_timed,sample_bridge,boundary_record
from m2_stop_clock import commit_clock

def clean(x):
    if is_dataclass(x):return clean(asdict(x))
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,(np.integer,)):return int(x)
    if isinstance(x,(np.floating,float)):return float(x) if np.isfinite(x) else None
    return x

class RollingPlanner:
    def __init__(self,plant,method,goal_hint,config=None):
        self.method=method;self.config=config or Settings();self.sensor=RangeMap(plant,self.config.radius)
        self.kernel=Kernel(plant,method,goal_hint,self.config)
        self.geometry=Geometry(plant,self.kernel.geometry.limits) # controller thread owns this instance
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='M2-local-planner')
        self.future=None;self.pending=None;self.active=None;self.events=[];self.windows={};self.next_id=0
        self.last_request=-np.inf;self.last_x=-np.inf;self.last_state=None;self.last_rate=np.zeros(6);self.last_accel=np.zeros(6)
        self.rejections=0;self.closed=False;self.status='waiting';self.done=False;self.needs_stop=True;self.request=None
        self.requests={}
        self.last_now=0.;self.event_path=None
    def event(self,kind,now,**fields):
        item=dict(event=kind,release_s=float(now),wall_monotonic_s=time.monotonic(),method=self.method,**fields)
        self.events.append(clean(item))
        if self.event_path is not None:
            with self.event_path.open('a',encoding='utf8') as log:log.write(json.dumps(self.events[-1])+'\n')
        return item
    def _active_sample(self,now):
        a=self.active
        if a is None:return self.last_state.copy(),np.zeros(6),np.zeros(6)
        local=max(0.,now-a['activated']);bridge=a.get('bridge');duration=bridge.duration if bridge else 0.
        if bridge is not None and local<duration:return sample_bridge(bridge,local)
        curve=a['curve'];offset=a.get('trajectory_offset_s',0.);t=np.clip(local-duration+offset,0.,curve.duration)
        if a.get('stop_clock') is not None:return a['stop_clock'].sample(curve,local-duration+offset)
        if local-duration+offset>=curve.duration:return curve.evaluate([curve.duration],0)[0],np.zeros(6),np.zeros(6)
        return tuple(curve.evaluate([t],d)[0] for d in range(3))
    def _job(self,request,snapshot,start,rate,accel,old):
        # Immutable copies of request state and snapshot are the entire causal
        # worker input. No live plant state or future map is queried here.
        request['worker_started_s']=time.monotonic()
        if self.event_path is not None:
            self.kernel.trace_path=self.event_path.with_name('planner-live.jsonl')
        try:
            result=self.kernel.solve(snapshot,start,rate,accel,request['id'])
            result['request']=request;result['bridge']=None;result['handoff']=None;result['trajectory_offset_s']=0.
            if self.method!='hard':
                offset=max(0.,request['scheduled_switch_s']-request['submit_release_s']) if request['previous_id']>=0 else 0.
                t=time.monotonic();bridge,proof,record,offset=qualify_timed(self.kernel,old,result['curve'],offset,request['bridge_duration'])
                result['timing']['handoff_s']=time.monotonic()-t
                result['timing']['total_with_handoff_s']=time.monotonic()-result['wall_started']
                result['handoff']=dict(record,proof=clean(proof));result['bridge']=bridge;result['trajectory_offset_s']=offset
                if not proof.feasible:raise RuntimeError('Receiving-map handoff rejected: '+proof.failure_reason)
            else:
                result['handoff']=dict(kind='hard_switch',duration=0.,phase_rule='candidate absolute time is submission time; apply its available suffix at actual readiness')
            if self.method=='stop' and not result['final']:
                result['stop_clock']=commit_clock(result['curve'],self.config.replan_distance)
                clock=result['stop_clock']
                if clock is not None:result['handoff']['commit_brake']=asdict(clock)
            result['wall_completed']=time.monotonic();return result
        finally:
            request['worker_finished_s']=time.monotonic()
    def _submit(self,now,state,rate,accel):
        wall_before_reveal=time.monotonic()
        snap=self.sensor.reveal(now,state);self.geometry.set_snapshot(snap)
        live=self.active is not None and self.method!='stop';bridge_duration=.5
        switch_time=float(np.ceil((now+1.25-1e-9)/.02)*.02) if live else max(21.,now)
        old=self._active_sample(switch_time) if live else (state.copy(),np.zeros(6),np.zeros(6))
        start,v,a=state.copy(),rate.copy(),accel.copy()
        if live:
            # The state origin is the present measured whole-body pose. The
            # moving spline starts with the tangent already issued by control;
            # differencing a nearly straight arm can give singular polar rates.
            _,v,a=self._active_sample(now)
        else:v=np.zeros(6);a=np.zeros(6)
        # Moving boundary is predicted from the already committed curve, never
        # by peeking at future physical states. Current measured state is logged.
        request=dict(id=self.next_id,submit_release_s=float(now),wall_before_reveal_s=wall_before_reveal,wall_submit_s=time.monotonic(),scheduled_switch_s=float(switch_time),
                     previous_id=self.active['id'] if self.active else -1,current=boundary_record(state,rate,accel),
                     seed_boundary=boundary_record(start,v,a),committed_boundary=boundary_record(*old),
                     map=snap.metadata(),bridge_duration=bridge_duration)
        self.request=request;self.next_id+=1;self.last_request=now;self.last_x=state[0]
        self.requests[request['id']]=dict(request=request,snapshot=snap)
        if self.event_path is not None:
            evidence=self.event_path.parent/'rolling'/'windows'/('%03d'%request['id'])
            evidence.mkdir(parents=True,exist_ok=True)
            np.savez(evidence/'snapshot.npz',obstacle_rows=snap.obstacle_rows,ids=np.asarray(snap.ids),
                     origin=snap.origin,radius=snap.radius)
            (evidence/'request.json').write_text(json.dumps(clean(request)))
        self.event('submit',now,**request);self.future=self.pool.submit(self._job,request,snap,start.copy(),v.copy(),a.copy(),old)
        self.needs_stop=self.active is None or self.method=='stop'
    def tick(self,now,state6,rate6,accel6):
        if self.closed:return
        self.last_now=float(now)
        state=np.asarray(state6,float).copy();rate=np.asarray(rate6,float).copy();accel=np.asarray(accel6,float).copy()
        self.last_state=state;self.last_rate=rate;self.last_accel=accel
        if self.future is not None and self.future.done():
            try:
                result=self.future.result();self.windows[result['id']]=result
                if self.method=='hard' and self.active is not None:
                    offset=max(0.,now-result['request']['submit_release_s'])
                    if offset>=result['curve'].duration:raise RuntimeError('Hard-switch candidate already expired at actual readiness')
                    result['trajectory_offset_s']=offset
                    result['handoff'].update(trajectory_offset_s=offset,before=boundary_record(*self._active_sample(now)),
                        after=boundary_record(*(result['curve'].evaluate([offset],d)[0] for d in range(3))))
                self.event('complete',now,id=result['id'],wall_started_s=result['wall_started'],wall_completed_s=result['wall_completed'],
                           timing=result['timing'],map_hash=result['snapshot'].map_hash)
                if self.active is not None and self.method not in ('hard','stop') and now>result['request']['scheduled_switch_s']+1e-6:
                    self.event('reject',now,id=result['id'],reason='missed_causal_handoff_deadline');self.rejections+=1
                else:self.pending=result
            except Exception as exc:
                self.requests[self.request['id']].update(failure=dict(reason=type(exc).__name__+': '+str(exc),
                    wall_completed_s=time.monotonic(),release_completed_s=float(now),timing=getattr(exc,'timing',None),
                    proof=clean(getattr(exc,'proof',None)),layers=getattr(exc,'layers',None)))
                self.event('reject',now,id=self.request['id'],reason=type(exc).__name__+': '+str(exc),
                           timing=getattr(exc,'timing',None),proof=clean(getattr(exc,'proof',None)))
                self.rejections+=1
            self.future=None
        if self.pending is not None:
            target=max(21.,self.pending['request']['scheduled_switch_s']) if self.method not in ('hard','stop') else 21.
            if now>=target:
                if self.active is not None and self.method not in ('hard','stop') and abs(now-target)>1e-6:
                    self.event('reject',now,id=self.pending['id'],reason='missed_exact_control_grid_handoff',scheduled_switch_s=target)
                    self.pending=None;self.rejections+=1;return
                result=self.pending;self.pending=None;result['activated']=float(now);self.active=result;self.status='running';self.needs_stop=False
                self.event('activate',now,id=result['id'],source_id=result['request']['previous_id'],map_hash=result['snapshot'].map_hash,
                           handoff=result['handoff'],trajectory_offset_s=result['trajectory_offset_s'],bridge_start_s=float(now),bridge_end_s=float(now+(result['bridge'].duration if result['bridge'] else 0.)))
        elapsed=now-self.active['activated'] if self.active else 0.
        duration=(self.active['curve'].duration-self.active.get('trajectory_offset_s',0.)+(self.active['bridge'].duration if self.active['bridge'] else 0.)) if self.active else 0.
        if self.active and self.active.get('stop_clock') is not None:
            duration+=self.active['stop_clock'].duration-self.active['curve'].duration
        ended=bool(self.active is not None and elapsed>=duration)
        if ended and self.active['final']:
            if not self.done:self.event('plan_done',now,id=self.active['id'])
            self.done=True;self.status='done';return
        if self.future is None and self.pending is None and not self.done and not (self.active and self.active['final']):
            if self.active is None:
                ready=now-self.last_request>1.0
            elif self.method=='stop':ready=ended and now-self.last_request>1.0
            else:ready=(state[0]-self.last_x>=self.config.replan_distance or elapsed>=duration-2.1) and now-self.last_request>=self.config.replan_min_interval
            if ready:self._submit(now,state,rate,accel)
        if ended:self.status='waiting' if self.future is not None else 'stopped';self.needs_stop=True
        elif self.active is None:self.status='waiting' if self.future is not None else 'rejected'
    def sample(self,now):
        if self.last_state is None:raise RuntimeError('tick must provide the first causal state before sample')
        q,v,a=self._active_sample(now)
        return dict(q=q,dq=v,ddq=a,active_id=self.active['id'] if self.active else -1,status=self.status,needs_stop=self.needs_stop)
    def close(self):
        if self.closed:return
        self.closed=True
        if self.future is not None:self.future.cancel()
        self.pool.shutdown(wait=True) # Python 3.8 has no cancel_futures argument.
        if self.future is not None and not self.future.cancelled():
            try:
                result=self.future.result();self.windows[result['id']]=result
                self.event('complete_after_stop',self.last_now,id=result['id'],wall_started_s=result['wall_started'],wall_completed_s=result['wall_completed'],timing=result['timing'])
            except Exception as exc:
                self.requests[self.request['id']]['failure']=dict(reason=type(exc).__name__+': '+str(exc),timing=getattr(exc,'timing',None),
                    worker_started_s=self.request.get('worker_started_s'),wall_completed_s=time.monotonic(),proof=clean(getattr(exc,'proof',None)),layers=getattr(exc,'layers',None))
    def export(self,folder):
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        (folder/'events.jsonl').write_text(''.join(json.dumps(clean(x),ensure_ascii=False)+'\n' for x in self.events))
        (folder/'settings.json').write_text(json.dumps(clean(self.config),indent=2))
        for ident,entry in self.requests.items():
            out=folder/'windows'/('%03d'%ident);out.mkdir(parents=True,exist_ok=True);snap=entry['snapshot'];result=self.windows.get(ident)
            np.savez_compressed(out/'snapshot.npz',obstacle_rows=snap.obstacle_rows,ids=np.asarray(snap.ids),origin=snap.origin,radius=snap.radius)
            for name,obj in [('snapshot',snap.metadata()),('request',entry['request']),('failure',entry.get('failure'))]:
                (out/(name+'.json')).write_text(json.dumps(clean(obj),indent=2,ensure_ascii=False))
            if result is None:continue
            curve=result['curve']
            for name,obj in [('snapshot',snap.metadata()),('request',result['request']),('timing',result['timing']),('proof',result['proof']),('handoff',result['handoff']),('seed_layers',result['layers'])]:
                (out/(name+'.json')).write_text(json.dumps(clean(obj),indent=2,ensure_ascii=False))
            t=np.linspace(0,curve.duration,int(np.ceil(curve.duration/.02))+1)
            np.savez_compressed(out/'trajectory.npz',t=t,q=curve.evaluate(t,0),dq=curve.evaluate(t,1),ddq=curve.evaluate(t,2),
                                control_states=curve.control_states,span_durations=curve.span_durations,seed=result['route'],
                                final=result['final'],stationary_start=curve.stationary_start,stationary_end=curve.stationary_end,
                                boundary_ramp_duration=curve.boundary_ramp_duration,trajectory_offset_s=result.get('trajectory_offset_s',0.),
                                certified_braking_tail_s=curve.finish_ramp_duration)
            if result['bridge'] is not None:
                bridge=result['bridge'];bt=np.linspace(0,bridge.duration,int(np.ceil(bridge.duration/.01))+1);rows=[sample_bridge(bridge,x) for x in bt]
                np.savez_compressed(out/'bridge.npz',t=bt,q=np.array([x[0] for x in rows]),dq=np.array([x[1] for x in rows]),ddq=np.array([x[2] for x in rows]),coefficients=bridge.c,duration=bridge.duration)
