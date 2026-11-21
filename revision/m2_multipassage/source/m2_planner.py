"""Small causal 6D beam, warm L-BFGS-B and independent dense certification.

No gate labels or global obstacle coordinates enter this module. The same
coupled kernel is used by coupled/stop/hard. Ablations alter only declared DOFs
or the four registered regularizers.
"""
from dataclasses import dataclass
from types import SimpleNamespace
import json,math,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from m2_geometry import Geometry,wrap
from m2_body_batch import BodyBatch
from m2_lattice import LocalLattice
from scipy.interpolate import BSpline
from m2_spline import VectorSpline as SplineTrajectory,moving_boundary
from uav_continuum_whole_body_v2.types import ValidationResult

FIXED_THETA=np.deg2rad(80.);FIXED_PHI=np.pi
METHODS=('straight','fixed','sequential','unregularized','coupled','stop','hard')

@dataclass
class Settings:
    radius:float=2.4
    window:float=2.2
    knots:int=12
    speed:float=.27
    beam:int=12
    solve_budget:float=.88
    optimize_budget:float=.14
    self_margin:float=.015
    body_margin:float=.085
    arm_margin:float=.085
    replan_distance:float=.78
    replan_min_interval:float=1.8
    bridge_durations:tuple=(.40,.65,.90)

def bend(q):return q[...,4,None]*np.stack((np.cos(q[...,5]),np.sin(q[...,5])),axis=-1)

class Validator:
    def __init__(self,geometry):self.geometry=geometry;self.cached={}
    def validate(self,curve):
        g=self.geometry
        if hasattr(curve,'source') and hasattr(curve,'offset'):
            # A suffix contains no new geometric or dynamic state. Reuse the
            # full-curve certificate only on the identical immutable map.
            result=self.validate(curve.source)
            if result.feasible:
                return ValidationResult(True,'accepted',dict(result.metrics,certified_suffix_offset_s=curve.offset))
            return result
        cachekey=(g.map_hash,curve.control_states.tobytes(),curve.span_durations.tobytes(),curve.stationary_start,curve.stationary_end)
        if cachekey in self.cached:return self.cached[cachekey]
        if not np.isfinite(curve.duration) or curve.duration<=0 or curve.duration>900.:
            return ValidationResult(False,'invalid_reference_clock',{'duration':float(curve.duration)},0,0.,'reference clock outside native proof domain')
        ts=np.linspace(0,curve.duration,max(3,int(np.ceil(curve.duration/.04))+1))
        q=curve.evaluate(ts,0);v=curve.evaluate(ts,1);a=curve.evaluate(ts,2);j=curve.evaluate(ts,3)
        values=dict(speed=np.linalg.norm(v[:,:3],axis=1),acceleration=np.linalg.norm(a[:,:3],axis=1),jerk=np.linalg.norm(j[:,:3],axis=1),
          yaw_rate=abs(v[:,3]),theta_rate=abs(v[:,4]),phi_rate=np.where(q[:,4]>.03,abs(v[:,5]),0),
          yaw_acceleration=abs(a[:,3]),theta_acceleration=abs(a[:,4]),phi_acceleration=np.where(q[:,4]>.03,abs(a[:,5]),0))
        bounds=dict(speed=.60,acceleration=1.,jerk=3.,yaw_rate=np.deg2rad(45),theta_rate=np.deg2rad(50),phi_rate=np.deg2rad(60),
                    yaw_acceleration=np.deg2rad(90),theta_acceleration=np.deg2rad(100),phi_acceleration=np.deg2rad(120))
        metrics=dict(peaks={k:float(vv.max()) for k,vv in values.items()},limits=bounds)
        for key,array in values.items():
            fail=np.flatnonzero(array>bounds[key]+1e-8)
            if len(fail):
                i=int(fail[0]);return ValidationResult(False,'dynamics_rejected',metrics,i,float(ts[i]),key)
        proof=g.proof(curve);metrics['geometry']=proof
        if not proof['accepted']:
            w=proof['first'];return ValidationResult(False,'geometry_rejected',metrics,w['index'],w['time_s'],w['reason'])
        result=ValidationResult(True,'accepted',metrics);self.cached={cachekey:result};return result

class Kernel:
    def __init__(self,plant,method,goal,settings=None):
        if method not in METHODS:raise ValueError(method)
        self.settings=settings or Settings();self.method=method;self.goal=goal;self.coupled=method!='unregularized'
        if method in ('straight','fixed'):
            from m2_inspection_ik import fixed_family
            self.goal=dict(goal)
            self.goal['candidates'],self.goal['best_locked_axis_error_deg']=fixed_family(
                plant,goal,0. if method=='straight' else np.rad2deg(FIXED_THETA),np.rad2deg(FIXED_PHI),method=='fixed')
        lim=SimpleNamespace(body_margin=self.settings.body_margin,arm_margin=self.settings.arm_margin,tilt_max=0.)
        self.geometry=Geometry(plant,lim);self.geometry.plan_self_margin=self.settings.self_margin
        from m2_path_batch import PathBatch
        self.geometry.path_batch=PathBatch(self.geometry)
        self.fine=self.geometry;self.validator=Validator(self.geometry);self.snapshot=None
        self.roadmap=LocalLattice(self)
        self.last_curve=None
        self.last_opt=None;self.warm_points=None;self.warm_used=0
        self.trace_path=None
    def phase(self,name,**extra):
        self.current_phase=name
        if self.trace_path is None:return
        status=Path('/proc/self/status')
        memory={}
        if status.is_file():
            memory={s.split(':')[0]:int(s.split()[1]) for s in status.read_text().splitlines()
                    if s.startswith(('VmRSS:','VmHWM:','VmSwap:'))}
        record=dict(id=self.request_id,phase=name,wall=time.monotonic(),memory_kib=memory,**extra)
        with self.trace_path.open('a',encoding='utf8') as log:log.write(json.dumps(record)+'\n')
    def seed(self,start,end,final=False):
        return self.roadmap.search(start,end,final)
    def make_curve(self,route,rate,final,spans=None,stationary_start=True):
        K=self.settings.knots;delta=np.diff(route,axis=0);length=np.linalg.norm(delta[:,:3],axis=1)
        mag=np.minimum(1.,np.maximum(route[:-1,4],route[1:,4])/.12)
        demand=np.maximum.reduce([length/self.settings.speed,abs(delta[:,3])/.64,abs(delta[:,4])/.70,abs(delta[:,5])*mag/.86])
        if final:
            # The optical working pose has a short, constrained arrival. Budget
            # separate time for the approach, using the task frame and remaining
            # stopping distance in the plan itself (all switching methods).
            remaining=np.linalg.norm(route[:-1,:3]-route[-1,:3],axis=1)
            precision_speed=np.sqrt(.055**2+2.*.038*remaining)
            demand=np.maximum(demand,length/precision_speed)
        tau=np.r_[0.,np.cumsum(np.maximum(.04,demand))]
        if spans is None:
            knots=np.interp(np.linspace(0.,1.,K-2),.34*np.linspace(0.,1.,len(tau))+.66*tau/tau[-1],tau)
            spans=np.diff(knots)
        u=np.r_[np.zeros(4),np.cumsum(spans)[:-1],np.full(4,np.sum(spans))]
        tau=tau*(np.sum(spans)/tau[-1])
        B=BSpline(u,np.eye(K),3)(tau)
        greville=np.array([np.mean(u[j+1:j+4]) for j in range(K)])
        initial=np.column_stack([np.interp(greville,tau,route[:,j]) for j in range(6)])
        fixed=[0,K-1] if stationary_start else [0,1,K-1]
        if not stationary_start:initial[1]=initial[0]+np.asarray(rate)*spans[0]/3.
        free=np.setdiff1d(np.arange(K),fixed)
        # Fit the interior path by collocation. Ridge regularization suppresses
        # overshoot on underdetermined short prefixes while preserving endpoints.
        regular=.43
        mat=np.vstack((B[:,free],regular*np.eye(len(free))))
        rhs=np.vstack((route-B[:,fixed]@initial[fixed],regular*initial[free]))
        control=initial.copy();control[free]=np.linalg.lstsq(mat,rhs,rcond=None)[0]
        control[:,4]=np.clip(control[:,4],0.,self.geometry.theta_max)
        if not stationary_start:moving_boundary(control,spans,rate)
        return SplineTrajectory(control,np.asarray(spans),stationary_start=stationary_start,stationary_end=True,
                                boundary_ramp_duration=1.)
    def optimize(self,seed,rate,final,stationary_start):
        from m2_refine import refine
        return refine(self,seed,rate,final,stationary_start)
    def solve(self,snapshot,start,rate,accel,request_id=0,warm=None):
        started=time.monotonic();self.request_id=request_id;self.phase('snapshot');self.seed_diagnostics=[]
        try:return self._solve(snapshot,start,rate,accel,request_id,warm)
        except Exception as exc:
            timing=dict(getattr(exc,'timing',{}) or {})
            timing.update(wall_started_s=started,wall_completed_s=time.monotonic(),total_s=time.monotonic()-started,
                          failed_phase=self.current_phase)
            exc.timing=timing
            if not hasattr(exc,'layers'):exc.layers=self.seed_diagnostics
            raise
    def _solve(self,snapshot,start,rate,accel,request_id=0,warm=None):
        wall=time.monotonic();self.geometry.set_snapshot(snapshot);self.snapshot=snapshot
        first,_=self.geometry.path_batch.evaluate(np.asarray(start).reshape(1,6))
        min_start=np.array([self.settings.body_margin,self.settings.arm_margin,self.settings.self_margin,0.,-1.e-10])
        if np.any(first[0]<min_start):
            labels=('body_environment','arm_environment','self_collision','assembly_fit','unknown_space')
            reason=labels[int(np.flatnonzero(first[0]<min_start)[0])]
            p=ValidationResult(False,'initial_geometry',{'clearance_m':first[0].tolist()},0,0.,reason)
            raise PlanRejected(p,{},[])
        self.body_batch=BodyBatch(self.geometry)
        self.geometry.tight_search=False
        previous=warm if warm is not None else self.last_curve
        self.warm_points=previous.evaluate(np.linspace(0,previous.duration,160),0) if previous is not None else None
        self.warm_used=0
        qgoal=np.r_[self.goal['body'],np.deg2rad(self.goal.get('yaw_deg',0)),np.deg2rad(self.goal['theta_deg']),np.deg2rad(self.goal['phi_deg'])]
        end=start.copy();end[0]=min(qgoal[0],start[0]+self.settings.window,snapshot.known_frontier-.70)
        if end[0]<=start[0]+.20:raise RuntimeError('Insufficient observed whole-body horizon')
        final=end[0]>=qgoal[0]-1e-8
        if final:
            self.geometry.tight_search=True
            pool=self.goal.get('candidates',[self.goal]);ranked=[]
            for candidate in pool:
                q=np.r_[candidate['body'],np.deg2rad(candidate.get('yaw_deg',0)),np.deg2rad(candidate['theta_deg']),np.deg2rad(candidate['phi_deg'])]
                if self.method=='straight' and q[4]>.001:continue
                if self.method=='fixed' and (abs(q[4]-FIXED_THETA)>.001 or abs(wrap(q[5]-FIXED_PHI))>.001):continue
                nominal_yaw=np.deg2rad(self.goal.get('yaw_deg',0.))
                cost=(np.linalg.norm(q[:3]-start[:3])+.10*np.linalg.norm(bend(q)-bend(start))
                      +.39*abs(wrap(q[3]-nominal_yaw))+.10*abs(wrap(q[3]-start[3]))
                      +candidate.get('task_tool_offset_m',0.)+.5*candidate.get('task_camera_offset_m',0.))
                ranked.append((cost,q))
            if ranked:
                qs=np.array([q for _,q in ranked])
                gaps,_=self.geometry.path_batch.evaluate(qs,caps=(.120,.120,.045,.006))
                ranked=[(cost+32.*max(0.,.102-min(row[:2]))+9.*max(0.,.030-row[2]),q) for (cost,q),row in zip(ranked,gaps)
                        if row[0]>=self.settings.body_margin and row[1]>=self.settings.arm_margin and
                        row[2]>=self.settings.self_margin and row[3]>=0 and row[4]>=0]
            ranked.sort(key=lambda item:item[0])
            for _,candidate in ranked:
                if self.geometry.state(candidate,exact=True)['reason'] is None:
                    end=candidate.copy();break
            else:
                if self.method in ('straight','fixed') and end[0]-start[0]>.28:
                    final=False
                else:raise RuntimeError('No observed terminal task-family pose passes the registered margins and method DOFs')
        self.phase('seed');t=time.monotonic()
        approach=None
        if final:
            # Intercept the surveyed optical axis before advancing the TCP.
            # Current-map clearance of the whole terminal ray is checked here
            # and the fitted spline receives the usual independent swept proof.
            axis=np.asarray(self.goal['camera_axis'],dtype=float)
            length=min(.16,max(0.,(end[0]-start[0]-.25)/max(axis[0],.05)))
            if length>.035:
                candidate=end.copy();candidate[:3]-=length*axis
                ray=np.linspace(candidate,end,17)
                gaps,_=self.geometry.path_batch.evaluate(ray)
                if np.all(gaps>=min_start):approach=candidate
        route,layers,reached=self.seed(start,end if approach is None else approach,final)
        if reached and approach is not None:
            # Keep the unwrapped orientation chart chosen by the 6D search.
            terminal=route[-1].copy();terminal[:3]=end[:3]
            route=np.vstack((route,np.linspace(route[-1],terminal,5)[1:]))
        seedwall=time.monotonic()-t
        final=final and reached;stationary=np.linalg.norm(rate[:3])<.03
        self.phase('optimization',layers=len(layers))
        curve,base,opt=self.optimize(route,rate,final,stationary)
        self.phase('validation')
        t=time.monotonic()
        for candidate in (curve,base):
            curve=candidate;repair=1.
            for retry in range(6):
                self.phase('validation',retry=retry,duration_s=float(curve.duration))
                proof=self.validator.validate(curve)
                if proof.feasible or proof.status!='dynamics_rejected':break
                peaks=proof.metrics['peaks'];bounds=proof.metrics['limits']
                ratios=[(peaks[k]/bounds[k])**(1/(3 if k=='jerk' else 2 if 'acceleration' in k else 1)) for k in peaks]
                scale=max(1.,max(ratios)*1.025);repair*=scale
                controls=curve.control_states.copy();h=curve.span_durations*scale
                if not stationary:moving_boundary(controls,h,rate)
                curve=SplineTrajectory(controls,h,stationary_start=stationary,stationary_end=True,
                                       boundary_ramp_duration=curve.boundary_ramp_duration*scale)
            if proof.feasible:break
        timing=dict(seed_s=seedwall,optimization_s=opt['wall_s'],validation_s=time.monotonic()-t,total_s=time.monotonic()-wall,
                    wall_started_s=wall,wall_completed_s=time.monotonic(),clock_repair=repair,optimizer=opt,warm_seed_layers=self.warm_used)
        if not proof.feasible:raise PlanRejected(proof,timing,layers)
        self.phase('certified',duration_s=float(curve.duration))
        proof.metrics['ending_policy']=dict(stationary_end=True,braking_tail_s=curve.finish_ramp_duration,
            endpoint_velocity=curve.evaluate([curve.duration],1)[0].tolist(),endpoint_acceleration=curve.evaluate([curve.duration],2)[0].tolist(),
            normal_operation='accepted successor may replace the committed plan before its certified braking tail')
        self.last_curve=curve
        terminal_slack=0.
        if final:
            last,_=self.geometry.path_batch.evaluate(curve.evaluate([curve.duration],0))
            terminal_slack=max(0.,min(last[0,0]-self.settings.body_margin,last[0,1]-self.settings.arm_margin))
        return dict(id=request_id,curve=curve,proof=proof,snapshot=snapshot,route=route,layers=layers,final=final,timing=timing,
                    terminal_slack_m=terminal_slack,wall_started=wall,wall_completed=time.monotonic(),method=self.method)

class PlanRejected(RuntimeError):
    def __init__(self,proof,timing,layers):
        super().__init__(str(proof.failure_reason));self.proof=proof;self.timing=timing;self.layers=layers
