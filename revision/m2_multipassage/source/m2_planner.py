"""Small causal 6D beam, warm L-BFGS-B and independent dense certification.

No gate labels or global obstacle coordinates enter this module. The same
coupled kernel is used by coupled/stop/hard. Ablations alter only declared DOFs
or the four registered regularizers.
"""
from dataclasses import dataclass
from types import SimpleNamespace
import math,time
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
        lim=SimpleNamespace(body_margin=self.settings.body_margin,arm_margin=self.settings.arm_margin,tilt_max=0.)
        self.geometry=Geometry(plant,lim);self.geometry.plan_self_margin=self.settings.self_margin
        from m2_path_batch import PathBatch
        self.geometry.path_batch=PathBatch(self.geometry)
        self.fine=self.geometry;self.validator=Validator(self.geometry);self.snapshot=None
        self.roadmap=LocalLattice(self)
        self.last_curve=None
        self.last_opt=None;self.warm_points=None;self.warm_used=0
    def seed(self,start,end,final=False):
        return self.roadmap.search(start,end,final)
    def make_curve(self,route,rate,final,spans=None,stationary_start=True):
        K=self.settings.knots;delta=np.diff(route,axis=0);length=np.linalg.norm(delta[:,:3],axis=1)
        mag=np.minimum(1.,np.maximum(route[:-1,4],route[1:,4])/.12)
        demand=np.maximum.reduce([length/self.settings.speed,abs(delta[:,3])/.64,abs(delta[:,4])/.70,abs(delta[:,5])*mag/.86])
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
        started=time.monotonic();self.current_phase='snapshot';self.seed_diagnostics=[]
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
            pool=self.goal.get('candidates',[self.goal]);feasible=[]
            for candidate in pool:
                q=np.r_[candidate['body'],np.deg2rad(candidate.get('yaw_deg',0)),np.deg2rad(candidate['theta_deg']),np.deg2rad(candidate['phi_deg'])]
                if self.method=='straight' and q[4]>.001:continue
                if self.method=='fixed' and (abs(q[4]-FIXED_THETA)>.001 or abs(wrap(q[5]-FIXED_PHI))>.001):continue
                check=self.geometry.state(q,exact=True)
                if check['reason'] is None:
                    cost=np.linalg.norm(q[:3]-start[:3])+.1*np.linalg.norm(bend(q)-bend(start))
                    feasible.append((cost,q,check))
            if not feasible:raise RuntimeError('No observed terminal task-family pose passes the registered margins and method DOFs')
            feasible.sort(key=lambda x:x[0]);end=feasible[0][1].copy()
        self.current_phase='seed';t=time.monotonic();route,layers,reached=self.seed(start,end,final);seedwall=time.monotonic()-t
        final=final and reached;stationary=np.linalg.norm(rate[:3])<.03
        self.current_phase='optimization'
        curve,base,opt=self.optimize(route,rate,final or self.method=='stop',stationary)
        self.current_phase='validation'
        t=time.monotonic()
        for candidate in (curve,base):
            curve=candidate;repair=1.
            for retry in range(6):
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
        proof.metrics['ending_policy']=dict(stationary_end=True,braking_tail_s=curve.finish_ramp_duration,
            endpoint_velocity=curve.evaluate([curve.duration],1)[0].tolist(),endpoint_acceleration=curve.evaluate([curve.duration],2)[0].tolist(),
            normal_operation='accepted successor may replace the committed plan before its certified braking tail')
        self.last_curve=curve
        return dict(id=request_id,curve=curve,proof=proof,snapshot=snapshot,route=route,layers=layers,final=final,timing=timing,
                    wall_started=wall,wall_completed=time.monotonic(),method=self.method)

class PlanRejected(RuntimeError):
    def __init__(self,proof,timing,layers):
        super().__init__(str(proof.failure_reason));self.proof=proof;self.timing=timing;self.layers=layers
