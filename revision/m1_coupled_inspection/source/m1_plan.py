"""Surveyed-cell V2.1 adapter with method-constrained DOFs and the native loaded geometry."""
import argparse,json,math,sys,time,traceback
from pathlib import Path
from dataclasses import replace
import numpy as np
from m1_layout import HOVER,METHODS,AXIS
from m1_plant import build,task_pose,ObserverGeometry,cart
from m1_geometry import GalleryGeometry
from planner_adapter import (V21Adapter,WholeBodySeedPlanner,WholeBodyOptimizer,SweptVolumeValidator,
        ArmModel,LIB,load_planner_configuration,ARC_LENGTH,BASE_LENGTH,PLAN_THETA_MAX)
from uav_continuum_whole_body_v2.trajectory import SplineTrajectory,resample_states
from uav_continuum_whole_body_v2.seed_planner import SeedPlan
from uav_continuum_whole_body_v2.geometry import wrap_angle
from gate_live_optics import clean
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize

def writej(p,obj):
    Path(p).write_text(json.dumps(clean(obj),indent=2,allow_nan=False)+'\n')
def body_reference(geom,start,end):
    """Dynamic programming over a body-only longitudinal voxel lattice; no arm or gate labels."""
    ys=np.arange(-.42,.505,.025);zs=np.unique(np.r_[np.arange(2.04,2.361,.025),2.15,2.154,2.158])
    yz=np.array(np.meshgrid(ys,zs,indexing='ij')).reshape(2,-1).T
    # Between slices impose an isotropic gradient bound on the vehicle centreline.
    xs=np.linspace(start[0],end[0],int(np.ceil(abs(end[0]-start[0])/.038))+1)
    full=[];distances=[];slacks=[];parents=[];costs=None;last_yz=None
    a,_,_=geom.local(0.,0.);a=np.ascontiguousarray(a[a[:,7]==0])
    for k,x in enumerate(xs):
        choices=start[None,1:3] if k==0 else end[None,1:3] if k==len(xs)-1 else yz
        yaw=start[3] if k<len(xs)-1 else end[3]
        states=np.array([[x,y,z,yaw,0.,0.] for y,z in choices])
        gaps=np.array([geom.query(a,s)[0][0] for s in states]);ok=gaps>=geom.limits.body_margin
        choices=choices[ok];states=states[ok];gaps=gaps[ok]
        if not len(states):raise ValueError('body reference: no admissible lattice node at x=%.3f'%x)
        guard=np.maximum(0.,geom.limits.body_margin+.055-gaps)
        line=(1-k/(len(xs)-1))*start[1:3]+k/(len(xs)-1)*end[1:3]
        node=.003*np.sum((choices-line)**2,axis=1)+.34*guard**2
        if costs is None:prior=np.zeros(len(states),int);vals=node
        else:
            delta=cdist(last_yz,choices);edge=delta**2+np.where(delta<.164,0,np.inf)
            tmp=costs[:,None]+edge;prior=np.argmin(tmp,axis=0);vals=tmp[prior,np.arange(len(states))]+node
        full.append(states);parents.append(prior);costs=vals;last_yz=choices
    if not np.isfinite(costs).any():raise ValueError('body reference: disconnected lattice')
    j=int(np.argmin(costs));back=[]
    for k in reversed(range(len(xs))):
        back.append(full[k][j]);j=int(parents[k][j])
    route=np.array(back[::-1])
    headings=np.arctan2(np.gradient(gaussian_filter1d(route[:,1],1.7)),np.gradient(route[:,0]))
    # A body-only yaw pass balances tangent alignment and slew against actual
    # rigid-body support. Joint planning can change these headings again.
    angles=np.unique(np.r_[np.deg2rad(np.arange(-48,49,6)),start[3],end[3]])
    costs=np.where(abs(angles-start[3])<1e-7,0.,np.inf);parents=[]
    for i,state in enumerate(route[1:],1):
        gap=np.array([geom.query(a,np.r_[state[:3],v,0.,0.])[0][0] for v in angles])
        node=.13*(angles-headings[i])**2+12.*np.maximum(0.,geom.limits.body_margin+.027-gap)**2
        node[gap<geom.limits.body_margin]=np.inf
        if i==len(route)-1:node[abs(angles-end[3])>1e-7]=np.inf
        edge=(angles[:,None]-angles[None,:])**2
        c=costs[:,None]+edge
        ind=np.argmin(c,axis=0);parents.append(ind);costs=c[ind,np.arange(len(angles))]+node
    if np.isfinite(costs).any():
        j=int(np.argmin(costs))
        for i in range(len(route)-1,0,-1):
            route[i,3]=angles[j];j=parents[i-1][j]
    return route

class Seed(WholeBodySeedPlanner):
    def select_layer(self,paths,index):
        unique={}
        for cost,path in paths:
            key=path[-1].tobytes()
            if key not in unique:unique[key]=(cost,path)
        ordered=list(unique.values());chosen=[];keys=set()
        if ordered:
            chosen=[ordered[0]]
            keys.add(ordered[0][1][-1].tobytes())
        for project in (lambda s:(round(s[4]/.55),round(s[5]/1.0)),lambda s:(round(s[1]/.07),round(s[2]/.024)),lambda s:s.tobytes()):
            seen=set()
            for cost,path in ordered:
                s=path[-1];k=project(s)
                if k in seen:continue
                seen.add(k);raw=s.tobytes()
                if raw not in keys:chosen.append((cost,path));keys.add(raw)
                if len(chosen)>=self.config.seed_beam_width:break
            if len(chosen)>=self.config.seed_beam_width:break
        self.layer_log.append(dict(index=index,survivors=[p[-1].tolist() for _,p in chosen]))
        print('layer',index,'unique',len(unique),'beam',len(chosen),flush=True)
        if not chosen:raise RuntimeError('no safe transition into longitudinal layer %d'%index)
        self.last_path=np.asarray(chosen[0][1]).copy()
        return chosen
    def _clearance(self,s):
        k=s.tobytes()
        if k not in self._clearance_cache:
            g=self.platform_geometry
            self._clearance_cache[k]=(*g.search_clearances(s),g.physical_coupling(s))
        return self._clearance_cache[k]
    def _candidates(self,reference,index,current_shape,risk,phi_locked=False):
        ref=reference[index];method=self.method
        if index==0:return [self.start.copy()]
        if index==len(reference)-1:return [self.end.copy()]
        free=method not in ('straight','fixed')
        shapes=[self.start[4:6]] if not free else [np.deg2rad([t,p]) for p in (180,150,210,90,270,0) for t in (0,35,55,75,90,105) if t or p==180]
        if free:shapes.extend([self.end[4:6],self.start[4:6]])
        # Rotate shape and aircraft independently; positive/negative offsets are offered at every slice.
        offsets=[(0,0)] if method=='sequential' else [(y,z) for y in (0.,-.06,.06,-.12,.12) for z in (0.,-.023,.023)]
        yaws=[ref[3]] if method=='sequential' else np.unique(np.r_[0.,ref[3]+np.deg2rad([0.,-18.,18.])])
        candidates=[]
        for dy,dz in offsets:
            for yaw in yaws:
                for theta,phi in shapes:
                    s=np.array([ref[0],ref[1]+dy,ref[2]+dz,yaw,theta,phi])
                    body,arm,c=self._clearance(s)
                    if min(body,arm)<self.limits.body_margin or c['com_offset']>self.limits.com_max:continue
                    cost=self._node_cost(s,ref,True)
                    candidates.append((cost,s))
        candidates.sort(key=lambda item:item[0])
        if not candidates:
            self.empty_at=ref.tolist()
            return []
        # Retain distinct basins of position, bend magnitude and orientation after scoring.
        selected=[];used=set()
        for feat in (lambda s:(round(s[4]*2),round(s[5]*2)),lambda s:(round(s[1]*10),round(s[2]*16),round(s[3]*2)),lambda s:tuple(s)):
            seen=set()
            for _,s in candidates:
                k=feat(s)
                if k in seen:continue
                seen.add(k);b=s.tobytes()
                if b in used:continue
                used.add(b);selected.append(s)
                if len(selected)>=self.config.seed_shape_limit:return selected
                if len(seen)>=max(4,self.config.seed_shape_limit//3):break
        return selected
    def _node_cost(self,s,ref,risk):
        body,arm,com=self._clearance(s)
        gap=min(body,arm);target=self.limits.body_margin+.045
        d=s[:3]-ref[:3]
        return (7.*np.dot(d,d)+.22*wrap_angle(s[3]-ref[3])**2+
                .07*s[4]**2+.012*wrap_angle(s[5]-np.pi)**2+
                1550.*max(0.,target-gap)**2+.2*(com['com_offset']/.08)**2)
    def _edge_cost(self,a,b):
        k=(a.tobytes(),b.tobytes())
        if k in self._edge_cache:return self._edge_cache[k]
        delta=b-a;delta[3]=wrap_angle(delta[3]);delta[5]=wrap_angle(delta[5])
        # Collision filter is independent of beam ranking. Angularly close
        # representatives are expensive only when geometry actually permits an edge.
        if abs(delta[4])>1.95 or abs(delta[5])>2.16:cost=np.inf
        elif not self.platform_geometry.edge(a,b):cost=np.inf
        else:cost=np.dot(delta[:3],delta[:3])+1.35*delta[4]**2+.24*(min(1.,.5*(a[4]+b[4]))*delta[5])**2+.2*delta[3]**2
        self._edge_cache[k]=cost
        return cost

class Optimizer(WholeBodyOptimizer):
    def repair_clock(self,controls,durations,invalid,seed,seed_durations):
        for c,h in ((controls,durations),(seed,seed_durations)):
            for ratio in (1.16,1.34,1.59,1.90,2.25):
                clock=h*ratio
                if np.max(clock)>self.config.dt_max:break
                checked=self.validator.validate(self._spline(c,clock),stop_on_failure=True)
                if checked.feasible:return c,clock,checked
                if checked.failure_reason in ('body_collision','arm_collision','self_collision'):break
        return None
    def polish_certified(self,controls,dt,proof,trial,reference):
        g=self.platform_geometry;w=self.config.weights;scale=np.deg2rad(18.0)
        trial_ctrl,_=self._decode(trial)
        def score(c,h):
            curve=self._spline(c,h)
            times=np.linspace(0,curve.duration,101);s=curve.evaluate(times,0);v=curve.evaluate(times,1);a=curve.evaluate(times,2)
            cp=np.array([g.physical_coupling(q)['com_offset'] for q in s])
            rr=np.stack([np.interp(np.linspace(0,1,len(s)),np.linspace(0,1,len(reference)),col) for col in reference[:,:3].T],axis=1)
            vel2=v[:,4]**2+(np.sin(s[:,4])*v[:,5])**2
            acc2=a[:,4]**2+(np.sin(s[:,4])*a[:,5])**2
            return float(w['duration']*curve.duration+w['reference']*np.mean(np.sum((s[:,:3]-rr)**2,axis=1))+
                 w['com']*np.mean((cp/self.limits.com_max)**2)+w['arm_rate']*curve.duration*np.mean(vel2)/scale**2+
                 w['arm_acceleration']*curve.duration*np.mean(acc2)/(2*scale)**2+w['dt_smooth']*np.mean(np.diff(h)**2))
        def schedule(c):
            b=c[:,4,None]*np.c_[np.cos(c[:,5]),np.sin(c[:,5])]
            bend=np.diff(b,axis=0)
            power=np.array([np.mean(np.sum(bend[i:i+3]**2,axis=1)) for i in range(len(dt))])
            def obj(h):
                rate=bend[1:len(h)+1]/h[:,None]
                accel=np.diff(rate,axis=0)/(.5*(h[1:]+h[:-1]))[:,None]
                return (w['duration']*h.sum()+w['arm_rate']*np.sum(power/h)/scale**2+
                        w['arm_acceleration']*np.sum(np.sum(accel**2,axis=1)*(h[1:]+h[:-1])*.5)/(2*scale)**2+
                        w['dt_smooth']*np.mean(np.diff(h)**2))
            bounds=[(max(self.config.dt_min,.65*t),min(self.config.dt_max,1.85*t)) for t in dt]
            return minimize(obj,dt,method='L-BFGS-B',bounds=bounds,options={'maxiter':36}).x
        candidates=[];original=score(controls,dt)
        for cf in (0.,.18,.45,.72,1.):
            c=controls+cf*(trial_ctrl-controls)
            if self.locked:c[:,self.locked]=self._seed_base[:,self.locked]
            clock=schedule(c)
            for tf in (.25,.5,.75,1.):
                h=dt+tf*(clock-dt);cost=score(c,h)
                if cost<original-1e-6:candidates.append((cost,cf,tf,c,h))
        log=[]
        for cost,cf,tf,c,h in sorted(candidates,key=lambda x:x[0]):
            check=self.validator.validate(self._spline(c,h),stop_on_failure=True)
            log.append(dict(spatial_fraction=cf,time_fraction=tf,cost=cost,valid=check.feasible,reason=check.failure_reason))
            if check.feasible:return c,h,check,dict(initial_cost=original,accepted_cost=cost,trials=log)
        return controls,dt,proof,dict(initial_cost=original,accepted_cost=original,trials=log)
    def initial_span_clock(self,seed,uniform):
        delta=np.abs(np.diff(seed,axis=0));translation=np.linalg.norm(delta[:,:3],axis=1)
        demand=np.maximum.reduce([translation/self.config.nominal_speed,delta[:,3]/self.limits.yaw_rate_max,
                                  delta[:,4]/self.limits.theta_rate_max,delta[:,5]/self.limits.phi_rate_max])
        # A cubic span is influenced by three adjacent control differences.
        clocks=np.array([1.54*max(demand[i:i+3]) for i in range(len(uniform))])
        clocks[[0,-1]]*=1.55
        return np.clip(clocks, max(.72,self.config.dt_min),self.config.dt_max)
    def _clearance_samples(self,states):
        g=self.platform_geometry
        return [(*g.search_clearances(s),g.physical_coupling(s)) for s in states]
    def _bounds(self,n):
        bb=super()._bounds(n);locked=self.locked
        for i in range(self.config.optimization_basis_count*6):
            if i%6 in locked:bb[i]=(0.,0.)
        return bb
    def _make_basis(self,n):
        a=super()._make_basis(n);a[-1]=0
        return a
    def _spline(self,c,dt):
        if self.locked:c=c.copy();c[:,self.locked]=self._seed_base[:,self.locked]
        self.last_spline=super()._spline(c,dt);return self.last_spline

class IndustryAdapter(V21Adapter):
    def __init__(self,p,method):
        self.plant=p;self.method=method;self.is_super=True
        self.raw,c,l,m=load_planner_configuration(LIB/'config/planner_v2_realtime.yaml')
        bodymass=p.m.body_mass;mount=p.m.body('mount').id;tool=p.m.body('tool').id
        self.mass=replace(m,uav=float(bodymass[p.base]),arm_base=float(bodymass[mount]),tool=float(bodymass[tool]),
            flexible=float(p.mass-bodymass[p.base]-bodymass[mount]-bodymass[tool]))
        self.arm=ArmModel(arc_length=ARC_LENGTH,base_offset_z=BASE_LENGTH,theta_max=PLAN_THETA_MAX)
        self.limits=replace(l,body_margin=.060,arm_margin=.060,tilt_max=np.deg2rad(2.),z_max=2.38,z_min=2.015)
        w=dict(c.weights)
        w.update(arm_rate=.42,arm_acceleration=.12)
        if method=='unregularized':
            for k in ('com','arm_rate','arm_acceleration','dt_smooth'):w[k]=0.
        self.config=replace(c,knot_count=32,seed_beam_width=18,seed_shape_limit=46,theta_design_margin=0.,
             dt_max=7.5,nominal_speed=.14,optimizer_time_budget=95.,optimizer_max_iterations=62,
             optimization_basis_count=2,risk_margin=.045,output_dt=.02,weights=w,repair_attempts=0)
        self.proposal=GalleryGeometry(p,self.limits);self.fine=GalleryGeometry(p,self.limits)
        self.validator=SweptVolumeValidator(None,self.arm,self.mass,self.limits,self.config,platform_geometry=self.fine)
    def boundaries(self):
        task=task_pose(self.plant)
        shape=np.deg2rad([80,180]) if self.method=='fixed' else np.deg2rad([0,180])
        start=np.r_[HOVER,0,shape];end=np.r_[task['body'],0.,np.deg2rad([50,143])]
        if self.method in ('straight','fixed'):
            g=ObserverGeometry(self.plant);b=shape[0]*np.array([np.cos(shape[1]),np.sin(shape[1])])
            axis=g.world_axis_for(b)
            yaw=np.arctan2(AXIS[1],AXIS[0])-np.arctan2(axis[1],axis[0]) if np.linalg.norm(axis[:2])>.01 else 0.
            R=Rotation.from_euler('z',yaw).as_matrix();g.place_estimate(np.zeros(3),R,b)
            end=np.r_[np.asarray(task['tool'])-g.tip()[0],yaw,shape]
        return start,end
    def solve(self,out):
        out=Path(out);out.mkdir(parents=True,exist_ok=True);start,end=self.boundaries();t0=time.perf_counter()
        report=dict(method=self.method,start=start,end=end,map=self.proposal.map_hash,seed_algorithm='WholeBodySeedPlanner',
                    optimizer='WholeBodyOptimizer',weights=self.config.weights)
        try:
            reference_end=np.r_[task_pose(self.plant)['body'],0.,end[4:6]]
            ref=body_reference(self.proposal,start,reference_end);ref[:,4]=start[4];ref[:,5]=start[5]
            ref[-1]=end
            np.save(out/'body-reference.npy',ref)
            seed=Seed(None,self.arm,self.mass,self.limits,self.config,platform_geometry=self.proposal)
            seed.method=self.method;seed.start=start;seed.end=end;seed.layer_log=[]
            print('body reference',len(ref),round(time.perf_counter()-t0,2),flush=True)
            try:
                seed_plan=seed.plan(ref,current_state=start,terminal_state=end,_force_phi_unlocked=True)
                report['task_route']='full'
            except RuntimeError as error:
                path=getattr(seed,'last_path',np.empty((0,6)))
                if len(path)<5:raise
                path=path[:-1].copy();path[:,3]=np.unwrap(path[:,3]);path[:,5]=np.unwrap(path[:,5])
                ref=path.copy()
                self.config=replace(self.config,knot_count=len(path))
                self.validator.config=self.config
                seed_plan=SeedPlan(path,list(range(1,len(path))),list(range(len(path)-3)),
                      dict(expanded=self.proposal.calls['seed_edges'],shape_mode='theta_phi'))
                report.update(task_route='partial',full_route_rejection=str(error))
            seed_plan.diagnostics['terminal_shape']=seed_plan.states[-1,4:6].tolist()
            np.save(out/'seed.npy',seed_plan.states);report['seed']=seed_plan.diagnostics
            print('beam',round(time.perf_counter()-t0,2),seed_plan.diagnostics['expanded'],flush=True)
            opt=Optimizer(None,self.arm,self.mass,self.limits,self.config,self.validator,platform_geometry=self.proposal)
            opt.locked=([4,5] if self.method in ('straight','fixed') else [0,1,2,3] if self.method=='sequential' else [])
            result=opt.optimize(seed_plan,ref,wall_time_budget=125.,validation_reserve=12.)
            report.update(accepted=bool(result.success),status=result.status,iterations=result.iterations,
                optimizer_details=result.diagnostics,proof=result.validation.__dict__,objective=result.objective)
            tr=result.trajectory
            report['validation_line_search']=tr.metadata.get('validated_refinement')
            if result.success:
                np.savez_compressed(out/'certified.npz',t=tr.times,q=tr.states,dq=tr.velocities,dt=tr.metadata['span_durations'],
                                    controls=tr.metadata['control_states'])
            print('solver',result.status,result.validation.failure_reason,flush=True)
        except (ValueError,RuntimeError) as e:
            report.update(accepted=False,status=str(e),empty_at=getattr(locals().get('seed'),'empty_at',None))
        report.update(elapsed_s=time.perf_counter()-t0,calls=self.proposal.calls,beam_layers=getattr(locals().get('seed'),'layer_log',[]))
        writej(out/'decision.json',report)
        print('decision',report['accepted'],report['status'],round(report['elapsed_s'],2),flush=True)
        return report
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('method',choices=METHODS);parser.add_argument('output',type=Path)
    a=parser.parse_args();IndustryAdapter(build(),a.method).solve(a.output)
