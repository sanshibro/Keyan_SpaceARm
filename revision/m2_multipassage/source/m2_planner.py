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
from uav_continuum_whole_body_v2.trajectory import SplineTrajectory,resample_states
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
        self.fine=self.geometry;self.validator=Validator(self.geometry);self.snapshot=None
        self.catalog=np.array([(np.deg2rad(t),np.deg2rad(p)) for t in (40,70,85,110) for p in range(0,360,30)]+[(0.,np.pi)])
        self.last_curve=None
        self.last_opt=None;self.warm_points=None;self.warm_used=0
    def shapes(self,start):
        if self.method=='straight':return np.array([[0.,start[5]]])
        if self.method=='fixed':return np.array([[FIXED_THETA,FIXED_PHI]])
        candidates=np.vstack((start[4:6],self.catalog));valid=[]
        for theta,phi in candidates:
            r=self.geometry.self_report(theta,phi)
            if r['hazardous']>=self.settings.self_margin and r['assembly_min']>=0:valid.append([theta,phi])
        return np.asarray(valid)
    def seed(self,start,end,final=False):
        g=self.geometry;c=self.settings;shapes=self.shapes(start);n=c.knots
        xs=np.linspace(start[0],end[0],n);beam=[(0.,[start.copy()])];layers=[];self.seed_diagnostics=layers
        # A workspace sampling grid is fixed before observing any gate. Every
        # retained point is certified against the current snapshot only.
        ys=np.arange(-.48,.481,.04);zs=np.arange(1.86,2.401,.04)
        bodycomp=g.local(0.,0.)[0];bodycomp=np.ascontiguousarray(bodycomp[bodycomp[:,7]==0])
        last_valid=None
        for k,x in enumerate(xs[1:],1):
            alpha=k/(n-1);line=(1-alpha)*start+alpha*end
            warm=None
            if self.warm_points is not None and self.warm_points[0,0]<=x<=self.warm_points[-1,0]:
                warm=np.array([np.interp(x,self.warm_points[:,0],self.warm_points[:,j]) for j in range(6)])
                line[1:]=warm[1:];self.warm_used+=1
            if final and k==n-1:positions=np.array([end[:4]])
            else:
                yy=np.unique(np.r_[ys,line[1],start[1]]);zz=np.unique(np.r_[zs,line[2],start[2]])
                poses=np.array([[x,y,z,psi] for y in yy for z in zz for psi in (0.,)])
                gaps=self.body_batch.query(np.column_stack((poses,np.zeros((len(poses),2)))),reject_below=.085)
                feasible=np.flatnonzero(gaps>=c.body_margin+.011)
                costs=2*(poses[:,1]-line[1])**2+1.2*(poses[:,2]-line[2])**2+20*np.maximum(0.,.125-gaps)**2
                look=poses.copy();look[:,0]+=.30
                ahead=self.body_batch.query(np.column_stack((look,np.zeros((len(look),2)))),reject_below=.085)
                expanded=len(feasible)<3 or float(np.max(ahead))<.125
                if expanded:
                    # The seed is genuinely 6D: when rigid-body passage narrows,
                    # expand vehicle yaw before attempting arm adaptation.
                    roots=poses
                    extra=np.array([np.r_[p[:3],yaw] for p in roots for yaw in np.deg2rad(np.arange(-180,180,30)) if abs(yaw)>1e-8])
                    extra_gaps=self.body_batch.query(np.column_stack((extra,np.zeros((len(extra),2)))),reject_below=.085)
                    poses=np.vstack((poses,extra));gaps=np.r_[gaps,extra_gaps]
                    refined=np.array([np.r_[p[:3],p[3]+delta] for p in poses[np.argsort(gaps)[-36:]] for delta in np.deg2rad([-10,-5,5,10])])
                    refined_gaps=self.body_batch.query(np.column_stack((refined,np.zeros((len(refined),2)))),reject_below=.085)
                    poses=np.vstack((poses,refined));gaps=np.r_[gaps,refined_gaps]
                    costs=2*(poses[:,1]-line[1])**2+1.2*(poses[:,2]-line[2])**2+.02*(poses[:,3]-line[3])**2+20*np.maximum(0.,.125-gaps)**2
                    feasible=np.flatnonzero(gaps>=c.body_margin+.011)
                if len(feasible)<3:
                    possible=np.flatnonzero(gaps>-.03)
                    for j in possible[np.argsort(gaps[possible])[::-1][:48]]:
                        exact=g.convex_map.query(np.r_[poses[j],0.,0.],groups=(0,))[0]['lower']
                        gaps[j]=max(gaps[j],exact)
                    feasible=np.flatnonzero(gaps>=c.body_margin+.011)
                    costs=2*(poses[:,1]-line[1])**2+1.2*(poses[:,2]-line[2])**2+20*np.maximum(0.,.125-gaps)**2
                    if len(feasible):g.tight_search=True
                order=feasible[np.argsort(costs[feasible])];selected=[];keys=set()
                if expanded:
                    # Keep a low-cost member of each yaw stratum before filling
                    # by cost. A currently wider straight branch can terminate
                    # inside this same observed map while a turning branch survives.
                    yawkeys=[];first=[]
                    for j in order:
                        yy=int(round(poses[j,3]/(np.pi/6)))
                        if yy not in yawkeys:yawkeys.append(yy);first.append(j)
                    order=np.r_[first,order]
                for j in order:
                    key=(round(poses[j,1]/.06),round(poses[j,2]/.05),round(poses[j,3]/.3))
                    if key in keys:continue
                    # A short geometric support tube anticipates spline corner
                    # rounding without consulting an obstacle label or next map.
                    lead=np.r_[poses[j],0.,0.];lead[0]+=.10
                    trail=lead.copy();trail[0]-=.20
                    clearance=self.body_batch.query(np.array([lead,trail])).tolist()
                    if g.tight_search:
                        clearance=[max(old,g.convex_map.query(s,groups=(0,))[0]['lower']) if old<c.body_margin+.011 else old for old,s in zip(clearance,(lead,trail))]
                    if min(clearance)<c.body_margin+.011:continue
                    keys.add(key);selected.append(j)
                    if len(selected)>=(14 if expanded else 7):break
                positions=poses[selected]
                if not len(positions):
                    # The seed's extra corner-rounding reserve is not a hard
                    # constraint. At a tight local slice, refine the leading
                    # native candidates and retain those meeting the registered
                    # 85 mm constraint; the final spline remains independently
                    # certified at the same strict margin.
                    chosen=[]
                    for j in np.argsort(gaps)[::-1][:72]:
                        exact=g.convex_map.query(np.r_[poses[j],0.,0.],groups=(0,))[0]['lower']
                        if exact>=c.body_margin+.0001:chosen.append(int(j))
                        if len(chosen)>=7:break
                    positions=poses[chosen]
                    if len(chosen):g.tight_search=True
                if warm is not None:
                    lead=warm.copy();lead[0]+=.10;trail=warm.copy();trail[0]-=.10
                    if min(g.query(bodycomp,lead)[0][0],g.query(bodycomp,trail)[0][0])>=c.body_margin+.011:
                        positions=np.vstack((warm[:4],positions))
                if self.method=='sequential':positions=positions[:1]
            candidates=[]
            for p in positions:
                local_shapes=shapes
                if warm is not None and self.method not in ('straight','fixed'):
                    sr=g.self_report(*warm[4:6])
                    if sr['hazardous']>=c.self_margin and sr['assembly_min']>=0:local_shapes=np.vstack((warm[4:6],shapes))
                candidates.extend(np.r_[p,shape] for shape in (end[None,4:6] if final and k==n-1 else local_shapes))
            candidates=np.asarray(candidates)
            feasible=[];byshape={}
            for q in candidates:byshape.setdefault(tuple(q[4:]),[]).append(q)
            for shape,states in byshape.items():
                states=np.asarray(states);components=g.local(*shape)[0];gaps=g.query_batch(components,states)
                for q,gap in zip(states,gaps):
                    if not g.known(components,q):continue
                    if np.min(gap)<min(c.body_margin,c.arm_margin):
                        exact=g.convex_map.query(q,groups=tuple(np.flatnonzero(gap<min(c.body_margin,c.arm_margin))))
                        for group,res in exact.items():gap[group]=max(gap[group],res['lower'])
                    reserve=0. if (g.tight_search or (final and k==n-1)) else .007
                    if min(gap)>=min(c.body_margin,c.arm_margin)+reserve:feasible.append((q,float(min(gap))))
            proposals=[]
            for q,gap in feasible:
                bq=bend(q);node=.08*np.sum((q[1:3]-line[1:3])**2)+5*max(0.,.125-gap)**2
                if self.coupled:node+=.35*g.physical_coupling(q)['com_offset']**2
                options=[]
                for cost,path in beam:
                    prev=path[-1];delta=q-prev;db=bq-bend(prev)
                    edge=2*np.sum(delta[1:3]**2)+.04*np.dot(db,db)+.02*delta[3]**2
                    # Search connectivity is geometric; actual rate limits are
                    # imposed by positive span timing and dense validation.
                    # Rejecting a >.20 m or >.85 rad transition here eliminated
                    # every side-bending branch before it could be retimed.
                    if np.linalg.norm(delta[1:3])>.40 or np.linalg.norm(db)>2.2:continue
                    options.append((cost+edge+node,path))
                for cost,path in sorted(options,key=lambda z:z[0])[:2]:
                    proposals.append((cost,path,q))
            proposals.sort(key=lambda z:z[0]);nextbeam=[];seen=set();edge_cache={};edge_failures=[]
            # Retain directional morphology branches rather than filling the
            # tiny beam with five nearly identical straight-arm body poses.
            def append(item):
                cost,path,q=item;key=q.tobytes()
                if key in seen:return False
                edgekey=(path[-1].tobytes(),key)
                if edgekey not in edge_cache:
                    edge_cache[edgekey]=g.edge(path[-1],q)
                    if not edge_cache[edgekey] and len(edge_failures)<12:edge_failures.append(g.last_edge_failure)
                if not edge_cache[edgekey]:return False
                nextbeam.append((cost,path+[q.copy()]));seen.add(key);return True
            for item in proposals:
                if append(item):break
            # Position/yaw branches matter independently of bend direction.
            # Preserve early sideways/yaw choices before morphology quadrants.
            for yawbin in (-3,-2,-1,1,2,3):
                if len(nextbeam)>=c.beam-4:break
                for item in proposals:
                    q=item[2]
                    if int(np.clip(np.round(wrap(q[3])/(np.pi/6)),-3,3))==yawbin and append(item):break
            for quadrant in range(4):
                if len(nextbeam)>=c.beam:break
                for item in proposals:
                    q=item[2]
                    if q[4]>.15 and int(np.floor((q[5]%(2*np.pi)+np.pi/4)/(np.pi/2)))%4==quadrant and append(item):break
            for item in proposals:
                if len(nextbeam)>=c.beam:break
                append(item)
            layers.append(dict(x=float(x),body_candidates=len(positions),whole_candidates=len(feasible),survivors=len(nextbeam),proposals=len(proposals),edge_failures=edge_failures,beam_before=[p[-1].tolist() for _,p in beam],beam_after=[p[-1].tolist() for _,p in nextbeam]))
            if not nextbeam:break
            beam=nextbeam;last_valid=beam[0][1]
        if last_valid is None or len(last_valid)<4:
            raise RuntimeError('No connected local whole-body seed; '+str(layers[-1] if layers else {}))
        route=np.asarray(last_valid);reached=len(route)==n
        nonzero=np.flatnonzero(route[:,4]>.015)
        if len(nonzero):route[:nonzero[0],5]=route[nonzero[0],5] # the zero-bend chart has no physical azimuth
        route[:,3]=np.unwrap(route[:,3]);route[:,5]=np.unwrap(route[:,5])
        return route,layers,reached
    def make_curve(self,route,rate,final,spans=None,stationary_start=True):
        K=self.settings.knots;control=resample_states(route,K)
        if spans is None:
            delta=np.diff(control,axis=0);length=np.linalg.norm(delta[:,:3],axis=1)
            demand=np.maximum.reduce([length/self.settings.speed,abs(delta[:,3])/.5,abs(delta[:,4])/.65,
                                      abs(delta[:,5])*np.minimum(1.,np.maximum(control[:-1,4],control[1:,4])/.1)/.8])
            # Cubic spans each integrate about three adjacent polygon segments.
            spans=np.array([max(.20,float(np.mean(demand[max(0,i-1):min(len(demand),i+3)]))) for i in range(K-3)])
            spans*=len(demand)/(K-3)
        if not stationary_start:control[1]=control[0]+np.asarray(rate)*spans[0]/3.
        return SplineTrajectory(control,np.asarray(spans),stationary_start=stationary_start,stationary_end=True,
                                boundary_ramp_duration=.65)
    def optimize(self,seed,rate,final,stationary_start):
        c=self.settings;started=time.monotonic();base=self.make_curve(seed,rate,final,stationary_start=stationary_start)
        points=base.control_states.copy();dur=base.span_durations.copy();K=len(points);count=K-3
        # A spatial mode in all five non-progress coordinates and three smooth
        # log-duration modes give a small actual L-BFGS-B problem. All K-3 spans
        # retain distinct positive durations; the modes couple their changes.
        u=np.linspace(0,1,K);basis=np.sin(np.pi*u)[:,None];basis[:2]=0;basis[-1]=0
        hcoord=np.linspace(-1,1,count);clock_basis=np.column_stack((np.ones(count),hcoord,np.sin(np.pi*(hcoord+1)/2)))
        dim=5;initial=self.last_opt.copy() if self.last_opt is not None else np.zeros(dim+3);best=[np.inf,base,initial.copy()];calls=[0]
        def decode(v):
            p=points.copy();deformation=basis@v[:dim].reshape(1,dim)
            deformation[:,3:]*=np.minimum(1.,points[:,4,None]/.3)
            p[:,1:6]+=deformation
            if self.method=='straight':p[:,4:6]=points[:,4:6]
            if self.method=='fixed':p[:,4:6]=[FIXED_THETA,FIXED_PHI]
            if self.method=='sequential':p[:,:4]=points[:,:4]
            h=dur*np.exp(clock_basis@v[dim:]);
            if not stationary_start:p[1]=p[0]+np.asarray(rate)*h[0]/3
            return SplineTrajectory(p,h,stationary_start=stationary_start,stationary_end=True,boundary_ramp_duration=.65)
        def objective(v):
            if time.monotonic()-started>c.optimize_budget:raise TimeoutError('L-BFGS-B wall deadline')
            curve=decode(v);t=np.linspace(0,curve.duration,K+3);q=curve.evaluate(t,0);vel=curve.evaluate(t,1);acc=curve.evaluate(t,2)
            if np.min(q[:,4])<0 or np.max(q[:,4])>self.geometry.theta_max:
                return 1e6+1e5*np.sum(np.maximum(0.,-q[:,4])**2+np.maximum(0.,q[:,4]-self.geometry.theta_max)**2)
            gaps=np.array([[r['body'],r['arm']] if r['known'] else [-1,-1] for r in (self.geometry.state(s,check_self=False) for s in q)])
            penalty=2e4*np.mean(np.maximum(0.,.089-gaps)**2)+2e3*np.mean(np.maximum(0.,q[:,4]-self.geometry.theta_max)**2+np.maximum(0.,-q[:,4])**2)
            cost=.10*curve.duration+5*np.mean((q[:,1:3]-np.linspace(points[0,1:3],points[-1,1:3],len(q)))**2)+.01*np.mean(acc[:,:3]**2)+penalty
            cost+=.6*np.mean((np.maximum(0.,np.array([.11,.14])-gaps)/.05)**2)
            if self.coupled:
                cp=np.array([self.geometry.physical_coupling(s)['com_offset'] for s in q])
                cost+=.8*np.mean((cp/.08)**2)+.14*curve.duration*np.mean(vel[:,4:6]**2)+.05*np.mean(acc[:,4:6]**2)+.15*np.mean(np.diff(curve.span_durations)**2)
            calls[0]+=1
            if cost<best[0]:best[:]=[cost,curve,v.copy()]
            return cost
        bounds=[(-.06,.06),(-.06,.06),(-.18,.18),(-.18,.18),(-.28,.28)]+[(-.12,.25)]*3
        status='complete'
        try:
            result=minimize(objective,initial,method='L-BFGS-B',bounds=bounds,options=dict(maxiter=5,maxfun=140,ftol=1e-5,eps=3e-4));status=str(result.message)
        except TimeoutError:status='wall_budget'
        was_warm=self.last_opt is not None;self.last_opt=best[2].copy()
        return best[1],base,dict(wall_s=time.monotonic()-started,calls=calls[0],status=status,algorithm='L-BFGS-B',
                               warm_coefficients=was_warm,initial=initial.tolist(),best=best[2].tolist())
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
        t=time.monotonic();proof=self.validator.validate(curve)
        if not proof.feasible and proof.status=='geometry_rejected':
            curve=base;proof=self.validator.validate(curve)
        # Dynamic feasibility is repaired inside the planner, identically for
        # all methods, by minimum uniform clock scaling; no execution slow play.
        repair=1.
        if not proof.feasible and proof.status=='dynamics_rejected':
            peaks=proof.metrics['peaks'];bounds=proof.metrics['limits']
            ratios=[(peaks[k]/bounds[k])**(1/(3 if k=='jerk' else 2 if 'acceleration' in k else 1)) for k in peaks]
            repair=max(1.,max(ratios)*1.025)
            controls=curve.control_states.copy();h=curve.span_durations*repair
            if not stationary:controls[1]=controls[0]+rate*h[0]/3
            curve=SplineTrajectory(controls,h,stationary_start=stationary,stationary_end=True,
                                   boundary_ramp_duration=curve.boundary_ramp_duration*repair)
            proof=self.validator.validate(curve)
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
