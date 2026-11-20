"""Independent signed convex-clearance audit on shared MjModel/private MjData.

No physics/control state or model flags are modified. Positive distances are
bounded by feasible convex-combination witnesses and separating support planes.
Intersections use EPA; unresolved pairs are reported and rejected in strict mode.
Distances are native collision-convex distances, not triangle-soup distances.
"""
import argparse
import hashlib
import importlib
import itertools
import json
import time
from pathlib import Path
import numpy as np


def closest_simplex(p,q):
    """Closest feasible convex combination of at most five Minkowski vertices."""
    w=p-q;best=None
    for n in range(1,min(4,len(w))+1):
        for combo in itertools.combinations(range(len(w)),n):
            ids=np.asarray(combo,int);v=w[ids]
            if n==1:weights=np.ones(1)
            else:
                edge=v[1:]-v[0]
                z=np.linalg.lstsq(edge.T,-v[0],rcond=1e-12)[0]
                weights=np.r_[1.-z.sum(),z]
                if weights.min() < -1e-10:continue
                weights=np.maximum(weights,0.);weights/=weights.sum()
            point=weights@v;norm2=float(point@point)
            if best is None or norm2<best[0]:
                active=weights>1e-11;best=(norm2,ids[active],weights[active]/weights[active].sum())
    _,ids,weights=best
    return p[ids],q[ids],weights@p[ids],weights@q[ids]


def penetration(support,p,q,tolerance,max_iterations):
    """EPA bounds the minimum translation out of convex overlap."""
    from scipy.spatial import ConvexHull,QhullError
    directions=list(np.eye(3))+list(-np.eye(3))
    directions.extend(np.asarray(v,float) for v in itertools.product((-1.,1.),repeat=3))
    for direction in directions:
        a,b=support(direction);p=np.vstack((p,a));q=np.vstack((q,b))
    w=p-q;_,ids=np.unique(np.round(w,13),axis=0,return_index=True);p=p[ids];q=q[ids]
    previous_upper=np.inf
    for iteration in range(max_iterations):
        w=p-q
        try:hull=ConvexHull(w)
        except QhullError:
            return dict(lower=-tolerance,upper=0.,distance=0.,a=p.mean(0),b=q.mean(0),
                        converged=False,iterations=iteration+1,status='degenerate EPA seed')
        # ConvexHull equations use outward normals and inside <= 0.
        if np.max(hull.equations[:,3])>1e-8:
            return dict(lower=-tolerance,upper=0.,distance=0.,a=p.mean(0),b=q.mean(0),
                        converged=False,iterations=iteration+1,status='EPA seed does not enclose origin')
        depths=-hull.equations[:,3];k=int(np.argmin(depths));normal=hull.equations[k,:3]
        depth=max(0.,float(depths[k]));a,b=support(normal)
        support_depth=float(normal@(a-b));previous_upper=min(previous_upper,support_depth)
        # A coplanar polygon is triangulated by Qhull; its normal projection may
        # lie in another triangle on the same closest supporting plane.
        witnesses=[]
        for face in hull.simplices[np.abs(depths-depth)<1e-9]:
            _,_,wa,wb=closest_simplex(p[face],q[face]);witnesses.append((np.linalg.norm(wa-wb),wa,wb))
        _,wa,wb=min(witnesses,key=lambda item:item[0])
        if support_depth-depth<=tolerance:
            return dict(lower=-support_depth,upper=-depth,distance=-.5*(support_depth+depth),
                        a=wa,b=wb,converged=True,iterations=iteration+1,status='overlap EPA')
        if np.min(np.linalg.norm(w-(a-b),axis=1))<1e-11:break
        p=np.vstack((p,a));q=np.vstack((q,b))
    return dict(lower=-previous_upper,upper=-depth,distance=-.5*(previous_upper+depth),a=wa,b=wb,
                converged=previous_upper-depth<=tolerance,iterations=max_iterations,status='EPA iteration limit')


def convex_distance(support,initial_direction=None,lower=0.,tolerance=1e-4,max_iterations=96):
    """support(d) returns the pair realizing max d dot (A-B)."""
    direction=np.array([1.,0.,0.]) if initial_direction is None else np.asarray(initial_direction,float)
    if np.linalg.norm(direction)<1e-12:direction=np.array([1.,0.,0.])
    a,b=support(-direction);p=a[None,:];q=b[None,:];lb=max(0.,float(lower));history=[]
    for iteration in range(max_iterations):
        upper=float(np.linalg.norm(a-b))
        if upper<1e-9:return penetration(support,p,q,tolerance,max_iterations)
        direction=(a-b)/upper;pa,pb=support(-direction)
        lb=max(lb,float(direction@(pa-pb)),0.)
        if lb>upper+2e-8:raise ArithmeticError('Inconsistent support/AABB lower bound and feasible witness')
        if upper-lb<=tolerance and lb>1e-10:
            return dict(lower=min(lb,upper),upper=upper,distance=.5*(min(lb,upper)+upper),a=a,b=b,
                        converged=True,iterations=iteration+1,status='separated GJK')
        new=pa-pb
        if any(np.linalg.norm(new-old)<1e-12 for old in history):break
        history.append(new)
        p,q,a,b=closest_simplex(np.vstack((p,pa)),np.vstack((q,pb)))
    upper=float(np.linalg.norm(a-b))
    if upper<1e-8:return penetration(support,p,q,tolerance,max_iterations)
    return dict(lower=min(lb,upper),upper=upper,distance=.5*(min(lb,upper)+upper),a=a,b=b,
                converged=upper-lb<=tolerance,iterations=iteration+1,status='GJK stalled')


class DistanceAudit:
    def __init__(self,model,tolerance_m=1e-4,strict=True):
        import mujoco
        self.m=model;self.d=mujoco.MjData(model);self.mj=mujoco
        self.tolerance=float(tolerance_m);self.strict=bool(strict);self.mesh={};self.warm={}
        self.names=[model.geom(i).name or ('geom%d'%i) for i in range(model.ngeom)]
        self.body_names=[model.body(i).name or ('body%d'%i) for i in range(model.nbody)]
        self.gates=np.asarray([i for i,n in enumerate(self.names) if n.startswith('G1_')],int)
        if not len(self.gates):raise ValueError('No registered G1 gate geoms')
        root=int(model.body('uav').id);robot_bodies=[]
        for b in range(model.nbody):
            k=b
            while k and k!=root:k=int(model.body_parentid[k])
            if k==root:robot_bodies.append(b)
        self.robot_bodies=set(robot_bodies)
        self.robot=np.asarray([i for i in range(model.ngeom) if int(model.geom_bodyid[i]) in self.robot_bodies
            and (model.geom_contype[i] or model.geom_conaffinity[i])],int)
        rigid={int(model.body(n).id) for n in ('uav','mount')}
        self.categories={int(i):'body' if int(model.geom_bodyid[i]) in rigid else 'arm_tool' for i in self.robot}
        # Physical bit-1 solids only: extra bit-4 CAD geoms are external-envelope geometry.
        physical=[int(i) for i in self.robot if (int(model.geom_contype[i])&1) and (int(model.geom_conaffinity[i])&1)]
        excluded=set(map(int,model.exclude_signature));explicit={tuple(sorted((int(a),int(b)))) for a,b in zip(model.pair_geom1,model.pair_geom2)}
        parent_filter=not (int(model.opt.disableflags)&int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT))
        counts=dict(same_weld=0,parent_weld=0,explicit_exclusion=0,mask=0);pairs=[]
        for a,b in itertools.combinations(physical,2):
            ba,bb=int(model.geom_bodyid[a]),int(model.geom_bodyid[b])
            wa,wb=int(model.body_weldid[ba]),int(model.body_weldid[bb])
            # Fixed assemblies are one body for collision purposes; no fabricated
            # distance between permanently joined subparts enters self clearance.
            if wa==wb:counts['same_weld']+=1;continue
            if (a,b) in explicit:pairs.append((a,b));continue
            pa=int(model.body_weldid[model.body_parentid[wa]])
            pb=int(model.body_weldid[model.body_parentid[wb]])
            if parent_filter and wa and wb and (wa==pb or wb==pa):counts['parent_weld']+=1;continue
            signature=(min(ba,bb)<<16)+max(ba,bb)
            if signature in excluded:counts['explicit_exclusion']+=1;continue
            if not ((int(model.geom_contype[a])&int(model.geom_conaffinity[b])) or
                    (int(model.geom_contype[b])&int(model.geom_conaffinity[a]))):counts['mask']+=1;continue
            pairs.append((a,b))
        self.self_pairs=np.asarray(pairs,int).reshape(-1,2)
        self.external={group:np.asarray([(int(i),int(j)) for i in self.robot if self.categories[int(i)]==group
            for j in self.gates if (int(model.geom_contype[i])&int(model.geom_conaffinity[j])) or
                                   (int(model.geom_contype[j])&int(model.geom_conaffinity[i]))],int).reshape(-1,2)
            for group in ('body','arm_tool')}
        self.all_ids=np.unique(np.r_[self.robot,self.gates]);self.local_aabb=model.geom_aabb.copy()
        mesh_sizes={};cache={}
        for i in self.all_ids:
            if model.geom_type[i]!=mujoco.mjtGeom.mjGEOM_MESH:continue
            mid=int(model.geom_dataid[i])
            if mid not in cache:
                start=int(model.mesh_vertadr[mid]);num=int(model.mesh_vertnum[mid])
                v=model.mesh_vert[start:start+num].astype(float)
                try:
                    from scipy.spatial import ConvexHull
                    if len(v)>8:v=v[ConvexHull(v).vertices]
                except (ImportError,ValueError,RuntimeError):pass
                cache[mid]=v;mesh_sizes[str(mid)]=dict(native_vertices=num,hull_vertices=len(v))
            self.mesh[int(i)]=cache[mid]
        self.registry=dict(robot_geoms=len(self.robot),physical_self_geoms=len(physical),self_pairs=len(pairs),
            external_pairs={k:len(v) for k,v in self.external.items()},excluded_counts=counts,
            body_exclusion_signatures=sorted(excluded),explicit_geom_pairs=[list(p) for p in sorted(explicit)],
            self_pair_names=[[self.names[a],self.names[b]] for a,b in pairs],mesh_reduction=mesh_sizes,
            filter_rule='MuJoCo 3.2.3 bitmasks, weld groups, FILTERPARENT and explicit body exclusions; fixed assemblies omitted.',
            reference='https://github.com/google-deepmind/mujoco/blob/3.2.3/src/engine/engine_collision_driver.c')
        fingerprint=hashlib.sha256()
        for key in ('geom_type','geom_size','geom_pos','geom_quat','geom_bodyid','geom_dataid','geom_contype',
                    'geom_conaffinity','mesh_vert','body_pos','body_quat','body_parentid','body_weldid','jnt_axis','jnt_pos','exclude_signature'):
            fingerprint.update(key.encode());fingerprint.update(np.ascontiguousarray(getattr(model,key)).view(np.uint8))
        self.registry['model_geometry_sha256']=fingerprint.hexdigest()
        self.last_pair={};self.calls=0;self.refinements=0

    def support(self,i,direction):
        m,d=self.m,self.d;R=d.geom_xmat[i].reshape(3,3);v=R.T@direction
        h=m.geom_size[i];kind=int(m.geom_type[i]);G=self.mj.mjtGeom
        if i in self.mesh:
            verts=self.mesh[i];p=verts[int(np.argmax(verts@v))]
        elif kind==int(G.mjGEOM_BOX):p=h*np.sign(v)
        elif kind==int(G.mjGEOM_SPHERE):p=v*h[0]/max(np.linalg.norm(v),1e-15)
        elif kind==int(G.mjGEOM_CAPSULE):
            p=v*h[0]/max(np.linalg.norm(v),1e-15);p[2]+=h[1]*np.sign(v[2])
        elif kind==int(G.mjGEOM_CYLINDER):p=np.r_[h[0]*v[:2]/max(np.linalg.norm(v[:2]),1e-15),h[1]*np.sign(v[2])]
        elif kind==int(G.mjGEOM_ELLIPSOID):p=h*h*v/max(np.linalg.norm(h*v),1e-15)
        else:raise ValueError('Unsupported bounded convex geom: '+self.names[i])
        return d.geom_xpos[i]+R@p

    def pair(self,a,b,lower=0.):
        key=(int(a),int(b));direction=self.warm.get(key,self.d.geom_xpos[a]-self.d.geom_xpos[b])
        def oracle(v):return self.support(a,v),self.support(b,-v)
        result=convex_distance(oracle,direction,lower,self.tolerance)
        result.update(geom_a=int(a),geom_b=int(b),name_a=self.names[a],name_b=self.names[b])
        self.warm[key]=result['a']-result['b'];self.refinements+=1
        if self.strict and not result['converged']:
            raise ArithmeticError('Unresolved convex distance '+str((self.names[a],self.names[b],result['lower'],result['upper'],result['status'])))
        return result

    def bounds(self):
        R=self.d.geom_xmat.reshape(-1,3,3)
        center=self.d.geom_xpos+(R@self.local_aabb[:,:3,None])[:,:,0]
        half=(abs(R)@self.local_aabb[:,3:,None])[:,:,0]
        return center-half,center+half

    def _scan(self,key,pairs,lo,hi):
        if not len(pairs):return dict(distance=np.inf,lower=np.inf,upper=np.inf,a=np.full(3,np.nan),b=np.full(3,np.nan),geom_a=-1,geom_b=-1,converged=True)
        aa,bb=pairs.T;delta=np.maximum(np.maximum(lo[aa]-hi[bb],lo[bb]-hi[aa]),0.)
        broad=np.linalg.norm(delta,axis=1);order=np.argsort(broad)
        previous=self.last_pair.get(key)
        if previous is not None:
            matches=np.flatnonzero((aa==previous[0])&(bb==previous[1]))
            if len(matches):order=np.r_[matches[0],order[order!=matches[0]]]
        best=None;minimum_lower=np.inf;count=0
        for k in order:
            lb=float(broad[k])
            if best is not None:
                # A zero AABB distance is not a signed lower bound during overlap.
                if lb>0 and lb>=best['upper']:continue
            result=self.pair(int(aa[k]),int(bb[k]),lb);count+=1
            minimum_lower=min(minimum_lower,result['lower'])
            if best is None or result['upper']<best['upper']:best=result
        best=dict(best);best['lower']=min(minimum_lower,best['upper'])
        best['distance']=.5*(best['lower']+best['upper']);best['pairs_refined']=count
        self.last_pair[key]=(best['geom_a'],best['geom_b'])
        return best

    def score(self,qpos):
        self.d.qpos[:]=np.asarray(qpos,float);self.mj.mj_kinematics(self.m,self.d)
        lo,hi=self.bounds();before=self.refinements
        results={k:self._scan(k,p,lo,hi) for k,p in self.external.items()}
        results['self']=self._scan('self',self.self_pairs,lo,hi)
        arm=results['arm_tool'];body=results['body'];whole=arm if arm['distance']<=body['distance'] else body
        self.calls+=1
        return dict(body_m=body['distance'],arm_tool_m=arm['distance'],self_m=results['self']['distance'],
            self_mm=1000*results['self']['distance'],near_robot=whole['a'].copy(),near_gate=whole['b'].copy(),
            arm_near_robot=arm['a'].copy(),arm_near_gate=arm['b'].copy(),
            body_near_robot=body['a'].copy(),body_near_gate=body['b'].copy(),
            self_near_a=results['self']['a'].copy(),self_near_b=results['self']['b'].copy(),
            details=results,pairs_refined=self.refinements-before)

    def audit_log(self,physics_path,run_path,output,step=1):
        with np.load(physics_path,allow_pickle=False) as f:
            qpos=f['qpos'][::step];us=f['us'][::step]
        release=json.loads(Path(run_path).read_text())['release_us'];t=(us-release)/1e6
        rows=[];timings=[]
        for q in qpos:
            before=time.perf_counter();rows.append(self.score(q));timings.append(time.perf_counter()-before)
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        fields=('body_m','arm_tool_m','self_m','self_mm','near_robot','near_gate','arm_near_robot','arm_near_gate',
                'body_near_robot','body_near_gate','self_near_a','self_near_b','pairs_refined')
        arrays={k:np.asarray([r[k] for r in rows]) for k in fields}
        for group in ('body','arm_tool','self'):
            for field in ('lower','upper','geom_a','geom_b','converged'):
                arrays[group+'_'+field]=np.asarray([r['details'][group][field] for r in rows])
        np.savez_compressed(output/'distance_trace.npz',t=t,**arrays)
        numeric=np.column_stack((t,arrays['body_m'],arrays['arm_tool_m'],arrays['self_mm'],arrays['near_robot'],arrays['near_gate'],arrays['arm_near_robot'],arrays['arm_near_gate']))
        np.savetxt(output/'distance_trace.csv',numeric,delimiter=',',header='t,body_m,arm_tool_m,self_mm,near_robot_x,near_robot_y,near_robot_z,near_gate_x,near_gate_y,near_gate_z,arm_near_robot_x,arm_near_robot_y,arm_near_robot_z,arm_near_gate_x,arm_near_gate_y,arm_near_gate_z',comments='')
        report=dict(tolerance_m=self.tolerance,frames=len(rows),sampling_step=int(step),
            adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            physics_sha256=hashlib.sha256(Path(physics_path).read_bytes()).hexdigest(),
            mean_ms_per_frame=1000*float(np.mean(timings)),p95_ms_per_frame=1000*float(np.quantile(timings,.95)),
            mean_pairs_refined=float(np.mean(arrays['pairs_refined'])),registry=self.registry,
            minima={k:float(np.min(arrays[k])) for k in ('body_m','arm_tool_m','self_mm')},
            scope='True integrated qpos on private kinematic data; no estimator or control input is read. '
                'Every selected recorded pose is scored; default step=1. Distance bounds are stored per frame.')
        (output/'distance_audit.json').write_text(json.dumps(report,indent=2)+'\n')
        return report


def analytic_tests(include_epa=False):
    tests=[]
    def run(name,A,B,truth):
        got=convex_distance(lambda v:(A(v),B(-v)),np.array([1.,.3,.2]),tolerance=2e-5)
        if not got['lower']-1e-8<=truth<=got['upper']+1e-8 or not got['converged']:
            raise AssertionError((name,truth,got))
        tests.append(dict(name=name,truth=truth,lower=got['lower'],upper=got['upper'],iterations=got['iterations']))
    sphere=lambda c,r:lambda v:np.asarray(c)+r*v/max(np.linalg.norm(v),1e-15)
    box=lambda c,h:lambda v:np.asarray(c)+np.asarray(h)*np.sign(v)
    run('sphere separated',sphere([0,0,0],.2),sphere([1,0,0],.3),.5)
    run('axis boxes separated',box([0,0,0],[.2,.3,.4]),box([1,.2,0],[.1,.2,.3]),.7)
    run('sphere diagonal box',sphere([1,1,1],.1),box([0,0,0],[.2,.2,.2]),np.sqrt(3)*.8-.1)
    # Planar box face gives an independent analytic support-plane distance for
    # an arbitrarily rotated convex hull, including the witness endpoint test.
    angle=.4;R=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
    verts=np.asarray(list(itertools.product((-.2,.2),(-.3,.3),(-.1,.1))))@R.T
    mesh=lambda v:verts[np.argmax(verts@v)]
    run('rotated convex to slab',mesh,box([0,0,1.],[5,5,.2]),.7)
    capsule=lambda c,r,h:lambda v:np.asarray(c)+r*v/max(np.linalg.norm(v),1e-15)+np.array([0.,0.,h*np.sign(v[2])])
    run('capsule axial separation',capsule([0,0,0],.1,.3),capsule([0,0,1.],.1,.2),.3)
    if include_epa:
        run('overlapping axis boxes',box([0,0,0],[.3,.2,.2]),box([.5,0,0],[.3,.2,.2]),-.1)
        run('overlapping spheres',sphere([0,0,0],.2),sphere([.4,0,0],.3),-.1)
    return tests


def platform_tests():
    """Dedicated offline model only; optional native-CCD flag is restored afterward."""
    import mujoco
    from round_g1_model import Geometry,make_plant
    from round_g1_distance import Solids
    result=dict(analytic=analytic_tests(True));plant=make_plant();g=Geometry(plant);old=Solids(g)
    before=time.perf_counter();audit=DistanceAudit(plant.m);result['build_seconds']=time.perf_counter()-before
    g.place([1.,0.,2.15],75.);audit.d.qpos[:]=g.d.qpos;mujoco.mj_kinematics(audit.m,audit.d)
    i=plant.m.geom('collision_landing_163').id;j=plant.m.geom('G1_sill').id
    ours=audit.pair(i,j);legacy=old.pair(i,0,0.)
    if not ours['lower']-2e-7<=.16377066<=ours['upper']+2e-7:raise AssertionError(('known GJK regression',ours))
    if abs(ours['distance']-.5*(legacy[0]+legacy[1]))>2*audit.tolerance:raise AssertionError('Solids known-pair mismatch')
    witness=np.zeros(6);mpr=mujoco.mj_geomDistance(plant.m,audit.d,i,j,10.,witness)
    probe=dict(independent_lower=ours['lower'],independent_upper=ours['upper'],old_solids=[legacy[0],legacy[1]],
        default_sdk_distance=float(mpr),default_sdk_witness_norm=float(np.linalg.norm(witness[:3]-witness[3:])))
    native=getattr(mujoco.mjtEnableBit,'mjENBL_NATIVECCD',None)
    if native is not None:
        saved=int(plant.m.opt.enableflags)
        try:
            plant.m.opt.enableflags=saved|int(native)
            value=mujoco.mj_geomDistance(plant.m,audit.d,i,j,10.,witness)
            probe.update(native_ccd_distance=float(value),native_ccd_witness_norm=float(np.linalg.norm(witness[:3]-witness[3:])),
                         native_matches=bool(ours['lower']-audit.tolerance<=value<=ours['upper']+audit.tolerance))
        finally:plant.m.opt.enableflags=saved
    result['reported_libccd_counterexample']=probe
    cross=[];speeds=[]
    for pos,theta in [([1.,0.,2.15],75.),([1.25,.03,2.12],50.),([1.60,-.04,2.20],85.)]:
        g.place(pos,theta);audit.d.qpos[:]=g.d.qpos;mujoco.mj_kinematics(audit.m,audit.d)
        for name in ('collision_central_equipment','collision_landing_163','visual_gripper_finger1'):
            if name not in audit.names:continue
            a=plant.m.geom(name).id
            for k,gate in enumerate(g.gates):
                b=plant.m.geom('G1_'+gate[0]).id;z=audit.pair(a,b)
                if z['lower']<0:continue  # Old Solids defines unsigned separated distance only.
                lo,hi,_,_=old.pair(a,k,0.)
                if abs(z['distance']-.5*(lo+hi))>2*audit.tolerance+1.5e-4:raise AssertionError((name,gate[0],z,lo,hi))
                cross.append(dict(theta=theta,geom=name,gate=gate[0],new=[z['lower'],z['upper']],old=[lo,hi]))
        before=time.perf_counter();frame=audit.score(g.d.qpos);speeds.append(time.perf_counter()-before)
    result.update(old_solids_cross_check=cross,registry=audit.registry,
        mean_score_ms=1000*float(np.mean(speeds)),max_score_ms=1000*float(np.max(speeds)))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--factory',default='round_g1_model:make_plant')
    p.add_argument('--physics');p.add_argument('--run');p.add_argument('--output',required=True);p.add_argument('--step',type=int,default=1)
    p.add_argument('--self-test',action='store_true');p.add_argument('--platform-test',action='store_true');args=p.parse_args()
    if args.platform_test:
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        report=platform_tests();Path(args.output).write_text(json.dumps(report,indent=2));print(report,flush=True);return
    if args.self_test:
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        Path(args.output).write_text(json.dumps(dict(analytic=analytic_tests()),indent=2));return
    module,name=args.factory.split(':');plant=getattr(importlib.import_module(module),name)()
    print(DistanceAudit(plant.m).audit_log(args.physics,args.run,args.output,args.step),flush=True)

if __name__=='__main__':main()
