"""Causal clipped-map queries and private robot-only kinematics for M2.

Only RangeMap reads environmental model coordinates. A planning Geometry accepts
an immutable MapSnapshot and cannot enumerate the scene's COL_* table.
"""
from dataclasses import dataclass
from functools import lru_cache
from types import SimpleNamespace
import hashlib,itertools,json,math,time,ctypes,subprocess,tempfile,os
from pathlib import Path
import mujoco,numpy as np
from model import SimArmBridge
from m1_geometry import library
from super_planning_geometry import sampled_times,wrap
from gate_distance_audit import DistanceAudit
from gate_distance_certificate import CertifiedDistance
from m2_self_registry import partition

SIGNS=np.array(list(itertools.product((-1.,1.),repeat=3)))

def map_batch_library():
    source=Path(__file__).with_name('m2_map.cc');folder=Path(tempfile.gettempdir())/('m2_map_'+hashlib.sha256(source.read_bytes()).hexdigest()[:14])
    folder.mkdir(exist_ok=True);lib=folder/'map.so'
    if not lib.exists():
        tmp=folder/('lib-%d.so'%os.getpid());subprocess.run(['c++','-O3','-std=c++17','-fPIC','-shared',str(source),'-o',str(tmp)],check=True);tmp.replace(lib)
    f=ctypes.CDLL(str(lib)).map_batch;fp=np.ctypeslib.ndpointer(np.float64,flags='C_CONTIGUOUS')
    f.argtypes=[ctypes.c_int,fp,ctypes.c_int,fp,ctypes.c_int,fp,fp];f.restype=None;return f

@dataclass(frozen=True)
class MapSnapshot:
    version:int
    stamp:float
    origin:tuple
    radius:float
    obstacle_rows:np.ndarray
    ids:tuple
    known_frontier:float
    map_hash:str
    cells:tuple=()
    def metadata(self):
        return dict(version=self.version,stamp=self.stamp,origin=list(self.origin),radius=self.radius,
                    known_frontier=self.known_frontier,map_hash=self.map_hash,ids=list(self.ids),
                    clipping='Finite-radius conservative voxel intersections; unknown region blocked',
                    row_count=len(self.obstacle_rows))

class RangeMap:
    """Equivalent causal range sensor; full scene is private to this sensor.

    Objects crossing the sensed ball are voxel-clipped. A voxel is observed only
    when all its corners lie in range. Conservative occupied boxes may enlarge
    the actual obstacle locally, never remove it in the certified inner ball.
    """
    def __init__(self,plant,radius=2.4,cell=.12):
        m=plant.m;d=mujoco.MjData(m);mujoco.mj_kinematics(m,d)
        self._objects=[];self.radius=float(radius);self.cell=float(cell);self.version=0
        for i in range(m.ngeom):
            name=m.geom(i).name or ''
            if not name.startswith('COL_'):continue
            R=d.geom_xmat[i].reshape(3,3);ctr=d.geom_xpos[i]+R@m.geom_aabb[i,:3]
            half=abs(R)@m.geom_aabb[i,3:];kind=m.geom_type[i];sz=m.geom_size[i]
            tube=kind in (mujoco.mjtGeom.mjGEOM_CAPSULE,mujoco.mjtGeom.mjGEOM_CYLINDER)
            row=np.r_[ctr,half,R[:,2],sz[0] if tube else 0.,sz[1]+(sz[0] if kind==mujoco.mjtGeom.mjGEOM_CAPSULE else 0.) if tube else 0.]
            self._objects.append((name,row))
    def reveal(self,t,state):
        origin=np.asarray(state[:3],float).copy();rows=[];ids=[]
        # Every accepted voxel is inside the ball. All intersecting voxels cover
        # the smaller certified ball R - sqrt(3)*cell, including its obstacles.
        radius=self.radius;inner=radius-math.sqrt(3)*self.cell
        for name,row in self._objects:
            c,h=row[:3],row[3:6]
            if np.linalg.norm(np.maximum(abs(c-origin)-h,0.))>radius:continue
            if np.max(np.linalg.norm(c+SIGNS*h-origin,axis=1))<=radius:
                rows.append(row);ids.append(name);continue
            stack=[(c.copy(),h.copy())]
            while stack:
                cc,hh=stack.pop()
                if np.linalg.norm(np.maximum(abs(cc-origin)-hh,0.))>radius:continue
                far=np.max(np.linalg.norm(cc+SIGNS*hh-origin,axis=1))
                if far<=radius:
                    rows.append(np.r_[cc,hh,0.,0.,1.,0.,0.]);ids.append(name);continue
                axis=int(np.argmax(hh))
                if 2*hh[axis]<=self.cell:continue
                nh=hh.copy();nh[axis]*=.5;off=np.zeros(3);off[axis]=nh[axis]
                stack.extend(((cc-off,nh),(cc+off,nh)))
        a=np.ascontiguousarray(rows,dtype=float).reshape(-1,11);a.setflags(write=False)
        self.version+=1;digest=hashlib.sha256(a.tobytes()+origin.tobytes()+str(self.version).encode()).hexdigest()
        return MapSnapshot(self.version,float(t),tuple(origin),inner,a,tuple(ids),float(origin[0]+inner),digest)

def original_pairs(m):
    physical=[i for i in range(m.ngeom) if m.geom_bodyid[i] and (int(m.geom_contype[i])&1) and (int(m.geom_conaffinity[i])&1)]
    excluded=set(map(int,m.exclude_signature));explicit={tuple(sorted((int(a),int(b)))) for a,b in zip(m.pair_geom1,m.pair_geom2)}
    parent_filter=not int(m.opt.disableflags)&int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)
    pairs=[]
    for a,b in itertools.combinations(physical,2):
        ba,bb=int(m.geom_bodyid[a]),int(m.geom_bodyid[b]);wa,wb=int(m.body_weldid[ba]),int(m.body_weldid[bb])
        if wa==wb:continue
        if (a,b) in explicit:pairs.append((a,b));continue
        pa=int(m.body_weldid[m.body_parentid[wa]]);pb=int(m.body_weldid[m.body_parentid[wb]])
        if parent_filter and wa and wb and (wa==pb or wb==pa):continue
        if (min(ba,bb)<<16)+max(ba,bb) in excluded:continue
        if not ((int(m.geom_contype[a])&int(m.geom_conaffinity[b])) or (int(m.geom_contype[b])&int(m.geom_conaffinity[a]))):continue
        pairs.append((a,b))
    return np.asarray(pairs,int).reshape(-1,2)

def self_auditor(m):
    """Reuse the certified GJK/support implementation without its old G1 scene."""
    a=CertifiedDistance.__new__(CertifiedDistance);a.m=m;a.d=mujoco.MjData(m);a.mj=mujoco
    a.tolerance=1e-5;a.strict=True;a.mesh={};a.warm={};a.last_pair={};a.calls=0;a.refinements=0;a.records=[]
    a.names=[m.geom(i).name or 'geom%d'%i for i in range(m.ngeom)]
    a.local_aabb=m.geom_aabb.copy();a.self_pairs=original_pairs(m);a.hazard,a.fit=partition(m,a.self_pairs)
    for i in np.unique(a.self_pairs):
        if m.geom_type[i]==mujoco.mjtGeom.mjGEOM_MESH:
            mid=int(m.geom_dataid[i]);start=int(m.mesh_vertadr[mid]);n=int(m.mesh_vertnum[mid])
            a.mesh[int(i)]=m.mesh_vert[start:start+n].astype(float)
    return a

class NativeSelf:
    """Same convex support geometry as the independent Python audit, in C++."""
    def __init__(self,m):
        source=Path(__file__).with_name('m2_self.cc');digest=hashlib.sha256(source.read_bytes()).hexdigest()[:14]
        folder=Path(tempfile.gettempdir())/('m2_self_'+digest);folder.mkdir(exist_ok=True);lib=folder/'self.so'
        if not lib.exists():
            tmp=folder/('lib-%d.so'%os.getpid());subprocess.run(['c++','-O3','-std=c++17','-fPIC','-shared',str(source),'-o',str(tmp)],check=True);tmp.replace(lib)
        self.library=ctypes.CDLL(str(lib));self.fn=self.library.self_clearance
        fp=np.ctypeslib.ndpointer(np.float64,flags='C_CONTIGUOUS');ip=np.ctypeslib.ndpointer(np.int32,flags='C_CONTIGUOUS')
        self.fn.argtypes=[ctypes.c_int,ip,fp,fp,fp,ip,ip,fp,ctypes.c_int,ip,ctypes.c_double,fp,ip];self.fn.restype=None
        self.external_fn=self.library.external_robust_clearance
        self.external_fn.argtypes=[ctypes.c_int,ip,fp,fp,fp,ip,ip,fp,ctypes.c_int,ip,ctypes.c_double,ctypes.c_int,ctypes.c_double,fp,fp,ip]
        self.external_fn.restype=None
        self.m=m;self.types=np.ascontiguousarray(m.geom_type,np.int32);self.size=np.ascontiguousarray(m.geom_size);self.aabb=np.ascontiguousarray(m.geom_aabb)
        self.offset=np.zeros(m.ngeom,np.int32);self.count=np.zeros(m.ngeom,np.int32);vertices=[];total=0
        for i in range(m.ngeom):
            if m.geom_type[i]!=mujoco.mjtGeom.mjGEOM_MESH:continue
            if not m.geom_bodyid[i] and not (m.geom(i).name or '').startswith('COL_'):continue
            mid=int(m.geom_dataid[i]);start=int(m.mesh_vertadr[mid]);n=int(m.mesh_vertnum[mid])
            v=m.mesh_vert[start:start+n].astype(float)
            if len(v)>8:
                from scipy.spatial import ConvexHull
                v=v[ConvexHull(v).vertices]
            self.offset[i]=total;self.count[i]=len(v);vertices.extend(v);total+=len(v)
        self.vertices=np.ascontiguousarray(vertices,float).reshape(-1,3)
    def query(self,d,pairs):
        pose=np.ascontiguousarray(np.column_stack((d.geom_xpos,d.geom_xmat)));pairs=np.ascontiguousarray(pairs,np.int32)
        out=np.zeros(4);which=np.zeros(2,np.int32)
        self.fn(self.m.ngeom,self.types,self.size,pose,self.aabb,self.offset,self.count,self.vertices,len(pairs),pairs,1e-5,out,which)
        return dict(lower=float(out[0]),upper=float(out[1]),pairs=int(out[2]),iterations=int(out[3]),which=which.tolist())

class ConvexMap:
    """Only robot geometry plus primitives explicitly contained in a snapshot.

    Robust support uses each vertex's horizontal-axis rotational ellipse plus
    its certified second-order Rodrigues remainder ball. This covers the whole
    roll/pitch disc without adding yaw uncertainty. Cylinders use a containing
    32-gon. The existing isotropic bounds remain valid independent certificates.
    """
    def __init__(self,geometry,snapshot):
        g=geometry;n=g.native_self;self.g=g;self.nr=len(g.ids);self.ns=len(snapshot.obstacle_rows);self.count=self.nr+self.ns
        self.types=np.r_[n.types[g.ids],np.where(snapshot.obstacle_rows[:,9]>0,5,6)].astype(np.int32)
        self.sizes=np.zeros((self.count,3));self.sizes[:self.nr]=n.size[g.ids]
        self.boxes=np.zeros((self.count,6));self.boxes[:self.nr]=n.aabb[g.ids]
        self.offset=np.r_[n.offset[g.ids],np.zeros(self.ns,dtype=np.int32)].astype(np.int32)
        self.nvert=np.r_[n.count[g.ids],np.zeros(self.ns,dtype=np.int32)].astype(np.int32)
        self.obpose=np.zeros((self.ns,12))
        for k,row in enumerate(snapshot.obstacle_rows):
            j=self.nr+k;self.obpose[k,:3]=row[:3]
            if row[9]>0:
                z=row[6:9]/np.linalg.norm(row[6:9]);x=np.cross([1,0,0] if abs(z[0])<.8 else [0,1,0],z);x/=np.linalg.norm(x);y=np.cross(z,x)
                self.obpose[k,3:]=np.column_stack((x,y,z)).ravel();self.sizes[j]=[row[9],row[10],0];self.boxes[j,3:]=[row[9],row[9],row[10]]
            else:self.obpose[k,3:]=np.eye(3).ravel();self.sizes[j]=row[3:6];self.boxes[j,3:]=row[3:6]
    def query(self,state,groups=(0,1),tilt=None):
        g=self.g;m=g.m;s=np.asarray(state,float);g.place(g.w,s);mujoco.mj_kinematics(m,g.w)
        pose=np.ascontiguousarray(np.vstack((np.column_stack((g.w.geom_xpos[g.ids],g.w.geom_xmat[g.ids])),self.obpose)))
        obs=g.snapshot.obstacle_rows;comp=g.components(g.w,s[:3]);ctr=comp[:,:3]+s[:3];half=comp[:,3:6]+comp[:,6,None]
        lo=np.min(ctr-half,axis=0);hi=np.max(ctr+half,axis=0)
        sep=np.maximum(obs[:,:3]-obs[:,3:6]-hi,lo-obs[:,:3]-obs[:,3:6]);keep=np.flatnonzero(np.max(sep,axis=1)<=.30)
        answer={}
        for group in groups:
            ids=np.flatnonzero(g.groups if group==0 else ~g.groups)
            if not len(keep):answer[group]=dict(lower=.30,upper=.30,pair=[None,None],support_queries=0);continue
            pairs=np.ascontiguousarray(np.column_stack((np.repeat(ids,len(keep)),np.tile(keep+self.nr,len(ids)))),np.int32)
            out=np.zeros(4);which=np.zeros(2,np.int32)
            g.native_self.external_fn(self.count,self.types,self.sizes,pose,self.boxes,self.offset,self.nvert,
                  g.native_self.vertices,len(pairs),pairs,1e-5,self.nr,g.limits.tilt_max if tilt is None else float(tilt),
                  np.ascontiguousarray(s[:3]),out,which)
            answer[group]=dict(lower=float(out[0]),upper=float(out[1]),pair=[g.names[g.ids[which[0]]],g.obnames[which[1]-self.nr]],
                               support_queries=int(out[2]),iterations=int(out[3]))
        return answer

class Geometry:
    """Shared immutable native model, private FK/data/caches per owner/thread."""
    def __init__(self,plant,limits=None,snapshot=None):
        self.m=m=plant.m;self.limits=limits or SimpleNamespace(body_margin=.085,arm_margin=.085,tilt_max=0.)
        self.fk=mujoco.MjData(m);self.w=mujoco.MjData(m);self.bridge=SimArmBridge()
        self.u=m.body('uav').id;self.root=int(m.jnt_qposadr[m.body_jntadr[self.u]])
        self.names=[m.geom(i).name or 'geom%d'%i for i in range(m.ngeom)]
        self.ids=np.array([i for i in range(m.ngeom) if m.geom_bodyid[i] and
          (m.geom_contype[i] or m.geom_conaffinity[i] or (m.geom_rgba[i,3]>.001 and m.geom_group[i] in (1,2)))],int)
        self.groups=np.isin(m.geom_bodyid[self.ids],[self.u,m.body('mount').id]);self.ab=m.geom_aabb[self.ids].copy()
        self.body=self.ids[self.groups];self.tool=self.ids[~self.groups];self.narrow=library()
        self.batch=map_batch_library()
        self.audit=self_auditor(m);self.native_self=NativeSelf(m);self.plan_self_margin=.015;self.theta_max=np.deg2rad(118)
        self.deadline=None;self.calls=dict(search=0,final=0,seed_edges=0);self.snapshot=None
        self.tight_search=False
        self.local=lru_cache(maxsize=4096)(self._local);self._self=lru_cache(maxsize=4096)(self._self_uncached)
        self.rigid_fit={}
        if snapshot is not None:self.set_snapshot(snapshot)
    def set_snapshot(self,snapshot):
        if not isinstance(snapshot,MapSnapshot):raise TypeError('Geometry requires immutable M2 MapSnapshot')
        self.snapshot=snapshot;self.obs=snapshot.obstacle_rows;self.map_hash=snapshot.map_hash;self.obnames=snapshot.ids
        self.convex_map=ConvexMap(self,snapshot)
    def check_deadline(self):
        if self.deadline is not None and time.monotonic()>self.deadline:raise TimeoutError('M2 geometry time budget')
    def place(self,d,state):
        s=np.asarray(state);d.qpos[self.root:self.root+3]=s[:3];d.qpos[self.root+3:self.root+7]=[np.cos(s[3]/2),0,0,np.sin(s[3]/2)]
        self.bridge.set_kinematic_shape(self.m,d,float(s[4]),float(s[5]))
    def components(self,d,offset):
        mat=d.geom_xmat[self.ids].reshape(-1,3,3)
        ctr=d.geom_xpos[self.ids]-offset+np.einsum('nij,nj->ni',mat,self.ab[:,:3])
        h=np.einsum('nij,nj->ni',abs(mat),self.ab[:,3:])
        reserve=2*math.sin(self.limits.tilt_max/2)*(np.linalg.norm(ctr,axis=1)+np.linalg.norm(h,axis=1))
        sphere=np.where(self.m.geom_type[self.ids]==mujoco.mjtGeom.mjGEOM_SPHERE,self.m.geom_size[self.ids,0],0.)
        upright=(self.m.geom_type[self.ids]==mujoco.mjtGeom.mjGEOM_CYLINDER)&(abs(mat[:,2,2])>1-1e-10)
        sphere=np.where(upright,-self.m.geom_size[self.ids,0],sphere)
        return np.ascontiguousarray(np.column_stack((ctr,h,reserve,~self.groups,self.ids,sphere)))
    def _local(self,theta,phi):
        self.place(self.fk,[0,0,0,0,theta,phi]);mujoco.mj_kinematics(self.m,self.fk)
        a=self.components(self.fk,np.zeros(3))
        bodyids=np.flatnonzero(self.m.body_mass>0)
        com=(self.m.body_mass[bodyids,None]*self.fk.xipos[bodyids]).sum(axis=0)/self.m.body_mass[bodyids].sum()
        return a,None,com
    def _self_uncached(self,theta,phi):
        a=self.audit;self.place(a.d,[0,0,0,0,theta,phi]);mujoco.mj_kinematics(self.m,a.d)
        haz=self.native_self.query(a.d,a.hazard);fit=[]
        for i,j in a.fit:
            key=(int(i),int(j));rigid=all('collision_gripper_finger' in a.names[k] for k in key)
            if rigid and key in self.rigid_fit:r=self.rigid_fit[key]
            else:
                r=self.native_self.query(a.d,np.array([[i,j]]))
                if rigid:self.rigid_fit[key]=r
            fit.append(dict(pair=[a.names[i],a.names[j]],distance=float(r['lower'])))
        assembly=min(x['distance'] for x in fit)
        return dict(hazardous=float(haz['lower']),registered=min(float(haz['lower']),assembly),assembly_min=assembly,
                    assembly_pair_distances=fit,hazard_pair=[a.names[k] for k in haz['which']],
                    certificate_tolerance_m=a.tolerance)
    def self_report(self,theta,phi):
        return self._self(float(theta),wrap(float(phi)) if theta>1e-10 else 0.)
    def query(self,components,s):
        if self.snapshot is None:raise RuntimeError('No causal map snapshot')
        # A certified 0.30 m cap permits a union-AABB broad phase. Every omitted
        # box lies farther than this cap from the complete robot bound, including
        # the tilt reserve. This changes no clearance decision at 85 mm.
        cap=.30;c,sn=np.cos(s[3]),np.sin(s[3]);R=np.array([[c,-sn,0],[sn,c,0],[0,0,1.]])
        ctr=components[:,:3]@R.T+np.asarray(s[:3]);half=components[:,3:6]@abs(R).T
        upright=components[:,9]<0;half[upright,:2]=-components[upright,9,None]
        spheres=components[:,9]>0;half[spheres]=components[spheres,9,None]
        lo=np.min(ctr-half-components[:,6,None],axis=0);hi=np.max(ctr+half+components[:,6,None],axis=0)
        sep=np.maximum(self.obs[:,:3]-self.obs[:,3:6]-hi,lo-self.obs[:,:3]-self.obs[:,3:6])
        keep=np.flatnonzero(np.max(sep,axis=1)<=cap)
        out=np.zeros(2);idx=np.zeros(4,np.int32)
        if not len(keep):return np.full(2,cap),idx
        subset=np.ascontiguousarray(self.obs[keep])
        transformed=np.ascontiguousarray(np.column_stack((ctr,half,components[:,6:])))
        self.narrow(len(components),transformed,len(subset),subset,np.zeros(6),out,idx)
        idx[1]=keep[idx[1]];idx[3]=keep[idx[3]];out=np.minimum(out,cap)
        return out,idx
    def known(self,components,s):
        c,sn=np.cos(s[3]),np.sin(s[3]);R=np.array([[c,-sn,0],[sn,c,0],[0,0,1.]])
        centers=components[:,:3]@R.T+s[:3];half=components[:,3:6]@abs(R).T
        upright=components[:,9]<0;half[upright,:2]=-components[upright,9,None]
        spheres=components[:,9]>0;half[spheres]=components[spheres,9,None]
        far=np.linalg.norm(abs(centers-np.asarray(self.snapshot.origin))+half,axis=1)+components[:,6]
        return bool(np.max(far)<=self.snapshot.radius+1e-10)
    def query_batch(self,components,states):
        states=np.ascontiguousarray(states,float);out=np.zeros((len(states),2))
        self.batch(len(components),components,len(self.obs),self.obs,len(states),states,out);return out
    def state(self,state,exact=False,check_self=True,tight=False):
        self.check_deadline();self.calls['final' if exact else 'search']+=1;s=np.asarray(state,float)
        a,_,_=self.local(float(s[4]),wrap(float(s[5])) if s[4]>1e-10 else 0.)
        dist,idx=self.query(a,s);known=self.known(a,s);tight_result={}
        groups=[i for i in (0,1) if dist[i]<self.limits.body_margin+.012 and (exact or tight or self.tight_search)]
        if groups:
            tight_result=self.convex_map.query(s,groups)
            for i in groups:dist[i]=max(dist[i],tight_result[i]['lower'])
        sr=self.self_report(s[4],s[5]) if check_self else None
        bad='unknown_space' if not known else 'command_range' if not -1e-9<=s[4]<=self.theta_max+1e-9 else 'self_collision' if sr and (sr['hazardous']<self.plan_self_margin or sr['assembly_min']<0) else 'body_environment' if dist[0]<self.limits.body_margin else 'arm_environment' if dist[1]<self.limits.arm_margin else None
        pairs=[[self.names[idx[i]],self.obnames[idx[i+1]]] if len(self.obs) else [None,None] for i in (0,2)]
        for i,r in tight_result.items():pairs[i]=r['pair']
        return dict(reason=bad,body=float(dist[0]),arm=float(dist[1]),raw_body=float(dist[0]),raw_arm=float(dist[1]),
                    body_pair=pairs[0],arm_pair=pairs[1],pair=pairs[int(np.argmin(dist))],self_contact=sr,known=known,convex_tightening=tight_result)
    def search_clearances(self,s):
        r=self.state(s);return r['body'],(-1. if r['reason'] in ('self_collision','command_range','unknown_space') else r['arm'])
    def physical_coupling(self,s):
        _,_,com=self.local(float(s[4]),wrap(float(s[5])) if s[4]>1e-10 else 0.)
        return dict(com_offset=float(np.linalg.norm(com[:2])),gravity_torque=float(self.m.body_subtreemass[self.u]*9.81*np.linalg.norm(com[:2])))
    def robustness_diagnostic(self,s,tilt_deg=2.):
        return self.convex_map.query(s,groups=(0,1),tilt=np.deg2rad(tilt_deg))
    def edge(self,a,b):
        self.calls['seed_edges']+=1;a=np.asarray(a);delta=np.asarray(b)-a;delta[3]=wrap(delta[3]);delta[5]=wrap(delta[5])
        n=max(1,int(np.ceil(max(np.linalg.norm(delta[:3])/.055,abs(delta[3])/.1,abs(delta[4])/.14,abs(delta[5])/.18))))
        self.last_edge_failure=None
        for u in np.linspace(0,1,n+1):
            state=a+u*delta;r=self.state(state,tight=True)
            if r['reason']:
                self.last_edge_failure=dict(fraction=float(u),state=state.tolist(),reason=r['reason'],body=r['body'],arm=r['arm'])
                return False
        return True
    def proof(self,spline,dt=.04,dp=.01,da=np.deg2rad(.5)):
        ts=sampled_times(spline,dt,dp,da);states=spline.evaluate(ts,0);first=None;body=arm=np.inf;bp=ap=None;count=0;selfmin=np.inf
        for i,(t,s) in enumerate(zip(ts,states)):
            r=self.state(s,True);count+=1
            if r['body']<body:body=r['body'];bp=r['body_pair']
            if r['arm']<arm:arm=r['arm'];ap=r['arm_pair']
            selfmin=min(selfmin,r['self_contact']['hazardous'])
            if r['reason']:first=dict(index=i,time_s=float(t),state=s.tolist(),**r);break
        return dict(accepted=first is None,first=first,samples=count,body_min=body,arm_min=arm,body_pair=bp,arm_pair=ap,
                    self_min=selfmin,self_gap_m=self.plan_self_margin,map_hash=self.map_hash,max_time_step_s=dt,
                    translation_step_m=dp,angular_step_deg=float(np.rad2deg(da)),tilt_bound_rad=self.limits.tilt_max,
                    acceptance='nominal 85 mm CAD clearance; roll/pitch uncertainty is separately diagnosed')

geometry_factory=Geometry
