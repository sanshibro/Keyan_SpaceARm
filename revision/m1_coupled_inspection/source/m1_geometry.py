"""Loaded SUPER geometry for the gallery, shared by the 6D planner and audit."""
from collections import OrderedDict
import ctypes,hashlib,math,subprocess,tempfile
from functools import lru_cache
from pathlib import Path
import mujoco,numpy as np
from model import SimArmBridge
from super_planning_geometry import SuperGeometry,COMMAND_THETA_MAX,SELF_GAP,wrap
from m1_plant import build
from m1_layout import blocks

def library():
    source=Path(__file__).with_name('m1_clearance.cc')
    folder=Path(tempfile.gettempdir())/('m1_geom_'+hashlib.sha256(source.read_bytes()).hexdigest()[:14])
    folder.mkdir(exist_ok=True);so=folder/'clearance.so'
    if not so.exists():
        import os
        tmp=folder/('lib-%d.so'%os.getpid())
        subprocess.run(['c++','-O3','-std=c++17','-fPIC','-shared',str(source),'-o',str(tmp)],check=True);tmp.replace(so)
    fn=ctypes.CDLL(str(so)).gallery_clearance
    fp=np.ctypeslib.ndpointer(np.float64,flags='C_CONTIGUOUS');ip=np.ctypeslib.ndpointer(np.int32,flags='C_CONTIGUOUS')
    fn.argtypes=[ctypes.c_int,fp,ctypes.c_int,fp,fp,fp,ip];fn.restype=None
    return fn

class GalleryGeometry(SuperGeometry):
    def __init__(self,plant,limits):
        self.map_hash=plant.scene_digest;self.limits=limits
        self.m=m=mujoco.MjModel.from_xml_string(plant.xml)
        self.fk=mujoco.MjData(m);self.w=mujoco.MjData(m);mujoco.mj_forward(m,self.w)
        self.bridge=SimArmBridge();self.u=m.body('uav').id;self.root=int(m.jnt_qposadr[m.body_jntadr[self.u]])
        self.names=[m.geom(i).name or 'geom_%d'%i for i in range(m.ngeom)]
        self.ids=np.array([i for i in range(m.ngeom) if m.geom_bodyid[i] and (m.geom_contype[i] or
              m.geom_conaffinity[i] or (m.geom_rgba[i,3]>.001 and m.geom_group[i] in (1,2)))])
        self.groups=np.isin(m.geom_bodyid[self.ids],[self.u,m.body('mount').id])
        self.body=self.ids[self.groups];self.tool=self.ids[~self.groups]
        self.ab=m.geom_aabb[self.ids].copy()
        self.eg=np.array([i for i,n in enumerate(self.names) if n.startswith('COL_')])
        self.obnames=[self.names[i][4:] for i in self.eg];self.obs=[]
        for i in self.eg:
            R=self.w.geom_xmat[i].reshape(3,3);ctr=self.w.geom_xpos[i];sz=m.geom_size[i];kind=m.geom_type[i]
            half=abs(R)@m.geom_aabb[i,3:];axis=R[:,2]
            tube=kind in (mujoco.mjtGeom.mjGEOM_CAPSULE,mujoco.mjtGeom.mjGEOM_CYLINDER)
            self.obs.append([*ctr,*half,*axis,sz[0] if tube else 0.,sz[1]+(sz[0] if kind==mujoco.mjtGeom.mjGEOM_CAPSULE else 0.) if tube else 0.])
        self.obs=np.asarray(self.obs);self.narrow=library()
        self.theta_max=COMMAND_THETA_MAX;self.deadline=None;self.calls=dict(search=0,final=0,seed_edges=0)
        m.geom_margin[self.ids]=SELF_GAP
        self.local=self._geometry_cache()
        self.com_lut={}
    def _geometry_cache(self):
        @lru_cache(maxsize=1024)
        def fk(theta,phi):
            self.place(self.fk,[-10,0,4,0,theta,phi])
            mujoco.mj_fwdPosition(self.m,self.fk)
            a=self.components(self.fk,[-10.,0,4.])
            hit=min(((float(c.dist),self.names[c.geom1],self.names[c.geom2]) for c in self.fk.contact[:self.fk.ncon]
                     if self.m.geom_bodyid[c.geom1] and self.m.geom_bodyid[c.geom2] and c.dist<=SELF_GAP),default=None)
            com=self.fk.subtree_com[self.u]-self.fk.xpos[self.u]
            return a,hit,com.copy()
        return fk
    def components(self,d,offset):
        mat=d.geom_xmat[self.ids].reshape(-1,3,3)
        ctr=d.geom_xpos[self.ids]-offset+np.einsum('nij,nj->ni',mat,self.ab[:,:3])
        h=np.einsum('nij,nj->ni',abs(mat),self.ab[:,3:])
        reserve=2*math.sin(self.limits.tilt_max/2)*(np.linalg.norm(ctr,axis=1)+np.linalg.norm(h,axis=1))
        sphere=np.where(self.m.geom_type[self.ids]==mujoco.mjtGeom.mjGEOM_SPHERE,self.m.geom_size[self.ids,0],0.)
        return np.ascontiguousarray(np.column_stack((ctr,h,reserve,~self.groups,self.ids,sphere)))
    def query(self,components,s):
        out=np.zeros(2);idx=np.zeros(4,np.int32)
        self.narrow(len(components),components,len(self.obs),self.obs,np.ascontiguousarray(s,dtype=float),out,idx)
        return out,idx
    def state(self,state,exact=False):
        self.check_deadline();self.calls['final' if exact else 'search']+=1
        s=np.asarray(state);a,b,c=self.local(float(s[4]),wrap(s[5]) if s[4]>1e-10 else 0.)
        if exact:
            self.place(self.w,s)
            mujoco.mj_kinematics(self.m,self.w)
            measured=self.components(self.w,s[:3])
            dist,idx=self.query(measured,np.r_[s[:3],0.,0.,0.])
        else:
            dist,idx=self.query(a,s)
        pairs=[[self.names[idx[i]],self.obnames[idx[i+1]]] for i in (0,2)]
        bad='self_collision' if b else 'command_range' if not -1e-9<=s[4]<=self.theta_max+1e-9 else 'body_environment' if dist[0]<self.limits.body_margin else 'arm_environment' if dist[1]<self.limits.arm_margin else None
        return dict(reason=bad,body=float(dist[0]),arm=float(dist[1]),body_pair=pairs[0],arm_pair=pairs[1],raw_body=float(dist[0]),raw_arm=float(dist[1]),pair=list(b[1:]) if b else pairs[int(np.argmin(dist))],self_contact=b)
    def measured(self,d):
        a=self.components(d,np.array([0.,0.,0.]));a[:,6]=0
        return self.query(a,np.zeros(6))
    def physical_coupling(self,s):
        _,_,com=self.local(float(s[4]),wrap(s[5]) if s[4]>1e-10 else 0.)
        return dict(com_offset=float(np.linalg.norm(com[:2])),gravity_torque=float(self.m.body_subtreemass[self.u]*9.81*np.linalg.norm(com[:2])))
