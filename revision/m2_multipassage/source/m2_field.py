"""Snapshot-only proposal distance field and robot-only surface samples."""
import ctypes
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import mujoco
from scipy.spatial import ConvexHull

@lru_cache(maxsize=1)
def library():
    cc=Path(__file__).with_suffix('.cc')
    lib=Path(tempfile.gettempdir())/('m2_field_'+hashlib.sha256(cc.read_bytes()).hexdigest()[:16]+'.so')
    if not lib.exists():
        tmp=lib.with_name(lib.stem+'-'+str(os.getpid())+'.so')
        subprocess.run(['c++','-O3','-std=c++17','-fopenmp','-march=native','-shared','-fPIC',str(cc),'-o',str(tmp)],check=True)
        tmp.replace(lib)
    dll=ctypes.CDLL(str(lib));dp=np.ctypeslib.ndpointer(np.float64,flags='C_CONTIGUOUS')
    fp=np.ctypeslib.ndpointer(np.float32,flags='C_CONTIGUOUS')
    I=ctypes.c_int;D=ctypes.c_double
    dll.local_field.argtypes=[I,dp,dp,D,I,I,I,ctypes.c_float,fp]
    dll.local_field.restype=None
    dll.pose_field.argtypes=[I,dp,I,dp,dp,D,I,I,I,fp,D,D,dp]
    dll.pose_field.restype=None
    return dll

class Field:
    def __init__(self,snap,start,end,step=.025):
        self.lib=library();self.snap=snap;self.step=step;self.cap=.175
        self.origin=np.floor((start[:3]-[.62,1.27,1.15])/step)*step
        extent=np.array([end[0]+.68,start[1]+1.27,start[2]+1.02])
        self.data=np.empty(np.maximum(3,np.ceil((extent-self.origin)/step).astype(int)+1),np.float32)
        self.lib.local_field(len(snap.obstacle_rows),snap.obstacle_rows,self.origin,step,*self.data.shape,self.cap,self.data)
    def query(self,points,states,reject=.085,cap=.162):
        q=np.ascontiguousarray(states,float);out=np.empty(len(q));p=np.ascontiguousarray(points,float)
        self.lib.pose_field(len(p),p,len(q),q,self.origin,self.step,*self.data.shape,self.data,cap,reject,out)
        return out

def local_surface(g,i):
    m=g.m;n=g.native_self;k=int(m.geom_type[i]);h=m.geom_size[i];G=mujoco.mjtGeom
    if k==G.mjGEOM_MESH:
        v=n.vertices[n.offset[i]:n.offset[i]+n.count[i]]
        hull=ConvexHull(v)
        pieces=[v]
        for tri in v[hull.simplices]:
            length=max(np.linalg.norm(tri[1]-tri[0]),np.linalg.norm(tri[2]-tri[0]),np.linalg.norm(tri[2]-tri[1]))
            count=int(np.ceil(length/.042))
            if count>1:
                a,b=np.triu_indices(count+1);u=a/count;w=(b-a)/count
                pieces.append(tri[0]+u[:,None]*(tri[1]-tri[0])+w[:,None]*(tri[2]-tri[0]))
        return np.vstack(pieces)
    if k==G.mjGEOM_BOX:
        axes=[np.linspace(-z,z,max(2,int(np.ceil(2*z/.038))+1)) for z in h]
        v=np.array(np.meshgrid(*axes,indexing='ij')).reshape(3,-1).T
        return v[np.any(abs(v)>=h-1.e-8,axis=1)]
    az=np.linspace(0.,2*np.pi,max(12,int(np.ceil(2*np.pi*h[0]/.035))),endpoint=False)
    if k==G.mjGEOM_CYLINDER:
        zs=np.linspace(-h[1],h[1],max(2,int(np.ceil(2*h[1]/.035))+1))
        rows=[[h[0]*np.cos(a),h[0]*np.sin(a),z] for a in az for z in zs]
        for rad in np.arange(0.,h[0],.035):rows.extend([[rad*np.cos(a),rad*np.sin(a),z] for a in az for z in (-h[1],h[1])])
        return np.array(rows)
    if k in (G.mjGEOM_CAPSULE,G.mjGEOM_SPHERE,G.mjGEOM_ELLIPSOID):
        half=h[1] if k==G.mjGEOM_CAPSULE else 0.
        r=h if k==G.mjGEOM_ELLIPSOID else np.repeat(h[0],3)
        return np.array([[r[0]*np.cos(v)*np.cos(a),r[1]*np.cos(v)*np.sin(a),r[2]*np.sin(v)+np.sign(v)*half]
                          for a in az for v in np.linspace(-np.pi/2,np.pi/2,9)])
    raise ValueError('Unsupported CAD proposal surface: '+str(k))

class RobotCloud:
    def __init__(self,g):
        self.g=g;self.local={i:local_surface(g,i) for i in g.ids}
        self.at=lru_cache(maxsize=256)(self.make)
    def make(self,group,theta,phi):
        g=self.g;d=g.w;g.place(d,[0,0,0,0,theta,phi]);mujoco.mj_kinematics(g.m,d)
        ids=g.body if group=='body' else g.tool
        pts=np.vstack([self.local[i]@d.geom_xmat[i].reshape(3,3).T+d.geom_xpos[i] for i in ids])
        # The points are an optimistic search guide. Every accepted reference
        # is independently checked against every registered full CAD link.
        pitch=.036 if group=='body' else .022
        _,use=np.unique(np.rint(pts/pitch).astype(np.int32),axis=0,return_index=True)
        p=pts[use]
        return np.ascontiguousarray(p[np.argsort(-np.linalg.norm(p,axis=1))])
