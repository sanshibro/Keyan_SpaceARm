"""Nominal rigid-body clearance over a private, immutable local map snapshot."""
import ctypes
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile

import mujoco
import numpy as np


@lru_cache(maxsize=1)
def native_batch():
    source = Path(__file__).with_suffix('.cc')
    digest = hashlib.sha256(source.read_bytes()+source.with_name('m2_self.cc').read_bytes()).hexdigest()[:16]
    folder = Path(tempfile.gettempdir())/('m2_body_batch_'+digest)
    folder.mkdir(exist_ok=True)
    path = folder/('body.dll' if os.name=='nt' else 'body.so')
    if not path.exists():
        tmp = folder/(str(os.getpid())+path.suffix)
        subprocess.run(['c++','-O3','-std=c++17','-fPIC','-shared',str(source),'-o',str(tmp)],check=True)
        tmp.replace(path)
    f = ctypes.CDLL(str(path)).rigid_batch
    fp = np.ctypeslib.ndpointer(np.float64,flags='C_CONTIGUOUS')
    ip = np.ctypeslib.ndpointer(np.int32,flags='C_CONTIGUOUS')
    num = ctypes.c_int
    f.argtypes = [num,num,ip,fp,fp,fp,ip,ip,fp,num,fp,ctypes.c_double,ctypes.c_double,fp]
    f.restype = None
    return f


class BodyBatch:
    """Uses the same body IDs, mesh support points and obstacle primitives as ConvexMap."""
    def __init__(self,g,group='body',shape=(0.,0.)):
        self.g = g
        self.snapshot = g.snapshot
        if self.snapshot is None:
            raise ValueError('A causal map snapshot is required')
        cm = g.convex_map
        if group not in ('body','arm'):
            raise ValueError('Expected body or arm component group')
        body = np.flatnonzero(g.groups if group=='body' else ~g.groups)
        if not len(body):
            raise ValueError('No registered rigid-body geometry')
        data=g.w
        g.place(data,np.r_[np.zeros(4),shape])
        mujoco.mj_kinematics(g.m,data)
        geo=g.ids[body]
        ctr=data.geom_xpos[geo]+np.einsum('nij,nj->ni',data.geom_xmat[geo].reshape(-1,3,3),g.m.geom_aabb[geo,:3])
        outer=np.linalg.norm(ctr,axis=1)+np.linalg.norm(g.m.geom_aabb[geo,3:],axis=1)
        body=body[np.argsort(-outer,kind='stable')]
        self.nr = len(body)
        ids = np.r_[body,np.arange(cm.nr,cm.count)]
        self.n = len(ids)
        self.types,self.sizes,self.boxes,self.offset,self.nvert = [
            np.ascontiguousarray(a[ids]) for a in (cm.types,cm.sizes,cm.boxes,cm.offset,cm.nvert)]
        geom_ids = g.ids[body]
        robot = np.column_stack((data.geom_xpos[geom_ids],data.geom_xmat[geom_ids]))
        self.poses = np.ascontiguousarray(np.vstack((robot,cm.obpose)))
        self.vertices = g.native_self.vertices
        self.fn = native_batch()

    def query(self,states,cap=.30,reject_below=0.):
        """A rejected row returns -inf only when a real upper witness is below reject_below."""
        states = np.ascontiguousarray(states,dtype=np.float64)
        if states.ndim!=2 or states.shape[1]!=6 or not np.isfinite(states).all():
            raise ValueError('Expected finite rows [x,y,z,yaw,theta,phi] in metres and radians')
        if not np.isfinite(cap) or cap<=0:
            raise ValueError('Positive distance cap required')
        if not 0.<=reject_below<=cap:
            raise ValueError('Invalid rejection threshold')
        if self.g.snapshot is not self.snapshot:
            raise ValueError('BodyBatch belongs to a prior map snapshot')
        out = np.empty(len(states),dtype=np.float64)
        self.fn(self.n,self.nr,self.types,self.sizes,self.poses,self.boxes,self.offset,self.nvert,
                self.vertices,len(states),states,cap,reject_below,out)
        return out
