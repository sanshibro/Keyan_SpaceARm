"""Exact-support robot geometry on immutable local snapshots, batched along a spline."""
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
def entry():
    source=Path(__file__).with_suffix('.cc')
    key=hashlib.sha256(source.read_bytes()+source.with_name('m2_self.cc').read_bytes()).hexdigest()[:16]
    lib=Path(tempfile.gettempdir())/('m2_path_'+key+'.so')
    if not lib.exists():
        tmp=lib.with_name(lib.stem+'-'+str(os.getpid())+lib.suffix)
        subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC',str(source),'-o',str(tmp)],check=True)
        tmp.replace(lib)
    f=ctypes.CDLL(str(lib)).path_clearances
    ip=np.ctypeslib.ndpointer(np.int32,flags='C_CONTIGUOUS')
    fp=np.ctypeslib.ndpointer(np.float64,flags='C_CONTIGUOUS')
    I=ctypes.c_int
    f.argtypes=[I,I,ip,fp,fp,ip,ip,fp,ip,fp,ip,I,fp,ctypes.c_double,fp,I,fp,fp,ip]
    f.restype=None
    return f

class PathBatch:
    def __init__(self,g):
        self.g=g
        self.fn=entry()
        self.group=np.ascontiguousarray(~g.groups,np.int32)
        mapping=np.full(g.m.ngeom,-1,int)
        mapping[g.ids]=np.arange(len(g.ids))
        records=[]
        for k,pairs in ((2,g.audit.hazard),(3,g.audit.fit)):
            a=mapping[pairs]
            if np.min(a)<0:raise ValueError('Missing native self pair in path proof')
            records.extend(np.column_stack((a,np.full(len(a),k))))
        self.internal=np.ascontiguousarray(records,np.int32)
        self.joints=np.array([[g.m.jnt_qposadr[g.m.joint(axis+str(i)).id] for axis in ('bx','by')]
                              for i in range(g.bridge.segments)])
    def evaluate(self,states,caps=(.132,.132,.04,.006)):
        g=self.g;c=g.convex_map
        states=np.asarray(states,float).reshape(-1,6)
        if not np.isfinite(states).all():raise ValueError('Nonfinite body or arm pose')
        poses=np.empty((len(states),len(g.ids),12))
        d=g.w
        angles=states[:,3]/2
        unit=np.column_stack((np.cos(states[:,5]),np.sin(states[:,5])))
        bend=states[:,4,None]*unit/g.bridge.segments
        for j,s in enumerate(states):
            d.qpos[g.root:g.root+3]=s[:3]
            d.qpos[g.root+3:g.root+7]=[np.cos(angles[j]),0,0,np.sin(angles[j])]
            d.qpos[self.joints]=bend[j]
            mujoco.mj_kinematics(g.m,d)
            poses[j,:,:3]=d.geom_xpos[g.ids];poses[j,:,3:]=d.geom_xmat[g.ids]
        out=np.zeros((len(states),5));pairs=np.full((len(states),4,2),-1,np.int32)
        self.fn(c.nr,c.ns,c.types,c.sizes,c.boxes,c.offset,c.nvert,g.native_self.vertices,self.group,
                c.obpose,self.internal,len(self.internal),np.ascontiguousarray(g.snapshot.origin),g.snapshot.radius,
                np.ascontiguousarray(caps),len(states),poses,out,pairs)
        return out,pairs
    def names(self,indices):
        g=self.g;nr=len(g.ids)
        return [None if i<0 else g.names[g.ids[i]] if i<nr else g.obnames[i-nr] for i in indices]
