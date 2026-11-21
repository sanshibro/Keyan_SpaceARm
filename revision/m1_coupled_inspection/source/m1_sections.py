"""Plane cuts through the logged native convex link geometry at three fixed abscissae."""
from functools import lru_cache
import numpy as np
import mujoco
from scipy.spatial import ConvexHull,QhullError
from gate_distance_audit import DistanceAudit

class Sections:
    def __init__(self,p):
        self.p=p;self.m=p.m;self.a=DistanceAudit(p.m,tolerance_m=.00015,strict=False,obstacle_prefix='COL_')
        self.body=set(self.a.external['body'][:,0]);self.arm=set(self.a.external['arm_tool'][:,0])
        self.world=set(self.a.external['body'][:,1])
    @lru_cache(maxsize=None)
    def shell(self,i):
        m=self.m;s=m.geom_size[i];kind=m.geom_type[i];G=mujoco.mjtGeom
        if kind==G.mjGEOM_MESH:
            k=m.geom_dataid[i];start=m.mesh_vertadr[k];verts=np.unique(m.mesh_vert[start:start+m.mesh_vertnum[k]],axis=0).astype(float)
        elif kind==G.mjGEOM_BOX:
            verts=np.array([[x,y,z] for x in (-s[0],s[0]) for y in (-s[1],s[1]) for z in (-s[2],s[2])])
        else:
            az=np.linspace(0,2*np.pi,33)[:-1]
            if kind==G.mjGEOM_CYLINDER:profile=[(s[0],-s[1]),(s[0],s[1])]
            elif kind==G.mjGEOM_SPHERE or kind==G.mjGEOM_CAPSULE:
                half=s[1] if kind==G.mjGEOM_CAPSULE else 0.
                profile=[(s[0]*np.cos(v),s[0]*np.sin(v)+(-half if v<0 else half)) for v in np.linspace(-np.pi/2,np.pi/2,18)]
            elif kind==G.mjGEOM_ELLIPSOID:
                verts=np.array([[s[0]*np.cos(v)*np.cos(u),s[1]*np.cos(v)*np.sin(u),s[2]*np.sin(v)] for v in np.linspace(-np.pi/2,np.pi/2,18) for u in az])
                profile=[]
            else:raise ValueError('Unexpected section geometry '+str(kind))
            if profile:verts=np.array([[r*np.cos(u),r*np.sin(u),z] for r,z in profile for u in az])
        hull=ConvexHull(verts)
        es=np.vstack((hull.simplices[:,[0,1]],hull.simplices[:,[1,2]],hull.simplices[:,[2,0]]))
        return verts,np.unique(np.sort(es,axis=1),axis=0)
    def cut(self,q,x):
        a=self.a;m=self.m;a.d.qpos[:]=q;mujoco.mj_kinematics(m,a.d);lo,hi=a.bounds()
        parts={'body':[],'arm':[],'walls':[]}
        for i in sorted(self.world | self.body | self.arm):
            if lo[i,0]>x or hi[i,0]<x:continue
            v,e=self.shell(i);w=v@a.d.geom_xmat[i].reshape(3,3).T+a.d.geom_xpos[i]
            ell=w[e[:,0]];r=w[e[:,1]];den=r[:,0]-ell[:,0]
            take=(np.abs(den)>1e-9)&((ell[:,0]-x)*(r[:,0]-x)<=0.)
            on=w[np.abs(w[:,0]-x)<1e-8,1:3]
            points=ell[take,1:3]+((x-ell[take,0])/den[take])[:,None]*(r[take,1:3]-ell[take,1:3])
            yz=np.unique(np.round(np.vstack((points,on)),6),axis=0)
            if len(yz)<3:continue
            try:poly=yz[ConvexHull(yz).vertices]
            except QhullError:continue
            group='walls' if i in self.world else 'body' if i in self.body else 'arm';parts[group].append(poly.tolist())
        return parts
