"""Bound the actual minimum of native convex collision solids at every saved free-flight pose."""
import itertools
import json
import math
import numpy as np
import pandas as pd
import mujoco
from round_g1_model import DATA, Geometry, make_plant
from round_g1_audit import save

TOL=.00015


def reduce_simplex(p,q):
    w=p-q
    best=None
    for n in range(1,len(w)+1):
        for ids in itertools.combinations(range(len(w)),n):
            ids=np.array(ids)
            v=w[ids]
            if n==1:
                weights=np.ones(1)
            else:
                edge=v[1:]-v[0]
                z=np.linalg.lstsq(edge.T,-v[0],rcond=1e-12)[0]
                weights=np.r_[1.-z.sum(),z]
                if weights.min() < -1e-10:continue
                weights=np.maximum(weights,0.)
                weights/=weights.sum()
            h=weights@v;s=h@h
            if best is None or s<best[0]:
                active=weights>1e-11
                best=(s,ids[active],weights[active]/weights[active].sum())
    _,ids,e=best
    return p[ids],q[ids],e@p[ids],e@q[ids]


class Solids:
    def __init__(self,g):
        self.g=g;self.mesh={}
        for i in g.ids:
            if g.m.geom_type[i]==mujoco.mjtGeom.mjGEOM_MESH:
                k=int(g.m.geom_dataid[i]);j=int(g.m.mesh_vertadr[k])
                self.mesh[i]=g.m.mesh_vert[j:j+g.m.mesh_vertnum[k]].astype(float)

    def extreme(self, i, direction):
        m,d=self.g.m,self.g.d
        R=d.geom_xmat[i].reshape(3,3);v=R.T@direction
        typ=m.geom_type[i];h=m.geom_size[i]
        if i in self.mesh:
            vertices=self.mesh[i];p=vertices[np.argmax(vertices@v)]
        elif typ==mujoco.mjtGeom.mjGEOM_BOX:
            p=h*np.sign(v)
        elif typ==mujoco.mjtGeom.mjGEOM_SPHERE:
            p=v*h[0]
        elif typ==mujoco.mjtGeom.mjGEOM_CAPSULE:
            p=v*h[0];p[2]+=h[1]*np.sign(v[2])
        elif typ==mujoco.mjtGeom.mjGEOM_CYLINDER:
            p=np.r_[h[0]*v[:2]/max(1e-15,np.linalg.norm(v[:2])),h[1]*np.sign(v[2])]
        elif typ==mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            p=h*h*v/max(1e-15,np.linalg.norm(h*v))
        else:raise ValueError('Unknown type %s'%typ)
        return d.geom_xpos[i]+R@p

    def pair(self,i,k,lb):
        g=self.g
        ctr=(g.glo[k]+g.ghi[k])/2.;half=(g.ghi[k]-g.glo[k])/2.
        direction=g.d.geom_xpos[i]-ctr;direction/=max(np.linalg.norm(direction),1e-15)
        p=np.array([self.extreme(i,-direction)])
        q=np.array([ctr+half*np.sign(direction)])
        a,b=p[0],q[0]
        for _ in range(64):
            ub=float(np.linalg.norm(a-b))
            if ub-lb<TOL:break
            direction=(a-b)/ub
            pa=self.extreme(i,-direction);pb=ctr+half*np.sign(direction)
            lb=max(lb,float(direction@(pa-pb)))
            if ub-lb<TOL:break
            p,q,a,b=reduce_simplex(np.vstack([p,pa]),np.vstack([q,pb]))
        assert ub>=lb-1e-7,(i,k,lb,ub)
        return min(lb,ub),ub,a,b

    def scan(self, raw, frame, details, group, release):
        g=self.g
        col='body_m' if group=='body' else 'arm_tool_m'
        picked=g.cat=='body' if group=='body' else g.cat!='body'
        best=float('inf');lowest=float('inf');record={};pairs=0;frames=0
        for idx in np.argsort(frame[col].values):
            broad=float(frame[col].iloc[idx])
            # The trace is rounded to 8 decimal places when written to CSV.
            if broad-1e-8>=best:break
            stamp=int(round(release+frame.t.iloc[idx]*1e6))
            kk=int(np.searchsorted(raw['us'],stamp));assert raw['us'][kk]==stamp
            g.from_record(raw['qpos'][kk]);ds=g.distances(*g.bounds())
            ds[~picked]=np.inf;frames+=1
            for ij in np.argsort(ds,axis=None):
                base=float(ds.flat[ij])
                if base>=best:break
                j,k=np.unravel_index(ij,ds.shape);i=int(g.ids[j]);pairs+=1
                lb,ub,a,b=self.pair(i,k,base)
                lowest=min(lowest,lb)
                if ub<best:
                    best=ub
                    record=dict(t=float(frame.t.iloc[idx]),k=kk,geom=g.m.geom(i).name,gate=g.gates[k][0],
                                robot=a.tolist(),wall=b.tolist())
        return dict(lower_m=min(lowest,best),upper_m=best,frames_refined=frames,
                    pairs_refined=pairs,**record)


def self_test():
    g=Geometry(make_plant());s=Solids(g)
    g.place([1.,0.,2.21],0)
    test=[]
    for name in ['collision_central_equipment','visual_gripper_finger1','collision_landing_163']:
        i=g.m.geom(name).id
        dir=np.array([0.,0.,1.]);q=s.extreme(i,dir)
        # Wide plane removes the other two coordinates from this independent analytic test.
        k=1;g.glo[k]=[-5.,-5.,2.8];g.ghi[k]=[5.,5.,4.]
        lo,hi,_,_=s.pair(i,k,0.)
        truth=2.8-q[2]
        assert lo-1e-8<=truth<=hi+1e-8 and hi-lo<=TOL, (name,lo,truth,hi)
        test.append([name,float(truth),lo,hi])
    return test


def main():
    test=self_test();audit=json.loads((DATA/'audit.json').read_text());out={}
    geo={k:Solids(Geometry(make_plant(k),k)) for k in ('study','wide')}
    for name,details in audit['runs'].items():
        g=geo['wide' if name.endswith('-wide') else 'study']
        folder=DATA/'runs'/name;raw=dict(np.load(str(folder/'physics.npz')))
        release=json.loads((folder/'run.json').read_text())['release_us']
        tr=pd.read_csv(DATA/'traces'/(name+'.csv'))
        out[name]={k:g.scan(raw,tr,details,k,release) for k in ('body','arm_tool')}
        b,a=out[name].values()
        print(name,round(100*b['upper_m'],3),round(100*a['upper_m'],3),flush=True)
    save(DATA/'convex-distance.json',dict(method='GJK feasible-simplex upper bound and rigorous support/AABB lower bound; all 50 Hz logged poses; native convex shapes and physical gate slabs',tolerance_m=TOL,plane_regression=test,runs=out))
    assert all(v['upper_m']-v['lower_m']<2*TOL for a in out.values() for v in a.values()),'global brackets need review'


if __name__=='__main__':main()
