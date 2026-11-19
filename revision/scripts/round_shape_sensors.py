"""Virtual surface strain and an independent optical-marker measurement of arm shape."""
import hashlib
import math
import numpy as np

MARKERS=np.array([[.018,.006,.004],[-.014,.011,-.004],[.002,-.017,.007],[-.004,-.003,-.015]])
RAD=.025
NODES=['seg%d'%i for i in range(8)]+['tool']
RING=RAD*np.array([[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]])
STATION=np.arange(9)*.035
GAUGE_STATIONS=np.r_[.11,.14,.17]
IDX=np.searchsorted(STATION,GAUGE_STATIONS,side='right')-1
FRACTION=(GAUGE_STATIONS-STATION[IDX])/.035


def stable_seed(label):
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:4],'little')


def frame_points(d,ids):
    return d.xpos[ids].copy(),d.xmat[ids].reshape(-1,3,3)


def surface_lengths(d,ids):
    centers,axes=frame_points(d,ids)
    strips=centers[:,None,:]+RING[None,:,:]@axes.transpose(0,2,1)
    points=(1-FRACTION[:,None,None])*strips[IDX]+FRACTION[:,None,None]*strips[IDX+1]
    return np.linalg.norm(np.diff(points,axis=0),axis=2).sum(axis=0)


def marker_cluster(d,bodyid):
    return d.xpos[bodyid]+MARKERS@d.xmat[bodyid].reshape(3,3).T


def marker_rot(points):
    # Orthogonal Procrustes from surveyed constellation; two calls for two tracked bodies.
    A=MARKERS-MARKERS.mean(axis=0)
    B=points-points.mean(axis=0)
    u,_,v=np.linalg.svd(A.T@B)
    corr=np.ones(3);corr[-1]=np.linalg.det(v.T@u.T)
    return (v.T*corr)@u.T


def mocap_bend(markers):
    R=marker_rot(markers[0]).T@marker_rot(markers[1])
    tangent=R[:,2]
    r=np.linalg.norm(tangent[:2])
    return np.zeros(2) if r<1e-10 else np.arctan2(r,tangent[2])*tangent[:2]/r


class Channels:
    def __init__(self,m,d_neutral,session):
        self.ids=[m.body(n).id for n in NODES]
        self.tracked=[m.body(n).id for n in ('mount','tool')]
        self.zero=surface_lengths(d_neutral,self.ids)
        assert np.allclose(self.zero,.06,atol=1e-7)
        self.rng=np.random.default_rng(stable_seed(session))
        self.gain=np.array([1.,.94,1.06,.97])*(1+self.rng.normal(0,.008,4))
        self.offset=self.rng.uniform(-.002,.002,4)
        self.drift=self.rng.uniform(-.000015,.000015,4)
        self.filtered=np.zeros(4)
        self.prior_t=None

    def read(self,d,t):
        axial=(surface_lengths(d,self.ids)-self.zero)/self.zero
        dt=.02 if self.prior_t is None else t-self.prior_t
        if dt<=0:raise ValueError('virtual-gauge timestamps must increase')
        self.filtered+=(1-np.exp(-dt/.045))*(axial-self.filtered)
        self.prior_t=t
        analog=self.offset+self.drift*t+self.gain*(self.filtered+.08*self.filtered*abs(self.filtered))
        signal=analog+self.rng.normal(0,.00045,4)
        counts=np.rint(np.clip((signal+.35)/.70,0,1)*4095).astype(np.int16)
        markers=np.array([marker_cluster(d,b) for b in self.tracked])
        return counts,markers


def vector_angles(b):
    b=np.asarray(b)
    return np.rad2deg(np.linalg.norm(b,axis=-1)),np.rad2deg(np.arctan2(b[...,1],b[...,0]))


def wrap_deg(x):
    return (np.asarray(x)+180.)%360.-180.


def features(adc,t,tare=None):
    # Only the straight initial interval is used to tare an acquisition session.
    t=np.asarray(t);adc=np.asarray(adc,dtype=float)
    mask=(t>=t[0])&(t<t[0]+.5)
    if tare is None:
        if mask.sum()<10:raise ValueError('short initial tare')
        tare=np.median(adc[mask],axis=0)
    x=(adc-tare)*(.70/4095.)
    return np.column_stack([x,x[:,0]-x[:,2],x[:,1]-x[:,3]]),tare


def poly(x):
    return np.column_stack([np.ones(len(x)),x]+[x[:,i]*x[:,j] for i in range(x.shape[1]) for j in range(i,x.shape[1])])


def fit_model(inputs,labels):
    X=np.vstack(inputs);Y=np.vstack(labels)
    mean=X.mean(0);sd=X.std(0)
    if np.any(sd<1e-6):raise ValueError('uninformative training channel')
    Z=poly((X-mean)/sd)
    penalty=.01*np.eye(Z.shape[1]);penalty[0,0]=0.
    coef=np.linalg.solve(Z.T@Z+penalty,Z.T@Y)
    return dict(mean=mean.tolist(),std=sd.tolist(),coef=coef.tolist(),alpha=.01)


def predict(model,X):
    Z=poly((X-np.asarray(model['mean']))/np.asarray(model['std']))
    return Z@np.asarray(model['coef'])
