"""Causal four-gauge health inference and state-conditioned bending uncertainty.

Time is seconds; bend-vector outputs are radians and angular variances are degree squared.
All observations supplied at runtime are box readings or the nominal box-only prediction.
"""
import numpy as np
from scipy.spatial import cKDTree

def polynomial(d):
    u,v=d.T
    return np.column_stack((np.ones(len(d)),u,v,u*u,u*v,v*v,u**3,u*u*v,u*v*v,v**3))

def split_pairs(adc,tare,gains):
    x=(np.asarray(adc,float)-tare)/gains
    return .5*(x[:,:2]-x[:,2:]),.5*(x[:,:2]+x[:,2:])

def ridge(X,Y,penalty):
    center=X.mean(axis=0);scale=np.maximum(X.std(axis=0),.0002)
    Z=np.column_stack((np.ones(len(X)),(X-center)/scale))
    reg=np.eye(Z.shape[1])*penalty*len(Z);reg[0,0]=0.
    coef=np.linalg.solve(Z.T@Z+reg,Z.T@Y)
    return center,scale,coef

def design(t,adc,tare,gains,forward):
    t=np.asarray(t,float)
    if len(t)==0 or len(t)!=len(adc) or not np.all(np.diff(t)>0):raise ValueError('Invalid acquisition timestamps')
    d,common=split_pairs(adc,tare,gains)
    residual=common-polynomial(d)@forward
    slow=d[0].copy();fast=d[0].copy();slow_r=residual[0].copy()
    play=d[0].copy();before=d[0].copy()
    ret=[];signature=[];phase=[];hist=[]
    for i in range(len(t)):
        dt=t[i]-t[i-1] if i else .025
        fast+=(-np.expm1(-dt/.22))*(d[i]-fast);slow+=(-np.expm1(-dt/1.50))*(d[i]-slow)
        slow_r+=(-np.expm1(-dt/.45))*(residual[i]-slow_r)
        play=np.clip(play,d[i]-.018,d[i]+.018)
        speed=(d[i]-fast)/.22;trail=d[i]-play;long=d[i]-slow
        magnitude=np.linalg.norm(d[i])
        radial=speed@d[i]/max(magnitude,.012)
        phase.append(0 if abs(radial)<.006 else (1 if radial>0 else 2))
        rail=float(np.any((adc[i]<=1)|(adc[i]>=4094)))
        # Opposing-pair parity and its recent evolution reveal departures from the
        # calibrated static surface. History terms depend only on samples up to i.
        signature.append(np.r_[residual[i],slow_r,residual[i]-slow_r,
                               magnitude,np.linalg.norm(speed),radial,trail@d[i]/max(magnitude,.012),rail])
        ret.append(np.r_[polynomial(d[i:i+1])[0,1:],residual[i],residual[i]*d[i],
                         trail,long,speed,trail*magnitude,long*magnitude,rail])
        hist.append(np.r_[residual[i],radial])
    return np.asarray(ret),np.asarray(signature),np.asarray(phase),np.asarray(hist)

def angle_features(b):
    return np.stack((np.degrees(np.linalg.norm(b,axis=-1)),np.degrees(np.arctan2(b[...,1],b[...,0]))),axis=-1)

def wrap(a):
    return (a+180.)%360.-180.

class HealthObserver:
    def __init__(self,model):
        self.m=model
        self.tree=cKDTree(model['library'])

    def point(self,features,nominal):
        m=self.m
        estimates=[nominal]
        for center,scale,coef in zip(m['centers'],m['scales'],m['coefficients']):
            x=np.column_stack((np.ones(len(features)),(features-center)/scale))
            estimates.append(x@coef)
        return np.stack(estimates,axis=1)

    def estimate(self,t,adc,tare,nominal):
        m=self.m
        x,signature,phase,history=design(t,adc,tare,m['gains'],m['forward'])
        normalized=(signature-m['class_center'])/m['class_scale']
        dist,idx=self.tree.query(normalized,k=48)
        weights=1./np.maximum(dist,.09)**2
        labels=m['library_labels'][idx]
        probs=np.column_stack([(weights*(labels==j)).sum(1) for j in range(4)])
        probs/=probs.sum(1,keepdims=True)
        # Prior to the first event no health class is known.
        belief=np.full(4,.25)
        for i,stamp in enumerate(t):
            dt=t[i]-t[i-1] if i else .025
            belief+=(-np.expm1(-dt/.16))*(probs[i]-belief);probs[i]=belief
        alternatives=self.point(x,nominal)
        bend=np.sum(probs[:,:,None]*alternatives,axis=1)
        angle=angle_features(bend);candidate=angle_features(alternatives)
        error=candidate-angle[:,None,:];error[:,:,1]=wrap(error[:,:,1])
        within=m['variance'][:,phase,:].transpose(1,0,2)
        spread=np.sum(probs[:,:,None]*(within+error**2),axis=1)
        novelty=dist[:,-1]/float(m['novelty_ref'])
        variance=np.maximum(spread*m['variance_scale'],[.16**2,.3**2])
        variance*=np.maximum(1.,np.minimum(novelty,6.))[:,None]**2
        variance[:,1]=np.maximum(variance[:,1],(.45/np.maximum(.04,np.sin(np.radians(np.minimum(angle[:,0],90.)))))**2)
        identified=(probs.max(1)>=.90)&(novelty<=1.)
        return dict(b=bend,vs=variance,prob=probs,identified=identified,novelty=novelty,
                    phase=phase,history=history,alternatives=alternatives)
