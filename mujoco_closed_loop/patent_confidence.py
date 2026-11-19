"""Timestamped two-source angular fusion. Kinematics, commands and ground truth are not inputs."""
from collections import deque
import numpy as np
def angles(b):
    return np.column_stack((np.rad2deg(np.linalg.norm(b,axis=1)),np.rad2deg(np.arctan2(b[:,1],b[:,0]))))
def wrap(x):return (x+180.)%360.-180.
def b_from(a):
    p=np.deg2rad(a[:,1]);return np.deg2rad(a[:,:1])*np.column_stack((np.cos(p),np.sin(p)))
def observable(theta):return np.sin(np.deg2rad(np.clip(theta,0,90)))**2

class BiasTeacher:
    """Accept only forward-validated updates to an additive bend-magnitude residual."""
    def __init__(self):
        self.values=np.zeros(3);self.previous=np.zeros(3);self.history=[deque(maxlen=40) for _ in range(3)]
        self.candidates=[None]*3;self.bad=np.zeros(3,int)
        self.accepted=0;self.rejected=0;self.rolled_back=0;self.used=0
    def weights(self,theta):
        # Overlapping bend bins interpolate correction through changing morphology.
        ref=np.array([0.,40.,85.]);return np.array([np.interp(theta,ref,np.eye(3)[i]) for i in range(3)])
    def correction(self,theta):return self.weights(theta)@self.values
    def sample(self,t,theta,vtheta,vs,rv,eligible):
        if not eligible:
            self.candidates=[None]*3
            return
        weights=self.weights(theta);j=int(np.argmax(weights))
        # If the strain itself is well constrained, an isolated camera discrepancy is insufficient.
        if theta<5 or vs<.65*rv:return
        self.used+=1
        target=float(np.clip(vtheta-theta,-15,15))
        current=self.correction(theta);pending=self.candidates[j]
        if pending is not None:
            proposal,creation,errors=pending
            if t-creation>1.8:self.candidates[j]=None
            else:
                errors.append((abs(target-current),abs(target-(current+weights[j]*(proposal-self.values[j]))),
                               abs(target-(current+weights[j]*(self.previous[j]-self.values[j])))))
                if len(errors)>=6:
                    base,new,prev=np.median(errors,axis=0)
                    if new+.12<base:
                        self.previous[j]=self.values[j];self.values[j]=proposal;self.accepted+=1;self.bad[j]=0
                    else:
                        self.rejected+=1;self.bad[j]=self.bad[j]+1 if prev+.5<base else 0
                        if self.bad[j]>=2:
                            self.values[j]=self.previous[j];self.bad[j]=0;self.rolled_back+=1
                    self.candidates[j]=None
                return
        buf=self.history[j];buf.append((t,target))
        recent=[v for ts,v in buf if t-ts<2.2]
        if len(recent)<10:return
        offset=np.median(recent)-current;spread=1.4826*np.median(abs(np.asarray(recent)-np.median(recent)))
        if abs(offset)<max(.65,.45*spread):return
        proposal=np.clip(self.values[j]+np.clip(.55*offset/max(.5,weights[j]),-2.5,2.5),-12,12)
        self.candidates[j]=(proposal,t,[])
        # Training pairs and the forward validation block are disjoint.
        buf.clear()

def fuse(t,bs,fresh,vs,age,capture,bv,cv,stable,profile,mode='both',delay=.20):
    t=np.asarray(t);n=len(t);dt=.02
    if n==0 or not np.allclose(np.diff(t),dt,atol=1e-7):raise ValueError('Requires 50 Hz synchronized control ticks')
    if mode not in ('both','teacher','strain'):raise ValueError(mode)
    s=angles(bs);v=angles(bv);allowed=np.all(np.isfinite(v),axis=1)&np.isfinite(cv)&(cv>0)
    strain_valid=np.isfinite(s).all(axis=1)
    fresh=np.asarray(fresh,dtype=bool)&strain_valid
    s=np.where(strain_valid[:,None],s,0.)
    ci=np.searchsorted(t,capture-1e-7);di=np.searchsorted(t,capture+delay-1e-7)
    delivery={int(d):(int(c),j) for j,(c,d) in enumerate(zip(ci,di)) if c<n and d<n and capture[j]>=t[0]-1e-7}
    vv=np.full_like(v,np.inf)
    vv[allowed,0]=profile['theta_k']/(cv[allowed]+profile['epsilon'])
    vv[allowed,1]=profile['phi_k']/(observable(v[allowed,0])*cv[allowed]+profile['epsilon'])
    if np.any(vs<=0) or not np.isfinite(vs).all():raise ValueError('Nonpositive sensor variance')
    ws=np.where(fresh[:,None],1./vs,0.);ws[s[:,0]<5,1]=0.
    vp=np.where(np.isfinite(vv),1./vv,0.);vp[v[:,0]<5,1]=0.
    teacher=BiasTeacher();corrected=s.copy();hist={};theta_bias=np.zeros(n);wtrace=np.full((n,2),np.nan)
    effective_w=np.zeros((n,2));updates=np.zeros((n,3),int);vis_used=np.zeros(n,bool)
    x=np.zeros((n,2));vel=x.copy();p00=x.copy();p01=x.copy();p11=x.copy();released=x.copy();unc=x.copy();plane=np.zeros(n,bool);observed=np.zeros(n,bool)
    for i in range(n):
        start=i
        if i in delivery and mode!='strain':
            c,j=delivery[i]
            if allowed[j]:
                hist[c]=j;start=c;vis_used[i]=True
                denom=ws[c]+vp[j];effective_w[i]=np.divide(vp[j],denom,out=np.zeros(2),where=denom>0)
                common=(ws[c]>0)&(vp[j]>0);wtrace[i,common]=effective_w[i,common]
            if mode=='teacher':
                teacher.sample(t[i],s[c,0],v[j,0],vs[c,0],vv[j,0],bool(stable[j] and allowed[j] and strain_valid[c] and age[c]<.045))
        if mode=='teacher':
            theta_bias[i]=teacher.correction(s[i,0]);corrected[i,0]=np.clip(s[i,0]+theta_bias[i],0.,120.)
            updates[i]=[teacher.accepted,teacher.rejected,teacher.rolled_back]
        for k in range(start,i+1):
            if k:
                xp=x[k-1]+dt*vel[k-1];dx=vel[k-1]
                pa=p00[k-1]+2*dt*p01[k-1]+dt*dt*p11[k-1]+.08**2+18.**2*dt**4/4
                pb=p01[k-1]+dt*p11[k-1]+18.**2*dt**3/2
                pc=p11[k-1]+18.**2*dt**2+1e-5;est_plane=plane[k-1]
            else:
                xp=corrected[0].copy() if fresh[0] else np.zeros(2)
                dx=np.zeros(2);pa=np.array([25. if fresh[0] else 120.**2,180.**2]);pb=np.zeros(2);pc=np.array([25.,40.**2]);est_plane=False
            j=hist.get(k);wp=ws[k].copy();obs=corrected[k].copy()
            if j is not None:
                both=wp+vp[j];obs[0]=(wp[0]*obs[0]+vp[j,0]*v[j,0])/both[0]
                if vp[j,1]>0:
                    e=wp[1]*np.exp(1j*np.deg2rad(obs[1]))+vp[j,1]*np.exp(1j*np.deg2rad(v[j,1]))
                    if abs(e)>1e-8*both[1]:obs[1]=np.rad2deg(np.angle(e))
                    else:both[1]=0.
                wp=both
            if wp[1]>0 and not est_plane:
                xp[1]=obs[1];dx=dx.copy();dx[1]=0.;est_plane=True
            eff=wp/(1.+pa*wp);err=obs-xp;err[1]=wrap(err[1])
            x[k]=xp+pa*eff*err;vel[k]=dx+pb*eff*err
            p00[k]=pa/(1.+pa*wp);p01[k]=pb/(1.+pa*wp);p11[k]=pc-pb**2*eff;plane[k]=est_plane
        released[i]=x[i];unc[i]=np.sqrt(p00[i]);observed[i]=fresh[i] or vis_used[i]
    released[:,0]=np.clip(released[:,0],0.,120.);released[:,1]=wrap(released[:,1])
    return dict(b=b_from(released),angle=released,posterior_sigma=unc,joint_vis_share=wtrace,visible_update=vis_used,
                bias=theta_bias,correction_log=updates,observed=observed,plane=(released[:,0]>=5)&plane,
                teacher_pairs=int(teacher.used))
