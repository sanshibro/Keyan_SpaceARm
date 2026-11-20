"""Timestamped bend and gauge-bias estimation in the two-dimensional bending plane (radians)."""
from dataclasses import dataclass
import numpy as np

I2=np.eye(2)
RAD=np.pi/180
METHODS=('vision','strain','fixed','proposed')

@dataclass(frozen=True)
class Settings:
    acceleration:float=22.0     # deg / s^(3/2), spectral SD of white acceleration
    position_rw:float=.11       # deg / sqrt(s)
    bias_rw:float=.09
    active_bias_rw:float=.95
    initial_bias_sd:float=.38
    fixed_strain_sd:float=1.25
    fixed_vision_sd:float=1.55
    fixed_acceleration:float=22.0
    strain_scale:float=1.
    vision_scale:float=1.
    vision_floor:float=.40
    bias_evidence_tau:float=.48
    bias_deadband:float=1.10
    vision_gate:float=13.82     # chi-square 2, 99.9%
    delay:float=.2

def check_packet(p,vision):
    t=p['t'];b=p['b'];r=p['R'];ok=p['valid'] if vision else np.ones(len(t),bool)
    if t.ndim!=1 or np.any(np.diff(t)<=0) or b.shape!=(len(t),2) or r.shape!=(len(t),2,2):
        raise ValueError('Require strictly ordered timestamps, 2-D bends and 2x2 covariances')
    if not np.isfinite(t).all() or not np.isfinite(b[ok]).all() or not np.isfinite(r[ok]).all():
        raise ValueError('Non-finite measurement marked valid')
    if ok.any() and (not np.allclose(r[ok],r[ok].transpose(0,2,1)) or np.min(np.linalg.eigvalsh(r[ok]))<=0):
        raise ValueError('Observation covariance must be symmetric positive definite')

class BendFilter:
    def __init__(self,c,augmented,t0):
        self.c=c;self.augmented=augmented;self.n=6 if augmented else 4
        self.Hv=np.zeros((2,self.n));self.Hv[:,:2]=I2
        self.Hs=self.Hv.copy()
        if augmented:self.Hs[:,4:]=I2
        self.I=np.eye(self.n);self.t=float(t0);self.x=np.zeros(self.n)
        init=[25,25,20,20]+([c.initial_bias_sd]*2 if augmented else [])
        self.P=np.diag(np.square(np.asarray(init)*RAD))
        self.last_st=-np.inf;self.last_s=np.zeros(2);self.last_sr=I2.copy()
        self.last_evidence=-np.inf;self.mean_pair=np.zeros(2);self.scatter=0.;self.activity=0.
        self.gain_s=0.;self.gain_v=0.;self.nis=np.nan;self.used_camera=False
    def copy(self):
        f=object.__new__(BendFilter);f.__dict__=self.__dict__.copy()
        f.x=self.x.copy();f.P=self.P.copy()
        return f
    def predict(self,t):
        dt=float(t-self.t)
        if dt < -2e-8:raise ValueError('Timestamps moved backward')
        if dt<=1e-10:return
        F=self.I.copy();F[:2,2:4]=dt*I2
        acc=self.c.acceleration if self.augmented else self.c.fixed_acceleration
        q=(acc*RAD)**2
        Q=np.zeros_like(self.P);Q[:2,:2]=(q*dt**3/3+(self.c.position_rw*RAD)**2*dt)*I2
        Q[:2,2:4]=Q[2:4,:2]=q*.5*dt**2*I2;Q[2:4,2:4]=q*dt*I2
        if self.augmented:
            # Excite bias dynamics when independent, sustained camera-strain disagreement occurs.
            # A lapse in valid images removes this extra excitation; background random walk remains.
            age=self.t-self.last_evidence
            excitation=self.activity*np.exp(-max(0.,age-.15)/.48)
            Q[4:,4:]=dt*RAD**2*(self.c.bias_rw**2+excitation*self.c.active_bias_rw**2)*I2
        self.x=F@self.x;self.P=F@self.P@F.T+Q;self.t=t
    def update(self,z,H,R,vision=False,bias_fraction=0.):
        e=z-H@self.x;cross=self.P@H.T;S=H@cross+R
        # S is 2x2 positive definite; solve explicitly to avoid a BLAS launch for each ADC event.
        a,b,d=S[0,0],.5*(S[0,1]+S[1,0]),S[1,1];det=a*d-b*b
        if det<=0:raise ArithmeticError('Nonpositive innovation covariance')
        inv=np.array([[d,-b],[-b,a]])/det;nis=float(e@inv@e)
        if vision:
            self.nis=nis
            if self.augmented and nis>self.c.vision_gate:
                self.gain_v=0.;self.used_camera=False;return
        K=cross@inv
        if self.augmented:
            # In a strain-only update the two summands b and delta are not separately observable.
            # Schmidt's constrained gain preserves the last optically anchored bias estimate.
            K[4:]*=bias_fraction
        self.x+=K@e
        A=self.I-K@H;self.P=A@self.P@A.T+K@R@K.T;self.P=.5*(self.P+self.P.T)
        gain=float(np.trace(K[:2])*.5)
        if vision:self.gain_v=gain;self.used_camera=True
        else:self.gain_s=gain
    def strain(self,t,z,R,paired=0.):
        self.predict(t);self.last_st=t;self.last_s=z;self.last_sr=R
        cov=R*self.c.strain_scale if self.augmented else I2*(self.c.fixed_strain_sd*RAD)**2
        self.update(z,self.Hs,cov,bias_fraction=paired)
    def camera(self,t,z,R,valid,quality,stable):
        self.predict(t);self.used_camera=False;self.nis=np.nan;self.gain_v=0.
        if not valid:return
        if not self.augmented:
            self.update(z,self.Hv,I2*(self.c.fixed_vision_sd*RAD)**2,vision=True);return
        cov=R*self.c.vision_scale+I2*(self.c.vision_floor*RAD)**2
        optical_sd=float(np.sqrt(.5*np.trace(cov)))/RAD
        bias_fraction=float(stable)*float(np.clip((quality-.10)/.45,0,1)*np.clip((5.-optical_sd)/2.5,0,1))
        if bias_fraction>0 and t-self.last_st<.08:
            # Detect coherent disagreement using independently timed observations before applying
            # the optical correction. Large, isolated residuals have little effect on this state.
            residual=self.last_s+self.x[2:4]*(t-self.last_st)-z
            elapsed=t-self.last_evidence
            if elapsed>.50:
                self.mean_pair=self.x[4:].copy();self.scatter=0.;weight=.07
            else:weight=-np.expm1(-max(0.,elapsed)/self.c.bias_evidence_tau)
            diff=residual-self.mean_pair;norm=np.linalg.norm(diff)/RAD
            clipped=diff*min(1.,12./max(norm,1e-10))
            self.mean_pair=self.mean_pair+weight*clipped
            self.scatter=(1-weight)*self.scatter+weight*min(norm,5.)**2
            self.last_evidence=t
            disagreement=np.linalg.norm(self.mean_pair-self.x[4:])/RAD
            tol=self.c.bias_deadband+.28*optical_sd+.20*np.sqrt(self.scatter)
            self.activity=float(np.clip((disagreement-tol)/1.75,0,1))*bias_fraction
        self.update(z,self.Hv,cov,vision=True,bias_fraction=bias_fraction)

def advance(f,t,s,j,paired=0.):
    end=np.searchsorted(s['t'],t+1e-9,side='right')
    for k in range(j,end):f.strain(s['t'][k],s['b'][k],s['R'][k],paired)
    f.predict(t);return int(end)

def estimate_clock(t,s,v,c=Settings(),augmented=True):
    """Release a causal estimate at each query time, with exact exposure-time rewind and ADC replay."""
    t=np.asarray(t);check_packet(s,False);check_packet(v,True)
    if not len(t) or np.any(np.diff(t)<=0):raise ValueError('Output clock must be monotone')
    n=len(t);z=np.full((n,2),np.nan);bias=np.zeros((n,2));cov=np.zeros((n,2,2))
    gs=np.zeros(n);gv=np.zeros(n);cv=np.zeros(n);cs=np.zeros(n);used=np.zeros(n,bool);nis=np.full(n,np.nan)
    arrival=v['t']+c.delay;ti=min(t[0],s['t'][0] if len(s['t']) else t[0],v['t'][0] if len(arrival) else t[0])
    master=BendFilter(c,augmented,ti);live=master.copy();ms=ls=ci=0
    for k,now in enumerate(t):
        got=False
        while ci<len(arrival) and arrival[ci]<=now+1e-9:
            exp=v['t'][ci]
            quality=float(v['quality'][ci])
            reliable=bool(v.get('ready',v['valid'])[ci])
            support=bool(augmented and reliable and v['valid'][ci] and ci and master.used_camera and exp-v['t'][ci-1]<.085)
            # Both bracketing optical exposures have arrived at this release. Jointly condition
            # the intervening ADC events on the same six-state posterior exactly once.
            prior=master.copy() if support else None;old_ms=ms
            ms=advance(master,exp,s,ms,float(support))
            master.camera(exp,v['b'][ci],v['R'][ci],bool(v['valid'][ci]),quality,reliable)
            if support and not master.used_camera:
                master=prior;ms=advance(master,exp,s,old_ms)
                master.camera(exp,v['b'][ci],v['R'][ci],True,quality,reliable)
            nis[k]=master.nis;used[k]=master.used_camera;gv[k]=master.gain_v;cv[k]=quality if v['valid'][ci] else 0.
            ci+=1;got=True
        if got:live=master.copy();ls=ms
        ls=advance(live,now,s,ls)
        if ls:
            z[k]=live.x[:2];cov[k]=live.P[:2,:2]
            if augmented:bias[k]=live.x[4:]
            older=np.searchsorted(s['t'],t[k-1]+1e-9,side='right') if k else 0
            if older<ls:gs[k]=live.gain_s
            sr=float(np.trace(live.last_sr)/2)
            cs[k]=1./(1.+sr/(RAD**2))
        if not got and ci and v['valid'][ci-1] and now-arrival[ci-1]<.065:cv[k]=float(v['quality'][ci-1])
    return dict(b=z,bias=bias,P=cov,gain_v=gv,gain_s=gs,confidence_v=cv,confidence_s=cs,accepted=used,nis=nis)

def vision_release(t,v,delay=.20):
    """Optical-only angular velocity compensates transport latency, stopping on the first missing frame."""
    ti=v['t'];b=v['b'];ok=v['valid'];pred=np.full((len(t),2),np.nan)
    if not len(ti):return pred
    rates=np.zeros_like(b);last=-np.inf
    for j in range(len(ti)):
        if not ok[j]:last=-np.inf;continue
        if j and ti[j]-last<.085:
            delta=(b[j]-b[j-1])/(ti[j]-last);speed=np.linalg.norm(delta)
            delta*=min(1.,np.deg2rad(65)/max(speed,1e-10))
            alpha=-np.expm1(-(ti[j]-last)/.12)
            rates[j]=(1-alpha)*rates[j-1]+alpha*delta
        last=ti[j]
    ii=np.searchsorted(ti+delay,t+1e-9,side='right')-1;idx=np.maximum(0,ii)
    use=(ii>=0)&ok[idx]&(t-ti[idx]-delay<.065)
    pred[use]=b[idx[use]]+rates[idx[use]]*(t[use]-ti[idx[use]])[:,None]
    return pred
def strain_release(t,s):
    idx=np.searchsorted(s['t'],t+1e-9,side='right')-1
    out=np.full((len(t),2),np.nan);use=idx>=0
    out[use]=s['b'][idx[use]];return out
