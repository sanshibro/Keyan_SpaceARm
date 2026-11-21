"""Vector-valued evaluation of the upstream cubic and its identical reference clock."""
import numpy as np
from scipy.interpolate import BSpline
from uav_continuum_whole_body_v2.trajectory import SplineTrajectory

def proof_times(curve,dt,dp,da):
    """Vectorized upstream 40 ms, 1 cm, 0.5° grid including original knot times."""
    if not np.isfinite(curve.duration) or not 0.<curve.duration<=900.:
        raise ValueError('invalid geometry proof duration')
    base=np.unique(np.r_[curve.span_boundaries,np.arange(0.,curve.duration,dt),curve.duration])
    q=np.asarray(curve.evaluate(base,0))
    if q.shape!=(len(base),6) or not np.isfinite(q).all():raise ValueError('non-finite geometry path')
    delta=np.diff(q,axis=0)
    phi=np.abs(np.arctan2(np.sin(delta[:,5]),np.cos(delta[:,5])))
    weighted=phi*np.minimum(1.,np.maximum(abs(q[:-1,4]),abs(q[1:,4])))
    steps=np.maximum.reduce((np.linalg.norm(delta[:,:3],axis=1)/dp,abs(delta[:,3])/da,
                             np.maximum(abs(delta[:,4]),weighted)/da))
    if np.max(steps)>2000.:raise ValueError('geometry proof sample budget exceeded')
    count=np.maximum(1,np.ceil(steps).astype(int))
    if count.sum()+1>100000:raise ValueError('geometry proof sample budget exceeded')
    interval=np.repeat(np.arange(len(count)),count)
    end=np.cumsum(count);order=np.arange(len(interval))+1-(end-count)[interval]
    values=base[interval]+np.diff(base)[interval]*(order/count[interval])
    values[end-1]=base[1:]
    return np.r_[base[0],values]

def moving_boundary(p,h,rate):
    """Set the measured tangent and project excessive entry acceleration at a clamped cubic knot."""
    p[1]=p[0]+rate*h[0]/3.
    acc=6./h[0]*((p[2]-p[1])/(h[0]+h[1])-rate/3.)
    size=np.linalg.norm(acc[:3])
    if size>.83:acc[:3]*=.83/size
    acc[3:]=np.clip(acc[3:],-np.deg2rad([73.,82.,98.]),np.deg2rad([73.,82.,98.]))
    p[2]=p[1]+(h[0]+h[1])*(rate/3.+h[0]*acc/6.)

class VectorSpline(SplineTrajectory):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.vector=BSpline(self.knots,self.control_states,self.degree,axis=0,extrapolate=False)
    def evaluate(self,times,derivative=0):
        t=np.atleast_1d(np.asarray(times,float))
        phase,r,a,j=self._phase(t)
        if derivative==0:return self.vector(phase)
        v=self.vector(phase,nu=1)
        if derivative==1:return v*r[:,None]
        acc=self.vector(phase,nu=2)
        if derivative==2:return acc*r[:,None]**2+v*a[:,None]
        if derivative==3:return self.vector(phase,nu=3)*r[:,None]**3+3*acc*(r*a)[:,None]+v*j[:,None]
        return np.zeros((len(t),6))
