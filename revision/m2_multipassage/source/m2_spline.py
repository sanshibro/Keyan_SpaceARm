"""Vector-valued evaluation of the upstream cubic and its identical reference clock."""
import numpy as np
from scipy.interpolate import BSpline
from uav_continuum_whole_body_v2.trajectory import SplineTrajectory

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
