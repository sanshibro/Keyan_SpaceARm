"""Select a dynamically admissible receiving phase for the native C² bridge."""
import math

import numpy as np
from super_handoff import coordinates, rates, accelerations


def limits(duration, coefficients):
    # Coefficients use dimensionless u=t/T. Evaluate every 10 ms before
    # commissioning a bridge. Geometry receives its own dense native check.
    u=np.linspace(0.,1.,int(math.ceil(duration/.01))+1)
    def derivative(order):
        basis=np.zeros((len(u),6))
        for p in range(order,6):
            basis[:,p]=math.factorial(p)/math.factorial(p-order)*u**(p-order)/duration**order
        return basis@coefficients
    q,v,a,j=(derivative(i) for i in range(4))
    b=q[:,:,4:];vb=v[:,:,4:];ab=a[:,:,4:]
    theta=np.linalg.norm(b,axis=2);radius=np.maximum(theta,1.e-8)
    unit=b/radius[:,:,None];tan=np.stack((-unit[:,:,1],unit[:,:,0]),axis=2)
    dr=np.sum(unit*vb,axis=2);dp=np.sum(tan*vb,axis=2)/radius
    ddr=np.sum(unit*ab,axis=2)+theta*dp**2
    ddp=(np.sum(tan*ab,axis=2)-2*dr*dp)/radius
    vel=np.linalg.norm(v[:,:,:3],axis=2).max(axis=1)
    acc=np.linalg.norm(a[:,:,:3],axis=2).max(axis=1)
    jerk=np.linalg.norm(j[:,:,:3],axis=2).max(axis=1)
    # At the straight pole the motion is a variation of bending vector,
    # irrespective of its indeterminate polar azimuth.
    dr=np.where(theta>.03,dr,np.linalg.norm(vb,axis=2))
    ddr=np.where(theta>.03,ddr,np.linalg.norm(ab,axis=2))
    metrics=np.stack((vel,acc,jerk,np.max(abs(v[:,:,3]),axis=1),np.max(abs(a[:,:,3]),axis=1),
        np.max(abs(dr),axis=1),np.max(abs(ddr),axis=1),
        np.max(np.where(theta>.03,abs(dp),0.),axis=1),np.max(np.where(theta>.03,abs(ddp),0.),axis=1)),axis=1)
    bounds=np.r_[.60,1.,3.,np.deg2rad([45.,90.,50.,100.,60.,120.])]
    return np.max(metrics/bounds,axis=1)


def receiving_options(old,curve,offset,first_duration):
    q,v,a=old
    start,velocity,acc=coordinates(q),rates(q,v),accelerations(q,v,a)
    options=[]
    for duration in (first_duration,.8,1.1,1.45,1.8,2.15,2.55,3.):
        a0=start;a1=velocity*duration;a2=.5*acc*duration**2
        phases=np.unique(np.clip(offset+duration+np.array([-1.20,-.80,-.45,-.20,0.,.20,.45,.80]),max(0.,offset-.20),curve.duration-1.e-6))
        if not len(phases) or phases[0]<0.:continue
        qs,vs,acs=(curve.evaluate(phases,k) for k in range(3))
        dest=np.array([coordinates(x) for x in qs])
        dest[:,3]=start[3]+(dest[:,3]-start[3]+np.pi)%(2*np.pi)-np.pi
        terminal_v=np.array([rates(x,y) for x,y in zip(qs,vs)])
        terminal_a=np.array([accelerations(x,y,z) for x,y,z in zip(qs,vs,acs)])
        pos=dest-a0-a1-a2;vel=terminal_v*duration-a1-2.*a2;ac=terminal_a*duration**2-2.*a2
        c=np.zeros((len(phases),6,6));c[:,0]=a0;c[:,1]=a1;c[:,2]=a2
        c[:,3]=10.*pos-4.*vel+.5*ac;c[:,4]=-15.*pos+7.*vel-ac;c[:,5]=6.*pos-3.*vel+.5*ac
        ratios=limits(duration,c)
        for i in np.flatnonzero(ratios<=1.):
            weight=duration*.21+ratios[i]*.36+.09*abs(phases[i]-offset-duration)
            options.append((float(weight),duration,float(phases[i]),float(ratios[i])))
    return sorted(options)
