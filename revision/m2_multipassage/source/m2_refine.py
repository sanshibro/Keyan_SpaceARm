"""Bounded spatial and nonuniform time refinement around a native six-DOF seed."""
import time
import mujoco
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize
from m2_spline import VectorSpline,moving_boundary

def refine(k,route,rate,final,stationary):
    ts=time.monotonic()
    base=k.make_curve(route,rate,final,stationary_start=stationary)
    points=base.control_states;h=base.span_durations;K=len(points);N=len(h)
    # Collocation is evaluated in the geometric parameter so it is independent
    # of the common smooth reference-clock ramps.
    u=np.unique(np.r_[np.linspace(0.,base.geometric_duration,39),np.cumsum(h)])
    Bsp=BSpline(base.knots,np.eye(K),3)
    B=Bsp(u);D=Bsp(u,nu=1);A=Bsp(u,nu=2)
    q0=B@points
    norm=np.array([.032,.050,.036,.19,.22,.26])
    perturb=np.array([.0014,.0014,.0014,.006,.006,.006])
    # Finite differences of the full convex support distance (translation and
    # orientation of the true CAD). Only the local objective is linearized.
    samples=[q0]+[q0+np.eye(6)[j]*perturb[j] for j in range(6)]
    dist,_=k.geometry.path_batch.evaluate(np.vstack(samples),caps=(.147,.147,.045,.004))
    dist=dist.reshape(7,len(u),5)
    initial=dist[0,:,:3];G=((dist[1:,:,:3]-initial)/perturb[:,None,None]).transpose(1,2,0)
    G=np.clip(G,-1.5,1.5)
    if k.coupled:
        def com(states):
            vals=[];g=k.geometry;d=g.fk
            for s in states:
                g.place(d,[0,0,0,0,s[4],s[5]]);mujoco.mj_kinematics(g.m,d);mujoco.mj_comPos(g.m,d)
                vals.append(d.subtree_com[g.u,:2].copy())
            return np.asarray(vals)
        pivot=com(q0)
        comgrad=np.zeros((len(u),2,6))
        for j in (4,5):comgrad[:,:,j]=(com(samples[j+1])-pivot)/perturb[j]
    mid=np.cumsum(h)-.5*h
    clock=np.column_stack([np.interp(u,mid,np.eye(N)[:,j]) for j in range(N)])
    free=np.arange(1 if stationary else 2,K-1)
    width=len(free)*6
    spatial_lim=np.ones((len(free),6))
    if k.method in ('straight','fixed'):spatial_lim[:,4:]=0.
    if k.method=='straight':spatial_lim[:,3]=0.
    if k.method=='sequential':spatial_lim[:,:4]=0.
    limits=[(-float(a),float(a)) for a in spatial_lim.ravel()]+[(-.10,.20)]*N
    reg=np.array([.099,.101,.0175]);weights=np.array([3.2e4,3.2e4,5.e4])
    bendfac=np.column_stack((np.ones(len(u)),np.minimum(1.,q0[:,4])))
    last={}
    def evaluate(x):
        if 'x' in last and np.array_equal(last['x'],x):return last['value']
        corr=np.zeros_like(points);corr[free]=x[:width].reshape(-1,6)*norm
        p=points+corr;dq=B@corr;ls=x[width:];factor=np.exp(clock@ls);dur=h*np.exp(ls)
        timecost=.10*dur.sum();gh=.10*dur
        gap=initial+np.einsum('ngd,nd->ng',G,dq)
        short=np.minimum(gap-reg,0)
        # Target working clearance is soft; only the complete 85/15 mm
        # native certificate admits the refined candidate.
        M=len(u);loss=np.sum(short*short*weights)/M
        gdq=2*np.einsum('ng,ngd->nd',short*weights,G)/M
        excess=np.maximum(p[:,4]-k.geometry.theta_max,0)+np.minimum(p[:,4],0)
        gp=B.T@gdq;gp[:,4]+=600*excess
        loss+=300*np.dot(excess,excess)
        vel=D@p;acc=A@p
        if k.coupled:
            co=pivot+np.einsum('ngd,nd->ng',comgrad,dq)
            loss+=32*np.sum(co*co)/M;gp+=B.T@(64*np.einsum('ng,ngd->nd',co,comgrad)/M)
        # Common base smoothness, plus the arm rate/acceleration and timing
        # terms only for the registered coupling-enabled planner.
        vw=np.array([.004,.004,.004,.008,.10 if k.coupled else 0,.10 if k.coupled else 0])
        aw=np.array([.026,.026,.026,.012,.055 if k.coupled else 0,.055 if k.coupled else 0])
        vweight=np.tile(vw,(M,1));aweight=np.tile(aw,(M,1));vweight[:,4:]*=bendfac**2;aweight[:,4:]*=bendfac**2
        lossvel=(vweight*vel**2).sum(axis=1)/factor/M
        lossacc=(aweight*acc**2).sum(axis=1)/factor**3/M
        loss+=lossvel.sum()+lossacc.sum()
        gp+=D.T@(2*vweight*vel/factor[:,None]/M)+A.T@(2*aweight*acc/factor[:,None]**3/M)
        gh-=clock.T@(lossvel+3*lossacc)
        if k.coupled:
            dh=np.diff(ls);loss+=.75*np.dot(dh,dh)
            gh[:-1]-=1.5*dh;gh[1:]+=1.5*dh
        # Trust region around the observed seed controls and coupling-independent
        # continuity of local control corrections.
        xs=x[:width].reshape(-1,6)
        ds=np.diff(xs,axis=0);loss+=.0025*np.sum(xs*xs)+.0035*np.sum(ds*ds)
        gg=gp[free]*norm+.005*xs;gg[:-1]-=.007*ds;gg[1:]+=.007*ds
        res=(float(loss+timecost),np.r_[gg.ravel(),gh]);last.update(x=x.copy(),value=res);return res
    x0=np.zeros(width+N);warm=k.last_opt is not None and len(k.last_opt)==len(x0)
    if warm:x0[:]=.25*k.last_opt
    opt=minimize(evaluate,x0,jac=True,method='L-BFGS-B',bounds=limits,options={'maxiter':28,'ftol':2e-7,'gtol':1e-4})
    x=opt.x;k.last_opt=x.copy()
    new=points.copy();new[free]+=x[:width].reshape(-1,6)*norm;new[:,4]=np.clip(new[:,4],0,k.geometry.theta_max)
    spans=h*np.exp(x[width:])
    if not stationary:moving_boundary(new,spans,np.asarray(rate))
    result=VectorSpline(new,spans,stationary_start=stationary,stationary_end=True,boundary_ramp_duration=base.boundary_ramp_duration)
    return result,base,dict(wall_s=time.monotonic()-ts,calls=opt.nfev,status=str(opt.message),
        algorithm='L-BFGS-B with convex-distance local Jacobians',warm_coefficients=warm,initial=x0.tolist(),best=x.tolist())
