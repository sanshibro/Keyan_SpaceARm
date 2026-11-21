"""Analytical regressions for trim direction, static transfer and projected escape."""
import json
import numpy as np
from m2_servo import GravityTrim,ShapeObservation,project_bend

def check():
    rotation=np.eye(3);calibration=np.diag([.45,.45,.03])
    trim=GravityTrim(calibration,True);off=GravityTrim(calibration,False)
    mass=2.295;centre=np.array([.012,-.008,-.23])
    for f in (trim,off):f.evaluate(np.zeros(3),mass,rotation,.02)
    acc,tau,angle=trim.evaluate(centre,mass,rotation,.02)
    ac0,t0,a0=off.evaluate(centre,mass,rotation,.02)
    # A forward mass increases the nose-down moment. PX4 commands nose up
    # (positive pitch FRD) to compensate; acceleration is backwards in ENU.
    assert np.allclose(tau,[-mass*9.81*.008,mass*9.81*.012])
    assert np.allclose(tau,t0) and acc[0]<0 and acc[1]>0 and np.max(abs(ac0))==0 and np.max(abs(a0))==0
    # Rotor thrust on the longitudinal COM axis has no moment under pitch.
    axial=GravityTrim(calibration,True)
    for deg in [-8.,-6.,0.,6.,8.]:
        u=np.deg2rad(deg);R=np.array([[np.cos(u),0,np.sin(u)],[0.,1.,0.],[-np.sin(u),0,np.cos(u)]])
        _, moment, correction=axial.evaluate(R@np.array([0.,0.,-.23]),mass,R,.02,.20)
        assert np.allclose(moment,0.,atol=1e-12) and np.allclose(correction,0.,atol=1e-12)
    R=np.eye(3)
    _, moment, _=axial.evaluate(centre,mass,R,.02,.20)
    assert np.allclose(moment,np.asarray([-.008,.012])*mass*(9.81+.20))
    for _ in range(400):steady=trim.evaluate(centre,mass,rotation,.02)[0]
    assert np.linalg.norm(steady)<1e-4 and np.linalg.norm(acc)>.1
    def clearance(b):return .006+.05*b[1]
    b=np.zeros(2);des=np.array([.4,-.15]);projected,reason=project_bend(b,des,.006,clearance)
    assert reason=='projected' and clearance(projected)>=.01099 and abs(projected[0]-des[0])<1e-8
    assert np.array_equal(project_bend(b,des,.016,clearance)[0],des)
    obs=ShapeObservation()
    obs.update(0,[0,0,0],[0,0,0],np.pi-.015,[1.,-.02])
    q,v,a=obs.update(.02,[.004,0,0],[.2,0,0],-np.pi+.015,[1.,.03])
    assert abs(v[3])<2 and np.isfinite(a).all()
    return dict(compensation_Nm_frd=tau.tolist(),incremental_tilt_deg=np.rad2deg(angle).tolist(),
         initial_command_acc_enu=acc.tolist(),steady_acc_enu=steady.tolist(),
         projected_gap_mm=1000*clearance(projected),yaw_derivative_across_wrap_rad_s=float(v[3]))
def native():
    import inspect,time
    from m2_flight import EstimatedMechanism,PERIOD
    from m2_plant import build
    from round_f_sitl import run
    p=build();obs=EstimatedMechanism(p)
    poses=np.column_stack((np.linspace(0,98,32),np.linspace(60,220,32)))
    times=[];clear=[];fitting=[];com=[];errors=[];recovery=[]
    for theta,phi in poses:
        t=time.perf_counter()
        b=np.deg2rad(theta)*np.array([np.cos(np.deg2rad(phi)),np.sin(np.deg2rad(phi))])
        h,fit=obs.read(np.zeros(3),np.eye(3),b);com.append(obs.com(np.zeros(3),np.eye(3),b))
        times.append(1000*(time.perf_counter()-t))
        clear.append(float(h['lower']));fitting.append(fit['all_lower'])
        lo,hi=obs.audit.bounds()
        exact=obs.audit._scan('reference',obs.hazard,lo,hi)
        if h['lower']>exact['upper']+1e-8:raise RuntimeError('Online lower bound exceeds full CAD upper witness')
        if h['lower']>exact['upper']+1e-8 or h['upper']<exact['lower']-1e-8:
            raise RuntimeError('Geometry certificate intervals are disjoint')
        errors.append(abs(h['lower']-exact['lower']))
        if .001<=h['lower']<.008 and len(recovery)<2:
            t0=time.perf_counter()
            def evaluate(shape):return obs.read(np.zeros(3),np.eye(3),shape,False)[0]['lower']
            next_b,kind=project_bend(b,b.copy(),h['lower'],evaluate)
            delta=evaluate(next_b)
            if delta<=h['lower'] or kind!='projected':raise RuntimeError('Active CAD recovery failed to increase separation')
            recovery.append(dict(initial_mm=1000*h['lower'],projected_mm=1000*delta,
                motion_deg=float(np.rad2deg(np.linalg.norm(next_b-b))),wall_ms=1000*(time.perf_counter()-t0)))
    src=inspect.getsource(run)
    assert 'translation_feedforward' in src and '2048' in src and 'yaw_enu' in src
    assert np.isfinite(times+clear).all() and np.all(np.array(fitting)>0)
    return dict(scene=p.scene_digest,physics_step_ms=1000*p.m.opt.timestep,shape_period_ms=1000*PERIOD,
        nposes=len(poses),pairs=len(obs.hazard),fit_pairs=len(obs.fit),com_diversity_mm=(1000*np.ptp(com,axis=0)).tolist(),
        minimum_hazard_mm=1000*min(clear),fit_lower_mm=(1000*np.min(fitting,axis=0)).tolist(),
        observation_fk_gjk_ms_median=float(np.median(times)),observation_fk_gjk_ms_max=max(times),
        full_CAD_check_max_difference_mm=1000*max(errors),active_recovery=recovery)

if __name__=='__main__':
    import argparse
    from pathlib import Path
    parser=argparse.ArgumentParser();parser.add_argument('--native',action='store_true');parser.add_argument('--output',type=Path)
    args=parser.parse_args();report=check()
    if args.native:report['native']=native()
    content=json.dumps(report,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(content)
    print(content)
