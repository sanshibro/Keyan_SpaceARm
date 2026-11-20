"""Score each independent trajectory from integrated robot geometry and the native clock."""
import argparse
import json
import hashlib
from pathlib import Path
import mujoco
import numpy as np
from gate_feedback_model import OUT,P,digest,dump,make_plant
from gate_distance_audit import DistanceAudit
from round_aug_vis import bend

def magnitude(b):
    return np.rad2deg(np.linalg.norm(b,axis=1))

def azimuth(b):
    return np.rad2deg(np.arctan2(b[:,1],b[:,0]))

def wrapped(x):
    return (x+180.)%360.-180.

def rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else None

def errors(estimate,truth,mask):
    exists=np.isfinite(estimate).all(axis=1)
    theta=magnitude(truth)
    m=mask&exists
    # At a straight configuration the bending plane has no physical azimuth.
    angular=m&(theta>=P['phi_min_deg'])&(magnitude(estimate)>=P['phi_min_deg'])
    theta_error=magnitude(estimate[m])-theta[m]
    phi_error=wrapped(azimuth(estimate[angular])-azimuth(truth[angular]))
    return dict(theta_rmse_deg=rms(theta_error),phi_rmse_deg=rms(phi_error),
                theta_peak_deg=float(np.max(abs(theta_error))) if m.any() else None,
                phi_peak_deg=float(np.max(abs(phi_error))) if angular.any() else None,
                samples=int(mask.sum()),available=int(m.sum()),azimuth_samples=int(angular.sum()))

def dwell_times(t,mask):
    result=np.zeros(len(t));start=None
    for i,ok in enumerate(mask):
        if not ok:
            start=None
        elif start is None or (i and t[i]-t[i-1]>.025):
            start=t[i]
        if start is not None:result[i]=t[i]-start
    return result

def score(folder,plant=None):
    folder=Path(folder)
    run=json.loads((folder/'run.json').read_text())
    info=json.loads((folder/'control-summary.json').read_text())
    if run['error'] is not None or run['release_us'] is None:raise ValueError('Incomplete flight acquisition '+str(folder))
    raw=np.load(folder/'physics.npz',allow_pickle=False)
    ctl=np.load(folder/'control.npz',allow_pickle=False)
    all_t=(raw['us']-run['release_us'])*1e-6
    keep=(all_t>=ctl['t'][0]-1e-6)&(all_t<=P['deadline']+1e-8)
    t=all_t[keep];qpos=raw['qpos'][keep]
    if t[-1]<P['deadline']-.021 or max(np.diff(t))>.02001:raise ValueError('Scoring clock has a gap')
    indices=np.searchsorted(ctl['t'],t+1e-9,side='right')-1
    if min(indices)<0:raise ValueError('No control observation at scoring time')
    if max(t-ctl['t'][indices])>.02001:raise ValueError('Flight/control clocks disagree')
    c={k:ctl[k][indices] for k in ctl.files}
    provenance=json.loads((folder/'provenance.json').read_text())
    xml_path=next((f for f in (folder/'model.xml',folder/'optics/sensor-canvas.xml') if f.exists()),None)
    if xml_path is not None:
        xml=xml_path.read_bytes()
        if hashlib.sha256(xml).hexdigest()!=provenance['plant_xml_sha256']:raise ValueError('Model fingerprint differs from flight')
        model=mujoco.MjModel.from_xml_string(xml.decode())
    else:
        plant=plant or make_plant(canvas=False)
        if hashlib.sha256(plant.xml.encode()).hexdigest()!=provenance['plant_xml_sha256']:raise ValueError('Flight model required')
        model=plant.m
    audit=DistanceAudit(model)
    m=model;d=audit.d;axis=np.asarray(info['goal_axis']);target=np.asarray(info['goal_xyz'])
    if (folder/'protocol.json').exists():
        path=folder/'protocol.json'
        if digest(path)!=provenance['protocol_sha256']:raise ValueError('Scene protocol differs from the flying copy')
        registered=json.loads(path.read_text())
        if 'target_axis_enu' in registered:
            fixed_axis=np.asarray(registered['target_axis_enu']);fixed_target=np.asarray(registered['target'])
            if not np.allclose(fixed_axis,axis,rtol=0.,atol=1e-9) or not np.array_equal(fixed_target,target):
                raise ValueError('Controller goal differs from the registered score reference')
            axis,target=fixed_axis,fixed_target
    series={k:[] for k in ('true_b','tip_error','axis_error_deg','body_clearance','arm_tool_clearance',
                             'self_clearance','near_robot','near_gate','body_lower','arm_lower','self_lower',
                             'body_upper','arm_upper','self_upper','rear','tip','tool_axis','body_pose','inside_aperture',
                             'body_pair','arm_pair','self_pair','arm_near_robot','arm_near_gate')}
    red=P['thresholds']['external_m']
    for q in qpos:
        dis=audit.score(q)
        if any(dis['details'][g]['lower']<red<=dis['details'][g]['upper'] for g in ('body','arm_tool')):
            tol=audit.tolerance
            try:
                audit.tolerance=1e-7
                dis=audit.score(q)
            finally:audit.tolerance=tol
        tip=d.site('tip').xpos.copy();tool_axis=d.site('tip').xmat.reshape(3,3)[:,2].copy()
        shape=bend(d.body('mount').xmat.reshape(3,3).T@d.body('tool').xmat.reshape(3,3))
        lo,hi=audit.bounds()
        gate=m.geom('G1_sill');lintel=m.geom('G1_lintel')
        gx=d.geom_xpos[gate.id,0];thickness=gate.size[0]
        intersect=audit.robot[(hi[audit.robot,0]>=gx-thickness)&(lo[audit.robot,0]<=gx+thickness)]
        aperture=bool(np.all(lo[intersect,1]>=d.geom_xpos[m.geom('G1_jamb_negative').id,1]+m.geom('G1_jamb_negative').size[1])
                      and np.all(hi[intersect,1]<=d.geom_xpos[m.geom('G1_jamb_positive').id,1]-m.geom('G1_jamb_positive').size[1])
                      and np.all(lo[intersect,2]>=d.geom_xpos[gate.id,2]+gate.size[2])
                      and np.all(hi[intersect,2]<=d.geom_xpos[lintel.id,2]-lintel.size[2]))
        lower=lambda g:dis['details'][g]['lower']
        high=lambda g:dis['details'][g]['upper']
        pair=lambda g:[dis['details'][g]['geom_a'],dis['details'][g]['geom_b']]
        s=dict(true_b=shape,tip_error=np.linalg.norm(tip-target),axis_error_deg=np.rad2deg(np.arccos(np.clip(tool_axis@axis,-1,1))),
               body_clearance=dis['body_m'],arm_tool_clearance=dis['arm_tool_m'],self_clearance=dis['self_m'],
               near_robot=dis['near_robot'],near_gate=dis['near_gate'],arm_near_robot=dis['arm_near_robot'],arm_near_gate=dis['arm_near_gate'],
               body_lower=lower('body'),arm_lower=lower('arm_tool'),self_lower=lower('self'),
               body_upper=high('body'),arm_upper=high('arm_tool'),self_upper=high('self'),
               tip=tip,tool_axis=tool_axis,body_pose=q[:7].copy(),body_pair=pair('body'),arm_pair=pair('arm_tool'),self_pair=pair('self'),
               inside_aperture=aperture,rear=float(lo[audit.robot,0].min()-gx-thickness))
        for name,val in s.items():series[name].append(val)
    series={k:np.asarray(v) for k,v in series.items()}
    view=info['case']['vision'];cut=(t>=P['window']['blackout_start']-1e-9)&(t<P['window']['blackout_end']-1e-9)
    blind=cut&(view=='occluded')
    estimated_exit=c['gate_exit'];true_exit=series['rear']>=0
    dock=true_exit&estimated_exit
    on_target=dock&(series['tip_error']<=P['thresholds']['position_m'])&(series['axis_error_deg']<=P['thresholds']['axis_deg'])
    dwell=dwell_times(t,on_target);done=np.flatnonzero(dwell>=P['thresholds']['dwell_s']-1e-8)
    terminal=(t>=P['terminal_window'][0])&(t<=P['terminal_window'][1])
    full=errors(c['est_b'],series['true_b'],np.ones(t.shape,bool))
    blackout=errors(c['est_b'],series['true_b'],blind)
    hold_diagnostic=errors(c['vision_hold_b'],series['true_b'],blind)
    difference=np.diff(t,append=P['deadline']);holds=c['hold']!='none'
    contact=int(raw['contact_steps'][keep][-1]) if 'contact_steps' in raw.files else int(run['contact_integration_steps'])
    safe=bool(series['body_lower'].min()>=red and series['arm_lower'].min()>=red and series['self_lower'].min()>0
              and contact==0 and series['inside_aperture'].all())
    dt=float(np.median(np.diff(t)))
    s=dict(case=info['case'],success=bool(safe and len(done)),safe=safe,
           crossed=bool(true_exit.any() and estimated_exit.any()),dwell_reached=bool(len(done)),
           settled_at_s=float(t[done[0]]) if len(done) else None,
           completion_s=float(t[done[0]]) if safe and len(done) else None,
           estimated_rear_exit_s=info['exit_time'],physical_rear_exit_s=float(t[np.flatnonzero(true_exit)[0]]) if true_exit.any() else None,
           body_min_m=float(series['body_clearance'].min()),arm_tool_min_m=float(series['arm_tool_clearance'].min()),
           self_min_m=float(series['self_clearance'].min()),physical_contact_steps=contact,
           hold_s=float(np.sum(difference[holds])),feedback_hold_s=float(np.sum(difference[c['hold']=='feedback'])),
           clearance_hold_s=float(np.sum(difference[c['hold']=='clearance'])),max_dwell_s=float(np.max(dwell)),
           tip_rmse_m=rms(series['tip_error'][terminal]),tip_peak_m=float(np.max(series['tip_error'][terminal])),
           axis_rmse_deg=rms(series['axis_error_deg'][terminal]),axis_peak_deg=float(np.max(series['axis_error_deg'][terminal])),
           theta_rmse_deg=full['theta_rmse_deg'],phi_rmse_deg=full['phi_rmse_deg'],
           theta_blackout_rmse_deg=blackout['theta_rmse_deg'],phi_blackout_rmse_deg=blackout['phi_rmse_deg'],
           whole_angle=full,blackout_angle=blackout,vision_hold_blackout=hold_diagnostic,
           dynamic_theta_range_deg=float(np.ptp(magnitude(series['true_b'][cut]))),
           dynamic_phi_range_deg=float(np.rad2deg(np.ptp(np.unwrap(np.deg2rad(azimuth(series['true_b'][cut])))))),
           true_max_bend_deg=float(magnitude(series['true_b']).max()),true_max_total_curvature_deg=run['max_total_curvature_deg'],
           source_valid_coverage=float(np.mean(c['shape_valid'])),terminal_after_gate_samples=int((terminal&dock).sum()),
           passed_inside_aperture=bool(series['inside_aperture'].all()),
           sample_period_s=dt,physics_samples=int(len(t)),max_control_alignment_s=float(max(t-ctl['t'][indices])),
           witness_bound_width_m={g:float(np.max(series[g+'_upper']-series[g+'_lower'])) for g in ('body','arm','self')},
           goal_xyz=target.tolist(),goal_axis=axis.tolist())
    for key,field in [('body','body_clearance'),('arm','arm_tool_clearance'),('self','self_clearance')]:
        i=int(np.argmin(series[field]));s[key+'_nearest']={'t':float(t[i]),'geoms':[audit.names[j] for j in series[key+'_pair'][i]]}
    trace=dict(t=t,**series,des_b=c['des_b'],est_b=c['est_b'],issued_b=c['issued_b'],vision_hold_b=c['vision_hold_b'],
               phase=c['phase'],hold=holds,hold_reason=c['hold'],gate_exit=estimated_exit,blackout=blind,
               dwell_s=dwell,shape_valid=c['shape_valid'],true_exit=true_exit)
    np.savez_compressed(folder/'score-trace.npz',**trace)
    s['artifacts']={n:digest(folder/n) for n in ('flight.ulg','flight.bag','physics.npz','control.npz','run.json','provenance.json','control-summary.json','score-trace.npz')}
    if not info['mechanics_only']:
        optical=json.loads((folder/'optics/complete.json').read_text())
        if not optical['frozen_sources_unchanged']:raise RuntimeError('Optical freeze audit failed')
        s['optics']=dict(frames=optical['frames'],valid=optical['valid'],native_csv_sha256=optical['native_csv_sha256'])
        for n in ('optics/native.csv','optics/index.jsonl','optics/observations.npz','sensors/strain_observations.npz','sensors/estimate_proposed.npz'):
            s['artifacts'][n]=digest(folder/n)
    s['scorer_sha256']=digest(__file__)
    dump(folder/'metrics.json',s)
    print(folder.name,s['success'],s['body_min_m'],s['arm_tool_min_m'],s['axis_rmse_deg'],flush=True)
    return s

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('case_folder',type=Path)
    args=parser.parse_args();score(args.case_folder)
