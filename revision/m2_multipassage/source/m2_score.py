"""Independent geometry, native dynamics and synchronized wrist-depth audit of an M2 flight."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image
from gate_distance_audit import DistanceAudit
from m2_self_registry import partition
from m2_scene import FACE,AXIS
from m2_plant import build,task_pose
from m2_gate import LOCK
from m1_score import bend

def read(p):return json.loads(Path(p).read_text())
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()
def dwell(t,ok,max_step):
    length=best=0.;prev=None
    for stamp,valid in zip(t,ok):
        if valid:
            length=length+stamp-prev if prev is not None and stamp-prev<=max_step else 0.
            best=max(best,length);prev=stamp
        else:length=0.;prev=None
    return float(best)
def write(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def logged_inputs(f,run):
    import rosbag
    with (f/'flight.ulg').open('rb') as stream:
        if stream.read(7)!=b'ULog\x01\x12\x35':raise ValueError('ULog signature')
    with (f/'flight.bag').open('rb') as stream:
        if stream.readline()!=b'#ROSBAG V2.0\n':raise ValueError('ROS bag signature')
    if run['error']:raise ValueError('flight run error')
    with rosbag.Bag(str(f/'flight.bag'),'r') as bag:
        topics=bag.get_type_and_topic_info().topics
        for name in ('/clock','/f/px4/local_position_ned','/f/plant/rotor_thrust_N','/f/arm/bend'):
            if name not in topics or topics[name].message_count<2:raise ValueError('missing time series '+name)
    return bool(run['packets']>0 and run['release_us'] is not None)
def score(f,out,plant=None):
    f=Path(f);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    p=plant or build();m=p.m;task=task_pose(p)
    run=read(f/'run.json');pr=read(f/'provenance.json')['mission'];su=read(f/'control-summary.json')
    if pr['scene']!=p.scene_digest or pr['freeze']!=LOCK:raise ValueError('plant or perception digest')
    native=logged_inputs(f,run)
    c=np.load(f/'control.npz');s=np.load(f/'physics.npz');t=c['t']
    ticks=run['release_us']+np.rint(1e6*t).astype(np.int64);idx=np.searchsorted(s['us'],ticks)
    if not np.array_equal(s['us'][idx],ticks):raise ValueError('independent clock alignment')
    xy=s['qpos'][idx];rot=s['rotor_N'][idx];tip=s['tcp'][idx]
    phase=c['phase'];motion=t>=su['motion_start_s']
    target=np.linalg.norm(tip-np.asarray(task['tool']),axis=1);d=p.d;d.eq_active[0]=0
    audit=DistanceAudit(m,tolerance_m=.00010,strict=False,obstacle_prefix='COL_')
    hazard,fit=partition(m,audit.self_pairs)
    mount=np.any(fit==m.geom('arm1').id,axis=1)
    if int(mount.sum())!=1 or len(fit)!=4:raise ValueError('CAD interface registry')
    groups=dict(body=audit.external['body'],arm=audit.external['arm_tool'],hazard=hazard,mount_fit=fit[mount],finger_fit=fit[~mount])
    from m2_geometry import NativeSelf
    fast=NativeSelf(m)
    bounds={k:[] for k in groups};upper_curve={k:[] for k in groups}
    degree=[];optical_degree=[];uvz=[];torque=[];shape=[];heading=[];tilt=[];cop=[]
    fk=m.body('mount').id;axes=np.asarray(AXIS);camera_id=m.camera('wrist_rgbd').id
    K=.5*360/np.tan(np.deg2rad(m.cam_fovy[camera_id])/2)
    for n,q in enumerate(xy):
        d.qpos[:]=q;d.qvel[:]=s['qvel'][idx[n]];d.ctrl[:]=s['ctrl'][idx[n]]
        mujoco.mj_forward(m,d);mujoco.mj_rnePostConstraint(m,d)
        r=d.subtree_com[m.body_rootid[fk]]-d.xpos[fk];w=d.cfrc_int[fk]
        R=d.xmat[p.base].reshape(3,3)
        torque.append((R.T@(w[:3]+np.cross(r,w[3:])))[:2])
        cop.append((d.subtree_com[p.base]-q[:3])[:2]);tangent=d.site('tip').xmat.reshape(3,3)[:,2]
        degree.append(np.rad2deg(np.arccos(np.clip(tangent@axes,-1.,1.))))
        camera_orientation=d.cam_xmat[camera_id].reshape(3,3)
        optical_degree.append(np.rad2deg(np.arccos(np.clip((-camera_orientation[:,2])@axes,-1.,1.))))
        relative=camera_orientation.T@(FACE-d.cam_xpos[camera_id]);z=-relative[2]
        uvz.append([239.5+K*relative[0]/z,179.5-K*relative[1]/z,z] if abs(z)>1e-9 else [1e6,1e6,0.])
        shape.append(bend(d.xmat[fk].reshape(3,3).T@d.body('tool').xmat.reshape(3,3)))
        heading.append(np.arctan2(R[1,0],R[0,0]));tilt.append(np.rad2deg(np.arccos(np.clip(R[2,2],-1,1))))
        for k,ps in groups.items():
            cert=fast.query(d,ps)
            if cert['lower']<=0 or cert['upper']-cert['lower']>.00011:
                # Native positive-distance bounds need signed EPA for overlap.
                audit.d.qpos[:]=q;mujoco.mj_kinematics(m,audit.d);lo,hi=audit.bounds()
                cert=audit._scan(k,ps,lo,hi)
            bounds[k].append(cert['lower']);upper_curve[k].append(cert['upper'])
    certificates={};curve={};grid=np.flatnonzero(motion)[::10]
    for k,ps in groups.items():
        lower=np.asarray(bounds[k]);upper=np.asarray(upper_curve[k]);n=int(np.argmin(upper))
        audit.d.qpos[:]=xy[n];mujoco.mj_kinematics(m,audit.d);lo,hi=audit.bounds()
        v=audit._scan(k,ps,lo,hi)
        if v['upper']<lower[n]-1e-7 or v['lower']>upper[n]+1e-7:raise ValueError('independent distance certificates disagree at '+k)
        certificates[k]=dict(lower_m=float(lower.min()),upper_m=float(upper[n]),t_s=float(t[n]),names=[v['name_a'],v['name_b']],crosscheck=[v['lower'],v['upper']])
        curve[k]=.5*(lower[grid]+upper[grid])
    degree=np.asarray(degree);uvz=np.asarray(uvz);u,v,z=uvz.T;shape=np.asarray(shape);torque=np.asarray(torque)
    error=np.linalg.norm(xy[:,:3]-c['ref_world'],axis=1)
    # Rotor channels in the log are in PX4 order. Resolve measured thrust
    # into physical body moments with the same registered SI allocation as
    # the plant; a single front/rear difference measures only pitch.
    from super_metric_model import allocation_matrix
    rotor_channels=np.asarray(read(f/'provenance.json')['rotor_channels_to_model'])
    if sorted(rotor_channels.tolist())!=list(range(4)):raise ValueError('rotor channel allocation')
    allocation=allocation_matrix(m)[:,rotor_channels]
    rotor_moment_flu=rot@allocation[1:].T
    diff=(rot[:,0]+rot[:,2])-(rot[:,1]+rot[:,3])
    optical_degree=np.asarray(optical_degree)
    eligible=(phase=='inspect')&(target<=.035)&(degree<=3)&(optical_degree<=3)&(z>=.12)&(z<=.18)&(u>=0)&(u<480)&(v>=0)&(v<360)
    optical_times=[];visible=[];verified=[]
    dataset=read(f/'optics/shared-scene.json')
    stream=read(f/'optics/complete.json')
    if dataset['scene_sha256']!=pr['scene'] or dataset['camera']!='wrist_rgbd' or dataset['view_specific_geometry']:raise ValueError('wrist/plant scene mismatch')
    for name,key in [('native.csv','native_csv_sha256'),('index.jsonl','index_sha256')]:
        if sha(f/'optics'/name)!=stream[key]:raise ValueError('native sensor log checksum: '+name)
    if abs(pr['camera_delay_s']-.20)>1e-9:raise ValueError('optical delay')
    for ln in (f/'optics/index.jsonl').read_text().splitlines():
        o=json.loads(ln);n=np.searchsorted(t,o['t']-1e-8)
        if n>=len(t) or abs(t[n]-o['t'])>1e-7 or hashlib.sha256(xy[n].astype(np.float64).tobytes()).hexdigest()!=o['sensor_qpos_sha256']:raise ValueError('RGB-D state mismatch')
        if abs(o['arrival_t']-o['t']-.20)>1e-7 or o['frame']!=len(optical_times):raise ValueError('optical timing or missing exposure')
        for key in ('rgb','depth'):
            if sha(f/'optics'/o[key])!=o[key+'_sha256']:raise ValueError('optical pixel checksum')
        ok=False;zdepth=0.
        if eligible[n]:
            px,py=np.rint(uvz[n,:2]).astype(int)
            with Image.open(f/'optics'/o['depth']) as im:patch=np.asarray(im)[max(0,py-2):py+3,max(0,px-2):px+3]
            valid=patch[patch>0]
            if len(valid)>=.80*25:
                zdepth=float(np.median(valid))*.001;ok=abs(z[n]-zdepth)<=.015
        optical_times.append(o['t']);visible.append(ok)
        if ok:verified.append([n,o['frame'],zdepth])
    optical_dwell=dwell(optical_times,visible,.061)
    if len(optical_times)!=stream['frames'] or not len(optical_times):raise ValueError('incomplete optical stream')
    first_dock=None;start=None
    for n,(stamp,accepted) in enumerate(zip(optical_times,visible)):
        if not accepted:start=None;continue
        if start is None or (n and stamp-optical_times[n-1]>.061):start=stamp
        if stamp-start>=1.-1e-7:first_dock=start;break
    move=motion&(phase!='inspect')
    change=move&(np.linalg.norm(np.gradient(shape,t,axis=0),axis=1)>np.deg2rad(3.))
    if not change.any():change=move
    insp=np.flatnonzero(phase=='inspect')
    jaw_ids=[m.jnt_qposadr[m.joint('grip'+str(i)).id] for i in range(3)]
    jaw_act=[m.actuator('grip'+str(i)).id for i in range(3)]
    jaw=xy[:,jaw_ids];cmd=s['ctrl'][idx][:,jaw_act]
    jaw_min=jaw.min(axis=0);jaw_max=jaw.max(axis=0)
    # MuJoCo enforces slide ranges through its compliant joint limits.
    # Record excursions and bound them to the compiled slide-limit margin.
    jaw_open=bool(np.max(abs(cmd))<1e-7)
    js=[m.joint('grip'+str(i)).id for i in range(3)]
    jaw_range=(m.jnt_range[js]);undershoot=np.maximum(0,jaw_range[:,0]-jaw_min);overshoot=np.maximum(0,jaw_max-jaw_range[:,1])
    terminal=np.flatnonzero((phase=='inspect')&(t>=t[-1]-1))
    reg=pr['registration']
    summary=dict(strategy=pr['case']['strategy'],seed=pr['case']['seed'],identity=reg['identity'],development=reg['kind']=='development',
        native_logs_verified=native,synchronized_rgbd=bool(len(optical_times)),rgbd_count=len(optical_times),external_lower_m=min(certificates[k]['lower_m'] for k in ('body','arm')),
        hazard_lower_m=certificates['hazard']['lower_m'],fit_lower_m=min(certificates[k]['lower_m'] for k in ('mount_fit','finger_fit')),certificates=certificates,contact_steps=int(run['contact_integration_steps']),
        gripper_open_command=jaw_open,grip_stroke_mm=dict(min=(1000*jaw_min).tolist(),max=(1000*jaw_max).tolist(),below_limit=(1000*undershoot).tolist(),above_limit=(1000*overshoot).tolist()),
        pose_dwell_s=dwell(t,eligible,.031),task_depth_dwell_s=optical_dwell,eligible_depth_count=int(eligible.sum()),verified_depth_count=len(verified),
        control_inspect_request_s=su['first_inspection_s'],elapsed_motion_observation_s=float(t[-1]-su['motion_start_s']),
        motion_to_inspect_s=float(first_dock-su['motion_start_s']) if first_dock is not None else None,position_tracking_peak_m=float(error[change].max()) if change.any() else 0.,
        thrust_diff_rms_N=float(np.sqrt(np.mean(diff[change]**2))) if change.any() else 0.,
        rotor_roll_pitch_rms_Nm=float(np.sqrt(np.mean(np.sum(rotor_moment_flu[change,:2]**2,axis=1)))) if change.any() else None,
        peak_roll_pitch_reaction_Nm=float(np.linalg.norm(torque[change],axis=1).max()) if change.any() else 0.,
        max_theta_deg=float(np.rad2deg(np.linalg.norm(shape,axis=1)).max()),end_body_m=xy[-1,:3].tolist(),
        terminal_error_m=float(np.median(target[terminal])) if len(terminal) else None,terminal_axis_deg=float(np.median(degree[terminal])) if len(terminal) else None,
        terminal_optical_m=float(np.median(z[terminal])) if len(terminal) else None,
        terminal_camera_axis_deg=float(np.median(optical_degree[terminal])) if len(terminal) else None,
        peak_tilt_integration_deg=run['max_tilt_integration_deg'],tilt_motion_p95_deg=float(np.percentile(np.asarray(tilt)[motion],95)),
        raw_ulog_sha256=sha(f/'flight.ulg'),raw_rosbag_sha256=sha(f/'flight.bag'))
    summary['success']=bool(jaw_open and summary['external_lower_m']>=.05 and summary['hazard_lower_m']>=.003 and summary['fit_lower_m']>=0 and not summary['contact_steps'] and optical_dwell>=1.-1e-7)
    np.savez_compressed(out/'series.npz',t=t,qpos=xy,tool=tip,ref=c['ref_world'],actual_b=shape,theta=np.linalg.norm(shape,axis=1),phi=np.arctan2(shape[:,1],shape[:,0]),yaw=heading,tilt=tilt,
        gravity_offset_xy=cop,mount_moment=torque,rotor_N=rot,thrust_diff=diff,tracking=error,depth_samples=verified,uvz=uvz,terminal_position_error=target,axis_error=degree,
        sample_t=t[grid],camera_axis_error=optical_degree,rotor_moment_flu=rotor_moment_flu,**{'gap_'+k:arr for k,arr in curve.items()})
    write(out/'physics-score.json',summary)
    from m2_rolling_audit import audit_rolling
    summary.update(audit_rolling(f,c,t,xy,summary))
    summary['evaluation_sources']={n:sha(Path(__file__).with_name(n)) for n in ('m2_score.py','m2_rolling_audit.py')}
    write(out/'metrics.json',summary)
    return summary
if __name__=='__main__':
    pp=argparse.ArgumentParser();pp.add_argument('flight',type=Path);pp.add_argument('output',type=Path);args=pp.parse_args()
    print(json.dumps(score(args.flight,args.output),ensure_ascii=False))
