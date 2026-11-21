"""Replay the saved states with external, self, task and synchronized real-image witnesses."""
import argparse,hashlib,json,math,time
from pathlib import Path
import cv2,mujoco,numpy as np
from m1_plant import build,task_pose,FACE,AXIS
from m1_geometry import GalleryGeometry
from gate_feedback_model import truth_bend
from gate_distance_audit import DistanceAudit
from gate_live_optics import clean
from uav_continuum_whole_body_v2.types import Limits
from round_aug_vis import bend
from m1_plan import writej

def dwell(t,valid):
    best=0.;first=None;start=None;limit=None
    for stamp,ok in zip(t,valid):
        if ok:
            if start is None:start=stamp
            best=max(best,stamp-start)
            if first is None and stamp-start>=1.-1e-7:first=float(stamp)
        else:start=None
    return best,first

def evaluate(folder,output):
    f=Path(folder);out=Path(output);out.mkdir(exist_ok=True,parents=True)
    p=build();task=task_pose(p);g=GalleryGeometry(p,Limits(body_margin=.06,arm_margin=.06,tilt_max=0.))
    run=json.loads((f/'run.json').read_text());prov=json.loads((f/'provenance.json').read_text());assert prov['mission']['scene']==p.scene_digest
    if run['error'] or not (f/'flight.ulg').is_file():raise ValueError('Incomplete registered flight')
    c=np.load(f/'control.npz');phys=np.load(f/'physics.npz')
    t=c['t'];grid=(phys['us']-run['release_us'])*1e-6;idx=np.searchsorted(np.round(grid,6),np.round(t,6))
    assert np.max(abs(grid[idx]-t))<1e-6
    Q=phys['qpos'][idx];Z=phys['rotor_N'][idx];tcp=phys['tcp'][idx]
    K=261.9016;cent=np.array([239.5,179.5]);fake=np.zeros(6)
    rows=[];audit=DistanceAudit(p.m,tolerance_m=.00015,strict=False,obstacle_prefix='COL_')
    vel=phys['qvel'][idx];ctrl=phys['ctrl'][idx];mount=p.m.body('mount').id;p.d.eq_active[0]=0
    reaction=[]
    for j,q in enumerate(Q):
        p.d.qpos[:]=q;p.d.qvel[:]=vel[j];p.d.ctrl[:]=ctrl[j];mujoco.mj_forward(p.m,p.d)
        mujoco.mj_rnePostConstraint(p.m,p.d)
        wrench=p.d.cfrc_int[mount];r=p.d.subtree_com[p.m.body_rootid[mount]]-p.d.xpos[mount]
        reaction.append(-(p.d.xmat[p.base].reshape(3,3).T@(wrench[:3]+np.cross(r,wrench[3:]))))
        cam=p.d.camera('wrist_rgbd');R=cam.xmat.reshape(3,3)
        point=R.T@(FACE-cam.xpos);z=-point[2]
        uv=cent+K*np.array([point[0],-point[1]])/z if abs(z)>.0001 else np.full(2,np.nan)
        tip=p.d.site('tip').xmat.reshape(3,3)[:,2]
        actual=bend(p.d.body('mount').xmat.reshape(3,3).T@p.d.body('tool').xmat.reshape(3,3))
        com=p.d.subtree_com[p.base]-q[:3];moment=np.cross(com,[0,0,-p.mass*9.81])
        pm=g.components(p.d,q[:3]);pm[:,6]=0
        lower,ids=g.query(pm,np.r_[q[:3],0.,0.,0.])
        rigid=p.d.body('uav').xmat.reshape(3,3);yaw=np.arctan2(rigid[1,0],rigid[0,0])
        rear=q[0]+np.min(pm[:,0]-pm[:,3])
        rows.append([*lower,float(np.rad2deg(np.arccos(np.clip(tip@AXIS,-1,1)))),z,*uv,*actual,*moment[:2],
                     float(np.rad2deg(np.arccos(np.clip(rigid[2,2],-1,1)))),yaw,rear,*com])
    A=np.asarray(rows)
    ext=A[:,:2];pose=np.linalg.norm(tcp-np.asarray(task['tool']),axis=1);during=t>=21
    if not during.any():raise ValueError('No scored interval')
    err=np.linalg.norm(Q[:,:3]-c['ref_world'],axis=1);fth=(Z[:,0]+Z[:,2])-(Z[:,1]+Z[:,3])
    # Exact minima of both possible external pair groups. Any remaining timestamp
    # whose conservative lower bound is greater than this upper witness cannot improve it.
    mins={};stations=[]
    for k,kind in enumerate(('body','arm_tool')):
        best=np.inf;winner=None;count=0
        for n in np.flatnonzero(during)[np.argsort(ext[during,k])]:
            if ext[n,k]>=best:break
            audit.d.qpos[:]=Q[n];mujoco.mj_kinematics(p.m,audit.d)
            lo,hi=audit.bounds();result=audit._scan(kind,audit.external[kind],lo,hi)
            count+=1
            if result['upper']<best:best=result['upper'];winner=(int(n),result)
        n,r=winner;mins[kind]=dict(m=float(r['distance']),t=float(t[n]),lower=float(r['lower']),upper=float(r['upper']),
                pair=[r['name_a'],r['name_b']],evaluations=count)
    # Fully sampled independent self audit, accelerate repeated pair bounds with the
    # same broad phase used by the signed-convex verifier.
    self_lower=[];pairs=audit.self_pairs;aa,bb=pairs.T
    for n in np.flatnonzero(during):
        audit.d.qpos[:]=Q[n];mujoco.mj_kinematics(p.m,audit.d);l,h=audit.bounds()
        delta=np.maximum(np.maximum(l[aa]-h[bb],l[bb]-h[aa]),0.)
        self_lower.append(np.linalg.norm(delta,axis=1).min())
    # Exact self/contact curve at 5 Hz. All 50 Hz samples retain the self AABB lower bound.
    ns=np.flatnonzero(during);sample=ns[::10];self_curve=[]
    for n in sample:
        audit.d.qpos[:]=Q[n];mujoco.mj_kinematics(p.m,audit.d);l,h=audit.bounds()
        r=audit._scan('self',pairs,l,h);self_curve.append(r['distance'])
    self_curve=np.array(self_curve);self_best=float(self_curve.min());self_remaining=0
    for j in np.argsort(self_lower):
        if self_lower[j]>=self_best:break
        n=ns[j]
        if j%10==0:continue
        audit.d.qpos[:]=Q[n];mujoco.mj_kinematics(p.m,audit.d);l,h=audit.bounds()
        r=audit._scan('self',pairs,l,h);self_best=min(self_best,float(r['upper']));self_remaining+=1
    inside=(A[:,3]>=.12)&(A[:,3]<=.18)&(A[:,4]>=0)&(A[:,4]<480)&(A[:,5]>=0)&(A[:,5]<360)
    eligible=(pose<=.035)&(A[:,2]<=3.)&inside&(c['phase']=='inspect')
    span,first=dwell(t,eligible)
    with (f/'optics/index.jsonl').open() as stream:optical=[json.loads(line) for line in stream]
    samples=[];view_ok=0;vis_count=0;visible=np.zeros(len(optical),bool)
    for o in optical:
        n=int(np.argmin(abs(t-o['t'])))
        if abs(t[n]-o['t'])>1e-6:raise ValueError('Wrist/flight time mismatch')
        if hashlib.sha256(np.asarray(Q[n],np.float64).tobytes()).hexdigest()!=o['sensor_qpos_sha256']:raise ValueError('Camera pose identity mismatch')
        if not eligible[n]:continue
        depth=cv2.imread(str(f/'optics'/o['depth']),cv2.IMREAD_UNCHANGED);u,v=np.rint(A[n,4:6]).astype(int)
        patch=depth[max(v-2,0):v+3,max(u-2,0):u+3]
        good=patch[patch>0];vis_count+=1
        if len(good) and abs(np.median(good)*.001-A[n,3])<.015:
            view_ok+=1;visible[o['frame']]=True;samples.append([n,o['frame'],int(np.median(good))])
    t_pass=[.0]*3
    for i,x in enumerate((1.05,1.793,3.07)):
        q=np.flatnonzero(during&(A[:,12]>x))
        t_pass[i]=float(t[q[0]]) if len(q) else None
    moving=np.linalg.norm(np.gradient(A[:,6:8],t,axis=0),axis=1)>np.deg2rad(3)
    transition=during&moving
    if not transition.any():transition=during
    z=np.load(f/'optics/observations.npz')
    optical_dwell,_=dwell(np.array([r['t'] for r in optical]),visible)
    success=bool(span>=1.-1e-7 and optical_dwell>=1.-1e-7 and all(v['lower']>=.06 for v in mins.values()) and run['contact_integration_steps']==0 and self_best>0)
    summary=dict(case=prov['mission']['case'],scene=p.scene_digest,classification='completed' if success else 'evaluated',
        task_success=success,task_dwell_s=span,first_pass_s=first,passages_s=t_pass,clearance=mins,
        self_clearance_5Hz_m=float(self_curve.min()),self_clearance_50Hz_m=self_best,self_additional_refinements=self_remaining,
        self_broad_phase_50Hz_m=float(np.min(self_lower)),optical_dwell_s=optical_dwell,
        collision_steps=run['contact_integration_steps'],end_phase=str(c['phase'][-1]),end_xyz=Q[-1,:3],end_tool=tcp[-1],
        final_pos_m=float(np.median(pose[t>=t[-1]-1])),final_axis_deg=float(np.median(A[t>=t[-1]-1,2])),
        final_optical_m=float(np.median(A[t>=t[-1]-1,3])),terminal_visible_frames=view_ok,terminal_checked_frames=vis_count,
        peak_tracking_m=float(err[transition].max()),peak_tilt_deg=float(A[transition,10].max()),
        max_gravity_moment_Nm=float(np.linalg.norm(A[transition,8:10],axis=1).max()),
        peak_mount_reaction_Nm=float(np.linalg.norm(np.array(reaction)[transition,:2],axis=1).max()),
        peak_front_rear_N=float(np.max(abs(fth[transition]))),rms_front_rear_N=float(np.sqrt(np.mean(fth[transition]**2))),
        orbit_frames=len(z['t']),tracked_frames=int(z['valid'].sum()),matching_exposures=len(optical),
        scene_geom_count=p.m.ngeom,command_clip_deg=float(np.degrees(np.linalg.norm(c['issued_b'],axis=1).max())),
        body_lateral_excursion_m=float(np.ptp(Q[during,1])),max_actual_bend_deg=float(np.rad2deg(np.linalg.norm(A[:,6:8],axis=1).max())))
    np.savez_compressed(out/'series.npz',t=t,body=Q[:,:3],qpos=Q,tcp=tcp,ref_xyz=c['ref_world'],ref_b=c['des_b'],
         est_b=c['est_b'],actual_b=A[:,6:8],external_lower=ext,self_lower_t=t[ns],self_lower=self_lower,
         self_exact_t=t[sample],self_exact=self_curve,moment=A[:,8:10],mount_reaction=np.array(reaction),rotors=Z,front_rear=fth,tracking=err,tilt=A[:,10],yaw=A[:,11],
         target_error=pose,target_angle=A[:,2],zcam=A[:,3],uv=A[:,4:6],rear_x=A[:,12],phase=c['phase'],hold=c['hold'],verified_views=np.array(samples))
    writej(out/'metrics.json',summary);print(json.dumps(clean(summary),ensure_ascii=False),flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('flight',type=Path);a.add_argument('output',type=Path);v=a.parse_args()
    evaluate(v.flight,v.output)
