"""Redundant terminal inverse kinematics using only robot kinematics and survey data."""
import numpy as np
import mujoco
from m2_plant import place


def goal_family(plant, task, yaw_grid=None):
    """Preserve the TCP datum or optical centre, and both forward axes and optical depth.

    Each member uses its own aircraft translation to compensate the roll of the
    camera/tool package induced by changing yaw. Obstacles are checked only by
    the caller using the current local map. No simulator state enters this FK.
    """
    m=plant.m;d=mujoco.MjData(m);origin=np.zeros(3)
    axis=np.asarray(task['axis']);camera_target=np.asarray(task['camera'])
    tool_target=np.asarray(task['tool']);neutral=float(task['yaw_deg'])
    yaw_grid=np.arange(-60.,60.01,3.) if yaw_grid is None else np.asarray(yaw_grid)
    def axis_at(yaw,params):
        place(m,d,origin,params[0],params[1],yaw)
        return d.body('tool').xmat.reshape(3,3)[:,2].copy()

    candidates=[]
    for offset in yaw_grid:
        yaw=neutral+offset
        params=np.array([task['theta_deg'],task['phi_deg']+offset],dtype=float)
        for attempt in range(8):
            residual=axis-axis_at(yaw,params)
            if np.linalg.norm(residual)<1e-9:break
            eps=.02;diff=[]
            for step in np.eye(2)*eps:
                diff.append((axis_at(yaw,params+step)-axis_at(yaw,params-step))/(2*eps))
            params+=np.linalg.lstsq(np.asarray(diff).T,residual,rcond=None)[0]
        final_axis=axis_at(yaw,params)
        if not 0<=params[0]<=120 or np.linalg.norm(final_axis-axis)>1e-7:continue
        tcp=d.site('tip').xpos.copy();c=d.camera('wrist_rgbd');eye=c.xpos.copy()
        optical=-c.xmat.reshape(3,3)[:,2].copy()
        if np.linalg.norm(optical-np.asarray(task['camera_axis']))>1e-7:continue
        align_tool=tool_target-tcp;align_eye=camera_target-eye
        seen=set()
        for u in (0.,.5,1.):
            center=(1-u)*align_tool+u*align_eye
            direction=camera_target-(center+tcp)
            direction-=optical*(direction@optical)
            direction/=max(np.linalg.norm(direction),1e-12)
            for shift in (0.,.008,.015):
                body=center+shift*direction;tool=body+tcp;camera=body+eye
                tool_error=np.linalg.norm(tool-tool_target);eye_error=np.linalg.norm(camera-camera_target)
                # Exact angular and optical-depth constraints. Bound TCP displacement
                # to half its task tolerance to leave a 17.5 mm tracking reserve.
                if tool_error>.0175 or eye_error>.024:continue
                if abs((camera-camera_target)@optical)>1e-7:continue
                key=tuple(np.round(body,8))
                if key in seen:continue
                seen.add(key)
                candidates.append(dict(body=body.tolist(),tool=tool.tolist(),camera=camera.tolist(),
                    yaw_deg=float(yaw),theta_deg=float(params[0]),phi_deg=float(params[1]%360.),
                    task_tool_offset_m=float(tool_error),task_camera_offset_m=float(eye_error),
                    eye_axis_depth_delta_m=float((camera-camera_target)@optical),
                    roll_compensation=float(u),datum_plane_shift_m=float(shift)))
    return candidates


def certify_tilt(data,auditor,native,pairs):
    """Independent certificate over the whole two-degree roll/pitch disc.

    Bound the Hausdorff motion of each rigidly transported CAD envelope.
    A 0.2-degree radial by 5-degree azimuth grid covers the disc; the
    unsampled-angle allowance is subtracted from every retained minimum.
    """
    model=auditor.m;qa=data.qpos[3:7].copy();base=data.qpos[:3].copy()
    lo,hi=auditor.bounds();a,b=pairs.T
    broad=np.linalg.norm(np.maximum(np.maximum(lo[a]-hi[b],lo[b]-hi[a]),0.),axis=1)
    centres=data.geom_xpos+(data.geom_xmat.reshape(-1,3,3)@model.geom_aabb[:,:3,None])[:,:,0]
    radii=np.linalg.norm(centres-base,axis=1)+np.linalg.norm(model.geom_aabb[:,3:],axis=1)
    max_tilt=np.deg2rad(2.);full_motion=2*np.sin(max_tilt/2)*radii[a]
    far=broad-full_motion>=.094
    skipped=float(np.min((broad-full_motion)[far])) if far.any() else np.inf
    near=np.ascontiguousarray(pairs[~far],np.int32)
    if not len(near):return dict(lower_m=skipped,cells=0)
    radius=float(radii[near[:,0]].max())
    covering=np.deg2rad(.1)+2*max_tilt*np.sin(np.deg2rad(1.25))
    residual=2*np.sin(covering/2)*radius
    best=np.inf;worst={};count=0
    try:
        for tilt in np.linspace(0.,2.,11):
            h=np.deg2rad(tilt)/2
            for bearing in (np.arange(0.,360.,5.) if tilt else [0.]):
                ang=np.deg2rad(bearing);c=np.r_[np.cos(h),np.sin(h)*np.cos(ang),np.sin(h)*np.sin(ang),0.]
                mujoco.mju_mulQuat(data.qpos[3:7],c,qa);mujoco.mj_kinematics(model,data)
                value=native.query(data,near);count+=1
                if value['lower']<best:
                    best=value['lower'];worst=dict(tilt_deg=tilt,bearing_deg=bearing,
                        pair=[model.geom(int(i)).name for i in value['which']])
    finally:
        data.qpos[3:7]=qa;mujoco.mj_kinematics(model,data)
    return dict(lower_m=min(skipped,best-residual),sample_min_m=best,unsampled_motion_m=residual,
                outer_rigid_radius_m=radius,cells=count,cover_rad=covering,
                near_pairs=len(near),far_pairs=int(far.sum()),witness=worst)


def geom_ids(reader):
    """Read native ID-colour raster; unresolved antialiased edges retain -1."""
    scene=reader.scene;old=scene.flags.copy()
    try:
        scene.flags[mujoco.mjtRndFlag.mjRND_SEGMENT]=1
        scene.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR]=1
        rgb=reader.render().astype(np.int32)
    finally:scene.flags[:]=old
    code=rgb[:,:,0]+256*rgb[:,:,1]+65536*rgb[:,:,2]
    table=np.full((scene.ngeom+1,2),-1,np.int32)
    for g in scene.geoms[:scene.ngeom]:
        if g.segid>=0:table[g.segid+1]=[g.objid,g.objtype]
    return table[np.where(code<=scene.ngeom,code,0)]


def audit_optical_family(plant,rows):
    from m2_scene import FACE,AXIS
    from m2_plant import option
    m=plant.m;d=mujoco.MjData(m);cid=m.camera('wrist_rgbd').id
    f=180/np.tan(np.deg2rad(m.cam_fovy[cid])/2)
    reader=mujoco.Renderer(m,360,480)
    bolts=[m.geom('COL_flange_bolt_%d'%i).id for i in range(8)]
    try:
        for row in rows:
            place(m,d,row['body'],row['theta_deg'],row['phi_deg'],row['yaw_deg'])
            reader.update_scene(d,camera='wrist_rgbd',scene_option=option())
            reader.enable_depth_rendering();depth=reader.render().copy();reader.disable_depth_rendering()
            labels=geom_ids(reader)
            points=[FACE+.00014*AXIS]+[d.geom_xpos[g]-.00283*AXIS for g in bolts]
            local=(np.asarray(points)-d.cam_xpos[cid])@d.cam_xmat[cid].reshape(3,3)
            look=[]
            for x,y,zz in local:
                z=-zz;i=int(round(239.5+f*x/z));j=int(round(179.5-f*y/z))
                if not (0<=i<480 and 0<=j<360):look.append(False);continue
                g,k=labels[j,i]
                visible=(k==mujoco.mjtObj.mjOBJ_GEOM and g>=0
                         and (m.geom(int(g)).name or '').startswith('M2_flange_')
                         and abs(depth[j,i]-z)<.002)
                look.append(bool(visible))
            row['optical_validation']['centre_depth_pixel_verified']=look[0]
            row['optical_validation']['socket_floor_depth_pixels_verified']=look[1:]
    finally:reader.close()


def audit(plant,folder):
    """Independently score the entire task-generated family on the static terminal snapshot."""
    import json
    import time
    from pathlib import Path
    from types import SimpleNamespace
    from m2_plant import task_pose
    from m2_geometry import RangeMap,Geometry
    from m2_scene import FACE
    from gate_distance_audit import DistanceAudit
    from m2_self_registry import partition
    task=task_pose(plant);start=time.perf_counter();rows=goal_family(plant,task)
    ik_ms=(time.perf_counter()-start)*1000
    sensor=RangeMap(plant);snap=sensor.reveal(0.,task['body'])
    g=Geometry(plant,SimpleNamespace(body_margin=.085,arm_margin=.085,tilt_max=np.deg2rad(2.)),snap)
    independent=DistanceAudit(plant.m,obstacle_prefix='COL_',tolerance_m=.0001,strict=False)
    hazard,fit=partition(plant.m,independent.self_pairs)
    groups=dict(body=independent.external['body'],arm=independent.external['arm_tool'],hazard=hazard,fit=fit)
    d=independent.d;m=plant.m
    camera_id=m.camera('wrist_rgbd').id;f=180/np.tan(np.deg2rad(m.cam_fovy[camera_id])/2)
    def angle(left,right):return float(np.rad2deg(np.arccos(np.clip(left@right,-1.,1.))))
    for row in rows:
        q=np.r_[row['body'],np.deg2rad([row['yaw_deg'],row['theta_deg'],row['phi_deg']])]
        online=g.state(q);row['planning']=dict(body=online['body'],arm=online['arm'],
            hazard=online['self_contact']['hazardous'],fit=online['self_contact']['assembly_min'],
            reason=online['reason'],body_pair=online['body_pair'],arm_pair=online['arm_pair'])
        place(m,d,row['body'],row['theta_deg'],row['phi_deg'],row['yaw_deg'])
        v=d.cam_xmat[camera_id].reshape(3,3).T@(FACE-d.cam_xpos[camera_id])
        dx,dy=f*v[0]/-v[2],-f*v[1]/-v[2]
        optical=-d.cam_xmat[camera_id].reshape(3,3)[:,2].copy()
        tool_axis=d.body('tool').xmat.reshape(3,3)[:,2].copy()
        row['optical_validation']=dict(flange_uv=[239.5+dx,179.5+dy],pixel_offset=float(np.hypot(dx,dy)),
            optical_depth_m=float(-v[2]),tcp_error_m=float(np.linalg.norm(d.site('tip').xpos-task['tool'])),
            eye_translation_m=float(np.linalg.norm(d.cam_xpos[camera_id]-task['camera'])),
            tool_axis_deg=angle(tool_axis,np.asarray(task['axis'])),optical_axis_deg=angle(optical,np.asarray(task['camera_axis'])))
        lo,hi=independent.bounds();dist={}
        for key,pairs in groups.items():
            v=independent._scan(key,pairs,lo,hi)
            dist[key]=dict(lower=float(v['lower']),upper=float(v['upper']),pair=[v['name_a'],v['name_b']])
        row['independent']=dist
        row['fit_pairs']=[dict(pair=[m.geom(int(i)).name,m.geom(int(j)).name],lower_m=independent.pair(int(i),int(j))['lower']) for i,j in fit]
        quat=d.qpos[3:7].copy()
        stress=[]
        for bearing in np.arange(0.,360.,15.):
            b=np.deg2rad(bearing);h=np.deg2rad(2.)/2
            tip_quat=np.r_[np.cos(h),np.sin(h)*np.cos(b),np.sin(h)*np.sin(b),0.]
            mujoco.mju_mulQuat(d.qpos[3:7],tip_quat,quat);mujoco.mj_kinematics(m,d)
            value=g.native_self.query(d,groups['arm'])
            stress.append(dict(tilt_deg=2.,bearing_deg=bearing,arm_lower_m=value['lower'],
                  pair=[m.geom(int(i)).name for i in value['which']]))
        row['tilt_boundary']=stress;row['tilted_min_m']=min(s['arm_lower_m'] for s in stress)
        if row['tilted_min_m']>=.085 and dist['hazard']['lower']>=.015:
            d.qpos[3:7]=quat;mujoco.mj_kinematics(m,d)
            row['tilt_certificate']={k:certify_tilt(d,independent,g.native_self,groups[k]) for k in ('arm','body')}
        row['jointly_feasible']=bool(not online['reason'] and min(dist['body']['lower'],dist['arm']['lower'])>=.085
            and dist['hazard']['lower']>=.015 and dist['fit']['lower']>=0.)
    audit_optical_family(plant,rows)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    out=dict(scene=plant.scene_digest,task=task,ik_ms=ik_ms,snapshot=snap.metadata(),candidates=rows)
    (folder/'terminal-ik.json').write_text(json.dumps(out,indent=2)+'\n')
    good=[v for v in rows if v['jointly_feasible']]
    rank=lambda r: (min(r['planning']['arm']-.085,r['planning']['body']-.085,r['planning']['hazard']-.015),-r['task_camera_offset_m']-r['task_tool_offset_m'])
    best=max(rows,key=rank)
    physical=[r for r in rows if min([s['lower_m'] for s in r.get('tilt_certificate',{}).values()] or [0])>=.085]
    print(json.dumps(dict(total=len(rows),feasible=len(good),tilt_certified=len(physical),best_proxy=best['planning'])))
    return out


if __name__=='__main__':
    import argparse
    from pathlib import Path
    from m2_plant import build
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);args=parser.parse_args()
    audit(build(),args.output)
