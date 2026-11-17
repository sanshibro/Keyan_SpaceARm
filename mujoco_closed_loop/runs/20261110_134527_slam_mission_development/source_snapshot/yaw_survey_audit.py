"""Fixed-base mounted-wrist survey development audit; sensor estimates register all rays."""
import sys,time,json
from pathlib import Path
from xml.etree.ElementTree import fromstring,tostring,SubElement as S
import mujoco,numpy as np,cv2
from model import model_xml,SimArmBridge
from scene import textured_room
from sensors import RGBDSensor,SurfaceLidar
from visual_odometry import RGBDOdometry
from fiducials import add_fiducials,FiducialLocalization
from robot_volume_prior import RobotVolumePrior
from perception_audit import camera_truth,transform
from estimation import pcc_camera_pose
from estimated_self_mask import EstimatedSelfMask
from depth_mapping import integrate_rgbd,self_hit_mask
from planner_adapter import V21Adapter,ROOT
from batched_mapper import BatchedVoxelMapper as IncrementalVoxelMapper
from evidence import freeze_run
from mission_scene import MissionScene
from qingyu_model import ARC_LENGTH,BASE_LENGTH
from tool_envelope import collision_samples

def workspace_endpoint_mask(frame,T_UC,radius):
    """Discard possible self endpoints; does not mark the workspace free."""
    yy,xx=np.indices(frame.depth_m.shape);z=frame.depth_m
    optical=np.stack([(xx-frame.K[0,2])*z/frame.K[0,0],(yy-frame.K[1,2])*z/frame.K[1,1],z],axis=-1)
    points=optical@T_UC[:3,:3].T+T_UC[:3,3]
    return np.isfinite(z)&(np.linalg.norm(points,axis=-1)<=radius)

def main(diagnostics=False,arm_sweep=False,mission=False,fixture_pose=False,sweep_deg=30.,workspace_self=False,visual_wrist_position=False,rock_pitch=False):
    if not 0<=sweep_deg<=120:raise ValueError('Survey bending exceeds single-segment range')
    out=ROOT/'runs'/(time.strftime('%Y%m%d_%H%M%S')+('_mission_survey' if mission else '_yaw_survey'));out.mkdir(parents=True)
    freeze_run(out)
    root=fromstring(model_xml(camera_profile='grasp_visible',nav_camera_profile='front_plus_survey'));textured_room(root);add_fiducials(root)
    mission_config=MissionScene() if mission else None
    if mission_config:
        mission_config.add(root);(out/'scene_config.json').write_text(json.dumps(mission_config.metadata(),indent=2))
    body_node=root.find("worldbody/body[@name='uav']");S(body_node,'joint',name='survey_yaw',axis='0 0 1',damping='2')
    S(root.find('actuator'),'position',name='survey_yaw',joint='survey_yaw',kp='100',kv='10')
    if rock_pitch:
        S(body_node,'joint',name='survey_pitch',axis='0 1 0',damping='2')
        S(root.find('actuator'),'position',name='survey_pitch',joint='survey_pitch',kp='100',kv='10')
    S(body_node,'site',name='survey_imu',pos='0 0 0',size='.005')
    sensor_node=S(root,'sensor');S(sensor_node,'gyro',name='survey_gyro',site='survey_imu')
    xml=tostring(root,encoding='unicode');(out/'model.xml').write_text(xml)
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    nav=RGBDSensor(m,'navigation',480,360);wrist=RGBDSensor(m,'wrist_rgbd',480,360);survey_cam=RGBDSensor(m,'survey_rgbd',480,360)
    nav_self=EstimatedSelfMask('navigation',nav_profile='front_plus_survey');wrist_self=EstimatedSelfMask('wrist_rgbd',nav_profile='front_plus_survey');survey_self=EstimatedSelfMask('survey_rgbd',nav_profile='front_plus_survey')
    no=FiducialLocalization(camera_truth(m,d,'navigation'));wo=FiducialLocalization(camera_truth(m,d,'wrist_rgbd'));self_volume=RobotVolumePrior(nav_profile='front_plus_survey');prior_statistics=[]
    uid=m.body('uav').id;U0=transform(d.xmat[uid].reshape(3,3),d.xpos[uid]);UC=np.linalg.solve(U0,camera_truth(m,d,'navigation'));US=np.linalg.solve(U0,camera_truth(m,d,'survey_rgbd'))
    bid=m.body('mount').id;tid=m.body('tool').id
    UB=np.linalg.solve(U0,transform(d.xmat[bid].reshape(3,3),d.xpos[bid]));TC=np.linalg.solve(transform(d.xmat[tid].reshape(3,3),d.xpos[tid]),camera_truth(m,d,'wrist_rgbd'))
    survey_localizer=FiducialLocalization(camera_truth(m,d,'survey_rgbd'));body_R=U0[:3,:3].copy();last_bending=np.zeros(2)
    workspace_radius=.03+BASE_LENGTH+ARC_LENGTH+max(.16,float(np.linalg.norm(collision_samples(),axis=1).max()))+.034+.020
    mapper=IncrementalVoxelMapper(.05,6.);lidar=SurfaceLidar();bridge=SimArmBridge();rows=[];images=0;scans=0;contacts=0;wall=time.perf_counter()
    diagnostic_maps={name:IncrementalVoxelMapper(.05,6.) for name in ('navigation','survey','wrist')} if diagnostics else {}
    # Physical arm movement aims the rigidly mounted wrist camera downward.
    for step in range(24000):
        t=d.time;theta=np.deg2rad(sweep_deg)*np.sin(np.pi*t/24)**2 if arm_sweep else 0.;phi=2*np.pi*t/12 if arm_sweep else 0.
        d.ctrl[:16]=bridge.move_joints(theta,phi);d.ctrl[m.actuator('survey_yaw').id]=2*np.pi*np.clip(t/20,0,1)
        if rock_pitch:d.ctrl[m.actuator('survey_pitch').id]=np.deg2rad(15)*np.sin(2*np.pi*t/24)
        mujoco.mj_step(m,d)
        contacts+=sum(any(m.geom_group[g] in (1,2) for g in (d.contact[k].geom1,d.contact[k].geom2)) for k in range(d.ncon))
        adr=m.sensor('survey_gyro').adr[0];body_R=body_R@cv2.Rodrigues(d.sensordata[adr:adr+3]*.001)[0]
        if step%250:continue
        if step%3000==0:print(f"survey {d.time:.1f}s",flush=True)
        nf=nav.read(d);wf=wrist.read(d);sf=survey_cam.read(d);a=no.update(nf);b=wo.update(wf);secondary=survey_localizer.update(sf)
        WU=a.T_WC@np.linalg.inv(UC) if a.valid else secondary.T_WC@np.linalg.inv(US) if secondary.valid else transform(body_R,U0[:3,3])
        if fixture_pose:
            # Explicit laboratory prior: the revolute fixture fixes position,
            # roll and pitch. Its yaw comes from integrated IMU gyro, not qpos.
            WU=transform(body_R,U0[:3,3])
        registered_wrist=None
        shape_usable=False
        if b.valid:
            registered_wrist,bending=pcc_camera_pose(WU,UB,b.T_WC,TC)
            last_bending=bending
            shape_usable=np.linalg.norm(registered_wrist[:3,3]-b.T_WC[:3,3])<=.020
            if visual_wrist_position:registered_wrist=b.T_WC.copy()
        if a.valid or fixture_pose:
            mask=workspace_endpoint_mask(nf,UC,workspace_radius) if workspace_self else nav_self.mask(nf,last_bending)
            registered_nav=WU@UC if fixture_pose else a.T_WC
            integrate_rgbd(mapper,nf,registered_nav,mask,stride=8);images+=1
            if diagnostics:integrate_rgbd(diagnostic_maps['navigation'],nf,registered_nav,mask,stride=8)
        # The underbody camera sees the moving arm. A stale shape projection
        # must not turn the unmodelled current arm into an external obstacle.
        # Pause this map feed on shape loss, using no truth fallback.
        if shape_usable or workspace_self:
            mask=workspace_endpoint_mask(sf,US,workspace_radius) if workspace_self else survey_self.mask(sf,bending)
            integrate_rgbd(mapper,sf,WU@US,mask,stride=5);images+=1
            if diagnostics:integrate_rgbd(diagnostic_maps['survey'],sf,WU@US,mask,stride=5)
        if shape_usable or (visual_wrist_position and b.valid):
            mask=workspace_endpoint_mask(wf,np.linalg.solve(WU,registered_wrist),workspace_radius) if workspace_self else wrist_self.mask(wf,bending)
            integrate_rgbd(mapper,wf,registered_wrist,mask,stride=6);images+=1
            if diagnostics:integrate_rgbd(diagnostic_maps['wrist'],wf,registered_wrist,mask,stride=6)
            last_bending=bending
            if shape_usable:prior_statistics.append(self_volume.integrate(mapper,WU,bending))
        if step%500==0:
            scan=lidar.read_body(m,d,uid);origin=WU[:3,3]+WU[:3,:3]@np.array([0,0,.2])
            mapper.integrate(scan['points']@WU[:3,:3].T+origin,origin,stride=2)
            mapper.integrate_free_rays(scan['free_endpoints']@WU[:3,:3].T+origin,origin,stride=2);scans+=1
        error=float(np.linalg.norm(registered_wrist[:3,3]-camera_truth(m,d,'wrist_rgbd')[:3,3])) if registered_wrist is not None else float('nan')
        true_WU=transform(d.xmat[uid].reshape(3,3),d.xpos[uid])
        # Truth residuals are diagnostic outputs only, never registration inputs.
        nav_error=np.linalg.norm(WU[:3,3]-true_WU[:3,3])
        nav_angle=np.arccos(np.clip((np.trace(WU[:3,:3].T@true_WU[:3,:3])-1)/2,-1,1))
        rows.append([d.time,float(theta),bool(a.valid),bool(b.valid),len(mapper.occupied),len(mapper.observed),error,nav_error,nav_angle,*last_bending])
    nav.close();wrist.close();survey_cam.close();nav_self.close();wrist_self.close();survey_self.close();snapshot=mapper.snapshot(stamp=d.time,metadata={'source':'mounted_wrist_physical_survey','localization':'surveyed_fiducial_RGBD','self_volume_prior':'estimated_eroded_collision_geometry_only_no_external_margin','initialization':'known_start_pose','robot_motion':'physical_yaw_fixture_gyro_measured_no_gimbal','secondary_camera':'rigid_underbody_RGBD','mapping_self_policy':'visible_prefix_stops_1.5_voxels_before_ambiguous_self_endpoint','self_mask_source':'estimated_shape_CAD_depth_projection_no_external_scene_labels','wrist_registration':'visual_orientation_plus_calibrated_PCC_position_prior','unknown_is_occupied':True})
    np.savez_compressed(out/'survey_log.npz',data=rows)
    if fixture_pose:snapshot.metadata['localization']='known_fixed_pivot_constraint_plus_integrated_IMU_gyro; wrist_shape_from_RGBD'
    if workspace_self:snapshot.metadata['self_mask_source']='calibrated_reachability_envelope_possible_self_endpoints; measured_prefix_only; workspace_not_filled_free'
    snapshot.metadata['sweep_deg']=sweep_deg
    snapshot.metadata['fixture_pitch_sweep_deg']=15 if rock_pitch else 0
    snapshot.metadata['mapping_self_policy']='measured_visible_ray_cells_excluding_ambiguous_endpoint_voxels'
    if visual_wrist_position:snapshot.metadata['wrist_registration']='surveyed_marker_RGBD_full_pose_no_PCC_position_projection'
    snapshot.save(out/'sensor_map.npz');adapter=V21Adapter(snapshot,survey_camera=True)
    start=np.r_[U0[:3,3],0,0,0];goal=mission_config.pregrasp if mission_config else [.1,0,1.5]
    trajectory,info=adapter.plan(start,goal,keep_yaw=True)
    for name,diagnostic_mapper in diagnostic_maps.items():
        diagnostic_mapper.snapshot(stamp=d.time,metadata={'diagnostic_only':True,'camera':name}).save(out/(name+'_contribution.npz'))
    result={'fixture_pose_constraint':fixture_pose,'workspace_endpoint_filter':workspace_self,'sweep_deg':sweep_deg,'arm_sweep':arm_sweep,'self_volume_updates':len(prior_statistics),'max_wrist_position_error_m':float(np.nanmax(np.asarray(rows)[:,6])),'registration':'surveyed_marker_RGBD_full_wrist_pose' if visual_wrist_position else 'visual_orientation_plus_PCC_position_prior_not_independent_wrist_translation','stage':'physical_yaw_fixture_multicamera_survey_only','full_task_accepted':False,'sim_seconds':d.time,'wall_seconds':time.perf_counter()-wall,'rgbd_integrations':images,'lidar_scans':scans,'contacts':contacts,'planner':info,'candidate_accepted':trajectory is not None}
    (out/'manifest.json').write_text(json.dumps(result,indent=2,default=str));np.savez_compressed(out/'survey_log.npz',data=rows)
    print(out);print(json.dumps(result,default=str))
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--diagnostics',action='store_true');parser.add_argument('--arm-sweep',action='store_true');parser.add_argument('--mission',action='store_true');parser.add_argument('--fixture-pose',action='store_true');parser.add_argument('--sweep-deg',type=float,default=30.);parser.add_argument('--workspace-self',action='store_true');parser.add_argument('--visual-wrist-position',action='store_true');parser.add_argument('--rock-pitch',action='store_true');args=parser.parse_args();main(args.diagnostics,args.arm_sweep,args.mission,args.fixture_pose,args.sweep_deg,args.workspace_self,args.visual_wrist_position,args.rock_pitch)
