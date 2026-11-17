"""Fixed-base mounted-wrist survey development audit; sensor estimates register all rays."""
import sys,time,json
from pathlib import Path
from xml.etree.ElementTree import fromstring,tostring
import mujoco,numpy as np
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

def main():
    out=ROOT/'runs'/(time.strftime('%Y%m%d_%H%M%S')+'_arm_survey');out.mkdir(parents=True)
    root=fromstring(model_xml(camera_profile='grasp_visible',nav_camera_profile='front_plus_survey'));textured_room(root);add_fiducials(root);xml=tostring(root,encoding='unicode');(out/'model.xml').write_text(xml)
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    nav=RGBDSensor(m,'navigation',480,360);wrist=RGBDSensor(m,'wrist_rgbd',480,360);survey_cam=RGBDSensor(m,'survey_rgbd',480,360)
    nav_self=EstimatedSelfMask('navigation',nav_profile='front_plus_survey');wrist_self=EstimatedSelfMask('wrist_rgbd',nav_profile='front_plus_survey');survey_self=EstimatedSelfMask('survey_rgbd',nav_profile='front_plus_survey')
    no=FiducialLocalization(camera_truth(m,d,'navigation'));wo=FiducialLocalization(camera_truth(m,d,'wrist_rgbd'));self_volume=RobotVolumePrior(nav_profile='front_plus_survey');prior_statistics=[]
    uid=m.body('uav').id;U0=transform(d.xmat[uid].reshape(3,3),d.xpos[uid]);UC=np.linalg.solve(U0,camera_truth(m,d,'navigation'));US=np.linalg.solve(U0,camera_truth(m,d,'survey_rgbd'))
    bid=m.body('mount').id;tid=m.body('tool').id
    UB=np.linalg.solve(U0,transform(d.xmat[bid].reshape(3,3),d.xpos[bid]));TC=np.linalg.solve(transform(d.xmat[tid].reshape(3,3),d.xpos[tid]),camera_truth(m,d,'wrist_rgbd'))
    mapper=IncrementalVoxelMapper(.05,6.);lidar=SurfaceLidar();bridge=SimArmBridge();rows=[];images=0;scans=0;contacts=0;wall=time.perf_counter()
    # Physical arm movement aims the rigidly mounted wrist camera downward.
    for step in range(24000):
        t=d.time;theta=np.deg2rad(90)*.5*(1-np.cos(np.pi*np.clip(t/6,0,1)))
        if t>18:theta=np.deg2rad(90)*.5*(1+np.cos(np.pi*np.clip((t-18)/6,0,1)))
        phi=np.pi+2*np.pi*np.clip((t-6)/12,0,1)
        d.ctrl[:16]=bridge.move_joints(theta,phi);mujoco.mj_step(m,d);contacts+=d.ncon
        if step%250:continue
        if step%3000==0:print(f"survey {d.time:.1f}s",flush=True)
        nf=nav.read(d);wf=wrist.read(d);a=no.update(nf);b=wo.update(wf)
        if a.valid:integrate_rgbd(mapper,nf,a.T_WC,nav_self.mask(nf,np.zeros(2)),stride=8);images+=1
        registered_wrist=None
        if a.valid and b.valid:
            registered_wrist,bending=pcc_camera_pose(a.T_WC@np.linalg.inv(UC),UB,b.T_WC,TC)
            integrate_rgbd(mapper,wf,registered_wrist,wrist_self.mask(wf,bending),stride=6);images+=1
            prior_statistics.append(self_volume.integrate(mapper,a.T_WC@np.linalg.inv(UC),bending))
            sf=survey_cam.read(d);integrate_rgbd(mapper,sf,a.T_WC@np.linalg.inv(UC)@US,survey_self.mask(sf,bending),stride=5);images+=1
        if step%500==0 and a.valid:
            WU=a.T_WC@np.linalg.inv(UC);scan=lidar.read(m,d,d.xpos[uid]+[0,0,.1],d.xmat[uid].reshape(3,3));origin=WU[:3,3]+WU[:3,:3]@np.array([0,0,.1])
            mapper.integrate(scan['points']@WU[:3,:3].T+origin,origin,stride=2)
            mapper.integrate_free_rays(scan['free_endpoints']@WU[:3,:3].T+origin,origin,stride=2);scans+=1
        error=float(np.linalg.norm(registered_wrist[:3,3]-camera_truth(m,d,'wrist_rgbd')[:3,3])) if registered_wrist is not None else float('nan')
        rows.append([d.time,float(theta),bool(a.valid),bool(b.valid),len(mapper.occupied),len(mapper.observed),error])
    nav.close();wrist.close();survey_cam.close();nav_self.close();wrist_self.close();survey_self.close();snapshot=mapper.snapshot(stamp=d.time,metadata={'source':'mounted_wrist_physical_survey','localization':'surveyed_fiducial_RGBD','self_volume_prior':'estimated_eroded_collision_geometry_only_no_external_margin','initialization':'known_start_pose','robot_motion':'fixed_base_theta_phi_sweep_no_gimbal','self_mask_source':'estimated_shape_CAD_depth_projection_no_external_scene_labels','wrist_registration':'visual_orientation_plus_calibrated_PCC_position_prior','unknown_is_occupied':True})
    np.savez_compressed(out/'survey_log.npz',data=rows)
    snapshot.save(out/'sensor_map.npz');adapter=V21Adapter(snapshot);trajectory,info=adapter.plan(np.array([0,0,1.5,0,0,0]),[.1,0,1.5])
    result={'self_volume_updates':len(prior_statistics),'max_wrist_position_error_m':float(np.nanmax(np.asarray(rows)[:,6])),'registration':'visual_orientation_plus_PCC_position_prior_not_independent_wrist_translation','stage':'fixed_base_mounted_camera_survey_only','full_task_accepted':False,'sim_seconds':d.time,'wall_seconds':time.perf_counter()-wall,'rgbd_integrations':images,'lidar_scans':scans,'contacts':contacts,'planner':info,'candidate_accepted':trajectory is not None}
    (out/'manifest.json').write_text(json.dumps(result,indent=2,default=str));np.savez_compressed(out/'survey_log.npz',data=rows)
    print(out);print(json.dumps(result,default=str))
if __name__=='__main__':main()
