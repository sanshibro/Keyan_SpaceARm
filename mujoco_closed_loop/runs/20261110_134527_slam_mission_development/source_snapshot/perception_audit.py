"""Sensor module development audit; fixture motion, known initial pose, no flight claim."""
import csv,json,time,sys
from pathlib import Path
from xml.etree.ElementTree import fromstring,SubElement as S,tostring
import numpy as np,mujoco
from model import model_xml,SimArmBridge
from sensors import RGBDSensor,SurfaceLidar,OPTICAL_FROM_GL
from visual_odometry import RGBDOdometry,detect_red_target
from estimation import relative_camera,orientation_shape
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'xtdrone_catkin_ws/src/uav_continuum_whole_body_v2/src'))
from uav_continuum_whole_body_v2.mapping import IncrementalVoxelMapper

def transform(R,p):
 T=np.eye(4);T[:3,:3]=R;T[:3,3]=p;return T

def camera_truth(m,d,name):
 i=m.camera(name).id;return transform(d.cam_xmat[i].reshape(3,3)@OPTICAL_FROM_GL,d.cam_xpos[i])

def main():
 out=ROOT/'runs'/(time.strftime('%Y%m%d_%H%M%S')+'_perception_audit');out.mkdir(parents=True)
 root=fromstring(model_xml());u=root.find("worldbody/body[@name='uav']")
 for g in u.iter('geom'):
  if g.get('group')!='2':g.set('group','1')
 for name,axis,kind in [('motion_x','1 0 0','slide'),('motion_yaw','0 0 1','hinge')]:
  S(u,'joint',name=name,axis=axis,type=kind,damping='5');S(root.find('actuator'),'position',name=name,joint=name,kp='500',kv='40')
 w=root.find('worldbody');rng=np.random.default_rng(8)
 for side in [-1,1]:
  S(w,'geom',name=f'wall{side}',type='box',pos=f'{side*1.6} 0 1.3',size='.03 2 1.3',rgba='.6 .6 .6 1')
  for i in range(450):
   y,z=rng.uniform(-1.9,1.9),rng.uniform(.1,2.5);c=rng.uniform(.05,.9,3);c[0]=min(c[0],c[1])
   S(w,'geom',type='box',pos=f'{side*1.565} {y} {z}',size=f'.002 {rng.uniform(.015,.055)} {rng.uniform(.015,.055)}',rgba=' '.join(map(str,[*c,1])))
 S(w,'geom',name='controlled_red_target',type='box',pos='-1.3 .1 .95',size='.04 .04 .04',rgba='1 .02 .02 1')
 xml=tostring(root,encoding='unicode');(out/'model.xml').write_text(xml)
 m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);mujoco.mj_forward(m,d)
 nav=RGBDSensor(m,'navigation',width=480,height=360);wrist=RGBDSensor(m,'wrist_rgbd',width=480,height=360)
 uid=m.body('uav').id;bid=m.body('mount').id;tid=m.body('tool').id
 U0=transform(d.xmat[uid].reshape(3,3),d.xpos[uid]);B0=transform(d.xmat[bid].reshape(3,3),d.xpos[bid]);T0=transform(d.xmat[tid].reshape(3,3),d.xpos[tid])
 UC=np.linalg.solve(U0,camera_truth(m,d,'navigation'));UB=np.linalg.solve(U0,B0);TC=np.linalg.solve(T0,camera_truth(m,d,'wrist_rgbd'))
 no=RGBDOdometry(camera_truth(m,d,'navigation'));wo=RGBDOdometry(camera_truth(m,d,'wrist_rgbd'));bridge=SimArmBridge();rows=[];mapper=IncrementalVoxelMapper(resolution=.08,maximum_range=6.);scans=0;clock=time.perf_counter();lidar=SurfaceLidar();raw_saved=False
 for step in range(6000):
  t=d.time;s=.5*(1-np.cos(np.pi*np.clip((t-.5)/5,0,1)))
  d.ctrl[:16]=bridge.move_joints(np.deg2rad(35)*s,np.deg2rad(45))
  d.ctrl[m.actuator('motion_x').id]=.08*s;d.ctrl[m.actuator('motion_yaw').id]=.15*s
  mujoco.mj_step(m,d)
  if step%100:continue
  nf=nav.read(d);wf=wrist.read(d);drop=2.4<t<2.8
  if drop:wf.rgb[:]=0
  ne=no.update(nf);we=wo.update(wf);shape_error=float('nan');nav_error=float('nan');det=detect_red_target(wf)
  if ne.valid:
   WU=ne.T_WC@np.linalg.inv(UC);nav_error=float(np.linalg.norm(WU[:3,3]-d.xpos[uid]))
   if we.valid:
    estimate=orientation_shape(relative_camera(WU,UB,we.T_WC,ne.stamp,we.stamp),TC)
    true_BC=np.linalg.solve(transform(d.xmat[bid].reshape(3,3),d.xpos[bid]),camera_truth(m,d,'wrist_rgbd'))
    actual=orientation_shape(true_BC,TC);shape_error=float(np.rad2deg(np.linalg.norm(estimate-actual)))
  target_error=float('nan')
  if det is not None and we.valid:
   surface=we.T_WC[:3,:3]@det['surface_center_C']+we.T_WC[:3,3]
   target_error=float(np.linalg.norm(surface-np.array([-1.26,.1,.95])))
  if step%500==0 and ne.valid:
   scan=lidar.read(m,d,d.xpos[uid]+d.xmat[uid].reshape(3,3)@np.array([0,0,.1]),d.xmat[uid].reshape(3,3))
   origin=WU[:3,3]+WU[:3,:3]@np.array([0,0,.1])
   registered=scan['points']@WU[:3,:3].T+origin
   mapper.integrate(registered,origin,stride=4);scans+=1
  rows.append([d.time,ne.valid,we.valid,drop,ne.inliers,we.inliers,nav_error,shape_error,det['pixels'] if det else 0,we.reprojection_px,target_error])
  if not raw_saved and ne.valid and we.valid:
   scan=lidar.read(m,d,d.xpos[uid]+d.xmat[uid].reshape(3,3)@np.array([0,0,.1]),d.xmat[uid].reshape(3,3))
   np.savez_compressed(out/'sensor_sample.npz',nav_rgb=nf.rgb,nav_depth_m=nf.depth_m,wrist_rgb=wf.rgb,wrist_depth_m=wf.depth_m,K=wf.K,stamp=d.time,lidar_points=scan['points'])
   (out/'calibration.json').write_text(json.dumps(dict(T_UC=UC.tolist(),T_UB=UB.tolist(),T_TC=TC.tolist(),depth_unit='metre',sensor_approximation='ideal_rendered_RGBD_not_RealSense_stereo',lidar='360x16,+/-30deg,2Hz in this audit,6m,3mm noise; nominal module supports10Hz'),indent=2));raw_saved=True
 nav.close();wrist.close()
 mapper.snapshot(stamp=d.time,metadata={'registration':'timestamped_RGBD_navigation_estimate','audit_scan_frequency_hz':2,'unknown_policy':'occupied'}).save(out/'sensor_map.npz')
 with (out/'estimates.csv').open('w') as f:
  writer=csv.writer(f);writer.writerow(['sim_time','nav_valid','wrist_valid','injected_dropout','nav_inliers','wrist_inliers','nav_position_error_m','bend_error_deg','target_pixels','wrist_reprojection_px','target_surface_world_error_m']);writer.writerows(rows)
 a=np.array(rows,dtype=float);visible=a[:,3]==0;valid=(a[:,1]>0)&(a[:,2]>0)&visible
 result=dict(stage='fixture_rgbd_odometry_audit',full_task_accepted=False,initialization='known_initial_camera_poses_only_from_simulator',feedback='subsequent_poses_from_RGBD_ORB_depth_PnP',frames=len(rows),valid_visible_fraction=float(valid.sum()/visible.sum()),max_nav_error_m=float(np.nanmax(a[:,6])),max_bend_error_deg=float(np.nanmax(a[:,7])),injected_dropout_rejected=bool(np.all(a[a[:,3]>0,2]==0)),target_detected_frames=int((a[:,8]>0).sum()),max_target_surface_error_m=float(np.nanmax(a[:,10])),lidar_scans=scans,occupied_voxels=len(mapper.occupied),observed_voxels=len(mapper.observed),sim_seconds=d.time,wall_seconds=time.perf_counter()-clock)
 result['acceptance_scope']='RGBD_odometry_and_compensated_shape_only_not_target_or_task'
 result['target_surface_30mm_passed']=bool(result['max_target_surface_error_m']<=.03)
 result['passed']=bool(result['valid_visible_fraction']>.9 and result['max_nav_error_m']<.1 and result['max_bend_error_deg']<5 and result['injected_dropout_rejected'])
 (out/'source.py').write_text(Path(__file__).read_text())
 (out/'manifest.json').write_text(json.dumps(result,indent=2));print(out);print(json.dumps(result))
if __name__=='__main__':main()
