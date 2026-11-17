"""SUPER GENERAL_360 simulated scanner profile; explicitly not Mid-360 fidelity."""
import json,time
from xml.etree.ElementTree import fromstring,tostring
import numpy as np,mujoco
from model import model_xml
from scene import textured_room
from sensors import SurfaceLidar
from batched_mapper import BatchedVoxelMapper
from planner_adapter import ROOT,V21Adapter
from self_space import add_rigid_self_prior

def main():
 out=ROOT/'runs'/(time.strftime('%Y%m%d_%H%M%S')+'_super_sensor_profile');out.mkdir(parents=True)
 root=fromstring(model_xml());textured_room(root);xml=tostring(root,encoding='unicode');m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);mujoco.mj_forward(m,d)
 scanner=SurfaceLidar.super_general360();mapper=BatchedVoxelMapper(.05,15);clock=time.perf_counter()
 # Initial configured mounting transform; registration is known initial pose only.
 origin=np.array([.24,0,1.6]);scan=scanner.read(m,d,origin,np.eye(3))
 mapper.integrate(scan['points']+origin,origin);mapper.integrate_free_rays(scan['free_endpoints']+origin,origin)
 pose=np.eye(4);pose[:3,3]=[0,0,1.5];prior=add_rigid_self_prior(mapper,pose)
 snapshot=mapper.snapshot(metadata={'profile':'SUPER_GENERAL_360_178deg_128lines_azimuth_decimated_1deg','mount_U':[.24,0,.1],'initialization':'known_initial_pose_only','self_prior':prior,'not_mid360':True});snapshot.save(out/'sensor_map.npz')
 adapter=V21Adapter(snapshot);trajectory,info=adapter.plan(np.array([0,0,1.5,0,0,0]),[.1,0,1.5])
 result={'stage':'static_sensor_profile_development_only','full_task_accepted':False,'official_config':'vendor/SUPER/mars_uav_sim/perfect_drone_sim/config/dense.yaml','not_mid360':True,'beam_count':len(scanner.directions),'hit_count':len(scan['points']),'self_occluded':scan['self_occluded_rays'],'wall_seconds':time.perf_counter()-clock,'planner':info,'candidate_accepted':trajectory is not None}
 (out/'manifest.json').write_text(json.dumps(result,indent=2,default=str));print(out);print(json.dumps(result,default=str))
if __name__=='__main__':main()
