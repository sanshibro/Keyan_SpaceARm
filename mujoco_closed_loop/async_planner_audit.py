"""Infrastructure audit: physics continues while historical-map planner runs."""
import argparse,json,time
from pathlib import Path
from xml.etree.ElementTree import fromstring,tostring
import numpy as np,mujoco
from model import model_xml
from async_planner import AsyncV21Planner
from planner_adapter import ROOT
from uav_continuum_whole_body_v2.mapping import MapSnapshot

def main():
 p=argparse.ArgumentParser();p.add_argument('map',type=Path);a=p.parse_args()
 out=ROOT/'runs'/(time.strftime('%Y%m%d_%H%M%S')+'_async_planner_audit');out.mkdir(parents=True)
 root=fromstring(model_xml(profile='v2'));u=root.find("worldbody/body[@name='uav']");u.set('pos','0 0 1.3');u.set('quat',f'{np.cos(-np.pi/4)} 0 0 {np.sin(-np.pi/4)}')
 m=mujoco.MjModel.from_xml_string(tostring(root,encoding='unicode'));d=mujoco.MjData(m)
 state=np.array([0,0,1.3,-np.pi/2,0,0]);snapshot=MapSnapshot.load(a.map);planner=AsyncV21Planner();planner.request(a.map,'v2',state,[0,-1.2,1.3],snapshot.map_hash)
 started=time.perf_counter();rows=[];trajectory=None;info={}
 try:
  for step in range(16000):
   mujoco.mj_step(m,d)
   if step%10==0:
    trajectory,info=planner.poll(state,np.zeros(6),snapshot.map_hash);rows.append([d.time,time.perf_counter()-started,info['status']])
    if info['status']!='planning':break
   if step%5==0:
    remaining=d.time-(time.perf_counter()-started)
    if remaining>0:time.sleep(remaining)
 finally:planner.close()
 result={'stage':'async_planner_infrastructure_only','full_task_accepted':False,'map_mode':'historical_frozen_regression','physical_mode':'fixed_base_no_trajectory_execution','physics_steps_while_planning':step+1,'sim_seconds':d.time,'wall_seconds':time.perf_counter()-started,'candidate_accepted':trajectory is not None,'planner':info}
 (out/'manifest.json').write_text(json.dumps(result,indent=2,default=str));(out/'schedule.json').write_text(json.dumps(rows))
 if trajectory is not None:np.savez_compressed(out/'candidate.npz',times=trajectory.times,states=trajectory.states,velocities=trajectory.velocities)
 print(out);print(json.dumps(result,default=str))
if __name__=='__main__':main()
