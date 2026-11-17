"""Geometry-only detector audit. State resets here are explicitly not flight control."""
from pathlib import Path
from xml.etree.ElementTree import fromstring,SubElement as S,tostring
import json,time,numpy as np,mujoco
from model import model_xml
from scene import textured_room
from sensors import RGBDSensor
from perception_audit import camera_truth
from visual_odometry import detect_supported_cylinder
ROOT=Path(__file__).resolve().parent
root=fromstring(model_xml(False,camera_profile='grasp_visible'));textured_room(root);w=root.find('worldbody')
S(w,'geom',name='support',type='box',pos='0 0 .953',size='.06 .06 .025',rgba='.4 .3 .2 1')
S(w,'geom',name='target',type='cylinder',pos='0 0 .998',size='.012 .02',rgba='1 .02 .02 1')
m=mujoco.MjModel.from_xml_string(tostring(root,encoding='unicode'));d=mujoco.MjData(m);sensor=RGBDSensor(m,'wrist_rgbd',480,360);rows=[]
for x,z in [(-.6,1.5),(-.3,1.55),(-.1,1.55),(0,1.55),(0,1.5064)]:
 d.qpos[:3]=[x,0,z];mujoco.mj_forward(m,d);frame=sensor.read(d);result=detect_supported_cylinder(frame,camera_truth(m,d,'wrist_rgbd'))
 error=float(np.linalg.norm(result['center_W']-[0,0,.998])) if result else None
 rows.append({'body_x':x,'body_z':z,'detected':result is not None,'error_m':error})
sensor.close();out=ROOT/'runs'/(time.strftime('%Y%m%d_%H%M%S')+'_target_geometry');out.mkdir(parents=True);(out/'report.json').write_text(json.dumps(rows,indent=2));print(out);print(json.dumps(rows))
