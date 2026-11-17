"""Pixel-only localization audit at configured camera viewpoints."""
from xml.etree.ElementTree import fromstring,tostring
from pathlib import Path
import numpy as np,mujoco,json,time
from model import model_xml
from scene import textured_room
from fiducials import add_fiducials,FiducialLocalization
from sensors import RGBDSensor
from perception_audit import camera_truth
root=fromstring(model_xml(False,camera_profile='grasp_visible',nav_camera_profile='underbody'));textured_room(root);add_fiducials(root)
m=mujoco.MjModel.from_xml_string(tostring(root,encoding='unicode'));d=mujoco.MjData(m);rows=[]
for name in ['navigation','wrist_rgbd']:
 sensor=RGBDSensor(m,name,480,360);mujoco.mj_forward(m,d);est=FiducialLocalization(camera_truth(m,d,name))
 for x in [-.6,-.3,0,.5]:
  d.qpos[:3]=[x,0,1.55];mujoco.mj_forward(m,d);f=sensor.read(d);p=est.update(f);truth=camera_truth(m,d,name)
  rows.append({'camera':name,'body_x':x,'valid':p.valid,'inliers':p.inliers,'position_error_m':float(np.linalg.norm(p.T_WC[:3,3]-truth[:3,3])),'reprojection_px':p.reprojection_px})
 sensor.close()
out=Path(__file__).resolve().parent/'runs'/(time.strftime('%Y%m%d_%H%M%S')+'_fiducial_audit');out.mkdir(parents=True);(out/'report.json').write_text(json.dumps(rows,indent=2));print(out);print(json.dumps(rows))
