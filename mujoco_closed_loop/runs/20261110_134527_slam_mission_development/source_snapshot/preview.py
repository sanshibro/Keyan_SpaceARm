"""Offscreen model inspection; observation cameras never feed control."""
from pathlib import Path
import mujoco,numpy as np,cv2
from model import model_xml,SimArmBridge
m=mujoco.MjModel.from_xml_string(model_xml());d=mujoco.MjData(m)
d.ctrl[:16]=SimArmBridge().move_joints(np.deg2rad(60),np.deg2rad(45))
for _ in range(3000):mujoco.mj_step(m,d)
r=mujoco.Renderer(m,height=480,width=640);views=[]
for az,el in [(90,-10),(0,-15),(90,-85)]:
 c=mujoco.MjvCamera();c.lookat[:]=[0,0,1.25];c.distance=1.2;c.azimuth=az;c.elevation=el;r.update_scene(d,camera=c);views.append(r.render().copy())
out=Path(__file__).resolve().parent/'model_preview.jpg';cv2.imwrite(str(out),cv2.cvtColor(np.concatenate(views,axis=1),cv2.COLOR_RGB2BGR));r.close();print(out)
