"""Synchronized offline replay of saved physics, explicitly labelled as replay."""
import argparse,csv
from pathlib import Path
import mujoco,numpy as np,cv2
p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
m=mujoco.MjModel.from_xml_path(str(a.run/'model.xml'));d=mujoco.MjData(m);states=np.load(a.run/'raw_dynamics_imu.npz')['data']
objects=list(csv.DictReader((a.run/'object_evaluation.csv').open()))
r=mujoco.Renderer(m,360,480);video=cv2.VideoWriter(str(a.run/'mission_replay.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(960,360))
for index,row in enumerate(states[::10]):
 d.time=row[0];d.qpos[:]=row[1:1+m.nq];d.qvel[:]=row[1+m.nq:1+m.nq+m.nv];mujoco.mj_forward(m,d);views=[]
 for zoom in [False,True]:
  c=mujoco.MjvCamera();c.lookat[:]=d.xpos[m.body('tool').id] if zoom else [.25,0,1.25];c.distance=.45 if zoom else 1.8;c.azimuth=90 if zoom else 115;c.elevation=-15;r.update_scene(d,camera=c);views.append(r.render().copy())
 image=cv2.cvtColor(np.concatenate(views,axis=1),cv2.COLOR_RGB2BGR);phase=objects[min(index*10,len(objects)-1)]['phase']
 cv2.putText(image,'PHYSICS REPLAY t=%.2f | %s'%(d.time,phase),(10,24),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);video.write(image)
video.release();r.close();print(a.run/'mission_replay.mp4')
