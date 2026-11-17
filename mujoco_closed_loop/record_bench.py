"""Offline replay of recorded dynamics for synchronized inspection only."""
import argparse
from pathlib import Path
import cv2,mujoco,numpy as np
p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args()
m=mujoco.MjModel.from_xml_path(str(args.run/'model.xml'));d=mujoco.MjData(m)
a=np.load(args.run/'raw_dynamics.npz')['data'];r=mujoco.Renderer(m,height=360,width=480)
v=cv2.VideoWriter(str(args.run/'contact_replay.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(960,360))
for row in a[::10]:
 d.time=row[0];d.qpos[:]=row[1:1+m.nq];d.qvel[:]=row[1+m.nq:1+m.nq+m.nv];mujoco.mj_forward(m,d);views=[]
 for az in [90,0]:
  c=mujoco.MjvCamera();c.lookat[:]=d.xpos[m.body('tool').id];c.lookat[2]-=.07;c.distance=.42;c.azimuth=az;c.elevation=-15;r.update_scene(d,camera=c);views.append(r.render().copy())
 frame=cv2.cvtColor(np.concatenate(views,axis=1),cv2.COLOR_RGB2BGR)
 cv2.putText(frame,'RECORDED PHYSICS REPLAY | FIXTURE BENCH | t=%.2fs'%d.time,(10,25),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
 v.write(frame)
v.release();r.close();print(args.run/'contact_replay.mp4')
