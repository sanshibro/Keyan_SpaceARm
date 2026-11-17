"""Render full installed SDK STL and its earlier decomposed LOD independently."""
from pathlib import Path
import sys,os
os.environ.setdefault('MUJOCO_GL','egl')
import cv2,mujoco,numpy as np
ROOT=Path(__file__).resolve().parents[2]
mesh=ROOT/'mujoco_closed_loop/vendor/qingyu_continuum_arm/meshes'
out=ROOT/'revision/data/gripper_review';out.mkdir(exist_ok=True,parents=True)
for variant in ['sdk_full','previous_lod','full_split']:
    assets=[];geoms=[]
    if variant=='sdk_full':
        assets.append('<mesh name="g" file="%s" scale=".001 .001 .001"/>'%(mesh/'gripper.STEP.stl'))
        # Rz(pi)Rx(pi/2), then RViz URDF visual translation in gripper frame.
        geoms.append('<geom type="mesh" mesh="g" pos="-.0021 -.023 .0708" xyaxes="-1 0 0 0 0 1"/>')
    else:
        for n in ['housing','finger0','finger1','finger2']:
            filename='gripper_'+n+('_lod' if variant=='previous_lod' else '')+'.obj'
            assets.append('<mesh name="%s" file="%s"/>'%(n,mesh/filename))
            geoms.append('<geom type="mesh" mesh="%s"/>'%n)
    xml='<mujoco><visual><global offwidth="1000" offheight="850" orthographic="true" fovy=".19"/><headlight ambient=".5 .5 .5" diffuse=".65 .65 .65"/></visual><default><geom mass="0" contype="0" conaffinity="0" rgba=".25 .28 .30 1"/></default><asset><texture type="skybox" builtin="flat" rgb1="1 1 1" width="256" height="1536"/>'+''.join(assets)+'</asset><worldbody>'+''.join(geoms)+'</worldbody></mujoco>'
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    renderer=mujoco.Renderer(m,height=760,width=900);cam=mujoco.MjvCamera();cam.lookat[:]=[0,0,.073];cam.distance=.35;cam.azimuth=120;cam.elevation=-24
    renderer.update_scene(d,camera=cam);cv2.imwrite(str(out/(variant+'.png')),cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR));renderer.close()
print(out)
