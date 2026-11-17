"""Regression: self filtering must not expose surfaces behind the robot."""
import numpy as np,mujoco
from sensors import SurfaceLidar
xml='<mujoco><worldbody><geom name="self" type="box" pos="1 0 1" size=".1 .2 .2" group="1"/><geom name="wall" type="box" pos="2 0 1" size=".1 .2 .2"/></worldbody></mujoco>'
m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);mujoco.mj_forward(m,d)
sensor=SurfaceLidar();sensor.directions=np.array([[1.,0,0]])
a=sensor.read(m,d,np.array([0.,0,1]),np.eye(3))
assert len(a['points'])==0 and len(a['free_endpoints'])==0 and a['self_occluded_rays']==1
m.geom_group[m.geom('self').id]=0
b=sensor.read(m,d,np.array([0.,0,1]),np.eye(3))
assert len(b['points'])==1 and abs(b['points'][0,0]-.9)<.02
sensor.directions=np.array([[0.,1,0]])
c=sensor.read(m,d,np.array([0.,0,1]),np.eye(3))
assert len(c['points'])==0 and np.allclose(c['free_endpoints'],[[0,6,0]])
print('PASS: self occludes rear wall; external first hit retained; no-return beam capped at6m')

# Exercise the generated robot too: primitive-only tests cannot detect an
# omitted group assignment in model construction.
from model import model_xml
for profile in ('front','front_plus_survey'):
    robot=mujoco.MjModel.from_xml_string(model_xml(camera_profile='grasp_visible',nav_camera_profile=profile))
    data=mujoco.MjData(robot);mujoco.mj_forward(robot,data)
    for i in range(robot.ngeom):
        if robot.geom_bodyid[i]!=0:
            assert robot.geom_group[i] in (1,2), (profile,i,'robot surface misclassified as environment')
    assert robot.geom_group[robot.geom('floor').id]==0
    sensor.directions=np.array([[0.,0.,-1.]])
    blocked=sensor.read(robot,data,np.array([0.,0.,1.6]),np.eye(3))
    assert blocked['self_occluded_rays']==1 and len(blocked['points'])==0
    assert len(blocked['free_endpoints'])==0
print('PASS: generated robot surfaces block lidar without becoming environment returns')
