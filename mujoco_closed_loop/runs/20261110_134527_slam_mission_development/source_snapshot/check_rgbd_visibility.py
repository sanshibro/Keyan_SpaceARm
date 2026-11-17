"""Rendered range evidence must be distinct from ordinary missing depth."""
import numpy as np
import mujoco
from sensors import RGBDSensor,RGBD
from depth_mapping import integrate_rgbd
from batched_mapper import BatchedVoxelMapper

def render(geometry):
    xml='<mujoco><statistic extent="1"/><visual><map znear=".005" zfar="30"/></visual><worldbody><camera name="cam" pos="0 0 1" xyaxes="0 -1 0 0 0 1"/>'+geometry+'</worldbody></mujoco>'
    model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    sensor=RGBDSensor(model,'cam',160,120)
    frame=sensor.read(data);sensor.close();return frame

pose=np.array([[0,0,1,0],[-1,0,0,0],[0,-1,0,1],[0,0,0,1.]])
empty=render('');assert empty.no_return_mask.all()
mapper=BatchedVoxelMapper(.05,6);integrate_rgbd(mapper,empty,pose,stride=3)
assert len(mapper.observed)>100 and not mapper.occupied
missing=RGBD(0,empty.rgb,np.full_like(empty.depth_m,np.nan),empty.K,'missing')
unknown=BatchedVoxelMapper(.05,6);integrate_rgbd(unknown,missing,pose,stride=3)
assert not unknown.observed and not unknown.occupied
blocked=render('<geom type="box" pos=".05 0 1" size=".01 .5 .5"/>')
assert not blocked.no_return_mask.any() and np.isnan(blocked.depth_m).all()
occluded=BatchedVoxelMapper(.05,6);integrate_rgbd(occluded,blocked,pose,stride=3)
assert not occluded.observed
surface=render('<geom type="box" pos="1 0 1" size=".1 2 2"/>')
prefix=BatchedVoxelMapper(.05,6)
integrate_rgbd(prefix,surface,pose,np.ones_like(surface.depth_m,dtype=bool),stride=1)
assert not prefix.occupied
assert (10,0,20) in prefix.observed
assert (18,0,20) not in prefix.observed and (22,0,20) not in prefix.observed
print('PASS: explicit beyond-range evidence clears visible rays; missing and too-close depth remain unknown')
print('PASS: ambiguous endpoint carves only its measured visible prefix, retaining surface and rear unknown')
