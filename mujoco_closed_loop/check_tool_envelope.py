"""Collision regression: an obstacle at the side-mounted camera must reject a trajectory."""
import numpy as np
from planner_adapter import V21Adapter
from tool_envelope import collision_samples
from uav_continuum_whole_body_v2.types import ArmModel,MassModel,Limits,PlannerConfig
from uav_continuum_whole_body_v2.environment import BoxObstacle,BoxWorld,HybridDistanceWorld
from uav_continuum_whole_body_v2.validator import SweptVolumeValidator
from uav_continuum_whole_body_v2.trajectory import SplineTrajectory
from uav_continuum_whole_body_v2.geometry import world_arm_points,world_arm_points_batch
state=np.array([0,0,1.5,0,0,0]);spline=SplineTrajectory(np.tile(state,(8,1)),np.ones(5))
world=HybridDistanceWorld(BoxWorld([BoxObstacle('camera_only_obstacle',(-.070267,.129267,1.108777),(.016,.016,.016))]))
base=ArmModel(arc_length=.28,base_offset_z=.0704);extended=ArmModel(arc_length=.28,base_offset_z=.0704,extra_tool_points=collision_samples())
a=SweptVolumeValidator(world,base,MassModel(),Limits(),PlannerConfig()).validate(spline)
b=SweptVolumeValidator(world,extended,MassModel(),Limits(),PlannerConfig()).validate(spline)
assert a.feasible and not b.feasible,(a.status,b.status)
print('PASS: original backbone alone misses the camera obstacle; explicit tool envelope rejects it:',b.failure_reason)
rng=np.random.default_rng(8);states=rng.normal(0,.3,(25,6));states[:,4]=abs(states[:,4])
batch=world_arm_points_batch(states,extended)
for i,s in enumerate(states):
    assert np.allclose(world_arm_points(s,base)[-1],world_arm_points(s,extended)[-1])
    assert np.allclose(batch[i],world_arm_points(s,extended))
print('PASS: tool envelope preserves terminal-tip indexing and scalar/batch equivalence')
