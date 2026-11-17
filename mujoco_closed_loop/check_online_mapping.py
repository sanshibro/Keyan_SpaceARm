"""A vacated obstacle needs repeated visible-free evidence; occlusion stays unknown."""
import numpy as np
from online_mapping import OnlineRayMapper
mapper=OnlineRayMapper();origin=np.array([0.,0.,1.])
mapper.begin_frame(0);mapper.integrate([[1.,0.,1.]],origin);mapper.finish_frame()
old=(20,0,20);wall=(40,0,20)
assert old in mapper.occupied
for i in range(1,4):
    mapper.begin_frame(i*.2);mapper.integrate([[2.,0.,1.]],origin);mapper.finish_frame()
    assert wall in mapper.occupied
    assert (old in mapper.occupied)==(i<3)
assert (50,0,20) not in mapper.observed
assert mapper.clearing_events[-1]['removed_voxels']==[old]
mapper.begin_frame(.8);mapper.integrate([[1.,0.,1.]],origin);mapper.finish_frame()
assert old in mapper.occupied
print('PASS: three independent measured free prefixes clear a vacated voxel; first-hit surfaces and hidden rear remain protected')
