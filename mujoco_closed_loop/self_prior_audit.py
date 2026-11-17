"""Audit a calibrated hardware-interior prior while preserving external unknown space."""
import argparse,time,json
import numpy as np
from planner_adapter import V21Adapter,ROOT
from batched_mapper import BatchedVoxelMapper
from self_space import add_rigid_self_prior
from uav_continuum_whole_body_v2.mapping import MapSnapshot

def main():
 p=argparse.ArgumentParser();p.add_argument('map');a=p.parse_args();out=ROOT/'runs'/(time.strftime('%Y%m%d_%H%M%S')+'_self_prior');out.mkdir(parents=True)
 old=MapSnapshot.load(a.map);mapper=BatchedVoxelMapper(old.resolution,6);mapper.occupied=set(map(tuple,old.occupied_indices));mapper.observed=set(map(tuple,old.observed_indices))
 pose=np.eye(4);pose[:3,3]=[0,0,1.5];prior=add_rigid_self_prior(mapper,pose)
 snap=mapper.snapshot(stamp=old.stamp,metadata={'sensor_source':a.map,'self_prior':prior});snap.save(out/'sensor_map.npz')
 adapter=V21Adapter(snap);trajectory,info=adapter.plan(np.array([0,0,1.5,0,0,0]),[.1,0,1.5])
 result={'stage':'rigid_self_prior_only','full_task_accepted':False,'prior':prior,'candidate_accepted':trajectory is not None,'planner':info}
 (out/'manifest.json').write_text(json.dumps(result,indent=2,default=str));print(out);print(json.dumps(result,default=str))
if __name__=='__main__':main()
