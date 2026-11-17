"""Vectorized ray sampling with the original V2 voxel/ray semantics."""
import numpy as np
from planner_adapter import ROOT  # adds the existing V2 library path
from uav_continuum_whole_body_v2.mapping import IncrementalVoxelMapper,voxel_indices,_voxel_keys

class BatchedVoxelMapper(IncrementalVoxelMapper):
    def _observe_indices(self,indices,occupied):
        self.observed.update(map(tuple,indices))
    def _rays(self,points,sensor_origin,stride,occupied):
        p=np.asarray(points,dtype=float).reshape(-1,3)[::max(1,int(stride))];o=np.asarray(sensor_origin,dtype=float).reshape(3)
        if not np.isfinite(p).all() or not np.isfinite(o).all():raise ValueError('Nonfinite map input')
        dist=np.linalg.norm(p-o,axis=1);p=p[(dist>1e-9)&(dist<=self.maximum_range+(0 if occupied else 1e-6))]
        dist=np.linalg.norm(p-o,axis=1);counts=np.maximum(2,np.ceil(dist/self.ray_step).astype(int)+1)
        if occupied:self.occupied.update(map(tuple,voxel_indices(p,self.resolution)))
        key_chunks=[];buffered=0
        def flush():
            if not key_chunks:return
            keys=np.unique(np.concatenate(key_chunks))
            excluded=getattr(self,'_prefix_excluded_keys',None)
            if excluded is not None:keys=np.setdiff1d(keys,excluded,assume_unique=True)
            mask=(1<<21)-1;offset=1<<20
            idx=np.column_stack([(keys>>42)-offset,((keys>>21)&mask)-offset,(keys&mask)-offset])
            self._observe_indices(idx,occupied);key_chunks.clear()
        for count in np.unique(counts):
            endpoints=p[counts==count]
            for start in range(0,len(endpoints),max(1,250000//count)):
                batch=endpoints[start:start+max(1,250000//count)]
                ray=o[None,None,:]+np.linspace(0,1,count)[None,:,None]*(batch[:,None,:]-o[None,None,:])
                idx=voxel_indices(ray.reshape(-1,3),self.resolution)
                keys=_voxel_keys(idx);key_chunks.append(keys);buffered+=len(keys)
                if buffered>=2000000:flush();buffered=0
        flush()
        return len(p)
    def integrate(self,points,sensor_origin,stride=1):return self._rays(points,sensor_origin,stride,True)
    def integrate_free_rays(self,endpoints,sensor_origin,stride=1):return self._rays(endpoints,sensor_origin,stride,False)
    def integrate_visible_prefix(self,endpoints,sensor_origin,stride=1):
        """Observe measured ray cells while excluding every endpoint voxel."""
        points=np.asarray(endpoints,dtype=float).reshape(-1,3)[::max(1,int(stride))]
        self._prefix_excluded_keys=np.unique(_voxel_keys(voxel_indices(points,self.resolution)))
        try:return self._rays(points,sensor_origin,1,False)
        finally:self._prefix_excluded_keys=None
