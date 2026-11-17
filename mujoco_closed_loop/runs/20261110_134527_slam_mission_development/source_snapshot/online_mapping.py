"""Sensor-evidence clearing for moving objects; separate from old frozen maps."""
import numpy as np
from batched_mapper import BatchedVoxelMapper
from uav_continuum_whole_body_v2.mapping import voxel_indices,voxel_centers

class OnlineRayMapper(BatchedVoxelMapper):
    def __init__(self,resolution=.05,maximum_range=6.,bounds=None,clear_confirmations=3):
        super().__init__(resolution,maximum_range)
        self.bounds=bounds;self.clear_confirmations=clear_confirmations
        self.stamp=0.;self.free_votes={};self.last_hit={};self.frame_free=set();self.frame_hits=set();self.clearing_events=[]

    def seed_snapshot(self,snapshot):
        if abs(snapshot.resolution-self.resolution)>1e-12:raise ValueError('Map resolution mismatch')
        observed=snapshot.observed_indices
        if self.bounds is not None:
            p=voxel_centers(observed,self.resolution)
            observed=observed[np.all((p>=self.bounds[0])&(p<=self.bounds[1]),axis=1)]
        self.occupied=set(map(tuple,snapshot.occupied_indices))
        self.observed=set(map(tuple,observed))|self.occupied

    def begin_frame(self,stamp):
        self.stamp=float(stamp);self.frame_free.clear();self.frame_hits.clear()

    def _observe_indices(self,indices,occupied):
        if self.bounds is not None:
            points=voxel_centers(indices,self.resolution)
            indices=indices[np.all((points>=self.bounds[0])&(points<=self.bounds[1]),axis=1)]
        cells=set(map(tuple,indices));self.observed.update(cells)
        if not occupied:self.frame_free.update(cells)

    def integrate(self,points,sensor_origin,stride=1):
        points=np.asarray(points,dtype=float).reshape(-1,3)[::max(1,int(stride))]
        origin=np.asarray(sensor_origin,dtype=float)
        if not np.isfinite(points).all() or not np.isfinite(origin).all():raise ValueError('Nonfinite map input')
        delta=points-origin;distance=np.linalg.norm(delta,axis=1)
        valid=(distance>1e-9)&(distance<=self.maximum_range)
        points=points[valid];delta=delta[valid];distance=distance[valid]
        hits=set(map(tuple,voxel_indices(points,self.resolution)))
        self.occupied.update(hits);self.observed.update(hits);self.frame_hits.update(hits)
        for cell in hits:self.last_hit[cell]=self.stamp;self.free_votes.pop(cell,None)
        # A first surface proves only the prefix is free. Keep a voxel-scale
        # reserve at the endpoint; never clear through or behind that surface.
        reserve=1.5*self.resolution;keep=distance>reserve
        prefix=origin+delta[keep]*((distance[keep]-reserve)/distance[keep])[:,None]
        super().integrate_free_rays(prefix,origin)
        return len(points)

    def finish_frame(self):
        removed=[]
        for cell in (self.frame_free-self.frame_hits)&self.occupied:
            self.free_votes[cell]=self.free_votes.get(cell,0)+1
            if self.free_votes[cell]>=self.clear_confirmations and self.stamp-self.last_hit.get(cell,float('-inf'))>=.4:
                self.occupied.remove(cell);removed.append(cell);self.free_votes.pop(cell,None)
        if removed:self.clearing_events.append(dict(stamp=self.stamp,removed_voxels=removed,reason='repeated_measured_visible_free_prefix'))
        return len(removed)
