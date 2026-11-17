"""Strict rigid self-interior prior; certifies no space outside calibrated hardware."""
import itertools,numpy as np

def rigid_self_cells(T_WU,resolution,erosion=.005):
    # Only UAV fuselage and Qingyu rigid base, not flexible shape or safety margins.
    R=T_WU[:3,:3];p=T_WU[:3,3];half=np.array([.14,.10,.06])-erosion
    lo=p-.25;hi=p+.25
    grid=np.stack(np.meshgrid(*[np.arange(np.floor(lo[k]/resolution),np.ceil(hi[k]/resolution)+1,dtype=int) for k in range(3)],indexing='ij'),axis=-1).reshape(-1,3)
    corners=np.array(list(itertools.product([0.,1.],repeat=3)))
    local=((grid[:,None,:]+corners[None,:,:])*resolution-p)@R
    box=np.all(np.abs(local)<=half,axis=2)
    base=(np.linalg.norm(local[:,:,:2],axis=2)<=.0675-erosion)&(local[:,:,2]>=-.1004+erosion)&(local[:,:,2]<=-.03-erosion)
    return grid[np.all(box|base,axis=1)]

def add_rigid_self_prior(mapper,T_WU):
    cells=rigid_self_cells(T_WU,mapper.resolution);keys=set(map(tuple,cells));conflicts=keys & mapper.occupied
    # Never erase a sensor obstacle even if it conflicts with the calibrated self volume.
    mapper.observed.update(keys-conflicts)
    return {'kind':'initial_calibrated_rigid_self_interior_only','erosion_m':.005,'cells_added':len(keys-conflicts),'obstacle_conflicts':len(conflicts),'external_space_added':False}
