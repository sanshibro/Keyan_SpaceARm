"""Register RGB-D rays using estimated camera poses; self hits never become obstacles."""
import numpy as np
import mujoco

def integrate_rgbd(mapper,frame,T_WC,self_mask=None,stride=6,self_policy="visible_prefix"):
    yy,xx=np.mgrid[0:frame.depth_m.shape[0]:stride,0:frame.depth_m.shape[1]:stride]
    z=frame.depth_m[yy,xx];good=np.isfinite(z)
    x=(xx-frame.K[0,2])*z/frame.K[0,0];y=(yy-frame.K[1,2])*z/frame.K[1,1]
    points=np.stack([x,y,z],axis=-1)[good];world=points@T_WC[:3,:3].T+T_WC[:3,3]
    own=np.zeros(len(world),dtype=bool) if self_mask is None else self_mask[yy,xx][good]
    mapper.integrate(world[~own],T_WC[:3,3])
    if self_policy=='visible_free':mapper.integrate_free_rays(world[own],T_WC[:3,3])
    if self_policy=='visible_prefix':
        # Every valid depth measures a visible segment, even if its endpoint
        # might be self or an external object. Keep the ambiguous endpoint
        # unknown and stop carving 1.5 voxels before the measured surface.
        if hasattr(mapper,'integrate_visible_prefix'):
            mapper.integrate_visible_prefix(world[own],T_WC[:3,3])
        else:
            delta=world[own]-T_WC[:3,3];distance=np.linalg.norm(delta,axis=1)
            reserve=1.5*mapper.resolution;valid=distance>reserve
            endpoints=delta[valid]*((distance[valid]-reserve)/distance[valid])[:,None]+T_WC[:3,3]
            mapper.integrate_free_rays(endpoints,T_WC[:3,3])
    no_return=getattr(frame,'no_return_mask',None)
    if no_return is not None:
        far=no_return[yy,xx]
        directions=np.stack([(xx-frame.K[0,2])/frame.K[0,0],(yy-frame.K[1,2])/frame.K[1,1],np.ones(xx.shape)],axis=-1)[far]
        directions/=np.linalg.norm(directions,axis=1,keepdims=True)
        endpoints=(directions*min(6.,mapper.maximum_range))@T_WC[:3,:3].T+T_WC[:3,3]
        mapper.integrate_free_rays(endpoints,T_WC[:3,3])
    return int((~own).sum()),int(own.sum())

def self_hit_mask(sensor):
    """Exact render-object self mask: development audit only, not final perception evidence."""
    r=sensor.renderer;r.enable_segmentation_rendering();seg=r.render().copy();r.disable_segmentation_rendering()
    geom=(seg[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))&(seg[:,:,0]>=0)
    mask=np.zeros(geom.shape,dtype=bool);ids=seg[:,:,0][geom]
    mask[geom]=sensor.model.geom_group[ids]>0
    return mask
