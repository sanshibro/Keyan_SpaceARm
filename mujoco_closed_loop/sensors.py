"""Rendered RGB-D and visible-surface scanning. No pose fields in measurements."""
from dataclasses import dataclass
import numpy as np
import mujoco

OPTICAL_FROM_GL=np.diag([1.,-1.,-1.])

@dataclass
class RGBD:
    stamp: float
    rgb: np.ndarray
    depth_m: np.ndarray
    K: np.ndarray
    frame: str
    # Simulator-only explicit first-surface-beyond-range evidence. Ordinary
    # missing RGB-D depth (NaN, stereo failure, too close) is never free space.
    no_return_mask: np.ndarray = None

class RGBDSensor:
    def __init__(self,model,camera,width=640,height=480,shadows=False):
        self.model=model;self.camera=camera;self.id=model.camera(camera).id
        self.renderer=mujoco.Renderer(model,height=height,width=width)
        # Ideal unshadowed illumination saves a second shadow-map geometry
        # pass. It changes RGB lighting, never depth, visibility or physics.
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=shadows
        focal=height/(2*np.tan(np.deg2rad(model.cam_fovy[self.id])/2))
        self.K=np.array([[focal,0,(width-1)/2],[0,focal,(height-1)/2],[0,0,1.]])
    def read(self,data):
        self.renderer.update_scene(data,camera=self.camera)
        rgb=self.renderer.render().copy();self.renderer.enable_depth_rendering()
        depth=self.renderer.render().copy();self.renderer.disable_depth_rendering()
        no_return=np.isfinite(depth)&(depth>6)
        depth[(depth<.15)|(depth>6)]=np.nan
        return RGBD(float(data.time),rgb,depth,self.K.copy(),self.camera+'_optical',no_return)
    def close(self):self.renderer.close()

class SurfaceLidar:
    """10Hz scanner approximation: 360 azimuth x 16 elevation, +/-30deg, 0.1..6m.
    Returns points in sensor frame; callers must register with estimated timed pose.
    Robot groups 1/2 block rays; self hits are discarded without seeing through them.
    """
    def __init__(self,seed=0,azimuth_count=360,elevation_count=16,vertical_fov=60.,maximum_range=6.):
        az,el=np.meshgrid(np.linspace(-np.pi,np.pi,azimuth_count,endpoint=False),np.deg2rad(np.linspace(-vertical_fov/2,vertical_fov/2,elevation_count)))
        self.directions=np.column_stack([(np.cos(el)*np.cos(az)).ravel(),(np.cos(el)*np.sin(az)).ravel(),np.sin(el).ravel()]);self.rng=np.random.default_rng(seed)
        self.maximum_range=maximum_range;self.vertical_fov=vertical_fov
    @classmethod
    def super_general360(cls,seed=0):
        # Official SUPER dense.yaml: GENERAL_360, 178deg, 128 lines, 15m,10Hz.
        # Azimuth decimated to1deg from official0.4deg for this CPU backend.
        return cls(seed,360,128,178.,15.)
    def read_body(self,model,data,body_id,offset=(0.,0.,.2)):
        """Generate local rays from a calibrated rigid body mount."""
        rotation=data.xmat[body_id].reshape(3,3)
        position=data.xpos[body_id]+rotation@np.asarray(offset)
        return self.read(model,data,position,rotation)
    def read(self,model,data,sensor_position,sensor_rotation):
        vec=np.ascontiguousarray(self.directions@np.asarray(sensor_rotation).T,dtype=np.float64)
        n=len(vec);geom=np.full(n,-1,dtype=np.int32);distance=np.full(n,-1.,dtype=np.float64)
        mask=np.array([1,1,1,0,0,0],dtype=np.uint8)
        mujoco.mj_multiRay(model,data,np.asarray(sensor_position,dtype=np.float64),vec.ravel(),mask,1,-1,geom,distance,n,self.maximum_range)
        hit=geom>=0;own=np.zeros(n,dtype=bool);own[hit]=np.isin(model.geom_group[geom[hit]],[1,2])
        valid=hit&~own&(distance>=.1)&(distance<=self.maximum_range)
        values=distance[valid]+self.rng.normal(0,.003,int(valid.sum()))
        free=~own&((distance<0)|(distance>self.maximum_range))
        return dict(stamp=float(data.time),frame='lidar',points=self.directions[valid]*values[:,None],directions=self.directions[valid],range_m=[.1,self.maximum_range],frequency_hz=10,vertical_fov_deg=self.vertical_fov,free_endpoints=self.directions[free]*self.maximum_range,self_occluded_rays=int(own.sum()))
