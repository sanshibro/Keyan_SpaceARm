"""Independent shape checker on a private kinematic copy of the new platform.

This is dense sampled checking, not a continuous-time collision certificate.
It never writes the executing simulation's state and cannot send SDK commands.
"""
import math
import numpy as np
import mujoco
from super_metric_model import build_super_metric
from model import SimArmBridge
from shape_limits import bounded_command

class SelfCollisionOracle:
    def __init__(self,clearance_m=.001,angular_step_deg=.5):
        xml,info=build_super_metric(fixed=True)
        self.model=mujoco.MjModel.from_xml_string(xml);self.data=mujoco.MjData(self.model)
        self.bridge=SimArmBridge();self.platform_id=info['platform_id']
        self.clearance_m=float(clearance_m);self.step=math.radians(angular_step_deg)
        if not 0<=self.clearance_m<=.02 or not 0<self.step<=math.radians(5):raise ValueError('Invalid checker resolution')
        self.model.geom_margin[:]=self.clearance_m
        self.model.geom_gap[:]=0.
        self.cache={}

    def check(self,theta,phi):
        theta,phi=bounded_command(theta,phi);phi=math.atan2(math.sin(phi),math.cos(phi)) if theta>1e-10 else 0.
        key=(float(theta),float(phi))
        if key in self.cache:return self.cache[key]
        self.bridge.set_kinematic_shape(self.model,self.data,theta,phi)
        mujoco.mj_forward(self.model,self.data)
        hits=[dict(a=self.model.geom(c.geom1).name,b=self.model.geom(c.geom2).name,clearance_m=float(c.dist))
              for c in self.data.contact[:self.data.ncon] if c.dist<self.clearance_m]
        result=dict(accepted=not hits,theta_rad=theta,phi_rad=phi,contacts=hits)
        if len(self.cache)>20000:self.cache.clear()
        self.cache[key]=result;return result

    def transition(self,start,end):
        a=np.array(start,float);b=np.array(end,float)
        if a.shape!=(2,) or b.shape!=(2,):raise ValueError('Expected theta/phi pairs')
        bounded_command(*a);bounded_command(*b)
        delta=b-a;delta[1]=math.atan2(math.sin(delta[1]),math.cos(delta[1]))
        n=max(1,int(math.ceil(np.max(np.abs(delta))/self.step)))
        for i in range(n+1):
            result=self.check(*(a+delta*i/n))
            if not result['accepted']:return dict(accepted=False,checked=i+1,witness=result,kind='dense_shape_transition')
        return dict(accepted=True,checked=n+1,witness=None,kind='dense_shape_transition')

    def sequence(self,shapes):
        shapes=np.asarray(shapes,dtype=float);checked=0
        if shapes.ndim!=2 or shapes.shape[1]!=2 or len(shapes)==0:raise ValueError('Expected nonempty Nx2 shapes')
        previous=shapes[0]
        for i,current in enumerate(shapes):
            result=self.transition(previous,current);checked+=result['checked']
            if not result['accepted']:return dict(result,segment=i,checked=checked,platform_id=self.platform_id)
            previous=current
        return dict(accepted=True,checked=checked,platform_id=self.platform_id,
            angular_step_deg=math.degrees(self.step),clearance_m=self.clearance_m,
            kind='dense_sampled_self_collision; not continuous certification')
