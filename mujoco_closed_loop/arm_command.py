"""Single-section command contract shared by simulation and Qingyu transport.

Internal units are radians. This module does not open ROS, a serial port or a
hardware client. Hardware adapters consume qingyu_joint_fields() explicitly.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np
from shape_limits import bounded_command, joint_bending_metrics

@dataclass(frozen=True)
class ArmCommand:
    sequence: int
    stamp_s: float
    theta_rad: float
    phi_rad: float
    enable: bool = True

    def normalized(self):
        if isinstance(self.sequence,bool) or not isinstance(self.sequence,int) or self.sequence<0:
            raise ValueError('sequence must be a non-negative integer')
        if not math.isfinite(self.stamp_s) or self.stamp_s<0:
            raise ValueError('stamp_s must be finite and non-negative')
        if not isinstance(self.enable,bool):raise ValueError('enable must be a bool')
        theta,phi=bounded_command(self.theta_rad,self.phi_rad)
        phi=math.atan2(math.sin(phi),math.cos(phi)) if theta>1e-10 else 0.
        return ArmCommand(self.sequence,self.stamp_s,theta,phi,self.enable)

    def qingyu_joint_fields(self):
        c=self.normalized()
        # Installed SDK API -> RViz bridge forwards phi1 unchanged. Its bending
        # plane is clockwise from +Y, whereas internal PCC uses CCW from +X:
        # beta_sdk = pi/2 - phi_pcc. Verified on the isolated installed nodes.
        beta=math.atan2(math.sin(math.pi/2-c.phi_rad),math.cos(math.pi/2-c.phi_rad)) if c.theta_rad>1e-10 else 0.
        return dict(theta1=math.degrees(c.theta_rad),phi1=math.degrees(beta),
                    theta2=0.,phi2=0.,theta3=0.,phi3=0.,enable=c.enable)

class MujocoArmBackend:
    """Apply commands to actuators; disabling removes their drive force.

    Passive joint damping/gravity remain active. No generalized state is set.
    A single backend owns its model's actuator coefficients.
    """
    def __init__(self,model,data,segments=8,collision_guard=None):
        self.model,self.data,self.segments=model,data,segments
        self.collision_guard=collision_guard
        self.ids=np.array([model.actuator('%s%d'%(axis,i)).id
            for i in range(segments) for axis in ('bx','by')],dtype=int)
        self.qpos_ids=np.array([model.jnt_qposadr[model.joint('%s%d'%(axis,i)).id]
            for i in range(segments) for axis in ('bx','by')],dtype=int)
        self.gain=model.actuator_gainprm[self.ids].copy()
        self.bias=model.actuator_biasprm[self.ids].copy()
        self.last=None;self.record=None

    def apply(self,command):
        c=command.normalized()
        if self.last is not None and (c.sequence<=self.last.sequence or c.stamp_s<self.last.stamp_s):
            raise ValueError('Out-of-order command; state and controls retained')
        if c.enable and self.collision_guard is not None:
            observed=self.observe();start=[observed['theta_rad'],observed['phi_rad']]
            verdict=self.collision_guard.transition(start,[c.theta_rad,c.phi_rad])
            if not verdict['accepted']:raise ValueError('Self-collision transition rejected: '+str(verdict['witness']))
        if c.enable:
            self.model.actuator_gainprm[self.ids]=self.gain
            self.model.actuator_biasprm[self.ids]=self.bias
            pair=[c.theta_rad*math.cos(c.phi_rad)/self.segments,
                  c.theta_rad*math.sin(c.phi_rad)/self.segments]
            self.data.ctrl[self.ids]=np.tile(pair,self.segments)
        else:
            self.model.actuator_gainprm[self.ids]=0.
            self.model.actuator_biasprm[self.ids]=0.
            self.data.ctrl[self.ids]=0.
        self.last=c
        self.record=dict(request=asdict(command),issued=asdict(c),
            theta_limited=abs(command.theta_rad-c.theta_rad)>1e-12,
            backend='mujoco_actuators',state_source='integrated_joint_coordinates')
        return self.record

    def observe(self):
        pairs=self.data.qpos[self.qpos_ids].reshape(-1,2)
        vector=pairs.sum(0);theta=float(np.linalg.norm(vector))
        return dict(stamp_s=float(self.data.time),theta_rad=theta,
            phi_rad=math.atan2(vector[1],vector[0]) if theta>1e-10 else 0.,
            source='mujoco_joint_coordinates',**joint_bending_metrics(pairs))
