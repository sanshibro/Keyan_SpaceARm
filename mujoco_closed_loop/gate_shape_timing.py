"""Rolling timing of a two-transition arm profile on an accepted single-gate base path.

The prediction consumes estimator pose, measured net bend and a local route
preview. It has no access to the plant state, route disturbance label or target
timestamps for the two transitions.
"""
import math
import numpy as np


def blend(u):
    a=np.clip(u,0.,1.)
    return a*a*a*(10.+a*(-15.+6.*a))


def blend_rate(u):
    a=np.clip(u,0.,1.)
    return 30.*a*a*(1.-a)**2


def enu_body_rotation(rpy):
    roll,pitch,yaw=rpy
    cr,sr=np.cos(roll),np.sin(roll);cp,sp=np.cos(pitch),np.sin(pitch);cy,sy=np.cos(yaw),np.sin(yaw)
    frd_ned=np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],
                      [sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],
                      [-sp,cp*sr,cp*cr]])
    return np.array([[0,1,0],[1,0,0],[0,0,-1]])@frd_ned@np.diag([1.,-1.,-1.])


class GateEnvelope:
    """Per-rigid-link conservative envelopes of the registered aircraft geometry."""
    def __init__(self, geometry, angle_pad):
        g=geometry
        nodes=g.m.geom_bodyid[g.ids]
        labels=np.unique(nodes)
        self.body=np.array([g.cat[np.flatnonzero(nodes==n)[0]]=='body' for n in labels])
        self.lo=[];self.hi=[]
        for theta in range(121):
            g.place([0.,0.,0.],theta)
            lo,hi=g.bounds()
            self.lo.append([lo[nodes==n].min(axis=0) for n in labels])
            self.hi.append([hi[nodes==n].max(axis=0) for n in labels])
        self.lo=np.array(self.lo);self.hi=np.array(self.hi)
        self.glo=g.glo.copy();self.ghi=g.ghi.copy()
        self.near=float(g.glo[:,0].min());self.far=float(g.ghi[:,0].max())
        # Maximum displacement per degree of the 0.52 m kinematic chain.
        self.pad=angle_pad

    def inspect(self,xyz,rotation,angles):
        angle=np.clip(np.asarray(angles),0.,120.)
        a=np.floor(angle).astype(int);b=np.ceil(angle).astype(int)
        pad=(self.pad*np.minimum(angle-a,b-angle))[:,None,None]
        lo=np.minimum(self.lo[a],self.lo[b])-pad
        hi=np.maximum(self.hi[a],self.hi[b])+pad
        center=(lo+hi)*.5@rotation.T+xyz[:,None,:]
        rad=(hi-lo)*.5@np.abs(rotation.T)
        low=center-rad;high=center+rad
        sep=np.maximum(np.maximum(low[:,:,None,:]-self.ghi,self.glo-high[:,:,None,:]),0.)
        pair=np.einsum('tijk,tijk->tij',sep,sep).min(axis=2)
        return (np.sqrt(pair[:,self.body].min(axis=1)),np.sqrt(pair[:,~self.body].min(axis=1)),
                high[:,:,0].max(axis=1)-self.near,low[:,:,0].min(axis=1)-self.far)


class RollingGateTiming:
    def __init__(self, geometry, config):
        self.cfg=config['controller'];self.g=geometry
        self.gate_deg=config['gate_angle_deg'];self.final_deg=config['terminal_angle_deg']
        self.up=config['bend_duration_s'];self.down=config['return_duration_s']
        self.times=np.arange(0.,self.cfg['horizon_s']+.001,self.cfg['predict_step_s'])
        self.bend_start=None;self.return_start=None
        self.seen_gate=False

    def sample(self,t):
        if self.bend_start is None:return 0.
        q=self.gate_deg*blend((t-self.bend_start)/self.up)
        if self.return_start is not None:
            q+=(self.final_deg-self.gate_deg)*blend((t-self.return_start)/self.down)
        return float(q)

    def update(self,t,observation,preview):
        c=self.cfg;h=self.times
        info=dict(action='retain',reason='no_new_transition',candidates=[])
        if observation is None:
            info['reason']='missing_feedback';return info
        xyz,vel,rpy,measured,age_pos,age_att=observation
        if not np.isfinite(np.r_[xyz,vel,rpy,measured,age_pos,age_att]).all() or min(age_pos,age_att)<0 or max(age_pos,age_att)>c['estimator_age_limit_s']:
            info['reason']='stale_feedback';return info
        R=enu_body_rotation(rpy)
        b,a,front,rear=self.g.inspect(xyz[None,:],R,[measured])
        front=float(front[0]);rear=float(rear[0])
        if front>=0. and rear<=0.:self.seen_gate=True
        info.update(x_m=float(xyz[0]),z_m=float(xyz[2]),angle_deg=float(measured),front_m=front,rear_m=rear,
                    body_now_m=float(b[0]),arm_tool_now_m=float(a[0]),seen_gate=self.seen_gate)
        if self.return_start is not None:
            info['reason']='following_accepted_return';return info
        if self.bend_start is None:
            if abs(xyz[2]-preview[0,2])>c['ready_altitude_tolerance_m'] or abs(vel[2])>c['ready_vz_mps']:
                info['reason']='settling_to_route';return info
        elif t<self.bend_start+self.up:
            info['reason']='following_accepted_bend';return info
        else:
            if not self.seen_gate or rear<c['tail_buffer_m']:
                info['reason']='holding_until_measured_rear_clears';return info
            if abs(measured-self.gate_deg)>c['bend_tolerance_deg']:
                info['reason']='waiting_for_measured_shape';return info
        # Preview is the presently issued path continued without clairvoyance about future pauses.
        vref=(preview[1]-preview[0])/(h[1]-h[0])
        xyz_pred=(preview+(xyz-preview[0])*np.exp(-h[:,None]/c['position_error_decay_s'])+
                  (vel-vref)*(h*np.exp(-h/c['velocity_error_decay_s']))[:,None])
        growing=self.bend_start is None
        start,end,duration=(0.,self.gate_deg,self.up) if growing else (self.gate_deg,self.final_deg,self.down)
        offsets=c['delay_options_s'] if growing else [0.]
        for delay in offsets:
            raw=(h-delay)/duration
            cmd=start+(end-start)*blend(raw)
            angles=np.clip(cmd+(measured-start)*np.exp(-h/2.),0.,120.)
            rates=np.abs((end-start)*blend_rate(raw)/duration)
            bc,ac,entry,exit_gap=self.g.inspect(xyz_pred,R,angles)
            inside=(entry>=-c['crossing_buffer_m'])&(exit_gap<=c['crossing_buffer_m'])
            geometry_ok=bool(bc.min()>=c['body_margin_m']+c['prediction_reserve_m'] and
                             ac.min()>=c['arm_margin_m']+c['prediction_reserve_m'])
            settled=bool(np.all(np.abs(angles[inside]-self.gate_deg)<=c['bend_tolerance_deg']) and
                         np.all(rates[inside]<=c['settled_rate_deg_s']))
            visible=bool(inside.any())
            admissible=geometry_ok and settled and (visible if growing else True)
            info['candidates'].append(dict(delay_s=delay,body_m=float(bc.min()),arm_tool_m=float(ac.min()),
                    gate_in_preview=visible,settled=settled,geometry_ok=geometry_ok,admissible=admissible))
        valid=[k for k in info['candidates'] if k['admissible']]
        if not valid:
            info['reason']='gate_beyond_preview' if growing and not any(k['gate_in_preview'] for k in info['candidates']) else 'no_valid_candidate'
            return info
        chosen=valid[-1];info['selected_delay_s']=chosen['delay_s']
        if growing and chosen['delay_s']>c['commit_within_s']:
            info['reason']='a_later_start_is_still_feasible';return info
        # The zero-delay trajectory must be validated itself, even when a later
        # transition is admissible. Preserve the last accepted command otherwise.
        if not info['candidates'][0]['admissible']:
            info['reason']='immediate_start_not_admissible';return info
        if growing:
            self.bend_start=t;info.update(action='bend',reason='last_valid_start_window')
        else:
            self.return_start=t;info.update(action='return',reason='rear_clear_and_return_validated')
        return info
