"""M2 native PX4 / MuJoCo loop with frozen causal shape feedback. Run on the simulation NUC."""
import argparse
from collections import deque
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
import mujoco

from gate_feedback_flight import FlightLoop,ramp
from gate_feedback_model import ObserverGeometry,LAUNCH,P,verify_frozen
from gate_live_shape import FrozenAssets,LiveEstimator,LiveGauge
from gate_shape_timing import enu_body_rotation
from gate_distance_audit import DistanceAudit
from m2_scene import HOVER
from m2_plant import build,task_pose,IndustryOptics
from m2_servo import GravityTrim,ShapeObservation,GEAR_TRIGGER,bound,shape_vector,project_bend
from m2_self_registry import partition
from round_f_sitl import run

METHODS=('straight','fixed','sequential','unregularized','coupled','stop','hard')
MOTION_START=21.
PERIOD=.020

class EstimatedMechanism:
    """Dedicated kinematic state buffer, independent of both the integrator and planner."""
    def __init__(self,p):
        self.fk=ObserverGeometry(p);self.model=p.m;self.base=p.base
        self.audit=DistanceAudit(p.m,tolerance_m=.0001,strict=False,obstacle_prefix='COL_')
        self.hazard,self.fit=partition(p.m,self.audit.self_pairs)
        if not len(self.hazard):raise ValueError('Empty hazard registry')
        self.mass=p.mass
        from m2_geometry import NativeSelf
        self.native=NativeSelf(self.model)
        self.finger_fit={}

    def narrow(self,pairs,label,lo,hi):
        native=self.native.query(self.audit.d,pairs)
        lower,upper=native['lower'],native['upper']
        if 0.<lower<=upper+1e-10 and upper-lower<.00010:
            a,b=native['which']
            return dict(lower=lower,upper=upper,geom_a=a,geom_b=b)
        # Independently resolve contacts / wide bounds with signed GJK + EPA.
        return self.audit._scan(label,pairs,lo,hi)
    def read(self,xyz,R,b,include_fit=True):
        self.fk.place_estimate(xyz,R,b);self.audit.d.qpos[:]=self.fk.d.qpos
        mujoco.mj_kinematics(self.model,self.audit.d)
        lo,hi=self.audit.bounds()
        hazard=self.narrow(self.hazard,'hazard',lo,hi)
        fit=None
        if include_fit:
            fit_rows=[]
            for a,c in self.fit:
                a,c=int(a),int(c)
                if (a,c) in self.finger_fit:row=self.finger_fit[a,c]
                else:
                    row=self.narrow(np.array([[a,c]]),'fit_%d_%d'%(a,c),lo,hi)
                    if all(self.model.geom(i).name.startswith('collision_gripper_finger') for i in (a,c)):
                        # The FK observes the registered open command. Jaw
                        # excursions in the live physics are independently scored.
                        self.finger_fit[a,c]=row
                fit_rows.append(row)
            fit=dict(min(fit_rows,key=lambda item:item['lower']))
            fit['all_lower']=[row['lower'] for row in fit_rows]
        return hazard,fit

    def com(self,xyz,R,b):
        self.fk.place_estimate(xyz,R,b)
        mujoco.mj_comPos(self.model,self.fk.d)
        return (self.fk.d.subtree_com[self.base]-self.fk.d.xpos[self.base]).copy()

class Controller:
    def __init__(self,p,method,folder):
        from m2_rolling import RollingPlanner
        from m2_inspection_ik import goal_family
        self.plant=p;self.method=method;self.folder=folder
        self.task=task_pose(p);self.mechanism=EstimatedMechanism(p)
        self.task['candidates']=goal_family(p,self.task)
        self.inspection_tcp=None
        self.rolling=RollingPlanner(p,method,self.task)
        self.rolling.event_path=folder/'rolling-live.jsonl'
        rotor_sites=[p.m.site('rotor'+str(i)).id for i in range(4)]
        if not np.all(p.m.site_bodyid[rotor_sites]==p.base) or not np.allclose(p.m.site_quat[rotor_sites],[1,0,0,0]):
            raise ValueError('Rotor thrust axes must be registered in the common body frame')
        self.trim=GravityTrim(p.check,method in ('coupled','hard','stop'),p.m.site_pos[rotor_sites].mean(axis=0))
        self.observation=ShapeObservation()
        self.ref=HOVER.copy();self.yaw=0.;self.phase='takeoff'
        self.des=np.zeros(2);self.issued=np.zeros(2);self.integral=np.zeros(2)
        self.ff=(np.zeros(3),np.zeros(3));self.yaw_rate=0.;self.exit_time=None;self.s=P['servo'];self.last_time=None

    def control(self,t,obs,xyz,velocity,rpy,age_p,age_r):
        fresh=bool(xyz is not None and velocity is not None and rpy is not None and
                   -.025<=age_p<=.25 and -.025<=age_r<=.25 and obs['valid'] and obs['age']<=.30)
        b=np.asarray(obs['b'],dtype=float);fresh=fresh and np.isfinite(b).all()
        dt=PERIOD if self.last_time is None else t-self.last_time
        self.last_time=t
        phase='takeoff' if t<12 else 'neutral_anchor' if t<16 else 'qualify_start'
        hold='none';q=np.r_[self.ref,self.yaw,np.linalg.norm(self.des),np.arctan2(self.des[1],self.des[0])]
        dq=np.zeros(6);ddq=np.zeros(6);gap=fit_gap=np.nan;pair=[-1,-1];fit_pair=[-1,-1]
        fit_all=np.full(len(self.mechanism.fit),np.nan)
        com=np.full(3,np.nan);tau=np.zeros(2);tilt=np.zeros(2);acc_trim=np.zeros(3)
        projection='nominal';window=-1;window_status='initial';sample_time=np.nan
        disabled_shape=(self.method=='straight')
        if t>=16 and t<MOTION_START and self.method=='fixed':
            self.des=shape_vector(np.deg2rad(80.)*ramp((t-16.)/4.),np.pi)
        if fresh:
            R=enu_body_rotation(rpy);yaw=np.arctan2(R[1,0],R[0,0])
            state,rate,accel=self.observation.update(t,xyz,velocity,yaw,b)
            hazard,fit=self.mechanism.read(xyz,R,b)
            gap=hazard['lower'];pair=[hazard['geom_a'],hazard['geom_b']]
            fit_gap=fit['lower'];fit_pair=[fit['geom_a'],fit['geom_b']]
            fit_all=np.asarray(fit['all_lower'])
            com=self.mechanism.com(xyz,R,b)
            if t>=MOTION_START-.5:
                self.rolling.tick(t,state,rate,accel)
            if t>=MOTION_START:
                s=self.rolling.sample(t);window=int(s['active_id']) if s['active_id'] is not None else -1;window_status=s['status']
                sample_time=float(s.get('local_time',np.nan))
                if window_status=='done':
                    phase='inspect'
                    q=np.asarray(s['q'],float).copy()
                    if self.exit_time is None:
                        self.exit_time=t
                        cy,sy=np.cos(q[3]),np.sin(q[3])
                        R_goal=np.array([[cy,-sy,0.],[sy,cy,0.],[0.,0.,1.]])
                        self.mechanism.fk.place_estimate(q[:3],R_goal,shape_vector(q[4],q[5]))
                        self.inspection_tcp=self.mechanism.fk.tip()[0].copy()
                    if self.method not in ('straight','fixed'):
                        self.des=self.mechanism.fk.geom_axis_bend(R,np.asarray(self.task['axis']))
                    self.mechanism.fk.place_estimate(np.zeros(3),R,b)
                    self.ref=self.inspection_tcp-self.mechanism.fk.tip()[0]
                    self.yaw=float(q[3])
                elif window_status in ('running','waiting','stopped','rejected'):
                    q,dq,ddq=(np.asarray(s[k],dtype=float).copy() for k in ('q','dq','ddq'))
                    if any(v.shape!=(6,) or not np.isfinite(v).all() for v in (q,dq,ddq)):
                        raise ValueError('Malformed committed window')
                    self.ref=q[:3].copy();self.yaw=float(q[3])
                    self.des=shape_vector(q[4],q[5])
                    phase='execute' if window_status=='running' else 'safe_prefix_end' if window_status=='rejected' else 'rolling_hold'
                    if window_status!='running':hold=window_status
                else:raise ValueError('Unknown rolling state '+str(window_status))
            acc_trim,tau,tilt=self.trim.evaluate(com,self.plant.mass,R,dt,ddq[2])
            if disabled_shape:self.des=np.zeros(2)
            if t>=12.:
                desired=self.des.copy()
                def gap_at(shape):
                    hazard_at,fit_at=self.mechanism.read(xyz,R,shape)
                    return hazard_at['lower'] if fit_at['lower']>=0. else fit_at['lower']
                if gap<GEAR_TRIGGER:
                    self.des,projection=project_bend(b,desired,gap,gap_at)
                error=self.des-b
                self.integral=bound(self.integral+dt*self.s['ki']*error,np.deg2rad(self.s['i_limit_deg']))
                unit=np.array([np.cos(q[5]),np.sin(q[5])])
                bend_rate=dq[4]*unit+q[4]*dq[5]*np.array([-unit[1],unit[0]])
                # HIL identification on separate development traces measured
                # 0.034–0.037 s actuator lag along the two bending axes.
                lead=.035*bend_rate if phase=='execute' and projection=='nominal' else np.zeros(2)
                target=bound(self.des+self.s['kp']*error+self.integral+lead,np.deg2rad(self.s['command_max_deg']))
                if disabled_shape and gap>=GEAR_TRIGGER:target[:]=0.
                # On recovery the integral correction may pull towards the very
                # leg being avoided. Project the final command as well.
                if gap<GEAR_TRIGGER:target,_=project_bend(b,target,gap,gap_at)
                self.issued+=bound(target-self.issued,dt*np.deg2rad(self.s['command_rate_deg_s']))
        elif t>=MOTION_START:
            phase='feedback_hold';hold='feedback'
        self.phase=phase
        self.ff=(dq[:3].copy(),ddq[:3]+acc_trim)
        self.yaw_rate=float(dq[3])
        return dict(t=float(t),phase=phase,hold=hold,valid=fresh,shape_valid=bool(obs['valid']),
            shape_age=obs['age'],age_p=age_p,age_r=age_r,
            des_b=self.des.copy(),est_b=b.copy(),issued_b=self.issued.copy(),integral_b=self.integral.copy(),
            px4_xyz=np.array(xyz if xyz is not None else [np.nan]*3),px4_rpy=np.array(rpy if rpy is not None else [np.nan]*3),
            ref_world=self.ref.copy(),reference_yaw=self.yaw,plan_q=q,plan_dq=dq,plan_ddq=ddq,active_window=window,
            window_status=window_status,local_trajectory_time=sample_time,ff_enu=self.ff[0].copy(),acc_ff_enu=self.ff[1].copy(),
            est_hazard_gap=gap,est_fit_gap=fit_gap,est_fit_gaps=fit_all,est_hazard_pair=pair,est_fit_pair=fit_pair,projection=projection,
            trim_enabled=self.trim.enabled,est_com_offset_enu=com,required_roll_pitch_Nm_frd=tau,trim_angle_frd=tilt,trim_accel_enu=acc_trim)

class Loop(FlightLoop):
    def __init__(self,plant,folder,case,horizon):
        self.plant=plant;self.folder=folder;self.case=case;self.mechanics=False;self.horizon=horizon
        self.ctrl=Controller(plant,case['strategy'],folder)
        self.camera=self.gauge=self.filter=None;self.strain_pending=[];self.visual_pending=deque()
        self.rows=[];self.last_source=None;self.yaw_enu=0.;self.yaw_rate_enu=0.;self.translation_feedforward=(np.zeros(3),np.zeros(3))
        self.jaw_joints=[plant.m.joint('grip'+str(i)).id for i in range(3)]
        self.jaw_actuators=[plant.m.actuator('grip'+str(i)).id for i in range(3)]
        self.jaw_qpos=plant.m.jnt_qposadr[self.jaw_joints]
        if not np.allclose(plant.m.jnt_range[self.jaw_joints],[[0.,.024]]*3):
            raise ValueError('CAD jaw stroke changed')
        # Qingyu slide axes point toward the tool center. Zero is the open stop.
        plant.d.ctrl[self.jaw_actuators]=0.
    def start(self,us):
        self.release=us;assets=FrozenAssets()
        self.gauge=LiveGauge(self.plant,str(self.case['seed']),'nominal',assets=assets,horizon_s=self.horizon+4.)
        self.filter=LiveEstimator(assets=assets)
        self.camera=IndustryOptics(self.plant,self.folder/'optics',condition='natural')
        self.strain_pending.extend(self.gauge.sample(0.,self.plant))
    def tick(self,*args):
        self.plant.d.ctrl[self.jaw_actuators]=0.
        previous=len(self.rows)
        result=super().tick(*args)
        if len(self.rows)>previous:
            self.rows[-1]['grip_command_m']=self.plant.d.ctrl[self.jaw_actuators].copy()
            self.rows[-1]['grip_observed_m']=self.plant.d.qpos[self.jaw_qpos].copy()
        self.yaw_enu=self.ctrl.yaw;self.yaw_rate_enu=self.ctrl.yaw_rate;self.translation_feedforward=self.ctrl.ff
        return result
    def save(self):
        self.ctrl.rolling.close()
        self.ctrl.rolling.export(self.folder/'rolling')
        if self.camera:self.camera.close()
        for node in (self.gauge,self.filter):
            if node is not None:node.save(self.folder/'sensors')
        if self.rows:np.savez_compressed(self.folder/'control.npz',**{k:np.asarray([r[k] for r in self.rows]) for k in self.rows[0]})
        summary=dict(task=self.ctrl.task,scene=self.plant.scene_digest,motion_start_s=MOTION_START,
            first_inspection_s=self.ctrl.exit_time,captures=len(self.rows),case=self.case,
            committed_inspection_tcp=(self.ctrl.inspection_tcp.tolist() if self.ctrl.inspection_tcp is not None else None),
            ff_kind='PX4 offboard acceleration / offset-COM rotor thrust moment',rate_P_I_s=self.ctrl.trim.integration_tau.tolist(),
            rotor_centroid_flu_m=self.ctrl.trim.rotor_center_body.tolist(),
            stiffness_Nm_per_rad=self.ctrl.trim.stiffness.tolist(),max_extra_tilt_deg=5.,
            fit_geometry_names=[[self.plant.m.geom(int(a)).name,self.plant.m.geom(int(b)).name] for a,b in self.ctrl.mechanism.fit],
            jaw_open_command_m=[0.,0.,0.],jaw_stroke_range_m=[0.,.024])
        (self.folder/'control-summary.json').write_text(json.dumps(summary,indent=2)+'\n')

def fly(strategy,seed,instance,out,development,gate_path):
    from m2_gate import authorize,LOCK
    registration=authorize(strategy,seed,development,gate_path)
    runtime=inspect.getsource(run)
    if 'translation_feedforward' not in runtime or 'yaw_rate_enu' not in runtime or 'yaw_enu' not in runtime:
        raise RuntimeError('Native HIL sender must pass translation, heading and its planned rate to PX4')
    p=build();case=dict(method='proposed',strategy=strategy,seed=seed,case='%s-%d'%(strategy,seed))
    lock=verify_frozen();out=out.resolve()
    if lock!=LOCK:raise ValueError('Perception identity differs from the M2 registration')
    if out.exists():raise FileExistsError(out)
    loop=Loop(p,out,case,70.)
    mission=dict(case=case,scene=p.scene_digest,freeze=lock,registration=registration,camera_delay_s=.20,motion_start_s=MOTION_START,
        actuator_limit_deg_s=float(P['servo']['command_rate_deg_s']),
        gate_sha256=hashlib.sha256(gate_path.read_bytes()).hexdigest())
    try:
        run(out,case['case'],.4,reference_fn=lambda _t:('takeoff',HOVER-LAUNCH,0.),score_end=loop.horizon,noise_seed=seed,
            mission_record=mission,plant_factory=lambda:p,flight_loop=loop,instance=instance,
            protocol_name='m2_rolling_inspection.json',wall_limit_s=3600)
    finally:
        if out.is_dir():loop.save()
        else:loop.ctrl.rolling.close()
    if not (out/'flight.ulg').is_file():raise RuntimeError('No native PX4 ULog export')
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('strategy',choices=METHODS);a.add_argument('seed',type=int);a.add_argument('instance',type=int)
    a.add_argument('out',type=Path);a.add_argument('--gate',type=Path,required=True);a.add_argument('--development',action='store_true')
    k=a.parse_args();fly(k.strategy,k.seed,k.instance,k.out,k.development,k.gate)
