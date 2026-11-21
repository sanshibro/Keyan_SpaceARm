"""Closed-loop industrial inspection using real HIL motor outputs and causal frozen sensing."""
import argparse,hashlib,json,math,shutil
from collections import deque
from pathlib import Path
import numpy as np
from gate_feedback_flight import FlightLoop,initial_reference,ramp
from gate_feedback_model import ObserverGeometry,limit_vector,LAUNCH,P,verify_frozen
from gate_live_shape import FrozenAssets,LiveEstimator,LiveGauge
from gate_shape_timing import enu_body_rotation
from round_f_sitl import run
from m1_geometry import GalleryGeometry
from m1_layout import HOVER,FACE,AXIS,METHODS,SEEDS
from m1_plant import build,IndustryOptics,task_pose
from m1_plan import IndustryAdapter,writej
from uav_continuum_whole_body_v2.types import Limits

class Controller:
    def __init__(self,p,decision,cert):
        self.fk=ObserverGeometry(p)
        self.geom=GalleryGeometry(p,Limits(body_margin=.06,arm_margin=.06,tilt_max=0))
        self.task=task_pose(p);self.axis=AXIS
        self.method=decision['method'];self.accepted=decision['accepted'];self.elapsed=0.
        self.start=np.asarray(decision['start']);self.end=np.asarray(decision['end'])
        self.q=np.zeros((1,6));self.t=np.zeros(1)
        if self.accepted:
            with np.load(cert) as f:self.q=f['q'];self.t=f['t']
        self.integral=np.zeros(2);self.issued=np.zeros(2);self.des=np.zeros(2)
        self.ref=HOVER.copy();self.yaw=0.;self.phase='takeoff';self.checked=0;self.exit_time=None
        self.s=P['servo'];self.holding=False
    def sample(self,clock):
        j=min(np.searchsorted(self.t,clock),len(self.t)-1)
        if j==0:return self.q[0].copy()
        lo=j-1;u=np.clip((clock-self.t[lo])/(self.t[j]-self.t[lo]),0,1)
        return (1-u)*self.q[lo]+u*self.q[j]
    def control(self,t,obs,xyz,velocity,rpy,age_p,age_r):
        fresh=xyz is not None and velocity is not None and rpy is not None and -.025<=age_p<=.25 and -.025<=age_r<=.25 and obs['valid'] and obs['age']<=.30
        estimate=np.asarray(obs['b']);g=self.fk;clear=np.full(2,np.nan);ahead=clear.copy();hit=False
        q=self.sample(self.elapsed);hold='none'
        if fresh:
            R=enu_body_rotation(rpy);g.place_estimate(xyz,R,estimate)
            components=self.geom.components(g.d,xyz);components[:,6]=0
            clear=self.geom.query(components,np.r_[xyz,0.,0,0])[0]
            # Check the measured pose and a short braking projection with the actual attitude.
            v=np.asarray(velocity);a=min(.24,.10+np.linalg.norm(v)/(2*.75))*v
            ahead=self.geom.query(components,np.r_[xyz+a,0.,0,0])[0]
            hit=g.has_self_contact();self.checked+=1
        if t<12:self.phase='takeoff'
        elif t<16:self.phase='neutral_anchor'
        elif not self.accepted:
            self.phase='rejected_hover';hold='planner_rejected'
        elif t<21.:
            self.phase='qualify_start'
            if self.method=='fixed':
                self.des=self.start[4]*ramp((t-16)/4.)*np.array([np.cos(self.start[5]),np.sin(self.start[5])])
        else:
            self.phase='execute'
            if not fresh:hold='feedback'
            elif hit:hold='self'
            elif min(np.min(clear),np.min(ahead))<.060:hold='clearance'
            elif np.linalg.norm(xyz-self.ref)>.14:hold='tracking'
            if hold=='none':
                self.elapsed=min(self.t[-1],self.elapsed+.02);q=self.sample(self.elapsed)
            self.ref=q[:3].copy();self.yaw=float(q[3])
            self.des=q[4]*np.array([np.cos(q[5]),np.sin(q[5])])
            if self.elapsed>=self.t[-1]-.001 and fresh:
                self.phase='inspect'
                if self.exit_time is None:self.exit_time=t
                # Common, measured-attitude tool servo. Fixed-morphology methods
                # retain the same fixed shape here as in the preceding passage.
                if self.method not in ('straight','fixed'):self.des=g.geom_axis_bend(R,AXIS)
                g.place_estimate(np.zeros(3),R,estimate)
                self.ref=np.asarray(self.task['tool'])-g.tip()[0]
            elif hold in ('clearance','self') and fresh:
                if not self.holding:self.brake=np.asarray(xyz).copy()
                self.ref=self.brake.copy()
        self.holding=hold in ('clearance','self') and fresh
        if fresh and not hit and t>=12:
            e=self.des-estimate
            self.integral=limit_vector(self.integral+.02*self.s['ki']*e,np.deg2rad(self.s['i_limit_deg']))
            target=limit_vector(self.des+self.s['kp']*e+self.integral,np.deg2rad(self.s['command_max_deg']))
            if self.method=='straight':target=np.zeros(2)
            self.issued+=limit_vector(target-self.issued,.02*np.deg2rad(self.s['command_rate_deg_s']))
        return dict(t=t,phase=self.phase,hold=hold,valid=bool(fresh),des_b=self.des.copy(),est_b=estimate.copy(),
                issued_b=self.issued.copy(),px4_xyz=np.asarray(xyz if xyz is not None else [np.nan]*3),
                px4_rpy=np.asarray(rpy if rpy is not None else [np.nan]*3),ref_world=self.ref.copy(),reference_yaw=self.yaw,
                spline_t=self.elapsed,est_body=clear[0],est_arm=clear[1],predicted_body=ahead[0],predicted_arm=ahead[1],self_contact=hit)

class IndustryLoop(FlightLoop):
    def __init__(self,p,folder,case,decision,planfolder,score_end):
        self.plant=p;self.folder=folder;self.case=case;self.mechanics=False;self.horizon=score_end
        self.ctrl=Controller(p,decision,planfolder/'certified.npz')
        self.camera=self.gauge=self.filter=None;self.strain_pending=[];self.visual_pending=deque()
        self.rows=[];self.last_source=None;self.yaw_enu=0.
    def start(self,us):
        self.release=us;assets=FrozenAssets()
        self.gauge=LiveGauge(self.plant,str(self.case['seed']),'nominal',assets=assets,horizon_s=self.horizon+4.)
        self.filter=LiveEstimator(assets=assets)
        self.camera=IndustryOptics(self.plant,self.folder/'optics',condition='natural')
        self.strain_pending.extend(self.gauge.sample(0.,self.plant))
    def tick(self,*args):
        output=super().tick(*args)
        self.yaw_enu=self.ctrl.yaw
        return output
    def save(self):
        if self.camera:self.camera.close()
        for n in (self.gauge,self.filter):
            if n is not None:n.save(self.folder/'sensors')
        if self.rows:np.savez_compressed(self.folder/'control.npz',**{k:np.asarray([r[k] for r in self.rows]) for k in self.rows[0]})
        writej(self.folder/'control-summary.json',dict(task=self.ctrl.task,scene=self.plant.scene_digest,
               end_of_plan=self.ctrl.exit_time,checks=self.ctrl.checked,captures=len(self.rows),case=self.case))

def fly(out,planfolder,strategy,seed,instance,development):
    if (seed in SEEDS)==development:raise ValueError('separate pilot and registered seeds')
    lock=verify_frozen();p=build();decision=json.loads((planfolder/'decision.json').read_text())
    if decision['method']!=strategy or decision['map']!=p.scene_digest:raise ValueError('mismatched plan')
    if decision['accepted']:
        with np.load(planfolder/'certified.npz') as f:T=f['t'][-1]
        score_end=21.+T+11.
    else:score_end=21.
    c=dict(method='proposed',strategy=strategy,seed=seed,case='%s-%d'%(strategy,seed))
    loop=IndustryLoop(p,out,c,decision,planfolder,score_end)
    mission=dict(case=c,scene=p.scene_digest,plan_sha256=hashlib.sha256((planfolder/'decision.json').read_bytes()).hexdigest(),
                 freeze=lock,camera_delay_s=.2)
    try:
        run(out,c['case'],.4,reference_fn=initial_reference,score_end=score_end,noise_seed=seed,
            mission_record=mission,plant_factory=lambda:p,flight_loop=loop,instance=instance,
            protocol_name='m1_coupled_inspection.json',wall_limit_s=5400)
    finally:
        if out.is_dir():loop.save()
    if not (out/'flight.ulg').is_file():raise RuntimeError('PX4 ULog missing')
    print('logged',out,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('strategy',choices=METHODS);ap.add_argument('seed',type=int)
    ap.add_argument('instance',type=int);ap.add_argument('planfolder',type=Path);ap.add_argument('out',type=Path)
    ap.add_argument('--development',action='store_true');a=ap.parse_args()
    fly(a.out,a.planfolder,a.strategy,a.seed,a.instance,a.development)
