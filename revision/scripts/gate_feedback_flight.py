"""One physical PX4/MuJoCo flight with causal wrist, strain and position feedback."""
import argparse
from collections import deque
from pathlib import Path
import time
import numpy as np
from gate_shape_timing import enu_body_rotation
from gate_feedback_model import (P, SPECFILE, ROOT, OUT, LAUNCH, ObserverGeometry,
                                 cart, digest, dump, limit_vector, make_plant,
                                 truth_bend, verify_frozen)
from round_f_plant import CLOCK_ORIGIN_US
from round_f_sitl import run
from gate_live_shape import FrozenAssets, LiveEstimator, LiveGauge
from gate_live_optics import LiveOptics

DT = .020
HOVER = np.array([0., 0., P['pose']['hover_z']])
STAGE = HOVER + [P['pose']['staging_x'], 0., 0.]
EXIT = HOVER + [P['pose']['exit_x'], 0., 0.]
TARGET = np.array(P['target'])

def ramp(value):
    u = float(np.clip(value, 0., 1.))
    return u**3 * (10. - 15.*u + 6.*u*u)

def interpolate_angles(a, b, fraction):
    return cart(a[0]+fraction*(b[0]-a[0]), a[1]+fraction*((b[1]-a[1]+180.)%360.-180.))

def initial_reference(t):
    return 'takeoff', HOVER-LAUNCH, 0.

class GateControl:
    """Uses only a declared shape observation, measured PX4 state and the surveyed gate."""
    def __init__(self, geometry):
        self.g = geometry
        self.axis = geometry.world_axis_for(cart(*P['final_deg']))
        self.integral = np.zeros(2)
        self.issued = np.zeros(2)
        self.ref = HOVER.copy()
        self.des = np.zeros(2)
        self.shaping = self.advance = self.docking = 0.
        self.exit_time = None
        self.exit_shape = None
        self.last_valid_b = np.zeros(2)
        self.phase = 'takeoff'
        self.dock_origin = None
        self.previous_hold = 'none'
        self.checked = 0
        self.servo = P['servo']

    def control(self, t, obs, xyz, velocity, rpy, age_p, age_r):
        g = self.g
        pose_fresh = (xyz is not None and velocity is not None and rpy is not None
                      and -.025 <= age_p <= .25 and -.025 <= age_r <= .25)
        shape_fresh = obs['valid'] and obs['age'] <= .30
        fresh = bool(pose_fresh and shape_fresh)
        hold = 'none'
        body_gap = arm_gap = rear = np.nan
        forecast_body = forecast_arm = np.nan
        estimate = np.asarray(obs['b'], dtype=float)
        if obs['valid']:
            self.last_valid_b = estimate.copy()
        if fresh:
            R = enu_body_rotation(rpy)
            body_gap, arm_gap, rear = g.info(xyz, R, estimate)
            # Forward swept translation out to a braking horizon. The current
            # and predicted positions bracket the short, monotone approach.
            speed=max(0.,velocity[0])
            braking_distance=.035+.12*speed+speed*speed/(2.*.75)
            advance = np.array([max(braking_distance,self.ref[0]-xyz[0]), 0., 0.])
            mid = g.info(xyz, R, estimate, advance/2)
            ahead = g.info(xyz, R, estimate, advance)
            forecast_body = min(body_gap,mid[0],ahead[0])
            forecast_arm = min(arm_gap,mid[1],ahead[1])
            self.checked += 1

        if t < 27.6:
            self.phase = 'takeoff' if t < 12 else 'neutral_anchor' if t < 14 else 'prebend'
            fraction = ramp((t-14.)/7.)
            self.des = cart(P['pre_deg'][0]*fraction, P['pre_deg'][1])
            if fresh:
                self.ref = HOVER+(STAGE-HOVER)*ramp((t-15.)/8.)
        elif self.exit_time is None:
            self.phase = 'cross'
            if not fresh:
                hold = 'feedback'
            else:
                self.shaping += DT
                self.des = interpolate_angles(P['pre_deg'], P['gate_deg'], ramp(self.shaping/4.2))
                if min(forecast_body,forecast_arm) < P['thresholds']['external_m']:
                    hold = 'clearance'
                else:
                    self.advance += DT
                    self.ref = STAGE+(EXIT-STAGE)*ramp(self.advance/10.)
                if rear >= .02 and xyz[0] > P['gate']['center_x']:
                    self.exit_time = t
                    self.dock_origin = self.ref.copy()
                    self.exit_shape = self.des.copy()
        else:
            self.phase = 'dock'
            if not fresh:
                hold = 'feedback'
            elif min(body_gap,arm_gap) < P['thresholds']['external_m']:
                hold = 'clearance'
            else:
                self.docking += DT
                fraction = ramp(self.docking/5.0)
                final_shape = g.geom_axis_bend(R,self.axis)
                self.des = (1.-fraction)*self.exit_shape+fraction*final_shape
                g.place_estimate(np.zeros(3), R, estimate)
                tool_offset = g.tip()[0]
                self.ref = (1.-fraction)*self.dock_origin + fraction*(TARGET-tool_offset)
        if hold!='none' and self.previous_hold=='none' and pose_fresh:
            # Brake from the measured aircraft position when its flying reference leads it.
            self.ref[0]=xyz[0]
        self.previous_hold=hold
        if fresh:
            error = self.des-estimate
            s=self.servo
            self.integral = limit_vector(self.integral+DT*s['ki']*error,np.deg2rad(s['i_limit_deg']))
            req = limit_vector(self.des+s['kp']*error+self.integral,np.deg2rad(s['command_max_deg']))
            self.issued += limit_vector(req-self.issued,DT*np.deg2rad(s['command_rate_deg_s']))
        return dict(t=t,phase=self.phase,hold=hold,valid=fresh,shape_valid=bool(obs['valid']),
                    des_b=self.des.copy(),est_b=estimate.copy(),issued_b=self.issued.copy(),
                    integral_b=self.integral.copy(),px4_xyz=np.asarray(xyz if xyz is not None else [np.nan]*3),
                    px4_rpy=np.asarray(rpy if rpy is not None else [np.nan]*3),ref_world=self.ref.copy(),
                    age_p=age_p,age_r=age_r,shape_age=obs['age'],est_body=body_gap,est_arm=arm_gap,
                    predicted_body=forecast_body,predicted_arm=forecast_arm,est_rear=rear,
                    gate_exit=self.exit_time is not None)

class FlightLoop:
    def __init__(self, plant, folder, case, mechanics):
        self.plant,self.folder,self.case,self.mechanics=plant,folder,case,mechanics
        self.ctrl=GateControl(ObserverGeometry(plant))
        self.camera=self.gauge=self.filter=None
        self.strain_pending=[]
        self.visual_pending=deque()
        self.rows=[]
        self.last_source=None

    def start(self, us):
        self.release=us
        if not self.mechanics:
            assets=FrozenAssets()
            self.gauge=LiveGauge(self.plant,str(self.case['seed']),self.case['material'],assets=assets,
                                 horizon_s=P['release_end']+3.,drift_window=(18.,22.))
            self.filter=LiveEstimator(assets=assets)
            self.camera=LiveOptics(self.plant,self.folder/'optics',condition=self.case['vision'])
            self.strain_pending.extend(self.gauge.sample(0.,self.plant))

    def tick(self,t,us,xyz,velocity,rpy,pos_ms,att_ms):
        k=round(t/.004)
        if self.gauge:
            self.strain_pending.extend(self.gauge.sample(t,self.plant))
            if t>=P['camera']['start_s']-1e-10 and k%10==0:
                if rpy is None:
                    raise RuntimeError('No PX4 attitude at wrist exposure')
                snap=self.camera.capture(t,self.plant.d.qpos,rpy,P['window']['blackout_start'])
                if snap is not None:
                    self.visual_pending.append(snap)
        if k%5==0:
            if self.filter:
                arrived=[]
                while self.visual_pending and self.visual_pending[0]['arrival_t']<=t+1e-9:
                    arrived.append(self.visual_pending.popleft())
                self.last_source=self.filter.step(t,self.strain_pending,arrived)
                self.strain_pending=[]
            method=self.case['method']
            if method=='oracle':
                source=dict(b=truth_bend(self.plant),age=0.,valid=True)
            else:
                source=self.last_source[method]
            stamp=(us-CLOCK_ORIGIN_US)/1e6
            row=self.ctrl.control(t,source,xyz,velocity,rpy,stamp-pos_ms/1000.,stamp-att_ms/1000.)
            self.plant.bend_command=self.ctrl.issued.copy()
            row['vision_hold_b']=(self.last_source['vision_hold']['b'].copy() if self.last_source
                                  else np.full(2,np.nan))
            row['optical_age']=(self.last_source['proposed']['vision_age'] if self.last_source else np.nan)
            self.rows.append(row)
        return self.ctrl.phase,self.ctrl.ref-LAUNCH,float(np.rad2deg(np.linalg.norm(self.ctrl.des)))

    def save(self):
        if self.camera is not None:
            self.camera.close()
        for node in (self.gauge,self.filter):
            if node is not None:node.save(self.folder/'sensors')
        if self.rows:
            np.savez_compressed(self.folder/'control.npz',
                                **{k:np.asarray([r[k] for r in self.rows]) for k in self.rows[0]})
        dump(self.folder/'control-summary.json',dict(goal_xyz=TARGET.tolist(),goal_axis=self.ctrl.axis.tolist(),
             exit_time=self.ctrl.exit_time,geometry_checks=self.ctrl.checked,mechanics_only=self.mechanics,
             case=self.case,n_queries=len(self.rows)))

    def bag_sample(self, bag, us):
        if not self.rows:return
        from genpy import Time
        from geometry_msgs.msg import Vector3Stamped
        from std_msgs.msg import String
        stamp=Time(us//1000000,(us%1000000)*1000)
        row=self.rows[-1]
        for field in ('des_b','est_b','issued_b'):
            message=Vector3Stamped()
            message.header.stamp=stamp
            message.header.frame_id='arm_mount'
            message.vector.x,message.vector.y=row[field]
            bag.write('/gate/shape/'+field,message,stamp)
        bag.write('/gate/phase',String(data=row['phase']),stamp)
        bag.write('/gate/hold_reason',String(data=row['hold']),stamp)

def fly(args):
    freeze_hash=verify_frozen()
    if args.formal:
        if args.mechanics or args.case is None:raise ValueError('A formal case requires the full sensor chain')
        c=next(x for x in P['cases'] if x['case']==args.case)
        folder=OUT/'formal'/c['case']
    else:
        if args.seed in P['seeds']:raise ValueError('Development must use separate seeds')
        c=dict(method=args.method,material=args.material,vision=args.view,seed=args.seed,case=args.name)
        if not c['case']:raise ValueError('Supply a fresh development run name')
        folder=OUT/'development'/c['case']
    if args.mechanics and c['method']!='oracle':raise ValueError('Mechanics pilot requires oracle')
    if folder.exists():raise FileExistsError(folder)
    plant=make_plant(canvas=not args.mechanics)
    loop=FlightLoop(plant,folder,c,args.mechanics)
    manifest=dict(case=c,perception_lock_sha256=freeze_hash,mechanics_only=args.mechanics,
                  controller_sha256=digest(__file__),model_builder_sha256=digest(Path(__file__).with_name('gate_feedback_model.py')),
                  observed_axis_world=loop.ctrl.axis.tolist(),payload_kg=P['payload_kg'])
    try:
        run(folder,c['case'],.4,reference_fn=initial_reference,score_end=P['release_end'],noise_seed=c['seed'],
            protocol_name=SPECFILE.name,mission_record=manifest,plant_factory=lambda:plant,
            flight_loop=loop,instance=args.instance,wall_limit_s=1800)
    finally:
        if folder.is_dir():loop.save()
    if not (folder/'flight.ulg').is_file():raise RuntimeError('Missing PX4 ULog in '+str(folder))
    print('completed flight',folder,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--formal',action='store_true');p.add_argument('--case');p.add_argument('--name')
    p.add_argument('--mechanics',action='store_true');p.add_argument('--instance',type=int,default=6)
    p.add_argument('--method',choices=P['methods'],default='proposed')
    p.add_argument('--material',choices=P['materials'],default='differential')
    p.add_argument('--view',choices=P['views'],default='occluded');p.add_argument('--seed',type=int,default=26112090)
    fly(p.parse_args())
