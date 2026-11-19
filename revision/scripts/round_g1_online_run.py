"""Couple the online single-gate scheduler to PX4 estimates and native simulated joint feedback."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from functools import partial
import numpy as np
from gate_shape_timing import GateEnvelope, RollingGateTiming
from round_f_plant import CLOCK_ORIGIN_US, LAUNCH
from round_f_sitl import run,digest
from round_g1_model import ROOT, Geometry, make_plant
from round_g1_plan import reference

OLD=ROOT/'revision/data/round_g1'
PROTO=ROOT/'revision/protocols/round_g1_online.json'
P=json.loads(PROTO.read_text())
DEST=Path(os.environ.get('ROUND_G1O_OUTPUT',str(ROOT/'revision/data/round_g1_online'))).expanduser().resolve()
PLAN=json.loads((OLD/'plans.json').read_text())['selected']['variable_timed']


def phase_time(t,condition):
    if t is None or condition=='nominal':return t
    cfg=P['pause'];start=cfg['deceleration_start_s'];r=cfg['ramp_s'];hold=cfg['stationary_s']
    if t<=start:return t
    # Integrate a quintic velocity blend, preserving C2 continuity of the base reference.
    def integrated(u):return 2.5*u**4-3.*u**5+u**6
    x=t-start
    if x<r:return start+x-r*integrated(x/r)
    if x<=r+hold:return start+.5*r
    if x<2*r+hold:return start+.5*r+r*integrated((x-r-hold)/r)
    return t-r-hold


class Route:
    def __init__(self,condition):
        self.condition=condition
        self.grid=np.arange(0.,72.001,.025)
        self.path=np.vstack([reference(t,PLAN,'variable_timed')[1]+LAUNCH for t in self.grid])
    def base(self,t):
        tau=phase_time(t,self.condition)
        return reference(tau,PLAN,'variable_timed')
    def preview(self,t,dt):
        tau=phase_time(t,self.condition)
        return np.column_stack([np.interp(tau+dt,self.grid,self.path[:,k]) for k in range(3)])


class Policy:
    def __init__(self,folder,route,method):
        self.folder=folder;self.route=route;self.method=method
        self.planner=RollingGateTiming(GateEnvelope(Geometry(make_plant()),P['controller']['angle_pad_per_degree_m']),P)
        self.next_us=0;self.log=None
    def command(self,us,t,ref,phase_angle,xyz,vel,rpy,pos_ms,att_ms,bend):
        if self.log is None:self.log=(self.folder/'timing.jsonl').open('x',buffering=1)
        if us>=self.next_us:
            self.next_us=us+int(round(1e6*P['controller']['update_s']))
            age_pos=(us-CLOCK_ORIGIN_US)/1e6-pos_ms/1000.
            age_att=(us-CLOCK_ORIGIN_US)/1e6-att_ms/1000.
            obs=None if xyz is None or vel is None or rpy is None else (np.asarray(xyz),np.asarray(vel),np.asarray(rpy),bend,age_pos,age_att)
            preview=self.route.preview(t,self.planner.times)
            before=time.perf_counter()
            dec=self.planner.update(t,obs,preview)
            wall=1000*(time.perf_counter()-before)
            row=dict(us=us,t=t,phase_s=phase_time(t,self.route.condition),method=self.method,
                position=None if xyz is None else list(xyz),velocity=None if vel is None else list(vel),
                rpy_frd_ned=rpy,pos_ms=pos_ms,att_ms=att_ms,age_pos_s=age_pos,age_att_s=age_att,
                measured_deg=bend,compute_ms=wall,
                bend_start=self.planner.bend_start,return_start=self.planner.return_start,**dec)
            self.log.write(json.dumps(row,allow_nan=False)+'\n')
        if self.method=='online':return self.planner.sample(t)
        if self.method=='clock':return reference(t,PLAN,'variable_timed')[2]
        return phase_angle
    def close(self):
        if self.log:self.log.close()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('seed',type=int);ap.add_argument('variant')
    args=ap.parse_args();method,cond=args.variant.split('_')
    if args.variant not in P['seed_order'][str(args.seed)]:raise ValueError('Unregistered flight')
    case='%d-%s'%(args.seed,args.variant);folder=DEST/'runs'/case
    route=Route(cond);policy=Policy(folder,route,method)
    mission=dict(seed=args.seed,mode=args.variant,plan_sha256=digest(OLD/'plans.json'),
        parent_protocol_sha256=digest(ROOT/'revision/protocols/round_g1.json'),
        input='PX4 LOCAL_POSITION_NED and ATTITUDE plus sampled simulated net arm bend',
        reference='identical per condition',estimator_only_for_body=True,
        wrapper_sha256={f.name:digest(f) for f in [Path(__file__),ROOT/'mujoco_closed_loop/gate_shape_timing.py',Path(__file__).with_name('round_g1_plan.py')]})
    try:
        run(folder,case,.4,reference_fn=route.base,score_end=52.,noise_seed=args.seed,protocol_name=PROTO.name,
            mission_record=mission,plant_factory=make_plant,shape_policy=policy)
    finally:policy.close()


if __name__=='__main__':main()
