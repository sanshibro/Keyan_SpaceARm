"""Supplementary matched 2 x 2 split of the outward and return timing policies."""
import argparse
import json
import numpy as np
from round_f_sitl import run,digest
from round_g1_online_run import DEST,OLD,PROTO,P,Policy,Route,phase_time,make_plant
from gate_shape_timing import blend
from round_g1_model import ROOT

SPLIT=ROOT/'revision/protocols/round_g1_online_split.json'
SPEC=json.loads(SPLIT.read_text())


def adopt_phase_bend(planner,t,method):
    if method=='phasebend' and t>=14. and planner.bend_start is None:
        planner.bend_start=14.


def split_command(t,method,events):
    if t is None:return 0.
    tau=phase_time(t,'pause')
    up=blend((tau-14.)/10.) if method=='phasebend' else (0. if 'bend' not in events else blend((t-events['bend'])/10.))
    down=blend((tau-34.)/8.) if method=='onlinebend' else (0. if 'return' not in events else blend((t-events['return'])/8.))
    # A missing outward acceptance leaves the previous shape command in place.
    if method=='onlinebend' and 'bend' not in events:return 0.
    return float(75.*up-55.*down)


class CrossedPolicy(Policy):
    def command(self,us,t,ref,phase_angle,xyz,vel,rpy,pos_ms,att_ms,bend):
        adopt_phase_bend(self.planner,t,self.method)
        super().command(us,t,ref,phase_angle,xyz,vel,rpy,pos_ms,att_ms,bend)
        events={}
        if self.planner.bend_start is not None:events['bend']=self.planner.bend_start
        if self.planner.return_start is not None:events['return']=self.planner.return_start
        return split_command(t,self.method,events)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('seed',type=int);ap.add_argument('variant')
    a=ap.parse_args();assert a.variant in SPEC['order'][str(a.seed)]
    method,cond=a.variant.split('_');case='%d-%s'%(a.seed,a.variant);folder=DEST/'runs'/case
    route=Route(cond);policy=CrossedPolicy(folder,route,method)
    mission=dict(seed=a.seed,mode=a.variant,plan_sha256=digest(OLD/'plans.json'),parent_protocol_sha256=digest(PROTO),
        wrapper_sha256={p.name:digest(p) for p in [ROOT/'revision/scripts/round_g1_online_split.py',
                    ROOT/'revision/scripts/round_g1_online_run.py',ROOT/'mujoco_closed_loop/gate_shape_timing.py']})
    try:
        run(folder,case,.4,reference_fn=route.base,score_end=52.,noise_seed=a.seed,protocol_name=SPLIT.name,
            mission_record=mission,plant_factory=make_plant,shape_policy=policy)
    finally:policy.close()


if __name__=='__main__':main()
