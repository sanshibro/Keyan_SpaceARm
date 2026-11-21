"""Re-evaluate recorded M2 window requests on their original finite snapshots."""
import argparse
import cProfile
import json
from pathlib import Path
import pstats
import time

import numpy as np

from m2_geometry import MapSnapshot
from m2_handoff import qualify_timed
from m2_planner import Kernel
from m2_plant import build
from m2_rolling import clean
from m2_spline import VectorSpline


def read(file):return json.loads(file.read_text(encoding='utf8'))

def recheck(flight,out,profile_id):
    p=build();task=read(flight/'control-summary.json')['task']
    k=Kernel(p,'coupled',task);root=flight/'rolling/windows';results=[]
    for folder in sorted(root.iterdir()):
        req=read(folder/'request.json');i=req['id'];mapdata=req['map']
        with np.load(folder/'snapshot.npz') as data:
            snap=MapSnapshot(mapdata['version'],mapdata['stamp'],tuple(data['origin']),float(data['radius']),
                data['obstacle_rows'],tuple(data['ids'].tolist()),mapdata['known_frontier'],mapdata['map_hash'])
        prev=req['previous_id']
        if prev>=0:
            with np.load(root/('%03d'%prev)/'trajectory.npz') as s:
                k.last_curve=VectorSpline(s['control_states'],s['span_durations'],stationary_start=bool(s['stationary_start']),
                         stationary_end=bool(s['stationary_end']),boundary_ramp_duration=float(s['boundary_ramp_duration']))
                k.last_opt=np.array(read(root/('%03d'%prev)/'timing.json')['optimizer']['best'])
        seed=[np.array(req['seed_boundary'][v]) for v in ('q','dq','ddq')]
        old=[np.array(req['committed_boundary'][v]) for v in ('q','dq','ddq')]
        profiler=cProfile.Profile() if i==profile_id else None
        if profiler:profiler.enable()
        started=time.monotonic()
        try:
            result=k.solve(snap,*seed,i);curve=result['curve']
            lag=max(0.,req['scheduled_switch_s']-req['submit_release_s']) if prev>=0 else 0.
            b,proof,handoff,o=qualify_timed(k,old,curve,lag,req['bridge_duration'])
            row=dict(id=i,accepted=proof.feasible,timing=result['timing'],bridge_s=handoff['duration'],
                     receiving_phase_s=o,bridge_peak=handoff.get('dynamics_peak_ratio'),elapsed_s=time.monotonic()-started)
        except Exception as e:
            row=dict(id=i,accepted=False,reason=str(e),timing=getattr(e,'timing',None),elapsed_s=time.monotonic()-started)
        if profiler:
            profiler.disable()
            with (out.parent/(out.stem+'-profile.txt')).open('w') as writer:
                pstats.Stats(profiler,stream=writer).strip_dirs().sort_stats('tottime').print_stats(32)
        results.append(row)
        print(i,row['accepted'],row.get('reason'),round(row['elapsed_s'],3),row.get('bridge_s'),flush=True)
    out.write_text(json.dumps(clean(results),indent=2)+'\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('flight',type=Path);ap.add_argument('out',type=Path)
    ap.add_argument('--profile-id',type=int,default=-1);args=ap.parse_args()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    recheck(args.flight,args.out,args.profile_id)
