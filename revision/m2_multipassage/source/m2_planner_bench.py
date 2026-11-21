"""Static M2 planner timing and causal-map checks. Never starts a flight."""
import json,time,traceback
from pathlib import Path
import numpy as np
from m2_plant import build,task_pose
from m2_geometry import Geometry,RangeMap
from m2_planner import Kernel
from m2_scene import HOVER

def main():
    p=build();sensor=RangeMap(p);start=np.r_[HOVER,0.,0.,np.pi];snap=sensor.reveal(21,start)
    print('snapshot',snap.map_hash,len(snap.ids),snap.origin,snap.radius,flush=True)
    g=Geometry(p,snapshot=snap)
    for theta,phi in ((0,180),(70,160),(75,90),(80,180)):
        ts=time.monotonic();r=g.self_report(np.deg2rad(theta),np.deg2rad(phi))
        print('self',theta,phi,r,'wall',time.monotonic()-ts,flush=True)
    kernel=Kernel(p,'coupled',task_pose(p))
    try:
        result=kernel.solve(snap,start,np.zeros(6),np.zeros(6));print('SOLVED',result['timing'],result['proof'],flush=True)
    except Exception as exc:
        print('REJECT',type(exc).__name__,str(exc),getattr(exc,'timing',None),'LAYERS',getattr(exc,'layers',None),flush=True)

if __name__=='__main__':main()
