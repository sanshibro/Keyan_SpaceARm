"""Read-only adapter audit on sensor map and historical frozen-map regression."""
import argparse,json,time
from pathlib import Path
import numpy as np
from planner_adapter import V21Adapter,ROOT
from uav_continuum_whole_body_v2.mapping import MapSnapshot

def main():
    p=argparse.ArgumentParser();p.add_argument('--map',type=Path,required=True);p.add_argument('--historical',action='store_true');args=p.parse_args()
    out=ROOT/'runs'/(time.strftime('%Y%m%d_%H%M%S')+'_planner_audit');out.mkdir(parents=True)
    snapshot=MapSnapshot.load(args.map);adapter=V21Adapter(snapshot,'v2' if args.historical else 'qingyu')
    state=np.array([0,0,1.3,-np.pi/2,0,0]) if args.historical else np.array([0,0,1.5,0,0,0])
    goal=np.array([0,-1.2,1.3]) if args.historical else np.array([.1,0,1.5])
    trajectory,diagnostics=adapter.plan(state,goal)
    if trajectory is not None:
        np.savez_compressed(out/'candidate.npz',times=trajectory.times,states=trajectory.states,velocities=trajectory.velocities,accelerations=trajectory.accelerations,jerks=trajectory.jerks)
    manifest=dict(stage='V21_map_planner_adapter_only',full_task_accepted=False,map_source=str(args.map),mode='historical_frozen_map_regression_not_online_MuJoCo' if args.historical else 'MuJoCo_sensor_map',candidate_accepted=trajectory is not None,diagnostics=diagnostics)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else str(v)));print(out);print(json.dumps(manifest,default=str))
if __name__=='__main__':main()
