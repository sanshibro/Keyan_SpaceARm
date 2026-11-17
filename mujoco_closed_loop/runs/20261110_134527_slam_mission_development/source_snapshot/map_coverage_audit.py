"""Locate unknown required envelope points without changing the map."""
import argparse,json,time
from pathlib import Path
import numpy as np
from planner_adapter import V21Adapter,ROOT
from uav_continuum_whole_body_v2.mapping import MapSnapshot
from uav_continuum_whole_body_v2.geometry import world_arm_points,world_body_points
p=argparse.ArgumentParser();p.add_argument('map',type=Path);args=p.parse_args()
snapshot=MapSnapshot.load(args.map);adapter=V21Adapter(snapshot);state=np.array([0,0,1.5,0,0,0]);origin=np.array([0,0,1.6]);report={}
for name,points in [('body',world_body_points(state,adapter.limits)),('arm',world_arm_points(state,adapter.arm))]:
 known=adapter.world._known_mask(points);unknown=points[~known];delta=unknown-origin
 elevation=np.rad2deg(np.arctan2(delta[:,2],np.linalg.norm(delta[:,:2],axis=1)))
 report[name]={'samples':len(points),'unknown_samples':len(unknown),'unknown_points_world':unknown.tolist(),'unknown_min_elevation_deg':float(elevation.min()) if len(elevation) else None,'unknown_max_elevation_deg':float(elevation.max()) if len(elevation) else None,'below_lidar_minus30deg':int((elevation<-30).sum())}
out=ROOT/'runs'/(time.strftime('%Y%m%d_%H%M%S')+'_map_coverage');out.mkdir(parents=True)
(out/'report.json').write_text(json.dumps(report,indent=2));print(out);print(json.dumps({k:{a:b for a,b in v.items() if a!='unknown_points_world'} for k,v in report.items()}))
