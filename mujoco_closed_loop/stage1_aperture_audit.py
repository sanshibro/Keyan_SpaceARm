"""Check the three stage-1 openings against the robot's collision geometry.

The preview welds the aircraft to the world, so MuJoCo omits contacts between
some fixed bodies. Signed geom distance still reports geometric penetration.
"""
import argparse
import json
import numpy as np
import mujoco

from model import SimArmBridge
from stage1_scene import PASSAGES, build_scene


def audit():
    model=mujoco.MjModel.from_xml_string(build_scene())
    data=mujoco.MjData(model)
    bridge=SimArmBridge()
    uav=model.body('uav').id
    robot=[i for i in range(model.ngeom)
           if model.geom_bodyid[i]>=uav
           and model.geom_contype[i] and model.geom_conaffinity[i]]
    rows=[]

    def margin(walls):
        return min(mujoco.mj_geomDistance(model,data,g,r,2,None)
                   for g in walls for r in robot)

    for name,y,xlo,xhi,zlo,zhi in PASSAGES:
        walls=[model.geom(name+'_'+part).id
               for part in ('left','right','sill','lintel')]
        center=(xlo+xhi)/2
        bridge.set_kinematic_shape(model,data,0,0)
        straight_best=(-float('inf'),None,None)
        for x in np.linspace(center-.05,center+.05,5):
            for z in np.arange(1.15,1.751,.005):
                model.body_pos[uav]=[x,y,z]
                mujoco.mj_forward(model,data)
                d=margin(walls)
                if d>straight_best[0]:
                    straight_best=(d,float(x),float(z))

        bridge.set_kinematic_shape(model,data,np.pi/2,np.pi)
        bent_best=(-float('inf'),None)
        for z in np.arange(1.24,1.371,.005):
            sweep_min=float('inf')
            for yy in np.linspace(y-.35,y+.35,71):
                model.body_pos[uav]=[center,yy,z]
                mujoco.mj_forward(model,data)
                sweep_min=min(sweep_min,margin(walls))
            if sweep_min>bent_best[0]:
                bent_best=(sweep_min,float(z))

        rows.append({'gate':name,'width_m':round(xhi-xlo,3),
                     'height_m':round(zhi-zlo,3),
                     'straight_best_margin_m':round(straight_best[0],4),
                     'bent_90deg_sweep_margin_m':round(bent_best[0],4),
                     'bent_uav_height_m':round(bent_best[1],3)})
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',help='Optional JSON report path')
    args=parser.parse_args()
    rows=audit()
    report={'method':'signed MuJoCo geom distance; gate center plus lateral samples for straight arm; ±0.35 m longitudinal sweep for 90° bent arm',
            'results':rows}
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    if any(row['straight_best_margin_m']>=0 or row['bent_90deg_sweep_margin_m']<=.01
           for row in rows):
        raise SystemExit('Aperture constraint failed')
