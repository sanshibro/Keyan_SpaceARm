"""Preflight qualification of the scenario, never an input to online planning.

The six-dimensional V2.1 envelope checks prove that a straight arm cannot
cross these openings with tracking reserve and that a bent, altitude-adjusted
state can. MuJoCo contacts independently check the latter physical pose.
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path
from xml.etree.ElementTree import tostring

import mujoco
import numpy as np

from mission_scene import MissionScene
from model import SimArmBridge
from planner_adapter import (LIB, ARC_LENGTH, BASE_LENGTH, ArmModel,
                             collision_samples, load_planner_configuration,
                             world_arm_points, world_body_points)
from stage1_scene import build_flight_base


def point_box_distance(points,center,half_size):
    q=np.abs(points-np.asarray(center))-np.asarray(half_size)
    return np.linalg.norm(np.maximum(q,0.),axis=1)+np.minimum(np.max(q,axis=1),0.)


def aperture_boxes(door):
    y,(x0,x1),(z0,z1)=door
    return [((0,y,z0/2),(1.57,.03,z0/2)),
            ((0,y,(z1+2.6)/2),(1.57,.03,(2.6-z1)/2)),
            (((-1.57+x0)/2,y,(z0+z1)/2),((x0+1.57)/2,.03,(z1-z0)/2)),
            (((x1+1.57)/2,y,(z0+z1)/2),((1.57-x1)/2,.03,(z1-z0)/2))]


def qualify():
    scene=MissionScene.three_gaps()
    _,_,limits,_=load_planner_configuration(LIB/'config/planner_v2_realtime.yaml')
    limits=replace(limits,body_height=max(limits.body_height,.42),
                   body_margin=.01+.03,arm_margin=.01+.03)
    arm=ArmModel(arc_length=ARC_LENGTH,base_offset_z=BASE_LENGTH,
                 extra_tool_points=collision_samples(approved_stage1=True))
    root=build_flight_base();scene.add(root)
    model=mujoco.MjModel.from_xml_string(tostring(root,encoding='unicode'))
    data=mujoco.MjData(model);bridge=SimArmBridge()
    results=[]
    for index,door in enumerate(scene.doors()):
        y,(x0,x1),(z0,z1)=door;x=(x0+x1)/2
        boxes=aperture_boxes(door)

        def margin(state):
            body=world_body_points(state,limits)
            tool=world_arm_points(state,arm)
            body_clearance=min(np.min(point_box_distance(body,*box)) for box in boxes)
            arm_clearance=min(np.min(point_box_distance(tool,*box)) for box in boxes)-arm.radius
            return min(body_clearance-limits.body_margin,arm_clearance-limits.arm_margin)

        longitudinal=np.linspace(y-.35,y+.35,29)
        def sweep(z,theta,phi):
            return min(float(margin(np.array([x,yy,z,0.,theta,phi]))) for yy in longitudinal)

        straight_best=max((sweep(float(z),0.,0.),float(z))
                          for z in np.arange(1.20,1.751,.025))
        bent_height=z0+.40
        bent_theta=np.deg2rad(110.)
        bent_margin=sweep(bent_height,bent_theta,np.pi/2)
        bridge.set_kinematic_shape(model,data,bent_theta,np.pi/2)
        wall_ids={model.geom(f'door_{part}_{index}').id
                  for part in ('lower','upper','left','right')}
        contact_count=0
        for yy in longitudinal:
            data.qpos[:3]=[x,yy,bent_height]
            mujoco.mj_forward(model,data)
            contact_count+=sum(c.dist<-.001 and
                               (c.geom1 in wall_ids or c.geom2 in wall_ids)
                               for c in data.contact)
        bridge.set_kinematic_shape(model,data,0.,0.)
        straight_physical_safe=[]
        for z in np.arange(1.15,1.801,.01):
            collision=False
            for yy in longitudinal:
                data.qpos[:3]=[x,yy,z]
                mujoco.mj_forward(model,data)
                if any(c.dist<-.001 and
                       (c.geom1 in wall_ids or c.geom2 in wall_ids)
                       for c in data.contact):
                    collision=True;break
            if not collision:straight_physical_safe.append(round(float(z),3))
        results.append(dict(gap=index+1,opening_width_m=x1-x0,
                            opening_height_m=z1-z0,straight_best_residual_m=straight_best[0],
                            straight_best_height_m=straight_best[1],
                            straight_physical_safe_heights_m=straight_physical_safe,
                            bent_height_m=bent_height,bent_theta_deg=110.,bent_phi_deg=90.,
                            bent_min_residual_m=bent_margin,physical_contact_count=contact_count))
    report=dict(method='preflight scene geometry; V2.1 body/arm envelopes and margins plus MuJoCo contact sweep',
                runtime_planner_uses_scene_truth=False,results=results)
    report['passed']=all(r['straight_best_residual_m']<0 and
                         not r['straight_physical_safe_heights_m'] and
                         r['bent_min_residual_m']>.025 and
                         r['physical_contact_count']==0 for r in results)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path)
    args=parser.parse_args();report=qualify()
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if not report['passed']:raise SystemExit('Aperture qualification failed')
