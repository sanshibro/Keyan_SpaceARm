"""Qingyu SDK three-finger CAD, with explicitly approximate slide actuation."""
from xml.etree.ElementTree import SubElement as S
from pathlib import Path
import math

def add_gripper(asset,tool,acts,exclusions):
    mesh_dir=Path(__file__).resolve().parent/'vendor/qingyu_continuum_arm/meshes'
    for name in ['housing','finger0','finger1','finger2']:
        S(asset,'mesh',name='gripper_'+name,file=str(mesh_dir/('gripper_'+name+'_lod.obj')))
    S(tool,'geom',type='mesh',mesh='gripper_housing',density='0',contype='0',conaffinity='0',group='2',rgba='.2 .2 .2 1')
    for i,angle in enumerate([150,30,270]):
        a=math.radians(angle);x,y=math.cos(a),math.sin(a)
        finger=S(tool,'body',name=f'finger{i}')
        S(finger,'joint',name=f'grip{i}',type='slide',axis=f'{-x} {-y} 0',range='0 .024',damping='1')
        S(finger,'geom',type='mesh',mesh=f'gripper_finger{i}',density='0',contype='0',conaffinity='0',group='2',rgba='.2 .2 .2 1')
        S(finger,'geom',name=f'pad{i}',type='box',pos=f'{.033*x} {.033*y} .112',size='.005 .005 .018',euler=f'0 0 {a}',mass=str(.05/3),rgba='.8 .5 .1 .4')
        acts.append((f'grip{i}',800,5));exclusions.append(('tool',f'finger{i}'))
