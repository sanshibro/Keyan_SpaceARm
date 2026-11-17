"""Controlled textured room for development; textures are not landmark truth inputs."""
from xml.etree.ElementTree import SubElement as S
import numpy as np

def textured_room(root,seed=8,back_wall=False,long_room=False):
    u=root.find("worldbody/body[@name='uav']")
    for g in u.iter('geom'):
        if g.get('group') not in ('2','3','4'):g.set('group','1')
    w=root.find('worldbody');rng=np.random.default_rng(seed)
    length=3.4 if long_room else 2.
    for side in [-1,1]:
        S(w,'geom',type='box',pos=f'{side*1.6} 0 1.3',size=f'.03 {length} 1.3',rgba='.6 .6 .6 1')
        for _ in range(200):
            y,z=rng.uniform(-length+.1,length-.1),rng.uniform(.1,2.5)
            shade=rng.uniform(.12,.78)
            c=(shade*.70,shade*.85,shade) if rng.random()<.13 else (shade,shade*1.02,shade*1.04)
            S(w,'geom',type='box',pos=f'{side*1.565} {y} {z}',size=f'.002 {rng.uniform(.015,.055)} {rng.uniform(.015,.055)}',contype='0',conaffinity='0',rgba=' '.join(map(str,[*c,1])))

    # Controlled floor texture lets the arm-mounted camera track during downward survey.
    for _ in range(160):
        x,y=rng.uniform(-1.45,1.45),rng.uniform(-length+.1,length-.1)
        shade=rng.uniform(.08,.8)
        S(w,'geom',type='box',pos=f'{x} {y} .001',size=f'{rng.uniform(.015,.05)} {rng.uniform(.015,.05)} .001',contype='0',conaffinity='0',rgba=f'{shade} {shade} {shade} 1')
    if back_wall:
        far_y=3.2 if long_room else 1.9
        S(w,'geom',name='far_wall',type='box',pos=f'0 {far_y} 1.3',size='1.6 .03 1.3',rgba='.48 .53 .57 1')
        for x in (-1.05,-.35,.35,1.05):
            S(w,'geom',name=f'far_wall_seam_{x}',type='box',pos=f'{x} {far_y-.035} 1.3',size='.003 .002 1.3',
              contype='0',conaffinity='0',rgba='.34 .40 .44 1')
        for _ in range(80):
            x,z=rng.uniform(-1.5,1.5),rng.uniform(.12,2.48);shade=rng.uniform(.20,.80)
            S(w,'geom',type='box',pos=f'{x} {far_y-.035} {z}',size=f'{rng.uniform(.012,.038)} .002 {rng.uniform(.012,.038)}',
              contype='0',conaffinity='0',rgba=f'{shade*.84} {shade*.96} {shade} 1')
        for name,x,z,sx,sz in [('left',-.45,1.,.008,.50),('right',.45,1.,.008,.50),
                              ('top',0,1.50,.45,.008),('bottom',0,.50,.45,.008)]:
            S(w,'geom',name='target_zone_'+name,type='box',pos=f'{x} {far_y-.042} {z}',size=f'{sx} .003 {sz}',
              contype='0',conaffinity='0',rgba='.11 .65 .69 1')
