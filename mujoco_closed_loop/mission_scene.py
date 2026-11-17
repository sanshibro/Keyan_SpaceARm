"""Frozen development scene: one real portal, a supported target, return region."""
from dataclasses import dataclass,asdict
from xml.etree.ElementTree import SubElement as S
import math

@dataclass(frozen=True)
class MissionScene:
    start: tuple=(-.6,-1.2,1.5)
    pregrasp: tuple=(-.6,0.,1.5)
    exit: tuple=(-.6,-1.2,1.55)
    door_y: float=-.6
    opening_x: tuple=(-1.15,-.05)
    opening_z: tuple=(1.10,1.95)
    target_shift_y: float=0.
    start_yaw: float=0.
    gap_count: int=1
    target_y: float=0.
    texture_seed: int=8

    def metadata(self):return asdict(self)

    @classmethod
    def three_gaps(cls,target_shift_y=0.,start_yaw=0.):
        return cls(start=(-.64,-2.57,1.65),pregrasp=(0.,2.35,1.5),
                   exit=(-.64,-2.57,1.55),door_y=-1.40,
                   target_shift_y=target_shift_y,start_yaw=start_yaw,
                   gap_count=3,target_y=2.35)

    def doors(self):
        if self.gap_count==1:return [(self.door_y,self.opening_x,self.opening_z)]
        if self.gap_count==3:
            # Follow the approved Stage 1 alternating passage layout. The
            # The 0.70 m high openings match the approved Stage 1 passage
            # logic: a hanging arm has no collision-free through-height,
            # while a bent arm and lowered UAV can cross.
            return [(self.door_y,(-1.10,-.20),(.95,1.65)),
                    (-.05,(-.15,.75),(1.00,1.70)),
                    (1.35,(-.90,0.),(.96,1.66))]
        raise ValueError('gap_count must be 1 or 3')

    def navigation_route(self):
        if self.gap_count==1:return [self.pregrasp]
        route=[]
        for door_y,(x0,x1),(z0,z1) in self.doors():
            center=(x0+x1)/2
            route.append((center,door_y-.72,1.55))  # stop and scan before risk zone
            route.append((center,door_y+.34,z0+.40))
        return route+[self.pregrasp]

    def scan_route_indices(self):
        return tuple(range(0,2*self.gap_count,2)) if self.gap_count==3 else ()

    def add(self,root):
        u=root.find("worldbody/body[@name='uav']")
        u.set('pos',' '.join(map(str,self.start)))
        u.set('quat',f'{math.cos(self.start_yaw/2)} 0 0 {math.sin(self.start_yaw/2)}')
        world=root.find('worldbody')
        for index,(door_y,opening_x,opening_z) in enumerate(self.doors()):
            x0,x1=opening_x;z0,z1=opening_z;suffix='' if self.gap_count==1 else f'_{index}'
            def wall(name,xlo,xhi,zlo,zhi):
                S(world,'geom',name=name+suffix,type='box',pos=f'{(xlo+xhi)/2} {door_y} {(zlo+zhi)/2}',
                  size=f'{(xhi-xlo)/2} .03 {(zhi-zlo)/2}',rgba='.43 .48 .51 1')
            wall('door_lower',-1.57,1.57,0,z0);wall('door_upper',-1.57,1.57,z1,2.6)
            wall('door_left',-1.57,x0,z0,z1);wall('door_right',x1,1.57,z0,z1)
            if self.gap_count==3:
                for name,x,z,sx,sz in [('left',x0,(z0+z1)/2,.009,(z1-z0)/2),
                                        ('right',x1,(z0+z1)/2,.009,(z1-z0)/2),
                                        ('top',(x0+x1)/2,z1,(x1-x0)/2,.009)]:
                    S(world,'geom',name=f'gap_frame_{index}_{name}',type='box',pos=f'{x} {door_y-.035} {z}',
                      size=f'{sx} .003 {sz}',contype='0',conaffinity='0',rgba='.10 .62 .69 1')
        if self.gap_count==3:
            for index,x,y,radius in [(0,.55,-.72,.11),(1,-1.20,.56,.10)]:
                S(world,'geom',name=f'obstacle_post_{index}',type='cylinder',pos=f'{x} {y} 1.05',
                  size=f'{radius} 1.05',rgba='.28 .35 .40 1')
        S(world,'geom',name='support',type='box',pos=f'0 {self.target_y} .953',size='.06 .12 .025',rgba='.4 .3 .2 1')
        obj=S(world,'body',name='object',pos=f'0 {self.target_y+self.target_shift_y} .998');S(obj,'freejoint',name='object_free')
        S(obj,'geom',name='target',type='cylinder',size='.012 .02',mass='.04',friction='1 .01 .001',rgba='1 .02 .02 1')

