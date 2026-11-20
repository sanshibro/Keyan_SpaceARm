"""Registered stepped dynamics and a private estimator FK copy for the oblique single-gate task."""
from xml.etree import ElementTree as ET
from types import SimpleNamespace
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from arm_command import ArmCommand
from round_f_plant import Plant, LAUNCH
from gate_feedback_spec import ROOT, SPECFILE, P, OUT, digest, dump, verify_frozen
from round_g1_model import Geometry
from round_aug_vis import bend

def cart(theta,phi):
    return np.deg2rad(theta)*np.array([np.cos(np.deg2rad(phi)),np.sin(np.deg2rad(phi))])
def limit_vector(v,r):
    n=np.linalg.norm(v)
    return v*(r/n) if n>r else v
def boxes():
    g=P['gate'];x=g['center_x'];half=g['thickness']/2;w=g['half_width'];s=g['sill_z'];l=g['lintel_z']
    beam=g['beam_depth_z'];post=g['post_width_y']
    sections=[('sill',-w,w,s-beam,s),('lintel',-w,w,l,l+beam),
              ('jamb_negative',-w-post,-w,0,l+beam),('jamb_positive',w,w+post,0,l+beam)]
    return [(n,np.array([x-half,a,c],float),np.array([x+half,b,d],float)) for n,a,b,c,d in sections]
class FlightPlant(Plant):
    def issue(self,k,us,controls,theta):
        b=getattr(self,'bend_command',np.zeros(2))
        mag=np.linalg.norm(b)
        self.arm.apply(ArmCommand(k,us*1e-6,mag,np.arctan2(b[1],b[0]) if mag else 0.))
        self.d.ctrl[self.rotors]=12.*np.asarray(controls)
        return float(np.degrees(mag))
def make_plant(canvas=True):
    def scene(root):
        world=root.find('worldbody');vehicle=world.find("body[@name='uav']")
        for geom in vehicle.iter('geom'):
            if geom.get('name','').startswith(('visual_','qingyu_rviz_')) and geom.get('contype')=='0':
                geom.set('contype','4');geom.set('conaffinity','0')
        for name,lo,hi in boxes():
            ET.SubElement(world,'geom',name='G1_'+name,type='box',group='4',pos=' '.join(map(str,(hi+lo)*.5)),
                          size=' '.join(map(str,(hi-lo)*.5)),contype='2',conaffinity='5',rgba='.55 .63 .67 .32')
        tool=vehicle.find(".//body[@name='tool']")
        if P['payload_kg']:
            site=tool.find("site[@name='tip']")
            # The native tool derives inertia from geoms. The load enters that same
            # rigid-body sum and has its own physical gate and self-collision surface.
            if tool.find('inertial') is not None:
                raise ValueError('Tool switched away from geometry-derived inertia')
            ET.SubElement(tool,'geom',name='alignment_probe_load',type='sphere',
                          pos=site.get('pos'),size='.012',mass=str(P['payload_kg']),
                          group='1',contype='1',conaffinity='1',rgba='.43 .42 .37 1')
        if canvas:
            from gate_live_optics import add_sensor_canvas
            add_sensor_canvas(root)
    return FlightPlant(scene_update=scene)

class ObserverGeometry(Geometry):
    def __init__(self,plant):
        super().__init__(SimpleNamespace(m=plant.m,d=mujoco.MjData(plant.m)))
        self.gates=boxes()
        self.glo=np.array([o[1] for o in self.gates]);self.ghi=np.array([o[2] for o in self.gates])
        self.freebody=self.m.body('uav').id
        self.local_mount=None
        self.place_estimate(np.zeros(3),np.eye(3),np.zeros(2))
        self.local_mount=self.d.body('mount').xmat.reshape(3,3).copy()
    def place_estimate(self,xyz,R,b):
        self.d.qpos[:3]=xyz;self.d.qpos[3:7]=np.roll(Rotation.from_matrix(R).as_quat(),1)
        theta=np.linalg.norm(b)
        self.bridge.set_kinematic_shape(self.m,self.d,min(theta,np.deg2rad(120)),np.arctan2(b[1],b[0]) if theta else 0.)
        mujoco.mj_kinematics(self.m,self.d)
    def info(self,xyz,R,b,extra=None):
        self.place_estimate(xyz,R,b);lo,hi=self.bounds()
        if extra is not None:lo+=extra;hi+=extra
        distance=self.distances(lo,hi).min(axis=1);rigid=self.cat=='body'
        return float(distance[rigid].min()),float(distance[~rigid].min()),float(lo[:,0].min()-self.ghi[0,0])
    def world_axis_for(self,b):
        self.place_estimate(np.zeros(3),np.eye(3),b)
        return self.tip()[1]
    def has_self_contact(self):
        mujoco.mj_forward(self.m,self.d)
        return any(c.dist<=0 and self.m.geom_bodyid[c.geom1] and self.m.geom_bodyid[c.geom2]
                   for c in self.d.contact[:self.d.ncon])
    def geom_axis_bend(self,R,axis):
        v=self.local_mount.T@R.T@axis
        return np.arctan2(np.linalg.norm(v[:2]),v[2])*v[:2]/max(np.linalg.norm(v[:2]),1e-12)
def truth_bend(plant):
    mount=plant.d.body('mount').xmat.reshape(3,3)
    tool=plant.d.body('tool').xmat.reshape(3,3)
    return bend(mount.T@tool)
