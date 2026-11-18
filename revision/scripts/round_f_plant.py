"""F physical plant, synthetic sensor boundary and nominal experiment commands."""
import math
from xml.etree import ElementTree as ET
import mujoco
import numpy as np

from round_e1_model import ROOT, contacts
from round_e1_run import command as bending_command
from arm_command import ArmCommand, MujocoArmBackend
from super_metric_model import build_super_metric, allocation_matrix

ENU_NED = np.array([[0,1,0],[1,0,0],[0,0,-1.]])
FLU_FRD = np.diag([1.,-1.,-1.])
LAUNCH = np.array([0.,0.,1.5])
# Front right, rear left, front left, rear right in PX4 FRD convention.
CHANNELS = np.array([3,1,0,2])
LAT, LON, AMSL = 47.397742, 8.545594, 488.
DT = .0005
TICK = .004
CLOCK_ORIGIN_US = 1000000


def reference(t):
    if t is None: return 'preflight', np.array([0.,0.,.65]), 0.
    phase = ('liftoff' if t<9 else 'hover' if t<14 else 'translate' if t<22 else
             'bend' if t<36 else 'recover')
    u = np.clip((t-14.)/4.,0.,1.)
    return phase, np.array([.3*(1.-math.cos(math.pi*u)),0.,.65]), (bending_command(t-22) if phase=='bend' else 0.)


class Plant:
    def __init__(self):
        root=ET.fromstring(build_super_metric(fixed=False,segments=8)[0])
        root.find('option').set('timestep',str(DT))
        uav=root.find(".//body[@name='uav']")
        ET.SubElement(uav,'site',name='f_imu',pos='0 0 0',size='.002',rgba='0 0 0 0')
        sensors=ET.SubElement(root,'sensor')
        for tag in ('accelerometer','gyro'):
            ET.SubElement(sensors,tag,name=tag,site='f_imu')
        equality=ET.SubElement(root,'equality')
        ET.SubElement(equality,'weld',name='launch_support',body1='uav',solref='.002 1')
        self.xml=ET.tostring(root,encoding='unicode')
        self.m=mujoco.MjModel.from_xml_string(self.xml)
        self.d=mujoco.MjData(self.m)
        mujoco.mj_forward(self.m,self.d)
        self.base=self.m.body('uav').id
        self.rotors=np.array([self.m.actuator('thrust%d'%i).id for i in CHANNELS])
        self.arm=MujocoArmBackend(self.m,self.d,8)
        self.contact_count=0
        self.max_curve=0.
        self.minimum_body_up=1.
        self.mass=float(self.m.body_subtreemass[self.base])
        # Each column of A maps one PX4 motor's force into collective and FRD moments.
        a=allocation_matrix(self.m)[:,CHANNELS].copy()
        a[1:]=FLU_FRD@a[1:]
        # Official quad_x normalized mix rows (also verified against this build at runtime).
        mixer=np.array([[-1,1,1],[1,-1,1],[1,1,-1],[-1,-1,-1]],float)
        self.check=a[1:]@mixer
        if (np.max(np.abs(self.check-np.diag(np.diag(self.check))))>3e-5 or
            np.min(np.diag(self.check))<=0 or np.max(np.abs(a[1:].sum(1)))>3e-5):
            raise ValueError('Rotor positions, numbering or reaction direction disagree with PX4')
        self.noise=np.random.default_rng(241118)

    def issue(self,k,us,controls,theta):
        issued=self.arm.apply(ArmCommand(k,us/1e6,math.radians(theta),math.pi))
        self.d.ctrl[self.rotors]=12.*np.asarray(controls)
        return math.degrees(issued['issued']['theta_rad'])

    def step(self,free):
        for _ in range(round(TICK/DT)):
            mujoco.mj_step(self.m,self.d)
            if free:
                self.contact_count+=int(self.d.ncon>0)
                quat=self.d.qpos[3:7]
                self.minimum_body_up=min(self.minimum_body_up,1.-2.*float(quat[1]**2+quat[2]**2))
                q=self.d.qpos[self.arm.qpos_ids].reshape(-1,2)
                self.max_curve=max(self.max_curve,float(np.degrees(np.linalg.norm(q,axis=1).sum())))
        mujoco.mj_forward(self.m,self.d)
        if not np.isfinite(self.d.qpos).all() or not np.isfinite(self.d.qvel).all():
            raise ArithmeticError('Nonfinite plant state')

    def sensor(self):
        # MuJoCo linear accelerometer is specific force at the specified site.
        R=self.d.xmat[self.base].reshape(3,3)
        R_ned_frd=ENU_NED@R@FLU_FRD
        acc=FLU_FRD@self.d.sensor('accelerometer').data+self.noise.normal(0,.015,3)
        gyro=FLU_FRD@self.d.sensor('gyro').data+self.noise.normal(0,.0008,3)
        mag=R_ned_frd.T@np.array([.215,.012,.429])+self.noise.normal(0,.0003,3)
        p=self.d.xpos[self.base]-LAUNCH
        alt=AMSL+p[2]+self.noise.normal(0,.018)
        press=1013.25*(1-alt/44330.77)**5.25588
        return acc,gyro,mag,press,alt

    def gps(self):
        p=self.d.xpos[self.base]-LAUNCH
        v=ENU_NED@self.d.qvel[:3]
        return (round((LAT+p[1]/111132.)*1e7),round((LON+p[0]/(111320.*math.cos(math.radians(LAT))))*1e7),
                round((AMSL+p[2])*1000),round(np.linalg.norm(v)*100),*(round(a*100) for a in v))
