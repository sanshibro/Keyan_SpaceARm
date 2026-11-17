"""Stage 1 approximation. SI units; no state overwrite during integration."""
from xml.etree.ElementTree import Element, SubElement as S, tostring
import json
import math
from pathlib import Path
import yaml
from gripper import add_gripper
from qingyu_model import ARC_LENGTH, BASE_LENGTH

def add_super_visual(asset,u,tool,visual):
    """Reuse the archived SUPER airframe and reference D435 CAD as massless visuals."""
    here=Path(__file__).resolve().parent
    assembly=here/'vendor/super_converted/full_assembly'
    d435=here/'vendor/reference_install/d435'
    parts=json.loads((here/'vendor/reference_install/d435_manifest.json').read_text())['parts']
    colors={'Lidar':'.83 .87 .89 1','metal_lidar':'.21 .25 .28 1',
        'carbon':'.09 .11 .14 1','motor':'.19 .20 .22 1',
        'prop':'.085 .095 .11 1','metal':'.27 .30 .32 1',
        'lander':'.12 .13 .15 1','PCB':'.13 .29 .22 1',
        'yellow':'.9 .7 .10 1','green':'.17 .65 .23 1',
        'Led_front':'.12 .75 .85 1','Led_back':'.88 .18 .14 1',
        'mars_logo':'.9 .9 .9 1','usb':'.65 .68 .70 1'}
    for path in sorted(assembly.glob('*.obj')):
        S(asset,'mesh',name='super_full_'+path.stem,file=str(path))
        S(u,'geom',name='super_assembly_'+path.stem,type='mesh',mesh='super_full_'+path.stem,
          rgba=colors.get(path.stem,'.45 .45 .45 1'),mass='0',contype='0',conaffinity='0',group='2')
    for part in parts:
        name=part['mesh'];S(asset,'mesh',name='ref_'+name,file=str(d435/(name+'.stl')))
        S(u,'geom',name='body_'+name,type='mesh',mesh='ref_'+name,
          pos='.166 0 0',xyaxes='0 -1 0 0 0 1',rgba=part['rgba'],
          mass='0',contype='0',conaffinity='0',group='2')
        S(tool,'geom',name='wrist_'+name,type='mesh',mesh='ref_'+name,
          pos='.070267 .129267 .010823',
          xyaxes='.70710678 -.70710678 0 -.40557979 -.40557979 -.81915204',
          rgba=part['rgba'],mass='0',contype='0',conaffinity='0',group='2')
    for name in ('fuselage','navigation_camera_housing','d435','lidar_mast_visual','lidar_head_visual'):
        for geom in list(u.iter('geom')):
            if geom.get('name')==name:geom.set('rgba','0 0 0 0')
    for site in u.findall('site'):
        if site.get('name','').startswith('rotor'):site.set('rgba','0 0 0 0')
    sdk_visual=yaml.safe_load((here/'vendor/qingyu_continuum_arm/config/continuum_arm_visualize.yaml').read_text())
    count=int(sdk_visual['resolution']);radius=float(sdk_visual['marker_scale'])/2
    ds=ARC_LENGTH/8
    for i in range(8):
        u.find(f".//body[@name='seg{i}']/geom[@name='arm{i}']").set('rgba','0 0 0 0')
    for j in range(count):
        arc=ARC_LENGTH*j/(count-1);i=min(7,int(arc/ds));local_z=arc-(i+.5)*ds
        seg=u.find(f".//body[@name='seg{i}']")
        S(seg,'geom',name=f'qingyu_rviz_backbone_{j}',type='sphere',pos=f'0 0 {local_z:.6f}',
          size=str(radius),rgba='.075 .075 .075 1',mass='0',contype='0',conaffinity='0',group='2')
    S(visual,'headlight',ambient='.35 .35 .35',diffuse='.55 .55 .55')

def model_xml(fixed=True, segments=8, profile="qingyu", camera_profile='original', nav_camera_profile='front',discretization=None,super_visual=False):
    if profile not in ('qingyu','v2'):
        raise ValueError('Unknown arm model profile: '+profile)
    length = ARC_LENGTH if profile == "qingyu" else .22
    base_length = BASE_LENGTH if profile == "qingyu" else .10
    discretization=discretization or ('trapezoid' if profile=='qingyu' else 'endpoint')
    if discretization not in ('trapezoid','endpoint'):raise ValueError('Unknown arm discretization')
    root=Element('mujoco', model='aerial_continuum_stage1')
    S(root,'compiler',angle='radian',autolimits='true')
    # Fixed metric rendering clip planes shared by full scene and self projection.
    S(root,'statistic',extent='1')
    visual=S(root,'visual');S(visual,'map',znear='.005',zfar='30')
    S(root,'option', timestep='0.001',integrator='implicitfast',gravity='0 0 -9.81',cone='elliptic',impratio='10',noslip_iterations='3',iterations='100',tolerance='1e-9')
    default=S(root,'default');S(default,'geom', friction='1 0.01 0.001',condim='4')
    S(default,'joint',damping='0.15',armature='0.0001')
    asset=S(root,'asset')
    mesh_path=Path(__file__).resolve().parent/'vendor/qingyu_continuum_arm/meshes/A01Base_visual_lod.obj'
    S(asset,'mesh',name='qingyu_base',file=str(mesh_path),scale='.001 .001 .001')
    w=S(root,'worldbody');S(w,'light',pos='0 -2 4')
    S(w,'geom',name='floor',type='plane',size='5 5 .1',rgba='.3 .35 .4 1')
    u=S(w,'body',name='uav',pos='0 0 1.5')
    if not fixed:S(u,'freejoint',name='flight')
    S(u,'geom',name='fuselage',type='box',size='.14 .10 .06',mass='1.42' if nav_camera_profile in ('underbody','front_plus_survey','front_housed') else '1.5',rgba='.1 .3 .4 1')
    S(u,'geom',name='lidar_mast_visual',type='cylinder',pos='0 0 .13',size='.012 .07',density='0',contype='0',conaffinity='0',group='4',rgba='.18 .2 .22 1')
    S(u,'geom',name='lidar_head_visual',type='cylinder',pos='0 0 .2',size='.045 .025',density='0',contype='0',conaffinity='0',group='4',rgba='.08 .13 .17 1')
    for i,(x,y) in enumerate([(.2,.2),(-.2,.2),(-.2,-.2),(.2,-.2)]):
        S(u,'site',name=f'rotor{i}',pos=f'{x} {y} 0',size='.025')
    if nav_camera_profile in ('underbody','front_plus_survey'):
        # Survey candidate below the rigid base: the old -70 mm origin saw
        # the base in front of the UAV's lower safety envelope during yaw.
        S(u,'geom',name='navigation_camera_housing',type='box',pos='.166 0 -.13',xyaxes='0 1 0 -1 0 0',size='.045 .012 .015',mass='.072',rgba='.6 .6 .6 1')
        S(u,'geom',name='navigation_camera_mount',type='capsule',fromto='.14 0 -.04 .166 0 -.13',size='.005',mass='.008')
        S(u,'camera',name='survey_rgbd' if nav_camera_profile=='front_plus_survey' else 'navigation',pos='.15 0 -.13',xyaxes='0 1 0 -1 0 0',fovy='70')
    if nav_camera_profile=='front_housed':
        S(u,'geom',name='navigation_camera_housing',type='box',pos='.166 0 0',xyaxes='0 -1 0 0 0 1',size='.045 .012 .015',mass='.072',rgba='.6 .6 .6 1')
        S(u,'geom',name='navigation_camera_mount',type='capsule',fromto='.14 0 -.04 .166 0 0',size='.005',mass='.008',rgba='.12 .16 .18 1')
    if nav_camera_profile!='underbody':
        optical_x='.21' if super_visual and nav_camera_profile=='front_housed' else '.15'
        S(u,'camera',name='navigation',pos=f'{optical_x} 0 0',xyaxes='0 -1 0 0 0 1',fovy='70')
    b=S(u,'body',name='mount',pos='0 0 -.03',quat='0 0 1 0')
    S(b,'geom',type='cylinder',size=f'.0675 {base_length/2}',pos=f'0 0 {base_length/2}',mass='.3',rgba='.2 .2 .2 .25')
    S(b,'geom',type='mesh',mesh='qingyu_base',pos=f'0 0 {BASE_LENGTH}',density='0',contype='0',conaffinity='0',group='2')
    parent=b; acts=[]; exclusions=[]; ds=length/segments
    for i in range(segments):
        first=base_length+(ds/2 if discretization=='trapezoid' else 0)
        seg=S(parent,'body',name=f'seg{i}',pos=f'0 0 {first if i==0 else ds}')
        axes=[('bx','0 1 0'),('by','-1 0 0')]
        # Symmetric alternating axis order cancels the leading artificial
        # torsion of repeated two-axis rotations; planning remains theta/phi.
        if discretization=='trapezoid' and i%2:axes.reverse()
        for name,axis in axes:S(seg,'joint',name=f'{name}{i}',axis=axis,range='-.4 .4')
        ends=f'0 0 {-ds/2} 0 0 {ds/2}' if discretization=='trapezoid' else f'0 0 0 0 0 {ds}'
        S(seg,'geom',name=f'arm{i}',type='capsule',fromto=ends,size='.034',mass=str(.25/segments),rgba='.2 .65 .6 1')
        exclusions.append((parent.get('name'),seg.get('name')))
        for j in ['bx','by']: acts.append((f'{j}{i}',80 if discretization=='trapezoid' else 35,1.8 if discretization=='trapezoid' else 1.2))
        parent=seg
    tool=S(parent,'body',name='tool',pos=f'0 0 {ds/2 if discretization=="trapezoid" else ds}')
    if camera_profile=='grasp_visible':
        # Engineering calibration candidate: fixed elevated bracket, no gimbal.
        S(tool,'geom',name='bracket',type='box',pos='0 0 .045',size='.025 .025 .045',mass='.065')
        S(tool,'geom',name='camera_standoff',type='capsule',fromto='0 .025 0 .070267 .129267 .010823',size='.006',mass='.015')
        S(tool,'geom',name='d435',type='box',pos='.070267 .129267 .010823',xyaxes='.70710678 -.70710678 0 -.40557979 -.40557979 -.81915204',size='.045 .012 .015',mass='.07',rgba='.7 .7 .7 1')
        S(tool,'camera',name='wrist_rgbd',pos='.061 .12 .02',xyaxes='.70710678 -.70710678 0 -.40557979 -.40557979 -.81915204',fovy='69')
    else:
        S(tool,'geom',name='bracket',type='box',pos='0 0 .045',size='.025 .025 .045',mass='.08')
        S(tool,'geom',name='d435',type='box',pos='.045 0 .065',size='.015 .045 .012',mass='.07',rgba='.7 .7 .7 1')
        S(tool,'camera',name='wrist_rgbd',pos='.061 0 .065',xyaxes='0 -1 0 0 0 1',fovy='69')
    S(tool,'site',name='tip',pos='0 0 .16',size='.006')
    add_gripper(asset,tool,acts,exclusions)
    if super_visual:add_super_visual(asset,u,tool,visual)
    exclusions.append((parent.get('name'),'tool'))
    if segments>1 and discretization=='endpoint':
        # At 120 deg the second capsule's rounded root overlaps the rigid
        # base by about 30--40 micrometres. It is the same joined continuum
        # boundary, like the already excluded neighbouring capsule caps.
        exclusions.append(('mount','seg1'))
    # Overlapping neighboring capsules discretize one continuous soft volume.
    for i in range(segments):
        for j in range(i+2,min(segments,i+4)):
            exclusions.append((f'seg{i}',f'seg{j}'))
    contact=S(root,'contact')
    for a,b in exclusions:S(contact,'exclude',body1=a,body2=b)
    act=S(root,'actuator')
    for name,kp,kv in acts:
        S(act,'position',name=name,joint=name,kp=str(kp),kv=str(kv),forcerange='-4 4')
    if not fixed:
        for i in range(4):S(act,'general',name=f'thrust{i}',site=f'rotor{i}',gear=f'0 0 1 0 0 {0.02*(-1)**i}',ctrlrange='0 12')
    # Sensor first-hit filtering reserves group 1 for robot collision surfaces
    # and group 2 for robot visual meshes; external scene geometry stays group 0.
    for geom in u.iter('geom'):
        if geom.get('group') is None:geom.set('group','1')
    return tostring(root,encoding='unicode')

class SimArmBridge:
    """Single-segment simulated actuator mapping in radians, without ROS/serial.

    This is not a complete SDK bridge. The legacy enable=False path commands
    a neutral shape; it does not implement hardware torque disable.
    """
    def __init__(self,segments=8): self.segments=segments
    def set_kinematic_shape(self,model,data,theta,phi):
        """Internal estimator/CAD model only; never call on live physics."""
        command=self.move_joints(theta,phi)
        for i in range(self.segments):
            for j,name in enumerate(('bx','by')):
                data.qpos[model.jnt_qposadr[model.joint(f'{name}{i}').id]]=command[2*i+j]
    def move_joints(self,theta,phi,enable=True):
        import numpy as np
        if not all(math.isfinite(v) for v in (theta,phi)) or not 0<=theta<=math.radians(120):
            raise ValueError('Invalid single-segment shape')
        if not enable:return np.zeros(2*self.segments)
        return np.tile([theta*math.cos(phi)/self.segments,theta*math.sin(phi)/self.segments],self.segments)
