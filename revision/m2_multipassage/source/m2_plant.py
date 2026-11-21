"""Actual native SUPER + Qingyu plant with the single shared M2 corridor.

build() is the live NUC plant. portable() is a static-only local model rebuilt
from the prior original robot XML and content-addressed CAD assets; it has no
flight stepping API. No collision masks or robot contact exclusions are removed.
"""
import hashlib,json,os,tarfile
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET
import mujoco,numpy as np
from m2_scene import AXIS,FACE,STANDOFF,HOVER,CAMERAS

STAGE=Path(__file__).resolve().parents[1];SCENE=Path(os.environ.get('M2_SCENE',str(STAGE/'scene')))
def vec(x):return ' '.join('%.9g'%i for i in x)
def option():
    opt=mujoco.MjvOption();opt.geomgroup[:]=[1,1,1,0,0,0];opt.sitegroup[:]=0
    return opt
def install(root,assets=SCENE,robot_setup=True):
    assets=Path(assets);world=root.find('worldbody');vehicle=world.find("body[@name='uav']")
    if robot_setup:
        for geom in vehicle.iter('geom'):
            if geom.get('name','').startswith(('visual_','qingyu_rviz_')) and geom.get('contype')=='0':
                geom.set('contype','4');geom.set('conaffinity','0')
        tool=vehicle.find(".//body[@name='tool']")
        if tool.find("geom[@name='alignment_probe_load']") is None:
            if tool.find('inertial') is not None:raise ValueError('Native tool no longer derives inertia from geoms')
            ET.SubElement(tool,'geom',name='alignment_probe_load',type='sphere',pos=tool.find("site[@name='tip']").get('pos'),
                size='.012',mass='.045',group='1',contype='1',conaffinity='1',rgba='.43 .42 .37 1')
    cell=ET.parse(assets/'cell.xml').getroot()
    for elem in cell.find('asset'):
        if 'file' in elem.attrib:elem.set('file',str((assets/elem.get('file')).resolve()))
        root.find('asset').append(elem)
    for elem in cell.find('worldbody'):world.append(elem)
    for name,pos,look,angle in CAMERAS:
        forward=np.asarray(look)-pos;forward/=np.linalg.norm(forward);right=np.cross(forward,[0,0,1.]);right/=np.linalg.norm(right);up=np.cross(right,forward)
        ET.SubElement(world,'camera',name=name,pos=vec(pos),xyaxes=vec([*right,*up]),fovy=str(angle))
    vis=root.find('visual')
    if vis is None:vis=ET.SubElement(root,'visual')
    g=vis.find('global')
    if g is None:g=ET.SubElement(vis,'global')
    g.set('offwidth','1600');g.set('offheight','1000')
    for lamp in world.findall('light'):lamp.set('ambient','.30 .31 .32')

def build(assets=SCENE):
    from gate_feedback_model import FlightPlant
    p=FlightPlant(scene_update=lambda root:install(root,assets))
    p.scene_digest=json.loads((Path(assets)/'manifest.json').read_text())['scene_sha256']
    return p

def place(model,data,xyz,theta_deg,phi_deg,yaw_deg=0):
    """Private static FK only; never call on the live plant data."""
    data.qpos[:3]=xyz;yaw=np.deg2rad(yaw_deg);data.qpos[3:7]=[np.cos(yaw/2),0,0,np.sin(yaw/2)]
    b=np.deg2rad(theta_deg)*np.array([np.cos(np.deg2rad(phi_deg)),np.sin(np.deg2rad(phi_deg))])
    for i in range(8):
        for j,axis in enumerate(('bx','by')):data.qpos[model.jnt_qposadr[model.joint(axis+str(i)).id]]=b[j]/8
    mujoco.mj_kinematics(model,data);mujoco.mj_camlight(model,data)

def task_pose(plant):
    d=mujoco.MjData(plant.m);place(plant.m,d,[0,0,0],50,143)
    cam=d.camera('wrist_rgbd');rot=cam.xmat.reshape(3,3).copy();axis=-rot[:,2]
    site=d.site('tip');body=FACE-STANDOFF*axis-cam.xpos
    toolaxis=d.body('tool').xmat.reshape(3,3)[:,2]
    return dict(body=body.tolist(),tool=(body+site.xpos).tolist(),axis=AXIS.tolist(),theta_deg=50.,phi_deg=143.,yaw_deg=0.,
      camera=(body+cam.xpos).tolist(),camera_axis=axis.tolist(),tool_axis=toolaxis.tolist())

def portable(assets=SCENE,robot_xml=None,cad_archive=None):
    parent=STAGE.parent
    robot_xml=Path(robot_xml or parent/'M1_六维全身联合规划与机臂耦合_20261121/work/portability/flight/model.xml')
    cad_archive=Path(cad_archive or parent/'冻结感知_单门双自由度精细作业闭环_20261120/records/registered-cad-assets.tgz')
    root=ET.fromstring(robot_xml.read_bytes());world=root.find('worldbody');asset=root.find('asset')
    # Only the frozen native robot and the original lighting survive; replace the old environment.
    for e in list(world):
        if e.tag=='body' and e.get('name')=='uav':continue
        if e.tag=='light':continue
        world.remove(e)
    for e in list(asset):
        if e.get('name','').startswith(('M1_','M0_')):asset.remove(e)
    memory={}
    with tarfile.open(cad_archive,'r:gz') as archive:
        manifest=json.loads(archive.extractfile('asset-hashes.json').read())
        byname={Path(k).name:v for k,v in manifest.items()}
        for node in root.findall('.//*[@file]'):
            old=node.get('file');checksum,name=manifest.get(old,byname.get(Path(old).name,(None,None)))
            if not name:raise FileNotFoundError('Native CAD asset missing: '+old)
            payload=archive.extractfile(name).read()
            if hashlib.sha256(payload).hexdigest()!=checksum:raise IOError('CAD checksum mismatch: '+name)
            key='cad_'+checksum+Path(old).suffix;memory[key]=payload;node.set('file',key)
    install(root,assets,robot_setup=False)
    for node in root.findall('.//*[@file]'):
        old=node.get('file')
        if old in memory:continue
        path=Path(old);key='m2_'+path.name;memory[key]=path.read_bytes();node.set('file',key)
    xml=ET.tostring(root,encoding='unicode');m=mujoco.MjModel.from_xml_string(xml,assets=memory);d=mujoco.MjData(m)
    return SimpleNamespace(m=m,d=d,xml=xml,base=m.body('uav').id,mass=float(m.body_subtreemass[m.body('uav').id]),
        scene_digest=json.loads((Path(assets)/'manifest.json').read_text())['scene_sha256'],static_only=True)

def __getattr__(name):
    if name not in ('SharedCanvas','IndustryOptics'):raise AttributeError(name)
    from gate_live_optics import LiveOptics,SensorCanvas
    from sensors import RGBDSensor
    class SharedCanvas(SensorCanvas):
        def __init__(self,plant,folder,texture_seed=0):
            self.mujoco=mujoco;self.model=plant.m;self.data=mujoco.MjData(plant.m)
            self.sensor=RGBDSensor(plant.m,'wrist_rgbd',480,360);self.K=self.sensor.K.copy();self.scene_option=option();self.nq=plant.m.nq
            Path(folder,'shared-scene.json').write_text(json.dumps(dict(scene_sha256=plant.scene_digest,camera='wrist_rgbd',
              world_model_sha256=hashlib.sha256(plant.xml.encode()).hexdigest(),geom_groups=list(map(int,self.scene_option.geomgroup)),view_specific_geometry=False)))
    class IndustryOptics(LiveOptics):canvas_class=SharedCanvas
    globals().update(SharedCanvas=SharedCanvas,IndustryOptics=IndustryOptics)
    return globals()[name]
