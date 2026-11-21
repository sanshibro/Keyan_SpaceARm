"""Native SUPER/Qingyu plant, shared industrial optics and registered optical task."""
import hashlib,json,os
from pathlib import Path
from xml.etree import ElementTree as ET
import cv2,mujoco,numpy as np
from scipy.spatial.transform import Rotation
from gate_feedback_model import FlightPlant,ObserverGeometry,cart
from gate_live_optics import LiveOptics,SensorCanvas
from m1_layout import AXIS,FACE,STANDOFF,HOVER
from sensors import RGBDSensor

STAGE=Path(__file__).resolve().parents[1]
SCENE=Path(os.environ.get('M1_SCENE',str(STAGE/'scene')))
def vec(x):return ' '.join('%.9g'%i for i in x)
def option():
    opt=mujoco.MjvOption()
    opt.geomgroup[:]=[1,1,1,0,0,0]
    opt.sitegroup[:]=0
    return opt

def build(assets=SCENE):
    assets=Path(assets)
    def install(root):
        world=root.find('worldbody');vehicle=world.find("body[@name='uav']")
        for geom in vehicle.iter('geom'):
            if geom.get('name','').startswith(('visual_','qingyu_rviz_')) and geom.get('contype')=='0':
                geom.set('contype','4');geom.set('conaffinity','0')
        tool=vehicle.find(".//body[@name='tool']")
        assert tool.find('inertial') is None
        ET.SubElement(tool,'geom',name='alignment_probe_load',type='sphere',pos=tool.find("site[@name='tip']").get('pos'),
                      size='.012',mass='.045',group='1',contype='1',conaffinity='1',rgba='.43 .42 .37 1')
        cell=ET.parse(assets/'cell.xml').getroot()
        for elem in cell.find('asset'):
            if 'file' in elem.attrib:elem.set('file',str((assets/elem.get('file')).resolve()))
            root.find('asset').append(elem)
        for elem in cell.find('worldbody'):world.append(elem)
        for name,pos,look,angle in [('observer_oblique',[5.45,-5.5,4.32],[1.70,.02,1.80],29),
                                   ('observer_side',[1.74,-6,2.24],[1.74,0,1.99],40),
                                   ('observer_dock',[3.99,-1.31,2.85],[3.37,.18,1.85],26),
                                   ('observer_end',[4.50,-.48,2.22],[1.42,.08,2.01],52)]:
            forward=np.asarray(look)-pos;forward/=np.linalg.norm(forward)
            right=np.cross(forward,[0,0,1.]);right/=np.linalg.norm(right);up=np.cross(right,forward)
            ET.SubElement(world,'camera',name=name,pos=vec(pos),xyaxes=vec([*right,*up]),fovy=str(angle))
        vis=root.find('visual')
        if vis is None:vis=ET.SubElement(root,'visual')
        g=vis.find('global')
        if g is None:g=ET.SubElement(vis,'global')
        g.set('offwidth','1280');g.set('offheight','720')
        for lamp in world.findall('light'):
            lamp.set('ambient','.30 .31 .32')
        # The transparent cut face and opaque ribs are permanently part of all views.
    p=FlightPlant(scene_update=install)
    p.scene_digest=json.loads((assets/'manifest.json').read_text())['scene_sha256']
    return p

class SharedCanvas(SensorCanvas):
    def __init__(self,plant,folder,texture_seed=0):
        self.mujoco=mujoco;self.model=plant.m;self.data=mujoco.MjData(plant.m)
        self.sensor=RGBDSensor(plant.m,'wrist_rgbd',480,360);self.K=self.sensor.K.copy()
        self.scene_option=option();self.nq=plant.m.nq
        Path(folder,'shared-scene.json').write_text(json.dumps(dict(scene_sha256=plant.scene_digest,
             camera='wrist_rgbd',world_model_sha256=hashlib.sha256(plant.xml.encode()).hexdigest(),geom_groups=list(map(int,self.scene_option.geomgroup)))))
class IndustryOptics(LiveOptics):
    canvas_class=SharedCanvas

def task_pose(plant):
    fk=ObserverGeometry(plant)
    q=cart(50.,143.);fk.place_estimate(np.zeros(3),np.eye(3),q)
    mujoco.mj_camlight(plant.m,fk.d)
    cam=fk.d.camera('wrist_rgbd');rot=cam.xmat.reshape(3,3).copy()
    tip,axis=fk.tip();body=FACE-STANDOFF*(-rot[:,2])-cam.xpos
    return dict(body=body.tolist(),tool=(body+tip).tolist(),axis=AXIS.tolist(),
                theta_deg=50.,phi_deg=143.,yaw_deg=0.,camera=(body+cam.xpos).tolist(),
                camera_axis=(-rot[:,2]).tolist(),tool_axis=axis.tolist())

def optics_survey(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);p=build()
    task=task_pose(p);g=ObserverGeometry(p);g.place_estimate(task['body'],np.eye(3),cart(50,143))
    canvas=SharedCanvas(p,out)
    rgb,depth=canvas.capture(0,g.d.qpos);cv2.imwrite(str(out/'flange-wrist.png'),rgb);cv2.imwrite(str(out/'flange-depth.png'),depth)
    d=canvas.data;cam=d.camera('wrist_rgbd');v=cam.xmat.reshape(3,3).T@(FACE-cam.xpos)
    K=canvas.K;u=K[0,2]+K[0,0]*v[0]/-v[2];w=K[1,2]-K[1,1]*v[1]/-v[2]
    renderer=mujoco.Renderer(p.m,height=640,width=1080)
    for name in ('observer_oblique','observer_dock'):
        renderer.update_scene(d,camera=name,scene_option=option())
        cv2.imwrite(str(out/(name+'.jpg')),cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR))
    renderer.close();canvas.close()
    result=dict(task=task,face=FACE.tolist(),scene_sha256=p.scene_digest,centre_uv=[u,w],
                optical_z_m=-v[2],depth_mm_at_centre=int(depth[round(w),round(u)]),
                depth_valid_fraction=float(np.mean(depth>0)),angle_tool_axis_deg=float(np.rad2deg(np.arccos(np.clip(np.dot(task['tool_axis'],AXIS),-1.,1.)))),
                mj_version=mujoco.__version__,geoms=p.m.ngeom,excluded_view_specific_meshes=0)
    (out/'registered-task.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
    assert .12<=-v[2]<=.18 and 150<=result['depth_mm_at_centre']<=180
if __name__=='__main__':
    import sys
    optics_survey(sys.argv[1])
