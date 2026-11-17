"""Render known targets through the two actual MuJoCo sensor cameras."""
from pathlib import Path
from xml.etree.ElementTree import fromstring,SubElement,tostring
import os,sys,json,math
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np,mujoco,cv2
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'mujoco_closed_loop'))
from super_metric_model import build_super_metric
from model import SimArmBridge

out=ROOT/'revision/data/camera_alignment';out.mkdir(parents=True,exist_ok=True)
xml,info=build_super_metric(fixed=True);base=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(base)
SimArmBridge().set_kinematic_shape(base,d,math.pi/2,math.pi);mujoco.mj_forward(base,d)
results=[]
for name,parent,forward,up,right in [('navigation','uav',[1,0,0],[0,0,1],[0,-1,0]),('wrist_rgbd','tool',[0,0,1],[-1,0,0],[0,-1,0])]:
    R=d.xmat[base.body(parent).id].reshape(3,3)
    f=R@forward;u=R@up;x=R@right;c=base.camera(name).id
    optical=-d.cam_xmat[c].reshape(3,3)[:,2]
    center=d.cam_xpos[c]+f*.35
    root=fromstring(xml);world=root.find('worldbody')
    # Normal is the expected mechanical forward direction, not read back from
    # the camera. Four asymmetric markers detect backwards/upside-down views.
    marks=[('center',0.,0.,[.15,.55,.27,1]),('left',-.095,0.,[.16,.31,.70,1]),
           ('right',.095,0.,[.75,.43,.12,1]),('top',0.,.095,[.75,.16,.16,1])]
    for label,h,v,color in marks:
        p=center+x*h+u*v
        SubElement(world,'geom',name='target_'+label,type='sphere',pos=' '.join(map(str,p)),
            size='.023',rgba=' '.join(map(str,color)),contype='0',conaffinity='0',group='5')
    m=mujoco.MjModel.from_xml_string(tostring(root,encoding='unicode'));data=mujoco.MjData(m)
    data.qpos[:]=d.qpos;mujoco.mj_forward(m,data)
    renderer=mujoco.Renderer(m,height=480,width=640);opt=mujoco.MjvOption();opt.geomgroup[3]=0;opt.geomgroup[5]=1;opt.sitegroup[:]=0
    renderer.update_scene(data,camera=name,scene_option=opt);rgb=renderer.render().copy()
    cv2.imwrite(str(out/(name+'.png')),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR));renderer.close()
    rr=rgb[:,:,0].astype(float);gg=rgb[:,:,1].astype(float);bb=rgb[:,:,2].astype(float)
    red=(rr>gg*1.6)&(rr>bb*1.6)&(rr>65)
    blue=(bb>rr*1.5)&(bb>gg*1.3)&(bb>60)
    yred,xred=np.where(red);yblue,xblue=np.where(blue)
    passed=len(yred)>20 and float(yred.mean())<240 and len(xblue)>20 and float(xblue.mean())<320
    assert passed,(name,len(yred),len(xblue))
    results.append(dict(camera=name,optical_forward_world=optical.tolist(),expected_forward_world=f.tolist(),
        angular_error_deg=float(np.degrees(np.arccos(np.clip(optical@f,-1,1)))),
        red_top_pixels=len(yred),blue_left_pixels=len(xblue),target_visible_and_upright=bool(passed)))
(out/'manifest.json').write_text(json.dumps(dict(platform_id=info['platform_id'],
    scope='simulated camera-axis and image-orientation regression; synthetic known targets',cameras=results),indent=2)+'\n')
print(json.dumps(results,indent=2))
