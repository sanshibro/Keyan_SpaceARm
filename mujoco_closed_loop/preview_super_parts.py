"""Review converted official CAD as isolated parts, not a validated aircraft assembly."""
from pathlib import Path
from xml.etree.ElementTree import Element,SubElement as S,tostring
import json,mujoco,numpy as np,cv2
ROOT=Path(__file__).resolve().parent;directory=ROOT/'vendor/super_converted'
report=json.loads((directory/'provenance.json').read_text());views=[]
for name,part in report['parts'].items():
 root=Element('mujoco');a=S(root,'asset');S(a,'mesh',name='part',file=str(directory/(name+'.obj')))
 w=S(root,'worldbody');S(w,'light',pos='0 0 1',diffuse='1 1 1')
 S(w,'geom',type='mesh',mesh='part',quat='0.70710678 0.70710678 0 0' if name!='foot' else '1 0 0 0',rgba='.15 .35 .45 1')
 m=mujoco.MjModel.from_xml_string(tostring(root,encoding='unicode'));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
 camera=mujoco.MjvCamera();camera.lookat[:]=[0,0,0];camera.distance=.40 if name=='main_board' else .24;camera.azimuth=75;camera.elevation=-60
 r=mujoco.Renderer(m,480,640);r.update_scene(d,camera=camera);frame=cv2.cvtColor(r.render(),cv2.COLOR_RGB2BGR);r.close()
 cv2.putText(frame,name+' | official STEP -> OBJ',(15,28),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
 cv2.putText(frame,'Isolated CAD inspection; assembly/dynamics pending',(15,460),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1);views.append(frame)
cv2.imwrite(str(directory/'parts_preview.jpg'),np.concatenate([np.concatenate(views[:2],axis=1),np.concatenate(views[2:],axis=1)],axis=0));print(directory/'parts_preview.jpg')
