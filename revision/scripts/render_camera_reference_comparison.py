"""Same-view native sensor assembly versus the registered MuJoCo assembly."""
from pathlib import Path
import os, sys, json, zipfile, hashlib
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from xml.etree.ElementTree import Element, SubElement as S, tostring
from prepare_custom_landing import read_ply

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'mujoco_closed_loop'))
from super_metric_model import build_super_metric
native=json.loads((ROOT/'revision/assets/cad16_geometry_manifest.json').read_text())
registration=json.loads((ROOT/'revision/data/body_camera_registration.json').read_text())
R=np.array(registration['rotation']);t=np.array(registration['translation'])
out=ROOT/'revision/data/camera_alignment'
generated=ROOT/'revision/generated/camera_reference';generated.mkdir(parents=True,exist_ok=True)
root=Element('mujoco',model='native_camera_reference')
S(root,'compiler',angle='radian')
visual=S(root,'visual');S(visual,'global',offwidth='1000',offheight='850',orthographic='true',fovy='.19')
S(visual,'headlight',ambient='.55 .55 .55',diffuse='.65 .65 .65',specular='.15 .15 .15')
asset=S(root,'asset');S(asset,'texture',name='white',type='skybox',builtin='flat',rgb1='1 1 1',width='256',height='1536')
world=S(root,'worldbody');body=S(world,'body',name='reference',pos='0 0 1.5')
with zipfile.ZipFile(ROOT/'revision/assets/cad16_native_meshes.zip') as z:
    for i in [98,99,100]+list(range(102,137)):
        c=native['components'][i];raw=z.read(c['file'])
        assert hashlib.sha256(raw).hexdigest()==c['sha256']
        v,f=read_ply(raw);v=v@R.T+t
        path=generated/('part%03d.obj'%i)
        with path.open('w') as h:
            for p in v:h.write('v '+' '.join('%.9g'%q for q in p)+'\n')
            for p in f+1:h.write('f '+' '.join(map(str,p))+'\n')
        S(asset,'mesh',name='part%d'%i,file=str(path))
        col=c['color']
        if isinstance(col,str):col=[int(col[k:k+2],16)/255 for k in (1,3,5)]
        elif i==100:col=[.25,.45,.55]
        S(body,'geom',type='mesh',mesh='part%d'%i,rgba=' '.join(map(str,list(col)+[1.])),contype='0',conaffinity='0')

images=[]
for name,xml in [('Native CAD reference',tostring(root,encoding='unicode')),('Registered SUPER assembly',build_super_metric(fixed=True)[0])]:
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    m.vis.global_.fovy=.19
    renderer=mujoco.Renderer(m,height=680,width=800)
    cam=mujoco.MjvCamera();cam.lookat[:]=[.025,0,1.585];cam.distance=1.;cam.azimuth=130;cam.elevation=-10;cam.orthographic=True
    opt=mujoco.MjvOption();opt.geomgroup[3]=0;opt.sitegroup[:]=0
    renderer.update_scene(d,camera=cam,scene_option=opt);images.append(renderer.render().copy());renderer.close()
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':10})
fig,axes=plt.subplots(1,2,figsize=(10,4.7),gridspec_kw={'wspace':.03})
for ax,img,label in zip(axes,images,['(a) Native CAD reference','(b) Registered SUPER assembly']):
    ax.imshow(img);ax.axis('off');ax.set_title(label,loc='left',fontsize=12)
fig.text(.5,.035,'Same viewpoint and scale; native camera-to-MID360 transform retained.',ha='center',fontsize=10)
fig.subplots_adjust(left=.02,right=.98,top=.92,bottom=.08)
fig.savefig(out/'native_camera_comparison.png',dpi=200,facecolor='white')
fig.savefig(out/'native_camera_comparison.pdf',facecolor='white')
print('Native CAD and current assembly comparison saved.')
