"""Generate the wrist RGB-D input for stock ORB-SLAM3, with no markers or world pose fields."""
import argparse
import json
from pathlib import Path
from xml.etree import ElementTree as ET
import cv2
import mujoco
import numpy as np
from round_shape_pixels import EVIDENCE,OUTPUT,calibration
from round_g1_model import make_plant
from sensors import RGBDSensor

def surface_image(path,seed):
    rng=np.random.default_rng(seed)
    size=1024
    shade=np.zeros((size,size),np.float32)
    for n,amplitude in [(9,14),(38,18),(120,21),(350,14),(1024,5)]:
        base=rng.normal(0,amplitude,(n,n)).astype('float32')
        shade+=cv2.resize(base,(size,size),interpolation=cv2.INTER_CUBIC)
    img=np.clip(164+shade,45,240).astype('uint8')
    # Uncoded mineral flecks and hairline scratches anchored to the static material.
    for _ in range(5000):
        x,y=rng.integers(0,size,2);radius=int(rng.integers(1,6))
        cv2.ellipse(img,(int(x),int(y)),(radius,max(1,radius//2)),float(rng.integers(0,180)),
                    0,360,int(rng.integers(55,223)),-1)
    assert cv2.imwrite(str(path),cv2.cvtColor(img,cv2.COLOR_GRAY2BGR))

def model_for(work,plain):
    base=make_plant();root=ET.fromstring(base.xml);asset=root.find('asset');world=root.find('worldbody')
    for g in world.findall('geom'):
        if g.get('name','').startswith('G1_'):
            g.set('group','0');g.set('rgba','.55 .59 .60 1')
    if not plain:
        for j,kind in enumerate(('floor','far','left','right','rear','gate')):
            tex=work/('finish-'+kind+'.png')
            if not tex.exists():surface_image(tex,71321+j)
            ET.SubElement(asset,'texture',name='tex_'+kind,type='2d',file=str(tex.resolve()))
            ET.SubElement(asset,'material',name='mat_'+kind,texture='tex_'+kind,
                          texrepeat='1 1',specular='.04',shininess='.05',texuniform='false')
        floor=world.find("geom[@name='floor']");floor.set('material','mat_floor');floor.set('rgba','.94 .95 .96 1')
        for kind,pos,axis,size in [
            ('far','3.35 0 1.8','0 1 0 -1.5707963','1.9 2.7 .012'),
            ('rear','-2.7 0 1.8','0 1 0 1.5707963','1.9 2.7 .012'),
            ('left','.3 -2.6 1.8','1 0 0 -1.5707963','3.1 1.9 .012'),
            ('right','.3 2.6 1.8','1 0 0 1.5707963','3.1 1.9 .012')]:
            ET.SubElement(world,'geom',name='finish_'+kind,type='box',pos=pos,axisangle=axis,size=size,
                          contype='0',conaffinity='0',material='mat_'+kind,group='0')
        for g in world.findall('geom'):
            if g.get('name','').startswith('G1_'):g.set('material','mat_gate')
    m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    assert m.nq==base.m.nq and m.nbody==base.m.nbody
    return m

def run(case,plain,store):
    cv2.setNumThreads(1);work=store/('plain' if plain else 'textured');work.mkdir(parents=True,exist_ok=True)
    dest=work/case;dest.mkdir(exist_ok=True)
    sentinel=dest/'complete.json'
    if sentinel.exists():return
    m=model_for(work,plain);d=mujoco.MjData(m);sensor=RGBDSensor(m,'wrist_rgbd',480,360)
    src=EVIDENCE/'runs'/case;record=np.load(str(src/'physics.npz'))
    release=json.loads((src/'run.json').read_bytes())['release_us']
    time=(record['us']-release)/1e6;idx=np.flatnonzero((time>=12)&(time<46))[::2]
    if plain:idx=idx[:150]
    lines=[];mini=[];K=sensor.K
    try:
        for j,i in enumerate(idx):
            t=time[i];d.qpos[:]=record['qpos'][i];d.time=t;mujoco.mj_forward(m,d);p=sensor.read(d)
            rgb='r%04d.jpg'%j;depth='d%04d.png'%j
            assert cv2.imwrite(str(dest/rgb),cv2.cvtColor(p.rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,92])
            mm=np.rint(1000*np.nan_to_num(p.depth_m,nan=0.)).astype('uint16')
            assert cv2.imwrite(str(dest/depth),mm,[cv2.IMWRITE_PNG_COMPRESSION,1])
            lines.append('%.6f %s %s\n'%(t,rgb,depth))
            if j%5==0:mini.append(cv2.resize(p.rgb,(240,180),interpolation=cv2.INTER_AREA))
    finally:sensor.close()
    (dest/'frames.txt').write_text(''.join(lines))
    tmpl=(Path.home()/'桌面/Test/ORB_SLAM3/Examples/RGB-D/TUM1.yaml').read_text()
    import re
    vals={'Camera1.fx':K[0,0],'Camera1.fy':K[1,1],'Camera1.cx':K[0,2],'Camera1.cy':K[1,2],
          'Camera.width':480,'Camera.height':360,'Camera.fps':25,'Camera.RGB':0,'RGBD.DepthMapFactor':1000.,
          **{'Camera1.'+k:0. for k in ('k1','k2','k3','p1','p2')}}
    for k,v in vals.items():
        tmpl,n=re.subn('^'+re.escape(k)+r':.*$',k+': '+str(v),tmpl,flags=re.M)
        assert n==1,k
    (dest/'settings.yaml').write_text(tmpl)
    result=OUTPUT/'slam'/('plain' if plain else 'textured')/case;result.mkdir(parents=True,exist_ok=True)
    (result/'settings.yaml').write_text(tmpl)
    np.savez_compressed(result/'wrists.npz',t=time[idx[::5]],rgb=mini)
    sentinel.write_text(json.dumps(dict(case=case,n=len(idx),start=float(time[idx[0]]),end=float(time[idx[-1]]))))
    print(case,'plain' if plain else 'textured',len(idx),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('case',nargs='+');a.add_argument('--plain',action='store_true')
    a.add_argument('--store',type=Path,default=Path('/tmp/g1_wrist_orb_20261119'))
    args=a.parse_args()
    for case in args.case:run(case,args.plain,args.store)
