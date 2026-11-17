"""Native 16-series CAD appearance review. Geometry is read-only and at 1:1 scale."""
from pathlib import Path
import sys,json,time,math,hashlib,os
import numpy as np
ROOT=Path(__file__).resolve().parent
CAD=ROOT.parent/'16_机腹模块化拆装/16_机腹模块化拆装/配色与爆炸渲染_20261116'
sys.path.append(str(CAD/'tools/renderer_venv/Lib/site-packages'))
import mitsuba as mi
import drjit as dr
mi.set_variant('cuda_ad_rgb')
from PIL import Image
OUT=ROOT/'CAD真实模型预览';OUT.mkdir(exist_ok=True)
manifest_path=CAD/'render/input/gauge60_20261117/manifest_gauge60.json'
manifest=json.loads(manifest_path.read_text('utf-8-sig'));parts=manifest['components']
CACHE=CAD/'render/cache'
os.chdir(CACHE)

def shader(i):
    p=parts[i];c=p['color']
    if isinstance(c,str):
        c=c.lstrip('#');v=np.array([int(c[j:j+2],16)/255 for j in (0,2,4)])
        v=np.where(v<=.04045,v/12.92,((v+.055)/1.055)**2.4)
    else:v=np.array(c[:3])
    role=p['material_role'];lum=v@np.array([.2126,.7152,.0722]);v=lum+.9*(v-lum)
    mix=.30
    if role in ['amber','copper']:mix=.12
    elif role in ['camera_lens','camera_mask','camera_cover','camera_bezel']:mix=.05
    elif role in ['carbon','carbon_plate','graphite']:mix=.22
    v=(1-mix)*v+mix*np.array([.56,.57,.59])
    b=dict(type='principled',base_color=dict(type='rgb',value=v.tolist()),roughness=max(.36,float(p['roughness'])),metallic=float(p['metallic'])*.65)
    if p.get('optical_cover') or role=='camera_cover':return dict(type='thindielectric',int_ior=1.49,ext_ior=1.000277)
    if p.get('texture'):
        b['base_color']=dict(type='bitmap',filename=f'{i:04}_000_texture.png')
        b=dict(type='mask',opacity=dict(type='bitmap',filename=f'{i:04}_000_texture_alpha.png',raw=True),nested=b)
    return b

def render(name,direction,gate=True):
    width,height=1800,1400;target=np.array([.002,-.105,.16]);span=.75
    direction=np.array(direction,float);direction/=np.linalg.norm(direction)
    half=.51 if gate else .40
    camera=mi.ScalarTransform4f.look_at(origin=(target+direction*4).tolist(),target=target.tolist(),up=[0,1,0])@mi.ScalarTransform4f.scale([half,half,1])
    scene=dict(type='scene',integrator=dict(type='path',max_depth=6,hide_emitters=True),sensor=dict(type='orthographic',to_world=camera,near_clip=.00001,far_clip=100,film=dict(type='hdrfilm',width=width,height=height,pixel_format='rgba',rfilter=dict(type='tent')),sampler=dict(type='independent',sample_count=256)),environment=dict(type='constant',radiance=dict(type='rgb',value=[.46,.47,.49])))
    def light(pos,size,val):
        return dict(type='rectangle',to_world=mi.ScalarTransform4f.look_at(origin=(target+np.array(pos)*span).tolist(),target=target.tolist(),up=[0,1,0])@mi.ScalarTransform4f.scale([size*span,size*span,1]),emitter=dict(type='area',radiance=dict(type='rgb',value=val)))
    scene['key']=light([-3,5,5],1.8,[7.6]*3);scene['fill']=light([4,2,2],1.6,[5.4,5.5,5.6]);scene['rim']=light([-2,3,-4],1.3,[5.4,5.6,5.8])
    for i in range(len(parts)):
        scene[f'p{i}']=dict(type='ply',filename=f'{i:04}_000.ply',bsdf=shader(i))
    # New geometry review aperture; never assigned historical performance numbers.
    # Native model is Y-up, forward +Z, metres.
    if gate:
        def box(label,center,half,color):
            scene[label]=dict(type='cube',to_world=mi.ScalarTransform4f.translate(center)@mi.ScalarTransform4f.scale(half),bsdf=dict(type='roughplastic',diffuse_reflectance=dict(type='rgb',value=color),alpha=.35))
        for label,center,half in [('left',[-.32,-.10,.22],[.035,.30,.022]),('right',[.32,-.10,.22],[.035,.30,.022]),('upper',[0,.20,.22],[.355,.025,.022]),('lower',[0,-.40,.22],[.355,.025,.022])]:
            box('gate_'+label,center,half,[.44,.48,.51])
        for label,center,half in [('upper',[0,.176,.194],[.285,.002,.002]),('lower',[0,-.376,.194],[.285,.002,.002])]:box('edge_'+label,center,half,[.15,.30,.36])
    start=time.time();print('RENDER',name,len(parts),flush=True)
    obj=mi.load_dict(scene);img=mi.render(obj,spp=256,seed=117);dr.eval(img);dr.sync_thread()
    rgba=np.array(mi.Bitmap(img).convert(mi.Bitmap.PixelFormat.RGBA,mi.Struct.Type.UInt8,True),copy=True)
    alpha=rgba[:,:,3:4].astype(float)/255
    rgb=(rgba[:,:,:3]*alpha+255*(1-alpha)).round().astype('uint8')
    Image.fromarray(rgb).save(OUT/(name+'.png'))
    result=dict(kind='native CAD visual review; not experimental result',configuration='native 90-degree assembly',native_component_count=len(parts),original_geometry_modified=False,uniform_scale=1.,units='m',camera='orthographic',direction=direction.tolist(),size=[width,height],manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),render_seconds=time.time()-start)
    (OUT/(name+'.json')).write_text(json.dumps(result,indent=2),encoding='utf-8');print('DONE',result['render_seconds'],flush=True)

if __name__=='__main__':
    render('整机门洞_原生CAD_90度',[1.,.18,1.2],True)
    render('整机近景_原生CAD_90度',[1.,.06,.45],False)
