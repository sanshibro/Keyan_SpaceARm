"""Bake the same engineering scene for optical observations and rendering."""
import hashlib,json
from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np
from PIL import Image,ImageFilter,ImageDraw
from m2_scene import boxes,pipes,GATES,FACE,AXIS,CAMERAS

STAGE=Path(__file__).resolve().parents[1]
COLORS={'deck':(.52,.56,.57),'wall':(.65,.70,.71),'glass':(.64,.76,.79),
        'steel':(.46,.53,.55),'bulkhead':(.60,.67,.69),'duct':(.62,.66,.65),
        'copper':(.55,.41,.30),'flange':(.50,.54,.55)}
def vec(v):return ' '.join('%.9g'%x for x in v)
def surface(path,base,seed):
    rng=np.random.default_rng(seed);n=512
    raw=rng.normal(128,24,(n,n)).clip(0,255).astype('uint8')
    low=np.asarray(Image.fromarray(raw).filter(ImageFilter.GaussianBlur(12)),float)-128
    fine=rng.normal(0,3.8,(n,n));vertical=rng.normal(0,1.8,(1,n))
    a=np.asarray(base)[None,None,:]*255+(low*2.5+fine+vertical)[:,:,None]
    im=Image.fromarray(np.uint8(np.clip(a,0,255)));d=ImageDraw.Draw(im)
    for _ in range(30):
        x,y=rng.integers(0,n,2);length=int(rng.integers(4,70));c=tuple(int(np.clip(v*255-12,0,255)) for v in base)
        d.line((int(x),int(y),int(x+length),int(y+rng.integers(-2,3))),fill=c,width=1)
    im.save(path)

def bake(folder=STAGE/'scene'):
    out=Path(folder);out.mkdir(parents=True,exist_ok=True)
    root=ET.Element('mujoco',model='M2_shared_industrial_corridor');asset=ET.SubElement(root,'asset');world=ET.SubElement(root,'worldbody')
    for j,(name,color) in enumerate(COLORS.items()):
        surface(out/(name+'.png'),color,26112100+j)
        ET.SubElement(asset,'texture',name='M2_tex_'+name,type='2d',file=name+'.png')
        ET.SubElement(asset,'material',name='M2_mat_'+name,texture='M2_tex_'+name,texrepeat='3 3',texuniform='true',
                      rgba=('1 1 1 .14' if name=='glass' else '1 1 1 1'),specular='.22',shininess='.28',reflectance='0')
    def twin(name,kind,mat,**kw):
        kw={k:(vec(v) if isinstance(v,(list,tuple,np.ndarray)) else str(v)) for k,v in kw.items()}
        ET.SubElement(world,'geom',name='M2_'+name,type=kind,group='0',contype='0',conaffinity='0',material='M2_mat_'+mat,**kw)
        ET.SubElement(world,'geom',name='COL_'+name,type=kind,group='4',contype='2',conaffinity='5',rgba='0 0 0 0',**kw)
    for name,lo,hi,mat in boxes():twin(name,'box',mat,pos=(lo+hi)/2,size=(hi-lo)/2)
    for name,a,b,r,mat in pipes():twin(name,'cylinder',mat,fromto=[*a,*b],size=r)
    # Raised flange bolts are physical solids, not tracking markers.
    z=AXIS/np.linalg.norm(AXIS);u=np.cross(z,[0,0,1.]);u/=np.linalg.norm(u);v=np.cross(z,u)
    for j in range(8):
        ang=j*np.pi/4;p=FACE-.006*z+.083*(np.cos(ang)*u+np.sin(ang)*v)
        twin('flange_bolt_%d'%j,'cylinder','steel',fromto=[*(p-.005*z),*(p+.008*z)],size=.010)
    # Bulkhead fasteners and duct ribs add actual geometry in shared views.
    for g in (GATES[0],GATES[2]):
        for side in (-1,1):
            for h in (1.48,2.68):
                y=g['y']+side*(g['width']/2+.08)
                twin('%s_bolt_%s_%s'%(g['name'],side,h),'cylinder','steel',fromto=[g['x']-.064,y,h,g['x']+.064,y,h],size=.012)
    for j,x in enumerate((2.19,2.37,2.56,2.75,2.93)):
        twin('duct_rib_%d'%j,'box','steel',pos=[x,0,2.451],size=[.008,.803,.221])
    ET.indent(root);ET.ElementTree(root).write(out/'cell.xml',encoding='utf-8',xml_declaration=True)
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file() and p.name!='manifest.json'}
    scene_hash=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    manifest=dict(scene_sha256=scene_hash,files=files,gates=GATES,face=FACE.tolist(),axis=AXIS.tolist(),
       shared_views=True,view_specific_geometry=False,texture='fixed-seed natural brushed/coated metal',
       box_geometry=[dict(name=n,lo=a.tolist(),hi=b.tolist(),material=m) for n,a,b,m in boxes()],
       pipe_geometry=[dict(name=n,a=a.tolist(),b=b.tolist(),radius=r,material=m) for n,a,b,r,m in pipes()],
       registration_status='development: static feasibility must be audited before flight')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest
if __name__=='__main__':
    import sys
    print(json.dumps(bake(Path(sys.argv[1]) if len(sys.argv)>1 else STAGE/'scene')))
