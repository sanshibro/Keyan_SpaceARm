"""Bake the locked M0 metalwork and M1 obstacles into one UV-mapped MuJoCo asset library."""
import hashlib,json,os,re
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from m0_scene import inspection_scene,Piece,brick,turning,rgb,basis,C as OLD_C
from m1_layout import FACE,AXIS,STANDOFF,GATE_HALF,SILL,LINTEL,blocks,pipes,longitudinal

BASE=Path(os.getenv('M1_BAKE_DIR',str(Path(__file__).resolve().parents[1]/'records/scene')))
BASE.mkdir(parents=True,exist_ok=True)

def txt(v):return ' '.join(f'{x:.7g}' for x in v)
def writej(p,v):p.write_bytes((json.dumps(v,indent=2,ensure_ascii=False)+'\n').encode())
def hashfile(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def planar_texture(family,index,n=768):
    rng=np.random.default_rng(9017+index)
    w=rng.normal(size=(n,n));f=gaussian_filter(w,17,mode='wrap');f/=f.std()
    mid=gaussian_filter(w,2.2,mode='wrap');mid/=mid.std()
    y,x=np.indices(w.shape)
    micro=gaussian_filter(w,(.62,7.2),mode='wrap');micro/=micro.std()
    base=216+3*f+2.4*mid+2.8*w
    if family in ('steel','pipe'):
        base=215+3*f+2.1*mid+4.3*micro+1.2*w
    if family=='deck':
        cell=n/4;line=np.minimum(np.minimum(x%cell,cell-x%cell),np.minimum(y%cell,cell-y%cell))
        base-=23*np.exp(-.5*(line/1.5)**2)+7*np.exp(-.5*(line/3.3)**2)
    # Small machining scratches, mineral inclusions and panel inspection marks, anchored in material UV.
    for j in range(150 if family!='concrete' else 240):
        px,py=rng.integers(6,n-40,2);length=int(rng.integers(2,23));delta=float(rng.uniform(7,34))
        if family in ('steel','pipe'):base[py:py+1,px:px+length]-=delta
        else:base[py:py+int(rng.integers(1,4)),px:px+int(rng.integers(2,5))]-=delta
    if family=='concrete':
        for a,b in ((86,220),(539,408),(346,630)):
            d=np.hypot((x-a),(y-b));base-=15*np.exp(-(d-5)**2/2)-5*np.exp(-(d-8)**2/3)
    col=np.stack((base*1.014,base,base*.991),-1)
    p=BASE/(family+'.png');Image.fromarray(np.uint8(col.clip(0,255))).save(p);return p

def add_piece(parts,mesh,center,orient,color,group,name,metal=.45,rough=.43,opacity=1,grid=0):
    v,n,f=mesh;parts.append(Piece(v@orient.T+center,n@orient.T,f,color,metal,rough,opacity,grid,group,name))

def bake():
    legacy,_=inspection_scene();parts=[];static=blocks();pipe=pipes();B=basis(AXIS)
    for p in legacy:
        v=p.v.copy()
        if p.group.startswith('flange'):
            v=FACE+.72*(v-OLD_C)
        elif p.group=='support':
            v+=FACE-OLD_C
        elif p.group in ('sill','lintel','jambminus','jambplus'):
            v[:,2]=np.interp(v[:,2],[-.1,SILL,2.385,2.80],[-.1,SILL,LINTEL,2.78])
            v[:,1]=np.interp(v[:,1],[-.85,-.38,.38,.85],[-.85,-GATE_HALF,GATE_HALF,.85])
        else:v[:,0]=longitudinal(v[:,0])
        parts.append(Piece(v,p.n,p.f,p.color,p.metal,p.rough,p.opacity,p.grid,p.group,p.name))
    for name in ('service_spine','exit_baffle','return_cabinet','near_guard','roof'):
        l,h=map(np.asarray,static[name]);s=h-l;centre=(l+h)/2
        coat={'service_spine':'#8c9da0','exit_baffle':'#9aadae','return_cabinet':'#b4b9b2',
              'near_guard':'#a8c2c3','roof':'#c7d4d0'}[name]
        alpha=.105 if name in ('roof','near_guard') else 1.
        add_piece(parts,brick(s,.009),centre,np.eye(3),coat,name,name,opacity=alpha)
        if alpha==1.:
            for j in range(4):
                dz=.035*j
                pos=np.array([centre[0],l[1]+.002,h[2]-.05-dz]) if name=='exit_baffle' else np.array([centre[0],h[1]-.001,h[2]-.05-dz])
                add_piece(parts,brick([.11,.003,.006],.001),pos,np.eye(3),'#668185',name,'press slot')
    for name,(a,b,r) in pipe.items():
        a,b=np.asarray(a),np.asarray(b);axis=(b-a)/np.linalg.norm(b-a);h=np.linalg.norm(b-a)
        profile=[(0,0),(r*.95,0),(r,.004),(r,h-.004),(r*.96,h),(0,h)]
        add_piece(parts,turning(profile,60),a,basis(axis),'#8d9e9d',name,'riser pipe',.58,.36)
        for u in (.06,.35,.69,.93):
            rr=r+.004;hh=.012
            add_piece(parts,turning([(r,0),(rr,0),(rr,hh),(r,hh)],60),a+axis*(u*h),basis(axis),
                      '#54827e' if u in (.35,.69) else '#76898a',name,'coupling collar',.60,.4)
    # The inspection datum is on a real, visible plate in the lip plane. Its
    # copper engraving, centre dimple and bolt circle are physical scene assets.
    add_piece(parts,turning([(0,0),(.061,0),(.063,.002),(.063,.009),(0,.009)],96),FACE,B,'#759091','flange_front','surveyed face',.5)
    for rr,col,ww in ((.035,'#c3915f',.0012),(.052,'#abc1bc',.0007),(.006,'#dfba83',.001)):
        add_piece(parts,turning([(rr-ww,-.00025),(rr+ww,-.00025),(rr+ww,.0001),(rr-ww,.0001)],80),FACE,B,col,'flange_front','engraved datum',.5)
    root=ET.Element('mujoco',model='m1_inspection_cell');assets=ET.SubElement(root,'asset');world=ET.SubElement(root,'worldbody')
    for j,family in enumerate(('steel','concrete','deck','pipe')):
        file=planar_texture(family,j)
        ET.SubElement(assets,'texture',name='M1_'+family,type='2d',file=file.name)
    groups=defaultdict(list)
    for p in parts:
        family=('deck' if p.grid else 'pipe' if p.group in pipe else 'concrete' if p.group in ('sidewall','foundation','service_deck','approach_deck','roof','near_guard') and p.metal<.4 else 'steel')
        color=tuple(np.round(rgb(p.color),4))
        groups[(p.group,family,color,round(p.opacity,3),round(p.metal,2),round(p.rough,2))].append(p)
    matids={};names=[]
    for j,((group,family,color,alpha,metal,rough),ls) in enumerate(groups.items()):
        mat=(family,color,alpha,metal,rough)
        if mat not in matids:
            key=f'M1_mat_{len(matids):03}';matids[mat]=key
            ET.SubElement(assets,'material',name=key,texture='M1_'+family,rgba=txt([*color,alpha]),
                shininess=f'{.13+.5*(1-rough):.4f}',specular=f'{.06+.25*metal:.4f}')
        key=f'M1_{group}_{j:03}';fname=key+'.obj';names.append(key)
        # Material UVs follow metric world faces so vertical walls retain texture.
        verts=[];uvs=[];shell=[];offset=0
        for p in ls:
            tri=p.v[p.f];nor=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
            dominant=np.argmax(abs(nor),axis=1);uvw=np.empty((len(tri),3,2))
            for ax,(a,b) in enumerate(((1,2),(0,2),(0,1))):
                take=dominant==ax;uvw[take]=tri[take][:,:,[a,b]]/(.40 if family=='deck' else .56)
            points=np.concatenate((tri,uvw),axis=2).reshape(-1,5)
            points,inv=np.unique(np.round(points,7),axis=0,return_inverse=True)
            verts.extend(points[:,:3]);uvs.extend(points[:,3:]);shell.extend(inv.reshape(-1,3)+offset+1);offset+=len(points)
        with (BASE/fname).open('w',encoding='ascii',newline='\n') as f:
            for v in verts:f.write('v '+txt(v)+'\n')
            for uv in uvs:f.write('vt '+txt(uv)+'\n')
            for face in shell:f.write('f '+' '.join(f'{i}/{i}' for i in face)+'\n')
        ET.SubElement(assets,'mesh',name=key,file=fname)
        ET.SubElement(world,'geom',name=key,type='mesh',mesh=key,material=matids[mat],group='0',contype='0',conaffinity='0')
    # Surveyed primitives share the same dimensions and datum as the metalwork.
    # Contact hulls are analytical, with no standalone scene or camera-specific exclusions.
    def addsolid(n,**a):
        ET.SubElement(world,'geom',name='COL_'+n,group='4',contype='2',conaffinity='5',rgba='0 0 0 0',**a)
    for n,(l,h) in static.items():
        l,h=map(np.asarray,(l,h));addsolid(n,type='box',pos=txt((l+h)/2),size=txt((h-l)/2))
    for n,(a,b,r) in pipe.items():addsolid(n,type='cylinder',fromto=txt([*a,*b]),size=txt([r+.004]))
    a=FACE;b=FACE+AXIS*.223
    addsolid('flange',type='cylinder',fromto=txt([*a,*b]),size='.1075')
    a=FACE+AXIS*.20;b=np.array([3.813,.754,1.49])
    addsolid('bracket',type='capsule',fromto=txt([*a,*b]),size='.042')
    xml=ET.tostring(root,encoding='utf-8');(BASE/'cell.xml').write_bytes(xml)
    files={p.name:hashfile(p) for p in sorted(BASE.iterdir()) if p.suffix in ('.xml','.obj','.png')}
    writej(BASE/'manifest.json',dict(meshes=len(groups),parts=len(parts),scene_sha256=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),
      files=files,flange_center=FACE.tolist(),axis=AXIS.tolist(),optical_working_distance_m=STANDOFF,visual_nodes=names,
      cols={'boxes':static,'pipes':pipe},shared_between=['wrist_rgbd','navigation','observer_oblique','observer_side']))
    print('Compiled',len(parts),'parts,',len(groups),'UV-mapped shared meshes',flush=True)
if __name__=='__main__':bake()
