"""M0/M1 Piece and metric-world UV baker for the single shared M2 scene."""
import hashlib,json
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from m0_scene import rgb
from m2_scene import GATES,FACE,AXIS,STANDOFF,CAMERAS,boxes,pipes
from m2_architecture import architecture_parts,architecture_metadata
from m2_flange import flange_parts,flange_metadata

STAGE=Path(__file__).resolve().parents[1]
LOCK=Path(__file__).with_name('m2_collision_lock.json')
def vec(v):return ' '.join('%.9g'%x for x in v)
def texture(folder,family,index,n=768):
    rng=np.random.default_rng(9017+index);w=rng.normal(size=(n,n))
    low=gaussian_filter(w,17,mode='wrap');low/=low.std();mid=gaussian_filter(w,2.2,mode='wrap');mid/=mid.std()
    micro=gaussian_filter(w,(.62,7.2),mode='wrap');micro/=micro.std();y,x=np.indices(w.shape)
    base=216+3*low+2.4*mid+2.8*w
    if family in ('steel','pipe'):base=215+3*low+2.1*mid+4.3*micro+1.2*w
    if family=='deck':
        cell=n/4;line=np.minimum(np.minimum(x%cell,cell-x%cell),np.minimum(y%cell,cell-y%cell))
        base-=14*np.exp(-.5*(line/1.5)**2)+4*np.exp(-.5*(line/3.3)**2)
    for _ in range(150 if family!='concrete' else 240):
        px,py=rng.integers(6,n-40,2);length=int(rng.integers(2,23));delta=float(rng.uniform(7,34))
        if family in ('steel','pipe'):base[py:py+1,px:px+length]-=delta
        else:base[py:py+int(rng.integers(1,4)),px:px+int(rng.integers(2,5))]-=delta
    col=np.stack((base*1.014,base,base*.991),-1);path=folder/(family+'.png');Image.fromarray(np.uint8(col.clip(0,255))).save(path)
    return path

def canonical_collisions():
    lock=json.loads(LOCK.read_text());nodes=lock['geoms'];digest=hashlib.sha256(json.dumps(nodes,sort_keys=True).encode()).hexdigest()
    if digest!=lock['collision_geometry_sha256']:raise ValueError('Collision registration lock checksum changed')
    table={g['name']:g for g in nodes}
    for name,lo,hi,_ in boxes():
        node=table['COL_'+name]
        if not np.allclose(np.fromstring(node['pos'],sep=' '),(lo+hi)/2,atol=5e-9) or not np.allclose(np.fromstring(node['size'],sep=' '),(hi-lo)/2,atol=5e-9):
            raise ValueError('Layout box disagrees with authorized collision lock: '+name)
    for name,a,b,r,_ in pipes():
        node=table['COL_'+name]
        if not np.allclose(np.fromstring(node['fromto'],sep=' '),np.r_[a,b],atol=5e-8) or abs(float(node['size'])-r)>1e-8:raise ValueError('Layout cylinder changed: '+name)
    return lock,table

def enclosure(piece,node,alternatives=None):
    if node['type']=='box':
        local=[]
        for candidate in (alternatives or [node]):
            c=np.fromstring(candidate['pos'],sep=' ');h=np.fromstring(candidate['size'],sep=' ');local.append(np.min(h-abs(piece.v-c),axis=1))
        margin=float(np.min(np.max(local,axis=0)))
    elif node['type']=='cylinder':
        p=np.fromstring(node['fromto'],sep=' ').reshape(2,3);v=p[1]-p[0];length=np.linalg.norm(v);v/=length
        q=piece.v-p[0];z=q@v;r=np.linalg.norm(q-z[:,None]*v,axis=1);margin=float(min(z.min(),length-z.max(),float(node['size'])-r.max()))
    else:raise ValueError(node['type'])
    if margin < -5e-8:raise ValueError('%s/%s extends beyond %s by %.4f mm'%(piece.group,piece.name,node['name'],-1000*margin))
    return margin

def bake(folder=STAGE/'scene'):
    out=Path(folder);out.mkdir(parents=True,exist_ok=True);lock,collisions=canonical_collisions()
    arch=architecture_parts();flange=flange_parts();parts=arch+flange;checks=[]
    for piece in arch:
        family=piece.group.split('_')[0];alternatives=None
        if family in ('gate1','gate3') and piece.name in ('registered aperture liner','machined shoulder','folded satin bezel','teal registered reveal'):
            alternatives=[collisions['COL_'+family+'_'+s] for s in ('sill','lintel','negative','positive')]
        checks.append(dict(group=piece.group,name=piece.name,cover=[n['name'] for n in alternatives] if alternatives else 'COL_'+piece.group,
                           vertex_min_slack_m=enclosure(piece,collisions['COL_'+piece.group],alternatives)))
    flange_info=flange_metadata()
    root=ET.Element('mujoco',model='M2_shared_industrial_corridor');asset=ET.SubElement(root,'asset');world=ET.SubElement(root,'worldbody')
    for j,family in enumerate(('steel','concrete','deck','pipe')):
        texture(out,family,j);ET.SubElement(asset,'texture',name='M2_tex_'+family,type='2d',file=family+'.png')
    pipe_groups={n for n,*_ in pipes()};groups=defaultdict(list)
    for p in parts:
        family='deck' if p.grid else ('pipe' if p.group in pipe_groups else 'concrete' if p.group in ('far_wall','near_guard','terminal_panel','entry_deck','deck') and p.metal<.4 else 'steel')
        color=tuple(np.round(rgb(p.color),4));groups[(p.group,family,color,round(p.opacity,4),round(p.metal,3),round(p.rough,3))].append(p)
    materials={};visual=[]
    for j,((group,family,color,alpha,metal,rough),pieces) in enumerate(groups.items()):
        mk=(family,color,alpha,metal,rough)
        if mk not in materials:
            material='M2_mat_%03d'%len(materials);materials[mk]=material
            ET.SubElement(asset,'material',name=material,texture='M2_tex_'+family,rgba=vec([*color,alpha]),
                shininess='%.4f'%(.13+.5*(1-rough)),specular='%.4f'%(.06+.25*metal))
        key='M2_%s_%03d'%(group,j);file=key+'.obj';verts=[];uvs=[];faces=[];offset=0
        # Original M1 metric-world UV projection and vertex/UV deduplication.
        for p in pieces:
            tri=p.v[p.f];normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);dominant=np.argmax(abs(normal),axis=1);uv=np.empty((len(tri),3,2))
            for axis,(a,b) in enumerate(((1,2),(0,2),(0,1))):
                take=dominant==axis;uv[take]=tri[take][:,:,[a,b]]/(.40 if family=='deck' else .56)
            points=np.concatenate((tri,uv),axis=2).reshape(-1,5);points,inv=np.unique(np.round(points,7),axis=0,return_inverse=True)
            verts.extend(points[:,:3]);uvs.extend(points[:,3:]);faces.extend(inv.reshape(-1,3)+offset+1);offset+=len(points)
        with (out/file).open('w',encoding='ascii',newline='\n') as f:
            for v in verts:f.write('v '+vec(v)+'\n')
            for uv in uvs:f.write('vt '+vec(uv)+'\n')
            for face in faces:f.write('f '+' '.join('%d/%d'%(i,i) for i in face)+'\n')
        ET.SubElement(asset,'mesh',name=key,file=file);ET.SubElement(world,'geom',name=key,type='mesh',mesh=key,material=materials[mk],group='0',contype='0',conaffinity='0');visual.append(key)
    for node in lock['geoms']:ET.SubElement(world,'geom',**node)
    current=[dict(sorted(g.attrib.items())) for g in world if g.get('name','').startswith('COL_')]
    if current!=lock['geoms']:raise AssertionError('Collision attributes or order changed')
    ET.indent(root);ET.ElementTree(root).write(out/'cell.xml',encoding='utf-8',xml_declaration=True)
    visual_policy=dict(legacy_infinite_floor_rgba=[1,1,1,0],studio_skybox_rgb=[1,1,1],legacy_floor_physical_contact_retained=True,
                       industrial_entry_and_main_decks_visible=True,shared_all_cameras=True,observer_cameras=CAMERAS)
    (out/'visual-policy.json').write_text(json.dumps(visual_policy,indent=2),encoding='utf-8')
    declared=sorted({'cell.xml','visual-policy.json'}|{e.get('file') for e in asset if e.get('file')});files={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in declared}
    digest=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    meta=dict(scene_sha256=digest,collision_geometry_sha256=lock['collision_geometry_sha256'],authorized_collision_predecessor=lock['authorized_scene_sha256'],
       collision_count=len(current),visual_meshes=len(visual),parts=len(parts),files=files,gates=GATES,face=FACE.tolist(),axis=AXIS.tolist(),optical_working_distance_m=STANDOFF,
       shared_views=True,view_specific_geometry=False,observer_camera_definitions=CAMERAS,
       observer_camera_sha256=hashlib.sha256(json.dumps(CAMERAS,sort_keys=True).encode()).hexdigest(),
       visual_nodes=visual,architecture=architecture_metadata(),flange=flange_info,
       architecture_enclosure_checks=checks,minimum_architecture_slack_m=min(c['vertex_min_slack_m'] for c in checks),
       source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name('m2_architecture.py'),Path(__file__).with_name('m2_flange.py'),LOCK)})
    (out/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps({k:meta[k] for k in ('scene_sha256','collision_geometry_sha256','collision_count','visual_meshes','parts','minimum_architecture_slack_m')}),flush=True)
    return meta
if __name__=='__main__':
    import sys
    bake(Path(sys.argv[1]) if len(sys.argv)>1 else STAGE/'scene')
