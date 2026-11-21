"""Mitsuba/OptiX softbox render of the same metric cell and archived CAD transforms."""
from collections import defaultdict
import gc
import os
import time
import numpy as np
from PIL import Image
import mitsuba as mi
import drjit as dr
mi.set_variant("cuda_ad_rgb")
from m0_data import QA,WORK,Run,plant,body_poses,matrix,unit,write_json
from m0_scene import inspection_scene,robot_parts,rgb
from m0_render import HERO,GATE_CAM,DOCK_CAM,Renderer
from m0_plates import TIMES

S=mi.ScalarTransform4f
def material(color,metal,rough,opacity,grid):
    col=np.asarray(color);lin=np.where(col<=.04045,col/12.92,((col+.055)/1.055)**2.4)
    albedo=dict(type="rgb",value=lin.tolist())
    if grid:
        i,j=np.indices((512,512));x=np.minimum(i%102.4,102.4-i%102.4);y=np.minimum(j%102.4,102.4-j%102.4)
        minor=np.minimum(x,y)<1.3
        major=np.minimum(np.minimum(i,512-i),np.minimum(j,512-j))<1.8
        val=np.broadcast_to(lin,(512,512,3)).copy()
        val[minor]*=.81;val[major]*=.72
        albedo=dict(type="bitmap",bitmap=mi.Bitmap(val.astype('f4')),raw=True,wrap_mode="repeat")
    bs=dict(type="principled",base_color=albedo,metallic=metal*.90,roughness=max(.26,rough*.88),specular=.46)
    if opacity<1:
        bs=dict(type="blendbsdf",weight=1-np.sqrt(1-opacity),transparent=dict(type="null"),material=bs)
    return mi.load_dict(bs)

def section(p):
    vs=[];ns=[]
    for ids in p.f:
        poly=list(zip(p.v[ids],p.n[ids]));out=[]
        for (a,na),(b,nb) in zip(poly,poly[1:]+poly[:1]):
            if a[1]>=-.22:out.append((a,na))
            if (a[1]>=-.22)!=(b[1]>=-.22):
                u=(-.22-a[1])/(b[1]-a[1]);out.append((a+u*(b-a),na+u*(nb-na)))
        for k in range(1,len(out)-1):
            for pt,n in (out[0],out[k],out[k+1]):vs.append(pt);ns.append(n)
    vs=np.array(vs).reshape(-1,3);ns=np.array(ns).reshape(-1,3)
    return vs,ns,np.arange(len(vs),dtype='i4').reshape(-1,3)

class Studio:
    def __init__(self,m,partset,cut=False):
        self.m=m;static,_=inspection_scene()
        scene={'type':'scene','integrator':{'type':'path','max_depth':9,'rr_depth':5,'hide_emitters':True}}
        # A large diffuse photographic cyclorama catches soft shadows under the raised section.
        scene['backdrop']={'type':'rectangle','to_world':S.translate([1,0,-.095])@S.scale([65,65,1]),
                         'bsdf':{'type':'diffuse','reflectance':{'type':'rgb','value':[.86,.86,.84]}}}
        scene['sky']={'type':'constant','radiance':{'type':'rgb','value':[.61,.68,.74]}}
        for i,(p,at,sz,power) in enumerate((([3.0,-3.9,5.5],[1,0,1.7],[2.75,1.9],[3.4,3.15,2.9]),
              ([-2.5,-.8,4.7],[1,0,1.8],[2.25,1.3],[1.4,1.53,1.67]),
              ([.8,3.1,4.1],[1,0,1.9],[2.5,1.4],[1.6,1.58,1.47]))):
            scene['softbox'+str(i)]={'type':'rectangle','to_world':S.look_at(origin=p,target=at,up=[0,0,1])@S.scale(sz+[1]),
                    'emitter':{'type':'area','radiance':{'type':'rgb','value':power}}}
        groups=defaultdict(list)
        for p in static+partset:
            k=(p.body,tuple(np.round(rgb(p.color),5)),p.metal,p.rough,p.opacity,p.grid)
            pts=(p.v,p.n,p.f)
            if cut and p.group in ('sill','lintel','jambminus','jambplus'):
                pts=section(p)
            if len(pts[2]):groups[k].append(pts)
        bybody={};materials={}
        for j,((body,color,metal,rough,alpha,grid),rows) in enumerate(groups.items()):
            key=(color,metal,rough,alpha,grid)
            if key not in materials:materials[key]=material(*key)
            vertices=[];normals=[];faces=[];nv=0
            for v,n,f in rows:
                vertices.append(v);normals.append(n);faces.append(f+nv);nv+=len(v)
            v=np.concatenate(vertices).astype('f4');n=np.concatenate(normals).astype('f4');fac=np.concatenate(faces).astype('u4')
            props=mi.Properties();props['bsdf']=materials[key]
            name='material_mesh_'+str(j)
            mesh=mi.Mesh(name,len(v),len(fac),props,has_vertex_normals=True,has_vertex_texcoords=bool(grid))
            par=mi.traverse(mesh);par['vertex_positions']=v.ravel();par['vertex_normals']=n.ravel();par['faces']=fac.ravel()
            if grid:par['vertex_texcoords']=np.ascontiguousarray(v[:,:2]*2).ravel()
            par.update()
            if body:
                bybody.setdefault(body,{'type':'shapegroup'})[name]=mesh
            else:scene[name]=mesh
        for body,group in bybody.items():
            key='part_'+str(body);scene[key]=group
            scene['pose_'+str(body)]={'type':'instance','group':{'type':'ref','id':key}}
        self.scene=mi.load_dict(scene);self.params=mi.traverse(self.scene)
        self.ids=list(bybody)
    def pose(self,q):
        if q is not None:R,T,_=body_poses(self.m,np.asarray(q)[None,:])
        for b in self.ids:
            mat=matrix(np.eye(3),[40,40,40]) if q is None else matrix(R[0,b],T[0,b])
            self.params['pose_'+str(b)+'.to_world']=mi.AffineTransform4f(mat)
        self.params.update()

def render(studio,cam,size,spp,seed):
    width,height=size;at=np.asarray(cam.at);eye=at+5*unit(cam.toward_eye)
    tm=S.look_at(origin=eye,target=at,up=[0,0,1])@S.scale([cam.width/2]*2+[1])
    out=np.empty((height,width,3),'f4')
    tile=320
    for y in range(0,height,tile):
        for x in range(0,width,tile):
            w=min(tile,width-x);h=min(tile,height-y)
            sen=mi.load_dict({'type':'orthographic','to_world':tm,'near_clip':.02,'far_clip':30.,
                'sampler':{'type':'independent','sample_count':spp},
                'film':{'type':'hdrfilm','width':width,'height':height,'crop_offset_x':x,'crop_offset_y':y,
                        'crop_width':w,'crop_height':h,'pixel_format':'rgb','rfilter':{'type':'box'}}})
            # Probe a central film ray in full-camera coordinates against the publication projection.
            if x==0 and y==0:
                uv=np.array([.39,.54]);global_uv=(np.array([x,y])+uv*[w,h])/[width,height]
                ray,_=sen.sample_ray(0,.15,mi.Point2f(*map(float,uv)),mi.Point2f(.5,.5))
                right,up,back=cam.axes();expected=at+right*(global_uv[0]-.5)*cam.width-up*(global_uv[1]-.5)*cam.width*height/width
                origin=np.array(ray.o).reshape(3);direction=np.array(ray.d).reshape(3)
                if np.linalg.norm(np.cross(expected-origin,direction))>2e-5:raise ValueError("PBR registration")
            img=mi.render(studio.scene,sensor=sen,spp=spp,seed=seed+17*x+231*y)
            out[y:y+h,x:x+w]=np.array(img)[:,:,:3]
    # Denoise the full stitched image, making the tile boundary invisible.
    den=mi.OptixDenoiser([width,height])
    result=np.array(den(mi.TensorXf(out)))
    result=np.maximum(result*1.24,0)
    result=result/(1.+.48*np.maximum(result-.80,0))
    result=np.where(result<=.0031308,12.92*result,1.055*result**(1/2.4)-.055)
    return Image.fromarray(np.uint8(np.clip(result,0,1)*255+.5))

def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--test',action='store_true');a=ap.parse_args()
    model=plant();r=Renderer(model);parts=robot_parts(model)
    flights={m:Run('26112131-differential-occluded-'+m) for m in ('proposed','strain','fixed','oracle')}
    f=flights['proposed'];studio=Studio(model,parts)
    if a.test:
        studio.pose(f.pose(52));render(studio,DOCK_CAM,(760,456),12,23).save(QA/'ray-flange-test.png');return
    receipt=[];cache=WORK/'studio';cache.mkdir(exist_ok=True)
    size=(2260,1200);spp=24
    studio.pose(None);r.set_pose(None)
    bg=np.asarray(render(studio,HERO,size,spp,41),float);gb=np.asarray(r.render(HERO,size),float)
    result=bg.copy()
    for stamp,alpha in zip(TIMES,(.46,.63,.79,1.)):
        q=f.pose(stamp);studio.pose(q);r.set_pose(q)
        im=np.asarray(render(studio,HERO,size,spp,41),float);gl=np.asarray(r.render(HERO,size),float)
        # Same calibrated cameras. The CAD silhouette and shadow define the matte;
        # Monte Carlo noise in empty space cannot erase earlier physical poses.
        mask=np.max(abs(gl-gb),axis=2)
        weight=np.minimum(1.,mask/1.6)[:,:,None]*alpha
        result=(1-weight)*result+weight*im
        print('Hero',stamp,flush=True)
    Image.fromarray(np.uint8(result.clip(0,255))).save(cache/'hero.png')
    for cut,cam,methods,tag in ((False,DOCK_CAM,('strain','oracle','proposed'),'dock'),
                               (True,GATE_CAM,('strain','fixed','proposed'),'gate')):
        if cut:
            del studio;gc.collect();dr.flush_malloc_cache();studio=Studio(model,parts,True)
        for m in methods:
            flight=flights[m];ti=flight.t[flight.closest()] if cut else 52.;studio.pose(flight.pose(ti))
            im=render(studio,cam,(1060,636),32,73)
            im.save(cache/(tag+'-'+m+'.png'));print(tag,m,flush=True)
            receipt.append(dict(method=m,view=tag,state_s=float(ti),spp=32))
    write_json(QA/'pathtrace.json',dict(renderer='Mitsuba '+mi.__version__+' / CUDA OptiX',
               version=mi.variant(),hero_spp=spp,hero_size=list(size),compared=receipt,projection_error_limit_m=.00002))
if __name__=='__main__':main()
