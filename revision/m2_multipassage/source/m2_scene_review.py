"""One large shared-scene axonometric plate and three equal-scale static details.

Extra UAVs in the overview are explicit copies of the three independently
certified static poses. No interpolated trajectory or flight is represented.
"""
import argparse,json
from pathlib import Path
import mujoco,numpy as np
from PIL import Image,ImageDraw,ImageFont
from m2_plant import portable,build,place,option,STAGE

FIELDS=('type','dataid','objtype','objid','category','texcoord','segid','matid','size','pos','mat','rgba','emission','specular','shininess','reflectance','modelrbound','transparent','label')
def snapshot(g):return {k:(getattr(g,k).copy() if isinstance(getattr(g,k),np.ndarray) else getattr(g,k)) for k in FIELDS}
def restore(g,source):
    for k,v in source.items():
        if isinstance(v,np.ndarray):getattr(g,k)[:]=v
        else:setattr(g,k,v)
def font(n,bold=False):
    for p in ('C:/Windows/Fonts/timesbd.ttf' if bold else 'C:/Windows/Fonts/times.ttf',
              '/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf'):
        if Path(p).exists():return ImageFont.truetype(p,n)
    return ImageFont.load_default()
def camera(look,dist,azimuth,elevation):
    c=mujoco.MjvCamera();c.type=mujoco.mjtCamera.mjCAMERA_FREE;c.lookat[:]=look;c.distance=dist;c.azimuth=azimuth;c.elevation=elevation;return c
def project(scene,xyz,width,height):
    a,b=scene.camera;pos=(a.pos+b.pos)/2;forward=(a.forward+b.forward)/2;forward/=np.linalg.norm(forward);up=(a.up+b.up)/2;up/=np.linalg.norm(up)
    right=np.cross(forward,up);right/=np.linalg.norm(right);q=np.asarray(xyz)-pos;x=q@right;y=q@up
    if not a.orthographic:
        depth=q@forward;x*=a.frustum_near/depth;y*=a.frustum_near/depth
    extent=(a.frustum_top-a.frustum_bottom);wide=extent*width/height
    return np.array([(x/wide+.5)*width,(.5-y/extent)*height])
def make(out=STAGE/'qa/static',native=False):
    out=Path(out);record=json.loads((out/'preflight.json').read_text());p=build() if native else portable();m,d=p.m,p.d
    original_fovy=m.vis.global_.fovy;original_ortho=m.vis.global_.orthographic
    m.vis.global_.orthographic=True;m.vis.global_.fovy=4.90
    renderer=mujoco.Renderer(m,height=900,width=1600);wide=camera([2.47,0,1.66],8.0,57,-35);copies=[]
    for row in record['gate_witnesses']:
        place(m,d,row['xyz'],row['theta_deg'],row['phi_deg']);renderer.update_scene(d,camera=wide,scene_option=option())
        copies.append([snapshot(g) for g in renderer.scene.geoms[:renderer.scene.ngeom]
                       if int(g.objtype)==int(mujoco.mjtObj.mjOBJ_GEOM) and int(g.objid)>=0 and m.geom_bodyid[int(g.objid)]>0])
    for robot in copies[:-1]:
        for g in robot:
            if renderer.scene.ngeom>=renderer.scene.maxgeom:raise ValueError('Static pose mesh capacity exceeded')
            restore(renderer.scene.geoms[renderer.scene.ngeom],g);renderer.scene.ngeom+=1
    overview=Image.fromarray(renderer.render().copy());draw=ImageDraw.Draw(overview)
    for j,row in enumerate(record['gate_witnesses']):
        uv=project(renderer.scene,np.asarray(row['xyz'])+[0,0,.22],1600,900);cx,cy=map(float,uv)
        draw.ellipse((cx-17,cy-17,cx+17,cy+17),fill='white',outline='#286c75',width=2);draw.text((cx-7,cy-14),str(j+1),font=font(24,True),fill='#286c75')
    overview.save(out/'three-gate-static-overview.png');renderer.close()
    panel=Image.new('RGB',(1600,1440),'white');draw=ImageDraw.Draw(panel)
    draw.text((24,13),'Three registered static feasibility poses',font=font(30,True),fill='#233941')
    draw.text((24,53),'Shared industrial geometry; the three UAV instances denote separate static configurations.',font=font(21),fill='#5e7178')
    panel.paste(overview,(0,82));draw.line((20,980,1580,980),fill='#aebabc',width=1)
    detail=mujoco.Renderer(m,height=324,width=520);m.vis.global_.fovy=1.10;positions=[]
    for j,row in enumerate(record['gate_witnesses']):
        place(m,d,row['xyz'],row['theta_deg'],row['phi_deg'])
        cam=camera(np.asarray(row['xyz'])+[.03,0,-.15],2.3,57,-12)
        detail.update_scene(d,camera=cam,scene_option=option());pixels=Image.fromarray(detail.render().copy())
        pixels.save(out/(row['gate']+'-closeup.png'));x=10+j*535;panel.paste(pixels,(x,1024))
        draw=ImageDraw.Draw(panel);draw.text((x+10,990),'(%s) %s | theta %g deg, phi %g deg'%('abc'[j],row['gate'],row['theta_deg'],row['phi_deg']),font=font(21,True),fill='#263e48')
        ex=row['external'];a=project(detail.scene,ex['a'],520,324);b=project(detail.scene,ex['b'],520,324)
        for uv in (a,b):
            xx,yy=uv
            if 0<=xx<520 and 0<=yy<324:draw.ellipse((x+xx-3,1024+yy-3,x+xx+3,1024+yy+3),fill='#b47c45')
        if all(0<=uv[0]<520 and 0<=uv[1]<324 for uv in (a,b)):
            draw.line((x+a[0],1024+a[1],x+b[0],1024+b[1]),fill='#b47c45',width=2)
        draw.text((x+10,1354),'External >= %.2f mm | hazard >= %.2f mm'%(1e3*ex['lower'],1e3*row['self_hazardous']['lower']),font=font(20),fill='#285d68')
        draw.text((x+10,1383),'Fit >= %.3f mm | identical scale'% (1e3*row['self_fit']['lower']),font=font(19),fill='#5a6d72')
        positions.append(dict(gate=row['gate'],lookat=cam.lookat.tolist(),distance=cam.distance,azimuth=cam.azimuth,elevation=cam.elevation,orthographic_height_m=1.10))
    detail.close();m.vis.global_.fovy=original_fovy;m.vis.global_.orthographic=original_ortho
    panel.save(out/'three-gate-static-review.png')
    meta=dict(scene_sha256=p.scene_digest,physical_geometry_unchanged=True,static_pose_instances=3,simultaneous_flight=False,
      overview_camera=dict(lookat=wide.lookat.tolist(),distance=8,azimuth=57,elevation=-35,orthographic_height_m=4.90),closeups=positions)
    (out/'static-review-composition.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');return meta
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,default=STAGE/'qa/static');a.add_argument('--native',action='store_true');v=a.parse_args();print(json.dumps(make(v.output,v.native)))
