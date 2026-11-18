"""Render the archived F flight with four independently animated CAD propellers.

The pump driving this visual is the archived HIL channel, not a motor speed sensor.
Angular speed is deliberately reduced to resolve each blade in the video.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont

from round_f_plant import Plant, ROOT, CHANNELS

F=ROOT/'revision/data/round_f'
RAW=F/'runs/final'
FPS=24
SPEED=2.
TURNS_PER_SIM_SECOND=1.2  # Display rate at unit motor output. Not a measured RPM.
W,H=1280,660
COLOR='#192f3b'; GREY='#4b6270'; TEAL='#17716c'; BLUE='#345c79'; COPPER='#a85534'; LIGHT='#d9e2e4'
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-'
def font(px,bold=False): return ImageFont.truetype(FONT+('Bold' if bold else 'Regular')+'.ttc',px,index=2)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def both(r):
    cs,ss=np.cos(r),np.sin(r)
    return np.array([[cs,-ss,0],[ss,cs,0],[0,0,1.]])


def split_model(plant,folder):
    info=json.loads((ROOT/'revision/data/super_metric_geometry.json').read_text())
    part=info['parts']['prop'];path=ROOT/part['file'];assert sha(path)==part['sha256']
    axes=np.array([r['axis_center_m'] for r in info['rotors']])
    v=[];f=[]
    for line in path.read_text().splitlines():
        if line.startswith('v '):v.append(line[2:])
        elif line.startswith('f '):f.append([int(s)-1 for s in line.split()[1:]])
    coords=np.array([list(map(float,s.split())) for s in v])
    faces=np.asarray(f);assert faces.shape[1]==3
    owner=np.argmin(np.linalg.norm(coords[:,None,:2]-axes[None,:,:2],axis=2),axis=1)
    assert np.all(owner[faces]==owner[faces[:,0],None])
    root=ET.fromstring(plant.xml);asset=root.find('asset');body=root.find(".//body[@name='uav']")
    old=body.find("geom[@name='visual_super_prop']");style=dict(old.attrib);body.remove(old)
    asset.remove(asset.find("mesh[@name='metric_prop']"));stats=[]
    for i in range(4):
        verts=np.flatnonzero(owner==i);tris=faces[owner[faces[:,0]]==i]
        ids=np.searchsorted(verts,tris);assert np.array_equal(verts[ids],tris)
        path_i=folder/('propeller%d.obj'%i)
        with path_i.open('w') as fd:
            fd.writelines('v '+v[j]+'\n' for j in verts)
            for a,b,c in ids+1:fd.write('f %d %d %d\n'%(a,b,c))
        name='f_blades_%d'%i;ET.SubElement(asset,'mesh',name=name,file=str(path_i))
        ET.SubElement(body,'geom',dict(style,name=name,mesh=name))
        stats.append(dict(cad_index=i,vertices=len(verts),triangles=len(tris)))
    m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    assert (m.nq,m.nv,m.nu,m.nbody)==(plant.m.nq,plant.m.nv,plant.m.nu,plant.m.nbody)
    for k in ('body_mass','body_inertia','jnt_qposadr','actuator_gear'):
        np.testing.assert_allclose(getattr(m,k),getattr(plant.m,k),rtol=0,atol=1e-12)
    return m,axes,stats


class View:
    def __init__(self,model,axes):
        self.m=model;self.d=mujoco.MjData(model);self.axes=axes
        self.base=model.body('uav').id
        self.ids={model.geom('f_blades_%d'%m).id:k for k,m in enumerate(CHANNELS)}
        self.render=mujoco.Renderer(model,height=480,width=480)
        self.opt=mujoco.MjvOption();self.opt.geomgroup[0]=0;self.opt.geomgroup[3]=0;self.opt.sitegroup[:]=0
        self.cam=mujoco.MjvCamera();self.cam.orthographic=1;self.cam.distance=1.0
    def draw(self,q,angles,top):
        d=self.d;d.qpos[:]=q;mujoco.mj_forward(self.m,d)
        U=d.xmat[self.base].reshape(3,3);origin=d.xpos[self.base]
        centers=origin+self.axes[CHANNELS]@U.T
        self.cam.lookat[:]=origin+([0,0,0] if top else [.018,0,-.17])
        self.cam.azimuth=90 if top else 116
        self.cam.elevation=-89.99 if top else -21
        self.m.vis.global_.fovy=.435 if top else .71
        self.render.update_scene(d,camera=self.cam,scene_option=self.opt)
        used=set()
        for g in self.render.scene.geoms[:self.render.scene.ngeom]:
            if g.objtype!=int(mujoco.mjtObj.mjOBJ_GEOM) or g.objid not in self.ids:continue
            ch=self.ids[g.objid];used.add(ch)
            spin=U@both(angles[ch])@U.T
            # Rigid render-space rotation about the registered shaft, after MuJoCo mesh recentering.
            g.pos[:]=centers[ch]+spin@(g.pos.astype(float)-centers[ch])
            g.mat[:]=spin@g.mat.astype(float)
        assert used==set(range(4))
        im=Image.fromarray(self.render.render().copy()).convert('RGB')
        left,right=self.render.scene.camera
        eye=(np.array(left.pos)+right.pos)/2
        upp=np.array(left.up);side=np.cross(left.forward,upp)
        span=left.frustum_top-left.frustum_bottom
        xy=np.column_stack((240+480*((centers-eye)@side)/span,240-480*((centers-eye)@upp)/span))
        return im,xy


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    read=lambda p:json.loads(p.read_text())
    rec=read(RAW/'run.json');prov=read(RAW/'provenance.json')
    expected=read(F/'summary.json')['cases'][-1]['sha256']
    for n in ('physics.npz','hil.csv','flight.ulg'):assert sha(RAW/n)==expected[n]
    with np.load(str(RAW/'physics.npz')) as z:us,qpos,stored_rotor=(z[k].copy() for k in ('us','qpos','rotor_N'))
    wire=pd.read_csv(RAW/'hil.csv');tick=wire.us.to_numpy(dtype=np.int64);motor=wire[['m1','m2','m3','m4']].values
    assert np.all(np.diff(tick)==4000) and np.all((motor>=0)&(motor<=1))
    np.testing.assert_allclose(12*motor[np.searchsorted(tick,us)],stored_rotor,atol=1e-6,rtol=0)
    native=Plant();assert hashlib.sha256(native.xml.encode()).hexdigest()==prov['plant_xml_sha256']
    sign=-np.sign(native.m.actuator_gear[native.rotors,5])
    np.testing.assert_equal(sign,[1,1,-1,-1])
    # The archived plant holds each command for the following 4 ms, so use its left endpoint.
    omega=(2*np.pi*TURNS_PER_SIM_SECOND)*np.sqrt(motor)*sign
    phase=np.vstack((np.zeros(4),np.cumsum(omega[:-1]*np.diff(tick)[:,None]/1e6,axis=0)))
    assert np.all(np.diff(phase,axis=0)[motor[:-1]==0]==0)
    release=rec['release_us'];start=release-4000000;end=release+39000000
    frame_us=np.round(np.linspace(start,end,int((end-start)*FPS/(SPEED*1e6))+1)).astype(np.int64)
    def nearest(a,u):
        j=int(np.clip(np.searchsorted(a,u),1,len(a)-1))
        return j if a[j]-u<u-a[j-1] else j-1
    def motor_angles(u):return np.array([np.interp(u,tick,phase[:,i]) for i in range(4)])
    with tempfile.TemporaryDirectory(prefix='f-cad-rotors-') as temp:
        m,axes,stats=split_model(native,Path(temp));view=View(m,axes)
        hold_q=qpos[nearest(us,release+30000000)]
        p0=motor_angles(release+30000000);p1=motor_angles(release+30100000)
        im0,hubs=view.draw(hold_q,p0,True);im1,_=view.draw(hold_q,p1,True)
        delta=np.abs(np.asarray(im0).astype(float)-np.asarray(im1));delta=delta.max(2)>14
        yy,xx=np.indices(delta.shape)
        changed=[]
        for x,y in hubs:
            radius=np.hypot(xx-x,yy-y);changed.append(int((delta&(radius>20)&(radius<86)).sum()))
        assert min(changed)>100,(hubs,changed)
        before,_=view.draw(hold_q,motor_angles(start),True)
        stopped,_=view.draw(hold_q,motor_angles(start+200000),True);assert np.array_equal(before,stopped)
        header,sub,lab,sm,val=font(28,True),font(21,True),font(18),font(15),font(23)
        ff=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s','1280x660',
              '-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p',
              '-movflags','+faststart',str(out/'replay.mp4')],stdin=subprocess.PIPE)
        labels=['右前','左后','左前','右后']
        def pix(e,h):return (round(945+(e+.075)*294),round(349-236*h))
        def txt(b,p,s,ft=lab,fill=COLOR,anchor=None):
            x,y=p
            if anchor:
                width,height=ft.getsize(s)
                x-=width*({'l':0,'m':.5,'r':1}[anchor[0]])
                y-=height*({'t':0,'m':.5,'b':1}[anchor[1]])
            b.text((round(x),round(y)),s,font=ft,fill=fill)
        still=Image.new('RGB',(900,355),'white');pen=ImageDraw.Draw(still)
        txt(pen,(20,7),'右前旋翼 M1 · 连续局部视图',sub)
        for j,dt in enumerate([0.,1/12,2/12]):
            stamp=release+round((30+dt)*1e6);k=nearest(us,stamp)
            img,center=view.draw(qpos[k],motor_angles(stamp),True)
            x,y=np.rint(center[0]).astype(int);r=109
            crop=img.crop((x-r,y-r,x+r,y+r)).resize((238,238),Image.LANCZOS)
            still.paste(crop,(30+300*j,57))
            txt(pen,(150+300*j,311),'t = %.3f s'%(30+dt),lab,BLUE,anchor='mt')
        still.save(out/'detail.png')
        with (out/'rotor-frames.csv').open('w',newline='') as fp:
            wr=csv.writer(fp);wr.writerow(['frame','relative_s','qpos_us','hil_us']+
                    ['motor%d_N'%(k+1) for k in range(4)]+['visual_motor%d_phase_rad'%(k+1) for k in range(4)])
            for j,stamp in enumerate(frame_us):
                t=(stamp-release)/1e6;k=nearest(us,stamp);i=np.searchsorted(tick,stamp,side='right')-1
                w=wire.iloc[i];ang=motor_angles(stamp);th=12*motor[i]
                wr.writerow([j,round(t,6),us[k],tick[i],*th,*ang])
                side,_=view.draw(qpos[k],ang,False);top,hubs=view.draw(qpos[k],ang,True)
                canvas=Image.new('RGB',(W,H),'white');draw=ImageDraw.Draw(canvas)
                txt(draw,(28,14),'悬停弯臂时的旋翼响应',header)
                txt(draw,(951,24),'SUPER  /  Qingyu     ·     PX4 SITL',lab,GREY)
                draw.line((28,69,1250,69),fill=LIGHT,width=2)
                txt(draw,(28,86),'a   整机姿态',sub);txt(draw,(473,86),'b   四个旋翼 · 俯视',sub)
                txt(draw,(930,86),'c   机身轨迹',sub)
                canvas.paste(side.resize((447,447),Image.LANCZOS),(8,119))
                canvas.paste(top.resize((438,438),Image.LANCZOS),(463,123))
                draw=ImageDraw.Draw(canvas)
                for n,(x,y) in enumerate(hubs):
                    hx=463+x*438/480;hy=123+y*438/480
                    # Leader and label lie toward the exterior corner away from the airframe.
                    sx=1 if x>240 else -1;sy=1 if y>240 else -1
                    lx=int(np.clip(hx+sx*70,490,870));ly=int(np.clip(hy+sy*67,145,533))
                    draw.line((hx+sx*9,hy+sy*9,lx,ly),fill=GREY,width=1)
                    draw.rectangle((lx-18,ly-12,lx+18,ly+12),fill='white')
                    txt(draw,(lx,ly),'M%d'%(n+1),sm,BLUE,anchor='mm')
                for h in (0,.3,.6,.9):
                    _,y=pix(0,h);draw.line((952,y,1246,y),fill=LIGHT);txt(draw,(943,y-1),'%.1f'%h,sm,GREY,anchor='rm')
                for e in (0,.3,.6,.9):
                    x,y=pix(e,0);draw.line((x,y,x,y+4),fill=GREY);txt(draw,(x,y+5),'%.1f'%e,sm,GREY,anchor='mt')
                txt(draw,(936,117),'相对高度 (m)',sm,GREY);txt(draw,(1103,375),'向东位移 (m)',sm,GREY,anchor='mt')
                ids=np.arange(np.searchsorted(tick,release),i+1,20)
                ref=[pix(0,.65),pix(.6,.65)];draw.line(ref,fill='#adbac0',width=2)
                if len(ids)>1:draw.line([pix(e,h) for e,h in zip(wire.ex.values[ids],wire.ez.values[ids])],fill=TEAL,width=2)
                x,y=pix(w.rx,w.rz);draw.rectangle((x-4,y-4,x+4,y+4),outline=COPPER,width=2)
                x,y=pix(w.ex,w.ez);draw.ellipse((x-4,y-4,x+4,y+4),fill=TEAL)
                txt(draw,(936,405),'旋翼推力 (N)',sub)
                for n,force in enumerate(th):
                    x=937+(n%2)*160;y=445+(n//2)*45
                    txt(draw,(x,y),'M%d %s'%(n+1,labels[n]),sm,GREY)
                    txt(draw,(x+114,y-3),'%.2f'%force,lab,TEAL,anchor='rt')
                txt(draw,(476,557),'俯视旋向：M1、M2 逆时针，M3、M4 顺时针',sm,GREY)
                draw.line((28,592,1250,592),fill=LIGHT,width=2)
                phase_name={'preflight':'起飞前','liftoff':'起飞','hover':'定点悬停','translate':'向东平移','bend':'悬停弯臂','recover':'恢复悬停'}[w.phase]
                txt(draw,(28,604),'%+.1f s   %s'%(t,phase_name),val)
                txt(draw,(344,608),'臂弯角 %.1f°'%w.net_deg,lab)
                txt(draw,(556,610),'桨叶转速按推力缩放示意；姿态与推力取自 F 轮日志。',lab,GREY)
                txt(draw,(1190,608),'2× 回放',lab,GREY)
                if j==round((30+4)/SPEED*FPS):canvas.save(out/'poster.png')
                ff.stdin.write(canvas.tobytes())
        ff.stdin.close()
        if ff.wait():raise RuntimeError('Video encoding failed')
        view.render.close()
    audit=dict(flight='F / final-full-log',source_sha256={k:expected[k] for k in ('physics.npz','hil.csv','flight.ulg')},
        physics_plant_xml_sha256=prov['plant_xml_sha256'],cad_prop_sha256=sha(ROOT/'revision/generated/super_metric/prop.obj'),
        assembly=stats,all_triangles_accounted_for=sum(s['triangles'] for s in stats),
        channel_to_cad=CHANNELS.tolist(),directions_top_view=['CCW','CCW','CW','CW'],
        shaft_speed_telemetry=False,visual_phase='Integral of signed 2*pi*1.2*sqrt(normalized HIL output) per simulation second; reduced for temporal resolution. Angular phase is illustrative.',
        frames=len(frame_us),fps=FPS,replay_rate=SPEED,interval_relative_s=[-4.,39.],
        max_qpos_alignment_ms=max(abs(us[nearest(us,t)]-t) for t in frame_us)/1000.,
        fixed_pose_zero_thrust_pixels_changed=0,fixed_pose_active_pixels_changed_per_rotor=changed)
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(audit['frames'],changed,audit['max_qpos_alignment_ms'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);args=p.parse_args();run(args.out.resolve())
