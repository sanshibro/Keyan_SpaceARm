"""Render registered wrist exposures and observer views on the same native MjModel."""
import argparse,hashlib,json,math,os,shutil,subprocess
from pathlib import Path
from functools import lru_cache
import mujoco,numpy as np
from PIL import Image,ImageDraw,ImageFont
from m1_replay import build,option,writej
from m1_layout import SEEDS,METHODS,LABELS
from m1_sections import Sections

@lru_cache(maxsize=32)
def face(size,bold=False,serif=False):
    name=('DejaVuSerif' if serif else 'DejaVuSans')+('-Bold' if bold else '')+'.ttf'
    path=Path('/usr/share/fonts/truetype/dejavu')/name
    if not path.exists():
        import matplotlib
        path=Path(matplotlib.get_data_path())/'fonts/ttf'/name
    return ImageFont.truetype(str(path),size)
def title(draw,xy,s,n,b=False,serif=False,col='#233b3b'):draw.text(xy,s,font=face(n,b,serif),fill=col)
class Flight:
    def __init__(self,case,scene):
        self.case=case;self.flat=case/'flight';self.name=case.name.rsplit('-',1)[0]
        r=json.loads((self.flat/'run.json').read_text());pr=json.loads((self.flat/'provenance.json').read_text());assert pr['mission']['scene']==scene
        f=np.load(self.flat/'physics.npz');c=np.load(self.flat/'control.npz');self.t=(f['us']-r['release_us'])*1.e-6
        self.q=f['qpos'];self.tip=f['tcp'];self.motor=f['rotor_N'];self.control_t=c['t'];self.phase=c['phase'];self.hold=c['hold']
        self.index=[json.loads(x) for x in (self.flat/'optics/index.jsonl').read_text().splitlines()]
        self.opt_t=np.array([x['t'] for x in self.index])
        self.c=case/'audit/metrics.json';self.metrics=json.loads(self.c.read_text()) if self.c.exists() else {}
    def nearest(self,t):
        i=int(np.argmin(abs(self.opt_t-t)));o=self.index[i];j=int(np.argmin(abs(self.t-o['t'])))
        assert abs(self.t[j]-o['t'])<1e-6 and hashlib.sha256(np.asarray(self.q[j],np.float64).tobytes()).hexdigest()==o['sensor_qpos_sha256']
        return o,j
    def wrist(self,o):
        path=self.flat/'optics'/o['rgb'];assert hashlib.sha256(path.read_bytes()).hexdigest()==o['rgb_sha256']
        return Image.open(path).convert('RGB')
class Visual:
    def __init__(self):
        p=build();self.p=p;self.opt=option()
        p.m.vis.global_.offwidth=2560;p.m.vis.global_.offheight=1440
        self.r=mujoco.Renderer(p.m,height=1440,width=2560);self.small=mujoco.Renderer(p.m,height=486,width=864)
    def data(self,q,t=0):
        d=self.p.d;d.eq_active[0]=0;d.qpos[:]=q;d.time=t;mujoco.mj_forward(self.p.m,d)
    def freeview(self,at,angle,elev,distance):
        cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE;cam.lookat[:]=at
        cam.azimuth=angle;cam.elevation=elev;cam.distance=distance
        return cam
    def shot(self,q,cam='observer_oblique',mask=False,small=False):
        self.data(q);r=self.small if small else self.r
        r.update_scene(self.p.d,camera=cam,scene_option=self.opt);pic=r.render().copy()
        if not mask:return Image.fromarray(pic)
        r.enable_segmentation_rendering();seg=r.render().copy();r.disable_segmentation_rendering()
        ids=seg[:,:,0];valid=(seg[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))&(ids>=0)
        moving=np.zeros(ids.shape,bool);moving[valid]=self.p.m.geom_bodyid[ids[valid]]!=0
        return Image.fromarray(pic),Image.fromarray(np.uint8(moving)*255)
    def project(self,points,camera):
        c=self.p.d.camera(camera);pos=c.xpos;R=c.xmat.reshape(3,3);v=(points-pos)@R
        f=405./math.tan(math.radians(self.p.m.cam_fovy[self.p.m.camera(camera).id])/2)
        return np.c_[720+f*v[:,0]/(-v[:,2]),405-f*v[:,1]/(-v[:,2])]
    def close(self):self.r.close();self.small.close()

def hero_and_sections(v,flights,out):
    f=flights['coupled'];t=f.t;scored=np.flatnonzero(t>=21)
    required=[0.,1.0,1.85,3.26]
    indices=[scored[np.argmin(abs(f.q[scored,0]-x))] for x in required]
    use=np.flatnonzero(f.phase=='inspect')
    if len(use):
        stamp=f.metrics.get('first_pass_s') or f.control_t[use[-1]]
        indices[-1]=int(np.argmin(abs(t-stamp)))
    q=f.q[indices[0]].copy();q[0]=-30.;background=v.shot(q)
    img=background.copy()
    for j,alpha in zip(indices,(.52,.67,.83,1.)):
        rgb,mask=v.shot(f.q[j],mask=True)
        # Native segmentation can assign glazing to a pixel even when it visibly
        # transmits the aircraft. Recover the transmitted pixels and their shadows
        # from the identical illuminated scene at the same viewpoint.
        diff=np.max(np.abs(np.asarray(rgb,dtype=int)-np.asarray(background,dtype=int)),axis=2)
        opacity=np.maximum(np.asarray(mask)/255.,np.clip(diff/5.,0.,1.))
        img.paste(rgb,(0,0),Image.fromarray(np.uint8(255*alpha*opacity)))
    # Both the mesh visibility mask and the transmitted intensity come from the native scene.
    img.save(out/'hero-native.png')
    from m1_layout import FACE,GATE_X,GATE_HALF,LINTEL,pipes
    landmark_points=np.asarray([[GATE_X,-GATE_HALF,LINTEL],pipes()['riser_return'][0],FACE])
    writej(out/'projection.json',dict(size=img.size,logical_viewport=[1440,810],landmarks=v.project(landmark_points,'observer_oblique'),poses=[dict(t=t[j],centre=v.project(f.q[j:j+1,:3],'observer_oblique')[0]) for j in indices],
        body=v.project(f.q[scored[::3],:3],'observer_oblique'),tool=v.project(f.tip[scored[::3]],'observer_oblique'),scene=v.p.scene_digest))
    axis_targets=[('beam',[.94,-.02,2.035],123,-12,1.75),('riser',[1.65,-.07,2.07],66,-16,1.74),
                  ('flange',[3.40,.11,1.90],116,-18,1.62)]
    records={};cuts={};slicer=Sections(v.p)
    for panel,center,a,e,d in axis_targets:
        cam='observer_dock' if panel=='flange' else v.freeview(center,a,e,d);records[panel]={};cuts[panel]={}
        for method,flight in flights.items():
            sampled=np.flatnonzero(flight.t>=21);x=center[0] if panel!='flange' else 3.26
            j=sampled[np.argmin(abs(flight.q[sampled,0]-x))]
            if panel=='flange' and flight.metrics.get('first_pass_s') is not None:
                j=np.argmin(abs(flight.t-flight.metrics['first_pass_s']))
            o,j=flight.nearest(flight.t[j]);obs=v.shot(flight.q[j],cam)
            obs.save(out/(panel+'-'+method+'.jpg'),quality=94)
            flight.wrist(o).save(out/(panel+'-'+method+'-wrist.png'))
            records[panel][method]=dict(t=o['t'],arrival=o['arrival_t'],x=flight.q[j,0],sha256=o['rgb_sha256'])
            plane={'beam':1.,'riser':1.65,'flange':3.04}[panel]
            jc=sampled[np.argmin(abs(flight.q[sampled,0]-plane))]
            oc,jc=flight.nearest(flight.t[jc]);cuts[panel][method]=dict(plane=plane,t=oc['t'],parts=slicer.cut(flight.q[jc],plane))
    writej(out/'sections.json',records)
    writej(out/'crosssections.json',cuts)
def movie(v,flights,out):
    encoder=shutil.which('ffmpeg')
    if not encoder:
        import imageio_ffmpeg
        encoder=imageio_ffmpeg.get_ffmpeg_exe()
    proc=subprocess.Popen([encoder,'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r','20','-i','-',
                   '-an','-c:v','libx264','-threads','2','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'m1_multiview.mp4')],stdin=subprocess.PIPE)
    chapters=[('I   Low service opening','straight',1.00),('II   Offset riser and instrument tap','fixed',1.53),
              ('III   Turning bay','sequential',2.34),('IV   Oblique flange inspection','unregularized',3.24)]
    zoom={'straight':v.freeview([.94,-.02,2.035],123,-12,1.75),'fixed':v.freeview([1.65,-.07,2.07],66,-16,1.74),
          'sequential':v.freeview([2.18,-.09,2.06],115,-15,1.83),'unregularized':'observer_dock'}
    f=flights['coupled'];capture=[]
    for name,other,station in chapters:
        ids=np.flatnonzero(f.t>=21);ref=f.t[ids[np.argmin(abs(f.q[ids,0]-station))]]
        if station>3:ref=f.metrics.get('first_pass_s') or ref
        # One common release clock per chapter; below-stream end is explicitly labelled.
        for k in range(108):
            timepoint=max(14.,ref-4.0)+k*.08
            page=Image.new('RGB',(1920,1080),'#fafbf8');draw=ImageDraw.Draw(page)
            title(draw,(36,22),name,36,True,True);title(draw,(1410,33),'PAIRED FLIGHT   /   SEED '+str(SEEDS[0]),18)
            for col,key in enumerate((other,'coupled')):
                flight=flights[key];o,j=flight.nearest(timepoint);xx=32+col*944
                title(draw,(xx,93),LABELS[key],27,True)
                wide=v.shot(flight.q[j],'observer_oblique',small=True).resize((894,503),Image.Resampling.LANCZOS if hasattr(Image,'Resampling') else Image.LANCZOS)
                page.paste(wide,(xx,139))
                wrist=flight.wrist(o).resize((456,342));page.paste(wrist,(xx,686))
                title(draw,(xx,651),'Wrist RGB-D   /   original exposure',18)
                title(draw,(xx+479,651),'Fixed inspection view',18)
                close=v.shot(flight.q[j],zoom[other],small=True).resize((415,233))
                page.paste(close,(xx+479,686))
                title(draw,(xx+479,928),'Exposure %.2f s   /   delivery %.2f s'%(o['t'],o['arrival_t']),17)
                title(draw,(xx+479,956),'Body x  %.2f m'%flight.q[j,0],17)
                phase=flight.phase[np.argmin(abs(flight.control_t-o['t']))].replace('_',' ')
                title(draw,(xx+479,984),phase,18,True)
                if timepoint>flight.opt_t[-1]+.04:title(draw,(xx+479,1010),'Final recorded exposure',15,col='#95623e')
                capture.append((name,k,key,o['frame'],o['t']))
            proc.stdin.write(np.asarray(page).tobytes())
    proc.stdin.close()
    if proc.wait()!=0:raise RuntimeError('ffmpeg')
    writej(out/'video-index.json',capture)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('folder',type=Path);a.add_argument('out',type=Path);a.add_argument('--movie',action='store_true')
    vrs=a.parse_args();out=vrs.out;out.mkdir(parents=True,exist_ok=True);v=Visual()
    flights={m:Flight(vrs.folder/(m+'-'+str(SEEDS[0])),v.p.scene_digest) for m in METHODS}
    try:
        hero_and_sections(v,flights,out)
        if vrs.movie:movie(v,flights,out)
    finally:v.close()
