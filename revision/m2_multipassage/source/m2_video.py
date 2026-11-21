"""Native M2 exposure-synchronous offline replay; stored wrist pixels are never replaced.

Each video row uses its own actual exposure and exact matching physics qpos for
both world cameras. Comparison rows align to the same requested release time;
the exact exposure timestamp and +0.20 s delivery timestamp are always printed.
Ended/missing rows remain explicitly blank, without held synthetic continuation.
"""
import argparse,json,os,shutil,subprocess
from pathlib import Path
import mujoco,numpy as np
from PIL import Image,ImageDraw,ImageFont
from m2_review_data import load_manifest,read
from m2_plant import build,portable,option,STAGE

def font(size,bold=False):
    paths=['C:/Windows/Fonts/timesbd.ttf' if bold else 'C:/Windows/Fonts/times.ttf',
           '/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf']
    for p in paths:
        if Path(p).exists():return ImageFont.truetype(p,size)
    return ImageFont.load_default()
def contain(image,size):
    image=image.copy();image.thumbnail(size,Image.Resampling.LANCZOS);panel=Image.new('RGB',size,'white')
    panel.paste(image,((size[0]-image.width)//2,(size[1]-image.height)//2));return panel
class Replay:
    def __init__(self,records):
        self.records=records;self.plant=portable() if os.name=='nt' else build();self.data=mujoco.MjData(self.plant.m)
        if any(r.scene!=self.plant.scene_digest for r in records):raise ValueError('Replay model is not the frozen flight scene')
        self.renderer=mujoco.Renderer(self.plant.m,height=360,width=640)
    def native(self,qpos,camera):
        self.data.qpos[:]=qpos;mujoco.mj_kinematics(self.plant.m,self.data);mujoco.mj_camlight(self.plant.m,self.data)
        self.renderer.update_scene(self.data,camera=camera,scene_option=option())
        return Image.fromarray(self.renderer.render().copy())
    def row(self,record,t):
        panel=Image.new('RGB',(1920,420),'white');d=ImageDraw.Draw(panel)
        d.text((20,7),record.label+'  |  seed '+str(record.seed),font=font(25,True),fill=record.color)
        exp=record.exposure(t,verify=True,require_tolerance=.026)
        if exp is None:
            d.text((650,180),'No recorded exposure at this release time',font=font(26),fill='#777777');return panel
        q=record.physics['qpos'][exp['physics_index']]
        panel.paste(self.native(q,'observer_oblique'),(0,52));panel.paste(self.native(q,'observer_side'),(640,52))
        with Image.open(exp['rgb_path']) as wrist:panel.paste(contain(wrist.convert('RGB'),(480,360)),(1280,52))
        d=ImageDraw.Draw(panel);d.text((1765,75),'Exposure',font=font(20),fill='#333333');d.text((1765,103),'%.3f s'%exp['t'],font=font(21),fill='#222222')
        d.text((1765,154),'Arrival',font=font(20),fill='#333333');d.text((1765,182),'%.3f s'%exp['arrival_t'],font=font(21),fill='#222222')
        d.text((1765,234),'Frame %d'%exp['frame'],font=font(18),fill='#333333');d.text((1765,273),'Pose + pixels',font=font(18),fill='#333333');d.text((1765,301),'hash verified',font=font(18),fill='#333333')
        return panel
    def frame(self,t):
        # Up to two comparison rows are padded to exactly 1920 x 1080.
        height=max(1080,120+420*len(self.records));im=Image.new('RGB',(1920,height),'white');d=ImageDraw.Draw(im)
        d.text((24,14),'M2 | Recorded native multi-passage flight | release time %.3f s'%t,font=font(29,True),fill='#18252c')
        d.text((24,52),'Oblique world view                         Side world view                         Original wrist RGB-D input',font=font(22),fill='#444444')
        for n,r in enumerate(self.records):im.paste(self.row(r,t),(0,92+420*n))
        return im
    def close(self):self.renderer.close()

def movie(manifest,out,fps=10.,start=None,end=None,frames_only=False):
    records=load_manifest(manifest)
    if not records:raise ValueError('No actual flights in manifest')
    if len(records)>2:raise ValueError('Use at most two rows for the registered 1080p comparison; split larger groups into paired videos')
    replay=Replay(records);out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    first=max(float(r.frame_t[0]) for r in records);last=max(float(r.frame_t[-1]) for r in records)
    start=first if start is None else start;end=last if end is None else end
    if end<start:raise ValueError('Empty replay interval')
    frame_dir=out.parent/(out.stem+'-frames');pipe=None;count=0
    if frames_only:frame_dir.mkdir(exist_ok=True)
    else:
        exe=shutil.which('ffmpeg') or os.environ.get('FFMPEG')
        if not exe:raise RuntimeError('ffmpeg missing; use --frames-only or set FFMPEG')
        pipe=subprocess.Popen([exe,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size','1920x1080','-framerate',str(fps),'-i','-',
          '-an','-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out)],stdin=subprocess.PIPE)
    try:
        for k,t in enumerate(np.arange(start,end+1e-8,1/fps)):
            image=replay.frame(float(t))
            if frames_only:image.save(frame_dir/('%06d.png'%k))
            else:pipe.stdin.write(np.asarray(image,dtype=np.uint8).tobytes())
            count+=1
    finally:
        replay.close()
        if pipe:
            pipe.stdin.close()
            if pipe.wait():raise RuntimeError('ffmpeg encoding failed')
    certificate=dict(output=str(out),video_frames=count,fps=fps,start_s=start,end_s=end,scene_sha256=replay.plant.scene_digest,
      original_wrist_pixels=True,exact_exposure_pose=True,all_rows_same_requested_release_clock=True,
      sources=[dict(flight=str(r.root),verified_frames=len(r.verified),available_frames=len(r.frames),status=r.status()) for r in records])
    (out.parent/(out.stem+'-sync.json')).write_text(json.dumps(certificate,indent=2),encoding='utf-8')
    return certificate
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('manifest',type=Path);a.add_argument('output',type=Path);a.add_argument('--fps',type=float,default=10);a.add_argument('--start',type=float);a.add_argument('--end',type=float);a.add_argument('--frames-only',action='store_true');v=a.parse_args()
    print(json.dumps(movie(v.manifest,v.output,v.fps,v.start,v.end,v.frames_only)))
