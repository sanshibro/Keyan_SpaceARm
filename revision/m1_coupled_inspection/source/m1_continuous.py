"""Continuous registered first- and third-person flight, from approach to inspection."""
import argparse,shutil,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from m1_media import Visual,Flight,title
from m1_layout import SEEDS
from m1_replay import writej
def run(data,out):
    v=Visual();flight=Flight(data/('coupled-'+str(SEEDS[0])),v.p.scene_digest)
    with np.load(flight.case/'audit/series.npz') as s:
        dat={k:s[k] for k in ('t','actual_b','phase','target_error','target_angle','zcam')}
    ff=shutil.which('ffmpeg')
    if not ff:
        import imageio_ffmpeg
        ff=imageio_ffmpeg.get_ffmpeg_exe()
    cmd=[ff,'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r','20','-i','-','-an',
         '-c:v','libx264','-threads','4','-preset','fast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'m1_continuous.mp4')]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE);rows=[]
    try:
        first=flight.opt_t[0];last=flight.opt_t[-1]
        for t in np.arange(first,last+.001,.10):
            o,j=flight.nearest(t);n=int(np.argmin(abs(dat['t']-o['t'])))
            page=Image.new('RGB',(1920,1080),'#fafbf9');d=ImageDraw.Draw(page)
            title(d,(37,30),'Industrial gallery inspection',41,True,True);title(d,(1340,54),'Continuous flight    /    2× replay',24)
            title(d,(37,144),'External observer',27,True);title(d,(1210,144),'Physical wrist camera',27,True)
            world=v.shot(flight.q[j],'observer_oblique',small=True);page.paste(world.resize((1147,645),Image.Resampling.LANCZOS),(32,195))
            wrist=flight.wrist(o);page.paste(wrist.resize((662,496),Image.Resampling.LANCZOS),(1210,225))
            b=dat['actual_b'][n];magnitude=np.rad2deg(np.linalg.norm(b));plane=np.rad2deg(np.arctan2(b[1],b[0]))
            title(d,(37,877),'Aircraft    x %.2f    y %.2f    z %.2f m'%tuple(flight.q[j,:3]),25)
            title(d,(37,926),'Arm    θ %.1f°'%magnitude+('    φ %.1f°'%plane if magnitude>=8. else ''),25)
            title(d,(1210,760),'Exposure %.2f s   /   delivery %.2f s'%(o['t'],o['arrival_t']),22)
            phase=str(dat['phase'][n]);title(d,(1210,823),phase.replace('_',' '),25,True)
            if phase=='inspect':
                title(d,(1210,881),'Tool error    %.2f cm   /   %.2f°'%(100*dat['target_error'][n],dat['target_angle'][n]),23)
                title(d,(1210,925),'Optical distance    %.1f cm'%(100*dat['zcam'][n]),23)
            proc.stdin.write(np.asarray(page).tobytes());rows.append([o['frame'],o['t'],o['arrival_t'],o['rgb_sha256']])
        proc.stdin.close()
        if proc.wait():raise RuntimeError('ffmpeg encode failed')
    finally:
        if proc.poll() is None:proc.terminate()
        v.close()
    writej(out/'continuous-index.json',{'playback':2.,'fps':20,'seed':SEEDS[0],'exposures':rows})
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('folder',type=Path);a.add_argument('out',type=Path);v=a.parse_args();run(v.folder,v.out)
