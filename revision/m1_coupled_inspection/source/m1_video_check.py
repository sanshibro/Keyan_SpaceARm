"""Decode the actual delivered films and compare the wrist panels with their source exposures."""
import hashlib,json
from pathlib import Path
import imageio_ffmpeg
import numpy as np
from PIL import Image
from m1_plates import ROOT,FILM,DATA,SEED
def read(p):return json.loads(p.read_text(encoding='utf8'))
def checks(film,index,chosen,continuous):
    counted,seconds=imageio_ffmpeg.count_frames_and_secs(str(film));frames=imageio_ffmpeg.read_frames(str(film),pix_fmt='rgb24')
    meta=next(frames);assert tuple(meta['size'])==(1920,1080) and abs(meta['fps']-20)<1e-6
    assert counted==(len(index) if continuous else len(index)//2)
    logs={};samples=[]
    try:
        for n,raw in enumerate(frames):
            if n not in chosen:continue
            pic=np.frombuffer(raw,np.uint8).reshape(1080,1920,3)
            if continuous:shots=[('coupled',index[n][0],index[n][1],(1210,225,662,496),Image.Resampling.LANCZOS)]
            else:
                shots=[]
                for col,(_,_,method,frame,t) in enumerate(index[2*n:2*n+2]):shots.append((method,frame,t,(32+944*col,686,456,342),None))
            for method,frame,t,(x,y,w,h),resampling in shots:
                folder=DATA/(method+'-'+str(SEED))/'flight/optics'
                if method not in logs:logs[method]=[json.loads(s) for s in (folder/'index.jsonl').read_text().splitlines()]
                src=logs[method][frame];path=folder/src['rgb']
                assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==src['rgb_sha256'] and abs(t-src['t'])<1e-8
                assert abs(src['arrival_t']-t-.2)<1e-8
                with Image.open(path) as im:
                    expected=im.convert('RGB').resize((w,h),resampling)
                error=np.abs(pic[y:y+h,x:x+w].astype(float)-np.asarray(expected))
                mae=float(error.mean());assert mae<6,(film,n,method,mae)
                samples.append(dict(video_frame=n,method=method,source_frame=frame,exposure=t,delivery=src['arrival_t'],mean_absolute_8bit_error=mae))
            if n==max(chosen):break
    finally:frames.close()
    return {'size':list(meta['size']),'fps':meta['fps'],'frames':counted,'seconds':seconds,'verified_panels':samples}
def main():
    multi=read(FILM/'video-index.json');long=read(FILM/'continuous-index.json')['exposures']
    result={'multi':checks(FILM/'m1_multiview.mp4',multi,[0,54,107,108,201,216,288,324,378,431],False),
            'continuous':checks(FILM/'m1_continuous.mp4',long,np.linspace(0,len(long)-1,11).astype(int).tolist(),True)}
    for p in ('m1_multiview','m1_continuous'):
        assert (FILM/(p+'.mp4')).stat().st_size<20*1024**2
    (ROOT/'qa/video-verification.json').write_text(json.dumps(result,indent=2)+'\n','utf8')
    print([(k,x['frames'],x['seconds'],len(x['verified_panels']),max(s['mean_absolute_8bit_error'] for s in x['verified_panels'])) for k,x in result.items()])
if __name__=='__main__':main()
