"""Render the registered review group and verify the actual wrist input in the encoded movie."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import imageio_ffmpeg
import numpy as np
from PIL import Image

from gatereview_visuals import METHODS, WristFrames


def write_json(path,data):
    part=path.with_suffix(path.suffix+'.part')
    part.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    part.replace(path)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for x in iter(lambda:f.read(1<<20),b''):
            h.update(x)
    return h.hexdigest()


def verify_movie(stage,cases,start,end,fps):
    movie=stage/'review/gate_multiview_comparison.mp4'
    streams={m:WristFrames(stage/'formal'/c) for m,c in cases.items()}
    counts={m:0 for m in METHODS};errors={m:[] for m in METHODS};times=[]
    reader=imageio_ffmpeg.read_frames(str(movie),pix_fmt='rgb24')
    meta=next(reader);width,height=meta['size']
    if (width,height)!=(1500,950) or abs(meta['fps']-fps)>.001:
        raise ValueError('Unexpected review video dimensions or rate')
    qa=stage/'qa/final';qa.mkdir(parents=True,exist_ok=True)
    checkpoints={round((t-start)*fps):t for t in [26,30,32.25,33.5,43,52]}
    samples={}
    try:
        for i,buffer in enumerate(reader):
            stamp=start+i/fps;times.append(stamp)
            rgb=np.frombuffer(buffer,dtype=np.uint8).reshape(height,width,3)
            sample={}
            for j,m in enumerate(METHODS):
                row=streams[m].at(stamp);raw=streams[m].image(row);raw.thumbnail((292,165))
                x=16+j*298+(292-raw.width)//2;y=501
                encoded=rgb[y+3:y+raw.height-3,x+3:x+raw.width-3]
                original=np.asarray(raw,dtype=np.uint8)[3:-3,3:-3]
                mae=float(np.abs(encoded.astype(np.int16)-original).mean())
                errors[m].append(mae)
                if mae>10:raise ValueError(f'{m} wrist tile differs at {stamp:.3f}s ({mae:.3f}/255)')
                if row['mask_fraction']>=1.-1.e-8:
                    if original.max()!=0 or encoded.mean()>1.:raise ValueError('Encoded full-black wrist tile differs')
                    counts[m]+=1
                if i in checkpoints:
                    sample[m]={'exposure_s':row['t'],'arrival_s':row['arrival_t'],'rgb_sha256':row['rgb_sha256'],'mae_levels':mae}
            if i in checkpoints:
                key=f'{stamp:.2f}'
                samples[key]=sample
                im=Image.fromarray(rgb);im.save(qa/('video-'+key+'.png'))
    finally:
        reader.close()
    if len(times)!=round((end-start)*fps)+1 or any(n!=4*fps for n in counts.values()):
        raise ValueError(f'Video frame count or shutter coverage changed: {len(times)}, {counts}')
    return {'file':movie.name,'sha256':sha(movie),'bytes':movie.stat().st_size,'resolution':[width,height],
            'fps':fps,'displayed_start_s':start,'displayed_end_s':end,'frames':len(times),
            'verified_wrist_tiles':len(times)*len(METHODS),'full_black_video_frames_by_method':counts,
            'wrist_mae_8bit':{m:{'mean':float(np.mean(v)),'max':float(np.max(v))} for m,v in errors.items()},
            'checkpoints':samples}


def run(stage,protocol,registered):
    qa=stage/'qa';review=stage/'review';qa.mkdir(exist_ok=True);review.mkdir(exist_ok=True)
    model=stage/'records/registered-cad-assets.tgz'
    cameras=qa/'cameras.json';write_json(cameras,protocol['review_cameras'])
    case_map={r['method']:r['case'] for r in registered}
    for case in case_map.values():
        op=stage/'formal'/case/'optics'
        verified=json.loads((op/'pixels-verified.json').read_bytes())
        if verified['index_sha256']!=sha(op/'index.jsonl') or verified['images_verified']!=2*verified['frames']:
            raise ValueError('RGB-D archive not verified for '+case)
    source=Path(__file__).parent/'gatereview_visuals.py';start,end,fps=23,55,8
    env={**os.environ,'GATE_REGISTERED_CAD':str(model),'GATE_FFMPEG':imageio_ffmpeg.get_ffmpeg_exe()}
    epochs=[str(t) for t in protocol['review_epochs_s']]
    cmd=[sys.executable,str(source),'--root',str(stage),'--seed',str(protocol['representative_seed']),
         '--material','differential','--vision','occluded','--out',str(review),'--curves','--barcode','--keyframes','--video',
         '--model-module','gate_portable_model','--cameras',str(cameras),'--epochs',*epochs,
         '--start',str(start),'--end',str(end),'--fps',str(fps)]
    subprocess.run(cmd,check=True,env=env)
    proof=verify_movie(stage,case_map,start,end,fps)
    proof['renderer_sha256']=sha(source);proof['model_archive_sha256']=sha(model)
    proof['cases']=case_map;write_json(review/'video-verification.json',proof)
    print('Rendered five paired flights and verified',proof['verified_wrist_tiles'],'delivered camera views.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',type=Path);p.add_argument('--watch',action='store_true')
    args=p.parse_args();stage=args.stage.resolve();spec=json.loads((stage/'protocol.json').read_bytes())
    registered=[x for x in spec['cases'] if x['seed']==spec['representative_seed'] and
                (x['material'],x['vision'])==('differential','occluded')]
    assert {x['method'] for x in registered}==set(METHODS)
    while True:
        ready=sum((stage/'formal'/x['case']/'optics/pixels-verified.json').is_file() for x in registered)
        if ready==len(METHODS):break
        if not args.watch:raise SystemExit(f'Representative RGB-D streams verified: {ready}/5')
        time.sleep(45)
    run(stage,spec,registered)

if __name__=='__main__':
    main()
