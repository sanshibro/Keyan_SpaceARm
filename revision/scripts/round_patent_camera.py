"""Replay a moving partial occlusion through native ORB-SLAM3 without passing its schedule to the estimator."""
import hashlib,json,os,subprocess
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from round_wrist_fusion import CASES,readj,savej
from round_shape_pixels import OUTPUT,ROOT
from round_wrist_run import build
OUT=ROOT/'revision/data/round_patent_confidence'
OLD=OUTPUT/'slam/textured'
INPUT=OUTPUT/'rgbd-inputs/textured'
PILOT='241118-online_pause'
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def exposure(t):
    if 31-1e-7<=t<31.32-1e-7:return 1.
    if 26<=t<31:return .88*min(1.,(t-26)/1.1)
    return 0.
def images(case):
    src=INPUT/case;rows=[]
    for line in (src/'frames.txt').read_text().splitlines():
        ts,rgb,depth=line.split();rows.append((float(ts),src/rgb,src/depth))
    return rows
def altered(rgb,dep,frac):
    x=round(rgb.shape[1]*frac)
    if x:
        rgb[:,:x]=0;dep[:,:x]=0
    return rgb,dep
def run():
    temp=Path('/tmp/g1_wrist_orb_20261119');orb=Path.home()/'桌面/Test/ORB_SLAM3';exe=build(orb,temp);cv2.setNumThreads(1)
    OUT.mkdir(exist_ok=True)
    for case in CASES:
        dest=OUT/'camera'/case;dest.mkdir(parents=True,exist_ok=True)
        if (dest/'graded.json').is_file() and (dest/'graded.csv').is_file():continue
        work=OUT/'inputs'/case;work.mkdir(parents=True,exist_ok=True)
        text=[];samples=[];samples_t=[];key_t=[24.,28.,31.12,37.];count=0;digest=hashlib.sha256()
        for i,(t,rgb,depth) in enumerate(images(case)):
            val=exposure(t)
            if val>0:
                cr,cd=altered(cv2.imread(str(rgb)),cv2.imread(str(depth),cv2.IMREAD_UNCHANGED),val)
                rp=work/('r%04d.png'%i);dp=work/depth.name
                assert cv2.imwrite(str(rp),cr) and cv2.imwrite(str(dp),cd);count+=1
            else:
                rp=work/rgb.name;dp=work/depth.name
                for a,b in [(rp,rgb),(dp,depth)]:
                    if not a.exists():os.link(b,a)
            text.append('%.6f %s %s'%(t,rp.name,dp.name))
            digest.update(rp.read_bytes());digest.update(dp.read_bytes())
            if key_t and abs(t-key_t[0])<=.02001:
                samples.append(cv2.cvtColor(cv2.imread(str(rp)),cv2.COLOR_BGR2RGB));samples_t.append(t);key_t.pop(0)
        (work/'frames.txt').write_text('\n'.join(text)+'\n')
        np.savez_compressed(dest/'samples.npz',t=samples_t,rgb=samples)
        pending=dest/'tracking.tmp';csv=dest/'graded.csv'
        with (dest/'graded.log').open('w') as out:
            rc=subprocess.run([str(exe),str(orb/'Vocabulary/ORBvoc.txt'),str(INPUT/case/'settings.yaml'),str(work),
                               'natural',str(pending)],cwd=work,stdout=out,stderr=subprocess.STDOUT,timeout=420)
        rows=pd.read_csv(pending);assert rc.returncode==0 and len(rows)==len(text)
        pending.replace(csv);savej(dest/'graded.json',dict(frames=len(text),affected_frames=count,input_sha256=digest.hexdigest(),
            tracking_sha256=sha(csv),binary_sha256=sha(exe),native_binary_condition='natural on physically masked RGB and depth',
            complete_blackout_capture=[31,31.32],source=case))
        print(case,'frames',len(rows),'map ids',rows.map_id.value_counts().to_dict(),flush=True)
if __name__=='__main__':run()
