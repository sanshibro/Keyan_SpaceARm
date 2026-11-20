"""Evaluate static-view overlap on a dedicated development flight."""
import argparse
import json
import tempfile
from pathlib import Path
import cv2
import numpy as np
from gate_feedback_model import OUT, make_plant, dump
from gate_live_optics import SensorCanvas,AnchoredRGBD

def check(case,report):
    p=make_plant(canvas=True)
    raw=np.load(case/'physics.npz')
    run=json.loads((case/'run.json').read_text())
    t=(raw['us']-run['release_us'])*1e-6
    images=[]
    anchors=[17.8,19.4,21.3,23.5,25.0,26.2,27.0,27.56]
    restored=[32.04,32.32,33.,34.,36.,39.,42.,46.,52.]
    with tempfile.TemporaryDirectory() as tmp:
        sensor=SensorCanvas(p,tmp)
        estimator=AnchoredRGBD(sensor.K)
        fs={}
        try:
            for stamp in anchors+restored:
                i=np.argmin(abs(t-stamp));im,z=sensor.capture(t[i],raw['qpos'][i])
                if 26.4<stamp<28:
                    k=round(im.shape[1]*.68*(stamp-26.4)/1.6);im[:,:k]=0;z[:,:k]=0
                gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
                fs[stamp]=estimator.features(gray,z)
                images.append(cv2.resize(im,(180,135),interpolation=cv2.INTER_AREA))
        finally:sensor.close()
        results={}
        for b in restored:
            fits=[]
            for a in anchors:
                f=estimator.fit(fs[a],fs[b]) if fs[a] is not None and fs[b] is not None else None
                if f is not None:fits.append(dict(reference=a,inliers=f[1]['inliers']))
            results[str(b)]=fits
        dump(report,dict(development_case=case.name,anchor_times=anchors,overlap=results))
        print(results)
        img=images+[np.full_like(images[0],255)]*((-len(images))%5)
        panel=np.vstack([np.hstack(img[i:i+5]) for i in range(0,len(img),5)])
        cv2.imwrite(str(report.with_suffix('.jpg')),panel)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('case',type=Path);a.add_argument('report',type=Path);args=a.parse_args();check(args.case,args.report)
