"""Exploratory angular-information proxy from the spatial distribution of depth-supported image corners.
The original registered poses, synchronized clocks and predeclared comparison remain unchanged.
"""
import cv2
import numpy as np
from patent_confidence import fuse
from round_patent_camera import OUT, INPUT, PILOT, images
from round_patent_eval import GAUGES, packet, optical
from round_shape_pixels import OUTPUT
from round_wrist_fusion import CASES, metrics, readj, savej

cv2.setNumThreads(1)
def spread(rgb,depth):
    gray=cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
    mask=cv2.erode((depth>0).astype(np.uint8),np.ones((5,5),np.uint8))
    pts=cv2.goodFeaturesToTrack(gray,320,.015,5.,mask=mask)
    if pts is None or len(pts)<4:return 0.
    # The least-supported normalized image direction is insensitive to a rotation of the feature cloud.
    xy=pts[:,0,:]/np.array([gray.shape[1],gray.shape[0]])
    return float(max(0.,np.linalg.eigvalsh(np.cov(xy.T,bias=True))[0]))
def extract(paths):
    return np.array([spread(cv2.imread(str(r)),cv2.imread(str(d),cv2.IMREAD_UNCHANGED)) for _,r,d in paths])
def run():
    folder=OUT/'support';folder.mkdir(exist_ok=True)
    pilot=images(PILOT);ts=np.array([t for t,_,_ in pilot]);cal=extract(pilot);ref=float(np.quantile(cal[(ts>=14)&(ts<46)],.25))
    profile=readj(OUT/'visual-calibration.json')
    savej(folder/'calibration.json',dict(reference=ref,pilot=PILOT,role='Exploratory, motivated by observed partial occlusion. Pilot-only reference; requires a new test flight.'))
    table={};packed={}
    for case in CASES:
        raw=images(case);tcam=np.array([xiv[0] for xiv in raw]);orig=extract(raw)
        inp=OUT/'inputs'/case
        actual=[(float(t),inp/r,inp/d) for t,r,d in [l.split() for l in (inp/'frames.txt').read_text().splitlines()]]
        alt=extract(actual)
        for cond,sp in [('natural',orig),('graded',alt),('occluded',orig)]:
            v=dict(np.load(OUT/'observations'/case/(cond+'.npz')))
            # The legacy native executable blanks long-loss frames internally. No pose from these
            # frames reaches fusion. Compute spatial support only when a registered pose exists.
            if cond=='occluded':sp=np.where(v['cv']>0,sp,np.nan)
            mult=np.nan_to_num(np.clip(sp/ref,0.,1.),nan=0.)
            # Observable proxy uses the actual camera pixels. Neither the tracking status nor the simulated schedule is its input.
            adjusted=v['cv']*mult
            np.savez_compressed(folder/(case+'-'+cond+'.npz'),t=tcam,spread=sp,factor=mult,cv0=v['cv'],cv=adjusted)
            for g in GAUGES:
                strain=dict(np.load(OUT/'observations'/case/(g+'.npz')))
                args=optical(v);args[2]=adjusted
                z=fuse(*packet(strain),*args,profile,mode='both')
                packed.setdefault((cond,g),[]).append((case,strain['t'],z))
    for (cond,g),parts in packed.items():
        table[cond+'/'+g]={}
        for phase,(lo,hi) in {'all':(14,46),'partial':(27.3,31.2),'dark':(31.2,31.52)}.items():
            refdata=[];est=[];pl=[];wt=[]
            for case,t,z in parts:
                keep=(t>=lo-1e-8)&(t<hi-1e-8)
                refdata.append(np.load(OUTPUT/'runs'/case/'scoring_truth.npz')['mocap_b'][keep]);est.append(z['b'][keep])
                pl.append(z['plane'][keep]);wt.extend(z['joint_vis_share'][keep,0])
            out=metrics(np.concatenate(est),np.concatenate(refdata),np.concatenate(pl));w=np.asarray(wt)
            out['joint_share']=float(np.nanmean(w)) if np.isfinite(w).any() else None
            table[cond+'/'+g][phase]=out
    savej(OUT/'support-scores.json',table)
    print('Reference and paired partial-mask diagnostics',ref,
          [(g,table['graded/'+g]['partial']['theta_rmse_deg'],table['graded/'+g]['partial']['joint_share']) for g in GAUGES],flush=True)
if __name__=='__main__':run()
