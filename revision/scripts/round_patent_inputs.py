"""Independently calibrate camera quality and prepare causal sensor packets."""
import numpy as np
import pandas as pd
from round_patent_camera import OUT,OLD,PILOT
from round_slim_box import OUT as SLIM
from round_g1_model import make_plant
from round_wrist_fusion import CASES,DELAY,visual_packets,readj,savej
from round_shape_pixels import OUTPUT
from round_shape_sensors import vector_angles,wrap_deg
from round_patent_variance import observable,predictors,var_eval

def quality(tab,eligible,profile):
    n=tab.matched.to_numpy();f=tab.features.to_numpy();rp=tab.reproj_px.fillna(9.).to_numpy()
    count=np.minimum(1.,n/profile['matched_ref'])
    frac=np.minimum(1.,(n/np.maximum(1,f))/profile['inlier_ref'])
    photo=np.exp(-.5*(rp/profile['pixel_ref'])**2)
    age=0;out=np.zeros(len(tab))
    for i,yes in enumerate(eligible):
        age=age+1 if yes else 0
        if yes:out[i]=np.sqrt(count[i]*frac[i])*photo[i]*min(age/4.,1.)
    return out
def visual(cam):
    folder=OLD/PILOT;tab=pd.read_csv(folder/'natural.csv');z=np.load(folder/'natural-estimate.npz')
    training=(z['capture']>=14)&(z['capture']<46)&np.isfinite(z['visual_b']).all(1)
    ta,pa=vector_angles(z['visual_b']);truth,pp=vector_angles(z['truth_at_capture']);std=tab.loc[training]
    params=dict(matched_ref=float(std.matched.quantile(.75)),inlier_ref=float((std.matched/std.features.clip(1)).quantile(.75)),
         pixel_ref=max(1.,float(std.reproj_px.quantile(.90))),epsilon=.0001,pilot=PILOT)
    cv=quality(tab,np.isfinite(z['visual_b']).all(1),params);use=training&(cv>0);c=cv[use];valid_phi=use&(truth>=5)&(ta>=5)
    params.update(theta_k=max(.2**2,float(np.mean((ta[use]-truth[use])**2*(c+params['epsilon'])))),
        phi_k=max(.2**2,float(np.mean(wrap_deg(pa[valid_phi]-pp[valid_phi])**2*(observable(ta[valid_phi])*cv[valid_phi]+params['epsilon'])))),
        teacher_threshold=float(np.quantile(c,.45)),teacher_pixel_max=float(std.reproj_px.quantile(.9)),calibration_frames=int(use.sum()))
    savej(OUT/'visual-calibration.json',params)
    target=OUT/'observations';target.mkdir(exist_ok=True)
    for case in CASES:
        dest=target/case;dest.mkdir(exist_ok=True)
        for cond in ('natural','graded','occluded','native-long'):
            path=OLD/case/'occluded.csv' if cond=='native-long' else OUT/'camera'/case/(cond+'-registered.csv')
            df=pd.read_csv(path);cap,b,legacy,_=visual_packets(case,path,cam)
            eligible=np.isfinite(b).all(1);cv=quality(df,eligible,params);theta,_=vector_angles(b)
            stable=np.zeros(len(cap),bool);ok=0
            for j in range(len(cap)):
                prior=df.reproj_px.to_numpy()[max(0,j-4):j+1]
                good=cv[j]>=params['teacher_threshold'] and np.isfinite(prior).all() and np.max(prior)<=1.35*params['teacher_pixel_max']
                ok=ok+1 if good else 0;stable[j]=ok>=4
            bridge=np.zeros(len(cap),bool) if cond=='native-long' else np.load(OUT/'camera'/case/(cond+'-map.npz'))['bridged']
            # A newly registered map also needs several confident observations before becoming a teacher.
            edge=np.flatnonzero(bridge&~np.r_[False,bridge[:-1]])
            for i in edge:stable[i:i+8]=False
            np.savez_compressed(dest/(cond+'.npz'),capture=cap,visual_b=b,cv=cv,legacy_q=legacy,stable_teacher=stable,
                                bridge=bridge,matches=df.matched.to_numpy(),reproj=df.reproj_px.to_numpy())
    print('visual calibration',params,flush=True)
def strain():
    model=readj(SLIM/'p30/predictor.json');vm=readj(OUT/'strain-variance.json')
    for case in CASES:
        dest=OUT/'observations'/case;dest.mkdir(parents=True,exist_ok=True)
        for name,key in [('nominal','primary'),('zero_offset','return_offset'),('material_memory','memory/combined')]:
            z=np.load(SLIM/key/case/'strain.npz');t,ts=z['t_tick'],z['t']
            ad=z['adc'];be,X=predictors(ad,z['tare'],ts,model);v=var_eval(vm,X)
            j=np.searchsorted(ts,t+1e-8,side='right')-1;initial=np.maximum(j,0)
            variance=v[initial].copy();variance[j<0]=[5**2,120**2]
            np.savez_compressed(dest/(name+'.npz'),t=t,b=z['raw_b'],fresh=z['fresh'],vs=variance,
                                rail=np.where(j>=0,((ad[initial]<=0)|(ad[initial]>=4095)).any(1),False),age=np.where(j>=0,t-ts[initial],np.inf))
if __name__=='__main__':visual(make_plant());strain()
