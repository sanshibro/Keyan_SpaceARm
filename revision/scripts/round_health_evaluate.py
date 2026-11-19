"""Run all frozen estimators without labels, then score against the independent marker reconstruction."""
import argparse,hashlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from patent_confidence import fuse
from strain_health import HealthObserver,angle_features,wrap
from round_health_fit import POINT,digest
from round_health_flights import OUT,SPEC
from round_health_data import FLIGHT_STATES
from round_slim_run import features_at
from round_shape_sensors import predict
from round_patent_camera import OUT as PREV
from round_patent_variance import predictors,var_eval
from round_wrist_fusion import metrics,readj,savej,filter_bend,DELAY

CASES=[e['case'] for e in SPEC['flight']]
METHODS=('raw_nominal','strain','vision','legacy','previous','previous_teacher','health','health_teacher')
PHASES={'all':(14,46),'before':(14,31.2),'blind_window':(31.2,35.2),'recovered':(36,46)}
def load_model():
    file=OUT/'model/health-model.npz';meta=readj(OUT/'freeze.json')
    if digest(file)!=meta['model_sha256']:raise ValueError('Frozen model fingerprint mismatch')
    with np.load(file) as z:return HealthObserver(dict(z))
def control(case):
    folder=OUT/'runs'/case;start=readj(folder/'run.json')['release_us']
    with np.load(folder/'physics.npz') as f:t=(f['us']-start)/1e6
    return t[(t>=0)&(t<46.31)]
def to_ticks(t,ts,arr,prior):
    ix=np.searchsorted(ts,t+1e-8,side='right')-1
    output=np.broadcast_to(prior,(len(t),)+arr.shape[1:]).copy().astype(arr.dtype)
    valid=ix>=0;output[valid]=arr[ix[valid]]
    fresh=valid&np.r_[True,np.diff(ix)>0]
    return output,fresh,np.where(valid,t-ts[np.maximum(ix,0)],np.inf)
def visual_output(t,cap,bv):
    i=np.searchsorted(cap+DELAY,t+1e-7,side='right')-1;k=np.maximum(i,0)
    good=(i>=0)&(t-cap[k]<DELAY+.061)
    out=np.full((len(t),2),np.nan);out[good]=bv[k[good]]
    return out,k
def strain_packet(folder,name,t,observer):
    with np.load(folder/(name+'.npz')) as f:raw=dict(f)
    a,ts,tare=raw['adc'],raw['t'],raw['tare']
    prev_b,X=predictors(a,tare,ts,POINT);prev_var=var_eval(readj(PREV/'strain-variance.json'),X)
    z=observer.estimate(ts,a,tare,prev_b);prior=predict(POINT,features_at(raw['initial_adc'][None,:],tare))[0]
    b,fresh,age=to_ticks(t,ts,z['b'],prior)
    vs=to_ticks(t,ts,z['vs'],np.array([120.**2,180.**2]))[0]
    old=to_ticks(t,ts,prev_b,prior)[0];oldvs=to_ticks(t,ts,prev_var,np.array([120.**2,180.**2]))[0]
    ps=to_ticks(t,ts,z['prob'],np.full(4,.25))[0]
    raw_angles=angle_features(z['alternatives'])[:,:,0];blended=angle_features(z['b'])[:,0]
    block=observer.m['systematic'][:,z['phase']].T
    bias_var=np.sum(z['prob']*(block+(raw_angles-blended[:,None])**2),axis=1)
    systematic=to_ticks(t,ts,bias_var,120.**2)[0]
    return dict(b=b,vs=vs,old=old,oldvs=oldvs,age=age,fresh=fresh,p=ps,sys=systematic,
                identified=to_ticks(t,ts,z['identified'],False)[0],novelty=to_ticks(t,ts,z['novelty'],0.)[0])
def estimate(task):
    case,name=task;t=control(case)
    folder=OUT/'records'/case;dest=OUT/'traces'/case;dest.mkdir(exist_ok=True,parents=True)
    obs=load_model();g=strain_packet(folder/'strain',name,t,obs);prof=readj(PREV/'visual-calibration.json')
    for cond in ('natural','occluded'):
        with np.load(folder/(cond+'.npz')) as f:v=dict(f)
        cap,bv,cv,stable=[v[k] for k in ('capture','visual_b','cv','stable_teacher')]
        visual,k=visual_output(t,cap,bv); vis_ok=np.isfinite(visual).all(1)
        f=lambda b,var,mode:fuse(t,b,g['fresh'],var,g['age'],cap,bv,cv,stable,prof,mode=mode)
        legacy,legacy_plane,_=filter_bend(t,g['old'],cap,bv,v['native_q'],[.5,1.],strain_fresh=g['fresh'])
        prev=f(g['old'],g['oldvs'],'both');prior_teacher=f(g['old'],g['oldvs'],'teacher')
        strain=f(g['b'],g['vs'],'strain');new=f(g['b'],g['vs'],'both');teacher=f(g['b'],g['vs'],'teacher')
        b=np.stack([g['old'],strain['b'],visual,legacy,prev['b'],prior_teacher['b'],new['b'],teacher['b']])
        plane=np.stack([angle_features(g['old'])[:,0]>=5,strain['plane'],angle_features(visual)[:,0]>=5,legacy_plane,
                        prev['plane'],prior_teacher['plane'],new['plane'],teacher['plane']])
        # A systematic residual can persist across repeated updates. Estimate its floor from
        # 1-second residual means on separate bench motions and the original visual pilot.
        cam_var=prof['theta_k']/(cv[k]+prof['epsilon'])
        weight=np.where(vis_ok,g['vs'][:,0]/(g['vs'][:,0]+cam_var),0.)
        conservative=np.sqrt(new['posterior_sigma'][:,0]**2+(1.-weight)**2*g['sys']+weight**2*float(obs.m['camera_systematic']))
        np.savez_compressed(dest/(name+'-'+cond+'.npz'),t=t,b=b,plane=plane,raw_health=g['b'],
            s=g['vs'][:,0]**.5,p=g['p'],identified=g['identified'],novelty=g['novelty'],common_weight=new['joint_vis_share'][:,0],
            accepted=new['visible_update'],mode_updates=teacher['correction_log'],iid_sigma=new['posterior_sigma'][:,0],
            reported_sigma=conservative,adc_fresh=g['fresh'])
    print('Released all variants',case,name,flush=True)
def interval(t,e,s):
    good=np.isfinite(e)&np.isfinite(s)
    return dict(n=int(good.sum()),coverage=float(np.mean(abs(e[good])<=1.96*s[good])),
        median_half_width_deg=float(np.median(1.96*s[good])))
def score():
    table={};per={};limits={};optical={}
    for name in FLIGHT_STATES:
        table[name]={}
        for cond in ('natural','occluded'):
            samples=[]
            for case in CASES:
                folder=OUT/'records'/case
                with np.load(OUT/'traces'/case/(name+'-'+cond+'.npz')) as f:z=dict(f)
                # Ground truth is first opened by the scorer, after all sensor-only estimates were released.
                with np.load(folder/'strain/reference.npz') as f:
                    np.testing.assert_array_equal(z['t'],f['t']);truth=f['mocap_b']
                samples.append((z,truth));full=(z['t']>=14)&(z['t']<46)
                per[case+'/'+name+'/'+cond]={m:metrics(z['b'][i,full],truth[full],z['plane'][i,full]) for i,m in enumerate(METHODS)}
                if name=='healthy':
                    p=np.load(folder/(cond+'.npz'));cap=p['capture'];cv=p['cv']
                    start=np.flatnonzero((cap>=35.-1e-7)&(cv>0))
                    reg=readj(OUT/'camera'/case/(cond+'-registration.json'))
                    optical[case+'/'+cond]=dict(input_frames=len(cap),valid=int((cv>0).sum()),native_poses=reg['native_anchored'],
                        auxiliary_poses=reg['verified_additional'],after_uncover_s=float(cap[start[0]]-35.+DELAY) if len(start) else None)
            rows={}
            for phase,(lo,hi) in PHASES.items():
                mask=[(z['t']>=lo-1e-8)&(z['t']<hi-1e-8) for z,_ in samples]
                truth=np.vstack([ref[k] for (_,ref),k in zip(samples,mask)])
                def concatenate(k):
                    return np.concatenate([z[k][m] for (z,_),m in zip(samples,mask)])
                b=np.concatenate([z['b'][:,m] for (z,_),m in zip(samples,mask)],axis=1)
                plane=np.concatenate([z['plane'][:,m] for (z,_),m in zip(samples,mask)],axis=1)
                same=np.isfinite(b[METHODS.index('vision')]).all(1)
                row={method:dict(all=metrics(b[i],truth,plane[i]),common=metrics(b[i,same],truth[same],plane[i,same])) for i,method in enumerate(METHODS)}
                tv=angle_features(truth)[:,0];err=angle_features(b[METHODS.index('health')])[:,0]-tv
                sensor_err=angle_features(concatenate('raw_health'))[:,0]-tv
                row['intervals']={s:interval(None,err,concatenate(s)) for s in ('reported_sigma','iid_sigma')}
                row['intervals']['strain']=interval(None,sensor_err,concatenate('s'))
                w=concatenate('common_weight');ok=np.isfinite(w)
                row['simultaneous_visual_share']=dict(n=int(ok.sum()),mean=float(w[ok].mean()) if ok.any() else None)
                row['identified_fraction']=float(concatenate('identified').mean())
                rows[phase]=row
            table[name][cond]=rows
    savej(OUT/'flight-scores.json',table);savej(OUT/'case-scores.json',per);savej(OUT/'optical.json',optical)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--score-only',action='store_true');a=ap.parse_args()
    if not a.score_only:
        with ProcessPoolExecutor(3) as pool:list(pool.map(estimate,[(case,g) for case in CASES for g in FLIGHT_STATES]))
    score()
