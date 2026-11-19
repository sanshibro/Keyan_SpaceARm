"""Fit and freeze health experts on bench motions. No test motion or test flight is opened."""
import hashlib
import numpy as np
from round_health_data import motions,STATES,PROFILE,box_source
from round_health_flights import OUT
from round_slim_box import OUT as SLIM
from round_slim_run import features_at,interp_b
from round_shape_sensors import predict
from round_wrist_fusion import readj,savej
from round_patent_camera import OLD,PILOT
from strain_health import HealthObserver,design,split_pairs,polynomial,ridge,angle_features,wrap

POINT=readj(SLIM/'p30/predictor.json')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def reference(folder):
    with np.load(folder/'reference.npz') as a:return {k:a[k] for k in ('t','mocap_b','length_m')}
def observations(folder,name,ref,replica=0):
    if replica:
        raw,lab,transition,_=box_source(folder.name+'-mount'+str(replica),ref['t'],ref['length_m'],name)
    else:
        with np.load(folder/(name+'.npz')) as f:raw=dict(f)
        with np.load(folder/(name+'-labels.npz')) as f:lab=f['classes'];transition=f['transition']
    truth=interp_b(raw['t'],ref['t'],ref['mocap_b'])
    b0=predict(POINT,features_at(raw['adc'],raw['tare']))
    return raw,lab,transition,truth,b0

def fit():
    keys=motions();tr=[OUT/'bench'/k for k,_,_ in keys['train']]
    val=[OUT/'bench'/k for k,_,_ in keys['validation']]
    gain=np.asarray(PROFILE['slope_counts_per_unit_ratio'])
    pair=[];common=[]
    for f in tr:
        raw=np.load(f/'healthy.npz');d,c=split_pairs(raw['adc'],raw['tare'],gain)
        pair.append(polynomial(d));common.append(c)
    forward=np.linalg.lstsq(np.vstack(pair),np.vstack(common),rcond=1e-7)[0]
    lib=[];tags=[];xs=[[] for _ in STATES];ys=[[] for _ in STATES]
    for folder in tr:
        ref=reference(folder)
        for j,name in enumerate(STATES):
            for replica in range(3):
                raw,label,transition,truth,_=observations(folder,name,ref,replica)
                X,S,phase,_=design(raw['t'],raw['adc'],raw['tare'],gain,forward)
                valid=(label==j)&~transition&(raw['t']>1.3)
                ix=np.flatnonzero(valid);n=min(190,len(ix))
                select=ix[np.linspace(0,len(ix)-1,n,dtype=int)]
                lib.append(S[select]);tags.append(np.full(n,j))
                if j:xs[j].append(X[valid][::2]);ys[j].append(truth[valid][::2])
        print('Fitted training features',folder.name,flush=True)
    experts=[ridge(np.vstack(xs[j]),np.vstack(ys[j]),.001) for j in (1,2,3)]
    S=np.vstack(lib);center=np.median(S,axis=0)
    scale=np.maximum(np.percentile(abs(S-center),75,axis=0),[.004,.004,.004,.004,.0015,.0015,.025,.012,.012,.006,1.])
    m=dict(gains=gain,forward=forward,centers=np.stack([x[0] for x in experts]),scales=np.stack([x[1] for x in experts]),
        coefficients=np.stack([x[2] for x in experts]),library=(S-center)/scale,library_labels=np.concatenate(tags),
        class_center=center,class_scale=scale,variance=np.ones((4,3,2)),variance_scale=np.ones(2),novelty_ref=np.array(1.))
    observer=HealthObserver(m)
    phase_errors={};block={};novel=[]
    for folder in val[:2]:
        ref=reference(folder)
        for j,name in enumerate(STATES):
            for replica in range(2):
                raw,lab,trans,truth,b0=observations(folder,name,ref,replica)
                X,S,phase,_=design(raw['t'],raw['adc'],raw['tare'],gain,forward)
                rr=angle_features(truth)
                delta=angle_features(observer.point(X,b0)[:,j])-rr;delta[:,1]=wrap(delta[:,1])
                valid=(lab==j)&~trans&(raw['t']>1.3)
                distance=observer.tree.query((S-center)/scale,k=48)[0][:,-1];novel.extend(distance[valid])
                for seg in range(3):
                    for axis in range(2):
                        sel=valid&(phase==seg)&((rr[:,0]>=5) if axis else True)
                        phase_errors.setdefault((j,seg,axis),[]).extend(delta[sel,axis])
                    for sec in np.unique(np.floor(raw['t']).astype(int)):
                        ids=valid&(phase==seg)&(np.floor(raw['t'])==sec)
                        if ids.sum()>=15:block.setdefault((j,seg),[]).append(float(np.mean(delta[ids,0])))
    m['novelty_ref']=np.array(max(.1,np.quantile(novel,.99)))
    system=np.ones((4,3))
    for j in range(4):
        for seg in range(3):
            for axis in range(2):
                err=phase_errors.get((j,seg,axis),[])
                if len(err)<24:err=sum((phase_errors.get((j,other,axis),[]) for other in range(3)),[])
                m['variance'][j,seg,axis]=max([.16,.30][axis]**2,np.mean(np.square(err)))
            bias=block.get((j,seg),[])
            system[j,seg]=np.mean(np.square(bias)) if len(bias)>=3 else m['variance'][j,seg,0]
    m['systematic']=system
    rv=reference(val[2]);ratios=[[],[]]
    for j,name in enumerate(STATES):
        raw,lab,transition,truth,b0=observations(val[2],name,rv)
        z=observer.estimate(raw['t'],raw['adc'],raw['tare'],b0)
        rt=angle_features(truth);err=angle_features(z['b'])-rt;err[:,1]=wrap(err[:,1])
        for axis in range(2):
            sel=(lab==j)&~transition&((rt[:,0]>=5) if axis else True)
            ratios[axis].extend(abs(err[sel,axis])/np.sqrt(z['vs'][sel,axis]))
    m['variance_scale']=np.array([max(.25,(np.quantile(v,.95)/1.96)**2) for v in ratios])
    with np.load(OLD/PILOT/'natural-estimate.npz') as pilot:
        cap=pilot['capture'];e=angle_features(pilot['visual_b'])[:,0]-angle_features(pilot['truth_at_capture'])[:,0]
        means=[np.mean(e[(cap>=k)&(cap<k+1)&np.isfinite(e)]) for k in range(14,46) if np.isfinite(e[(cap>=k)&(cap<k+1)]).sum()>=15]
    m['camera_systematic']=np.array(float(np.mean(np.square(means))))
    path=OUT/'model'/'health-model.npz';path.parent.mkdir(exist_ok=True)
    if path.exists():raise FileExistsError('Frozen fit already exists. Use a new round to change a fitted model.')
    np.savez_compressed(path,**m)
    savej(OUT/'freeze.json',dict(model_sha256=digest(path),bench_training=[p.name for p in tr],bench_validation=[p.name for p in val],
        split='First two validation motions estimate expert and one-second bias variances; third scales marginal intervals. Pilot visual scale from previous independent flight.',
        input_hashes={p.name:digest(p/'reference.npz') for p in tr+val},point_model_sha256=digest(SLIM/'p30/predictor.json'),
        code_files={p.name:digest(p) for p in [__import__('pathlib').Path(__file__),__import__('pathlib').Path(__import__('strain_health').__file__)]},
        confidence_threshold=.90,neighbors=48,library_size=len(m['library']),novelty_ref=float(m['novelty_ref']),
        variance_multiplier=m['variance_scale'].tolist(),camera_systematic_theta_sd_deg=float(np.sqrt(m['camera_systematic']))))
    print('FROZEN',digest(path),flush=True)
if __name__=='__main__':fit()
