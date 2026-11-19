"""Paired SLIM-M material-path, ADC and angle evaluation on the archived G1 movements."""
import argparse, hashlib,json
from pathlib import Path
import numpy as np
import mujoco
from round_slim_box import OUT,SPEC
from round_shape_pixels import OUTPUT as OLD,EVIDENCE
from round_g1_model import make_plant,SimArmBridge
from round_shape_sensors import features,fit_model,predict,stable_seed,vector_angles
from round_wrist_fusion import CASES,PHASES,METHODS,DELAY,filter_bend,metrics,readj,savej
from slim_sensor import SlimGaugeRig,SlimReadout

PRELOADS=SPEC['preloads']
def features_at(adc,tare):
    x=(np.asarray(adc,float)-tare)/4095.
    return np.column_stack((x,x[:,0]-x[:,2],x[:,1]-x[:,3]))
def interp_b(ts,t,b):
    return np.column_stack([np.interp(ts,t,b[:,j]) for j in range(2)])
def accuracy(est,ref):
    return metrics(est,ref,vector_angles(est)[0]>=5.)
def label(p):return 'p%02d'%round(100*p)

def geometry():
    plant=make_plant();m,d=plant.m,plant.d
    rigs=[SlimGaugeRig(m,SPEC,p) for p in PRELOADS]
    dest=OUT/'kinematics';dest.mkdir(exist_ok=True)
    all_folders=sorted((OLD/'bench').glob('*.npz'))
    for case in [f.stem for f in all_folders]+CASES:
        if case in CASES:
            folder=EVIDENCE/'runs'/case;raw=np.load(str(folder/'physics.npz'))
            ref=np.load(str(OLD/'runs'/case/'scoring_truth.npz'))
            origin=readj(folder/'run.json')['release_us'];t=ref['t'];truth=ref['mocap_b']
            ix=np.searchsorted(raw['us'],origin+np.rint(t*1e6).astype('int64'))
            np.testing.assert_array_equal(raw['us'][ix],origin+np.rint(t*1e6).astype('int64'))
            states=raw['qpos'][ix]
        else:
            raw=np.load(str(OLD/'bench'/(case+'.npz')));t=raw['t'];truth=raw['mocap_b']
            states=np.tile(m.qpos0,(len(t),1));states[:,plant.arm.qpos_ids]=raw['plant_joint']
        path=np.empty((len(t),len(rigs),4))
        for i,q in enumerate(states):
            d.qpos[:]=q;mujoco.mj_kinematics(m,d)
            path[i]=[r.lengths(d) for r in rigs]
        np.savez_compressed(dest/(case+'.npz'),t=t,length_m=path,mocap_b=truth,preloads=PRELOADS)
    angles=np.arange(0,121,5);planes=np.arange(0,360,15)
    shape=np.empty((len(angles),len(planes),len(rigs),4));bridge=SimArmBridge()
    for i,theta in enumerate(angles):
        for j,phi in enumerate(planes):
            d.qpos[:]=m.qpos0
            bridge.set_kinematic_shape(m,d,np.deg2rad(theta),np.deg2rad(phi))
            mujoco.mj_kinematics(m,d)
            shape[i,j]=[r.lengths(d)/r.free-1. for r in rigs]
    np.testing.assert_allclose(shape[0],np.broadcast_to(np.array(PRELOADS)[None,:,None],shape[0].shape),atol=1e-6)
    np.savez_compressed(OUT/'sweep.npz',bend_deg=angles,plane_deg=planes,preloads=PRELOADS,extension=shape)
    out={}
    for j,p in enumerate(PRELOADS):
        e=shape[:,:,j]
        out[label(p)]=dict(installed_span_mm=25*(1+p),min_120_percent=float(e[-1].min()*100),max_120_percent=float(e[-1].max()*100),
            first_slack_test_angle_deg=next((int(angles[i]) for i,a in enumerate(e) if a.min()<-1e-8),None),
            overrange_channel_samples=int(np.sum(e>1+1e-9)))
    savej(OUT/'sweep.json',out);print('120 degrees',out,flush=True)

def acquire(case,p,scale=1.,shift=False,memory=None):
    path=OUT/'kinematics'/(case+'.npz');source=np.load(str(path));t=source['t']
    profile=readj(OUT/'box-profile.json');noise=np.load(str(OUT/'stationary-box.npz'))['events']
    box=SlimReadout(SPEC,profile,noise,stable_seed('box-v1-'+case))
    data=box.sample(t,source['length_m'][:,PRELOADS.index(p)],p,scale,profile['offset_stress_counts'] if shift else None,memory)
    return t,source['mocap_b'],data

def train():
    files=sorted((OLD/'bench').glob('*.npz'));results={}
    for p in PRELOADS:
        out=OUT/label(p);out.mkdir(exist_ok=True);Xs={};Ys={};raws={}
        for f in files:
            key=f.stem;t,mocap,d=acquire(key,p)
            assert not d['overrange'].any()
            Xs[key]=features_at(d['adc'],d['tare']);Ys[key]=interp_b(d['t'],t,mocap);raws[key]=d
            if p==SPEC['preload_primary'] or key=='test-00':
                (out/'bench').mkdir(exist_ok=True)
                np.savez_compressed(out/'bench'/f.name,**d,mocap_b=Ys[key])
        names=[f.stem for f in files if f.stem.startswith('train-')]
        assert len(names)==8
        model=fit_model([Xs[n] for n in names],[Ys[n] for n in names]);model['training_sessions']=names
        savej(out/'predictor.json',model);stats={}
        for split in ('train','validation','test'):
            keys=[k for k in Xs if k.startswith(split+'-')]
            est=np.vstack([predict(model,Xs[k]) for k in keys])
            d=np.vstack([raws[k]['strain'] for k in keys]);codes=np.vstack([raws[k]['adc'] for k in keys])
            stats[split]={**accuracy(est,np.vstack([Ys[k] for k in keys])), 'extension_range_pct':[float(d.min()*100),float(d.max()*100)],
                'fraction_any_slack':float((d<-1e-8).any(1).mean()),'fraction_any_adc_rail':float(((codes==0)|(codes==4095)).any(1).mean())}
        savej(out/'bench-scores.json',stats);results[label(p)]=stats
    savej(OUT/'bench-scores.json',results);print('Bench test',[(k,round(v['test']['theta_rmse_deg'],3)) for k,v in results.items()],flush=True)

def estimates(case,p,tag,scale=1.,shift=False,memory=None,model_path=None):
    model=readj(model_path if model_path is not None else OUT/label(p)/'predictor.json')
    t,truth,d=acquire(case,p,scale,shift,memory)
    adc_t=d['t'];valid=~d['overrange'].any(1)
    prediction=predict(model,features_at(d['adc'][valid],d['tare']))
    initial=predict(model,features_at(d['initial_adc'][None,:],d['tare']))[0]
    ts=adc_t[valid];j=np.searchsorted(ts,t+1e-8,side='right')-1
    held=np.vstack((initial[None,:],prediction))[j+1]
    fresh=np.r_[True,np.diff(j)>0]
    val=readj(OUT/label(p)/'bench-scores.json')['validation']
    sigma=[max(.5,val['theta_rmse_deg']),max(1.,val['phi_rmse_deg'])]
    out=OUT/tag/case;out.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(out/'strain.npz',**d,t_tick=t,raw_b=held,fresh=fresh)
    rec={}
    for cond in ('natural','occluded'):
        old=OLD/'slam/textured'/case/(cond+'-estimate.npz')
        z=np.load(str(old));np.testing.assert_allclose(z['t'],t,atol=1e-8)
        cap,bv,q=z['capture'],z['visual_b'],z['quality']
        fu,fd,weight=filter_bend(t,held,cap,bv,q,sigma,strain_fresh=fresh)
        sm,sd,_=filter_bend(t,held,cap,bv,np.zeros_like(q),sigma,strain_fresh=fresh)
        v=z['b'][2];b=np.stack((held,sm,v,fu));direction=np.stack((vector_angles(held)[0]>=5,sd,vector_angles(v)[0]>=5,fd))
        if cond=='occluded':
            blackout=(t>=31.22)&(t<35.2)
            assert np.all(weight[blackout]==0) and not np.isfinite(v[blackout]).any()
        if tag=='primary' or (cond=='occluded' and case=='241119-online_pause'):
            np.savez_compressed(out/(cond+'.npz'),t=t,raw_b=b,methods=METHODS,truth=truth,direction=direction,vision_weight=weight)
        rec[cond]=(b,direction,truth,t)
    return rec,dict(n=len(d['t']),rail=int(d['rail'].any(1).sum()),slack=int(d['slack'].any(1).sum()),over=int(d['overrange'].any(1).sum()),
        min_percent=float(d['strain'].min()*100),max_percent=float(d['strain'].max()*100),same_tick_coalesced=int(np.maximum(np.diff(j)-1,0).sum()))

def flight():
    all_scores={}
    for tag,p,scale,shift in [('primary',.3,1.,False),('natural_mount',0.,1.,False),('mount_35mm',.4,1.,False),
                              ('gain_half',.3,.5,False),('gain_double',.3,2.,False),('return_offset',.3,1.,True)]:
        by_case={};acq={}
        for case in CASES:by_case[case],acq[case]=estimates(case,p,tag,scale,shift)
        score={}
        for cond in ('natural','occluded'):
            score[cond]={}
            for phase,(lo,hi) in PHASES.items():
                bs=[];ds=[];ys=[]
                for case in CASES:
                    b,d,y,t=by_case[case][cond];mask=(t>=lo)&(t<hi)
                    bs.append(b[:,mask]);ds.append(d[:,mask]);ys.append(y[mask])
                b=np.concatenate(bs,axis=1);dire=np.concatenate(ds,axis=1);truth=np.concatenate(ys)
                shared=np.isfinite(b[2]).all(1)
                score[cond][phase]={m:{'all':metrics(b[h],truth,dire[h]),'shared_visual':metrics(b[h,shared],truth[shared],dire[h,shared])} for h,m in enumerate(METHODS)}
        case_scores={case:{} for case in CASES}
        for case in CASES:
            for cond,(b,d,y,t) in by_case[case].items():
                sel=(t>=14)&(t<46);case_scores[case][cond]={m:metrics(b[h,sel],y[sel],d[h,sel]) for h,m in enumerate(METHODS)}
        all_scores[tag]=dict(conditions=score,per_case=case_scores,sensors=acq,preload=p,gain_scale=scale,end_zero_sensitivity=shift)
        print(tag,'fused theta',round(score['natural']['all']['fusion']['all']['theta_rmse_deg'],3),'blackout',
              round(score['occluded']['during']['fusion']['all']['theta_rmse_deg'],3),flush=True)
    savej(OUT/'flight-scores.json',all_scores)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('task',choices=('geometry','train','flight'));option=a.parse_args()
    {'geometry':geometry,'train':train,'flight':flight}[option.task]()
