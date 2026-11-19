"""Check direction dependence in box archives and isolate dynamic effects in paired simulated flights."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.optimize import lsq_linear
from round_slim_box import OUT,SPEC,FILES
from round_slim_run import acquire,estimates,features_at,interp_b,accuracy
from round_wrist_fusion import readj,savej,CASES,PHASES
from round_shape_sensors import fit_model,predict,vector_angles
from slim_sensor import memory_components
from round_wrist_pack import pack,sha

ROOT=OUT/'memory';PROFILE=readj(OUT/'box-profile.json')
TESTS={'loop':dict(half_width=.02,tau_s=2.,play_weight=.5,transient_weight=0.),
       'relax':dict(half_width=.02,tau_s=2.,play_weight=0.,transient_weight=.02),
       'combined':dict(half_width=.02,tau_s=2.,play_weight=.5,transient_weight=.02)}
def archive_record(key):
    df=pd.read_csv(FILES/key/'dual_strain_auto.csv').iloc[::4].copy()
    t=df.rel_time.to_numpy();valid=(df.usable.to_numpy()==1)&(df.tracking_state.to_numpy()==2)&(df.theta_age.to_numpy()<.15)
    theta=df.orb_theta_deg.to_numpy();phi=np.deg2rad(df.orb_fai_deg.to_numpy())
    vec=theta[:,None]*np.column_stack((np.cos(phi),np.sin(phi)))
    for j in range(2):vec[:,j]=np.interp(t,t[valid],vec[valid,j])
    alpha=np.deg2rad([r['equivalent_angle_deg'] for r in PROFILE['transfer_check']]);unit=np.array([np.cos(alpha),np.sin(alpha)])
    proj=vec@unit
    p=SPEC['preload_primary'];strain=p+(1+p)*SPEC['arm_radius_m']/.28*np.deg2rad(proj)
    curve=SPEC['resistance_curve_25C'];ratio=np.interp(strain,curve['engineering_strain'],curve['relative_resistance_change'])
    neutral=np.interp(p,curve['engineering_strain'],curve['relative_resistance_change'])
    codes=df[['ch1','ch2','ch3']].to_numpy()
    zeros=PROFILE['session_stats'][key]['neutral_codes'];g=np.array(PROFILE['slope_counts_per_unit_ratio'][:3])
    nominal=zeros+g*(ratio-neutral)
    mask=valid&df.phase.isin(['moving','holding']).to_numpy()
    return dict(t=t,vec=vec,proj=proj,ratio=ratio,actual=codes,base=nominal,mask=mask,phase=df.phase.to_numpy(),ids=df.point_index.to_numpy())
def pairs(d):
    t,vec,proj=d['t'],d['vec'],d['proj'];rate=(proj[4:]-proj[:-4])/(t[4:]-t[:-4])[:,None]
    rate=np.pad(rate,((2,2),(0,0)),mode='edge')
    candidates=np.flatnonzero(d['mask']&(d['phase']=='moving')&(np.linalg.norm(vec,axis=1)>10))[::3]
    tree=cKDTree(vec[candidates]);out=[]
    for ch in range(3):
        groups={}
        for typ in ['opposite','same']:
            distances=[];used=set()
            for ii,ix in enumerate(candidates):
                if ix in used or abs(rate[ix,ch])<1:continue
                ids=candidates[tree.query_ball_point(vec[ix],.8)]
                ids=[j for j in ids if j not in used and 2<=t[j]-t[ix]<=60 and d['ids'][ix]!=d['ids'][j] and
                     abs(rate[j,ch])>=1 and ((rate[ix,ch]*rate[j,ch]<0)==(typ=='opposite'))]
                if not ids:continue
                j=min(ids,key=lambda j:np.linalg.norm(vec[j]-vec[ix]))
                distances.append(abs(d['actual'][j,ch]-d['actual'][ix,ch]));used.update((ix,j))
            groups[typ]=dict(n=len(distances),median_abs_count_difference=float(np.median(distances)) if distances else None)
        out.append(groups)
    return out
def calibrate():
    cal=archive_record(SPEC['source_sessions'][-1]);test=archive_record(SPEC['hardware_session_holdout'])
    gain=np.asarray(PROFILE['slope_counts_per_unit_ratio'][:3])
    def get_basis(d,h,tau):
        p,s=memory_components(d['t'],d['ratio'],h,tau)
        return np.stack((p*gain,s*gain),axis=-1)
    def fit(X,d,idx,ch):
        z=X[idx,ch];y=(d['actual']-d['base'])[idx,ch]
        return lsq_linear(z,y,bounds=([-1.,-.1],[1.,.1])).x
    def rmse(e):return float(np.sqrt(np.mean(e**2)))
    cut=cal['t'][round(.68*len(cal['t']))]
    idx1=cal['mask']&(cal['t']<cut);idx2=cal['mask']&~idx1
    best=[(1e5,None)]*3
    for h in [.008,.02,.04]:
        for tau in [.5,2.,5.]:
            X=get_basis(cal,h,tau)
            for j in range(3):
                v=fit(X,cal,idx1,j);err=rmse(np.clip(cal['base'][idx2,j]+X[idx2,j]@v,0,4095)-cal['actual'][idx2,j])
                if err<best[j][0]:best[j]=(err,(h,tau))
    rows=[];profile={k:[] for k in ('half_width','tau_s','play_weight','transient_weight')}
    for j,(_,choice) in enumerate(best):
        h,tau=choice;X=get_basis(cal,h,tau);v=fit(X,cal,cal['mask'],j);Z=get_basis(test,h,tau);mask=test['mask']
        base=test['base'][mask,j];y=test['actual'][mask,j];dyn=np.clip(base+Z[mask,j]@v,0,4095)
        rows.append(dict(ch=j+1,frozen_on=SPEC['source_sessions'][-1],checked_on=SPEC['hardware_session_holdout'],
                    static_rmse_counts=rmse(np.clip(base,0,4095)-y),memory_rmse_counts=rmse(dyn-y),n=int(mask.sum())))
        for key,value in zip(profile,[h,tau,*v]):profile[key].append(float(value))
    for v in profile.values():v.append(float(np.median(v)))
    return profile,dict(heldout=rows,opposite_and_same_direction_pairs={'fit':pairs(cal),'holdout':pairs(test)},
        pair_rule='Within 0.8 degree in the measured bend-vector plane; different movement windows, 2 to 60 seconds apart; projected speed magnitude at least 1 degree/s; per-group disjoint pairs. No direct strip-length reference.',
        orientation='Historical optical base-to-tool orientation estimates conditional local strain; aggregate channel dynamics can also contain arm, mounting and timing effects.')
def retrain(memory):
    train_x=[];train_y=[];tests=[]
    for key in sorted(p.stem for p in (OUT/'kinematics').glob('*.npz') if p.stem.startswith(('train-','test-','validation-'))):
        t,truth,d=acquire(key,.3,memory=memory)
        x=features_at(d['adc'],d['tare']);y=interp_b(d['t'],t,truth)
        if key.startswith('train-'):train_x.append(x);train_y.append(y)
        elif key.startswith('test-'):tests.append((x,y))
    assert len(train_x)==8 and len(tests)==3
    model=fit_model(train_x,train_y);savej(ROOT/'retrained-predictor.json',model)
    X,Y=map(np.vstack,zip(*tests))
    return accuracy(predict(model,X),Y)
def flight_profile(name,mem,retrained=False):
    pooled={key:[[],[],[]] for key in ('natural','occluded')}
    for case in CASES:
        rec,_=estimates(case,.3,'memory/'+name,memory=mem,model_path=ROOT/'retrained-predictor.json' if retrained else None)
        for cond in ('natural','occluded'):
            b,_,truth,t=rec[cond];lo,hi=PHASES['all' if cond=='natural' else 'during'];q=(t>=lo)&(t<hi)
            for group,values in zip(pooled[cond],[b[0,q],b[3,q],truth[q]]):group.append(values)
        if case==CASES[1]:
            b,_,truth,t=rec['occluded'];folder=ROOT/name/case
            np.savez_compressed(folder/'estimated.npz',t=t,b=b,truth=truth)
    out={}
    for cond,arrays in pooled.items():
        raw,fused,truth=map(np.vstack,arrays);out[cond]=dict(fusion=accuracy(fused,truth),strain=accuracy(raw,truth))
    return out
def main():
    ROOT.mkdir(exist_ok=True);emp,transfer=calibrate()
    profiles={**TESTS,'historical_fit':emp}
    res={'profiles':profiles,'historical_check':transfer,'heldout_bench_retrained':retrain(TESTS['combined']),'simulated':{}}
    for name,mem in profiles.items():res['simulated'][name]=flight_profile(name,mem)
    res['simulated']['combined_retrained']=flight_profile('combined_retrained',TESTS['combined'],True)
    path=OUT/'memory-scores.json';savej(path,res)
    material=SPEC['resistance_curve_25C'];base=lambda e:np.interp(e,material['engineering_strain'],material['relative_resistance_change'])
    t=np.arange(0,36.001,.025);eps=np.interp(t,[0,2,8,16,22,29,35,36],[.3,.3,.5,.5,.3,.3,.5,.5])
    stat=base(eps);h,s=memory_components(t,stat,.02,2.)
    np.savez_compressed(ROOT/'controlled-loop.npz',t=t,eps=eps,static=stat,loop=stat+.5*h,relax=stat+.02*s,combined=stat+.5*h+.02*s)
    assert np.allclose(memory_components(t,np.ones_like(t),.02,2.),0)
    a,b=memory_components(t[:423],stat[:423],.02,2.);np.testing.assert_array_equal(a,h[:423]);np.testing.assert_array_equal(b,s[:423])
    dip=np.argmin(abs(t-5));back=np.argmin(abs(t-19));assert abs(eps[dip]-eps[back])<1e-10 and abs(.5*(h[back]-h[dip])-.02)<1e-8
    # Frozen gain, noise and input clock in both arms of the controlled test.
    _,_,legacy=acquire(CASES[0],.3);_,_,zero=acquire(CASES[0],.3,memory=dict(half_width=.02,tau_s=2.,play_weight=0.,transient_weight=0.))
    np.testing.assert_array_equal(legacy['adc'],zero['adc']);np.testing.assert_array_equal(legacy['t'],zero['t'])
    archive=OUT/'evidence/slim-memory.tar.gz';files={f.relative_to(OUT).as_posix():f for f in ROOT.rglob('*') if f.is_file()}
    files['memory-scores.json']=path;files['datasheet-plateaus.json']=OUT/'datasheet-plateaus.json';pack(archive,files)
    assert archive.stat().st_size<24*1024**2
    idx=readj(OUT/'evidence-index.json');idx[archive.name]=dict(bytes=archive.stat().st_size,sha256=sha(archive),members=len(files))
    savej(OUT/'evidence-index.json',idx)
    print('holdout',transfer['heldout'],'paired',transfer['opposite_and_same_direction_pairs']['holdout'],flush=True)
    for k,v in res['simulated'].items():print(k,[round(v[c]['fusion']['theta_rmse_deg'],3) for c in ('natural','occluded')],flush=True)
if __name__=='__main__':main()
