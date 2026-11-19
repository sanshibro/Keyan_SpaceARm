"""Check identification, fundamental ambiguity, causal release and native run evidence."""
import numpy as np
import mujoco
from scipy.spatial import cKDTree
from scipy.signal import savgol_filter
from rosbag import Bag
from round_health_data import motions,STATES,SLIM_SPEC,PROFILE,geometry
from round_health_fit import reference,observations,POINT
from round_health_evaluate import load_model,control,strain_packet,CASES,METHODS
from round_health_flights import OUT
from round_slim_box import OUT as SLIM
from round_slim_run import features_at
from round_shape_sensors import predict,marker_cluster,mocap_bend
from round_g1_model import make_plant,SimArmBridge
from round_wrist_fusion import readj,savej
from round_patent_camera import OUT as PREV
from slim_sensor import SlimGaugeRig,SlimReadout
from patent_confidence import fuse
from strain_health import angle_features

def confusion(probs,truth,identified):
    winner=np.argmax(probs,axis=1)
    decisions=np.where(identified,winner,4)
    counts=np.zeros((4,5),int);np.add.at(counts,(truth,decisions),1)
    marginal={}
    for key,pval,expected in [('offset',probs[:,1]+probs[:,3],np.isin(truth,[1,3])),
                              ('memory',probs[:,2]+probs[:,3],np.isin(truth,[2,3]))]:
        hard=pval>=.5;sens=hard[expected].mean();spec=(~hard[~expected]).mean()
        high=(pval>=.90)|(pval<=.10)
        marginal[key]=dict(balanced_accuracy=float(.5*(sens+spec)),sensitivity=float(sens),specificity=float(spec),
            high_confidence_fraction=float(high.mean()),high_confidence_accuracy=float(np.mean(hard[high]==expected[high])) if high.any() else None)
    return dict(counts=counts.tolist(),threshold=.90,high_confidence_fraction=float(identified.mean()),
        high_confidence_accuracy=float(np.mean(winner[identified]==truth[identified])) if identified.any() else None,
        forced_balanced_accuracy=float(np.mean([np.mean(winner[truth==j]==j) for j in range(4)])),marginal=marginal)
def bench(observer):
    pp=[];yy=[];dec=[];phase_rows={};per={};pairs={}
    for key,_,_ in motions()['test']:
        folder=OUT/'bench'/key;ref=reference(folder)
        for j,name in enumerate(STATES):
            raw,labels,transition,truth,b0=observations(folder,name,ref)
            ts=raw['t'];z=observer.estimate(ts,raw['adc'],raw['tare'],b0)
            a=angle_features(truth);speed=np.gradient(savgol_filter(a[:,0],25,2),ts)
            valid=(labels==j)&~transition&(ts>1.3)&(a[:,0]>=8)
            pp.append(z['prob'][valid]);yy.extend(labels[valid]);dec.extend(z['identified'][valid])
            error=angle_features(z['b'])[:,0]-a[:,0];nom=angle_features(b0)[:,0]-a[:,0]
            per[key+'/'+name]=dict(n=int(valid.sum()),nominal_rmse=float(np.mean(nom[valid]**2)**.5),health_rmse=float(np.mean(error[valid]**2)**.5))
            for label,mask in [('loading',speed>2.),('holding',abs(speed)<=2.),('unloading',speed< -2.)]:
                keep=valid&mask
                if keep.any():phase_rows.setdefault(name+'/'+label,[]).extend(np.column_stack((nom[keep],error[keep])).tolist())
            # Equal-shape loading and unloading, nearest vector within 0.8 degrees and separated in time.
            up=np.flatnonzero(valid&(speed>2.))[::4];down=np.flatnonzero(valid&(speed< -2.))
            obs=[]
            if len(up) and len(down):
                tree=cKDTree(np.rad2deg(truth[down]));idx=tree.query_ball_point(np.rad2deg(truth[up]),.8);used=set()
                for i,cand in zip(up,idx):
                    opts=[down[h] for h in cand if down[h] not in used and abs(ts[down[h]]-ts[i])>1.]
                    if not opts:continue
                    k=min(opts,key=lambda h:np.linalg.norm(truth[h]-truth[i]));used.add(k)
                    obs.append(float(np.linalg.norm(raw['adc'][i].astype(float)-raw['adc'][k])/2.))
            pairs[key+'/'+name]=dict(n=len(obs),median_rms_channel_code_difference=float(np.median(obs)) if obs else None)
    results={k:dict(n=len(v),nominal_rmse=float(np.mean(np.asarray(v)[:,0]**2)**.5),
                    health_rmse=float(np.mean(np.asarray(v)[:,1]**2)**.5)) for k,v in phase_rows.items()}
    return dict(classification=confusion(np.vstack(pp),np.array(yy),np.array(dec)),per_motion=per,phases=results,equal_shape_pairs=pairs)
def indistinguishable(observer):
    plant=make_plant();m,d=plant.m,plant.d;rig=SlimGaugeRig(m,SLIM_SPEC,.3);bridge=SimArmBridge()
    t=np.arange(0.,5.02,.02);noise=np.load(SLIM/'stationary-box.npz')['events'];boxes=[];angles=[]
    for end in (35.,43.):
        lengths=[];true=[]
        for ts in t:
            u=np.clip((ts-.8)/2.,0,1);deg=35.+(end-35.)*u*u*(3.-2.*u)
            d.qpos[:]=m.qpos0;bridge.set_kinematic_shape(m,d,np.radians(deg),np.pi);mujoco.mj_kinematics(m,d)
            lengths.append(rig.lengths(d));true.append(np.degrees(np.linalg.norm(mocap_bend(np.stack([marker_cluster(d,m.body(k).id) for k in ('mount','tool')])))))
        box=SlimReadout(SLIM_SPEC,PROFILE,noise,1117)
        boxes.append(box.sample(t,np.asarray(lengths),.3));angles.append(true)
    original,other=boxes
    assert np.array_equal(original['t'],other['t']) and np.array_equal(original['tare'],other['tare'])
    off=other['adc'].astype(int)-original['adc'];observed=original['adc']+off
    assert np.array_equal(observed,other['adc']) and not original['rail'].any() and not other['rail'].any()
    outs=[]
    for q in [observed,other['adc']]:
        b0=predict(POINT,features_at(q,other['tare']))
        outs.append(observer.estimate(other['t'],q,other['tare'],b0)['b'])
    assert np.array_equal(*outs)
    return dict(physical_bends_deg=[float(row[-1]) for row in angles],max_channel_changes_counts=np.max(abs(off),axis=0).tolist(),
        peak_observed_code_difference=int(np.max(abs(observed-other['adc']))),max_estimate_difference_rad=float(np.max(abs(outs[0]-outs[1]))),
        unavoidable_worst_case_error_deg=.5*abs(float(angles[1][-1]-angles[0][-1])),
        scope='Constructed identical four-channel histories. This establishes ambiguity for an allowed directional offset, not how often real offsets have this structure.')
def run():
    obs=load_model();savej(OUT/'bench-scores.json',bench(obs));amb=indistinguishable(obs)
    prof=readj(PREV/'visual-calibration.json');prefix=0.;estdiff=0.;count=0;gap_count=0
    flight_info={};pp=[];yy=[];dec=[]
    for case in CASES:
        origin=OUT/'runs'/case;meta=readj(origin/'run.json')
        with (origin/'flight.ulg').open('rb') as fh:magic=fh.read(7)
        assert magic==b'ULog\x01\x12\x35'
        with Bag(str(origin/'flight.bag'),'r') as bag:
            topics={k:int(v.message_count) for k,v in bag.get_type_and_topic_info()[1].items()}
        flight_info[case]=dict(ulog_bytes=(origin/'flight.ulg').stat().st_size,topics=topics,contact_steps=meta['contact_integration_steps'],
            peak_curvature_deg=meta['max_total_curvature_deg'],native_sensor_packets=meta['packets'])
        root=OUT/'records'/case; t=control(case)
        for j,name in enumerate(STATES):
            with np.load(root/'strain'/(name+'.npz')) as f:r=dict(f)
            b0=predict(POINT,features_at(r['adc'],r['tare']));z=obs.estimate(r['t'],r['adc'],r['tare'],b0)
            il=np.load(root/'strain'/(name+'-labels.npz'));mask=(il['classes']==j)&~il['transition']&(r['t']>=14)&(r['t']<46)
            pp.append(z['prob'][mask]);yy.extend(il['classes'][mask]);dec.extend(z['identified'][mask])
            for seconds in (25.8,33.04,36.1):
                keep=r['t']<=seconds;short=obs.estimate(r['t'][keep],r['adc'][keep],r['tare'],b0[keep])
                err=float(np.max(abs(short['b']-z['b'][keep])));prefix=max(prefix,err)
                assert err<1e-10 and np.allclose(short['vs'],z['vs'][keep],atol=1e-10)
            trace=np.load(OUT/'traces'/case/(name+'-occluded.npz'))
            blind=(t>=31.2-1e-8)&(t<35.2-1e-8)
            assert not trace['accepted'][blind].any() and not np.isfinite(trace['common_weight'][blind]).any()
            assert np.isfinite(trace['b'][METHODS.index('health'),blind]).all()
            assert not np.ptp(trace['mode_updates'][blind],axis=0).any()
            if name=='mixed':
                g=strain_packet(root/'strain',name,t,obs)
                v=dict(np.load(root/'occluded.npz'));p=lambda key:v[key]
                for seconds in (29.04,33.04,35.94):
                    keep=t<=seconds;cam=p('capture')<=t[keep][-1]+1e-8
                    result=fuse(t[keep],g['b'][keep],g['fresh'][keep],g['vs'][keep],g['age'][keep],
                        *[p(k)[cam] for k in ('capture','visual_b','cv','stable_teacher')],prof)
                    expected=trace['b'][METHODS.index('health'),keep];estdiff=max(estdiff,float(np.max(abs(result['b']-expected))));count+=1
                gap_count+=int(blind.sum())
    assert gap_count==600 and estdiff<1e-10
    savej(OUT/'audit.json',dict(ambiguity=amb,flight=flight_info,flight_classification=confusion(np.vstack(pp),np.array(yy),np.array(dec)),
        sensor_prefix_tests=36,maximum_sensor_prefix_difference_rad=prefix,fusion_prefix_tests=count,
        maximum_fusion_prefix_difference_rad=estdiff,blind_ticks_per_three_flights=gap_count))
    print('Completed independent sensitivity and clock audits',amb,flush=True)
if __name__=='__main__':run()
