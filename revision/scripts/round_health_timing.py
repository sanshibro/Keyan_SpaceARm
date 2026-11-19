"""Macro-average by complete motion and measure sustained shift alarms without changing frozen estimators."""
import numpy as np
from round_health_audit import confusion
from round_health_data import motions,STATES
from round_health_fit import reference,observations
from round_health_evaluate import load_model,CASES
from round_health_flights import OUT
from round_wrist_fusion import readj,savej
from strain_health import angle_features

def score_motion(obs,folder,domain):
    ref=reference(folder);injected=readj(folder/'generator.json')
    pp=[];yy=[];classified=[];alarms={};support={}
    for j,state in enumerate(STATES):
        raw,lab,transition,truth,b0=observations(folder,state,ref)
        t=raw['t'];z=obs.estimate(t,raw['adc'],raw['tare'],b0)
        eligible=(t>1.3)&(angle_features(truth)[:,0]>=8) if domain=='bench' else (t>=14)&(t<46)
        mask=eligible&(lab==j)&~transition
        support[state]=int(mask.sum());pp.extend(z['prob'][mask]);yy.extend(lab[mask]);classified.extend(z['identified'][mask])
        valid=(t>1.3) if domain=='bench' else (t>=14)&(t<46)
        marginal=z['prob'][:,1]+z['prob'][:,3]
        sustained=np.convolve((marginal>=.90).astype(int),np.ones(5,dtype=int),'full')[:len(t)]==5
        alarm=sustained&valid
        upward=alarm&~np.r_[False,alarm[:-1]]
        onset=injected[state]['onset_s'] if state in ('offset','mixed') else None
        pre=valid if onset is None else valid&(t<onset)
        post=np.flatnonzero(upward&(t>=onset)) if onset is not None else []
        alarms[state]=dict(onset_s=onset,false_episodes=int(np.sum(upward&pre)),baseline_samples=int(np.sum(pre)),
                           detection_delay_s=float(t[post[0]]-onset) if len(post) else None)
    c=confusion(np.array(pp),np.array(yy),np.array(classified))
    return dict(eligible=support,classifier=c,alarms=alarms)

def run():
    obs=load_model();out={}
    groups={'bench':{k:OUT/'bench'/k for k,_,_ in motions()['test']},
            'flight':{k:OUT/'records'/k/'strain' for k in CASES}}
    for dom,paths in groups.items():
        per={k:score_motion(obs,p,dom) for k,p in paths.items()}
        macro={name:float(np.mean([r['classifier']['marginal'][name]['balanced_accuracy'] for r in per.values()])) for name in ('offset','memory')}
        ds=[r['alarms'][s]['detection_delay_s'] for r in per.values() for s in ('offset','mixed')]
        n=sum(x is not None for x in ds);eligible=[float(x) for x in ds if x is not None]
        out[dom]=dict(macro_balanced_accuracy=macro,per_motion=per,
                      sustained_alarm=dict(positive_cases=len(ds),detected=n,min_delay_s=min(eligible) if n else None,
                       max_delay_s=max(eligible) if n else None,
                       false_episodes=sum(r['alarms'][s]['false_episodes'] for r in per.values() for s in STATES)))
        print(dom,macro,out[dom]['sustained_alarm'],flush=True)
    out['definitions']=dict(macro='Threshold 0.5, balance sensitivity and specificity within each motion, then give each held-out motion equal weight. Only the four predeclared in-scope profiles.',
        timing='Additional descriptive analysis: first rising sustained alarm after injection begins (marginal >=0.90 for five consecutive ADC samples). Onset and labels used only during scoring.')
    savej(OUT/'timing-reports.json',out)
if __name__=='__main__':run()
