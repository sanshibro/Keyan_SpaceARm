"""Select constant baseline and bias process parameters on one independent calibration flight."""
import argparse,dataclasses, itertools,json
import numpy as np
from augmented_bend_kf import RAD,Settings,estimate_clock
from round_aug_runtime import ROOT,OUT,PROTOCOL,SPEC,read_inputs,digest,write

def calibrate():
    case='241118-online_pause'
    assert case==SPEC['pilot'] and case not in SPEC['cases']
    labs=ROOT/'revision/data/round_shape_clearance/runs'/case/'scoring_truth.npz'
    with np.load(labs) as f:lt,lb=f['t'],f['mocap_b']
    inputs=[];vrms=[]
    for source in ['nominal','common','differential','single']:
        t,s,v=read_inputs(case,source,'natural')
        tr=np.column_stack([np.interp(t,lt,lb[:,j]) for j in range(2)])
        # An additional validation outage is independent of all three held-out streams.
        bad=(v['t']>=30.)&(v['t']<34.)
        v={k:a.copy() for k,a in v.items()}
        v['valid'][bad]=False;v['ready'][bad]=False;v['b'][bad]=np.nan;v['quality'][bad]=0.
        inputs.append((source,t,s,v,tr))
    # For a delayed camera, the motion bandwidth matters as much as the noise ratio.
    # Include an optical-dominant fixed filter as well as nominal-data noise levels.
    fixed=[dict(fixed_strain_sd=a,fixed_vision_sd=b,fixed_acceleration=q) for a,b,q in
           [(.5,.45,22),(.5,.9,22),(.9,.45,22),(1.4,.45,22),(.5,.9,4),(.9,.45,4),
            (1.4,.45,4),(3.,.25,4),(3.,.25,1.),(3.,.25,.4),(5.,.25,1.),(5.,.35,4.),(2.,.45,1.)]]
    # Near-zero process RW and coherent disagreement enable passive bias calibration on
    # nominal data and faster convergence when optical and material inverse observations diverge.
    adapt=[dict(bias_rw=a,active_bias_rw=b,bias_deadband=d) for a,b,d in [(.06,.70,1.1),(.10,.95,1.1),(.16,1.25,.85),(.22,1.35,1.05)]]
    # Evaluate RGB-D uncertainty scaling and motion bandwidth only on the optical pilot.
    for acc,rw,active in [(8.,.35,1.35),(4.,.62,1.6),(2.2,.60,1.55),(6.,.48,1.5)]:
        adapt.append(dict(acceleration=acc,bias_rw=rw,active_bias_rw=active,bias_deadband=1.05,vision_floor=.18,vision_scale=.09))
    def evaluate(params,aug):
        settings=Settings(**params);scores={}
        for name,t,s,v,truth in inputs:
            tr=estimate_clock(t,s,v,settings,aug)['b']
            for key,sel in [('full',(t>=14)&(t<46)),('steady',(t>=24)&(t<46)),('gap',(t>=30.20)&(t<34.20))]:
                dv=(tr[sel]-truth[sel])/RAD;dth=(np.linalg.norm(tr[sel],axis=1)-np.linalg.norm(truth[sel],axis=1))/RAD
                scores[name+'/'+key]=float(.7*np.sqrt(np.mean(dth**2))+.3*np.sqrt(np.mean(np.sum(dv**2,axis=1))))
        return dict(parameters=params,balanced_loss=float(np.mean(list(scores.values()))),per_condition=scores)
    allresults={};configuration={}
    for name,candidates in [('fixed',fixed),('augmented',adapt)]:
        scores=[evaluate(c,name=='augmented') for c in candidates];best=min(scores,key=lambda u:u['balanced_loss'])
        allresults[name]=scores;configuration.update(best['parameters']);print('Pilot selected',name,best['parameters'],best['balanced_loss'],flush=True)
    write(OUT/'pilot-selection.json',dict(case=case,labels_sha256=digest(labs),settings=dataclasses.asdict(Settings(**configuration)),candidates=allresults,
        scoring='Regular, common, differential and single conditions given equal weight. Each includes full, steady, and a withheld four-second camera interval. Each metric is 70% magnitude RMSE and 30% 2-D vector RMS. Same event clocks and front ends are used for both reviewed filters.'))
def freeze():
    dest=OUT/'frozen-settings.json'
    if dest.exists():raise FileExistsError('Settings already frozen')
    fit=json.loads((OUT/'pilot-selection.json').read_bytes())
    files=['mujoco_closed_loop/augmented_bend_kf.py','mujoco_closed_loop/continuum_strain_inverse.py','mujoco_closed_loop/anchored_rgbd.py',
           'revision/scripts/round_aug_runtime.py','revision/scripts/round_aug_geometry.py','revision/scripts/round_aug_vis.py',
           'revision/scripts/round_aug_strain_common.py','revision/scripts/round_aug_strain_generate.py','revision/scripts/round_aug_calibration.py']
    for p in sorted((ROOT/'revision/scripts').glob('round_aug_vis*.py')):
        if p.name!='round_aug_visuals.py' and str(p.relative_to(ROOT)) not in files:files.append(str(p.relative_to(ROOT)))
    frozen={p:digest(ROOT/p) for p in files}
    hashes={}
    for case in SPEC['cases']:
        for cond in SPEC['conditions']:
            for suffix in ('.npz','-ready.json'):
                p=OUT/'vision'/case/(cond+suffix);hashes[p.relative_to(OUT).as_posix()]=digest(p)
        for s in SPEC['sources']:
            p=OUT/'strain'/case/(s+'.npz');hashes[p.relative_to(OUT).as_posix()]=digest(p)
    write(dest,dict(protocol=digest(PROTOCOL),pilot_selection=digest(OUT/'pilot-selection.json'),settings=fit['settings'],
                     strain_model=digest(OUT/'strain-model/model.json'),software=frozen,inputs=hashes))
    print('FROZEN',digest(dest),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');args=p.parse_args();freeze() if args.freeze else calibrate()
