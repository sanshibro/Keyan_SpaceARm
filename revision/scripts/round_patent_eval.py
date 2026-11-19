"""Paired replay on four frozen trajectories; only the scorer opens motion-capture labels."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from patent_confidence import angles, fuse
from round_patent_camera import OUT
from round_shape_pixels import OUTPUT
from round_wrist_fusion import CASES, DELAY, filter_bend, metrics, readj, savej

OPTICS=('natural','graded','occluded','native-long')
GAUGES=('nominal','zero_offset','material_memory')
METHODS=('strain','vision','previous','adaptive','corrected')
PHASES={'all':(14,46),'clear':(14,26),'weak':(27.3,31.2),'blink':(31.2,31.52),
        'same_four_seconds':(31.2,35.2),'restored':(36,46)}
def packet(z):
    return [z[k] for k in ('t','b','fresh','vs','age')]
def optical(v):
    return [v[k] for k in ('capture','visual_b','cv','stable_teacher')]
def evaluate(task):
    case,gname=task
    obs=OUT/'observations'/case; profile=readj(OUT/'visual-calibration.json')
    dest=OUT/'traces'/case;dest.mkdir(parents=True,exist_ok=True)
    with np.load(obs/(gname+'.npz')) as f:g=dict(f)
    t=g['t']
    for name in OPTICS:
        with np.load(obs/(name+'.npz')) as f:v=dict(f)
        s=packet(g); o=optical(v)
        if name==OPTICS[0]:strain=fuse(*s,*o,profile,mode='strain')
        adaptive=fuse(*s,*o,profile,mode='both')
        corrected=fuse(*s,*o,profile,mode='teacher')
        prev,prev_plane,legacy_share=filter_bend(t,g['b'],v['capture'],v['visual_b'],v['legacy_q'],[.5,1.],strain_fresh=g['fresh'])
        cap=v['capture']; last=np.searchsorted(cap+DELAY,t+1e-7,side='right')-1
        ix=np.maximum(0,last); has=(last>=0)&(t-cap[ix]<DELAY+.061)
        vis=np.full_like(prev,np.nan);vis[has]=v['visual_b'][ix[has]]
        estimates=np.stack((strain['b'],vis,prev,adaptive['b'],corrected['b']))
        planes=np.stack((strain['plane'],angles(vis)[:,0]>=5,prev_plane,adaptive['plane'],corrected['plane']))
        np.savez_compressed(dest/(gname+'-'+name+'.npz'),t=t,b=estimates,plane=planes,
            joint=corrected['joint_vis_share'],old_share=legacy_share,accepted=corrected['visible_update'],
            sigma=corrected['posterior_sigma'],bias=corrected['bias'],changes=corrected['correction_log'],
            pairs=corrected['teacher_pairs'])
    print(case,gname,flush=True)
def score():
    all_scores={};by_case={};changes={};recovery={}
    for gauge in GAUGES:
        all_scores[gauge]={}
        for condition in OPTICS:
            data=[]
            for case in CASES:
                key=gauge+'/'+condition+'/'+case
                folder=OUT/'traces'/case
                with np.load(folder/(gauge+'-'+condition+'.npz')) as f:z=dict(f)
                # These labels are never supplied to an estimator.
                with np.load(OUTPUT/'runs'/case/'scoring_truth.npz') as f:truth=f['mocap_b']
                p=(z['t']>=14)&(z['t']<46)
                by_case[key]={name:metrics(z['b'][i,p],truth[p],z['plane'][i,p]) for i,name in enumerate(METHODS)}
                changes[key]=dict(accepted_rejected_rollback=z['changes'][-1].tolist(),pairs=int(z['pairs']))
                if gauge=='nominal':
                    v=np.load(OUT/'observations'/case/(condition+'.npz'));t=z['t']
                    end={'graded':31.32,'occluded':35,'native-long':35}.get(condition)
                    if end:
                        rows=np.flatnonzero((v['capture']>=end-1e-7)&(v['cv']>0))
                        j=int(rows[0]) if len(rows) else None
                        if j is not None:
                            tick=np.searchsorted(t,v['capture'][j]+DELAY-1e-7)
                            jump=float(angles(z['b'][4,tick:tick+1])[0,0]-angles(z['b'][4,tick-1:tick])[0,0])
                        else:jump=None
                        recovery[condition+'/'+case]=dict(first_original_frame_capture_s=None if j is None else float(v['capture'][j]),
                            delay_from_uncover_to_release_s=None if j is None else round(float(v['capture'][j]+DELAY-end),4),
                            additional_registration=bool(v['bridge'][j]) if j is not None else False,theta_step_deg=jump)
                data.append((z,truth))
            phase_table={}
            for phase,(lo,hi) in PHASES.items():
                mask=[(z['t']>=lo-1e-8)&(z['t']<hi-1e-8) for z,_ in data]
                ref=np.concatenate([truth[k] for (_,truth),k in zip(data,mask)])
                shared=np.concatenate([np.isfinite(z['b'][1,k]).all(1) for (z,_),k in zip(data,mask)])
                methods={}
                for i,m in enumerate(METHODS):
                    b=np.concatenate([z['b'][i,k] for (z,_),k in zip(data,mask)])
                    plane=np.concatenate([z['plane'][i,k] for (z,_),k in zip(data,mask)])
                    methods[m]=dict(all=metrics(b,ref,plane),common=metrics(b[shared],ref[shared],plane[shared]))
                joint=np.concatenate([z['joint'][k,0] for (z,_),k in zip(data,mask)])
                old=np.concatenate([z['old_share'][k] for (z,_),k in zip(data,mask)])
                ok=np.isfinite(joint);methods['weight']=dict(common_ticks=int(ok.sum()),
                    adaptive_mean=None if not ok.any() else float(np.mean(joint[ok])),
                    previous_mean=None if not ok.any() else float(np.mean(old[ok])))
                phase_table[phase]=methods
            all_scores[gauge][condition]=phase_table
    savej(OUT/'paired-scores.json',all_scores);savej(OUT/'case-scores.json',by_case)
    savej(OUT/'teacher-and-recovery.json',dict(teachers=changes,recovery=recovery))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--jobs',type=int,default=3);a.add_argument('--score-only',action='store_true');opts=a.parse_args()
    if not opts.score_only:
        with ProcessPoolExecutor(opts.jobs) as pool:list(pool.map(evaluate,[(c,g) for c in CASES for g in GAUGES]))
    score()
