"""Open independent evaluation markers only after all sensor-only output trajectories are sealed."""
import json
from collections import defaultdict
import numpy as np
from round_aug_runtime import OUT,ROOT,SPEC,METHODS,digest,write
from augmented_bend_kf import RAD

def metrics(b,ref):
    valid=np.isfinite(b).all(1);th=np.linalg.norm(ref,axis=1)/RAD;pred=np.linalg.norm(b,axis=1)/RAD
    plane=valid&(th>=5.)&(pred>=5.)
    phi=(np.arctan2(b[plane,1],b[plane,0])-np.arctan2(ref[plane,1],ref[plane,0]))/RAD
    phi=(phi+180.)%360.-180.
    def rms(e):return float(np.sqrt(np.mean(e**2))) if len(e) else None
    return dict(n=len(ref),valid=int(valid.sum()),coverage=float(valid.mean()) if len(ref) else 0.,
                theta=rms(pred[valid]-th[valid]),phi=rms(phi),phi_n=int(plane.sum()),phi_eligible=int(np.sum(th>=5)),
                vector=rms(np.linalg.norm(b[valid]-ref[valid],axis=1)/RAD))
def combine(groups):
    r={}
    for key,pairs in groups.items():
        ref=np.concatenate([v[1] for v in pairs]);b=np.concatenate([v[0] for v in pairs],axis=1)
        r[key]={m:metrics(b[i],ref) for i,m in enumerate(METHODS)}
    return r
def mask(t,lo,hi):return (t>=lo-1e-8)&(t<hi-1e-8)
def score():
    lock=json.loads((OUT/'release-lock.json').read_bytes());freeze=json.loads((OUT/'frozen-settings.json').read_bytes())
    assert digest(OUT/'frozen-settings.json')==lock['frozen_settings']
    assert len(lock['files'])==len(SPEC['cases'])*len(SPEC['conditions'])*len(SPEC['sources'])
    for name,h in {**lock['files'],**lock['sensor_packets']}.items():assert digest(OUT/name)==h,name
    for name,h in lock['sensor_packets'].items():assert freeze['inputs'][name]==h,name
    group=defaultdict(list);equal=defaultdict(list);per={};optical={};recovery={};audit={};labels={}
    for case in SPEC['cases']:
        origin=ROOT/'revision/data/round_strain_health/records'/case/'strain/reference.npz'
        labels[case]=digest(origin)
        with np.load(origin) as f:t0,ref=f['t'],f['mocap_b']
        for cond in SPEC['conditions']:
            with np.load(OUT/'vision'/case/(cond+'.npz')) as f:cap=f['t'];obs=f['b'];usable=f['valid']
            optical_ref=np.column_stack([np.interp(cap,t0,ref[:,i]) for i in range(2)])
            for ph,(lo,hi) in SPEC['phases'].items():
                sel=mask(cap,lo,hi);op=obs[sel].copy();op[~usable[sel]]=np.nan
                optical[case+'/'+cond+'/'+ph]=metrics(op,optical_ref[sel])
            for source in SPEC['sources']:
                tag=source+'/'+cond;seq=np.load(OUT/'released'/case/(source+'-'+cond+'.npz'))
                with seq as seq:tr={k:seq[k] for k in seq.files}
                t=tr['t'];idx=np.searchsorted(t0,t)
                assert np.array_equal(t,t0[np.minimum(idx,len(t0)-1)])
                truth=ref[idx];b=tr['b'];dst=OUT/'scored'/case/(source+'-'+cond+'.npz')
                dst.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dst,truth=truth,**tr)
                for ph,(lo,hi) in SPEC['phases'].items():
                    ids=mask(t,lo,hi);k=tag+'/'+ph;part=(b[:,ids],truth[ids]);group[k].append(part)
                    if source in SPEC['primary_offsets']:group['drift/'+cond+'/'+ph].append(part)
                    visible=ids&np.isfinite(b[0]).all(1);equal[k].append((b[:,visible],truth[visible]))
                    per[case+'/'+k]={m:metrics(part[0][j],part[1]) for j,m in enumerate(METHODS)}
                if cond=='occluded':
                    blind=mask(t,*SPEC['phases']['blind']);d=tr['bias'][blind]
                    if tr['accepted'][blind].any() or np.any(tr['gain_v'][blind]) or np.ptp(d,axis=0).any():
                        raise AssertionError('Visual update or change in committed bias during the lost stream: '+case+'/'+source)
                    restore=np.flatnonzero((t>=35.2-1e-8)&tr['accepted'])
                    raw=np.flatnonzero((cap>=35-1e-8)&usable)
                    i=int(restore[0]) if len(restore) else None
                    step=(np.diff(np.linalg.norm(b[3],axis=1))-np.diff(np.linalg.norm(truth,axis=1)))/RAD
                    watch=mask(t[1:],t[i]-1e-8,t[i]+.6) if i is not None else np.zeros(len(t)-1,bool)
                    pre=mask(t,26.2,31.2)
                    recovery[case+'/'+source]=dict(optical_lock_s=None if not len(raw) else float(cap[raw[0]]+.2-35.),
                      first_accepted_s=None if i is None else float(t[i]-35.),post_lock_peak_excess_step_deg=float(np.max(abs(step[watch]))) if watch.any() else None,
                      blackout_points=int(blind.sum()),vision_updates=int(tr['accepted'][blind].sum()),
                      bias_change_during_blackout_rad=float(np.ptp(d,axis=0).max()),
                      mean_bias_before_blackout_deg=(np.mean(tr['bias'][pre],axis=0)/RAD).tolist(),
                      accepted_before=int(tr['accepted'][pre].sum()))
    pooled=combine(group);common=combine(equal)
    write(OUT/'scores.json',dict(cases=pooled,per_flight=per,common_visible=common,phase_bounds=SPEC['phases'],optical_at_exposure=optical,
                 recovery=recovery,truth_hash=labels,release_sha256=digest(OUT/'release-lock.json'),unit='degrees; pooled RMSE recomputed from samples, phi eligible at theta>=5 deg'))
    checks=[]
    def check(label,value,op,limit,case='pooled',required=True):
        good=value is not None and {'le':lambda:value<=limit+1e-12,'lt':lambda:value<limit,'ge':lambda:value+1e-12>=limit,'eq':lambda:abs(value-limit)<1e-12}[op]()
        checks.append(dict(metric=label,value=value,operation=op,limit=limit,scope=case,required=required,passed=bool(good)))
    for case in ['pooled']+SPEC['cases']:
        p=pooled if case=='pooled' else {k[len(case)+1:]:v for k,v in per.items() if k.startswith(case+'/')}
        hard=case=='pooled'
        n=p['nominal/natural/all'];check('healthy proposed theta RMSE',n['proposed']['theta'],'le',.9,case,hard)
        check('healthy strain theta RMSE',n['strain']['theta'],'le',1.2,case,hard)
        for cond,ph in [('natural','all'),('occluded','recovered')]:
            check('vision '+cond+' '+ph+' theta RMSE at release',p['nominal/'+cond+'/'+ph]['vision']['theta'],'le',1.8,case,hard)
        for src in ['nominal']+SPEC['primary_offsets']:
            q=p[src+'/occluded/blind'];check(src+' blind visual coverage',q['vision']['coverage'],'eq',0.,case,hard)
            check(src+' blind proposed coverage',q['proposed']['coverage'],'eq',1.,case,hard)
            if src=='nominal':continue
            clean=p[src+'/natural/steady']
            check(src+' steady proposed RMSE',clean['proposed']['theta'],'le',1.5,case,hard)
            check(src+' blind proposed RMSE',q['proposed']['theta'],'le',2.,case,hard)
            for who in ('strain','fixed'):
                red=1.-q['proposed']['theta']/q[who]['theta']
                check(src+' blind reduction vs '+who,red,'ge',.5,case,hard)
                check(src+' steady lower than '+who,clean['proposed']['theta'],'lt',clean[who]['theta'],case,hard)
    for case in SPEC['cases']:
        for src in ['nominal']+SPEC['primary_offsets']:
            rec=recovery[case+'/'+src];check(src+' optical reacquisition s',rec['optical_lock_s'],'lt',.6,case)
            check(src+' first accepted s',rec['first_accepted_s'],'lt',.6,case)
            check(src+' post-lock step residual degrees',rec['post_lock_peak_excess_step_deg'],'le',2.,case)
    checks.append(dict(metric='fused prefix equality',passed=bool(lock['prefix']) and all(v==0 for cases in lock['prefix'].values() for row in cases.values() for v in row.values()),required=True))
    out=dict(passed=all(q['passed'] for q in checks if q['required']),checks=checks)
    write(OUT/'acceptance.json',out)
    print('PASS' if out['passed'] else 'OPEN GATES',flush=True)
    for c in checks:
        if not c['passed'] and c['required']:print(c,flush=True)
if __name__=='__main__':score()
