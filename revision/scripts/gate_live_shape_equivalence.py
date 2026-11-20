"""Offline sample-for-sample audit against frozen batch code, without truth labels."""
import argparse
import inspect
import time
from pathlib import Path
import numpy as np
import augmented_bend_kf as frozen_kf
from continuum_strain_inverse import ContinuousStrainInverse
from round_aug_strain_common import material_for,DESIGN,sample_geometry
from gate_live_shape import FrozenAssets,StreamingStrainInverse,StreamingBox,LiveEstimator,writej,digest


def packet(path,keys):
    with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in keys}

def difference(a,b,name,atol=2e-11):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape or not np.array_equal(np.isnan(a),np.isnan(b)):
        raise AssertionError((name,'shape or NaN mismatch',a.shape,b.shape))
    if a.dtype.kind=='b':
        if not np.array_equal(a,b):raise AssertionError(name)
        return 0.
    finite=np.isfinite(a)&np.isfinite(b)
    result=float(np.max(abs(a[finite]-b[finite]))) if finite.any() else 0.
    if result>atol:raise AssertionError((name,result,atol))
    return result

def instrument_reference():
    # Instrument the unchanged frozen routine to expose full posterior diagnostics.
    # Neither state transition nor event ordering is altered by these observations.
    source=inspect.getsource(frozen_kf.estimate_clock)
    source=source.replace('for k,now in enumerate(t):',
        'full_x=np.zeros((n,6 if augmented else 4));full_P=np.zeros((n,6 if augmented else 4,6 if augmented else 4))\n    for k,now in enumerate(t):')
    source=source.replace('ls=advance(live,now,s,ls)',
        'ls=advance(live,now,s,ls)\n        full_x[k]=live.x;full_P[k]=live.P')
    source=source.replace('return dict(b=z,','return dict(full_x=full_x,full_P=full_P,b=z,')
    scope=dict(frozen_kf.__dict__);exec(compile(source,'<read-only frozen posterior instrumentation>','exec'),scope)
    return scope['estimate_clock']

def run(args):
    a=FrozenAssets(args.root,args.data,args.slim,args.protocol,not args.local)
    base=a.data;case=args.case
    model_before=digest(a.model_path);query=packet(base/'geometry'/(case+'.npz'),('t','length_m'))
    s=packet(base/'strain'/case/'differential.npz',('t','adc','tare','initial_adc','b','R'))
    v=packet(base/'vision'/case/'occluded.npz',('t','b','R','valid','quality','ready'))
    report=dict(model_sha256=model_before,adapter_sha256=digest(Path(__file__).with_name('gate_live_shape.py')),
                reference_commit='881a799',source='Independent old pilot sensor packets and sanitized geometry; no truth fields read.')
    # Linear-time stateful inverse versus original two-loop frozen inverse.
    before=time.perf_counter();stream=StreamingStrainInverse(a.model,s['tare'])
    rows=[stream.step(t,adc) for t,adc in zip(s['t'],s['adc'])]
    elapsed=time.perf_counter()-before
    batch=ContinuousStrainInverse(a.model).estimate(s['t'],s['adc'],s['tare'],s['initial_adc'])
    report['inverse']={k:difference(np.asarray([r[k] for r in rows]),batch[k],k) for k in batch}
    report['inverse_ms_per_adc']=1000*elapsed/len(rows)
    # Physical generator is fed current integrated surface spans one step at a time.
    for condition in ('nominal','differential'):
        box=StreamingBox(a.spec,a.profile,a.noise,case,condition,float(query['t'][-1])+1e-6,float(query['t'][0]))
        actual=[]
        for t,L in zip(query['t'],query['length_m']):actual.extend(box.observe_lengths(t,L))
        material=material_for('flight:'+case,DESIGN['heldout_material_fractional_variation'])
        expected,_=sample_geometry(case,query['t'],query['length_m'],a.spec,a.profile,a.noise,condition,material)
        report['physical_'+condition]=dict(t=difference([r['t'] for r in actual],expected['t'],'physical t',1e-12),
            adc=difference([r['adc'] for r in actual],expected['adc'],'physical ADC',0.),
            tare=difference(box.tare,expected['tare'],'tare',0.))
    # Original batch inference and online inference see the exact same release clock.
    clock=query['t'];online=LiveEstimator(a.settings,t0=float(clock[0]));si=vi=0
    before=time.perf_counter()
    for now in clock:
        ss=[];vv=[]
        while si<len(s['t']) and s['t'][si]<=now+1e-9:
            ss.append({k:s[k][si] for k in ('t','b','R')});si+=1
        while vi<len(v['t']) and v['t'][vi]+a.settings.delay<=now+1e-9:
            vv.append({k:v[k][vi] for k in v});vi+=1
        online.step(now,ss,vv)
    report['filter_ms_per_query']=1000*(time.perf_counter()-before)/len(clock)
    reference=instrument_reference();sensor={k:s[k] for k in ('t','b','R')}
    for name,augmented in [('fixed',False),('proposed',True)]:
        ref=reference(clock,sensor,v,a.settings,augmented)
        fields=dict(b='b',cov='P',bias='bias',state='full_x',P='full_P',gain_v='gain_v',gain_s='gain_s',
                    confidence_v='confidence_v',confidence_s='confidence_s',accepted='accepted',nis='nis')
        report[name]={k:difference([r[k] for r in online.rows[name]],ref[rk],name+'/'+k) for k,rk in fields.items()}
    report['vision']=difference([r['b'] for r in online.rows['vision']],frozen_kf.vision_release(clock,v,a.settings.delay),'vision')
    report['strain']=difference([r['b'] for r in online.rows['strain']],frozen_kf.strain_release(clock,sensor),'strain')
    if model_before!=digest(a.model_path):raise AssertionError('Frozen model changed')
    report['n_adc']=len(s['t']);report['n_query']=len(clock);report['n_visual']=len(v['t'])
    report['complexity']=('One physical/material transition per physics step; one inverse per fresh ADC; '
        'master advances each ADC once and live replays only the fixed transport-delay interval per arriving image. '
        'Append-only records are linear storage, never refitted or rescanned in the flight loop.')
    writej(args.output,report);print(report,flush=True)

if __name__=='__main__':
    root=Path(__file__).resolve().parents[2];p=argparse.ArgumentParser()
    p.add_argument('--root',default=root);p.add_argument('--data');p.add_argument('--slim');p.add_argument('--protocol')
    p.add_argument('--case',default='241118-online_pause');p.add_argument('--local',action='store_true')
    p.add_argument('--output',default=root/'revision/data/round_gate_closed_loop/adapter-equivalence.json')
    run(p.parse_args())
