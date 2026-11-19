"""Freeze single-seed pads before held-out scoring. Keep geometry decisions separate from physical contacts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

from round_shape_pixels import OUTPUT,SPEC
from round_shape_replay import METHODS,CONDITIONS,HEIGHTS
from round_shape_sensors import vector_angles,wrap_deg
from round_shape_bench import errors


def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def cases(split):
    return [f'{seed}-{m}' for seed in SPEC['cases'][split] for m in SPEC['cases']['methods']]


def window(z):
    return (z['t']>=26)&(z['t']<42)


def reserve(clear):
    return np.min(clear-np.array([.10,.07]).reshape(2,1),axis=-2)


def calibrate():
    selected=cases('calibration')
    result={}
    for method in METHODS:
        maxima=[]
        for case in selected:
            with np.load(str(OUTPUT/'runs'/case/'natural.npz')) as z:
                valid=window(z);h=METHODS.index(method)
                diff=reserve(z['clear'][h,valid])-reserve(z['reference_clear'][valid])
                maxima.append(float(np.nanmax(diff)))
        result[method]=max(0.,max(maxima))+.001
    out={'fit_cases':selected,'condition':'natural','pad_m':result,
         'meaning':'Empirical maximum overestimate of whole-body clearance reserve plus 1 mm; not a guaranteed hardware margin.'}
    f=OUTPUT/'frozen-pads.json'
    if f.exists():
        assert json.loads(f.read_bytes())==out
    else:dump(f,out)
    print(out,flush=True)


def classify(reference,estimate,pad):
    truth=reference>=-1e-12
    known=np.isfinite(estimate);accept=known&(estimate>=pad-1e-12)
    return dict(total=int(truth.size),reference_pass=int(truth.sum()),est_available=int(known.sum()),
                unknown_hold=int((~known).sum()),safe_released=int(np.sum(accept&truth)),
                false_pass=int(np.sum(accept&~truth)),false_reject_observed=int(np.sum(known&~accept&truth)),
                conservative_or_unknown_hold_on_safe=int(np.sum(~accept&truth)))


def basic_accuracy(b,btrue):
    mask=np.isfinite(b).all(axis=1)
    if not mask.any():return {'n':0,'coverage':0.}
    return dict(coverage=float(mask.mean()),**errors(b[mask],btrue[mask]))


def summarize():
    padfile=OUTPUT/'frozen-pads.json';frozen=json.loads(padfile.read_bytes())
    cases_test=cases('held_out')
    assert not set(cases_test)&set(frozen['fit_cases'])
    summary={};rows=[];witness={}
    for cond in CONDITIONS:
        summary[cond]={};witness[cond]={}
        zs=[dict(np.load(str(OUTPUT/'runs'/case/(cond+'.npz')))) for case in cases_test]
        for h,method in enumerate(METHODS):
            truth=np.concatenate([z['reference_b'] for z in zs])
            predicted=np.concatenate([z['bend'][h] for z in zs])
            shared=np.concatenate([np.isfinite(z['bend'][2]).all(1) for z in zs])
            world=np.concatenate([z['world'][h]-z['reference_world'] for z in zs])
            local=np.concatenate([z['local'][h]-z['reference_local'] for z in zs])
            tcp=np.linalg.norm(world[:,-1],axis=-1)*1000
            interior=np.linalg.norm(world[:,:-1],axis=-1).max(axis=-1)*1000
            interior_shape=np.linalg.norm(local[:,:-1],axis=-1).max(axis=-1)*1000
            mins=[];preds=[];dclear=[];bad=[]
            for case,z in zip(cases_test,zs):
                keep=window(z);t=z['t'][keep]
                rr=reserve(z['reference_clear'][keep]);pp=reserve(z['clear'][h,keep])
                mins.append(rr);preds.append(pp)
                dclear.extend(1000*(z['clear'][h,keep,1,0]-z['reference_clear'][keep,1,0]))
                badmask=(pp>=0)&(rr<0)
                if np.any(badmask):
                    i,k=np.unravel_index(np.argmax(np.where(badmask,pp-rr,-np.inf)),pp.shape)
                    bad.append(dict(case=case,t=float(t[i]),sill_raise_m=float(HEIGHTS[k]),
                                    true_reserve_mm=float(1000*rr[i,k]),estimated_reserve_mm=float(1000*pp[i,k])))
                for ix,ti in enumerate(t):
                    rows.append([cond,method,case,float(ti),float(rr[ix,0]),float(pp[ix,0])])
            rr=np.stack(mins);pp=np.stack(preds)
            def pair_metrics(a,b):
                return dict(raw=classify(a,b,0),empirical_pad=classify(a,b,frozen['pad_m'][method]))
            visible=np.isfinite(tcp)
            entry=dict(all_frames=basic_accuracy(predicted,truth),common_visible_frames=basic_accuracy(predicted[shared],truth[shared]),
                       world_tcp_rmse_mm=float(np.sqrt(np.nanmean(tcp**2))),
                       world_tcp_common_rmse_mm=float(np.sqrt(np.mean(tcp[shared]**2))),
                       world_interior_max_point_p95_mm=float(np.nanpercentile(interior,95)),
                       mount_interior_max_point_p95_mm=float(np.nanpercentile(interior_shape,95)),
                       nominal_arm_clearance_error_rmse_mm=float(np.sqrt(np.nanmean(np.square(dclear)))),
                       pad_m=frozen['pad_m'][method],decision_frames=pair_metrics(rr,pp),
                       decision_routes=pair_metrics(rr.min(axis=1),pp.min(axis=1)),
                       nominal_route=pair_metrics(rr[:,:,0].min(axis=1),pp[:,:,0].min(axis=1)))
            summary[cond][method]=entry
            if bad:witness[cond][method]=max(bad,key=lambda b:b['estimated_reserve_mm']-b['true_reserve_mm'])
    # Compare pose quality at capture and at delivery without fitting any delay to the scoring truth.
    diagnostics={}
    for case in cases_test:
        folder=OUTPUT/'runs'/case
        with np.load(str(folder/'natural.npz')) as z,np.load(str(folder/'observations.npz')) as obs:
            packets=obs['wrist_packets'];good=packets[:,4].astype(bool)
            true=z['reference_b'];pix=packets[:,-2:]
            measured=basic_accuracy(pix[good],true[good])
            delivery_idx=np.minimum(np.searchsorted(z['time_50'],packets[:,1]+.2-1e-8),len(z['time_50'])-1)
            later=z['reference_b_50'][delivery_idx]
            diagnostics[case]=dict(detected=int(good.sum()),captured=len(good),
                                   at_capture=measured,against_later_truth=basic_accuracy(pix[good],later[good]))
    with (OUTPUT/'gate-score-traces.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['condition','estimator','case','t_s','recorded_reserve_m','estimated_reserve_m'])
        w.writerows(rows)
    output=dict(held_out_cases=cases_test,pad_fingerprint=hashlib.sha256(padfile.read_bytes()).hexdigest(),results=summary,
                selected_counterexamples=witness,vision_timing=diagnostics,decision_window_s=[26,42],
                selected_rule='Largest estimated minus recorded conservative reserve among actual threshold flips per estimator.')
    dump(OUTPUT/'scores.json',output)
    for method in METHODS:
        e=summary['natural'][method]
        print(method,'theta/phi/TCP',round(e['all_frames']['theta_rmse_deg'],3),
              round(e['all_frames']['phi_rmse_deg'],3),round(e['world_tcp_rmse_mm'],2),
              'false pass raw/pad',e['decision_frames']['raw']['false_pass'],e['decision_frames']['empirical_pad']['false_pass'],
              'complete route accepted',e['decision_routes']['raw']['safe_released'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=('calibrate','score'))
    a=p.parse_args()
    if a.action=='calibrate':calibrate()
    else:summarize()
