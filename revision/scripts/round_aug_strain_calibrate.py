"""Calibrate one continuous material inverse on independent bench motions only.

No target flight is opened. Full motions define train/validation/test partitions.
Train and validation use independently perturbed material parameters. The prior
test-00..02 motions became development diagnostics when they exposed ADC clipping.
Fresh independent post-freeze testing is in round_aug_strain_validate.py.
"""
import argparse
from pathlib import Path
import numpy as np
from slim_sensor import memory_components
from continuum_strain_inverse import ContinuousStrainInverse,polynomial,fit_polynomial
from round_aug_strain_common import DESIGN,PLANT_MEMORY,environment,savej,sha,material_for,sample_geometry

def load_bench(path):
    # This reader is deliberately restricted to independent bench split paths.
    if not path.parent.name.startswith(('train-','validation-','test-')):
        raise ValueError('Only bench motions may provide calibration labels')
    with np.load(path,allow_pickle=False) as f:
        return {k:f[k] for k in ('t','length_m','mocap_b')}

def interpolate(ts,t,b):
    return np.column_stack([np.interp(ts,t,b[:,j]) for j in range(b.shape[1])])

def stats(b,truth):
    err=np.degrees(b-truth)
    theta=np.degrees(np.linalg.norm(b,axis=1)-np.linalg.norm(truth,axis=1))
    return dict(n=len(b),theta_rmse_deg=float(np.sqrt(np.mean(theta**2))),
                b_vector_rmse_deg=float(np.sqrt(np.mean(np.sum(err**2,axis=1)))),
                b_component_rmse_deg=np.sqrt(np.mean(err**2,axis=0)).tolist(),
                theta_p95_abs_deg=float(np.quantile(abs(theta),.95)))

def make_record(folder,spec,profile,noise,variation):
    ref=load_bench(folder/'reference.npz');key=folder.name
    material=material_for('bench:'+key,variation)
    raw,meta=sample_geometry('bench:'+key,ref['t'],ref['length_m'],spec,profile,noise,material=material)
    truth=interpolate(raw['t'],ref['t'],ref['mocap_b'])
    return dict(key=key,ref=ref,raw=raw,truth=truth,physical=meta)

def fit_material(records,spec,profile):
    curve=spec['resistance_curve_25C'];gain=np.asarray(profile['slope_counts_per_unit_ratio'])
    preload=spec['preload_primary'];neutral=np.interp(preload,curve['engineering_strain'],curve['relative_resistance_change'])
    data=[]
    for rec in records:
        ref,raw=rec['ref'],rec['raw']
        ratio=np.interp(ref['length_m']/spec['free_active_length_m']-1.,
                        curve['engineering_strain'],curve['relative_resistance_change'])
        measured=neutral+(raw['adc']-raw['tare'])/gain
        nominal=interpolate(raw['t'],ref['t'],ratio)
        data.append((ref['t'],ratio,raw['t'],measured-nominal))
    target=np.vstack([d[3] for d in data]);best=[(np.inf,None)]*4
    # The grid specifies physical timescale/width hypotheses, not health classes.
    # Per-channel continuous branch amplitudes are fitted by least squares.
    for h in (.014,.016,.018,.020,.022,.024,.026):
        for tau in (1.2,1.5,1.8,2.1,2.4,2.7,3.):
            ps=[];ss=[]
            for t,r,ts,_ in data:
                play,slow=memory_components(t,r,h,tau)
                ps.append(interpolate(ts,t,play));ss.append(interpolate(ts,t,slow))
            P,S=np.vstack(ps),np.vstack(ss)
            for j in range(4):
                x=np.column_stack((P[:,j],S[:,j]));w=np.linalg.lstsq(x,target[:,j],rcond=None)[0]
                if not (0<=w[0]<.90 and 0<=w[1]<=.10):continue
                mse=np.mean((x@w-target[:,j])**2)
                if mse<best[j][0]:best[j]=(mse,(h,tau,*w))
    if any(v[1] is None for v in best):raise ValueError('Unidentifiable material calibration')
    keys=('half_width','tau_s','play_weight','transient_weight')
    return {k:[float(item[1][i]) for item in best] for i,k in enumerate(keys)},[float(x[0]) for x in best]

def calibrate(args):
    out=Path(args.output);model_dir=out/'strain-model';model_dir.mkdir(parents=True,exist_ok=True)
    if (model_dir/'freeze.json').exists():raise FileExistsError('Frozen model exists; use a new output for a changed calibration')
    bench=Path(args.bench);spec,profile,noise=environment(args.slim,args.protocol)
    savej(model_dir/'generation-design.json',DESIGN)
    folders={split:sorted(bench.glob(split+'-*')) for split in ('train','validation','test')}
    if not all(folders.values()):raise ValueError('Need complete independent bench partitions')
    records={}
    for split in ('train','validation'):
        records[split]=[make_record(f,spec,profile,noise,DESIGN['calibration_material_fractional_variation']) for f in folders[split]]
    memory,fit_mse=fit_material(records['train'],spec,profile)
    print('Material fitted from independent train ADC/geometry:',memory,flush=True)
    curve=spec['resistance_curve_25C']
    model=dict(schema=1,method='Single causal play-plus-relaxation inverse and cubic opposing-pair shape map',
        gains_counts_per_ratio=profile['slope_counts_per_unit_ratio'],curve_strain=curve['engineering_strain'],
        curve_ratio=curve['relative_resistance_change'],preload=spec['preload_primary'],calibrated_memory=memory,
        shape_coefficients=np.zeros((10,2)).tolist(),even_coefficients=np.zeros((10,2)).tolist(),
        base_covariance_rad2=(np.eye(2)*np.deg2rad(.30)**2).tolist(),common_residual_scale=[.001,.001],
        nominal_sample_s=float(np.median(noise[:,0])))
    # Shape fit uses only corrected ADC; labels enter here only as bench supervision.
    inverse=ContinuousStrainInverse(model);X=[];Y=[];C=[]
    for rec in records['train']:
        raw=rec['raw'];z=inverse.estimate(raw['t'],raw['adc'],raw['tare'])
        e=z['estimated_strain'];d=.5*(e[:,:2]-e[:,2:]);c=.5*(e[:,:2]+e[:,2:])
        X.append(polynomial(d));Y.append(rec['truth']);C.append(c)
    x,y,c=np.vstack(X),np.vstack(Y),np.vstack(C)
    model['shape_coefficients']=fit_polynomial(x,y).tolist()
    model['even_coefficients']=fit_polynomial(x,c).tolist()
    geometric_x=np.vstack([polynomial(r['ref']['mocap_b']) for r in records['train']])
    geometric_y=np.vstack([r['ref']['length_m']/spec['free_active_length_m']-1. for r in records['train']])
    model['geometry_forward_coefficients']=fit_polynomial(geometric_x,geometric_y).tolist()
    residual=c-x@np.asarray(model['even_coefficients'])
    model['common_residual_scale']=np.maximum(np.sqrt(np.mean(residual**2,axis=0)),.0008).tolist()
    inverse=ContinuousStrainInverse(model);errors=[];val_rows={}
    for rec in records['validation']:
        raw=rec['raw'];z=inverse.estimate(raw['t'],raw['adc'],raw['tare'])
        errors.append(z['b']-rec['truth']);val_rows[rec['key']]=stats(z['b'],rec['truth'])
    error=np.vstack(errors);cov=error.T@error/len(error)
    # Retain a physical/model-mismatch floor rather than ADC quantization alone.
    eig,vec=np.linalg.eigh(cov);cov=(vec*np.maximum(eig,np.deg2rad(.30)**2))@vec.T
    model['base_covariance_rad2']=cov.tolist()
    inverse=ContinuousStrainInverse(model)
    # Frozen covariance calibration uses validation only; preserve off-diagonal terms.
    normalized=[]
    for rec in records['validation']:
        raw=rec['raw'];z=inverse.estimate(raw['t'],raw['adc'],raw['tare']);err=z['b']-rec['truth']
        normalized.extend(np.einsum('ni,nij,nj->n',err,np.linalg.inv(z['R']),err))
    coverage_scale=max(1.,float(np.quantile(normalized,.95)/5.991464547))
    model['base_covariance_rad2']=(cov*coverage_scale).tolist()
    model_path=model_dir/'model.json';savej(model_path,model)
    source_dir=Path(__file__).parent
    inverse_path=Path(__import__('continuum_strain_inverse').__file__)
    input_hashes={str(p.relative_to(bench)):sha(p) for split in ('train','validation')
                  for f in folders[split] for p in (f/'reference.npz',)}
    freeze=dict(model_sha256=sha(model_path),configuration_sha256=sha(model_dir/'generation-design.json'),
        material_parameter_fit='Grid width/time plus continuous linear amplitudes on training ADC and geometry only.',
        fit_mse_ratio2=fit_mse,training=[p.name for p in folders['train']],validation=[p.name for p in folders['validation']],
        inputs=input_hashes,code={p.name:sha(p) for p in [Path(__file__),inverse_path,source_dir/'round_aug_strain_common.py']},
        forbidden_target_inputs='No target flight, target mocap, flight label, target error or future ADC was opened.',
        validation_covariance_scale=coverage_scale,
        covariance_units='radian squared; second moment includes systematic residuals; consecutive samples can remain correlated')
    savej(model_dir/'freeze.json',freeze)
    # No numerical fitting after the freeze. These prior test motions are now
    # development diagnostics, not untouched evidence: test-00 exposed an ADC-rail
    # implementation issue. New independent motions are generated by validate.py.
    inverse=ContinuousStrainInverse(model_path);test_rows={};perturbed_rows={};prefix_max=0.
    all_errors=[];all_ref=[];all_point=[];all_R=[];phase={k:[[],[]] for k in ('hold','motion','return')}
    for folder in folders['test']:
        for variation,name in [(DESIGN['calibration_material_fractional_variation'],'nominal'),
                               (DESIGN['heldout_material_fractional_variation'],'material_transfer')]:
            rec=make_record(folder,spec,profile,noise,variation);raw=rec['raw']
            z=inverse.estimate(raw['t'],raw['adc'],raw['tare']);row=stats(z['b'],rec['truth'])
            (test_rows if name=='nominal' else perturbed_rows)[rec['key']]=row
            dest=out/'strain-bench'/rec['key'];dest.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest/(name+'.npz'),t=raw['t'],adc=raw['adc'],tare=raw['tare'],**z)
            np.savez_compressed(dest/(name+'-scoring.npz'),t=raw['t'],mocap_b=rec['truth'])
            if name=='nominal':
                all_errors.append(z['b']);all_ref.append(rec['truth']);all_point.append(z['point_b']);all_R.append(z['R'])
                cut=len(raw['t'])//2
                prefix=inverse.estimate(raw['t'][:cut],raw['adc'][:cut],raw['tare'])
                prefix_max=max(prefix_max,float(np.max(abs(prefix['b']-z['b'][:cut]))),float(np.max(abs(prefix['R']-z['R'][:cut]))))
                radial=np.gradient(np.linalg.norm(rec['truth'],axis=1),raw['t'])
                for label,mask in [('hold',abs(radial)<.025),('motion',abs(radial)>=.025),('return',radial<-.05)]:
                    phase[label][0].append(z['b'][mask]);phase[label][1].append(rec['truth'][mask])
    est,ref,point,covs=map(np.vstack,(all_errors,all_ref,all_point,all_R));e=est-ref
    nees=np.einsum('ni,nij,nj->n',e,np.linalg.inv(covs),e)
    report=dict(model_sha256=sha(model_path),validation=val_rows,nominal_diagnostic_by_motion=test_rows,
        diagnostic_material_perturbation_15_percent_by_motion=perturbed_rows,pooled_diagnostic_nominal=stats(est,ref),
        same_static_shape_map_without_material_inverse=stats(point,ref),
        by_motion_phase={k:stats(np.vstack(v[0]),np.vstack(v[1])) for k,v in phase.items()},
        nominal_joint95_coverage=float(np.mean(nees<=5.991464547)),causal_prefix_max_abs=prefix_max,
        scope='Synthetic development diagnostics on prior test-00..02 with empirical box noise/timing. '
              'These motions exposed the ADC-rail defect and are not untouched tests. '
              'Fresh post-freeze evidence is independent-bench-validation.json. Generated material law '
              'is conditional and is not an identification of new physical strips. No target flight scores used.')
    savej(model_dir/'bench-validation.json',report)
    print('FROZEN',sha(model_path),'DIAGNOSTIC',report['pooled_diagnostic_nominal'],'prefix',prefix_max,flush=True)

if __name__=='__main__':
    root=Path(__file__).resolve().parents[2];p=argparse.ArgumentParser()
    p.add_argument('--output',default=root/'revision/data/round_augmented_fusion')
    p.add_argument('--bench',default=root/'revision/data/round_strain_health/bench')
    p.add_argument('--slim',default=root/'revision/data/round_slim_sensor')
    p.add_argument('--protocol',default=root/'revision/protocols/round_slim_protocol.json')
    calibrate(p.parse_args())
