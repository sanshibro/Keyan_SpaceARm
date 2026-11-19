"""Independent checks of acquisition clocks, geometry, held-out data and the estimator boundary."""
import ast,hashlib, inspect,json, os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from round_g1_model import Geometry,make_plant
from round_shape_pixels import OUTPUT,EVIDENCE,ROOT
from round_shape_sensors import MARKERS,mocap_bend,features,predict,vector_angles
from round_shape_replay import place_only_estimated,geometry_points,clearance
from round_shape_score import classify,reserve
from round_wrist_fusion import CASES,METHODS,filter_bend,readj,savej
from gate_shape_timing import enu_body_rotation

def check():
    test={}
    source=Path.home()/'桌面/Test/dual_strain_ai/run_confidence_ekf_fusion.py'
    local=Path(__file__).with_name('round_shape_theta_reference.py')
    if source.exists():
        def symbols(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_bytes()).body if isinstance(n,ast.FunctionDef)}
        u,v=symbols(source),symbols(local)
        assert all(u[k]==v[k] for k in ('run_ekf','ekf_update'))
        test['reference_kernel_source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    rot=Rotation.from_rotvec([0.,.72,0.]).as_matrix();frame=Rotation.from_rotvec([.26,-.31,1.1]).as_matrix()
    markers=np.array([MARKERS,MARKERS@rot.T+[.2,0.,.24]])
    q=mocap_bend(markers);assert np.allclose(q,[.72,0],atol=1e-12)
    assert np.allclose(mocap_bend(markers@frame.T+[2.,.3,-1.]),q,atol=1e-12)
    test['mocap_global_rotation_translation_invariance']=True
    model=readj(OUTPUT/'bench/predictor.json');assert all(s.startswith('train-') for s in model['training_sessions'])
    files=sorted((OUTPUT/'bench').glob('*.npz'));split={s:sum(f.stem.startswith(s+'-') for f in files) for s in ('train','validation','test')}
    assert split=={'train':8,'validation':2,'test':3}
    test['bench_splits']=split;test['predictor_sha256']=hashlib.sha256((OUTPUT/'bench/predictor.json').read_bytes()).hexdigest()
    plain=pd.read_csv(OUTPUT/'slam/plain/241118-online_pause/natural.csv')
    test['plain_first_6s']={'frames':len(plain),'tracking_ok':int((plain.state==2).sum())}
    assert not (set(inspect.signature(filter_bend).parameters)&{'truth','reference','issued','command'})
    checks={}
    for case in CASES:
        checks[case]={}
        for cond in ('natural','occluded'):
            d=OUTPUT/'slam/textured'/case;z=np.load(str(d/(cond+'-estimate.npz')))
            t=z['t'];b=z['b'];w=z['vision_weight_on_delivery']
            tab=pd.read_csv(d/(cond+'.csv'))
            assert len(tab)==850 and np.isfinite(b[[0,1,3]]).all()
            if cond=='occluded':
                block=(t>=31.22)&(t<35.20)
                assert np.all(w[block]==0) and not np.isfinite(b[2,block]).any()
                recovered=readj(d/(cond+'-diagnostics.json'))['first_valid_after_restore_s']
            else:recovered='not applicable'
            checks[case][cond]=dict(camera_frames=len(tab),camera_obscured=int(tab.occluded.sum()),
                reacquired_original_map_s=recovered,maximum_estimated_fused_bend_deg=float(vector_angles(b[3])[0].max()))
        # Delayed updates may never leak backwards into an already published estimate.
        zz=np.load(str(d/'natural-estimate.npz'));cut=1300
        sel=(zz['capture']+.20<=zz['t'][cut-1]+1e-8)
        val=readj(OUTPUT/'bench/scores.json')['validation']
        sigma=[max(.5,val['theta_rmse_deg']),max(1.,val['phi_rmse_deg'])]
        out,_,_=filter_bend(zz['t'][:cut],zz['b'][0,:cut],zz['capture'][sel],zz['visual_b'][sel],zz['quality'][sel],sigma)
        assert np.allclose(out,zz['b'][3,:cut],atol=1e-10)
    test['causal_prefix_matches_full_run']=True;test['flight_verification']=checks
    return test

def geometry():
    # Apply the earlier screen without fitting a new buffer to these markerless cases.
    pad=readj(OUTPUT/'frozen-pads.json')['pad_m']['fusion']
    geom=Geometry(make_plant());out={}
    for cond in ('natural','occluded'):
        source={m:[] for m in METHODS};ref=[];tcp={m:[] for m in METHODS}
        for case in CASES:
            old=OUTPUT/'runs'/case;native=np.load(str(old/'natural.npz'));state=np.load(str(old/'observations.npz'))
            z=np.load(str(OUTPUT/'slam/textured'/case/(cond+'-estimate.npz')))
            k=native['state_index'];ids=np.flatnonzero((native['t']>=26)&(native['t']<42))
            np.testing.assert_allclose(native['t'],z['t'][k],atol=1e-7)
            ref.append(reserve(native['reference_clear'][ids]))
            for h,m in enumerate(METHODS):
                rows=[]
                for i in ids:
                    b=z['b'][h,k[i]]
                    xyz=state['px4_xyz'][i];r=enu_body_rotation(state['px4_rpy'][i])
                    if place_only_estimated(geom,xyz,r,b):
                        rows.append(reserve(clearance(geom)))
                        xyz_est,_=geometry_points(geom)
                        tcp[m].append(1000.*np.linalg.norm(xyz_est[-1]-native['reference_world'][i,-1]))
                    else:rows.append(np.full(len(native['heights']),np.nan))
                source[m].append(np.stack(rows))
        truth=np.stack(ref);out[cond]={}
        for m in METHODS:
            vals=np.stack(source[m])
            out[cond][m]=dict(frame_raw=classify(truth,vals,0),frame_with_prior_pad=classify(truth,vals,pad),
                route_raw=classify(truth.min(1),vals.min(1),0),route_with_prior_pad=classify(truth.min(1),vals.min(1),pad),
                tcp_on_available_rmse_mm=float(np.sqrt(np.mean(np.square(tcp[m])))))
    return dict(counterfactual_sill_heights_m=[0.,.16,.005],prior_empirical_pad_m=pad,
                scope='Paired replay of four trajectories, 80 instants, 33 related sill positions. Conservative CAD envelopes. Not new flights or a universal safety guarantee.',
                conditions=out)

if __name__=='__main__':
    savej(OUTPUT/'slam/audit.json',check());savej(OUTPUT/'slam/geometry.json',geometry())
