"""Independent checks on installation geometry, instrument limits, timing and data separation."""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from round_slim_box import OUT, SPEC
from round_g1_model import make_plant, SimArmBridge
from round_slim_run import PRELOADS, label
from round_wrist_fusion import CASES, readj, savej, filter_bend
from round_shape_pixels import OUTPUT as OLD
from slim_sensor import SlimGaugeRig, SlimReadout

def main():
    plant=make_plant(); m,d=plant.m,plant.d
    rig=SlimGaugeRig(m,SPEC)
    p=readj(OUT/'box-profile.json')
    readout=SlimReadout(SPEC,p,np.load(OUT/'stationary-box.npz')['events'],721)
    np.testing.assert_allclose(readout.ratio([-.1,0,.5,1]),[0,0,.6386,1.5662]);assert np.isnan(readout.ratio(1.01))
    d.qpos[:]=m.qpos0
    SimArmBridge().set_kinematic_shape(m,d,np.deg2rad(93),np.deg2rad(62));mujoco.mj_kinematics(m,d)
    base=rig.lengths(d)
    q=Rotation.from_euler('xyz',[30,16,-53],degrees=True).as_quat()
    d.qpos[:3]+=[1.4,-.7,.3];d.qpos[3:7]=q[[3,0,1,2]];mujoco.mj_kinematics(m,d)
    residual=float(np.max(abs(base-rig.lengths(d))));assert residual<1e-10
    s=np.load(OUT/'sweep.npz');sweeps={}
    for i,f in enumerate(PRELOADS):
        e=s['extension'][:,:,i,:];np.testing.assert_allclose(e[0],f,atol=1e-6)
        raw=p['neutral_codes']+(readout.ratio(e)-readout.ratio(f))*np.array(p['slope_counts_per_unit_ratio'])
        low=raw<.5
        ind=np.flatnonzero(low.any(axis=(1,2)));sweeps[label(f)]={'first_nominal_low_code_bend_deg':int(s['bend_deg'][ind[0]]) if len(ind) else None,
                'low_code_120_deg_channel_plane_samples':int(low[-1].sum()),'total_120_deg_channel_plane_samples':int(low[-1].size)}
    old=np.load(OLD/'slam/textured'/CASES[0]/'natural-estimate.npz')
    ob=readj(OLD/'bench/scores.json')['validation']
    sigma=[max(.5,ob['theta_rmse_deg']),max(1.,ob['phi_rmse_deg'])]
    ref=filter_bend(old['t'],old['b'][0],old['capture'],old['visual_b'],old['quality'],sigma)
    same=filter_bend(old['t'],old['b'][0],old['capture'],old['visual_b'],old['quality'],sigma,
                     strain_fresh=np.ones(len(old['t']),bool))
    for a,b in zip(ref,same):np.testing.assert_array_equal(a,b)
    np.testing.assert_allclose(ref[0],old['b'][3],atol=1e-10)
    train=readj(OUT/'p30/predictor.json')['training_sessions'];assert len(set(train))==8 and all(k.startswith('train-') for k in train)
    checks={}
    for case in CASES:
        a=np.load(OUT/'primary'/case/'strain.npz');z=np.load(OUT/'primary'/case/'occluded.npz')
        j=np.searchsorted(a['t'],a['t_tick']+1e-8,side='right')-1
        np.testing.assert_array_equal(a['fresh'],np.r_[True,np.diff(j)>0])
        muted=(z['t']>=31.2-1e-8)&(z['t']<35.2-1e-8)
        assert not np.any(z['vision_weight'][muted]) and not np.isfinite(z['raw_b'][2,muted]).any()
        assert np.isfinite(z['raw_b'][3,muted]).all()
        valid=(a['t']>=14)&(a['t']<46)
        low=(a['adc']==0).any(1);taut=(a['strain']>0).all(1)
        checks[case]=dict(analysis_box_updates=int(valid.sum()),analysis_low_code=int((low&valid).sum()),
                 analysis_low_code_all_taut=int((low&valid&taut).sum()),analysis_slack=int((a['slack'].any(1)&valid).sum()),
                 analysis_above_range=int((a['overrange'].any(1)&valid).sum()),analysis_hz=float(valid.sum()/32),
                 covered_output_ticks=int(muted.sum()),covered_finite_fused=int(np.isfinite(z['raw_b'][3,muted]).all(1).sum()),
                 covered_valid_slam=int(np.isfinite(z['raw_b'][2,muted]).all(1).sum()))
    savej(OUT/'audit.json',dict(rigid_transform_residual_m=residual,registered_preloads=PRELOADS,adc_sweep=sweeps,flight=checks,
        legacy_all_fresh_matches_default=True,train_motion_ids=train,box_curve_checked=True,
        virtual_fourth_channel='independent simulated material path; median effective gain and independent stationary-noise donor; no active fourth reference channel'))
    print(sweeps,checks,flush=True)
if __name__=='__main__':main()
