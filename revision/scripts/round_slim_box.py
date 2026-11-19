"""Derive timing and effective readout sensitivity from historical box sessions.

Historical vision is a commissioning reference only; it never enters the simulated strain-angle fit.
"""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from round_g1_model import ROOT
from round_wrist_fusion import savej,readj
SPEC=readj(ROOT/'revision/protocols/round_slim_protocol.json')
OUT=ROOT/'revision/data/round_slim_sensor'
FILES=Path.home()/'桌面/Test/dual_strain_ai/three_strain_final_v2/data'

def reference_holds(data):
    blocks=data.phase.ne(data.phase.shift())|data.point_index.ne(data.point_index.shift())|data.repeat_index.ne(data.repeat_index.shift())
    cols=['orb_theta_deg','orb_fai_deg','ch1','ch2','ch3']
    rows=[]
    for _,group in data.groupby(blocks.cumsum(),sort=False):
        if group.phase.iloc[0]!='holding':continue
        tail=group[group.rel_time>=group.rel_time.iloc[-1]-.3]
        tail=tail[(tail.usable==1)&(tail.tracking_state==2)].dropna(subset=cols)
        if len(tail)<5:continue
        theta=tail.orb_theta_deg.median()
        if theta<5:continue
        phi=np.rad2deg(np.angle(np.mean(np.exp(1j*np.deg2rad(tail.orb_fai_deg)))))
        rows.append([theta,phi,*tail[['ch1','ch2','ch3']].median().values])
    if len(rows)<15:raise ValueError('Too few independent commanded holds')
    return np.array(rows)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    ids=SPEC['source_sessions'];held=SPEC['hardware_session_holdout'];meta={};dfs={};noise=[]
    for name in [*ids,held]:
        f=FILES/name/'dual_strain_auto.csv';df=pd.read_csv(f);dfs[name]=df
        base=df[df.phase=='baseline']
        active=base[['ch1','ch2','ch3']].to_numpy(float);center=np.median(active,axis=0)
        assert np.all(base[['ch4','ch5','ch6']].to_numpy()==4095)
        delta=np.diff(base.rel_time);assert np.all(delta>0)
        meta[name]=dict(n=len(base),dt_median_s=float(np.median(delta)),arrival_hz=float(1/np.mean(delta)),
                        neutral_codes=center.tolist(),noise_sd_code=active.std(axis=0).tolist(),
                        source_sha256=hashlib.sha256(f.read_bytes()).hexdigest())
        if name in ids:noise.append(np.column_stack([delta,active[1:]-center]))
    quiet=np.concatenate(noise)
    np.savez_compressed(OUT/'stationary-box.npz',events=quiet)
    cal=ids[-1];A=reference_holds(dfs[cal]);B=reference_holds(dfs[held])
    curve=SPEC['resistance_curve_25C'];ex=np.array(curve['engineering_strain']);ey=np.array(curve['relative_resistance_change'])
    preload=SPEC['preload_primary'];offset=np.interp(preload,ex,ey)
    def variation(rows,angle):
        strain=preload+(1+preload)*SPEC['arm_radius_m']/.28*np.deg2rad(rows[:,0])*np.cos(np.deg2rad(rows[:,1])-angle)
        return np.interp(strain,ex,ey)-offset
    slopes=[];checks=[]
    for k in range(3):
        a=A[:,2+k]-meta[cal]['neutral_codes'][k];b=B[:,2+k]-meta[held]['neutral_codes'][k]
        def res(params):return params[0]*variation(A,params[1])-a
        trials=[least_squares(res,[1800.,ang],bounds=([0.,-4*np.pi],[50000.,4*np.pi]),loss='soft_l1',f_scale=25.)
                for ang in np.arange(-np.pi,np.pi,np.pi/2)]
        best=min(trials,key=lambda s:s.cost);g,ang=best.x
        pred=g*variation(B,ang)
        if not 0<g<49999:raise ValueError('Readout sensitivity is unidentifiable')
        slopes.append(float(g))
        checks.append(dict(sensor=k+1,effective_slope=float(g),n_fit=len(a),n_check=len(b),equivalent_angle_deg=float(np.rad2deg(ang)%360),
                           holdout_rmse_counts=float(np.sqrt(np.mean((pred-b)**2))),holdout_r2=float(1.-np.sum((pred-b)**2)/np.sum((b-b.mean())**2))))
    four=lambda v:[*np.asarray(v,dtype=float).tolist(),float(np.median(v))]
    c=meta[cal]['neutral_codes'];end=dfs[cal][dfs[cal].phase=='final_zero'][['ch1','ch2','ch3']].median().values
    profile=dict(slope_counts_per_unit_ratio=four(slopes),neutral_codes=four(c),stationary_sessions=ids,
            calibration_session=cal,transfer_check_session=held,transfer_check=checks,session_stats=meta,
            offset_stress_counts=four(end-c),interpretation='Effective installed gain inferred from optical angle and nominal radius. Hardware session transfer is checked independently. Offset stress is the observed end-minus-start code change and may also include pose change. Channel 4 borrows the median gain and a pooled recorded idle-noise trace.')
    savej(OUT/'box-profile.json',profile)
    print([(round(c['effective_slope']),round(c['holdout_r2'],3),round(c['holdout_rmse_counts'],1)) for c in checks],
          'raw Hz',[round(meta[s]['arrival_hz'],1) for s in [*ids,held]],flush=True)
if __name__=='__main__':main()
