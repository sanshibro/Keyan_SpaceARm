"""Match coupled trajectories at the same x and compare dynamics within a common pipe window."""
import json
from pathlib import Path
import numpy as np
def interp(x,values,station):
    order=np.argsort(x);xs=x[order];y=values[order];xs,idx=np.unique(xs,return_index=True)
    return np.interp(station,xs,y[idx])
def measure(root,out):
    seeds=(26112161,26112167,26112173);rows=[]
    for seed in seeds:
        paired={}
        for strategy in ('coupled','unregularized','sequential'):
            base=root/(strategy+'-'+str(seed));d=np.load(base/'audit/series.npz')
            q=np.load(base/'plan/certified.npz');ref=np.load(base/'plan/body-reference.npy')
            t=d['t'];x=d['body'][:,0];valid=(t>=21)&(d['phase']=='execute')
            if not valid.any():continue
            # Each profile is interpolated only inside its own recorded x extent.
            xv=x[valid];yb=d['body'][valid,1];start=max(.8,xv.min(),ref[0,0]);end=min(3.05,xv.max(),ref[-1,0])
            if start>=end:continue
            stations=np.linspace(start,end,191)
            # First forward passage selects one measured position per station and
            # avoids mixing the return from a stop with the inbound displacement.
            first_ids=[np.flatnonzero(xv>=s)[0] for s in stations]
            measured=yb[first_ids];reference=interp(ref[:,0],ref[:,1],stations)
            difference=measured-reference;j=int(np.argmax(abs(difference)))
            planned=interp(q['q'][:,0],q['q'][:,1],stations)-reference
            k=int(np.argmax(abs(planned)));sub=valid&(x>=.80)&(x<=1.92)
            b=d['actual_b'][sub];a=np.rad2deg(np.linalg.norm(b,axis=1));yaw=np.rad2deg(d['yaw'][sub])
            phi=np.rad2deg(np.unwrap(np.arctan2(b[:,1],b[:,0])));phi=phi[a>=15]
            osc=float(np.ptp(phi)) if len(phi)>1 else None
            paired[strategy]=dict(max_signed_departure_cm=float(100*difference[j]),departure_x_m=float(stations[j]),
               max_signed_planned_departure_cm=float(100*planned[k]),planned_departure_x_m=float(stations[k]),
               body_y_range_cm=float(100*np.ptp(yb)),pipe_t=np.asarray([t[sub][0],t[sub][-1]]).tolist(),
               theta_range_deg=[float(a.min()),float(a.max())],phi_range_deg=osc,
               yaw_range_deg=float(np.ptp(yaw)),pipe_tracking_peak_cm=float(100*d['tracking'][sub].max()),
               pipe_thrust_diff_rms_N=float(np.sqrt(np.mean(d['front_rear'][sub]**2))),
               pipe_root_peak_Nm=float(np.max(np.linalg.norm(d['mount_reaction'][sub,:2],axis=1))),
               pipe_tilt_peak_deg=float(d['tilt'][sub].max()),certified_duration_s=float(q['t'][-1]))
        rows.append(dict(seed=seed,strategies=paired))
    (out/'m1_mechanism.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n',encoding='utf8')
    return rows
