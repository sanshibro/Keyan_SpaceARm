"""Explain each registered endpoint from the controller, verifier and original wrist records."""
import json
from pathlib import Path
import numpy as np
from m1_plates import DATA, OUT, read

def inspect(base, destination):
    result=[]
    for m in read(base/'matrix.json'):
        c=m['case'];d=base/c['case']
        plan=read(d/'plan/decision.json')
        with np.load(d/'flight/control.npz') as z,np.load(d/'audit/series.npz') as s:
            t=z['t'];dt=np.diff(t,append=t[-1]);active=z['phase']=='execute'
            holds={}
            for kind in ('self','clearance','tracking','feedback'):
                mask=active&(z['hold']==kind)
                if not mask.any():continue
                edges=np.diff(np.r_[False,mask,False].astype(int))
                begins=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
                duration=np.asarray([dt[i:j].sum() for i,j in zip(begins,ends)])
                k=int(np.argmax(duration));n=int(begins[k]);j=int(ends[k])
                # Exact self-distances in audit/series.npz are sampled at 5 Hz.
                ts=s['self_exact_t'];idx=np.searchsorted(t,ts).clip(0,len(t)-1)
                sample=mask[idx];common=(ts>=t[n])&(ts<=t[j-1]+1e-8)
                holds[kind]=dict(cumulative_s=float(dt[mask].sum()),longest_s=float(duration[k]),
                    interval_s=[float(t[n]),float(t[j-1])],body_x_m=float(s['body'][n,0]),
                    minimum_actual_self_during_holds_mm=float(1000*s['self_exact'][sample].min()) if sample.any() else None,
                    minimum_actual_self_during_longest_mm=float(1000*s['self_exact'][common].min()) if common.any() else None)
            checks={'geometry_body_60mm':m['clearance']['body']['lower']>=.06,'geometry_tool_60mm':m['clearance']['arm_tool']['lower']>=.06,
                    'positive_self_clearance':m['self_interval_50Hz_m'][0]>0,'contact_free':m['collision_steps']==0,
                    'task_dwell_1s':m['task_dwell_s']>=1.-1e-7,'wrist_depth_dwell_1s':m['optical_dwell_s']>=1.-1e-7}
            assert all(checks.values())==m['task_success']
            result.append(dict(case=c,route=plan['task_route'],success=m['task_success'],end=m['end_phase'],
                actual_end_x_m=float(m['end_xyz'][0]),checks=checks,hold=holds))
    destination.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    return result

if __name__=='__main__':
    inspect(DATA,OUT/'m1_execution_audit.json')
