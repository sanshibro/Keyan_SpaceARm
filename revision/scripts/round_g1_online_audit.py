"""Score native flights and independently replay every rolling timing decision from sensor records."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import mujoco
import numpy as np
import pandas as pd
from pyulog import ULog
from scipy.spatial.transform import Rotation
from gate_shape_timing import RollingGateTiming, GateEnvelope, blend, enu_body_rotation
from round_g1_audit import score, save
from round_g1_distance import Solids, TOL
from round_f_plant import ENU_NED, LAUNCH, CLOCK_ORIGIN_US
from round_g1_model import Geometry,make_plant
from round_g1_online_run import DEST, OLD, P, PROTO, PLAN, Route, phase_time
from round_g1_plan import reference


def records(p):
    return [json.loads(x) for x in p.read_text().splitlines()]


def native_observations(folder,rows):
    u=ULog(str(folder/'flight.ulg'))
    a=u.get_dataset('vehicle_local_position').data
    b=u.get_dataset('vehicle_attitude').data
    t=a['timestamp'].astype(float)/1000
    tt=np.array([z['pos_ms'] for z in rows])
    origin=np.array(json.loads((folder/'run.json').read_text())['ekf_launch_origin_ned'])
    p=np.column_stack([np.interp(tt,t,a[k]) for k in ('x','y','z')])
    v=np.column_stack([np.interp(tt,t,a[k]) for k in ('vx','vy','vz')])
    err=float(np.abs((p-origin)@ENU_NED.T+LAUNCH-np.array([z['position'] for z in rows])).max())
    ev=float(np.abs(v@ENU_NED.T-np.array([z['velocity'] for z in rows])).max())
    att_ms=b['timestamp'].astype(float)/1000
    at=np.array([z['att_ms'] for z in rows])
    qq=np.column_stack([np.interp(at,att_ms,b['q[%d]'%j]) for j in (1,2,3,0)])
    rr=Rotation.from_quat(qq)
    observed=Rotation.from_euler('xyz',[z['rpy_frd_ned'] for z in rows])
    er=float(np.rad2deg((rr.inv()*observed).magnitude()).max())
    assert err<.02 and ev<.04 and er<.5,(err,ev,er)
    return dict(position_interp_max_m=err,velocity_interp_max_m_s=ev,attitude_interp_max_deg=er,
                exact_pos_stamp_count=int(sum(np.isin(tt,t))),exact_att_stamp_count=int(sum(np.isin(at,att_ms))))


def replay(folder,geom,route,mode):
    rows=records(folder/'timing.jsonl');pr=RollingGateTiming(geom,P)
    wall=pd.read_csv(folder/'hil.csv')
    tm=np.array([r['us'] for r in rows]);wi=np.searchsorted(wall.us.to_numpy(),tm)
    np.testing.assert_array_equal(np.diff(tm),200000)
    np.testing.assert_array_equal(wall.us.to_numpy()[wi],tm)
    np.testing.assert_allclose([r['measured_deg'] for r in rows],wall.net_deg.to_numpy()[wi],atol=1e-7)
    evs=[]
    for q in rows:
        obs=(np.array(q['position']),np.array(q['velocity']),np.array(q['rpy_frd_ned']),q['measured_deg'],q['age_pos_s'],q['age_att_s'])
        d=pr.update(q['t'],obs,route.preview(q['t'],pr.times))
        assert all(d[k]==q[k] for k in ('action','reason','candidates')),(folder.name,q['t'])
        assert q['bend_start']==pr.bend_start and q['return_start']==pr.return_start
        if d['action']!='retain':
            evs.append(dict(action=d['action'],t=q['t'],x=q['position'][0],rear=q['rear_m'],
                            body=q['body_now_m'],tool=q['arm_tool_now_m'],decision=d['reason']))
    lookup={q['action']:q['t'] for q in evs}
    times=[q['compute_ms'] for q in rows]
    return lookup,dict(accepted=evs,decisions=len(rows),candidate_evaluations=sum(len(q['candidates']) for q in rows),
             stale_ticks=sum(q['reason'] in ('stale_feedback','missing_feedback') for q in rows),
             blocked_candidate_ticks=sum(q['reason']=='no_valid_candidate' for q in rows),
             compute_p95_ms=float(np.percentile(times,95)),compute_max_ms=float(max(times)),
             max_feedback_age_ms=float(1000*max(max(q['age_pos_s'],q['age_att_s']) for q in rows)),
             native=native_observations(folder,rows))


def applied(method,condition,events):
    if method=='online':return events
    if method=='clock' or condition=='nominal':return {'bend':14.,'return':34.}
    return {'bend':14.,'return':34.+P['pause']['effective_delay_s']}


def offline_command(t,method,condition,events):
    if t is None:return 0.
    if method!='online':
        index=t if method=='clock' else phase_time(t,condition)
        return reference(index,PLAN,'variable_timed')[2]
    b=events.get('bend');r=events.get('return')
    q=0. if b is None else P['gate_angle_deg']*blend((t-b)/P['bend_duration_s'])
    if r is not None:q+=(P['terminal_angle_deg']-P['gate_angle_deg'])*blend((t-r)/P['return_duration_s'])
    return float(q)


def score_one(folder,env,solids):
    seed,variant=folder.name.split('-');method,condition=variant.split('_');route=Route(condition)
    outcome=json.loads((folder/'run.json').read_text())
    if outcome['error'] or outcome['release_us'] is None:
        return dict(case=folder.name,seed=int(seed),mode=variant,pass_all=False,failed='flight_transport'),{'error':outcome['error']}
    events,review=replay(folder,env,route,method)
    action=applied(method,condition,events)
    def fn(t):
        phase,pos,_=route.base(t);return phase,pos,offline_command(t,method,condition,events)
    row,detail=score(folder,{},expected_reference=fn,protocol_file=PROTO,plan_file=OLD/'plans.json',trace_dir=DEST/'traces')
    tr=pd.read_csv(DEST/'traces'/(folder.name+'.csv'))
    raw=dict(np.load(str(folder/'physics.npz')));release=outcome['release_us']
    wire=pd.read_csv(folder/'hil.csv')
    free=(wire.us>=release)&(wire.us<=release+52e6);tf=(wire.us[free].to_numpy()-release)/1e6
    if method=='online':
        b=events.get('bend');r=events.get('return')
        command=np.zeros(len(tf)) if b is None else P['gate_angle_deg']*blend((tf-b)/P['bend_duration_s'])
        if r is not None:command+=(P['terminal_angle_deg']-P['gate_angle_deg'])*blend((tf-r)/P['return_duration_s'])
        np.testing.assert_allclose(command,wire.request_deg[free].to_numpy(),atol=1e-8)
    exact={k:solids.scan(raw,tr,detail,k,release) for k in ('body','arm_tool')}
    assert all(v['upper_m']-v['lower_m']<=2*TOL for v in exact.values())
    row.update(method=method,condition=condition,body_convex_cm=100*exact['body']['upper_m'],
               arm_convex_cm=100*exact['arm_tool']['upper_m'],online_ticks=review['decisions'],
               compute_p95_ms=review['compute_p95_ms'],max_feedback_age_ms=review['max_feedback_age_ms'])
    for label in ('bend','return'):
        tt=action.get(label);row[label+'_t']=tt
        for col in ('x','rear_m','front_m','bend_deg'):
            row[label+'_'+col]=float(np.interp(tt,tr.t,tr[col])) if tt is not None else None
    detail.update(timing=review,convex=exact,executed_transition_times=action)
    detail['sha256']['timing.jsonl']=hashlib.sha256((folder/'timing.jsonl').read_bytes()).hexdigest()
    return row,detail


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--follow',action='store_true')
    ap.add_argument('--partial',action='store_true');args=ap.parse_args()
    (DEST/'traces').mkdir(parents=True,exist_ok=True);res=DEST/'per-run';res.mkdir(exist_ok=True)
    geom=Geometry(make_plant());env=GateEnvelope(geom,P['controller']['angle_pad_per_degree_m'])
    solids=Solids(geom);runs={};rows=[]
    for s,variants in P['seed_order'].items():
        for v in variants:
            name=s+'-'+v;folder=DEST/'runs'/name
            if args.follow:
                limit=time.monotonic()+240
                while not (folder/'flight.ulg').is_file() and time.monotonic()<limit:time.sleep(1.)
            if not (folder/'flight.ulg').is_file() and args.partial:continue
            row,detail=score_one(folder,env,solids)
            save(res/(name+'.json'),dict(row=row,detail=detail));rows.append(row);runs[name]=detail
            print(name,row['pass_all'],row['failed'],*(round(row[k],2) if row.get(k) is not None else None for k in
                ['body_convex_cm','arm_convex_cm','return_t','return_rear_m','target_time_s']),flush=True)
    if args.partial:return
    assert len(rows)==15
    for seed,variants in P['seed_order'].items():
        for cond in ('nominal','pause'):
            refs=[]
            for v in variants:
                if not v.endswith(cond):continue
                f=DEST/'runs'/(seed+'-'+v)
                w=pd.read_csv(f/'hil.csv');zero=json.loads((f/'run.json').read_text())['release_us']
                pick=(w.us>=zero)&(w.us<=zero+52e6);refs.append(w.loc[pick,['rx','ry','rz']].to_numpy())
            for q in refs[1:]:np.testing.assert_array_equal(q,refs[0])
    pd.DataFrame(rows).to_csv(DEST/'metrics.csv',index=False)
    save(DEST/'audit.json',dict(runs=runs,base_references_identical_within_condition=True,independent_flights=len(rows),
        nominal_phase_control='same command at every tick as the absolute clock; one shared baseline, no duplicate independent observation'))


if __name__=='__main__':main()
