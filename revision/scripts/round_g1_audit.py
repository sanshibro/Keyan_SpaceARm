"""Independently score physical poses against four gate solids and the shared tool target."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import mujoco
import numpy as np
import pandas as pd
import rosbag
from pyulog import ULog
from round_f_audit import hashes, wire
from round_g1_model import Geometry, make_plant, DATA, D, LAUNCH, P, PROTOCOL, TARGET, AXIS, SHIFT
from round_g1_plan import reference, wide_reference

MODES=('straight','fixed','variable_nominal','variable_timed')


def save(p, x):
    p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def read(p):
    return json.loads(p.read_text())


def surface_points(g, i, direction):
    """Feasible points on each native convex shape. An upper bound is a real point-to-box distance."""
    m,d=g.m,g.d
    R=d.geom_xmat[i].reshape(3,3)
    center=d.geom_xpos[i]
    h=m.geom_size[i]
    typ=m.geom_type[i]
    if typ==mujoco.mjtGeom.mjGEOM_MESH:
        mesh=int(m.geom_dataid[i]);a=int(m.mesh_vertadr[mesh])
        v=m.mesh_vert[a:a+m.mesh_vertnum[mesh]].astype(float)
    else:
        axes=np.vstack((np.eye(3),-np.eye(3),direction))
        local=axes @ R
        local/=np.maximum(1e-12,np.linalg.norm(local,axis=1,keepdims=True))
        if typ==mujoco.mjtGeom.mjGEOM_BOX:
            v=np.vstack((np.sign(local)*h, np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])*h))
        elif typ==mujoco.mjtGeom.mjGEOM_CAPSULE:
            v=h[0]*local;v[:,2]+=h[1]*np.sign(local[:,2])
        elif typ==mujoco.mjtGeom.mjGEOM_SPHERE:
            v=local*h[0]
        elif typ==mujoco.mjtGeom.mjGEOM_CYLINDER:
            xy=local[:,:2];unit=xy/np.maximum(1e-12,np.linalg.norm(xy,axis=1,keepdims=True))
            v=np.column_stack([unit*h[0],h[1]*np.sign(local[:,2])])
        elif typ==mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            v=local*h*h/np.maximum(1e-12,np.linalg.norm(local*h,axis=1,keepdims=True))
        else:
            raise ValueError('Unsupported surface %s'%typ)
    return v@R.T+center


def witness(g, q, cat):
    g.from_record(q)
    lo,hi=g.bounds()
    distances=g.distances(lo,hi)
    sel=g.cat==cat if cat!='arm_tool' else g.cat!='body'
    distances[~sel]=np.inf
    pairs=np.argsort(distances,axis=None)
    best=math.inf; lower=float(distances.flat[pairs[0]])
    ans=None
    for ij in pairs:
        if distances.flat[ij]>=best or not np.isfinite(distances.flat[ij]):break
        j,k=np.unravel_index(ij,distances.shape)
        i=int(g.ids[j])
        center=g.d.geom_xpos[i];direction=np.clip(center,g.glo[k],g.ghi[k])-center
        points=surface_points(g,i,direction)
        wall=np.clip(points,g.glo[k],g.ghi[k]);ds=np.linalg.norm(points-wall,axis=1)
        ix=int(np.argmin(ds))
        if ds[ix]<best:
            best=float(ds[ix])
            ans=dict(geom=g.m.geom(i).name,gate=g.gates[k][0],robot=points[ix].tolist(),wall=wall[ix].tolist())
    return dict(lower_m=lower,upper_m=best,**ans)


def dwell(t, passed):
    ids=np.flatnonzero(passed)
    if not len(ids):return 0.,None
    chunks=np.split(ids,np.flatnonzero(np.diff(ids)>1)+1)
    max_d=max(float(t[part[-1]]-t[part[0]]) for part in chunks)
    qualifying=[t[v[0]]+P['terminal_dwell_s'] for v in chunks
                if t[v[-1]]-t[v[0]]>=P['terminal_dwell_s']-1e-9]
    return max_d, (float(min(qualifying)) if qualifying else None)


def log_check(folder, plant, a, w, ix, release, deadline, mode, seed, plan):
    stamps=a['us'];us=w.us.to_numpy(np.int64)
    assert np.all(np.diff(us)==4000) and np.all(np.diff(stamps)==20000)
    np.testing.assert_array_equal(us[ix],stamps)
    motors=w[['m1','m2','m3','m4']].values
    np.testing.assert_allclose(a['rotor_N'],12*motors[ix],atol=1e-7)
    np.testing.assert_allclose(a['ctrl'][:,plant.rotors],a['rotor_N'],atol=1e-7)
    np.testing.assert_allclose(a['qpos'][:,:3]-LAUNCH,w[['ex','ey','ez']].values[ix],atol=1e-8)
    joints=a['qpos'][:,plant.arm.qpos_ids].reshape(-1,8,2)
    np.testing.assert_allclose(np.degrees(np.linalg.norm(joints,axis=2).sum(axis=1)),w.total_deg.values[ix],atol=1e-7)
    fn=wide_reference() if mode=='wide' else lambda tt:reference(tt,plan,mode)
    for n in range(0,len(ix),11):
        tr=(stamps[n]-release)/1e6
        _,pos,deg=fn(None if tr<0 else tr)
        np.testing.assert_allclose(pos,a['ref'][n],atol=1e-8)
        assert abs(deg-w.request_deg.values[ix[n]])<1e-7
    prov=read(folder/'provenance.json')
    assert prov['plant_xml_sha256']==hashlib.sha256(plant.xml.encode()).hexdigest()
    assert prov['protocol_sha256']==hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    assert prov['mission']['plan_sha256']==hashlib.sha256((DATA/'plans.json').read_bytes()).hexdigest()
    assert (prov['mission']['seed'],prov['mission']['mode'])==(seed,mode)
    u=ULog(str(folder/'flight.ulg'));get=lambda n:u.get_dataset(n).data
    gps=get('vehicle_gps_position')
    zero=int(np.median(gps['time_utc_usec'].astype(np.int64)-gps['timestamp'].astype(np.int64)-4000))
    assert zero==prov['clock_origin_us']
    out=get('actuator_outputs');pts=out['timestamp'].astype(np.int64)+zero
    mask=(pts>=release)&(pts<=release+deadline*1e6);j=np.searchsorted(us,pts[mask])
    np.testing.assert_array_equal(us[j],pts[mask])
    native=np.clip((np.column_stack([out['output[%d]'%k][mask] for k in range(4)])-1000)/1000,0,1)
    motor_res=float(np.abs(native-motors[j]).max())
    assert motor_res<1e-5
    free=(us>=release)&(us<=release+deadline*1e6)
    status=get('vehicle_status')
    j=np.searchsorted(status['timestamp'].astype(np.int64)+zero,us[free],side='right')-1
    assert np.all(status['nav_state'][j]==14) and np.all(status['arming_state'][j]==2) and not np.any(status['failsafe'][j])
    assert np.all(w.armed[free]) and np.all(w.offboard[free])
    fields={'/clock':stamps[:,None],'/f/evaluation/mujoco_position':a['qpos'][:,:3]-LAUNCH,
            '/f/plant/rotor_thrust_N':a['rotor_N'],'/f/reference/position':a['ref']}
    counts={n:0 for n in fields}
    residual=0.
    with rosbag.Bag(str(folder/'flight.bag')) as bag:
        for n,msg,stamp in bag.read_messages(topics=list(fields)):
            k=counts[n]; assert stamps[k]==stamp.to_nsec()//1000
            obs=([msg.clock.to_nsec()//1000] if n=='/clock' else [msg.point.x,msg.point.y,msg.point.z]
                 if hasattr(msg,'point') else msg.data)
            residual=max(residual,float(abs(np.asarray(obs)-fields[n][k]).max()))
            counts[n]+=1
    assert residual<1e-7 and all(k==len(stamps) for k in counts.values())
    return dict(ros_residual=residual,ulog_motor_residual=motor_res,ulog_dropouts=len(u.dropouts),
                bag_sample_count=len(stamps),native_actuator_samples=int(sum(mask)),source_commit=prov['source_commit'])


def score(folder, plans):
    case=folder.name;seed,mode=case.split('-');seed=int(seed);wide=mode=='wide'
    scene='wide' if wide else 'study'
    plant=make_plant(scene);g=Geometry(plant,scene)
    np.testing.assert_allclose(plant.mass,2.25,atol=1e-10)
    w=wire(folder);a=dict(np.load(str(folder/'physics.npz')))
    meta=read(folder/'run.json')
    if meta['error'] or meta['release_us'] is None:
        return dict(case=case,seed=seed,mode=mode,pass_all=False,error=meta['error']),dict(meta=meta)
    release=meta['release_us'];deadline=39. if wide else P['deadline_s']
    stamps=a['us'];ix=np.searchsorted(w.us.values,stamps)
    logs=log_check(folder,plant,a,w,ix,release,deadline,mode,seed,None if wide else plans[mode])
    t=(stamps-release)/1e6
    keep=np.flatnonzero((t>=0)&(t<=deadline))
    if wide:
        end=np.load(str(D/'receiver_trajectory.npz'))['states'][-1,:3]+SHIFT
    data=[]
    for k in keep:
        q=a['qpos'][k];g.from_record(q)
        tip,axis=g.tip()
        np.testing.assert_allclose(tip,a['tcp'][k],atol=1e-9)
        lo,hi=g.bounds(); ds=g.distances(lo,hi).min(axis=1)
        b,l,h=[float(ds[g.cat==v].min()) for v in ('body','arm','tool')]
        rear=float(lo[:,0].min()-g.ghi[0,0])
        front=float(hi[:,0].max()-g.glo[0,0])
        err=np.linalg.norm((q[:3]-end) if wide else (tip-TARGET))
        deg=np.rad2deg(np.arccos(np.clip(axis@AXIS,-1.,1.)))
        data.append([t[k],*q[:3],*tip,w.net_deg.values[ix[k]],b,l,h,rear,front,err,deg,
                     np.linalg.norm(q[:3]-LAUNCH-a['ref'][k])])
    tr=pd.DataFrame(data,columns='t x y z tip_x tip_y tip_z bend_deg body_m link_m tool_m rear_m front_m target_m axis_deg base_error_m'.split())
    tr['arm_tool_m']=tr[['link_m','tool_m']].min(axis=1)
    tr.to_csv(DATA/'traces'/(case+'.csv'),index=False,float_format='%.8f')
    tv=tr.t.values; clearly_behind=tr.rear_m.values>0
    after=tv>=(26 if wide else P['terminal_evaluation_s'][0])
    valid=after&clearly_behind&(tr.target_m.values<=P['terminal_position_tolerance_m'])
    if not wide:valid&=(tr.axis_deg.values<=P['terminal_axis_tolerance_deg'])
    longest,done=dwell(tv,valid)
    wb={}
    for label,col in [('body','body_m'),('arm','link_m'),('tool','tool_m'),('arm_tool','arm_tool_m')]:
        j=int(tr[col].idxmin());wb[label]=dict(t=float(tv[j]),k=int(keep[j]),**witness(g,a['qpos'][keep[j]],label))
        assert abs(wb[label]['lower_m']-tr[col].iloc[j])<1e-9
    contacts=int(meta['contact_integration_steps']);peak=meta['max_total_curvature_deg']
    tests=dict(body=bool(tr.body_m.min()>=P['gate_body_margin_m']),arm_tool=bool(tr.arm_tool_m.min()>=P['gate_arm_tool_margin_m']),
               curvature=bool(peak<=P['hard_total_bend_deg']),contacts=contacts==0,rear=bool(clearly_behind[-1]),target=done is not None)
    gate=(tr.front_m>=0)&(tr.rear_m<=0)
    angles=w.net_deg.to_numpy();v=np.gradient(angles,.004)
    tw=(w.us.values-release)/1e6;st=(tw>=14)&(tw<=42)
    row=dict(case=case,seed=seed,mode=mode,pass_all=all(tests.values()),failed=','.join(k for k,vv in tests.items() if not vv),
        contact_steps=contacts,body_cm=100*tr.body_m.min(),arm_tool_cm=100*tr.arm_tool_m.min(),
        link_cm=100*tr.link_m.min(),tool_cm=100*tr.tool_m.min(),body_upper_cm=100*wb['body']['upper_m'],
        arm_tool_upper_cm=100*wb['arm_tool']['upper_m'],cap_peak_deg=peak,max_bend_rate_deg_s=float(abs(v[st]).max()),
        rear_clear_time_s=float(tr.t[tr.rear_m>0].iloc[0]) if clearly_behind.any() else None,
        target_time_s=done,dwell_s=longest,final_error_cm=100*tr.target_m.iloc[-1],
        final_axis_deg=tr.axis_deg.iloc[-1],gate_tracking_peak_cm=100*tr.base_error_m[gate].max() if gate.any() else None,
        gate_overlap_initial_s=float(tr.t[gate].min()) if gate.any() else None,
        gate_overlap_end_s=float(tr.t[gate].max()) if gate.any() else None)
    return row,dict(witness=wb,checks=tests,logs=logs,sha256=hashes(folder))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');a=ap.parse_args()
    runs=DATA/'runs';(DATA/'traces').mkdir(exist_ok=True);(DATA/'per_run').mkdir(exist_ok=True)
    plans=read(DATA/'plans.json')['selected']
    labels=['241119-wide']+['%s-%s'%(s,m) for s,ms in P['seed_order'].items() for m in ms]
    rows=[];audit={}
    for label in labels:
        folder=runs/label
        if not (folder/'flight.ulg').is_file() and a.partial:continue
        if not (folder/'run.json').is_file():raise RuntimeError('Run incomplete: '+label)
        row,details=score(folder,plans);rows.append(row);audit[label]=details
        save(DATA/'per_run'/(label+'.json'),dict(row=row,details=details))
        print(label,row['pass_all'],row.get('failed'),round(row.get('body_cm',0),2),round(row.get('arm_tool_cm',0),2),flush=True)
    if not a.partial:
        pd.DataFrame(rows).to_csv(DATA/'metrics.csv',index=False)
        save(DATA/'audit.json',dict(runs=audit,groups={m:[x['pass_all'] for x in rows if x['mode']==m] for m in MODES}))


if __name__=='__main__':
    main()
