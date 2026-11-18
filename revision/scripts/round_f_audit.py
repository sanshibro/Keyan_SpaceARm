"""Reconstruct scoring intervals independently from HIL, native ULog, ROS1 bag and generalized states."""
import argparse, csv, gzip, hashlib, json
from pathlib import Path

import numpy as np
import pandas as pd
import rosbag
from pyulog import ULog
from round_f_plant import Plant,LAUNCH,ENU_NED,ROOT


def read(p): return json.loads(p.read_text())
def save(p,x): p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def rms(x): return float(np.sqrt(np.mean(np.sum(x*x,axis=1))))
def hashes(folder):
    files=('flight.ulg','flight.bag','physics.npz','hil.csv','events.jsonl','run.json','provenance.json')
    result={}
    for n in files:
        p=folder/n
        content=p.read_bytes() if p.is_file() else gzip.decompress(p.with_suffix('.csv.gz').read_bytes())
        result[n]=hashlib.sha256(content).hexdigest()
    return result

def wire(folder):
    path=folder/'hil.csv'
    return pd.read_csv(path if path.is_file() else folder/'hil.csv.gz')


def audit(folder,raw,plant):
    meta=read(folder/'run.json');prov=read(folder/'provenance.json')
    w=wire(folder);us=w.us.values.astype(np.int64)
    release=meta['release_us'];end=release+39000000
    u=ULog(str(folder/'flight.ulg')); d=lambda n:u.get_dataset(n).data
    phase=w.phase.values; free=(us>=release)&(us<=end)
    ref=w[['rx','ry','rz']].values; p=w[['ex','ey','ez']].values
    motors=w[['m1','m2','m3','m4']].values
    err=p-ref
    ranges={}
    for ph in ('liftoff','hover','translate','bend','recover'):
        pick=np.flatnonzero((phase==ph)&free)
        ranges[ph]=dict(first_us=int(us[pick[0]]),last_us=int(us[pick[-1]]),ticks=len(pick),
                        rms_3d_m=rms(err[pick]),max_3d_m=float(np.linalg.norm(err[pick],axis=1).max()),
                        endpoint_3d_m=float(np.linalg.norm(err[pick[-1]])))
    hover=np.flatnonzero(phase=='hover');trans=np.flatnonzero(phase=='translate');bend=phase=='bend'
    dis=float(p[trans[-250:],0].mean()-p[hover[-250:],0].mean())
    # Raw per-run bag stamps and ROS simulated clock must be identical integer microseconds.
    with np.load(str(folder/'physics.npz')) as z: a={k:z[k] for k in z.files}
    ids=np.array([plant.m.jnt_qposadr[plant.m.joint('%s%d'%(axis,i)).id] for i in range(8) for axis in ('bx','by')])
    q=a['qpos'];mu=a['us'];wi=np.searchsorted(us,mu);assert np.array_equal(us[wi],mu)
    arm=np.degrees(np.linalg.norm(q[:,ids].reshape(-1,8,2).sum(1),axis=1))
    curve=np.degrees(np.linalg.norm(q[:,ids].reshape(-1,8,2),axis=2).sum(1))
    truth=q[:,:3]-LAUNCH
    mtu=(mu>=release)&(mu<=end)
    logs={};bag_max={};n_bag=0
    bi={name:0 for name in ('clock','est','position','ref','motors','thrust','arm')}
    bag_names={'/clock':'clock','/f/px4/local_position_ned':'est','/f/evaluation/mujoco_position':'position',
               '/f/reference/position':'ref','/f/px4/hil_motor_channels':'motors','/f/plant/rotor_thrust_N':'thrust','/f/arm/bend':'arm'}
    with rosbag.Bag(str(folder/'flight.bag')) as bag:
        for topic,msg,stamp in bag.read_messages():
            key=bag_names[topic];t=stamp.to_nsec()//1000
            if not release<=t<=end:continue
            j=int(np.searchsorted(mu,t));i=wi[j];assert mu[j]==t
            bi[key]+=1
            if key=='clock':
                assert msg.clock.to_nsec()==stamp.to_nsec();continue
            if key in ('est','position','ref'):
                val=np.array([msg.point.x,msg.point.y,msg.point.z])
                expected={'position':truth[j],'ref':ref[i],'est':w.loc[i,['north','east','down']].values.astype(float)}[key]
            elif key=='arm':
                val=np.degrees(msg.position);expected=[w.request_deg.iat[i],w.issued_deg.iat[i],arm[j],curve[j]]
            else:
                val=msg.data;expected=motors[i] if key=='motors' else a['ctrl'][j,plant.rotors]
            bag_max[key]=max(bag_max.get(key,0.),float(np.abs(np.array(val)-expected).max()))
    assert all(v==int(mtu.sum()) for v in bi.values())
    assert max(bag_max.values())<1e-7
    qpos_err=float(np.max(abs(truth[mtu]-p[wi[mtu]])))
    thrust_err=float(np.max(abs(a['ctrl'][:,plant.rotors]-12*motors[wi])))
    assert qpos_err<1e-7 and thrust_err<1e-6 and np.max(abs(a['ctrl'][:,plant.rotors]-a['rotor_N']))<1e-7
    gps=d('vehicle_gps_position')
    # The GPS packet precedes its same-tick HIL_SENSOR clock pulse by one simulator tick.
    offsets=(gps['time_utc_usec'].astype(np.int64)-gps['timestamp'].astype(np.int64))-4000
    offset=int(np.median(offsets));assert offset==prov['clock_origin_us']
    out=d('actuator_outputs');t=out['timestamp'].astype(np.int64)+offset;sel=(t>=release)&(t<=end)
    idx=np.searchsorted(us,t[sel]);assert np.array_equal(us[idx],t[sel])
    pwm=np.column_stack([out['output[%d]'%i][sel] for i in range(4)])
    norm=np.clip((pwm-1000.)/1000.,0,1);u_err=float(abs(norm-motors[idx]).max())
    wrong=float(abs(norm[:,[1,0,2,3]]-motors[idx]).max())
    assert u_err<2e-6 and wrong>.05
    est=d('vehicle_local_position');ex=np.column_stack([est[k] for k in ('x','y','z')])
    wire_ms=w.estimator_ms.values[free]*1000
    ekf_rx=np.column_stack([np.interp(wire_ms,est['timestamp'],col) for col in ex.T])
    ekf_err=rms(ekf_rx-w[['north','east','down']].values[free])
    truth_ekf=rms((ekf_rx-np.asarray(meta['ekf_launch_origin_ned']))@ENU_NED-p[free])
    stat=d('vehicle_status');st=stat['timestamp'].astype('int64')+offset
    selst=np.searchsorted(st,us[free],side='right')-1
    mode_ok=bool(np.all(stat['nav_state'][selst]==14) and np.all(stat['arming_state'][selst]==2)
                 and not np.any(stat['failsafe'][selst]) and np.all(w.offboard.values[free]) and np.all(w.armed.values[free]))
    sens=d('sensor_combined');st=sens['timestamp'].astype(np.int64)+offset;good=(st>=release)&(st<=end)
    ai=np.searchsorted(us,st[good]);native_acc=np.column_stack([sens['accelerometer_m_s2[%d]'%i][good] for i in range(3)])
    ae=float(np.percentile(abs(native_acc-w[['ax','ay','az']].values[ai]),99))
    sv=d('trajectory_setpoint');sf=(sv['timestamp']+offset>=release+18000000)&(sv['timestamp']+offset<=release+36000000)
    sp=(np.column_stack([sv[k][sf] for k in ('x','y','z')])-meta['ekf_launch_origin_ned'])@ENU_NED
    sp_err=float(np.max(abs(np.median(sp,axis=0)-[.6,0,.65])))
    sampled_tilt=float(np.degrees(np.arccos(np.clip(1.-2.*(q[mtu,4]**2+q[mtu,5]**2),-1,1))).max())
    peak_tilt=meta.get('max_tilt_integration_deg',sampled_tilt)
    pct=float(len(idx)/free.sum()); gap=int(np.diff(t[sel]).max()); residual=int(np.abs(w.px4_act_us.values[free]-us[free]).max())
    protocol=read(ROOT/'revision/protocols/round_f.json')['acceptance']
    checks=dict(hover=ranges['hover']['rms_3d_m']<=protocol['hover_3d_rms_m'],
                translation_endpoint=ranges['translate']['endpoint_3d_m']<=protocol['translation_endpoint_error_3d_m'],
                translation_rms=ranges['translate']['rms_3d_m']<=protocol['translation_3d_rms_m'],
                translation_distance=abs(dis-.6)<=protocol['translation_endpoint_displacement_tolerance_m'],
                bend_position=ranges['bend']['rms_3d_m']<=protocol['bend_hold_3d_rms_m'],
                bend_response=float(w.net_deg.values[bend].max())>=protocol['bend_tip_response_peak_min_deg'],
                peak_tilt=peak_tilt<=protocol['airborne_peak_tilt_deg'],curvature=meta['max_total_curvature_deg']<=120,
                no_contacts=meta['contact_integration_steps']==0,offboard=mode_ok,
                logs=pct>.8 and not u.dropouts and u_err<2e-6 and residual<=4000 and ae<.15 and ekf_err<.025 and sp_err<.02)
    counts={p:len(d(p)['timestamp']) for p in ['sensor_combined','vehicle_gps_position','vehicle_magnetometer','vehicle_air_data',
               'estimator_status','vehicle_local_position','trajectory_setpoint','vehicle_status','actuator_outputs']}
    return dict(label=folder.name,pitch_rate_i=float(u.initial_parameters['MC_PITCHRATE_I']),phases=ranges,
                release_us=release,scoring_end_us=end,code=prov['source_commit'],mass_kg=prov['mass_kg'],
                displacement_east_m=dis,peak_tilt_deg=peak_tilt,tilt_source='every physics step' if 'max_tilt_integration_deg' in meta else '50 Hz qpos',
                max_total_curvature_deg=meta['max_total_curvature_deg'],peak_net_bend_deg=float(w.net_deg.values[bend].max()),
                maximum_bend_hold_deviation_m=ranges['bend']['max_3d_m'],airborne_contact_steps=meta['contact_integration_steps'],
                checks=checks,passed=bool(all(checks.values())),sha256=hashes(folder),
                reconstruction=dict(px4_boot_to_hil_us=offset,gps_minus_prior_pulse_us=int(np.median(offsets)+4000-offset),
                overlap_native_motor_samples=len(idx),fraction_native_per_wire=pct,max_native_motor_gap_us=gap,
                wire_ticks=int(free.sum()),max_hil_actuator_stamp_error_us=residual,max_native_normalized_motor_error=u_err,
                swapped_first_two_channels_max_error=wrong,max_qpos_position_error_m=qpos_err,max_actuator_write_error_N=thrust_err,
                bag_samples_per_topic=bi,bag_max_abs_error_by_topic=bag_max,ulog_dropouts=len(u.dropouts),ulog_topic_samples=counts,
                ros_telemetry_vs_native_ekf_rms_m=ekf_err,native_ekf_vs_plant_3d_rms_m=truth_ekf,
                native_accel_vs_wire_p99_m_s2=ae,native_held_setpoint_max_component_error_m=sp_err,
                native_motor_first_us=int(t[sel][0]),native_motor_last_us=int(t[sel][-1]),gps_offset_outliers=int((offsets!=offset).sum())))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('source',type=Path);parser.add_argument('out',type=Path)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False);plant=Plant()
    cases=['reference','candidate','repeat-reference','qualification','final']
    results=[audit(args.source/k,args.out,plant) for k in cases]
    final=results[-1];assert final['reconstruction']['native_motor_last_us']>=final['scoring_end_us']-8000
    save(args.out/'summary.json',dict(protocol='revision/protocols/round_f.json',cases=results,display_run='final'))
    save(args.out/'alignment.json',dict(axes='World ENU (east,north,up); PX4 local NED; body FLU -> FRD with diag(+1,-1,-1).',
         alignment_rule='HIL and ROS simulated timestamps in microseconds; add the reported offset to PX4 boot time. GPS packets are sent immediately before the matching IMU; infer the origin from GPS UTC minus uORB sample time minus 4 ms.',
         physics='qpos at 50 Hz includes free-body quaternion wxyz and integrated joints; ctrl holds the actual following interval drive. HIL CSV at 250 Hz, MuJoCo integrates eight 0.5 ms steps per HIL interval.',
         scope='ROS bag written by the running simulator bridge using typed messages on its simulation clock. PX4 control transport is MAVLink over loopback.',**final['reconstruction']))
    s=wire(args.source/'final')
    s=s[(s.us>=final['release_us'])&(s.us<=final['scoring_end_us'])].iloc[::5].copy()
    s.insert(0,'since_release_s',(s.us-final['release_us'])/1e6)
    s.to_csv(args.out/'display_50hz.csv',index=False,float_format='%.8g')
    with (args.out/'phase_metrics.csv').open('w',newline='') as f:
        wr=csv.writer(f);wr.writerow(['run','pitch_I','phase','first_HIL_us','last_HIL_us','rms_3d_m','endpoint_3d_m','max_3d_m','run_passed'])
        for case in results:
            for k,v in case['phases'].items():wr.writerow([case['label'],case['pitch_rate_i'],k,v['first_us'],v['last_us'],v['rms_3d_m'],v['endpoint_3d_m'],v['max_3d_m'],case['passed']])
    print([(c['label'],c['passed'],round(c['phases']['bend']['rms_3d_m'],4)) for c in results])
