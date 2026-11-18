"""Reconstruct flight scores and mounting moments independently from the saved records."""
import argparse
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
import pandas as pd
import rosbag
from pyulog import ULog

from round_f_plant import CHANNELS, LAUNCH, ROOT, Plant
from round_fp_run import P, setpoint
from round_fp_flex import fast_outward
from round_f_audit import hashes, wire


def read(path):
    return json.loads(path.read_text())


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x)**2)))


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def sensor_sign_check():
    # Same engine and sensor as below, known stationary 2 kg mass 0.1 m forward.
    xml = """<mujoco><worldbody><body name="holder"><geom size=".02" mass="1"/>
      <body name="load"><site name="origin"/><geom size=".02" mass="2" pos=".1 0 0"/></body>
      </body></worldbody><sensor><torque name="torque" site="origin"/></sensor></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    reaction = -d.sensor('torque').data.copy()
    np.testing.assert_allclose(reaction, [0., 1.962, 0.], atol=1e-9)
    return reaction.tolist()


class MountProbe:
    def __init__(self, plant):
        tree = ET.fromstring(plant.xml)
        mount = tree.find(".//body[@name='mount']")
        ET.SubElement(mount, 'site', name='fp_mount_origin', pos='0 0 0', size='.001')
        ET.SubElement(tree.find('sensor'), 'torque', name='fp_mount_moment', site='fp_mount_origin')
        self.m = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding='unicode'))
        for field in ('body_mass', 'body_inertia', 'jnt_qposadr', 'actuator_gear'):
            np.testing.assert_array_equal(getattr(self.m, field), getattr(plant.m, field))
        self.d = mujoco.MjData(self.m)
        self.base = self.m.body('uav').id
        self.mount = self.m.body('mount').id
        self.arms = np.array([self.m.site('rotor%d' % i).pos for i in CHANNELS])
        self.hub = self.arms.mean(axis=0)
        self.mass = self.m.body_subtreemass[self.base]

    def sample(self, q, v, ctrl, thrust):
        m, d = self.m, self.d
        d.eq_active[0] = 0
        d.qpos[:] = q
        d.qvel[:] = v
        d.ctrl[:] = ctrl
        mujoco.mj_forward(m, d)
        base_R = d.xmat[self.base].reshape(3, 3)
        mount_R = d.xmat[self.mount].reshape(3, 3)
        g = base_R.T @ m.opt.gravity
        com = base_R.T @ (d.subtree_com[self.base] - d.xpos[self.base]) - self.hub
        grav = np.cross(com, self.mass * g)[1]
        rotor = np.dot(-(self.arms[:, 0] - self.hub[0]), thrust)
        # Reaction applied by the arm assembly to the aircraft about the mount site.
        total_mount = -(base_R.T @ mount_R @ d.sensor('fp_mount_moment').data)[1]
        arm_com = base_R.T @ (d.subtree_com[self.mount] - d.xpos[self.mount])
        mount_g = np.cross(arm_com, m.body_subtreemass[self.mount] * g)[1]
        return com[0], grav, rotor, total_mount, mount_g, total_mount - mount_g


def audit(folder, plant, probe, trace_dir):
    seed, variant = folder.name.split('-')
    mission = read(folder/'provenance.json')
    run = read(folder/'run.json')
    protocol = 'round_fp_flex.json' if variant == 'fastbend' else 'round_fp.json'
    assert mission['protocol_sha256'] == hashlib.sha256((ROOT/'revision/protocols'/protocol).read_bytes()).hexdigest()
    assert mission['plant_xml_sha256'] == hashlib.sha256(plant.xml.encode()).hexdigest()
    assert run['error'] is None and mission['mission']['seed'] == int(seed)
    w = wire(folder)
    us = w.us.values.astype(np.int64)
    release = run['release_us']
    t = (us-release)/1e6
    free = (t >= 0) & (t <= P['score_end_s'])
    np.testing.assert_equal(np.diff(us), 4000)
    motors = w[['m1','m2','m3','m4']].values
    pos = w[['ex','ey','ez']].values
    ref = w[['rx','ry','rz']].values
    err = np.linalg.norm(pos-ref, axis=1)
    bend = w.net_deg.values
    speed = np.gradient(bend, .004)
    expected = [fast_outward(v)[2] if variant == 'fastbend' else setpoint(v,variant)[2] for v in t[free]]
    np.testing.assert_allclose(w.request_deg.values[free], expected, atol=1e-9)
    np.testing.assert_allclose(w.issued_deg.values[free], expected, atol=1e-9)
    np.testing.assert_allclose(ref[free], np.tile(P['target_ENU_m'], (free.sum(),1)), atol=1e-12)
    a = np.load(str(folder/'physics.npz'))
    k = (a['us']>=release) & (a['us']<=release+P['score_end_s']*1e6)
    stamps = a['us'][k]; ix=np.searchsorted(us, stamps)
    np.testing.assert_array_equal(us[ix], stamps)
    q, v, ctrl, thrust = (a[n][k] for n in ('qpos','qvel','ctrl','rotor_N'))
    state = q[:,plant.arm.qpos_ids].reshape(-1,8,2)
    angles = np.degrees(np.linalg.norm(state.sum(1),axis=1))
    curvature = np.degrees(np.linalg.norm(state,axis=2).sum(1))
    np.testing.assert_allclose(angles, bend[ix], atol=1e-7)
    np.testing.assert_allclose(curvature, w.total_deg.values[ix], atol=1e-7)
    pose_res=float(np.max(abs(q[:,:3]-LAUNCH-pos[ix])))
    thrust_res=float(np.max(abs(ctrl[:,plant.rotors]-12*motors[ix])))
    np.testing.assert_allclose(thrust, ctrl[:,plant.rotors], atol=1e-8)
    assert pose_res < 1e-7 and thrust_res < 1e-7
    u = ULog(str(folder/'flight.ulg'))
    data=lambda n:u.get_dataset(n).data
    gps=data('vehicle_gps_position')
    to_hil=int(np.median(gps['time_utc_usec'].astype(np.int64)-gps['timestamp'].astype(np.int64)-4000))
    assert to_hil==mission['clock_origin_us']
    out=data('actuator_outputs')
    native=out['timestamp'].astype(np.int64)+to_hil
    mask=(native>=release)&(native<=release+P['score_end_s']*1e6)
    times=native[mask]; idx=np.searchsorted(us,times); np.testing.assert_equal(us[idx],times)
    normalized=np.clip((np.column_stack([out['output[%d]'%i][mask] for i in range(4)])-1000)/1000,0,1)
    motor_res=float(np.max(abs(normalized-motors[idx])))
    signals=data('vehicle_status')
    j=np.searchsorted(signals['timestamp'].astype(np.int64)+to_hil,us[free],side='right')-1
    mode=bool(np.all(signals['nav_state'][j]==14) and np.all(signals['arming_state'][j]==2) and
              not np.any(signals['failsafe'][j]) and np.all(w.offboard.values[free]) and np.all(w.armed.values[free]))
    topics={'/clock':stamps[:,None], '/f/evaluation/mujoco_position':pos[ix],
        '/f/reference/position':ref[ix], '/f/px4/hil_motor_channels':motors[ix],
        '/f/plant/rotor_thrust_N':thrust, '/f/px4/local_position_ned':w[['north','east','down']].values[ix],
        '/f/arm/bend':np.column_stack((w.request_deg.values[ix],w.issued_deg.values[ix],angles,curvature))}
    count={n:0 for n in topics}; diff={n:0. for n in topics}
    with rosbag.Bag(str(folder/'flight.bag')) as bag:
        for name,msg,stamp in bag.read_messages():
            stamp_us=stamp.to_nsec()//1000
            if stamp_us<stamps[0] or stamp_us>stamps[-1]:continue
            ii=count[name]; assert stamps[ii]==stamp_us
            if name=='/clock':observed=[msg.clock.to_nsec()//1000]
            elif hasattr(msg,'point'):observed=[msg.point.x,msg.point.y,msg.point.z]
            elif name=='/f/arm/bend':observed=np.degrees(msg.position)
            else:observed=msg.data
            diff[name]=max(diff[name],float(np.max(abs(np.asarray(observed)-topics[name][ii]))));count[name]+=1
    assert min(count.values())==max(count.values())==len(stamps) and max(diff.values())<1e-7
    force=motors*12
    margin=np.minimum(force,12-force).min(axis=1)
    pitch=np.degrees(np.arcsin(np.clip(2*(q[:,3]*q[:,5]-q[:,6]*q[:,4]),-1.,1.)))
    tilt=np.degrees(np.arccos(np.clip(1.-2.*(q[:,4]**2+q[:,5]**2),-1.,1.)))
    rate=data('rate_ctrl_status'); integ=np.interp(stamps,rate['timestamp'].astype(np.int64)+to_hil,rate['pitchspeed_integ'])
    moments=np.array([probe.sample(*args) for args in zip(q,v,ctrl,thrust)])
    frame=pd.DataFrame(dict(s=t[ix],east=pos[ix,0],north=pos[ix,1],up=pos[ix,2],error_cm=100*err[ix],bend_deg=bend[ix],
        issued_deg=w.issued_deg.values[ix],speed_dps=speed[ix],pitch_deg=pitch,tilt_deg=tilt,front_N=force[ix][:,[0,2]].mean(1),
        rear_N=force[ix][:,[1,3]].mean(1),margin_N=margin[ix],pitch_i_norm=integ,com_x_m=moments[:,0],
        gravity_Nm=moments[:,1],rotor_Nm=moments[:,2],mount_Nm=moments[:,3],mount_gravity_Nm=moments[:,4],mount_residual_Nm=moments[:,5]))
    frame.to_csv(trace_dir/(folder.name+'.csv'),index=False,float_format='%.9g')
    begin=(t>=14)&(t<16); b0=int(np.searchsorted(t,24)); q0=int(np.argmin(abs(t[ix]-24)))
    before=(frame.s>=14)&(frame.s<16)
    coverage=len(times)/int(free.sum()); gap=int(np.diff(times).max())
    final_gap_ms=(release+int(P['score_end_s']*1e6)-int(times[-1]))/1000
    age=float(np.abs(w.px4_act_us.values[free]-us[free]).max()/1000)
    log=dict(motor_max_abs=motor_res,pose_max_m=pose_res,thrust_max_N=thrust_res,ulog_dropouts=len(u.dropouts),
             native_motor_coverage=coverage,max_native_gap_ms=gap/1000.,last_native_gap_ms=final_gap_ms,
             max_HIL_age_ms=age,bag_samples_per_topic=len(stamps),bag_max_residual=max(diff.values()),
             time_offset_us=to_hil,offboard_all=mode,integrator_samples=len(rate['timestamp']))
    rc=P['eligibility']
    pre_error=rms(err[begin]);pre_speed=rms(np.linalg.norm(v[before.values,:3],axis=1))
    checks=dict(logs=bool(motor_res<2e-6 and not u.dropouts and coverage>.80 and age<=4 and final_gap_ms<=24),
                flight=bool(mode and run['contact_integration_steps']==0),curvature=bool(run['max_total_curvature_deg']<=120),
                prehover=bool(pre_error<=rc['max_prebend_position_rms_m'] and pre_speed<=rc['max_prebend_speed_rms_m_s']))
    row=dict(seed=int(seed),variant=variant,pre_rms_cm=pre_error*100,pre_speed_cm_s=pre_speed*100,
             curve_max_deg=run['max_total_curvature_deg'],tilt_flight_deg=run['max_tilt_integration_deg'],
             contacts=run['contact_integration_steps'],precheck=all(checks.values()))
    def window(name,start,end):
        mask=(t>=start)&(t<=end); ids=np.flatnonzero(mask); peak=ids[np.argmax(err[ids])]
        f=frame[(frame.s>=start)&(frame.s<=end)]
        row.update({name+'_peak_cm':100*float(err[peak]),name+'_peak_after_s':float(t[peak]-start),
             name+'_end_cm':100*float(err[ids[-1]]),name+'_speed_dps':float(abs(speed[mask]).max()),
             name+'_tilt_deg':float(f.tilt_deg.max()),name+'_margin_N':float(margin[mask].min()),
             name+'_sat_ms':4*int(np.sum(margin[mask]<=.00001)),name+'_mount_resid_Nm':float(f.mount_residual_Nm.abs().max())})
    rise=3 if variant=='fastbend' else 6
    window('out',16.,16.+rise+2.);window('out_common',16.,24.)
    duration=8 if variant=='slow' else 4
    window('ret',24.,24.+duration+3.);window('ret_common',24.,35.)
    j16=int(np.searchsorted(t,16)); k16=int(np.argmin(abs(t[ix]-16)))
    return row,dict(log=log,checks=checks,sha256=hashes(folder),source=mission['source_commit'],
              outward_start_pos=pos[j16].tolist(),outward_start_vel=v[k16,:3].tolist(),
              start_pos=pos[b0].tolist(),start_vel=v[q0,:3].tolist(),start_bend_deg=float(bend[b0]),
              mass_kg=mission['mass_kg'],plant_sha256=mission['plant_xml_sha256'],
              pitch_rate_i=float(u.initial_parameters['MC_PITCHRATE_I']))


def main():
    args=argparse.ArgumentParser();args.add_argument('round',type=Path);root=args.parse_args().round
    traces=root/'traces';traces.mkdir(exist_ok=True)
    plant=Plant();probe=MountProbe(plant)
    sign=sensor_sign_check()
    rows=[];details={};pairs=[]
    for seed in P['seed_order']:
        for variant in ('straight','nominal','slow','fastbend'):
            label=seed+'-'+variant
            row,detail=audit(root/'runs'/label,plant,probe,traces)
            rows.append(row);details[label]=detail
        n,s=[details[seed+'-'+v] for v in ('nominal','slow')]
        pos_diff=float(np.linalg.norm(np.subtract(n['start_pos'],s['start_pos'])))
        vel_diff=float(np.linalg.norm(np.subtract(n['start_vel'],s['start_vel'])))
        angle_diff=abs(n['start_bend_deg']-s['start_bend_deg'])
        pair_ok=(pos_diff<=.03 and vel_diff<=.04 and angle_diff<=1)
        by={x['variant']:x for x in rows if x['seed']==int(seed)}
        fast=details[seed+'-fastbend']
        d16=np.linalg.norm(np.subtract(n['outward_start_pos'], fast['outward_start_pos']))
        v16=np.linalg.norm(np.subtract(n['outward_start_vel'], fast['outward_start_vel']))
        pairs.append(dict(seed=int(seed),outward_start_position_difference_cm=100*float(d16),
          outward_start_velocity_difference_cm_s=100*float(v16),return_start_position_difference_cm=100*pos_diff,
          return_start_velocity_difference_cm_s=100*vel_diff,return_start_shape_difference_deg=angle_diff,
          pair_matched=bool(pair_ok),slow_minus_nominal_cm=by['slow']['ret_peak_cm']-by['nominal']['ret_peak_cm'],
          fastbend_minus_nominal_cm=by['fastbend']['out_peak_cm']-by['nominal']['out_peak_cm']))
    pd.DataFrame(rows).to_csv(root/'metrics.csv',index=False,float_format='%.9g')
    pd.DataFrame(pairs).to_csv(root/'pairs.csv',index=False,float_format='%.9g')
    save(root/'audit.json',dict(static_probe_reaction_Nm=sign,runs=details,pairs=pairs,
          all_integrity_checks=all(all(v['checks'].values()) for v in details.values()),
          all_pairs_matched=all(v['pair_matched'] for v in pairs)))
    for p in pairs: print(p)


if __name__=='__main__':
    main()
