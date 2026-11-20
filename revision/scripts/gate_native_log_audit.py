"""Cross-check PX4 uORB motor messages, ROS morphology feedback, and MuJoCo rotor loads."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from pyulog import ULog
import rosbag


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def flight(folder, signature):
    with (folder/'hil.csv').open(newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if r['armed'] == '1' and r['released'] == '1']
    hil_time = np.array([int(r['px4_act_us']) for r in rows], dtype=np.int64)
    motors = np.array([[float(r[k]) for k in ('m1','m2','m3','m4')] for r in rows])
    with np.load(folder/'physics.npz', allow_pickle=False) as p:
        physics_us = p['us']
        thrust = p['rotor_N']
    clock_origin = int(physics_us[0])
    log = ULog(str(folder/'flight.ulg'), ['actuator_outputs', 'vehicle_status', 'vehicle_attitude',
                                           'vehicle_local_position', 'trajectory_setpoint'])
    native = {d.name: d.data for d in log.data_list if d.multi_id == 0}
    act = native['actuator_outputs']
    # The simulated MAVLink clock starts at clock_origin. uORB is stamped from PX4 boot.
    us = hil_time - clock_origin
    ix = np.minimum(np.searchsorted(act['timestamp'], us), len(act['timestamp'])-1)
    same = act['timestamp'][ix] == us
    pwm = np.column_stack([act[f'output[{i}]'] for i in range(4)]).astype(float)
    motor_error = np.abs((pwm[ix[same]]-1000.)/1000. - motors[same])
    if len(motor_error) == 0:
        raise ValueError('PX4 and HIL clocks have no overlapping motor packets: '+folder.name)
    # A message-type change or stale playback would break the clock or vector matches.
    shapes = ('des_b','est_b','issued_b')
    bag_times = {k: [] for k in shapes}
    vectors = {k: [] for k in shapes}
    thrust_us, rotor = [], []
    with rosbag.Bag(str(folder/'flight.bag')) as bag:
        topics = bag.get_type_and_topic_info().topics
        for topic, msg, at in bag.read_messages(topics=['/gate/shape/'+k for k in shapes]+['/f/plant/rotor_thrust_N']):
            stamp = at.to_nsec()//1000
            if topic == '/f/plant/rotor_thrust_N':
                thrust_us.append(stamp)
                rotor.append(msg.data)
            else:
                key = topic.rsplit('/',1)[1]
                bag_times[key].append(stamp)
                vectors[key].append((msg.vector.x, msg.vector.y))
    recorded = json.loads((folder/'run.json').read_bytes())
    with np.load(folder/'control.npz', allow_pickle=False) as c:
        control_us = recorded['release_us'] + np.rint(c['t']*1e6).astype(np.int64)
        for key in shapes:
            if (not np.array_equal(bag_times[key], control_us) or
                    not np.array_equal(np.asarray(vectors[key]), c[key], equal_nan=True)):
                raise ValueError('Shape timestamps/values differ in ROS and controller log: '+folder.name+'/'+key)
    if not np.array_equal(physics_us, thrust_us):
        raise ValueError('Different ROS / physics clocks: '+folder.name)
    thrust_error = float(np.max(np.abs(np.asarray(rotor)-thrust)))
    finite_active = np.all(np.isfinite(motors),axis=1) & (np.ptp(motors,axis=1)>0)
    if not np.any(finite_active) or not len(native['trajectory_setpoint']['timestamp']):
        raise ValueError('Missing independent motor actuation or PX4 setpoints: '+folder.name)
    status = native['vehicle_status']
    return dict(input_sha256=signature, armed_offboard_status_samples=int(np.sum(
                    (status['arming_state']==2)&(status['nav_state']==14))),
                px4_sets={n:len(s['timestamp']) for n,s in native.items()},
                armed_release_motor_packets=len(motors), exact_timestamp_motor_matches=int(same.sum()),
                max_px4_pwm_to_hil_residual=float(np.max(motor_error)),
                px4_motor_fraction_in_float32_precision=float(np.mean(np.max(motor_error,axis=1)<2e-6)),
                min_hil_channel_span=float(np.min(np.ptp(motors,axis=0))),
                ros_shape_samples_per_topic=len(control_us), ros_shape_arrays_match=True,
                ros_thrust_samples=len(rotor), ros_vs_mujoco_max_thrust_err_N=thrust_error,
                ros_local_position_samples=topics['/f/px4/local_position_ned'].message_count,
                ros_shape_time_start_us=int(control_us[0]), ros_shape_time_end_us=int(control_us[-1]))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',type=Path);ap.add_argument('output',type=Path)
    ap.add_argument('--complete',action='store_true')
    args=ap.parse_args()
    result=json.loads((args.stage/'decisions/metrics.json').read_bytes())['cases']
    if args.complete and len(result)!=60:
        raise ValueError('60 independent flights required for final native audit')
    before=json.loads(args.output.read_bytes())['cases'] if args.output.exists() else {}
    audited={}
    certificates={}
    for case in result:
        folder=args.stage/'formal'/case
        signature={n:sha(folder/n) for n in ('hil.csv','flight.ulg','flight.bag','physics.npz','control.npz','run.json')}
        if any(result[case]['artifacts'].get(n, h)!=h for n,h in signature.items()):
            raise ValueError('Raw source differs from its independent score: '+case)
        old=before.get(case,{})
        audited[case]=old if old.get('input_sha256')==signature else flight(folder,signature)
        extra=result[case].get('distance_certificate')
        if extra:
            receipt=args.stage/'distance-certificates'/(case+'.json')
            if extra['file']!='distance-certificates/'+receipt.name or sha(receipt)!=extra['sha256']:
                raise IOError('Offline box-distance receipt differs: '+case)
            proof=json.loads(receipt.read_bytes())
            if (proof['inputs']!={n:signature[n] for n in proof['inputs']} or
                len(proof['solutions'])!=extra['certified_queries'] or
                any(g['upper_m']<g['lower_m'] or g['upper_m']-g['lower_m']>g['requested_precision_m']
                    for g in proof['solutions'])):
                raise ValueError('Closest-point certificate failed verification: '+case)
            certificates[case]=dict(sha256=extra['sha256'],**proof)
    proof=dict(trials=len(audited),cases=audited,distance_certificates=certificates,exact_timestamp_motor_matches=sum(
               x['exact_timestamp_motor_matches'] for x in audited.values()),
               matched_shape_samples_per_topic=sum(x['ros_shape_samples_per_topic'] for x in audited.values()),
               all_ros_shape_arrays_match=all(x['ros_shape_arrays_match'] for x in audited.values()),
               min_fraction_of_matching_pwm_values=min(x['px4_motor_fraction_in_float32_precision'] for x in audited.values()),
               maximum_ros_vs_physics_thrust_error_N=max(x['ros_vs_mujoco_max_thrust_err_N'] for x in audited.values()),
               offboard_active_in_every_flight=all(x['armed_offboard_status_samples']>0 for x in audited.values()))
    args.output.parent.mkdir(exist_ok=True,parents=True)
    tmp=args.output.with_suffix('.writing')
    tmp.write_text(json.dumps(proof,indent=2)+'\n')
    tmp.replace(args.output)
    print('Native uORB, ROS and physical state cross-check:',len(audited),'flights;',proof['exact_timestamp_motor_matches'],'synchronous PX4 motor records')


if __name__=='__main__':
    main()
