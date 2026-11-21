"""Run a loopback-only PX4 v1.13 SITL with the registered free-base MuJoCo assembly."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import select
import shutil
import signal
import socket
import subprocess
import time

import genpy
import mujoco
import numpy as np
import rosbag
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import JointState
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Float64MultiArray
from pymavlink import mavutil
from pymavlink.dialects.v20 import common as mav
from round_f_plant import Plant, ROOT, ENU_NED, LAUNCH, CLOCK_ORIGIN_US, TICK, reference, CHANNELS

BUILD=Path(os.environ.get('PX4_F_BUILD',str(Path.home()/'PX4-Autopilot/build/px4_sitl_default'))).resolve()
INSTANCE=6
SYS=INSTANCE+1
FREQ=round(1/TICK)


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p,values):
    temp=p.with_name(p.name+'.tmp')
    temp.write_text(json.dumps(values,indent=2,allow_nan=False)+'\n')
    temp.replace(p)


def prepare(runtime,pitch_i,instance=INSTANCE):
    mixer=BUILD/'src/lib/mixer/MultirotorMixer/mixer_multirotor_normalized.generated.h'
    rows=mixer.read_text().split('_config_quad_x[] {')[1].split('};')[0]
    import re
    a=np.array([list(map(float,k.split(','))) for k in re.findall(r'\{([^}]+)\}',rows)])
    assert np.array_equal(np.sign(a[:,:3]),[[-1,1,1],[1,-1,1],[1,1,-1],[-1,-1,-1]])
    shutil.copytree(BUILD/'etc',runtime/'etc')
    init=runtime/'etc/init.d-posix'
    params=init/'px4-rc.params'
    params.write_text(params.read_text()+'''
set MIXER quad_x
param set EKF2_MULTI_IMU 1
param set EKF2_MULTI_MAG 1
param set SENS_IMU_MODE 1
param set SENS_MAG_MODE 1
param set MPC_THR_HOVER 0.46
param set SDLOG_MODE 2
param set SDLOG_PROFILE 0
''')
    with params.open('a') as f: f.write('param set MC_PITCHRATE_I %.4f\n'%pitch_i)
    logger=runtime/'etc/logging/logger_topics.txt'
    logger.parent.mkdir(exist_ok=True)
    logger.write_text('\n'.join(('sensor_combined 0','vehicle_gps_position 0','vehicle_magnetometer 20',
                    'vehicle_air_data 20','vehicle_local_position 20','vehicle_attitude 20','estimator_status 20',
                    'vehicle_status 0','vehicle_status_flags 0','offboard_control_mode 0','trajectory_setpoint 0',
                    'vehicle_attitude_setpoint 20','vehicle_rates_setpoint 20','rate_ctrl_status 20',
                    'actuator_controls_0 4','actuator_outputs 0','vehicle_control_mode 0','vehicle_land_detected 0',
                    'vehicle_command 0','manual_control_setpoint 100'))+'\n')
    link=init/'px4-rc.mavlink'
    link.write_text(f'''#!/bin/sh
mavlink start -u {14580+instance} -o {14540+instance} -t 127.0.0.1 -r 1000000 -m onboard
mavlink stream -u {14580+instance} -s LOCAL_POSITION_NED -r 50
mavlink stream -u {14580+instance} -s ATTITUDE -r 20
''')
    return dict(binary_sha256=digest(BUILD/'bin/px4'),stock_rcS_sha256=digest(init/'rcS'),
                params_override=params.read_text(),mavlink_override=link.read_text(),native_logger_topics=logger.read_text(),
                normalized_mixer_rows=a.tolist(),mixer_sha256=digest(runtime/'etc/mixers/quad_x.main.mix'),
                px4_checkout=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(BUILD.parents[1])).decode().strip())


class Sender:
    def __init__(self,sock): self.sock=sock
    def write(self,packet): self.sock.sendall(packet)


class Bag:
    def __init__(self,path): self.file=rosbag.Bag(str(path),'w',compression='bz2')
    def sample(self,us,r,p,estimate,motor,thrust,arm):
        stamp=genpy.Time(us//1000000,(us%1000000)*1000)
        def point(topic,xyz,frame):
            m=PointStamped()
            m.header.stamp=stamp
            m.header.frame_id=frame
            m.point.x,m.point.y,m.point.z=xyz
            self.file.write(topic,m,stamp)
        self.file.write('/clock',Clock(stamp),stamp)
        point('/f/reference/position',r,'launch_enu')
        point('/f/evaluation/mujoco_position',p,'launch_enu')
        if estimate is not None: point('/f/px4/local_position_ned',estimate,'ekf_origin_ned')
        self.file.write('/f/px4/hil_motor_channels',Float64MultiArray(data=list(motor)),stamp)
        self.file.write('/f/plant/rotor_thrust_N',Float64MultiArray(data=list(thrust)),stamp)
        m=JointState(); m.header.stamp=stamp
        m.name=['request','issued','integrated_net','integrated_total']
        m.position=np.radians(arm).tolist()
        self.file.write('/f/arm/bend',m,stamp)


def run(out,tag,pitch_i,*,reference_fn=reference,score_end=39.,noise_seed=None,
        protocol_name='round_f.json',mission_record=None,plant_factory=Plant,shape_policy=None,
        flight_loop=None,instance=INSTANCE,wall_limit_s=200):
    if flight_loop is not None and shape_policy is not None:
        raise ValueError('Select one flight controller')
    sysid=instance+1
    out.mkdir(parents=True,exist_ok=False)
    runtime=out/'runtime'
    plant=plant_factory()
    if flight_loop is not None:
        (out/'model.xml').write_text(plant.xml)
        shutil.copyfile(ROOT/'revision/protocols'/protocol_name,out/'protocol.json')
    if noise_seed is not None:
        plant.noise=np.random.default_rng(noise_seed)
    prov=prepare(runtime,pitch_i,instance)
    prov.update(tag=tag,protocol_sha256=digest(ROOT/'revision/protocols'/protocol_name),
                plant_xml_sha256=hashlib.sha256(plant.xml.encode()).hexdigest(),
                source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT)).decode().strip(),
                script_sha256={s:digest(Path(__file__).with_name(s)) for s in ('round_f_plant.py','round_f_sitl.py')},
                mujoco=mujoco.__version__,rotor_channels_to_model=CHANNELS.tolist(),motor_gain_N=12.,
                differential_FRD=plant.check.tolist(),mass_kg=plant.mass,physics_dt_s=.0005,hil_dt_s=TICK,
                clock_origin_us=CLOCK_ORIGIN_US,imu_std_si=[.015,.0008],mag_std_gauss=.0003,baro_std_m=.018,
                gps='ideal WGS84 fix plus integer MAVLink quantization; one simulated receiver, 10 Hz')
    if mission_record is not None:
        prov['mission']=mission_record
    dump(out/'provenance.json',prov)
    server=socket.socket(); server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    server.bind(('127.0.0.1',4560+instance));server.listen(1);server.settimeout(35 if flight_loop else 18)
    ground=mavutil.mavlink_connection('udpin:127.0.0.1:%d'%(14540+instance),source_system=246,source_component=191)
    log=(out/'px4-console.txt').open('wb')
    env=dict(os.environ,PX4_SIM_MODEL='iris',PX4_ESTIMATOR='ekf2',PX4_SIM_HOST_ADDR='127.0.0.1')
    px=subprocess.Popen([str(BUILD/'bin/px4'),'-d','-i',str(instance),'-w',str(runtime),'-s','etc/init.d-posix/rcS'],
                        env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    event_file=(out/'events.jsonl').open('w',buffering=1)
    wire_file=(out/'hil.csv').open('w',newline='',buffering=65536)
    writer=csv.writer(wire_file)
    writer.writerow('us phase released px4_act_us armed flag offboard m1 m2 m3 m4 request_deg issued_deg net_deg total_deg ex ey ez rx ry rz estimator_ms north east down ax ay az gx gy gz mx my mz hpa baro_m'.split())
    bag=Bag(out/'flight.bag'); samples={k:[] for k in ('us','qpos','qvel','rotor_N','tcp','ref','ctrl','ncon')}
    if flight_loop is not None:
        samples['contact_steps']=[]
    sock=None
    k=0;released=None; last_phase=''; arm_since=None
    last_mode=False;armed=False;motor=np.zeros(4);flag=0;act_us=0;positions=[]
    estimate=None;est_ms=0; origin=None;ground_seen=False
    velocity=None;attitude=None;att_ms=0
    ts0=time.monotonic();npack=0; max_age=0;last_seq=-1;latest_ack={}
    err=None;us=CLOCK_ORIGIN_US
    def event(name,**fields):
        event_file.write(json.dumps(dict(us=us,event=name,**fields))+'\n')
    try:
        sock,_=server.accept();sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
        sender=mav.MAVLink(Sender(sock),srcSystem=1,srcComponent=200)
        parser=mav.MAVLink(None)
        event('simulator_connected')
        while True:
            us=CLOCK_ORIGIN_US+k*round(TICK*1e6)
            sim_t=k*TICK
            if sim_t>max(110.,score_end+45.) or time.monotonic()-ts0>wall_limit_s:
                raise TimeoutError('Qualification sequence did not finish before timeout')
            if px.poll() is not None: raise RuntimeError('PX4 exited during flight')
            while True:
                msg=ground.recv_match(blocking=False)
                if msg is None: break
                typ=msg.get_type()
                if msg.get_srcSystem()!=sysid: continue
                if typ=='HEARTBEAT':
                    ground_seen=True;last_mode=((msg.custom_mode>>16)&255)==6
                elif typ=='LOCAL_POSITION_NED':
                    estimate=[msg.x,msg.y,msg.z];est_ms=msg.time_boot_ms
                    velocity=[msg.vx,msg.vy,msg.vz]
                    if origin is None: positions.append(estimate)
                elif typ=='ATTITUDE':
                    attitude=[msg.roll,msg.pitch,msg.yaw];att_ms=msg.time_boot_ms
                elif typ=='COMMAND_ACK':
                    latest_ack[int(msg.command)]=int(msg.result)
                    event('ack',command=msg.command,result=msg.result)
                elif typ=='STATUSTEXT': event('px4_text',text=str(msg.text))
            if origin is None and sim_t>7 and len(positions)>=35:
                origin=np.median(positions[-35:],axis=0)
                event('fixed_ekf_launch_origin',ned=origin.tolist())
            if ground_seen and k%FREQ==0:
                ground.mav.heartbeat_send(6,8,0,0,4)
            phase_t=None if released is None else (k-released)*TICK
            phase,ref,theta=reference_fn(phase_t)
            shape=plant.arm.observe()
            if shape_policy is not None and phase_t is not None:
                xyz=None if origin is None or estimate is None else LAUNCH+ENU_NED@(np.asarray(estimate)-origin)
                vel=None if velocity is None else ENU_NED@velocity
                theta=shape_policy.command(us,phase_t,ref,theta,xyz,vel,attitude,est_ms,att_ms,
                                           shape['net_bending_deg'])
            if flight_loop is not None and phase_t is not None:
                xyz=None if origin is None or estimate is None else LAUNCH+ENU_NED@(np.asarray(estimate)-origin)
                vel=None if velocity is None else ENU_NED@velocity
                phase,ref,theta=flight_loop.tick(phase_t,us,xyz,vel,attitude,est_ms,att_ms)
            if phase_t is not None and phase_t>score_end: phase='log_tail'
            if k%10==0 and ground_seen:
                # A neutral virtual joystick supplies the monitored manual link in this isolated SITL.
                ground.mav.manual_control_send(sysid,0,0,0,0,0)
                ref_ned=ENU_NED@ref+(origin if origin is not None else 0.)
                yaw_ned = math.remainder(math.pi/2-getattr(flight_loop, "yaw_enu", 0.), 2*math.pi)
                motion = getattr(flight_loop, "translation_feedforward", None)
                if motion is None:
                    ground.mav.set_position_target_local_ned_send(us//1000,sysid,1,1,2552,
                                                *ref_ned,0,0,0,0,0,0,yaw_ned,0)
                else:
                    vel_ff, acc_ff = (ENU_NED @ component for component in motion)
                    ground.mav.set_position_target_local_ned_send(us//1000,sysid,1,1,2048,
                                                *ref_ned,*vel_ff,*acc_ff,yaw_ned,0)
            if origin is not None and sim_t>10 and k%FREQ==0:
                if not last_mode:
                    ground.mav.command_long_send(sysid,1,176,0,1,6,0,0,0,0,0)
                elif not armed:
                    ground.mav.command_long_send(sysid,1,400,0,1,0,0,0,0,0,0)
            acc,gyro,mag,press,alt=plant.sensor()
            if k%25==0:
                lat,lon,h,vel,vn,ve,vd=plant.gps()
                sender.hil_gps_send(us,3,lat,lon,h,35,55,vel,vn,ve,vd,65535,12)
            sender.hil_sensor_send(us,*acc,*gyro,*mag,press,0.,alt,18.,8191)
            # Startup is asynchronous. Once the mixer starts, wait for the response to this stimulus.
            until=time.monotonic()+(.018 if npack else .004)
            while time.monotonic()<until:
                rr,_,_=select.select([sock],[],[],max(0.,until-time.monotonic()))
                if not rr: break
                buf=sock.recv(65536)
                if not buf: raise ConnectionError('PX4 disconnected')
                for p in parser.parse_buffer(buf) or []:
                    if p.get_type()=='HIL_ACTUATOR_CONTROLS':
                        npack+=1;act_us=int(p.time_usec);last_seq=k
                        motor=np.array(p.controls[:4],float);armed=bool(p.mode&128);flag=int(p.flags)
                        if not np.isfinite(motor).all() or min(motor)<-1e-5 or max(motor)>1.00001:
                            raise ValueError('Out-of-range HIL motor outputs')
                if npack and act_us>=us: break
            if armed and arm_since is None:
                arm_since=sim_t;event('px4_armed')
            if released is None and armed and last_mode and motor.sum()*12>=plant.mass*9.81*.98:
                plant.d.eq_active[0]=0
                if noise_seed is not None:
                    # Paired airborne sensor noise is indexed by time since release.
                    plant.noise=np.random.default_rng([noise_seed,1])
                released=k;phase_t=0.;phase='liftoff';event('support_removed',thrust_N=12*motor.sum())
                if flight_loop is not None: flight_loop.start(us)
            if released is not None and (not armed or not last_mode):
                raise RuntimeError('Arming or OFFBOARD lost in free flight')
            if phase!=last_phase:
                event('phase',name_detail=phase)
                last_phase=phase
            pose=plant.d.xpos[plant.base]-LAUNCH
            issued=plant.issue(k,us,motor,theta)
            angles=[theta,issued,shape['net_bending_deg'],shape['total_curvature_deg']]
            # Stored qpos and position are at sensor time; thrust is issued for the following 4 ms.
            sample_tick = k if flight_loop is None or released is None else k-released
            if sample_tick%5==0:
                bag.sample(us,ref,pose,estimate,motor,motor*12,angles)
                if flight_loop is not None:
                    samples['contact_steps'].append(plant.contact_count)
                    if released is not None:
                        flight_loop.bag_sample(bag.file,us)
                for key,val in [('us',us),('qpos',plant.d.qpos.copy()),('qvel',plant.d.qvel.copy()),
                                ('rotor_N',motor*12),('tcp',plant.d.site('tip').xpos.copy()),
                                ('ref',ref.copy()),('ctrl',plant.d.ctrl.copy()),('ncon',plant.d.ncon)]:
                    samples[key].append(val)
            plant.step(released is not None)
            writer.writerow([us,phase,int(released is not None),act_us,int(armed),flag,int(last_mode),
                             *motor,*angles,*pose,*ref,est_ms,*(estimate or [float('nan')]*3),*acc,*gyro,*mag,press,alt])
            if released is not None: max_age=max(max_age,k-last_seq)
            if k%250==0:
                dump(out/'progress.json',dict(tag=tag,sim_s=sim_t,phase=phase,armed=armed,offboard=last_mode,
                     actuator_packets=npack,world_relative_enu_m=np.round(pose,4).tolist(),bend_deg=round(angles[2],2),
                     rotor_N=np.round(motor*12,3).tolist(),wall_s=round(time.monotonic()-ts0,1)))
                if released is None and sim_t>=48:
                    raise RuntimeError('No free lift-off in 48 simulated seconds; inspect estimator, arming and rotor signals')
            # Keep the simulator live beyond the final scoring stamp while the native logger drains.
            if phase_t is not None and phase_t>=score_end+1.: break
            k+=1
    except Exception as e:
        err=repr(e);event('error',detail=err)
        raise
    finally:
        wire_file.close(); event_file.close();bag.file.close()
        np.savez_compressed(str(out/'physics.npz'),**{key:np.asarray(a) for key,a in samples.items()})
        try:
            px.send_signal(signal.SIGINT)
            try: px.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(px.pid,signal.SIGTERM);px.wait(timeout=5)
        finally:
            if sock: sock.close()
            server.close();ground.close();log.close()
            dump(out/'run.json',dict(tag=tag,error=err,release_us=None if released is None else CLOCK_ORIGIN_US+4000*released,
                 end_us=us,packets=npack,contact_integration_steps=plant.contact_count,
                 max_tilt_integration_deg=float(np.degrees(np.arccos(np.clip(plant.minimum_body_up,-1.,1.)))),
                 max_total_curvature_deg=plant.max_curve,max_ticks_since_new_packet=max_age,ack_latest=latest_ack,
                 ekf_launch_origin_ned=None if origin is None else origin.tolist()))
            candidates=list(runtime.rglob('*.ulg'))
            if len(candidates)==1: shutil.copyfile(candidates[0],out/'flight.ulg')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('out',type=Path);parser.add_argument('--tag',required=True)
    parser.add_argument('--pitch-i',type=float,choices=(.2,.4),default=.2)
    args=parser.parse_args();run(args.out.resolve(),args.tag,args.pitch_i)
