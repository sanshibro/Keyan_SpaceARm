"""Sensor-feedback flight takeover of V2 on a frozen sensor-survey map.

This isolates actual free-flight takeover; it is not online SLAM, a door task,
rolling replanning, or the final navigation/grasp acceptance experiment.
"""
import argparse,json,time,queue,shutil,multiprocessing as mp
from pathlib import Path
from xml.etree.ElementTree import fromstring,tostring,SubElement as S
import numpy as np,mujoco
from model import model_xml,SimArmBridge
from qingyu_model import ARC_LENGTH,BASE_LENGTH
from evidence import freeze_run
from scene import textured_room
from fiducials import add_fiducials
from sensor_worker import worker,T,offer_latest
from sensors import OPTICAL_FROM_GL
from flight_feedback import InertialVisualState,RotorController
from mass_compensation import rigid_com_correction
from async_planner import AsyncV21Planner
from planner_adapter import V21Adapter,ROOT
from uav_continuum_whole_body_v2.mapping import MapSnapshot
from uav_continuum_whole_body_v2.geometry import coupling_metrics
from uav_continuum_whole_body_v2.types import ArmModel,MassModel
from uav_continuum_whole_body_v2.execution import tracking_progress_scale

def main(map_path,seconds=40):
    out=ROOT/'runs'/(time.strftime('%Y%m%d_%H%M%S')+'_navigation_development');out.mkdir(parents=True);freeze_run(out)
    input_snapshot=MapSnapshot.load(map_path)
    # The reference search may crop observed space internally; seed,
    # optimizer, validator and runtime clearance need the full sensor map.
    snapshot=input_snapshot
    shutil.copyfile(map_path,out/'input_map.npz')
    adapter=V21Adapter(snapshot,survey_camera=True)
    root=fromstring(model_xml(False,camera_profile='grasp_visible',nav_camera_profile='front_plus_survey'));textured_room(root);add_fiducials(root)
    u=root.find("worldbody/body[@name='uav']");S(u,'site',name='imu',size='.005')
    sensors=S(root,'sensor');S(sensors,'gyro',name='gyro',site='imu');S(sensors,'accelerometer',name='accel',site='imu')
    xml=tostring(root,encoding='unicode');(out/'model.xml').write_text(xml)
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    def body(name):
        i=m.body(name).id;return T(d.xmat[i].reshape(3,3),d.xpos[i])
    def camera(name):
        i=m.camera(name).id;return T(d.cam_xmat[i].reshape(3,3)@OPTICAL_FROM_GL,d.cam_xpos[i])
    U0=body('uav');nav0=camera('navigation');wrist0=camera('wrist_rgbd')
    calibration=dict(initial_nav=nav0,initial_wrist=wrist0,UC=np.linalg.solve(U0,nav0),UB=np.linalg.solve(U0,body('mount')),TC=np.linalg.solve(body('tool'),wrist0),localization_mode='surveyed_fiducials')
    (out/'calibration.json').write_text(json.dumps(calibration,default=lambda x:x.tolist(),indent=2))
    ctx=mp.get_context('spawn');inputs=ctx.Queue(maxsize=1);outputs=ctx.Queue();process=ctx.Process(target=worker,args=(xml,inputs,outputs,calibration,str(out)));process.start()
    ready=outputs.get(timeout=30)
    if 'error' in ready:raise RuntimeError(ready['error'])
    planner=AsyncV21Planner();state=InertialVisualState(U0[:3,3]);controller=RotorController();bridge=SimArmBridge()
    arm=ArmModel(arc_length=ARC_LENGTH,base_offset_z=BASE_LENGTH);mass=MassModel();com=np.zeros(3);bend=np.zeros(2);bend_rate=np.zeros(2);shape_stamp=0.
    reference=np.r_[U0[:3,3],0.,0.,0.];hold=reference.copy();rotors=[m.actuator('thrust%d'%i).id for i in range(4)]
    uid=m.body('uav').id;rng=np.random.default_rng(0);wall=time.perf_counter();rows=[];raw=[];events=[]
    trajectory=None;requested=False;phase=0.;rate=0.;healthy=False;status='sensor_initializing';plan_info={};accepted_at=None;planning_steps=0;contacts=0;saturated=0;interrupted=False;finished=None
    inputs.put((0.,d.qpos.copy(),d.qvel.copy()))
    try:
        for step in range(int(seconds*1000)):
            t=d.time
            while True:
                try:item=outputs.get_nowait()
                except queue.Empty:break
                if 'error' in item:raise RuntimeError(item['error'])
                corrected=item['nav_valid'] and state.correct(item['stamp'],item['WU'],t)
                if item['shape'] is not None and t-item['stamp']<.3:
                    interval=item['stamp']-shape_stamp
                    if interval>.01:bend_rate=.85*bend_rate+.15*(item['shape']-bend)/interval
                    bend=item['shape'];shape_stamp=item['stamp']
                events.append([t,item['stamp'],bool(corrected),item['nav_reason'],item['wrist_reason']])
            healthy=t-state.last_visual<.5 and t-shape_stamp<.5
            yaw=float(np.arctan2(state.R[1,0],state.R[0,0]));measured=np.r_[state.p,yaw,np.linalg.norm(bend),np.arctan2(bend[1],bend[0])]
            if t>3 and healthy and not requested:
                # Plan from the actual estimated configuration, not simulator state.
                planner.request(out/'input_map.npz','qingyu',measured,[.10,0,1.5],snapshot.map_hash,keep_yaw=True,survey_camera=True,progress_path=out/'planner_worker_progress.json');requested=True;status='planning'
                (out/'planning_request.json').write_text(json.dumps(dict(stamp=t,state=measured.tolist(),bending_rate=bend_rate.tolist(),goal=[.1,0,1.5])))
            if requested and planner.process is not None:
                planning_steps+=1
                if step%10==0:
                    # The switch helper uses bending-vector rates at theta=0.
                    theta=max(1e-8,float(np.linalg.norm(bend)));theta_rate=float(bend@bend_rate/theta)
                    phi_rate=float((bend[0]*bend_rate[1]-bend[1]*bend_rate[0])/(theta*theta)) if theta>1e-4 else 0.
                    measured_rate=np.r_[state.v,float(d.sensordata[m.sensor('gyro').adr[0]+2]),theta_rate,phi_rate]
                    candidate,info=planner.poll(measured,measured_rate,snapshot.map_hash)
                    if info['status']!='planning':
                        plan_info=info;status=info['status']
                        if candidate is not None:
                            trajectory=candidate;accepted_at=t;status='executing'
                            np.savez_compressed(out/'accepted_trajectory.npz',times=candidate.times,states=candidate.states,velocities=candidate.velocities)
            if trajectory is not None and step%20==0:
                reference=np.array([np.interp(phase,trajectory.times,trajectory.states[:,i]) for i in range(6)])
                clear=adapter.clearance(measured)
                safe=clear['body']>=adapter.limits.body_margin and clear['arm']>=adapter.limits.arm_margin
                error=np.linalg.norm(state.p-reference[:3]);rate=tracking_progress_scale(error,.025,.08) if healthy and safe else 0.
                status='executing' if rate>0 else 'tracking_or_visibility_hold'
                if phase>=trajectory.duration and error<.03:
                    if finished is None:finished=t
                    status='goal_hold'
            phase=min(trajectory.duration,phase+.001*rate) if trajectory is not None else 0.
            desired=reference[4]*np.array([np.cos(reference[5]),np.sin(reference[5])])
            command=desired+.2*(desired-bend)
            if healthy:d.ctrl[:16]=bridge.move_joints(min(np.linalg.norm(command),np.deg2rad(90)),np.arctan2(command[1],command[0]))
            if step%20==0:
                local_state=np.r_[np.zeros(4),measured[4:6]]
                com=coupling_metrics(local_state,arm,mass)['system_com']+rigid_com_correction(bend,controller.mass,survey=True)
            gyro=d.sensordata[m.sensor('gyro').adr[0]:m.sensor('gyro').adr[0]+3].copy()+rng.normal(0,.0002,3)
            accel=d.sensordata[m.sensor('accel').adr[0]:m.sensor('accel').adr[0]+3].copy()+rng.normal(0,.01,3)
            forces,sat=controller.command(state,gyro,reference[:3],com);d.ctrl[rotors]=forces;saturated+=int(sat)
            mujoco.mj_step(m,d);state.propagate(d.time,gyro,accel,.001);contacts+=d.ncon
            if step%50==0:
                offer_latest(inputs,(d.time,d.qpos.copy(),d.qvel.copy()))
            if step%10==0:
                rows.append([d.time,*d.xpos[uid],*state.p,*reference,phase,rate,np.linalg.norm(d.xpos[uid]-state.p),np.linalg.norm(d.xpos[uid]-reference[:3]),healthy])
                raw.append(np.r_[d.time,d.qpos,d.qvel,d.ctrl,gyro,accel])
            if step%1000==0:(out/'progress.json').write_text(json.dumps(dict(sim_time=d.time,status=status,phase=phase,estimated_position=state.p.tolist(),planner=plan_info),default=str))
            if (out/'STOP_REQUEST').exists():interrupted=True;break
            if finished is not None and t-finished>3:break
            if requested and trajectory is None and planner.process is None and t>8:break
            if step%5==0:
                delay=d.time-(time.perf_counter()-wall)
                if delay>0:time.sleep(delay)
    except KeyboardInterrupt:interrupted=True
    finally:
        planner.close()
        try:inputs.put_nowait(None)
        except queue.Full:pass
        process.join(timeout=3)
        if process.is_alive():process.terminate();process.join()
    elapsed=time.perf_counter()-wall;a=np.array(rows)
    np.savez_compressed(out/'flight.npz',data=a,columns=['time','true_x','true_y','true_z','estimated_x','estimated_y','estimated_z','reference_x','reference_y','reference_z','reference_yaw','reference_theta','reference_phi','phase','phase_rate','nav_error','tracking_error','healthy'])
    np.savez_compressed(out/'raw_dynamics_imu.npz',data=np.array(raw),nq=m.nq,nv=m.nv,nu=m.nu)
    (out/'perception_events.json').write_text(json.dumps(events))
    result=dict(stage='free_flight_V21_takeover_frozen_sensor_map_development',full_task_accepted=False,map_hash=snapshot.map_hash,planning_physics_steps=planning_steps,planner=plan_info,accepted_at=accepted_at,status=status,finished_at=finished,sim_seconds=d.time,wall_seconds=elapsed,realtime_factor=d.time/elapsed,contacts=contacts,saturated_steps=saturated,interrupted=interrupted,max_nav_error_m=float(a[:,15].max()),position_rms_m=float(np.sqrt(np.mean(a[:,16]**2))))
    result['passed']=finished is not None and not interrupted and contacts==0 and saturated==0 and result['position_rms_m']<.1
    (out/'manifest.json').write_text(json.dumps(result,default=str,indent=2));print(out);print(json.dumps(result,default=str))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('map',type=Path);parser.add_argument('--seconds',type=float,default=40);args=parser.parse_args();main(args.map,args.seconds)
