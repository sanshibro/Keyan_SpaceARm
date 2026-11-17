"""Asynchronous sensor-feedback flight development gate. No task acceptance."""
import argparse,csv,json,time,sys,multiprocessing as mp,queue
from pathlib import Path
from xml.etree.ElementTree import fromstring,SubElement as S,tostring
import mujoco,numpy as np
from model import model_xml,SimArmBridge
from evidence import freeze_run
from scene import textured_room
from sensor_worker import worker,T,offer_latest
from sensors import OPTICAL_FROM_GL
from estimation import orientation_shape
from flight_feedback import InertialVisualState,RotorController
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'xtdrone_catkin_ws/src/uav_continuum_whole_body_v2/src'))
from uav_continuum_whole_body_v2.geometry import coupling_metrics
from uav_continuum_whole_body_v2.types import ArmModel,MassModel

def main(seconds=12):
    out=ROOT/'runs'/(time.strftime('%Y%m%d_%H%M%S')+'_sensor_flight');out.mkdir(parents=True)
    executed_source=Path(__file__).read_text();freeze_run(out)
    root=fromstring(model_xml(False));textured_room(root)
    u=root.find("worldbody/body[@name='uav']");S(u,'site',name='imu',pos='0 0 0',size='.005')
    sen=S(root,'sensor');S(sen,'gyro',name='gyro',site='imu');S(sen,'accelerometer',name='accel',site='imu')
    xml=tostring(root,encoding='unicode');(out/'model.xml').write_text(xml)
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    def body(name):
        i=m.body(name).id;return T(d.xmat[i].reshape(3,3),d.xpos[i])
    def camera(name):
        i=m.camera(name).id;return T(d.cam_xmat[i].reshape(3,3)@OPTICAL_FROM_GL,d.cam_xpos[i])
    # Known initial placement and rigid calibration only. No subsequent truth used by controller.
    U0=body('uav');nav0=camera('navigation');wrist0=camera('wrist_rgbd')
    calibration=dict(initial_nav=nav0,initial_wrist=wrist0,UC=np.linalg.solve(U0,nav0),UB=np.linalg.solve(U0,body('mount')),TC=np.linalg.solve(body('tool'),wrist0))
    ctx=mp.get_context('spawn');inputs=ctx.Queue(maxsize=1);outputs=ctx.Queue();process=ctx.Process(target=worker,args=(xml,inputs,outputs,calibration));process.start()
    ready=outputs.get(timeout=30)
    if 'error' in ready:raise RuntimeError(ready['error'])
    inputs.put((0.,d.qpos.copy(),d.qvel.copy()));first=outputs.get(timeout=30)
    if 'error' in first:raise RuntimeError(first['error'])
    (out/'calibration.json').write_text(json.dumps({k:v.tolist() for k,v in calibration.items()},indent=2))
    state=InertialVisualState([0,0,1.5]);controller=RotorController();bridge=SimArmBridge();bend=np.zeros(2);last_shape=0.
    arm=ArmModel(arc_length=.28,base_offset_z=.0704);mass=MassModel();com=np.zeros(3)
    rotor_ids=[m.actuator('thrust%d'%i).id for i in range(4)];uid=m.body('uav').id;rng=np.random.default_rng(0)
    rows=[];events=[];dropped=0;saturated=0;contact_total=0;wall=time.perf_counter();raw=[]
    try:
        for step in range(int(seconds*1000)):
            t=d.time
            while True:
                try:item=outputs.get_nowait()
                except queue.Empty:break
                if 'error' in item:raise RuntimeError(item['error'])
                accepted=False
                if item['nav_valid']:accepted=state.correct(item['stamp'],item['WU'],t)
                if item['shape'] is not None and t-item['stamp']<.3:bend=item['shape'];last_shape=item['stamp']
                events.append([t,item['stamp'],item['nav_valid'],item['wrist_valid'],accepted,item['dropout'],item['nav_reason'],item['wrist_reason'],*(item['WU'][:3,3] if item['WU'] is not None else [float('nan')]*3)])
            healthy=t-state.last_visual<.5 and t-last_shape<.5
            s=.5*(1-np.cos(np.pi*np.clip((t-2)/6,0,1)))
            desired=np.deg2rad(30)*s*np.array([np.cos(np.pi/4),np.sin(np.pi/4)])
            commanded=desired+.2*(desired-bend);theta=min(np.linalg.norm(commanded),np.deg2rad(90));phi=np.arctan2(commanded[1],commanded[0])
            if healthy:d.ctrl[:16]=bridge.move_joints(theta,phi)
            if step%20==0:
                com=coupling_metrics([0,0,0,0,np.linalg.norm(bend),np.arctan2(bend[1],bend[0])],arm,mass)['system_com']
            gyro=d.sensordata[m.sensor('gyro').adr[0]:m.sensor('gyro').adr[0]+3].copy()+rng.normal(0,.0002,3)
            accel=d.sensordata[m.sensor('accel').adr[0]:m.sensor('accel').adr[0]+3].copy()+rng.normal(0,.01,3)
            target=np.array([.1*s if healthy else state.p[0],0,1.5])
            forces,sat=controller.command(state,gyro,target,com);d.ctrl[rotor_ids]=forces;saturated+=int(sat)
            mujoco.mj_step(m,d);state.propagate(d.time,gyro,accel,.001);contact_total+=d.ncon
            if step%50==0:
                dropped+=offer_latest(inputs,(d.time,d.qpos.copy(),d.qvel.copy()))
            if step%10==0:
                actual=d.xpos[uid].copy();tilt=np.rad2deg(np.arccos(np.clip(d.xmat[uid].reshape(3,3)[2,2],-1,1)))
                actual_shape=orientation_shape(np.linalg.solve(body('mount'),camera('wrist_rgbd')),calibration['TC'])
                bend_error=float(np.rad2deg(np.linalg.norm(actual_shape-bend)))
                rows.append([d.time,*actual,*state.p,*state.v,*bend,*forces,np.linalg.norm(actual-target),np.linalg.norm(actual-state.p),tilt,healthy,t-state.last_visual,t-last_shape,d.ncon,*actual_shape,bend_error])
                raw.append(np.r_[d.time,d.qpos.copy(),d.qvel.copy(),d.ctrl.copy(),gyro,accel])
            # Only real-time pacing sleeps; never wait for perception or planning.
            if step%5==0:
                remaining=d.time-(time.perf_counter()-wall)
                if remaining>0:time.sleep(remaining)
        runtime=time.perf_counter()-wall
    finally:
        try:inputs.put_nowait(None)
        except queue.Full:pass
        process.join(timeout=3)
        if process.is_alive():process.terminate();process.join()
    elapsed=runtime
    columns=['sim_time','true_x','true_y','true_z','estimated_x','estimated_y','estimated_z','vx','vy','vz','estimated_bx','estimated_by','rotor0','rotor1','rotor2','rotor3','tracking_error_m','nav_error_m','tilt_deg','healthy','nav_age_s','shape_age_s','contacts','true_bx','true_by','bend_error_deg']
    with (out/'flight.csv').open('w') as f:
        w=csv.writer(f);w.writerow(columns);w.writerows(rows)
    with (out/'perception_events.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['received_sim_time','observation_sim_time','nav_valid','wrist_valid','nav_correction_accepted','injected_dropout','nav_reason','wrist_reason','visual_x','visual_y','visual_z']);w.writerows(events)
    np.savez_compressed(out/'raw_dynamics_imu.npz',data=np.array(raw),nq=m.nq,nv=m.nv,nu=m.nu)
    a=np.array(rows);result=dict(stage='asynchronous_sensor_feedback_flight',full_task_accepted=False,initialization='known_spawn_and_initial_camera_alignment',navigation='RGBD_ORB_PnP_plus_noisy_simulated_gyro_accelerometer',arm_feedback='wrist_RGBD_with_estimated_UAV_compensation',com_feedforward='V2_PCC_mass_model_from_visual_shape_not_truth',sensor_process=True,sim_seconds=d.time,wall_seconds=elapsed,realtime_factor=d.time/elapsed,position_rms_m=float(np.sqrt(np.mean(a[:,16]**2))),max_nav_error_m=float(a[:,17].max()),max_tilt_deg=float(a[:,18].max()),contacts=contact_total,saturated_steps=saturated,skipped_sensor_snapshots=dropped,visual_corrections=len(events),rejected_corrections=state.rejected,max_bend_error_deg=float(a[:,25].max()),achieved_visual_hz=len(events)/d.time,dropout_invalid_frames=sum(bool(e[5]) and not bool(e[3]) for e in events))
    result['passed']=bool(result['position_rms_m']<.1 and result['max_nav_error_m']<.1 and result['max_tilt_deg']<15 and contact_total==0 and len(events)>20 and result['max_bend_error_deg']<5)
    (out/'source.py').write_text(executed_source);(out/'manifest.json').write_text(json.dumps(result,indent=2));print(out);print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=12);args=p.parse_args();main(args.seconds)
