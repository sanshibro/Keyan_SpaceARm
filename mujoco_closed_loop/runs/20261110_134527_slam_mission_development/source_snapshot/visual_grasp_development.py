"""Preset grasp-pose integration: sensor-feedback flight plus tactile contact grasp. Not navigation acceptance."""
import argparse,csv,json,time,sys,multiprocessing as mp,queue
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from pathlib import Path
from xml.etree.ElementTree import fromstring,SubElement as S,tostring
import mujoco,numpy as np
from model import model_xml,SimArmBridge
from evidence import freeze_run
from scene import textured_room
from fiducials import add_fiducials
from sensor_worker import worker,T,offer_latest
from sensors import OPTICAL_FROM_GL
from estimation import orientation_shape
from flight_feedback import InertialVisualState,RotorController
from mass_compensation import rigid_com_correction
from qingyu_model import ARC_LENGTH,BASE_LENGTH
from mission_scene import MissionScene
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'xtdrone_catkin_ws/src/uav_continuum_whole_body_v2/src'))
from uav_continuum_whole_body_v2.geometry import coupling_metrics
from uav_continuum_whole_body_v2.types import ArmModel,MassModel

def audit_entry_live_map(slam_link,map_path,rgbd_path,center,spline,tracking_reserve):
    """Build and validate a live sensor map off the physics thread."""
    from planner_adapter import V21Adapter
    started=time.monotonic()
    snapshot=slam_link.snapshot(map_path,rgbd_path,local_center=center,local_radius=1.)
    adapter=V21Adapter(snapshot,survey_camera=True,tracking_reserve=tracking_reserve)
    proof=adapter.validator.validate(spline)
    return adapter,proof,snapshot.map_hash,snapshot.stamp,time.monotonic()-started

def main(seconds=60,dashboard_video=False,gui=False,mission_map=None,seed=0,target_shift=0.,slam=False,survey_camera=False,three_gaps=False):
    if three_gaps and not slam:raise ValueError('--three-gaps needs --slam')
    if slam and survey_camera:raise ValueError('Approved Stage 1 assembly contains exactly two D435 cameras')
    overall_started=time.perf_counter()
    mission=bool(mission_map or slam)
    out=ROOT/'runs'/(time.strftime('%Y%m%d_%H%M%S')+('_slam_mission_development' if slam else '_mission_development' if mission else '_visual_grasp_development'));out.mkdir(parents=True)
    if slam and three_gaps:
        from aperture_qualification import qualify
        aperture_report=qualify()
        (out/'aperture_qualification.json').write_text(json.dumps(aperture_report,indent=2))
        if not aperture_report['passed']:
            raise RuntimeError('Three-gap physical and V2.1 aperture qualification failed')
    executed_source=Path(__file__).read_text();freeze_run(out)
    nav_profile='front_plus_survey' if mission_map else 'approved_stage1' if slam else 'front'
    if slam:
        from stage1_scene import build_flight_base
        root=build_flight_base()
    else:root=fromstring(model_xml(False,camera_profile='grasp_visible',nav_camera_profile=nav_profile))
    textured_room(root,back_wall=slam,long_room=three_gaps)
    if not slam:add_fiducials(root)
    u=root.find("worldbody/body[@name='uav']");S(u,'site',name='imu',pos='0 0 0',size='.005')
    world=root.find('worldbody')
    scene_config=(MissionScene.three_gaps(target_shift_y=target_shift,start_yaw=0.) if three_gaps
                  else MissionScene(target_shift_y=target_shift,start_yaw=0.)) if mission else None
    if scene_config:
        if mission_map:
            map_scene_path=Path(mission_map).parent/'scene_config.json'
            if not map_scene_path.exists():raise ValueError('Mission map needs its recorded scene_config.json')
            mapped_scene=json.loads(map_scene_path.read_text());current_scene=json.loads(json.dumps(scene_config.metadata()))
            # Exit is a navigation target, not physical scene geometry.
            for key in ('start','pregrasp','door_y','opening_x','opening_z','texture_seed'):
                if mapped_scene.get(key)!=current_scene[key]:raise ValueError('Mission map geometry differs from current scene: '+key)
            if three_gaps:
                for key in ('gap_count','target_y'):
                    if mapped_scene.get(key)!=current_scene[key]:raise ValueError('Mission map geometry differs from current scene: '+key)
        scene_config.add(root);(out/'scene_config.json').write_text(json.dumps(scene_config.metadata(),indent=2))
    else:
        u.set('pos','-.6 0 1.5')
        S(world,'geom',name='support',type='box',pos='0 0 .953',size='.06 .06 .025',rgba='.4 .3 .2 1')
        obj=S(world,'body',name='object',pos='0 0 .998');S(obj,'freejoint',name='object_free')
        S(obj,'geom',name='target',type='cylinder',size='.012 .02',mass='.04',friction='1 .01 .001',rgba='1 .02 .02 1')
    sen=S(root,'sensor');S(sen,'gyro',name='gyro',site='imu');S(sen,'accelerometer',name='accel',site='imu')
    # Simulated segment strain readouts. They are explicit sensor channels,
    # independent of the D435 images, and are never used as simulator truth.
    for i in range(8):
        for axis in ('bx','by'):
            S(sen,'jointpos',name=f'strain_{axis}{i}',joint=f'{axis}{i}')
    xml=tostring(root,encoding='unicode');(out/'model.xml').write_text(xml)
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    def body(name):
        i=m.body(name).id;return T(d.xmat[i].reshape(3,3),d.xpos[i])
    def camera(name):
        i=m.camera(name).id;return T(d.cam_xmat[i].reshape(3,3)@OPTICAL_FROM_GL,d.cam_xpos[i])
    # Known initial placement and rigid calibration only. No subsequent truth used by controller.
    U0=body('uav');nav0=camera('navigation');wrist0=camera('wrist_rgbd')
    calibration=dict(initial_nav=nav0,initial_wrist=wrist0,UC=np.linalg.solve(U0,nav0),UB=np.linalg.solve(U0,body('mount')),TC=np.linalg.solve(body('tool'),wrist0))
    if slam and survey_camera:calibration['US']=np.linalg.solve(U0,camera('survey_rgbd'))
    calibration['nav_profile']=nav_profile
    calibration['super_visual']=bool(slam)
    calibration['robot_visual_profile']='approved_stage1' if slam else 'development_base'
    calibration['localization_mode']='fast_lio_external' if slam else 'surveyed_fiducials'
    calibration['live_preview']=bool(dashboard_video or gui)
    preview_buffer=None
    if dashboard_video or gui:
        from shared_preview import SharedPreview
        preview_buffer=SharedPreview()
    slam_link=None
    if slam:
        from live_fast_lio import LiveFastLioLink
        slam_link=LiveFastLioLink(U0,out,seed,nav_profile=nav_profile)
    ctx=mp.get_context('spawn');inputs=ctx.Queue(maxsize=1);outputs=ctx.Queue();process=ctx.Process(target=worker,args=(xml,inputs,outputs,calibration,str(out),preview_buffer.spec if preview_buffer else None,preview_buffer.lock if preview_buffer else None));process.start()
    try:
        ready=outputs.get(timeout=30)
        if 'error' in ready:raise RuntimeError(ready['error'])
        inputs.put((0.,d.qpos.copy(),d.qvel.copy(),np.zeros(2),U0.copy() if slam else None));first=outputs.get(timeout=30)
        if 'error' in first:raise RuntimeError(first['error'])
    except BaseException:
        if process.is_alive():process.terminate();process.join()
        if preview_buffer is not None:preview_buffer.close()
        raise
    dashboard=None;preview=None;last_consumed_sensor_stamp=first['stamp']
    if dashboard_video or gui:
        from live_dashboard import LiveDashboard
        dashboard=LiveDashboard(xml,out,show=gui,map_path=mission_map,preview_buffer=preview_buffer,slam=slam)
    (out/'calibration.json').write_text(json.dumps({k:(v.tolist() if hasattr(v,'tolist') else v) for k,v in calibration.items()},indent=2))
    state=InertialVisualState(U0[:3,3]);state.R=U0[:3,:3].copy();controller=RotorController();bridge=SimArmBridge();bend=np.zeros(2);last_shape=0.;arm_bias=np.zeros(2)
    if slam:
        controller.position_gain=2.;controller.velocity_gain=3.;controller.integral_gain=.25
        controller.integral_leak_s=3.;controller.horizontal_acceleration_limit=.12
    arm=ArmModel(arc_length=ARC_LENGTH,base_offset_z=BASE_LENGTH);mass=MassModel();com=np.zeros(3)
    from async_frontier import AsyncFrontierSelector
    frontier_selector=AsyncFrontierSelector() if slam else None
    map_executor=ThreadPoolExecutor(max_workers=2) if slam else None
    frontier_snapshot_future=None;frontier_snapshot_state=None;frontier_snapshot_path=None;entry_navigation_future=None
    entry_navigation=None;entry_frontier=False;entry_leg=0;entry_route=scene_config.navigation_route() if scene_config else [];scan_route_indices=scene_config.scan_route_indices() if scene_config else ();entry_route_index=0;next_entry_attempt=12.;failed_entry_plans=0;entry_stable_since=None;frontier_survey_start=-1.;frontier_survey_end=-1.;frontier_survey_count=0;pre_gate_scan_count=0;held_entry_bend=np.zeros(2);exit_navigation=None;exit_final_navigation=None;exit_stage_done=False;exit_route_goals=[];exit_route_index=0;navigation_done=False;mission_finished=False
    if mission_map:
        from mission_navigation import MissionNavigator
        entry_navigation=MissionNavigator(mission_map,out,'entry',np.r_[U0[:3,3],0.,0.,0.],scene_config.pregrasp)
        exit_clearance_goal=np.asarray(scene_config.pregrasp)+np.array([0.,0.,.05])
        exit_navigation=MissionNavigator(mission_map,out,'exit_clearance',np.array([0,scene_config.target_y+target_shift,1.75,0,0,0.]),exit_clearance_goal,loaded=True)
        exit_final_navigation=MissionNavigator(mission_map,out,'exit_portal',np.r_[exit_clearance_goal,0.,0.,0.],scene_config.exit,loaded=True)
    target_realtime_factor=.5 if slam else 1.
    (out/'task_config.json').write_text(json.dumps(dict(seed=seed,mission_map=str(mission_map) if mission_map else None,online_map_connected=bool(slam),target_shift_y=target_shift,gap_count=scene_config.gap_count if scene_config else 0,navigation_route=entry_route,starts_in_known_hover=True,localization_mode=calibration['localization_mode'],navigation_camera_profile=nav_profile,super_visual=bool(slam),underbody_survey_camera=bool(slam and survey_camera),target_realtime_factor=target_realtime_factor),indent=2))
    rotor_ids=[m.actuator('thrust%d'%i).id for i in range(4)];uid=m.body('uav').id;rng=np.random.default_rng(seed)
    target_samples=[];last_target_stamp=-100.;body_target_samples=[];last_body_target_stamp=-100.;body_target_observations=0;loss_reference=None;relative_target=None;relative_velocity=np.zeros(3);previous_relative=None;previous_relative_stamp=None;descent_progress=0.;grasp_pose=None;visual_target=None;approach_start=None;descent_start=None;close_start=None;mission_start=None;object_rows=[];confirmed=False;lost=False;loss_since=None;hold_target=None;confirmation_time=None;phase='SENSOR_READY';pads=[m.geom('pad%d'%i).id for i in range(3)];target_id=m.geom('target').id;support_id=m.geom('support').id;object_id=m.body('object').id
    interrupted=False;end_reason='time_limit';rows=[];events=[];latest_target_reason='initializing';dropped=0;saturated=0;contact_total=0;wall=time.perf_counter();raw=[];schedule=[];shape_updates=0;strain_fallback_steps=0;hover_zupt_steps=0
    strain_addresses=np.array([[m.sensor(f'strain_{axis}{i}').adr[0] for i in range(8)] for axis in ('bx','by')])
    try:
        for step in range(int(seconds*1000)):
            t=d.time
            while True:
                try:item=outputs.get_nowait()
                except queue.Empty:break
                if 'error' in item:raise RuntimeError(item['error'])
                last_consumed_sensor_stamp=item['stamp']
                accepted=False;latest_target_reason=item.get('target_reason','unknown')
                if slam and item.get('target_body_coarse') is not None:
                    body_target_samples.append(item['target_body_coarse']);body_target_samples=body_target_samples[-5:]
                    last_body_target_stamp=item['stamp'];body_target_observations+=1
                if item.get('target') is not None and not confirmed and (not mission or navigation_done):
                    # Body-relative visual error cancels common navigation translation bias.
                    delta=state.R@item['tool_error_U'] if item.get('tool_error_U') is not None else item['target']-item['WU'][:3,3]+np.array([0,0,.5084])
                    if previous_relative is not None and .01<item['stamp']-previous_relative_stamp<.5:
                        observed_velocity=-(delta-previous_relative)/(item['stamp']-previous_relative_stamp)
                        if np.linalg.norm(observed_velocity)<1:relative_velocity=.5*relative_velocity+.5*observed_velocity
                    previous_relative=delta.copy();previous_relative_stamp=item['stamp']
                    relative_target=delta.copy()
                    adjusted=state.p+delta-state.v*max(0,t-item['stamp'])
                    target_samples.append(adjusted);last_target_stamp=item['stamp']
                    target_samples=target_samples[-5:]
                    if len(target_samples)>=3:visual_target=np.median(target_samples,axis=0)
                if item['nav_valid'] and not slam:accepted=state.correct(item['stamp'],item['WU'],t)
                if item['shape'] is not None and t-item['stamp']<.3:bend=item['shape'];last_shape=item['stamp'];shape_updates+=1
                events.append([t,item['stamp'],item['nav_valid'],item['wrist_valid'],accepted,item['dropout'],item['nav_reason'],item['wrist_reason'],*(item['WU'][:3,3] if item['WU'] is not None else [float('nan')]*3),item.get('nav_inliers'),item.get('wrist_inliers'),item.get('nav_reprojection_px'),item.get('wrist_reprojection_px'),item.get('nav_depth_fit_rms_m'),item.get('wrist_depth_fit_rms_m'),item.get('target_pose_source')])
            if slam:
                for stamp,pose,velocity in slam_link.drain_poses():
                    state.correct(stamp,pose,t,velocity_gain=.65,velocity_window=.35,orientation_gain=0.,position_gain=.45 if t>6. else .2,velocity_hint=velocity)
            # Visual shape remains primary. During loss of wrist features,
            # use the measured segment bend channels with independent noise.
            strain_measurement=d.sensordata[strain_addresses].sum(axis=1)+rng.normal(0,.002,2)
            if t-last_shape>.15:
                bend=strain_measurement
                last_shape=t
                strain_fallback_steps+=1
            healthy=t-state.last_visual<.5 and t-last_shape<.5
            s=0.
            touched=set()
            for k in range(d.ncon):
                c=d.contact[k];force=np.zeros(6);mujoco.mj_contactForce(m,d,k,force)
                # Reject finger/finger contact as a grasp signal. No target pose enters control.
                if force[0]>.01 and not (c.geom1 in pads and c.geom2 in pads):touched.update(p for p in pads if p in (c.geom1,c.geom2))
            if visual_target is not None and approach_start is None:
                approach_start=t
            phase='APPROACH' if approach_start is not None else 'TARGET_SEARCH'
            fresh=relative_target is not None and t-last_target_stamp<.3 and healthy
            error=np.zeros(3) if relative_target is None else relative_target-state.v*min(.3,max(0,t-last_target_stamp))
            if not fresh and loss_reference is None:loss_reference=state.p.copy()
            if fresh:loss_reference=None
            if fresh and np.linalg.norm(error[:2])<.008:
                if descent_start is None:descent_start=t
                descent_progress=min(1.,descent_progress+.001/12.)
            elif np.linalg.norm(error[:2])>.015:
                descent_progress=max(0.,descent_progress-.001/2.)
            if descent_start is not None:phase='DESCENT'
            requested=error+np.array([0,0,.06*(1-descent_progress)])
            reference=state.p+np.clip(requested,[-.08,-.08,-.015],[.08,.08,.015]) if fresh else loss_reference.copy()
            controller.integral=np.clip(controller.integral,-.10 if slam else -.03,.10 if slam else .03)
            if descent_progress>=1 and fresh and np.linalg.norm(error[:2])<.006 and abs(error[2])<.012 and np.linalg.norm(relative_velocity)<.03 and close_start is None:close_start=t
            if not fresh and not confirmed and visual_target is not None and t-last_target_stamp>.5:
                phase='TARGET_RECOVERY';descent_progress=0.;descent_start=None;close_start=None;d.ctrl[16:19]=0
                backoff=visual_target+np.array([-.12,0,.06]);backoff[2]=max(backoff[2],loss_reference[2])
                weight=.5-.5*np.cos(np.pi*np.clip((t-last_target_stamp-.5)/3,0,1))
                reference=loss_reference+(backoff-loss_reference)*weight
            if close_start is not None:
                phase='CLOSING';d.ctrl[16:19]=.024*np.clip((t-close_start)/.5,0,1)
            if close_start is not None and t-close_start>.6 and len(touched)==3 and not confirmed and not lost:
                confirmed=True;confirmation_time=t;mission_start=t;grasp_pose=state.p.copy();controller.mass=2.29;mass.tool=.24
            if confirmed:phase='CONTACT_CONFIRMED'
            if confirmed and mission_start is not None and t-mission_start>1.5 and len(touched)<2:
                if loss_since is None:loss_since=t
                if t-loss_since>.25 and not lost:
                    lost=True;hold_target=state.p.copy();controller.mass=2.25;mass.tool=.20
            else:loss_since=None
            if confirmed and t-mission_start>=1.5:phase='LIFT'
            if confirmed and t-mission_start>=6:phase='CARRY'
            if confirmed and t-mission_start>=15:phase='HOLD'
            if lost:phase='GRIP_LOST_HOLD'
            if slam and navigation_done and not confirmed and not fresh and body_target_samples and t-last_body_target_stamp<.5:
                coarse=np.median(body_target_samples,axis=0)
                coarse_goal=np.array([coarse[0]-.6,coarse[1],coarse[2]+.5])
                reference=state.p+np.clip(coarse_goal-state.p,[-.04,-.04,-.015],[.04,.04,.015])
                phase='BODY_D435_COARSE_ALIGN'
            # Keep the certified arm shape while the next online frontier is
            # selected. Releasing it here straightened the arm after every
            # stationary-bend leg, so the planner repeated that leg forever.
            desired=held_entry_bend.copy() if slam and not navigation_done else np.zeros(2)
            if slam and entry_navigation is None and not navigation_done and t<12.:
                survey_theta=np.deg2rad(50)*np.sin(np.pi*t/12)**2
                survey_phi=2*np.pi*t/12
                desired=survey_theta*np.array([np.cos(survey_phi),np.sin(survey_phi)])
                phase='SLAM_ACTIVE_SURVEY'
            elif slam and entry_navigation is None and not navigation_done and frontier_survey_start<=t<frontier_survey_end:
                if np.linalg.norm(held_entry_bend)<.05:
                    survey_phase=(t-frontier_survey_start)/(frontier_survey_end-frontier_survey_start)
                    survey_theta=np.deg2rad(50)*np.sin(np.pi*survey_phase)**2
                    survey_phi=2*np.pi*survey_phase
                    desired=survey_theta*np.array([np.cos(survey_phi),np.sin(survey_phi)])
                    phase='PRE_GATE_ACTIVE_SCAN'
                else:
                    phase='PRE_GATE_LIDAR_SCAN'
            # Inner physical position actuators already track the desired
            # bending. Avoid a second high-gain loop around delayed vision.
            commanded=desired;theta=min(np.linalg.norm(commanded),np.deg2rad(120));phi=np.arctan2(commanded[1],commanded[0])
            # On loss, retain the previous actuator reference. Feeding a
            # measured gravity-deflected shape back as a new setpoint adds
            # the gravity deflection repeatedly at every intermittent loss.
            if healthy:d.ctrl[:16]=bridge.move_joints(theta,phi)
            if step%20==0:
                com=coupling_metrics([0,0,0,0,np.linalg.norm(bend),np.arctan2(bend[1],bend[0])],arm,mass)['system_com']
                # Encoded rigid tool mass distribution versus the original uniform tool model.
                com+=rigid_com_correction(bend,controller.mass,survey=mission,loaded=confirmed and not lost,approved_stage1=slam)
            gyro=d.sensordata[m.sensor('gyro').adr[0]:m.sensor('gyro').adr[0]+3].copy()+rng.normal(0,.0002,3)
            accel=d.sensordata[m.sensor('accel').adr[0]:m.sensor('accel').adr[0]+3].copy()+rng.normal(0,.01,3)
            if slam and step%5==0:slam_link.publish_imu(t,accel,gyro)
            if slam and step and step%100==0:slam_link.publish_scan(m,d,uid,bend,contact_total==0)
            lift=(.25 if mission else .10)*.5*(1-np.cos(np.pi*np.clip((t-mission_start-1.5)/3,0,1))) if confirmed else 0.
            carry=.50*.5*(1-np.cos(np.pi*np.clip((t-mission_start-6)/7,0,1))) if confirmed and not mission else 0.
            target=hold_target if lost else (grasp_pose if confirmed else reference)+np.array([carry,0,lift])
            control=state
            if fresh and not confirmed:
                control=SimpleNamespace(p=state.p,R=state.R,v=relative_velocity.copy(),visual_servo=True)
            commanded_yaw=scene_config.start_yaw+(.5*np.sin(2*np.pi*t/12) if entry_navigation is None and t<12. else 0.) if slam else 0.
            stable_shape=held_entry_bend if slam and entry_navigation is None and not navigation_done else np.zeros(2)
            stable_now=np.linalg.norm(state.v)<.028 and np.linalg.norm(state.last_velocity_measurement)<.07 and abs(gyro[2])<.06 and np.linalg.norm(bend-stable_shape)<.04
            if stable_now:
                if entry_stable_since is None:entry_stable_since=t
            else:entry_stable_since=None
            entry_stable=entry_stable_since is not None and t-entry_stable_since>=.5
            if slam and t>=next_entry_attempt and entry_navigation is None and entry_navigation_future is None and frontier_selector.process is None and frontier_snapshot_future is None and not navigation_done and entry_stable and slam_link.map_scans>=20 and (out/'rgbd_online_map.npz').exists():
                map_path=out/(f'entry_online_map_{entry_leg:02d}.npz')
                yaw_est=np.arctan2(state.R[1,0],state.R[0,0])
                measured=np.r_[state.p,yaw_est,np.linalg.norm(bend),np.arctan2(bend[1],bend[0])]
                frontier_snapshot_state=measured.copy();frontier_snapshot_path=map_path
                frontier_snapshot_future=map_executor.submit(slam_link.snapshot,map_path,out/'rgbd_online_map.npz',state.p.copy(),1.)
            if slam and frontier_snapshot_future is not None and frontier_snapshot_future.done() and step%100==0:
                try:
                    snapshot=frontier_snapshot_future.result()
                    frontier_selector.request(frontier_snapshot_path,frontier_snapshot_state,entry_route[entry_route_index],snapshot.map_hash)
                except Exception as error:
                    (out/(f'entry_frontier_{entry_leg:02d}.json')).write_text(json.dumps(dict(status='map_snapshot_failed_hold',kind='hold',waypoint=None,information={'reason':type(error).__name__+': '+str(error)}),indent=2))
                    next_entry_attempt=t+3.
                frontier_snapshot_future=None
            if slam and frontier_selector.process is not None and step%100==0:
                selected=frontier_selector.poll()
                if selected is not None:
                    from mission_navigation import MissionNavigator
                    waypoint=selected.get('waypoint');kind=selected.get('kind','hold')
                    frontier_info=selected.get('information',{})
                    selection_state=frontier_selector.request_state
                    map_path=frontier_selector.map_path
                    selected['requested_state']=selection_state.tolist()
                    selected['result_position']=state.p.tolist()
                    drift=float(np.linalg.norm(state.p-selection_state[:3]))
                    selected['position_drift_m']=drift
                    if drift>.05:
                        waypoint=None;kind='hold';frontier_info=dict(reason='frontier_state_drift_reject',drift_m=drift)
                        selected.update(waypoint=None,kind=kind,information=frontier_info)
                    (out/(f'entry_frontier_{entry_leg:02d}.json')).write_text(json.dumps(selected,indent=2))
                    if waypoint is None:
                        if frontier_survey_count<3:
                            frontier_survey_start=t;frontier_survey_end=t+8.;frontier_survey_count+=1
                            next_entry_attempt=frontier_survey_end+1.
                        else:next_entry_attempt=t+3.
                    else:
                        entry_frontier=kind=='frontier'
                        entry_navigation_future=map_executor.submit(MissionNavigator,map_path,out,f'entry_leg_{entry_leg:02d}',selection_state,waypoint,frontier=entry_frontier,goal_shape=frontier_info.get('terminal_shape'),live_map_required=True)
                        entry_leg+=1
            if slam and entry_navigation_future is not None and entry_navigation_future.done() and step%100==0:
                try:entry_navigation=entry_navigation_future.result()
                except Exception as error:
                    failed_entry_plans+=1;next_entry_attempt=t+3.
                    (out/(f'entry_leg_{entry_leg-1:02d}_initialization_error.json')).write_text(json.dumps(dict(error=type(error).__name__+': '+str(error),sim_time=t),indent=2))
                entry_navigation_future=None
            if slam and confirmed and exit_navigation is None and slam_link.map_scans>=5:
                from mission_navigation import MissionNavigator
                map_path=out/'exit_online_map.npz';slam_link.snapshot(map_path,out/'rgbd_online_map.npz')
                exit_clearance_goal=np.asarray(scene_config.pregrasp)+np.array([0.,0.,.05])
                exit_route_goals=[exit_clearance_goal.tolist()]+(entry_route[-2::-1] if three_gaps else [])+[scene_config.exit]
                initial=np.r_[state.p,np.arctan2(state.R[1,0],state.R[0,0]),np.linalg.norm(bend),np.arctan2(bend[1],bend[0])]
                exit_navigation=MissionNavigator(map_path,out,'exit_route_00',initial,exit_route_goals[0],loaded=True)
            navigator=None
            if entry_navigation is not None and not navigation_done:navigator=entry_navigation
            elif exit_navigation is not None and confirmed and not lost and t-mission_start>=6:
                navigator=exit_navigation if slam or not exit_stage_done else exit_final_navigation
            if navigator is not None:
                if (slam and navigator.spline is not None and navigator.phase>=navigator.spline.duration
                    and state.correct_stationary_velocity(t,gyro,accel)):
                    hover_zupt_steps+=1
                nav_reference=navigator.update(t,state.p,state.R,state.v,bend,gyro,healthy) if t>=3 else navigator.reference.copy()
                if slam and navigator is entry_navigation:
                    refresh_future=getattr(navigator,'live_refresh_future',None)
                    if refresh_future is not None and refresh_future.done():
                        refresh_index=navigator.live_refresh_pending_index
                        first_refresh=not navigator.live_map_refreshed
                        try:
                            current_adapter,current_proof,map_hash,map_stamp,audit_wall=refresh_future.result()
                            current_state=np.r_[state.p,np.arctan2(state.R[1,0],state.R[0,0]),np.linalg.norm(bend),np.arctan2(bend[1],bend[0])]
                            current_clearance=current_adapter.clearance(current_state)
                            takeover_safe=(current_clearance['body']>=current_adapter.limits.body_margin and current_clearance['arm']>=current_adapter.limits.arm_margin and current_proof.feasible)
                            report=dict(safe=takeover_safe,current_clearance=current_clearance,validation=current_proof.__dict__,map_hash=map_hash,map_stamp=map_stamp,audit_wall_s=audit_wall,sim_time=t)
                        except Exception as error:
                            takeover_safe=False;report=dict(safe=False,error=type(error).__name__+': '+str(error),sim_time=t)
                        (out/(navigator.name+f'_live_refresh_{refresh_index:02d}.json')).write_text(json.dumps(report,default=str,indent=2))
                        navigator.live_refresh_future=None;navigator.live_map_refreshed=True
                        if takeover_safe:navigator.adapter=current_adapter;navigator.needs_map_refresh=False
                        elif first_refresh:navigator.spline=None;navigator.status='live_takeover_rejected'
                    refresh_due=(not navigator.live_map_refreshed or (navigator.needs_map_refresh and t-getattr(navigator,'last_live_refresh',-100.)>=.5))
                    if navigator.spline is not None and refresh_due and getattr(navigator,'live_refresh_future',None) is None:
                        refresh_index=getattr(navigator,'live_refresh_count',0)
                        latest_map=out/(navigator.name+f'_live_map_{refresh_index:02d}.npz')
                        navigator.live_refresh_future=map_executor.submit(audit_entry_live_map,slam_link,latest_map,out/'rgbd_online_map.npz',state.p.copy(),navigator.spline,navigator.tracking_reserve)
                        navigator.live_refresh_pending_index=refresh_index
                        navigator.last_live_refresh=t;navigator.live_refresh_count=refresh_index+1
                target=nav_reference[:3];reference=target.copy();control=state;commanded_yaw=float(nav_reference[3])
                if healthy:
                    desired_bend=float(nav_reference[4])*np.array([np.cos(nav_reference[5]),np.sin(nav_reference[5])])
                    # Slow integral feedforward cancels measured gravity sag.
                    # The world reference and tracking safety gate are not
                    # moved; only the physical position command is adjusted.
                    arm_bias=np.clip(arm_bias+.0005*np.clip(desired_bend-bend,-.1,.1),-.08,.08)
                    command_bend=desired_bend+arm_bias
                    command_theta=min(np.linalg.norm(command_bend),np.deg2rad(120))
                    d.ctrl[:16]=bridge.move_joints(command_theta,float(np.arctan2(command_bend[1],command_bend[0])))
                phase=('ENTRY_' if navigator is entry_navigation else 'EXIT_')+navigator.status.upper()
                if navigator is entry_navigation and navigator.done:
                    if slam and entry_frontier:
                        hold_goal=navigator.goal.copy()
                        held_entry_bend=float(nav_reference[4])*np.array([np.cos(nav_reference[5]),np.sin(nav_reference[5])])
                        navigator.close();entry_navigation=None;next_entry_attempt=t+.1
                        loss_reference=hold_goal
                        controller.integral[:]=0.;controller.velocity_integral[:]=0.
                    elif slam and entry_route_index+1<len(entry_route):
                        hold_goal=navigator.goal.copy();entry_route_index+=1
                        held_entry_bend=float(nav_reference[4])*np.array([np.cos(nav_reference[5]),np.sin(nav_reference[5])])
                        navigator.close();entry_navigation=None;entry_frontier=False;next_entry_attempt=t+.1
                        loss_reference=hold_goal
                        controller.integral[:]=0.;controller.velocity_integral[:]=0.
                        if entry_route_index-1 in scan_route_indices:
                            frontier_survey_start=t;frontier_survey_end=t+8.
                            next_entry_attempt=frontier_survey_end+.5
                            pre_gate_scan_count+=1
                            (out/f'pre_gate_scan_{pre_gate_scan_count:02d}.json').write_text(json.dumps({
                                'sim_time':t,'route_index':entry_route_index-1,
                                'stage_position_estimate':state.p.tolist(),
                                'scan_until_sim_time':frontier_survey_end,
                                'reason':'before_aperture_joint_planning'},indent=2))
                    else:navigation_done=True;loss_reference=navigator.goal.copy()
                if navigator is exit_navigation and navigator.done:
                    if slam:
                        if exit_route_index+1==len(exit_route_goals):mission_finished=True
                        else:
                            navigator.close();exit_route_index+=1
                            map_path=out/f'exit_online_map_{exit_route_index:02d}.npz'
                            slam_link.snapshot(map_path,out/'rgbd_online_map.npz')
                            initial=np.r_[state.p,np.arctan2(state.R[1,0],state.R[0,0]),np.linalg.norm(bend),np.arctan2(bend[1],bend[0])]
                            exit_navigation=MissionNavigator(map_path,out,f'exit_route_{exit_route_index:02d}',initial,exit_route_goals[exit_route_index],loaded=True)
                            controller.integral[:]=0.;controller.velocity_integral[:]=0.
                    else:exit_stage_done=True
                if navigator is exit_final_navigation and navigator.done:mission_finished=True
                if navigator.requested and navigator.planner.process is None and navigator.spline is None:
                    if slam and navigator is entry_navigation:
                        failed_entry_plans+=1
                        navigator.close();entry_navigation=None;next_entry_attempt=t+3.;loss_reference=state.p.copy()
                        if failed_entry_plans>=5:
                            end_reason='entry_online_frontier_retries_exhausted';break
                    else:
                        end_reason=('entry_' if navigator is entry_navigation else 'exit_')+navigator.status;break
            feedforward_scale=.5 if slam else 1.
            forces,sat=controller.command(control,gyro,target,com,yaw=commanded_yaw,
                target_velocity=navigator.reference_velocity*feedforward_scale if navigator else None,
                target_acceleration=navigator.reference_acceleration*feedforward_scale if navigator else None)
            d.ctrl[rotor_ids]=forces;saturated+=int(sat)
            mujoco.mj_step(m,d);state.propagate(d.time,gyro,accel,.001)
            schedule.append([d.time,time.perf_counter()-wall])
            if dashboard is not None and step%200==0:dashboard.submit(d,None,state.p,phase,last_consumed_sensor_stamp)
            for k in range(d.ncon):
                pair={d.contact[k].geom1,d.contact[k].geom2}
                if pair=={target_id,support_id} or (target_id in pair and any(p in pair for p in pads)):continue
                contact_total+=1
            if step%50==0:
                lio_pose=slam_link.latest_pose() if slam else None
                external_WU=lio_pose[1] if lio_pose is not None and d.time-lio_pose[0]<.3 else None
                dropped+=offer_latest(inputs,(d.time,d.qpos.copy(),d.qvel.copy(),strain_measurement.copy(),external_WU))
            if mission_finished:
                end_reason='completed_mission_hold';break
            if not mission and phase=='HOLD' and t-mission_start>=18 and len(touched)==3 and np.linalg.norm(state.p-target)<.03:
                end_reason='completed_hold';break
            if lost and loss_since is not None and t-loss_since>2:
                end_reason='grip_loss';break
            if (out/'STOP_REQUEST').exists():
                interrupted=True;end_reason='stop_requested';break
            if step%1000==0:
                nav_trace=navigator.trace[-1] if navigator is not None and navigator.trace else None
                (out/'progress.json').write_text(json.dumps({'sim_time':d.time,'phase':phase,'nav_age':t-state.last_visual,'shape_age':t-last_shape,'target_seen':visual_target is not None,'target_age':t-last_target_stamp,'target_fresh':fresh,'target_reason':latest_target_reason,'confirmed':confirmed,'lost':lost,'estimated_position':state.p.tolist(),'estimated_velocity':state.v.tolist(),'lio_velocity':state.last_velocity_measurement.tolist() if slam else None,'reference_position':reference.tolist(),'navigator_name':navigator.name if navigator else None,'navigator_phase':navigator.phase if navigator else None,'navigator_phase_rate':navigator.phase_rate if navigator else None,'tracking_budget_m':float(nav_trace[3]) if nav_trace is not None else None,'whole_body_error_m':float(nav_trace[4]) if nav_trace is not None else None,'frontier_survey_count':frontier_survey_count if slam else None,'pre_gate_scan_count':pre_gate_scan_count if slam else None,'route_index':entry_route_index if slam else None,'relative_error':error.tolist(),'descent_progress':descent_progress}))
            if step%10==0:
                object_rows.append([d.time,*d.xpos[object_id],len(touched),confirmed,phase,*(visual_target if visual_target is not None else [float('nan')]*3),*reference])
                actual=d.xpos[uid].copy();tilt=np.rad2deg(np.arccos(np.clip(d.xmat[uid].reshape(3,3)[2,2],-1,1)))
                actual_shape=orientation_shape(np.linalg.solve(body('mount'),camera('wrist_rgbd')),calibration['TC'])
                bend_error=float(np.rad2deg(np.linalg.norm(actual_shape-bend)))
                true_R=d.xmat[uid].reshape(3,3)
                rotation_error=np.arccos(np.clip((np.trace(state.R@true_R.T)-1)/2,-1,1))
                def roll_pitch(R):return np.rad2deg([np.arctan2(R[2,1],R[2,2]),np.arctan2(-R[2,0],np.hypot(R[2,1],R[2,2]))])
                rows.append([d.time,*actual,*state.p,*state.v,*bend,*forces,np.linalg.norm(actual-target),np.linalg.norm(actual-state.p),tilt,healthy,t-state.last_visual,t-last_shape,d.ncon,*actual_shape,bend_error,np.rad2deg(rotation_error),*roll_pitch(true_R),*roll_pitch(state.R)])
                raw.append(np.r_[d.time,d.qpos.copy(),d.qvel.copy(),d.ctrl.copy(),gyro,accel])
            # Only real-time pacing sleeps; never wait for perception or planning.
            if step%5==0:
                remaining=d.time/target_realtime_factor-(time.perf_counter()-wall)
                if remaining>0:time.sleep(remaining)
        runtime=time.perf_counter()-wall
    except KeyboardInterrupt:
        interrupted=True;end_reason='interrupted';runtime=time.perf_counter()-wall
    finally:
        try:inputs.put_nowait(None)
        except queue.Full:pass
        process.join(timeout=3)
        if process.is_alive():process.terminate();process.join()
        if frontier_selector is not None:frontier_selector.close()
        if map_executor is not None:map_executor.shutdown(wait=True)
        if entry_navigation is not None:entry_navigation.close()
        if exit_navigation is not None:exit_navigation.close()
        if exit_final_navigation is not None:exit_final_navigation.close()
        if dashboard is not None:dashboard.close()
        slam_stats=slam_link.close() if slam else None
    elapsed=runtime
    columns=['sim_time','true_x','true_y','true_z','estimated_x','estimated_y','estimated_z','vx','vy','vz','estimated_bx','estimated_by','rotor0','rotor1','rotor2','rotor3','tracking_error_m','nav_error_m','tilt_deg','healthy','nav_age_s','shape_age_s','contacts','true_bx','true_by','bend_error_deg','orientation_error_deg','true_roll_deg','true_pitch_deg','estimated_roll_deg','estimated_pitch_deg']
    with (out/'flight.csv').open('w') as f:
        w=csv.writer(f);w.writerow(columns);w.writerows(rows)
    with (out/'perception_events.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['received_sim_time','observation_sim_time','nav_valid','wrist_valid','nav_correction_accepted','injected_dropout','nav_reason','wrist_reason','visual_x','visual_y','visual_z','nav_inliers','wrist_inliers','nav_reprojection_px','wrist_reprojection_px','nav_depth_fit_rms_m','wrist_depth_fit_rms_m','target_pose_source']);w.writerows(events)
    with (out/'object_evaluation.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['sim_time','object_x','object_y','object_z','touched_fingers','confirmed','phase','detected_x','detected_y','detected_z','reference_x','reference_y','reference_z']);w.writerows(object_rows)
    np.savez_compressed(out/'raw_dynamics_imu.npz',data=np.array(raw),nq=m.nq,nv=m.nv,nu=m.nu)
    np.savez_compressed(out/'physics_wall_schedule.npz',data=np.array(schedule),columns=['sim_time','wall_time_since_physics_start'])
    a=np.array(rows);result=dict(stage='visual_target_approach_sensor_flight_contact_grasp_development',full_task_accepted=False,initialization='known_spawn_and_initial_camera_alignment',navigation='FAST_LIO_lidar_IMU' if slam else 'surveyed_ArUco_pixel_localization_plus_noisy_simulated_IMU',arm_feedback='wrist_RGBD_with_estimated_UAV_compensation_and_segment_strain',com_feedforward='V2_PCC_mass_model_from_visual_shape_not_truth',sensor_process=True,sim_seconds=d.time,wall_seconds=elapsed,realtime_factor=d.time/elapsed,position_rms_m=float(np.sqrt(np.mean(a[:,16]**2))),max_nav_error_m=float(a[:,17].max()),max_tilt_deg=float(a[:,18].max()),contacts=contact_total,saturated_steps=saturated,skipped_sensor_snapshots=dropped,visual_corrections=len(events),rejected_corrections=state.rejected,max_bend_error_deg=float(a[:,25].max()),achieved_visual_hz=len(events)/d.time,dropout_invalid_frames=sum(bool(e[5]) and not bool(e[3]) for e in events))
    final=[r for r in object_rows if r[0]>=max(0,d.time-3.)]
    final_carry=min(float(np.linalg.norm([r[1],r[2]-scene_config.target_y-target_shift])) for r in final) if mission else min(r[1] for r in final)
    limitation=('Known initial hover; online FAST-LIO and first-hit LiDAR map; '+('three-gap' if three_gaps else 'single-portal')+' development scene') if slam else 'Known initial hover, surveyed fiducials, controlled cylinder; mission development uses frozen sensor presurvey' if mission else 'Surveyed printed fiducial map and controlled cylinder; V2 obstacle navigation not connected yet'
    result.update(end_reason=end_reason,interrupted=interrupted,initialization_limitation=limitation,approach_started=approach_start,descent_started=descent_start,close_started=close_start,target_observations=len(target_samples),body_d435_coarse_observations=body_target_observations,contact_confirmed=confirmed,grip_lost=lost,confirmation_time=confirmation_time,final_min_object_lift_m=min(r[3]-.998 for r in final),final_min_object_carry_m=final_carry,final_min_touched_fingers=min(r[4] for r in final),binding_used=False)
    result['passed']=bool(result['position_rms_m']<.1 and result['max_nav_error_m']<.1 and result['max_tilt_deg']<15 and contact_total==0 and len(events)>20 and result['max_bend_error_deg']<5 and not interrupted and confirmed and not lost and result['final_min_object_lift_m']>.05 and result['final_min_object_carry_m']>.45 and result['final_min_touched_fingers']==3 and end_reason in ('completed_hold','completed_mission_hold'))
    gaps=np.diff(np.array(schedule)[:,1])*1000 if len(schedule)>1 else np.array([0.])
    result.update(initialization_wall_seconds=wall-overall_started,end_to_end_wall_seconds=time.perf_counter()-overall_started,target_realtime_factor=target_realtime_factor,control_sim_hz=1000,physics_wall_gap_p95_ms=float(np.percentile(gaps,95)),physics_wall_gap_max_ms=float(gaps.max()),perception_packets=len(events),visual_corrections=sum(bool(e[4]) for e in events),accepted_shape_observations=shape_updates,achieved_shape_hz=shape_updates/d.time,strain_fallback_steps=strain_fallback_steps,shape_fallback_source='simulated_jointpos_strain',hover_zupt_steps=hover_zupt_steps,arm_bias_final_rad=arm_bias.tolist())
    if mission:
        result.update(stage=('three_gap' if three_gaps else 'single_portal')+('_online_slam_development' if slam else '_navigation_grasp_loaded_exit_development'),seed=seed,target_shift_y=target_shift,gap_count=scene_config.gap_count,entry_route_index=entry_route_index,entry_route_goals=entry_route,exit_route_index=exit_route_index if slam else None,exit_route_goals=exit_route_goals if slam else None,navigation_completed=navigation_done,exit_completed=mission_finished,online_map_connected=bool(slam),slam_stats=slam_stats,online_entry_legs=entry_leg if slam else None,online_entry_planning_retries=failed_entry_plans if slam else None)
        result['passed']=bool(result['passed'] and mission_finished and result['final_min_object_carry_m']>.5)
    (out/'source.py').write_text(executed_source);(out/'manifest.json').write_text(json.dumps(result,indent=2));print(out);print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=60);p.add_argument('--dashboard-video',action='store_true');p.add_argument('--gui',action='store_true');p.add_argument('--mission-map',type=Path);p.add_argument('--slam',action='store_true');p.add_argument('--survey-camera',action='store_true');p.add_argument('--three-gaps',action='store_true');p.add_argument('--seed',type=int,default=0);p.add_argument('--target-shift',type=float,default=0.);args=p.parse_args();main(args.seconds,args.dashboard_video,args.gui,args.mission_map,args.seed,args.target_shift,args.slam,args.survey_camera,args.three_gaps)
