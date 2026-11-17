"""Separate-process rendering/odometry; snapshots are only used to generate sensors."""
import traceback,time,queue,multiprocessing as mp
from pathlib import Path
import visual_odometry as vo
import numpy as np,mujoco
import cv2
from sensors import RGBDSensor,OPTICAL_FROM_GL
from visual_odometry import RGBDOdometry,VisualEstimate,detect_red_target,detect_supported_cylinder
from fiducials import FiducialLocalization
from estimation import relative_camera,orientation_shape,pcc_camera_pose,pcc_tool_transform

def T(R,p):
    out=np.eye(4);out[:3,:3]=R;out[:3,3]=p;return out

def offer_latest(channel,snapshot):
    """Keep the most recent pending observation; never queue stale physics."""
    try:channel.put_nowait(snapshot);return 0
    except queue.Full:
        try:channel.get_nowait()
        except queue.Empty:pass
        try:channel.put_nowait(snapshot)
        except queue.Full:pass
        return 1

def worker(xml,inputs,outputs,calibration,record_directory=None,preview_spec=None,preview_lock=None):
    shared_preview=None
    try:
        cv2.setNumThreads(1)
        if preview_spec is not None:
            from shared_preview import SharedPreview
            shared_preview=SharedPreview(preview_spec,preview_lock)
        m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m)
        nav=RGBDSensor(m,'navigation',480,360);wrist=RGBDSensor(m,'wrist_rgbd',480,360)
        external_lio=calibration.get('localization_mode')=='fast_lio_external'
        survey=RGBDSensor(m,'survey_rgbd',320,240) if external_lio and 'US' in calibration else None
        Localizer=FiducialLocalization if calibration.get('localization_mode')=='surveyed_fiducials' else RGBDOdometry
        no=None if external_lio else Localizer(calibration['initial_nav'])
        wo=Localizer(calibration['initial_wrist'])
        map_inputs=map_process=None;last_map=-10.
        if external_lio:
            from rgbd_mapper_worker import run as map_worker
            map_ctx=mp.get_context('spawn');map_inputs=map_ctx.Queue(maxsize=1)
            map_process=map_ctx.Process(target=map_worker,args=(map_inputs,record_directory,calibration))
            map_process.start()
        sample_path=Path(record_directory)/'sensors' if record_directory else None
        if sample_path:sample_path.mkdir(exist_ok=True)
        last_sample=-10.;recorded_reasons=set();last_preview=-10.
        outputs.put({'ready':True})
        while True:
            snapshot=inputs.get()
            if snapshot is None:break
            stamp,qpos,qvel,*extra=snapshot
            measured_bend=extra[0] if extra else None
            external_WU=extra[1] if len(extra)>1 else None
            d.time=stamp;d.qpos[:]=qpos;d.qvel[:]=qvel;mujoco.mj_forward(m,d)
            nf=nav.read(d);wf=wrist.read(d)
            for frame,sensor in [(nf,nav),(wf,wrist)]:
                # Sensor-only depth gate: nearby moving arm/target features excluded.
                frame.feature_mask=(np.isfinite(frame.depth_m)&(frame.depth_m>.6)).astype(np.uint8)*255
            drop=4<stamp<4.4
            if drop:wf.rgb[:]=0
            if external_lio:
                valid=external_WU is not None and np.isfinite(external_WU).all()
                pose=external_WU@calibration['UC'] if valid else calibration['initial_nav']
                a=VisualEstimate(stamp,pose,valid,0,0.,'fast_lio_external' if valid else 'fast_lio_pose_unavailable')
            else:a=no.update(nf)
            b=wo.update(wf);shape=None;WU=None;visual_wrist_shape_valid=False
            if a.valid:
                WU=a.T_WC@np.linalg.inv(calibration['UC'])
                if b.valid:
                    candidate=orientation_shape(relative_camera(WU,calibration['UB'],b.T_WC,a.stamp,b.stamp),calibration['TC'])
                    if measured_bend is None or np.linalg.norm(candidate-measured_bend)<.035:
                        shape=candidate;visual_wrist_shape_valid=True
            body_target=None
            if external_lio and a.valid:
                coarse=detect_red_target(nf)
                if coarse is not None:
                    body_target=(a.T_WC@np.r_[coarse['surface_center_C'],1.])[:3]
            target=None;tool_error_U=None;target_reason='camera_pose_invalid';target_pose_source='none'
            if WU is not None and (visual_wrist_shape_valid or measured_bend is not None):
                if visual_wrist_shape_valid:
                    WC,_=pcc_camera_pose(WU,calibration['UB'],b.T_WC,calibration['TC'])
                    target_pose_source='wrist_rgbd'
                else:
                    # Camera still supplies target pixels/depth. Segment
                    # strain supplies its mount pose while visual pose is lost.
                    WC=WU@calibration['UB']@pcc_tool_transform(measured_bend)@calibration['TC']
                    target_pose_source='segment_strain_fallback'
                detected=detect_supported_cylinder(wf,WC)
                target_reason=vo.LAST_TARGET_DIAGNOSTIC.get('reason','unknown')
                if detected is not None:
                    target=detected['center_W']
                    object_C=np.linalg.solve(WC,np.r_[target,1.])
                    object_T=calibration['TC']@object_C
                    error_T=object_T[:3]-np.array([0,0,.128])
                    rotation_U_T=WU[:3,:3].T@WC[:3,:3]@calibration['TC'][:3,:3].T
                    tool_error_U=rotation_U_T@error_T
            if external_lio and WU is not None and measured_bend is not None and stamp-last_map>=.5:
                sf=survey.read(d) if survey is not None else None
                offer_latest(map_inputs,(stamp,nf,wf,sf,WU.copy(),np.asarray(measured_bend).copy()))
                last_map=stamp
            high_bend=shape is not None and np.linalg.norm(shape)>np.deg2rad(4)
            if sample_path and (stamp-last_sample>=2 or target_reason not in recorded_reasons or high_bend):
                np.savez_compressed(sample_path/('frame_%09.3f.npz'%stamp),stamp=stamp,nav_rgb=nf.rgb,nav_depth_m=nf.depth_m,nav_no_return=nf.no_return_mask,nav_K=nf.K,wrist_rgb=wf.rgb,wrist_depth_m=wf.depth_m,wrist_no_return=wf.no_return_mask,K=wf.K,nav_pose=a.T_WC,wrist_pose=b.T_WC,target_reason=target_reason)
                last_sample=stamp;recorded_reasons.add(target_reason)
            if shared_preview is not None and stamp-last_preview>=.2:
                shared_preview.write(stamp,nf,wf,a.valid,b.valid,a.reprojection_px,b.reprojection_px);last_preview=stamp
            outputs.put(dict(target_reason=target_reason,target_pose_source=target_pose_source,target_body_coarse=body_target,tool_error_U=tool_error_U,target=target,stamp=stamp,nav_valid=a.valid,wrist_valid=b.valid,WU=WU,shape=shape,dropout=drop,nav_reason=a.reason,wrist_reason=b.reason,nav_inliers=a.inliers,wrist_inliers=b.inliers,nav_reprojection_px=a.reprojection_px,wrist_reprojection_px=b.reprojection_px,nav_depth_fit_rms_m=getattr(a,'depth_fit_rms_m',None),wrist_depth_fit_rms_m=getattr(b,'depth_fit_rms_m',None)))
        nav.close();wrist.close()
        if survey is not None:survey.close()
        if map_process is not None:
            offer_latest(map_inputs,None)
            map_process.join(timeout=5)
            if map_process.is_alive():map_process.terminate();map_process.join()
            map_inputs.close();map_inputs.join_thread();map_process.close()
    except Exception:outputs.put({'error':traceback.format_exc()})
    finally:
        if shared_preview is not None:shared_preview.close()
