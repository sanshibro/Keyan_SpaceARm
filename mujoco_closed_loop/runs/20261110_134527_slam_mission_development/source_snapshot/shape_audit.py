"""Independent all-node PCC fit, rigid-camera check and finger-stroke audit."""
import json,time,csv,sys
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares
from model import model_xml,SimArmBridge
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'xtdrone_catkin_ws/src/uav_continuum_whole_body_v2/src'))
from uav_continuum_whole_body_v2.geometry import local_arm_sections
from uav_continuum_whole_body_v2.types import ArmModel

def run():
    out=ROOT/'runs'/(time.strftime('%Y%m%d_%H%M%S')+'_shape_audit');out.mkdir(parents=True)
    xml=model_xml();(out/'model.xml').write_text(xml);m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m)
    bridge=SimArmBridge();model=ArmModel(arc_length=.28,base_offset_z=.0704,segment_count=16)
    camera_local=np.zeros(9);mujoco.mju_quat2Mat(camera_local,m.cam_quat[m.camera('wrist_rgbd').id])
    mount=m.body('mount').id;tool=m.body('tool').id;cam=m.camera('wrist_rgbd').id
    nodes=[m.body('seg%d'%i).id for i in range(8)]+[tool]
    rows=[];raw=[];max_extrinsic=0.;start=time.perf_counter()
    for theta_deg in [0,30,60,90,120]:
        for phi_deg in [0,45,90,135,180,225,270,315]:
            theta,phi=np.deg2rad([theta_deg,phi_deg]);d.ctrl[:16]=bridge.move_joints(theta,phi)
            for step in range(3000):
                mujoco.mj_step(m,d)
                if step%20==0:raw.append([d.time,theta,phi,*d.qpos.copy(),*d.qvel.copy(),*d.ctrl.copy(),d.ncon])
            R=d.xmat[mount].reshape(3,3);pts=(d.xpos[nodes]-d.xpos[mount])@R
            def predicted(b):
                angle=np.linalg.norm(b);plane=np.arctan2(b[1],b[0]);return local_arm_sections(angle,plane,model)[1][[1,3,5,7,9,11,13,15,16]]
            fit=least_squares(lambda b:(predicted(b)-pts).ravel(),[theta*np.cos(phi)+1e-7,theta*np.sin(phi)],max_nfev=100)
            expected=predicted([theta*np.cos(phi),theta*np.sin(phi)])
            shape_error=np.linalg.norm(pts-expected,axis=1)
            bend_error=np.linalg.norm(fit.x-theta*np.array([np.cos(phi),np.sin(phi)]))
            cp=d.xmat[tool].reshape(3,3).T@(d.cam_xpos[cam]-d.xpos[tool])
            cr=d.xmat[tool].reshape(3,3).T@d.cam_xmat[cam].reshape(3,3)
            max_extrinsic=max(max_extrinsic,np.linalg.norm(cp-m.cam_pos[cam]),np.linalg.norm(cr-camera_local.reshape(3,3)))
            rows.append(dict(theta_deg=theta_deg,phi_deg=phi_deg,max_centerline_error_m=float(shape_error.max()),fitted_bend_error_deg=float(np.rad2deg(bend_error)),fit_residual_m=float(np.linalg.norm(fit.fun.reshape(-1,3),axis=1).max()),contacts=d.ncon))
    with (out/'shape.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    np.savez_compressed(out/'raw_states.npz',data=np.array(raw),columns=np.array(['sim_time','command_theta','command_phi']+['qpos%d'%i for i in range(m.nq)]+['qvel%d'%i for i in range(m.nv)]+['ctrl%d'%i for i in range(m.nu)]+['contacts']))
    strokes=[]
    for command in [0,.032,0]:
        d.ctrl[16:19]=min(command,.024)
        for _ in range(1000):mujoco.mj_step(m,d)
        strokes.append(d.qpos[-3:].copy().tolist())
    result=dict(stage='fixed_shape_and_extrinsic_audit',full_task_accepted=False,feedback='truth_development',cases=len(rows),max_centerline_error_m=max(r['max_centerline_error_m'] for r in rows),max_bend_error_deg=max(r['fitted_bend_error_deg'] for r in rows),max_camera_extrinsic_error=float(max_extrinsic),gripper_command_positions=strokes,wall_seconds=time.perf_counter()-start)
    result['passed']=bool(result['max_centerline_error_m']<=.02 and result['max_bend_error_deg']<=5 and max_extrinsic<=1e-6)
    (out/'manifest.json').write_text(json.dumps(result,indent=2));print(out);print(json.dumps(result))
if __name__=='__main__':run()
