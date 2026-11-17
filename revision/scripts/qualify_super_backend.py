"""Render geometry and integrate the registered SUPER development backend."""
from pathlib import Path
import sys,os,json,math,time,csv,argparse
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np
import mujoco,cv2
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'mujoco_closed_loop'))
from super_metric_model import build_super_metric
from model import SimArmBridge
from arm_command import ArmCommand,MujocoArmBackend

def renderer(model):
    r=mujoco.Renderer(model,height=850,width=900)
    cam=mujoco.MjvCamera();cam.lookat[:]=[.06,0,1.30];cam.distance=1.08;cam.azimuth=135;cam.elevation=-18
    if hasattr(cam,'orthographic'):cam.orthographic=True
    return r,cam

def picture(r,cam,data,overlay=False):
    opt=mujoco.MjvOption();opt.geomgroup[0]=0;opt.geomgroup[3]=int(overlay);opt.sitegroup[:]=0
    r.update_scene(data,camera=cam,scene_option=opt);return r.render().copy()

def save(p,image):cv2.imwrite(str(p),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))

def main(mode):
    out=ROOT/'revision/data/super_backend'/mode
    if (out/'manifest.json').exists():
        archived=out.with_name(mode+'_previous_'+time.strftime('%Y%m%d_%H%M%S'))
        if archived.exists():raise FileExistsError(str(archived))
        out.rename(archived)
    out.mkdir(parents=True,exist_ok=True)
    live=ROOT/'revision/live';xml,info=build_super_metric(fixed=mode!='hover')
    (out/'model.xml').write_text(xml);(out/'platform.json').write_text(json.dumps(info,indent=2)+'\n')
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    if mode=='geometry':
        SimArmBridge().set_kinematic_shape(m,d,math.radians(90),math.pi);mujoco.mj_forward(m,d)
        r,cam=renderer(m)
        for name,az,el in [('perspective',135,-18),('side',90,-3),('top',90,-89.9)]:
            cam.azimuth=az;cam.elevation=el
            for overlay in (False,True):
                save(out/('%s_%s.png'%(name,'collision' if overlay else 'cad')),picture(r,cam,d,overlay))
        r.close();print('geometry ready',out,flush=True);return
    backend=MujocoArmBackend(m,d);controller=None
    from platform_self_collision import SelfCollisionOracle
    precheck=SelfCollisionOracle().sequence([[0.,math.pi],[math.radians(118),math.pi],[0.,math.pi]])
    if not precheck['accepted']:raise RuntimeError('Motion rejected by independent self-collision precheck: '+str(precheck))
    if mode=='hover':
        from flight_feedback import RotorController
        from types import SimpleNamespace
        controller=RotorController.from_model(m)
    r,cam=renderer(m);cam.azimuth=120
    video=cv2.VideoWriter(str(out/'recorded_dynamics.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(900,850))
    rows=[];raw=[];events=[];wall=time.monotonic();u=m.body('uav').id
    rotor_ids=[m.actuator('thrust%d'%i).id for i in range(4)] if mode=='hover' else []
    saturation=0;max_position=0.;max_tilt=0.;peak=0.;violations=0;contact_steps=0
    def smooth(x):return .5-.5*math.cos(math.pi*np.clip(x,0,1))
    nsteps=14000
    for step in range(nsteps):
        t=d.time
        if t<5:theta=90*smooth((t-1)/4)
        elif t<7:theta=90+28*smooth((t-5)/2)
        elif t<9:theta=118
        else:theta=118*(1-smooth((t-9)/4))
        requested=ArmCommand(step,float(t),math.radians(theta),math.pi)
        backend.apply(requested)
        if controller is not None:
            state=SimpleNamespace(p=d.xpos[u].copy(),v=d.qvel[:3].copy(),R=d.xmat[u].reshape(3,3).copy())
            com=state.R.T@(d.subtree_com[u]-state.p)
            force,sat=controller.command(state,d.qvel[3:6],[0,0,1.5],com)
            d.ctrl[rotor_ids]=force;saturation+=int(sat)
        mujoco.mj_step(m,d)
        obs=backend.observe();peak=max(peak,obs['total_curvature_deg']);violations+=int(not obs['within_hard_limit'])
        max_position=max(max_position,float(np.linalg.norm(d.xpos[u]-[0,0,1.5])))
        max_tilt=max(max_tilt,float(math.degrees(math.acos(np.clip(d.xmat[u].reshape(3,3)[2,2],-1,1)))))
        contact_steps+=int(d.ncon>0)
        if d.ncon and len(events)<20:
            for c in d.contact[:d.ncon]:events.append(dict(t=float(d.time),a=m.geom(c.geom1).name,b=m.geom(c.geom2).name,distance=float(c.dist)))
        if step%10==0:
            rows.append([d.time,theta,obs['net_bending_deg'],obs['total_curvature_deg'],*d.xpos[u],d.ncon])
            raw.append(np.r_[d.time,d.qpos.copy(),d.qvel.copy(),d.ctrl.copy()])
        if step%100==0:
            image=picture(r,cam,d);frame=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            cv2.putText(frame,'MuJoCo integrated state | %s | t=%.1f s'%(mode,d.time),(22,35),cv2.FONT_HERSHEY_SIMPLEX,.62,(48,56,65),1,cv2.LINE_AA)
            cv2.putText(frame,'command %.1f deg | total curvature %.1f deg'%(theta,obs['total_curvature_deg']),(22,66),cv2.FONT_HERSHEY_SIMPLEX,.54,(48,56,65),1,cv2.LINE_AA)
            video.write(frame)
            cv2.imwrite(str(live/'frame.png'),frame)
            if step in (5000,8000):cv2.imwrite(str(out/('frame_%05d.png'%step)),frame)
        if step%2000==0:print(mode,'t',round(d.time,2),'bend',round(obs['total_curvature_deg'],2),'contacts',d.ncon,flush=True)
        if not np.isfinite(d.qpos).all() or max_position>1.0:
            events.append(dict(t=float(d.time),reason='integration_guard'));break
    video.release();r.close()
    with (out/'states.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['time_s','command_theta_deg','net_theta_deg','total_curvature_deg','uav_x','uav_y','uav_z','contacts']);w.writerows(rows)
    np.savez_compressed(out/'raw_dynamics.npz',data=np.array(raw),nq=m.nq,nv=m.nv,nu=m.nu)
    report=dict(platform_id=info['platform_id'],mode=mode,sim_seconds=float(d.time),wall_seconds=time.monotonic()-wall,
        physics='mj_step; qpos initialized once, no pose replay during integration',
        mass_kg=float(m.body_mass.sum()),mass_calibrated=False,command_cap_deg=118.,
        nominal_motion_precheck=precheck,
        peak_total_curvature_deg=peak,above_120_steps=violations,contact_steps=contact_steps,
        saturated_steps=saturation,max_position_error_m=max_position,max_tilt_deg=max_tilt,
        accepted_for_task_evaluation=False,events=events[:20])
    report['component_check_passed']=violations==0 and contact_steps==0 and (mode!='hover' or (max_position<.1 and max_tilt<15))
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['geometry','fixed','hover'],default='geometry');main(p.parse_args().mode)
