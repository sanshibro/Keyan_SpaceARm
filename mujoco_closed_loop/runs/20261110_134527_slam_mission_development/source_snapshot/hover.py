"""Rotor-driven free-flight development test. Explicit truth feedback, no PX4 claim."""
import csv,json,time,hashlib,argparse
from pathlib import Path
import mujoco
import numpy as np
from model import model_xml,SimArmBridge
ROOT=Path(__file__).resolve().parent

def vee(M):return np.array([M[2,1],M[0,2],M[1,0]])

def run(mode):
    out=ROOT/'runs'/('%s_hover_%s'%(time.strftime('%Y%m%d_%H%M%S'),mode));out.mkdir(parents=True)
    xml=model_xml(fixed=False);(out/'model.xml').write_text(xml)
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    bridge=SimArmBridge();mass=m.body_mass.sum();u=m.body('uav').id
    rotor_ids=[m.actuator('thrust%d'%i).id for i in range(4)]
    positions=np.array([[.2,.2],[-.2,.2],[-.2,-.2],[.2,-.2]])
    mix=np.array([np.ones(4),positions[:,1],-positions[:,0],[.02,-.02,.02,-.02]])
    inv=np.linalg.inv(mix);target=np.array([0.,0.,1.5]);integral=np.zeros(3);rows=[]
    wall=time.perf_counter();max_tilt=0.;max_pos=0.;contacts=0;saturated=0;err2=0.;count=0
    for step in range(30000):
        t=d.time
        # Smooth shape switching, same mass/controller across modes.
        shape=.5*(1-np.cos(np.pi*np.clip((t-5)/5,0,1)))
        theta=np.deg2rad(60)*shape if mode=='moving' else 0.
        phi=np.deg2rad(90)*.5*(1-np.cos(np.pi*np.clip((t-15)/5,0,1)))
        d.ctrl[:16]=bridge.move_joints(theta,phi)
        R=d.xmat[u].reshape(3,3);position=d.xpos[u];velocity=d.qvel[:3];omega=d.qvel[3:6]
        error=target-position;integral=np.clip(integral+error*.001,-.2,.2)
        acc=2*error-2.2*velocity+.15*integral+np.array([0,0,9.81])
        z=acc/np.linalg.norm(acc);y=np.cross(z,[1.,0,0]);y/=np.linalg.norm(y);x=np.cross(y,z);Rd=np.column_stack([x,y,z])
        er=.5*vee(Rd.T@R-R.T@Rd)
        # Counter static gravity moment of the coupled arm; development truth COM.
        r=R.T@(d.subtree_com[u]-position)
        gravity_moment=np.cross(r,R.T@np.array([0,0,-mass*9.81]))
        torque=-8.*er-1.6*omega-gravity_moment
        wrench=np.r_[max(0.,mass*np.dot(acc,R[:,2])),torque]
        raw=inv@wrench;forces=np.clip(raw,0,12);d.ctrl[rotor_ids]=forces
        mujoco.mj_step(m,d)
        tilt=np.arccos(np.clip(R[2,2],-1,1));e=np.linalg.norm(error)
        max_tilt=max(max_tilt,tilt);max_pos=max(max_pos,e);err2+=e*e;count+=1
        contacts+=d.ncon;saturated+=int(np.any(abs(raw-forces)>1e-8))
        if step%20==0:
            rows.append([d.time,*d.xpos[u],*error,*d.qpos[7:23],theta,phi,*forces,tilt,d.ncon])
        if not np.isfinite(d.qpos).all():raise RuntimeError('Nonfinite dynamics')
    with (out/'states.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['sim_time','x','y','z','ex','ey','ez']+['joint%d'%i for i in range(16)]+['command_theta','command_phi']+['rotor%d_N'%i for i in range(4)]+['tilt_rad','contacts']);w.writerows(rows)
    result=dict(stage='free_hover_truth_development',mode=mode,full_task_accepted=False,feedback='mujoco_truth_development',controller='custom_geometric_rotor_controller_not_PX4',mass_kg=float(mass),sim_seconds=d.time,wall_seconds=time.perf_counter()-wall,position_rms_m=float(np.sqrt(err2/count)),position_max_m=float(max_pos),max_tilt_deg=float(np.rad2deg(max_tilt)),contact_steps=contacts,saturated_steps=saturated,model_sha256=hashlib.sha256(xml.encode()).hexdigest())
    result['passed']=bool(result['position_rms_m']<=.10 and result['max_tilt_deg']<=15 and contacts==0)
    (out/'manifest.json').write_text(json.dumps(result,indent=2));print(out);print(json.dumps(result));return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['fixed','moving','both'],default='both');args=p.parse_args()
    for mode in (['fixed','moving'] if args.mode=='both' else [args.mode]):run(mode)
