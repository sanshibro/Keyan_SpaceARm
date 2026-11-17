"""Actuator-envelope bench test; fixed airframe, integrated MuJoCo arm dynamics."""
import sys,json,csv,time,math
from pathlib import Path
import numpy as np,mujoco
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'mujoco_closed_loop'))
from model import model_xml,SimArmBridge
from shape_limits import joint_bending_metrics

def main():
    out=ROOT/'revision/data/bending_dynamics';out.mkdir(exist_ok=True)
    rows=[];summaries=[]
    for phi_deg in [0,90,180,270]:
        model=mujoco.MjModel.from_xml_string(model_xml(fixed=True))
        data=mujoco.MjData(model);bridge=SimArmBridge();mujoco.mj_forward(model,data)
        jid=np.array([[model.jnt_qposadr[model.joint(name+str(i)).id] for name in ['bx','by']] for i in range(8)])
        aids=[model.actuator(name+str(i)).id for i in range(8) for name in ['bx','by']]
        peak=0.;violations=0;bounded=0;trace=[]
        for step in range(14000):
            t=float(data.time)
            u=np.clip((t-1)/6,0,1);u=10*u**3-15*u**4+6*u**5
            request=math.radians(120)*u
            data.ctrl[aids]=bridge.move_joints(request,math.radians(phi_deg))
            mujoco.mj_step(model,data)
            metrics=joint_bending_metrics(data.qpos[jid]);peak=max(peak,metrics['total_curvature_deg'])
            violations+=int(not metrics['within_hard_limit'])
            if step%10==0:
                row=[float(data.time),phi_deg,math.degrees(request),math.degrees(bridge.last_commanded_theta),metrics['net_bending_deg'],metrics['total_curvature_deg'],int(metrics['within_hard_limit'])]
                rows.append(row);trace.append(np.r_[data.time,data.qpos.copy(),data.qvel.copy(),data.ctrl.copy()])
        np.savez_compressed(out/('raw_phi_%03d.npz'%phi_deg),data=np.array(trace),nq=model.nq,nv=model.nv,nu=model.nu)
        summaries.append(dict(phi_deg=phi_deg,peak_total_curvature_deg=peak,hard_limit_violation_steps=violations,command_limit_events=bridge.command_limit_events))
    with (out/'bending.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['sim_time_s','phi_deg','requested_deg','issued_deg','net_bending_deg','total_curvature_deg','within_hard_limit']);w.writerows(rows)
    result=dict(test='fixed-airframe actuator dynamics qualification',integration='mj_step; no state overwrite after initialization',command_limit_deg=118,physical_limit_deg=120,planes=summaries,passed=all(x['hard_limit_violation_steps']==0 for x in summaries),scope='Actuator bench only. Does not establish free-flight, passage or perception performance.')
    (out/'manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
