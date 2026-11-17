"""Fixed-base physical scan; explicitly truth-based development evaluation."""
import csv,hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
import mujoco
from model import model_xml,SimArmBridge
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'xtdrone_catkin_ws/src/uav_continuum_whole_body_v2/src'))
from uav_continuum_whole_body_v2.geometry import local_arm_sections
from uav_continuum_whole_body_v2.types import ArmModel

def main():
    out=ROOT/'mujoco_closed_loop/runs'/time.strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True,exist_ok=False)
    xml=model_xml();(out/'model.xml').write_text(xml)
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);bridge=SimArmBridge()
    rows=[];t0=time.perf_counter()
    tool=m.body('tool').id
    for theta_deg in [0,30,60,90]:
        for phi_deg in [0,90,180,270]:
            theta,phi=np.deg2rad([theta_deg,phi_deg]);d.ctrl[:16]=bridge.move_joints(theta,phi)
            expected=local_arm_sections(theta,phi,ArmModel(arc_length=.28,base_offset_z=.0704))[1][-1]
            errors=[]
            for step in range(3000):
                mujoco.mj_step(m,d)
                if step>=2500:
                    p=d.xmat[m.body('mount').id].reshape(3,3).T@(d.xpos[tool]-d.xpos[m.body('mount').id])
                    errors.append(np.linalg.norm(p-expected))
            rows.append(dict(theta_deg=theta_deg,phi_deg=phi_deg,time=d.time,endpoint_error_m=float(np.mean(errors)),max_abs_joint=float(np.max(np.abs(d.qpos))),contacts=d.ncon))
    with (out/'scan.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    passed=all(r['endpoint_error_m']<=.02 for r in rows)
    result=dict(stage='fixed_base_endpoint_scan_only',passed=passed,full_task_accepted=False,feedback='truth_development',mujoco=mujoco.__version__,mass=float(m.body_mass.sum()),wall_seconds=time.perf_counter()-t0,sim_seconds=d.time,model_sha256=hashlib.sha256(xml.encode()).hexdigest(),max_endpoint_error_m=max(r['endpoint_error_m'] for r in rows))
    (out/'manifest.json').write_text(json.dumps(result,indent=2));print(out);print(json.dumps(result))
if __name__=='__main__':main()
