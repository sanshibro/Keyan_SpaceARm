"""Fixed-base physical model review, not navigation/grasp acceptance."""
import json,time
from pathlib import Path
import cv2,numpy as np,mujoco
from model import model_xml,SimArmBridge
from evidence import freeze_run

def main():
    out=Path(__file__).resolve().parent/'runs'/(time.strftime('%Y%m%d_%H%M%S')+'_qingyu_model_review');out.mkdir();freeze_run(out)
    xml=model_xml(camera_profile='grasp_visible');(out/'model.xml').write_text(xml)
    model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model);bridge=SimArmBridge();renderer=mujoco.Renderer(model,height=360,width=400);tiles=[];rows=[]
    for phi in [0,45]:
        for theta in [0,60,120]:
            data.ctrl[:16]=bridge.move_joints(np.deg2rad(theta),np.deg2rad(phi))
            for _ in range(3500):mujoco.mj_step(model,data)
            camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE;camera.lookat[:]=[0,0,1.2];camera.azimuth=110;camera.elevation=-12;camera.distance=1.05
            renderer.update_scene(data,camera=camera);tile=cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR)
            cv2.rectangle(tile,(0,0),(400,30),(20,20,20),-1)
            cv2.putText(tile,f'Command theta={theta} deg, phi={phi} deg',(9,21),cv2.FONT_HERSHEY_SIMPLEX,.5,(240,240,240),1,cv2.LINE_AA)
            tiles.append(tile);rows.append(dict(theta_deg=theta,phi_deg=phi,sim_time=data.time,contacts=data.ncon))
    renderer.close();canvas=np.vstack([np.hstack(tiles[:3]),np.hstack(tiles[3:])]);cv2.imwrite(str(out/'qingyu_model_review.jpg'),canvas)
    (out/'manifest.json').write_text(json.dumps(dict(stage='fixed_base_model_review_only',full_task_accepted=False,source='Qingyu SDK CAD/config with numerical flexible-body dynamics',mass_kg=float(model.body_mass.sum()),cases=rows),indent=2));print(out)

if __name__=='__main__':main()
