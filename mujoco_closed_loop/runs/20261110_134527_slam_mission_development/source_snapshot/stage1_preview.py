"""Orbitable primary 3D viewer for the stage-1 assembly and scene."""
import argparse,time
from pathlib import Path
import cv2,mujoco,numpy as np
from stage1_scene import build_scene
from model import SimArmBridge

def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',action='store_true');args=p.parse_args()
    out=Path(__file__).resolve().parent/'runs'/(time.strftime('%Y%m%d_%H%M%S')+'_stage1_review');out.mkdir(parents=True)
    xml=build_scene();(out/'scene.xml').write_text(xml)
    model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:]=[-.2,0,1.15];camera.azimuth=132;camera.elevation=-48;camera.distance=8.5
    dragging=[False,None];demo=False;bridge=SimArmBridge();demo_time=0.;gate_pose=False
    uav_id=model.body('uav').id;start_pos=model.body_pos[uav_id].copy()
    def reset_robot():
        model.body_pos[uav_id]=start_pos
        data.qpos[:]=0;data.qvel[:]=0;data.ctrl[:]=0
        mujoco.mj_forward(model,data)
    def show_bent_gate():
        model.body_pos[uav_id]=[.39,-.05,1.34]
        bridge.set_kinematic_shape(model,data,np.pi/2,np.pi)
        mujoco.mj_forward(model,data)
        camera.lookat[:]=[.39,-.05,1.24]
        camera.azimuth=90;camera.elevation=-8;camera.distance=1.55
    def mouse(event,x,y,flags,param):
        if event==cv2.EVENT_LBUTTONDOWN:dragging[:]=[True,(x,y)]
        elif event==cv2.EVENT_LBUTTONUP:dragging[0]=False
        elif event==cv2.EVENT_MOUSEMOVE and dragging[0]:
            dx=x-dragging[1][0];dy=y-dragging[1][1];camera.azimuth+=dx*.32;camera.elevation=np.clip(camera.elevation-dy*.25,-85,85);dragging[1]=(x,y)
        elif event==cv2.EVENT_MOUSEWHEEL:camera.distance=np.clip(camera.distance*(.88 if flags>0 else 1.12),.7,14)
    name='SUPER + Qingyu | Stage 1 3D scene'
    if not args.snapshot:cv2.namedWindow(name,cv2.WINDOW_NORMAL);cv2.resizeWindow(name,1440,810);cv2.setMouseCallback(name,mouse)
    with mujoco.Renderer(model,height=810,width=1440) as renderer:
        # Save inspectable robot and task views in addition to the overview.
        original=(camera.lookat.copy(),camera.azimuth,camera.elevation,camera.distance)
        for filename,lookat,azimuth,elevation,distance in [
            ('robot_detail.jpg',[-.64,-2.57,1.11],135,-12,1.05),
            ('body_camera_front.jpg',[-.64,-2.48,1.31],270,-10,.70),
            ('camera_alignment_front.jpg',[-.64,-2.48,1.10],270,-7,1.05),
            ('task_area.jpg',[-.2,0,1.2],132,-55,7.3)]:
            camera.lookat[:]=lookat;camera.azimuth=azimuth;camera.elevation=elevation;camera.distance=distance
            renderer.update_scene(data,camera=camera)
            cv2.imwrite(str(out/filename),cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR))
        for camera_name in ('navigation','wrist_rgbd'):
            renderer.update_scene(data,camera=camera_name)
            cv2.imwrite(str(out/(camera_name+'_view.jpg')),
                        cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR))
        show_bent_gate()
        renderer.update_scene(data,camera=camera)
        cv2.imwrite(str(out/'bent_gap_2.jpg'),
                    cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR))
        reset_robot()
        camera.lookat[:]=original[0];camera.azimuth=original[1];camera.elevation=original[2];camera.distance=original[3]
        first=True
        while True:
            if demo:
                theta=np.deg2rad(90)*.5*(1-np.cos(2*np.pi*demo_time/7))
                command=bridge.move_joints(float(theta),0.)
                for i in range(8):
                    for j,name in enumerate(('bx','by')):
                        data.ctrl[model.actuator(name+str(i)).id]=command[2*i+j]
                for _ in range(20):mujoco.mj_step(model,data)
                demo_time+=.02
            renderer.update_scene(data,camera=camera);rgb=renderer.render();bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            cv2.rectangle(bgr,(0,0),(1440,52),(17,27,32),-1)
            cv2.putText(bgr,'SUPER + QINGYU  /  SCENE REVIEW',(24,35),cv2.FONT_HERSHEY_SIMPLEX,.75,(245,242,230),2,cv2.LINE_AA)
            cv2.putText(bgr,'START  >  3 TIGHT GAPS  >  OBSTACLES  >  TARGET',(778,34),cv2.FONT_HERSHEY_SIMPLEX,.55,(170,221,226),1,cv2.LINE_AA)
            cv2.putText(bgr,'Drag: orbit   Wheel: zoom   1/2/3: views   4: bent at gap   R: reset   SPACE: arm demo   Q: close',(22,787),cv2.FONT_HERSHEY_SIMPLEX,.51,(235,240,240),1,cv2.LINE_AA)
            if first:cv2.imwrite(str(out/'overview.jpg'),bgr);first=False
            if args.snapshot:break
            cv2.imshow(name,bgr);k=cv2.waitKey(30)&255
            if k in (ord('q'),27):break
            if k==32:
                if gate_pose:reset_robot();gate_pose=False
                demo=not demo
            if k==ord('4'):
                demo=False;data.ctrl[:]=0;show_bent_gate();gate_pose=True
            if k==ord('r'):
                demo=False;reset_robot();gate_pose=False
                camera.lookat[:]=[-.64,-2.57,1.11];camera.azimuth=135;camera.elevation=-12;camera.distance=1.05
            if k==ord('1'):camera.lookat[:]=[-.2,0,1.15];camera.distance=8.5;camera.azimuth=132;camera.elevation=-48
            if k==ord('2'):camera.lookat[:]=[-.64,-2.57,1.11];camera.distance=1.05;camera.azimuth=135;camera.elevation=-12
            if k==ord('3'):camera.lookat[:]=[-.2,0,1.2];camera.distance=7.3;camera.azimuth=132;camera.elevation=-55
    if not args.snapshot:cv2.destroyAllWindows()
    print(out)
if __name__=='__main__':main()
