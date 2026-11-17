"""Independent online observer with the exact RGB-D previews consumed by control."""
import json,time,queue,traceback,multiprocessing as mp
from pathlib import Path
import numpy as np,cv2,mujoco

W,H=400,300

def _label(tile,title):
    tile=tile.copy();cv2.rectangle(tile,(0,0),(W,26),(15,15,15),-1)
    cv2.putText(tile,title,(9,18),cv2.FONT_HERSHEY_SIMPLEX,.44,(245,245,245),1,cv2.LINE_AA)
    return tile

def _run(xml,directory,inputs,show,map_path,preview_spec,preview_lock,slam):
    directory=Path(directory);writer=hero_writer=None;renderer=hero_renderer=None;frames=0;first=None;last=None
    shared_preview=None
    try:
        from shared_preview import SharedPreview
        shared_preview=SharedPreview(preview_spec,preview_lock)
        model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model);renderer=mujoco.Renderer(model,height=H,width=W)
        hero_renderer=mujoco.Renderer(model,height=480,width=640)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
        hero_renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
        writer=cv2.VideoWriter(str(directory/'online_dashboard.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),5,(W*4,H*2))
        hero_writer=cv2.VideoWriter(str(directory/'mission_view.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),5,(1280,720))
        if not writer.isOpened() or not hero_writer.isOpened():raise RuntimeError('Cannot open observer video writer')
        map_background=np.full((H,W,3),24,np.uint8)
        def pixel(p):return tuple(np.rint([W/2+75*p[0],H/2-75*p[1]]).astype(int))
        current_map=map_path
        if current_map:
            from planner_adapter import ROOT
            from uav_continuum_whole_body_v2.mapping import MapSnapshot
            snapshot=MapSnapshot.load(current_map)
            for points,color in [(snapshot.observed_points,(65,65,65)),(snapshot.occupied_points,(210,210,210))]:
                for p in points[(points[:,2]>1.4)&(points[:,2]<1.6)][::3]:
                    x,y=pixel(p)
                    if 0<=x<W and 0<=y<H:map_background[y,x]=color
        preview=None;track=[];timestamps=[];side_view_cache={}
        while True:
            item=inputs.get()
            if item is None:break
            if slam:
                maps=sorted(directory.glob('entry_online_map_*.npz'))
                newest=maps[-1] if maps else None
                if newest is not None and str(newest)!=current_map:
                    try:
                        from uav_continuum_whole_body_v2.mapping import MapSnapshot
                        snapshot=MapSnapshot.load(newest)
                        tile=np.full((H,W,3),24,np.uint8)
                        for points,color in [(snapshot.observed_points,(65,65,65)),(snapshot.occupied_points,(210,210,210))]:
                            for point in points[(points[:,2]>1.4)&(points[:,2]<1.6)][::3]:
                                x,y=pixel(point)
                                if 0<=x<W and 0<=y<H:tile[y,x]=color
                        map_background=tile;current_map=str(newest)
                    except (OSError,ValueError,EOFError):pass
            stamp=item['stamp'];first=stamp if first is None else first;last=stamp
            data.time=stamp;data.qpos[:]=item['qpos'];mujoco.mj_forward(model,data)
            p=np.asarray(item['estimated_position'])
            fresh_preview=shared_preview.read(min(stamp,item['consumed_sensor_stamp']))
            if fresh_preview is not None:preview=fresh_preview
            views=[];hero_main=None
            for title,azimuth,elevation,distance in [('OBSERVER: perspective',125,-25,1.7),('OBSERVER: side',90,-8,2.0),('OBSERVER: top',0,-89,2.2)]:
                cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE
                cam.lookat[:]=p+np.array([0.,.20,-.12]);cam.azimuth=azimuth;cam.elevation=elevation;cam.distance=distance
                if hero_main is None:
                    hero_renderer.update_scene(data,camera=cam)
                    hero_main=cv2.cvtColor(hero_renderer.render().copy(),cv2.COLOR_RGB2BGR)
                    rgb=cv2.resize(hero_main,(W,H),interpolation=cv2.INTER_AREA)
                    views.append(_label(cv2.convertScaleAbs(rgb,alpha=1.25,beta=12),title))
                    continue
                if frames%5 and title in side_view_cache:
                    views.append(side_view_cache[title]);continue
                renderer.update_scene(data,camera=cam);rgb=renderer.render().copy()
                side_view_cache[title]=_label(cv2.convertScaleAbs(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),alpha=1.25,beta=12),title)
                views.append(side_view_cache[title])
            track.append(pixel(p));map_tile=map_background.copy()
            if len(track)>1:cv2.polylines(map_tile,[np.array(track,dtype=np.int32)],False,(0,220,255),2)
            cv2.circle(map_tile,pixel(p),4,(0,255,100),-1)
            views.append(_label(map_tile,('ONLINE SLAM MAP' if slam else 'PRESURVEY MAP')+' z=1.5m + estimate' if current_map else 'Estimate XY (mapping in progress)' if slam else 'Estimate XY (map not connected)'))
            if preview is not None:
                for key,title in [('nav_rgb','USED SENSOR: navigation RGB'),('wrist_rgb','USED SENSOR: wrist RGB')]:
                    bgr=cv2.cvtColor(preview[key],cv2.COLOR_RGB2BGR)
                    views.append(_label(cv2.resize(bgr,(W,H)),title))
                depth=preview['wrist_depth_m'];valid=np.isfinite(depth)
                scaled=np.uint8(np.clip(np.nan_to_num(depth,nan=0)/2,0,1)*255)
                colored=cv2.applyColorMap(scaled,cv2.COLORMAP_JET);colored[~valid]=0
                views.append(_label(cv2.resize(colored,(W,H)),'USED SENSOR: wrist depth 0..2 m'))
            else:
                for title in ['Navigation RGB: waiting','Wrist RGB: waiting','Wrist depth: waiting']:views.append(_label(np.zeros((H,W,3),np.uint8),title))
            status=np.full((H,W,3),18,np.uint8)
            text=[f"Physics time: {stamp:.3f} s",'Phase: '+item['phase'],f"Estimate: {p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f} m",'Feedback: FAST-LIO + IMU' if slam else 'Feedback: surveyed ArUco + IMU',f"Sensor stamp: {preview['stamp']:.3f} s" if preview else 'Sensor stamp: unavailable',f"Sensor age: {stamp-preview['stamp']:.3f} s" if preview else '', 'Observer views use physics snapshots.', 'RGB-D panels are control-consumed frames.', 'Full navigation/grasp acceptance: pending']
            for i,line in enumerate(text):cv2.putText(status,line,(12,46+i*25),cv2.FONT_HERSHEY_SIMPLEX,.43,(220,220,220),1,cv2.LINE_AA)
            views.append(_label(status,'ONLINE STATUS'))
            canvas=np.vstack([np.hstack(views[:4]),np.hstack(views[4:])]);writer.write(canvas);frames+=1
            hero=cv2.resize(cv2.convertScaleAbs(hero_main,alpha=1.18,beta=10),(960,720),interpolation=cv2.INTER_CUBIC)
            cv2.rectangle(hero,(0,0),(960,38),(16,21,24),-1)
            cv2.putText(hero,f"SUPER + Qingyu | {item['phase']} | t={stamp:.1f}s",(18,26),cv2.FONT_HERSHEY_SIMPLEX,.62,(235,245,245),2,cv2.LINE_AA)
            sidebar=np.vstack([cv2.resize(views[4],(320,240),interpolation=cv2.INTER_AREA),
                               cv2.resize(views[5],(320,240),interpolation=cv2.INTER_AREA),
                               cv2.resize(views[3],(320,240),interpolation=cv2.INTER_AREA)])
            hero_canvas=np.hstack([hero,sidebar]);hero_writer.write(hero_canvas)
            timestamps.append(dict(video_frame=frames-1,physics_stamp=stamp,sensor_stamp=preview['stamp'] if preview else None))
            if frames==1:cv2.imwrite(str(directory/'online_dashboard_preview.jpg'),canvas)
            if preview is not None and not (directory/'mission_view_preview.jpg').exists():
                cv2.imwrite(str(directory/'mission_view_preview.jpg'),hero_canvas)
            if show:
                cv2.imshow('UAV + Qingyu live observer',hero_canvas);cv2.waitKey(1)
        (directory/'dashboard_manifest.json').write_text(json.dumps(dict(mode='online_observer_and_actual_control_consumed_sensor_previews',frames=frames,first_sim_stamp=first,last_sim_stamp=last,video_fps=5,mission_view='mission_view.mp4',observer_truth_is_display_only=True,display_enabled=show),indent=2))
        (directory/'dashboard_timestamps.json').write_text(json.dumps(timestamps))
    except Exception:
        (directory/'dashboard_error.txt').write_text(traceback.format_exc())
    finally:
        if writer is not None:writer.release()
        if hero_writer is not None:hero_writer.release()
        if renderer is not None:renderer.close()
        if hero_renderer is not None:hero_renderer.close()
        if shared_preview is not None:shared_preview.close()
        if show:cv2.destroyAllWindows()

class LiveDashboard:
    def __init__(self,xml,directory,show=False,map_path=None,preview_buffer=None,slam=False):
        self.directory=Path(directory);ctx=mp.get_context('spawn');self.inputs=ctx.Queue(maxsize=1);self.dropped=0
        from shared_preview import SharedPreview
        self.preview_buffer=preview_buffer if preview_buffer is not None else SharedPreview()
        self.process=ctx.Process(target=_run,args=(xml,str(directory),self.inputs,show,str(map_path) if map_path else None,self.preview_buffer.spec,self.preview_buffer.lock,slam));self.process.start()
    def submit(self,data,preview,position,phase,consumed_sensor_stamp=0.):
        packet=dict(stamp=float(data.time),qpos=data.qpos.copy(),estimated_position=np.asarray(position).copy(),phase=phase,consumed_sensor_stamp=consumed_sensor_stamp)
        try:self.inputs.put_nowait(packet)
        except queue.Full:
            self.dropped+=1
            try:self.inputs.get_nowait()
            except queue.Empty:pass
            try:self.inputs.put_nowait(packet)
            except queue.Full:pass
    def close(self):
        try:self.inputs.put(None,timeout=3)
        except queue.Full:pass
        self.process.join(timeout=20)
        if self.process.is_alive():self.process.terminate();self.process.join()
        self.preview_buffer.close()
        (self.directory/'dashboard_transport.json').write_text(json.dumps(dict(dropped_observer_packets=self.dropped,physics_waits_for_dashboard=False,pixel_transport='shared_memory_nonblocking',control_queue_carries_images=False)))
