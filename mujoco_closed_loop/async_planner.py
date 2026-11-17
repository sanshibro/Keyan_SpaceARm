"""Nonblocking V2 adapter process and conservative phase-zero takeover gate."""
import multiprocessing as mp,time,queue,json
from pathlib import Path
import numpy as np
from planner_adapter import V21Adapter
from uav_continuum_whole_body_v2.mapping import MapSnapshot
from uav_continuum_whole_body_v2.execution import whole_body_switch_deltas,whole_body_switch_gate

GATES={'position_gap_m':.05,'yaw_gap_rad':np.deg2rad(3),'bending_gap_rad':np.deg2rad(3),'linear_velocity_gap_m_s':.03,'yaw_rate_gap_rad_s':np.deg2rad(5),'bending_rate_gap_rad_s':np.deg2rad(5)}

def _solve(path,profile,state,goal,budget,result,keep_yaw=False,survey_camera=False,progress_path=None,loaded=False,tracking_reserve=0.,terminal_straight=False,terminal_shape=None):
    started=time.monotonic()
    def progress(stage,info):
        if progress_path:Path(progress_path).write_text(json.dumps(dict(stage=stage,worker_wall_s=time.monotonic()-started,diagnostics=info),default=str,indent=2))
    try:
        progress('loading_map',{})
        snapshot=MapSnapshot.load(path);adapter=V21Adapter(snapshot,profile,survey_camera=survey_camera,loaded=loaded,tracking_reserve=tracking_reserve)
        trajectory,info=adapter.plan(state,goal,budget=max(.1,budget-(time.monotonic()-started)),keep_yaw=keep_yaw,progress=progress,terminal_straight=terminal_straight,terminal_shape=terminal_shape)
        progress('finished',info)
        info['worker_total_wall_s']=time.monotonic()-started;result.put((trajectory,info))
    except Exception as error:result.put((None,{'status':'worker_error_hold','error':repr(error)}))

class AsyncV21Planner:
    def __init__(self):self.process=None;self.status='idle'
    def request(self,path,profile,state,goal,map_hash,budget=15.,keep_yaw=False,survey_camera=False,progress_path=None,loaded=False,tracking_reserve=0.,terminal_straight=False,terminal_shape=None):
        if self.process is not None:raise RuntimeError('Pending request must be polled or closed')
        ctx=mp.get_context('spawn');self.results=ctx.Queue(maxsize=1);self.hash=map_hash;self.started=time.monotonic();self.budget=budget
        self.process=ctx.Process(target=_solve,args=(str(path),profile,np.array(state),np.array(goal),budget,self.results,keep_yaw,survey_camera,progress_path,loaded,tracking_reserve,terminal_straight,terminal_shape));self.process.start();self.status='planning'
    def poll(self,current_state,current_rate,map_hash):
        if self.process is None:return None,{'status':self.status}
        try:trajectory,info=self.results.get_nowait()
        except queue.Empty:
            if time.monotonic()-self.started>self.budget:
                self.close();self.status='deadline_hold';return None,{'status':self.status}
            if not self.process.is_alive():
                self.process.join();self.process=None;self.status='worker_exited_hold';return None,{'status':self.status}
            return None,{'status':'planning'}
        self.process.join(timeout=.001)
        # Result delivered after numerical work is finished; close feeder without flight wait.
        self.finished_process=self.process;self.process=None
        if map_hash!=self.hash:info['status']='map_changed_reject';trajectory=None
        if time.monotonic()-self.started>self.budget:info['status']='deadline_reject';trajectory=None
        if trajectory is not None:
            gaps=whole_body_switch_deltas(current_state,current_rate,trajectory.states[0],trajectory.velocities[0]);violations=whole_body_switch_gate(gaps,GATES)
            info['takeover_gaps']=gaps;info['takeover_violations']=violations
            if violations:trajectory=None;info['status']='start_or_rate_mismatch_hold_requires_validated_bridge'
        self.status='candidate_ready' if trajectory is not None else info['status'];return trajectory,info
    def close(self):
        if self.process is not None:
            if self.process.is_alive():self.process.terminate()
            self.process.join(timeout=.05);self.process=None
