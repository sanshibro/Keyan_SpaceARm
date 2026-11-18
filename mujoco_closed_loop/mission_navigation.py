"""Single-owner V2 reference execution helper for mission development."""
import json,time
from collections import deque
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from planner_adapter import V21Adapter
from async_planner import AsyncV21Planner
from trajectory_tools import reconstruct_spline
from uav_continuum_whole_body_v2.mapping import MapSnapshot
from uav_continuum_whole_body_v2.execution import clearance_tracking_budget,clearance_progress_scale
from uav_continuum_whole_body_v2.geometry import world_arm_points,world_body_points
from uav_continuum_whole_body_v2.trajectory import SplineTrajectory

def choose_online_frontier(snapshot,state,final_goal,max_step=.5,min_step=.02):
    """Select a known six-DoF local goal; V2.1 proves the swept trajectory."""
    deadline=time.monotonic()+12.
    state=np.asarray(state,dtype=float)
    goal=np.asarray(final_goal,dtype=float)
    adapter=V21Adapter(snapshot,survey_camera=True,tracking_reserve=.03)
    if state[4]<np.deg2rad(1):state[4:6]=0.
    delta=goal-state[:3];distance=float(np.linalg.norm(delta))
    if distance<1e-9:return goal,'final',{'distance_m':0.}
    direction=delta/distance
    def safe(q):
        if not adapter.limits.z_min<=q[2]<=adapter.limits.z_max:return None
        clearance=adapter.clearance(q)
        return clearance if (clearance['body']>=adapter.limits.body_margin and
                             clearance['arm']>=adapter.limits.arm_margin) else None

    def certify(candidate,step,reason):
        if time.monotonic()>deadline:return None
        clearance=safe(candidate)
        if clearance is None:return None
        # The candidate is a six-dimensional state: body height and the PCC
        # shape change together before the independent dense V2.1 check.
        samples=max(8,int(np.ceil(np.linalg.norm(candidate[:3]-state[:3])/.025))+1)
        for u in np.linspace(0.,1.,samples)[1:-1]:
            if time.monotonic()>deadline or safe(state+(candidate-state)*u) is None:return None
        spline=SplineTrajectory(np.linspace(state,candidate,4),[6.],stationary_start=True,stationary_end=True)
        proof=adapter.validator.validate(spline,stop_on_failure=True)
        if not proof.feasible:return None
        info=dict(distance_m=float(step),reason=reason,altitude_adjustment_m=float(candidate[2]-state[2]),
                  terminal_shape=candidate[4:6].tolist(),clearance=clearance,validation=proof.__dict__)
        final=bool(np.linalg.norm(candidate[:3]-goal)<.005)
        return candidate[:3].copy(),'final' if final else 'frontier',info

    if safe(state) is None:
        return None,'hold',dict(reason='start_unverified_or_blocked',distance_to_goal_m=distance)
    steps=sorted({round(float(s),4) for s in (min(max_step,distance),.35,.25,.15,.08,.04,min_step)
                  if min_step<=s<=distance+1e-9} | ({round(distance,4)} if distance<min_step else set()),reverse=True)
    # First retain the current shape where the observed map already permits
    # progress. A body-only navigation reference never authorizes crossing.
    for step in steps:
        if step<.15 and distance>=.15:break
        candidate=state.copy();candidate[:3]=state[:3]+direction*step
        result=certify(candidate,step,'current_shape_progress')
        if result is not None:return result
    # Risk-zone seed search from the paper: vertical offsets and theta/phi
    # are considered in the same candidate state, then V2.1 validates the
    # entire transition. Try the present bending plane before activating
    # other planes. Unknown cells remain blocked.
    heights=list(dict.fromkeys(round(float(z),3) for z in
        (goal[2],(state[2]+goal[2])/2,state[2],goal[2]-.05,goal[2]+.05)))
    shapes=[(float(state[4]),float(state[5]))]
    planes=list(dict.fromkeys((float(state[5]),0.,np.pi/2,-np.pi/2,np.pi)))
    for theta in (np.deg2rad(60),np.deg2rad(90),np.deg2rad(110)):
        for phi in planes:shapes.append((theta,phi))
    for step in steps:
        for z in heights:
            for theta,phi in shapes:
                if time.monotonic()>deadline:
                    return None,'hold',dict(reason='joint_frontier_search_budget_exhausted',distance_to_goal_m=distance)
                candidate=state.copy();candidate[:3]=state[:3]+direction*step
                candidate[2]=z;candidate[4:6]=theta,phi
                result=certify(candidate,step,'joint_altitude_shape_progress')
                if result is not None:return result
    # A stationary adjustment is useful only when the latest sensor map also
    # proves a short forward lookahead at the resulting height and shape.
    for z in heights:
        for theta,phi in shapes[1:]:
            if time.monotonic()>deadline:
                return None,'hold',dict(reason='joint_frontier_search_budget_exhausted',distance_to_goal_m=distance)
            candidate=state.copy();candidate[2]=z;candidate[4:6]=theta,phi
            probe=candidate.copy();probe[:2]+=direction[:2]*min(.08,distance)
            if safe(probe) is None:continue
            result=certify(candidate,0.,'joint_preparation_with_forward_lookahead')
            if result is not None:return result
    return None,'hold',dict(reason='insufficient_known_joint_clearance_scan_required',distance_to_goal_m=distance)

class MissionNavigator:
    def __init__(self,map_path,directory,name,initial_state,goal,loaded=False,frontier=False,goal_shape=None,live_map_required=False):
        self.directory=Path(directory);self.name=name;self.goal=np.asarray(goal);self.loaded=loaded
        self.goal_shape=None if goal_shape is None else np.asarray(goal_shape,dtype=float)
        self.live_map_required=bool(live_map_required);self.live_map_refreshed=False
        # Online entry waypoints are followed by an explicit stationary
        # sensor survey. Their acceptance gate need only show a short stable
        # hover; the old 3 s continuous gate could never survive centimetre
        # FAST-LIO fluctuations and prevented the pre-gate scan from starting.
        self.hold_duration=.8 if (frontier or live_map_required) else 3.
        self.initial_state=np.asarray(initial_state).copy()
        self.anchor_limit=.03 if name=='exit_portal' else .005
        # 30 mm covers the measured 5.3 mm UAV localization error plus the
        # <=4.9 deg arm-shape error (~24 mm at 0.28 m); the previous 40 mm
        # reserve rejected the independently validated sensor-map passage.
        self.tracking_reserve=.03
        # The complete sensor map remains the runtime and final proof source.
        # A local observed subset shortens worker startup. Everything outside
        # it becomes unknown; all occupied hits are retained, so no cell is
        # made freer by this reduction.
        self.snapshot=MapSnapshot.load(map_path)
        observed=self.snapshot.observed_points
        lower=np.minimum(np.asarray(initial_state)[:2],self.goal[:2])-2.
        upper=np.maximum(np.asarray(initial_state)[:2],self.goal[:2])+2.
        selected=np.all((observed[:,:2]>=lower)&(observed[:,:2]<=upper),axis=1)&(observed[:,2]>=-1.)&(observed[:,2]<=3.)
        self.planning_snapshot=MapSnapshot(self.snapshot.resolution,self.snapshot.occupied_indices,
            np.vstack([self.snapshot.observed_indices[selected],self.snapshot.occupied_indices]),
            frame_id=self.snapshot.frame_id,source=self.snapshot.source,stamp=self.snapshot.stamp,
            metadata=dict(self.snapshot.metadata,local_parent_hash=self.snapshot.map_hash,
                          local_xy_bounds_m=[lower.tolist(),upper.tolist()],local_z_bounds_m=[-1.,3.]),
            traversed_positions=self.snapshot.traversed_positions)
        self.path=self.directory/(name+'_planning_map.npz')
        self.planning_snapshot.save(self.path)
        self.adapter=V21Adapter(self.snapshot,loaded=loaded,survey_camera=True,tracking_reserve=self.tracking_reserve)
        (self.directory/(name+'_navigation_config.json')).write_text(json.dumps(dict(goal=self.goal.tolist(),goal_shape=None if self.goal_shape is None else self.goal_shape.tolist(),loaded=loaded,planning_reserve_m=self.tracking_reserve,planning_body_margin_m=self.adapter.limits.body_margin,planning_arm_margin_m=self.adapter.limits.arm_margin,hard_body_margin_m=self.adapter.hard_limits.body_margin,hard_arm_margin_m=self.adapter.hard_limits.arm_margin,map_hash=self.snapshot.map_hash,planning_map_hash=self.planning_snapshot.map_hash,map_scope='conservative_local_observed_crop_full_map_runtime_and_validation',local_xy_bounds_m=[lower.tolist(),upper.tolist()],local_z_bounds_m=[-1.,3.]),indent=2))
        self.planner=AsyncV21Planner();self.reference=np.asarray(initial_state).copy();self.phase=0.;self.phase_rate=0.
        self.spline=None;self.trajectory=None;self.status='ready';self.info={};self.requested=False;self.done=False;self.hold_since=None
        self.recent_hold_positions=deque(maxlen=12)
        self.last_bend=np.zeros(2);self.bend_rate=np.zeros(2);self.tick=0;self.physics_steps_planning=0
        self.reference_velocity=np.zeros(3);self.reference_acceleration=np.zeros(3)
        self.trace=[];self.needs_map_refresh=False

    def receive_candidate(self, candidate, measured, rates, info):
        """Check the recovered spline on the full receiving map before installing it."""
        try:
            spline = reconstruct_spline(candidate)
            proof = self.adapter.handoff_review(measured, rates, spline)
        except ValueError as error:
            self.status = 'spline_identity_rejected'
            info['error'] = str(error)
            return False
        info['full_sensor_map_validation'] = proof.__dict__
        if not proof.feasible:
            self.status = 'full_sensor_map_validation_rejected'
            return False
        archive = self.directory / (self.name + '_trajectory.npz')
        try:
            with TemporaryDirectory(prefix='.candidate-', dir=self.directory) as directory:
                pending = Path(directory) / 'trajectory.npz'
                np.savez_compressed(pending, times=candidate.times, states=candidate.states,
                                    velocities=candidate.velocities, accelerations=candidate.accelerations,
                                    jerks=candidate.jerks, control_states=spline.control_states,
                                    span_durations=spline.span_durations)
                pending.replace(archive)
        except OSError as error:
            self.status = 'trajectory_archive_failed'
            info['error'] = str(error)
            return False
        self.spline, self.trajectory = spline, candidate
        self.status = 'executing'
        return True

    def prepare_synchronous_frontier(self,measured,velocity,gyro):
        """Plan a short observed frontier while MuJoCo physics is paused."""
        measured=np.asarray(measured,dtype=float)
        planning_state=measured.copy()
        if planning_state[4]<np.deg2rad(1):planning_state[4:6]=0.
        candidate,info=self.adapter.plan(planning_state,self.goal,budget=15.,keep_yaw=True,terminal_straight=self.goal_shape is None,terminal_shape=self.goal_shape)
        self.requested=True;self.info=info;self.status=info['status']
        if candidate is not None:
            from async_planner import GATES
            from uav_continuum_whole_body_v2.execution import whole_body_switch_deltas,whole_body_switch_gate
            rates=np.r_[velocity,gyro[2],0.,0.]
            gaps=whole_body_switch_deltas(measured,rates,candidate.states[0],candidate.velocities[0])
            violations=whole_body_switch_gate(gaps,GATES)
            info['takeover_gaps']=gaps;info['takeover_violations']=violations
            if not violations:
                if self.receive_candidate(candidate,measured,rates,info):
                    self.reference=candidate.states[0].copy()
            else:self.status='start_or_rate_mismatch_hold_requires_validated_bridge'
        (self.directory/(self.name+'_planner.json')).write_text(json.dumps(info,default=str,indent=2))
        return self.spline is not None

    def update(self,stamp,position,rotation,velocity,bending,gyro,healthy,dt=.001):
        measured=np.r_[position,np.arctan2(rotation[1,0],rotation[0,0]),np.linalg.norm(bending),np.arctan2(bending[1],bending[0])]
        self.bend_rate=.999*self.bend_rate+.001*(bending-self.last_bend)/dt;self.last_bend=bending.copy()
        if not self.requested and healthy:
            self.reference=measured.copy()
            planning_state=measured.copy()
            # A surveyed hover anchor is a reproducible nominal planning
            # start only while the actual estimated state is within 5 mm.
            # The independent takeover gate still checks the actual state.
            anchor_gap=float(np.linalg.norm(measured[:3]-self.initial_state[:3]))
            anchored=bool(anchor_gap<self.anchor_limit and abs(measured[3]-self.initial_state[3])<np.deg2rad(1))
            if anchored:planning_state[:4]=self.initial_state[:4]
            # At neutral the noisy bending-plane angle is undefined. Plan
            # from the held neutral command only within a 1-degree gap; the
            # unchanged takeover gate checks the actual measured bend/rate.
            if planning_state[4]<np.deg2rad(1):planning_state[4:6]=0.
            (self.directory/(self.name+'_request.json')).write_text(json.dumps(dict(measured_state=measured.tolist(),planning_state=planning_state.tolist(),neutral_snap_limit_deg=1.,hover_anchor_used=anchored,hover_anchor_gap_m=anchor_gap,hover_anchor_limit_m=self.anchor_limit),indent=2))
            self.planner.request(self.path,'qingyu',planning_state,self.goal,self.planning_snapshot.map_hash,budget=15.,keep_yaw=True,survey_camera=True,loaded=self.loaded,tracking_reserve=self.tracking_reserve,terminal_straight=self.goal_shape is None,terminal_shape=self.goal_shape,progress_path=self.directory/(self.name+'_worker_progress.json'))
            self.requested=True;self.status='planning'
        if self.planner.process is not None:self.physics_steps_planning+=1
        if self.tick%100==0:
            self.recent_hold_positions.append((stamp,position.copy()))
            if self.planner.process is not None:
                theta=max(1e-8,np.linalg.norm(bending));theta_rate=float(bending@self.bend_rate/theta)
                phi_rate=float((bending[0]*self.bend_rate[1]-bending[1]*self.bend_rate[0])/(theta*theta)) if theta>1e-4 else 0.
                rates=np.r_[velocity,gyro[2],theta_rate,phi_rate]
                candidate,info=self.planner.poll(measured,rates,self.planning_snapshot.map_hash)
                if info['status']!='planning':
                    self.info=info;self.status=info['status']
                    if candidate is not None:
                        self.receive_candidate(candidate,measured,rates,self.info)
                    (self.directory/(self.name+'_planner.json')).write_text(json.dumps(self.info,default=str,indent=2))
            if self.spline is not None:
                clearance=self.adapter.clearance(measured)
                if clearance['body']<self.adapter.hard_limits.body_margin or clearance['arm']<self.adapter.hard_limits.arm_margin:
                    self.needs_map_refresh=True
                validation=self.info['validation']['metrics']
                hard=self.adapter.hard_limits
                slack=min(clearance['body']-hard.body_margin,clearance['arm']-hard.arm_margin,validation['body_clearance_min']-hard.body_margin,validation['arm_clearance_min']-hard.arm_margin)
                budget=clearance_tracking_budget(slack,0.,.015,.08)
                arm_error=np.linalg.norm(world_arm_points(measured,self.adapter.arm)-world_arm_points(self.reference,self.adapter.arm),axis=1).max()
                error=max(np.linalg.norm(position-self.reference[:3]),float(arm_error))
                self.phase_rate=clearance_progress_scale(error,.025,budget) if healthy else 0.
                self.status='executing' if self.phase_rate>0 else 'tracking_or_visibility_hold'
                self.trace.append([stamp,self.phase,self.phase_rate,budget,error,clearance['body'],clearance['arm'],int(healthy),*self.reference])
                target_bend=np.zeros(2) if self.goal_shape is None else self.goal_shape[0]*np.array([np.cos(self.goal_shape[1]),np.sin(self.goal_shape[1])])
                recent=[p for sample_stamp,p in self.recent_hold_positions if stamp-sample_stamp<=.8]
                position_stable=(len(recent)>=8 and
                    max(np.linalg.norm(p-position) for p in recent)<(.02 if self.live_map_required else .01))
                if (self.phase>=self.spline.duration
                    and np.linalg.norm(position-self.goal)<.03
                    and np.linalg.norm(velocity)<(.025 if self.live_map_required else .018)
                    and abs(gyro[2])<.05
                    and position_stable
                    and np.linalg.norm(bending-target_bend)<np.deg2rad(5)):
                    if self.hold_since is None:self.hold_since=stamp
                    self.status='goal_hold'
                    if stamp-self.hold_since>=self.hold_duration:self.done=True
                else:self.hold_since=None
        if self.spline is not None:
            if self.live_map_required and not self.live_map_refreshed:
                self.phase_rate=0.;self.status='awaiting_live_map_validation'
            self.phase=min(self.spline.duration,self.phase+dt*self.phase_rate)
            self.reference=self.spline.evaluate([self.phase])[0]
            self.reference_velocity=self.spline.evaluate([self.phase],1)[0,:3]*self.phase_rate
            self.reference_acceleration=self.spline.evaluate([self.phase],2)[0,:3]*self.phase_rate**2
        self.tick+=1
        return self.reference.copy()

    def close(self):
        self.planner.close()
        np.savez_compressed(self.directory/(self.name+'_execution_trace.npz'),data=np.array(self.trace),columns=['sim_time','reference_phase','phase_rate','tracking_budget_m','whole_body_error_m','estimated_body_clearance_m','estimated_arm_clearance_m','sensors_healthy','reference_x','reference_y','reference_z','reference_yaw','reference_theta','reference_phi'])
        (self.directory/(self.name+'_execution.json')).write_text(json.dumps(dict(status=self.status,done=self.done,phase=self.phase,physics_steps_planning=self.physics_steps_planning),indent=2))
