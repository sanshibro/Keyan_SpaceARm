"""Reuse V2.1 reference/seed/optimizer/validator; unknown space stays blocked."""
from pathlib import Path
from dataclasses import replace
import sys,time
import numpy as np
ROOT=Path(__file__).resolve().parent
LIB=ROOT.parent/'xtdrone_catkin_ws/src/uav_continuum_whole_body_v2'
sys.path.insert(0,str(LIB/'src'))
from tool_envelope import collision_samples
from qingyu_model import ARC_LENGTH, BASE_LENGTH
from shape_limits import PLAN_THETA_MAX
from uav_continuum_whole_body_v2.configuration import load_planner_configuration
from uav_continuum_whole_body_v2.types import ArmModel
from uav_continuum_whole_body_v2.reference_planner import VoxelAStarReferencePlanner,path_states
from uav_continuum_whole_body_v2.horizon import local_horizon
from uav_continuum_whole_body_v2.seed_planner import WholeBodySeedPlanner
from uav_continuum_whole_body_v2.optimizer import WholeBodyOptimizer
from uav_continuum_whole_body_v2.validator import SweptVolumeValidator
from uav_continuum_whole_body_v2.geometry import world_arm_points,world_body_points
from uav_continuum_whole_body_v2.mapping import MapSnapshot
from uav_continuum_whole_body_v2.trajectory import SplineTrajectory

class _SerialQueryTree:
    """Avoid creating a CPU-count-sized thread pool for each local query."""
    def __init__(self,tree):self.tree=tree
    def query(self,*args,**kwargs):
        kwargs.pop('n_jobs',None);kwargs['workers']=1
        return self.tree.query(*args,**kwargs)

class V21Adapter:
    def __init__(self,snapshot,profile='qingyu',loaded=False,survey_camera=False,tracking_reserve=0.):
        self.snapshot=snapshot;self.world=snapshot.world(unknown_is_occupied=True)
        if self.world.tree is not None:self.world.tree=_SerialQueryTree(self.world.tree)
        self.raw,self.config,self.limits,self.mass=load_planner_configuration(LIB/'config/planner_v2_realtime.yaml')
        approved_stage1=snapshot.metadata.get('robot_visual_profile')=='approved_stage1'
        # Scenario-specific deployment envelope. The 0.42 m body height
        # covers the Mid-360 top (+0.197 m) and the landing rails (-0.14 m).
        # The sensor voxel radius is separately subtracted by MapWorld; the
        # 30 mm tracking reserve is added below. These settings are audited
        # against all three physical passages before online launch.
        if approved_stage1 and survey_camera:
            self.limits=replace(self.limits,body_height=max(self.limits.body_height,.42),
                                body_margin=.01,arm_margin=.01)
        elif survey_camera:self.limits.body_height=max(self.limits.body_height,.42)
        self.hard_limits=replace(self.limits)
        self.limits=replace(self.limits,body_margin=self.limits.body_margin+tracking_reserve,arm_margin=self.limits.arm_margin+tracking_reserve)
        self.arm=ArmModel(arc_length=ARC_LENGTH if profile=='qingyu' else .22,base_offset_z=BASE_LENGTH if profile=='qingyu' else .10,extra_tool_points=collision_samples(loaded=loaded,approved_stage1=approved_stage1) if profile=='qingyu' else ())
        # New runs reserve 2 degrees below the physical 120-degree envelope.
        # The arm bound is also used by the independent dense validator.
        self.arm=replace(self.arm,theta_max=PLAN_THETA_MAX)
        self.config=replace(self.config,theta_design_margin=0.)
        if loaded:self.mass.tool+=.04
        self.validator=SweptVolumeValidator(self.world,self.arm,self.mass,self.limits,self.config)
    def clearance(self,state):
        body=self.world.minimum_distance(world_body_points(state,self.limits))[0]
        arm=self.world.minimum_distance(world_arm_points(state,self.arm))[0]-self.arm.radius
        return dict(body=float(body),arm=float(arm))
    def short_direct_bridge(self,start,end,started,budget):
        """Certify a slow straight rest-to-rest bridge inside observed space."""
        if np.linalg.norm(end[:3]-start[:3])>.25:
            return None,None
        controls=np.linspace(start,end,4)
        for duration in (1.5,2.,3.,4.,6.,8.):
            if time.monotonic()-started>budget-.5:
                break
            spline=SplineTrajectory(controls,[duration],stationary_start=True,stationary_end=True)
            proof=self.validator.validate(spline)
            if proof.feasible:
                trajectory=spline.sample(self.config.output_dt,source='v21_validated_short_direct_bridge',metadata={
                    'span_durations':[duration], 'conditioned_initial_rate':None,
                    'bridge_geometry':'straight_interpolation', 'validation':proof.metrics})
                return trajectory,proof
        return None,None
    def reference_snapshot(self,state,goal,padding=1.):
        """Bound candidate centre enumeration, retaining every occupied voxel.

        Removed observed cells become unknown in the reference search. The
        optimizer and independent validator still use the complete input map.
        This can reject a route needing a larger detour; it cannot clear one.
        """
        lower=np.minimum(state[:3],np.asarray(goal)[:3])-padding
        upper=np.maximum(state[:3],np.asarray(goal)[:3])+padding
        lower[2]=self.limits.z_min-self.limits.body_height/2
        upper[2]=self.limits.z_max+self.limits.body_height/2
        points=self.snapshot.observed_points
        selected=np.all((points>=lower)&(points<=upper),axis=1)
        observed=np.vstack([self.snapshot.observed_indices[selected],self.snapshot.occupied_indices])
        return MapSnapshot(self.snapshot.resolution,self.snapshot.occupied_indices,observed,frame_id=self.snapshot.frame_id,source=self.snapshot.source,stamp=self.snapshot.stamp,metadata=dict(self.snapshot.metadata,reference_parent_hash=self.snapshot.map_hash,reference_candidate_bounds=[lower.tolist(),upper.tolist()]))
    def plan(self,state,goal,budget=15.,keep_yaw=False,progress=None,terminal_straight=False,terminal_shape=None):
        started=time.monotonic();diagnostics={'map_hash':self.snapshot.map_hash,'profile_arm_length':self.arm.arc_length,'unknown_is_occupied':True,'initial_clearance':self.clearance(state)}
        if progress:progress('initial_clearance',diagnostics)
        if diagnostics['initial_clearance']['body']<self.limits.body_margin or diagnostics['initial_clearance']['arm']<self.limits.arm_margin:
            return None,dict(diagnostics,status='start_unverified_or_blocked_hold',elapsed=time.monotonic()-started)
        try:
            if np.linalg.norm(np.asarray(goal)[:3]-state[:3])<=.25:
                end=np.asarray(state,dtype=float).copy();end[:3]=np.asarray(goal)[:3]
                if not keep_yaw:end[3]=np.arctan2(end[1]-state[1],end[0]-state[0])
                if terminal_shape is not None:end[4:6]=terminal_shape
                elif terminal_straight:end[4]=0.
                bridge,proof=self.short_direct_bridge(state,end,started,budget)
                if bridge is not None:
                    diagnostics.update(status='validated_short_direct_bridge',success=True,
                                       elapsed=time.monotonic()-started,validation=proof.__dict__,
                                       reference={'search':'V2_dense_validated_direct_bridge'},
                                       optimizer={'backend':'V2_spline_direct_bridge'})
                    return bridge,diagnostics
            reference_snapshot=self.reference_snapshot(state,goal)
            diagnostics['reference_candidate_observed_voxels']=len(reference_snapshot.observed_indices)
            diagnostics['full_observed_voxels']=len(self.snapshot.observed_indices)
            # Test the same continuous body envelope before enumerating every
            # possible A* centre. The full arm still needs seed + dense proof.
            centres=np.linspace(state[:3],np.asarray(goal)[:3],max(2,int(np.ceil(np.linalg.norm(np.asarray(goal)[:3]-state[:3])/(.5*self.snapshot.resolution)))+1))
            template=world_body_points(np.zeros(6),self.limits,azimuth_samples=16,vertical_samples=3)
            groups=centres[:,None,:]+template[None,:,:]
            direct=bool(np.all((centres[:,2]>=self.limits.z_min)&(centres[:,2]<=self.limits.z_max)) and self.world._known_mask(groups.reshape(-1,3)).all() and np.all(self.world.minimum_distances_grouped(groups)[0]>=self.limits.body_margin))
            planner=None
            if direct:
                points=np.array([state[:3],np.asarray(goal)[:3]])
                reference_info=dict(search='certified_continuous_body_bridge',expanded=0,terminal_reached=True,certified_terminal_shortcut=True)
            else:
                planner=VoxelAStarReferencePlanner(reference_snapshot,body_radius=self.limits.body_radius,body_height=self.limits.body_height,body_margin=self.limits.body_margin,z_min=self.limits.z_min,z_max=self.limits.z_max,time_budget=min(5.,budget/2))
                points,reference_info=planner.plan(state[:3],np.asarray(goal)[:3])
            # The original planner appends the continuous terminal bridge
            # after voxel simplification. Remove a voxel-centre kink only if
            # its existing continuous body guard certifies the direct bridge.
            if planner is not None and planner._continuous_bridge_is_valid(state[:3],np.asarray(goal)[:3]):
                points=np.array([state[:3],np.asarray(goal)[:3]]);reference_info['certified_terminal_shortcut']=True
            reference=path_states(points,default_shape=state[4:6]);reference[0]=state
            if keep_yaw:reference[:,3]=state[3]
            if terminal_shape is not None:reference[-1,4:6]=terminal_shape
            elif terminal_straight:reference[-1,4:6]=[0.,state[5]]
            diagnostics['heading_reference']='hold_initial' if keep_yaw else 'path_tangent'
            diagnostics['reference']=reference_info
            diagnostics['reference_states']=reference.tolist()
            if progress:progress('reference_ready',diagnostics)
            window,nearest=local_horizon(reference,state,self.config.horizon_distance)
            terminal=reference[-1].copy() if (terminal_straight or terminal_shape is not None) and np.linalg.norm(window[-1,:3]-np.asarray(goal)[:3])<.05 else None
            fixed_terminal_plane=terminal is not None and abs(terminal[5]-state[5])<1e-8
            seed=WholeBodySeedPlanner(self.world,self.arm,self.mass,self.limits,self.config).plan(window,current_state=state,terminal_state=terminal,_force_phi_locked=fixed_terminal_plane)
            if terminal is not None:
                seed.diagnostics['terminal_shape']=terminal[4:6].tolist()
                seed.states[-1]=terminal
            diagnostics['seed']=seed.diagnostics
            if progress:progress('seed_ready',diagnostics)
            remaining=budget-(time.monotonic()-started)
            if remaining<=0:return None,dict(diagnostics,status='deadline_before_optimization',elapsed=time.monotonic()-started)
            result=WholeBodyOptimizer(self.world,self.arm,self.mass,self.limits,self.config,self.validator).optimize(seed,window,wall_time_budget=remaining,validation_reserve=.15)
            diagnostics.update(status=result.status,success=result.success,elapsed=time.monotonic()-started,reference=reference_info,optimizer=result.diagnostics,validation=result.validation.__dict__ if result.validation else None)
            if diagnostics['elapsed']>budget:diagnostics['status']='deadline_exceeded_reject';return None,diagnostics
            return result.trajectory if result.success else None,diagnostics
        except Exception as error:
            diagnostics.update(status='planning_failed_hold',error=type(error).__name__+': '+str(error),elapsed=time.monotonic()-started);return None,diagnostics
