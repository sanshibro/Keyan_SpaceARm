"""Regression checks tied to observed failures, not implementation snapshots."""
import math,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'mujoco_closed_loop'),str(ROOT/'xtdrone_catkin_ws/src/uav_continuum_whole_body_v2/src')]
from model import SimArmBridge
from shape_limits import joint_bending_metrics
from uav_continuum_whole_body_v2.comparison_methods import InformedRrtStarWholeBodyMethod as RRT,ComparisonProblem
from uav_continuum_whole_body_v2.types import ArmModel,Limits,MassModel,PlannerConfig
from uav_continuum_whole_body_v2.geometry import world_arm_points
from uav_continuum_whole_body_v2.environment import BoxWorld

class RevisionGuards(unittest.TestCase):
    def test_periodic_angles_have_short_distance(self):
        a=np.zeros(6);b=a.copy();a[3]=a[5]=math.radians(179);b[3]=b[5]=math.radians(-179)
        expected=math.radians(2)*math.hypot(.25,.15)
        self.assertAlmostEqual(RRT._metric(a,b),expected,places=12)
        self.assertAlmostEqual(RRT._metric(a,b),RRT._metric(b,a),places=12)

    def test_pure_yaw_edge_detects_interior_tool_collision(self):
        arm=ArmModel();mid=np.array([0.,0.,1.5,0.,math.pi/2,0.])
        tip=world_arm_points(mid,arm)[-1]
        world=BoxWorld.from_dicts([dict(name='swept_tool_obstacle',center=tip.tolist(),size=[.025,.025,.025])])
        a=mid.copy();b=mid.copy();a[3]=-math.pi/2;b[3]=math.pi/2
        problem=ComparisonProblem(world=world,arm=arm,mass=MassModel(),limits=Limits(com_max=10.),config=PlannerConfig(),reference_states=np.array([a,b]),current_state=a)
        self.assertTrue(RRT._state_feasible(problem,a));self.assertTrue(RRT._state_feasible(problem,b))
        self.assertFalse(RRT._state_feasible(problem,mid))
        self.assertFalse(RRT._edge_feasible(problem,a,b))
        self.assertFalse(RRT._edges_feasible(problem,np.array([a]),np.array([b]))[0])

    def test_runtime_reserve_for_every_bending_plane(self):
        bridge=SimArmBridge()
        for phi in np.linspace(-math.pi,math.pi,25):
            q=bridge.move_joints(math.radians(120),float(phi)).reshape(-1,2)
            self.assertAlmostEqual(joint_bending_metrics(q)['total_curvature_deg'],118.,places=10)
        self.assertEqual(bridge.command_limit_events,25)
        for theta in (math.radians(120.1),float('nan'),float('inf'),-1.):
            with self.assertRaises(ValueError):bridge.move_joints(theta,0.)

    def test_evaluation_detects_overshoot_without_clipping(self):
        q=np.tile([math.radians(121)/8,0.],(8,1));original=q.copy()
        result=joint_bending_metrics(q)
        self.assertFalse(result['within_hard_limit']);self.assertAlmostEqual(result['total_curvature_deg'],121.)
        np.testing.assert_equal(q,original)

if __name__=='__main__':unittest.main(verbosity=2)
