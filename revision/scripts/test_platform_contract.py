"""Behavioral checks for the newly identified platform, not historical runs."""
from pathlib import Path
import sys,math,json,unittest
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'mujoco_closed_loop'))
from arm_command import ArmCommand,MujocoArmBackend
from super_metric_model import build_super_metric,allocation_matrix

class PlatformChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        xml,cls.info=build_super_metric(fixed=False)
        cls.model=mujoco.MjModel.from_xml_string(xml)
    def setUp(self):
        self.data=mujoco.MjData(self.model)
        self.backend=MujocoArmBackend(self.model,self.data)
    def tearDown(self):
        self.model.actuator_gainprm[self.backend.ids]=self.backend.gain
        self.model.actuator_biasprm[self.backend.ids]=self.backend.bias
    def test_single_section_units_wrap_and_limits(self):
        c=ArmCommand(0,0.,math.radians(120),math.radians(450))
        f=c.qingyu_joint_fields()
        self.assertAlmostEqual(f['theta1'],118);self.assertAlmostEqual(f['phi1'],90)
        for k in ('theta2','phi2','theta3','phi3'):self.assertEqual(f[k],0)
        self.assertEqual(ArmCommand(1,1.,0.,math.pi).normalized().phi_rad,0.)
        for a in (float('nan'),math.radians(121),-.1):
            with self.assertRaises(ValueError):ArmCommand(2,2.,a,0.).normalized()
    def test_disabled_actuator_does_not_command_neutral_pose(self):
        c=ArmCommand(0,0.,math.radians(90),.4);self.backend.apply(c)
        before=self.data.qpos.copy();self.backend.apply(ArmCommand(1,.1,c.theta_rad,c.phi_rad,False))
        mujoco.mj_forward(self.model,self.data)
        self.assertTrue(np.array_equal(before,self.data.qpos))
        self.assertTrue(np.allclose(self.data.actuator_force[self.backend.ids],0.))
        self.backend.apply(ArmCommand(2,.2,c.theta_rad,c.phi_rad,True));mujoco.mj_forward(self.model,self.data)
        self.assertGreater(np.linalg.norm(self.data.actuator_force[self.backend.ids]),0.)
    def test_out_of_order_commands_retain_controls(self):
        self.backend.apply(ArmCommand(5,2.,.2,.1));before=self.data.ctrl.copy()
        for c in (ArmCommand(5,3.,1.,1.),ArmCommand(6,1.,1.,1.)):
            with self.assertRaises(ValueError):self.backend.apply(c)
            self.assertTrue(np.array_equal(before,self.data.ctrl))
    def test_allocation_uses_compiled_rotor_lever_arms(self):
        a=allocation_matrix(self.model)
        self.assertTrue(np.allclose(abs(a[1:3]),.098995,atol=2e-6))
        target=np.array([22.,.13,-.11,.04]);force=np.linalg.solve(a,target)
        self.assertTrue(np.allclose(a@force,target,atol=1e-12))
        mujoco.mj_forward(self.model,self.data)
        for i in range(4):
            s=self.model.site('rotor%d'%i).id
            self.assertTrue(np.allclose(self.model.site_pos[s],self.info['rotors'][i]['axis_center_m']))
    def test_physical_collision_geometries_present(self):
        for name in ['collision_carbon','collision_central_equipment']+['collision_rotor_sweep%d'%i for i in range(4)]:
            g=self.model.geom(name).id
            self.assertEqual(self.model.geom_contype[g],1)
            self.assertEqual(self.model.geom_conaffinity[g],1)
        self.assertLess(max(abs(x) for x in self.info['calibration']['residual_span_m']),3e-6)
    def test_free_endpoints_do_not_imply_a_free_bending_transition(self):
        from platform_self_collision import SelfCollisionOracle
        oracle=SelfCollisionOracle()
        a=[0.,math.pi/2];b=[math.pi/2,math.pi/2]
        self.assertTrue(oracle.check(*a)['accepted'])
        self.assertTrue(oracle.check(*b)['accepted'])
        verdict=oracle.transition(a,b)
        self.assertFalse(verdict['accepted']);self.assertTrue(verdict['witness']['contacts'])
        guarded=MujocoArmBackend(self.model,self.data,collision_guard=oracle)
        before=self.data.ctrl.copy()
        with self.assertRaises(ValueError):guarded.apply(ArmCommand(0,0.,*b))
        self.assertTrue(np.array_equal(before,self.data.ctrl))
        guarded.apply(ArmCommand(1,1.,*b,enable=False))
        self.assertTrue(np.allclose(self.model.actuator_gainprm[guarded.ids],0.))
    def test_front_bend_and_periodic_plane_transition_are_clear(self):
        from platform_self_collision import SelfCollisionOracle
        oracle=SelfCollisionOracle()
        self.assertTrue(oracle.sequence([[0,math.pi],[math.radians(118),math.pi],[0,math.pi]])['accepted'])
        r=oracle.transition([math.pi/2,math.radians(179)],[math.pi/2,math.radians(-179)])
        self.assertTrue(r['accepted']);self.assertLess(r['checked'],8)

if __name__=='__main__':unittest.main(verbosity=2)
