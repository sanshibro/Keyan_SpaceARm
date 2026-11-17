"""Internal FK audit against the independent SDK-sized PCC reference."""
import numpy as np,mujoco
from model import model_xml,SimArmBridge
from planner_adapter import ROOT
from qingyu_model import ARC_LENGTH,BASE_LENGTH
from estimation import orientation_shape
from uav_continuum_whole_body_v2.geometry import local_arm_sections
from uav_continuum_whole_body_v2.types import ArmModel
reference=ArmModel(arc_length=ARC_LENGTH,base_offset_z=BASE_LENGTH);report={}
for method in ('endpoint','trapezoid'):
    m=mujoco.MjModel.from_xml_string(model_xml(discretization=method));d=mujoco.MjData(m);bridge=SimArmBridge();position=[];angle=[]
    for theta in np.deg2rad([0,30,60,90,120]):
        for phi in np.deg2rad(np.arange(0,360,45)):
            bridge.set_kinematic_shape(m,d,theta,phi);mujoco.mj_forward(m,d)
            mount=m.body('mount').id;tool=m.body('tool').id;R=d.xmat[mount].reshape(3,3)
            p=(d.xpos[tool]-d.xpos[mount])@R;expected=local_arm_sections(theta,phi,reference)[1][-1]
            T=np.eye(4);T[:3,:3]=R.T@d.xmat[tool].reshape(3,3)
            b=orientation_shape(T,np.eye(4));position.append(np.linalg.norm(p-expected));angle.append(np.rad2deg(np.linalg.norm(b-theta*np.array([np.cos(phi),np.sin(phi)]))))
    report[method]=dict(max_endpoint_error_m=max(position),max_bending_error_deg=max(angle))
print(report)
assert report['trapezoid']['max_endpoint_error_m']<.01
assert report['trapezoid']['max_bending_error_deg']<5.
assert report['trapezoid']['max_bending_error_deg']<report['endpoint']['max_bending_error_deg']/5
assert report['trapezoid']['max_endpoint_error_m']<report['endpoint']['max_endpoint_error_m']/3
print('PASS: same8 numerical bending pairs, substantially smaller PCC position and artificial-torsion error')
