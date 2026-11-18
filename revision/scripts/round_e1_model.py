"""Build a fixed-base, resolution-matched variant without altering the production XML."""
import math
from pathlib import Path
import sys
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mujoco_closed_loop"))
from model import ARC_LENGTH, BASE_LENGTH, SimArmBridge
from super_metric_model import build_super_metric

NATIVE = 8


def mass_properties(model, data, ids):
    masses = model.body_mass[ids]
    mass = masses.sum()
    centers = data.xipos[ids]
    center = np.average(centers, weights=masses, axis=0)
    inertia = np.zeros((3, 3))
    for i, pos, m in zip(ids, centers, masses):
        R = data.ximat[i].reshape(3, 3)
        v = pos - center
        inertia += R @ np.diag(model.body_inertia[i]) @ R.T
        inertia += m * (np.dot(v, v) * np.eye(3) - np.outer(v, v))
    return dict(mass=float(mass), com=center.tolist(), inertia=inertia.tolist())


def neutral(model):
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    return d


def properties(model, n):
    d = neutral(model)
    ids = np.array([model.body('seg%d' % i).id for i in range(n)])
    joints = np.array([model.joint('%s%d' % (axis, i)).id for i in range(n) for axis in ('bx','by')])
    acts = np.array([model.actuator('%s%d' % (axis, i)).id for i in range(n) for axis in ('bx','by')])
    dofs = model.jnt_dofadr[joints]
    def common(v):
        assert np.allclose(v, v[0], rtol=1e-10, atol=1e-14)
        return float(v[0])
    return dict(arm=mass_properties(model, d, ids),
                whole=mass_properties(model, d, np.arange(1, model.nbody)),
                distal_mass_kg=float(model.body_subtreemass[model.body('tool').id]),
                zero_tip_world=d.site('tip').xpos.tolist(),
                drive_stiffness_Nm_rad=common(model.actuator_gainprm[acts,0])/n,
                drive_bias_stiffness_Nm_rad=-common(model.actuator_biasprm[acts,1])/n,
                drive_damping_Nms_rad=-common(model.actuator_biasprm[acts,2])/n,
                passive_damping_Nms_rad=common(model.dof_damping[dofs])/n,
                reflected_inertia_kgm2=common(model.dof_armature[dofs])/n,
                equivalent_axis_range_rad=[common(model.jnt_range[joints,j])*n for j in (0,1)],
                torque_limits_Nm=[common(model.actuator_forcerange[acts,j]) for j in (0,1)],
                gravity=model.opt.gravity.tolist())


def build(n, dt):
    native = mujoco.MjModel.from_xml_string(build_super_metric(fixed=True,segments=NATIVE)[0])
    ref = properties(native, NATIVE)
    root = ET.fromstring(build_super_metric(fixed=True,segments=n)[0])
    root.find('option').set('timestep', str(dt))
    # The native neutral arm is coaxial with world z. Across N, preserve all
    # neutral composite moments, accounting for each slice's parallel-axis term.
    length = ARC_LENGTH
    ds = length/n
    mass = ref['arm']['mass']
    tensor = np.array(ref['arm']['inertia'])
    assert np.allclose(tensor, np.diag(np.diag(tensor))) and np.isclose(tensor[0,0],tensor[1,1])
    ix = (tensor[0,0] - mass*(length**2 - ds**2)/12)/n
    iz = tensor[2,2]/n
    assert 2*ix >= iz > 0
    for i in range(n):
        body = root.find(".//body[@name='seg%d']" % i)
        ET.SubElement(body, 'inertial', pos='0 0 0', mass=str(mass/n),
                      diaginertia='%.17g %.17g %.17g' % (ix, ix, iz))
        for j in body.findall('joint'):
            name = j.get('name')
            a = root.find("./actuator/position[@name='%s']" % name)
            a.set('kp', str(ref['drive_stiffness_Nm_rad']*n))
            a.set('kv', str(ref['drive_damping_Nms_rad']*n))
            j.set('damping', str(ref['passive_damping_Nms_rad']*n))
            j.set('armature', str(ref['reflected_inertia_kgm2']*n))
            j.set('range', ' '.join(str(x/n) for x in ref['equivalent_axis_range_rad']))
    contact = root.find('contact')
    pairs = {frozenset((e.get('body1'), e.get('body2'))) for e in contact}
    def local_seam(a,b):
        key = frozenset((a,b))
        if key not in pairs:
            ET.SubElement(contact,'exclude',body1=a,body2=b)
            pairs.add(key)
    for i in range(n):
        s = (i+.5)*ds
        if s <= length/NATIVE: local_seam('mount', 'seg%d'%i)
        if length-s <= length/NATIVE: local_seam('tool', 'seg%d'%i)
        for j in range(i+1,n):
            if (j-i)*ds <= 3*length/NATIVE+1e-10:
                local_seam('seg%d'%i, 'seg%d'%j)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    got = properties(model,n)
    for k,value in ref.items():
        if isinstance(value,dict):
            for name,v in value.items(): np.testing.assert_allclose(got[k][name],v,rtol=2e-10,atol=1e-12)
        else: np.testing.assert_allclose(got[k],value,rtol=2e-10,atol=1e-12)
    return model, got


def contacts(model,data):
    return [(model.geom(c.geom1).name or 'unnamed',model.geom(c.geom2).name or 'unnamed',float(c.dist))
            for c in data.contact[:data.ncon]]


def geometry_controls(model,n):
    data=neutral(model)
    rest=contacts(model,data)
    SimArmBridge(n).set_kinematic_shape(model,data,math.radians(60),math.pi/2)
    mujoco.mj_forward(model,data)
    witness=contacts(model,data)
    assert not rest and any('landing' in (a+b).lower() or 'skid' in (a+b).lower() for a,b,_ in witness), (rest,witness)
    return dict(neutral_contacts=rest, unsafe_phi90_theta60=witness)
