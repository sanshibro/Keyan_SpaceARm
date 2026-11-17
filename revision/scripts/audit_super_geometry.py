"""Check native-mesh coverage and sampled self-collision of the new assembly."""
from pathlib import Path
import sys,json,math
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'mujoco_closed_loop'))
from super_metric_model import build_super_metric
from model import SimArmBridge

def vertices(p):
    return np.array([[float(x) for x in l.split()[1:4]] for l in p.read_text().splitlines() if l.startswith('v ')])

def main():
    xml,info=build_super_metric();model=mujoco.MjModel.from_xml_string(xml)
    landing=json.loads((ROOT/'revision/data/custom_landing_geometry.json').read_text())
    max_outside=-1.;count=0
    for p in landing['parts']:
        v=vertices(ROOT/p['file']);c=p['collision'];q=np.array(c['quat_wxyz'])
        R=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
        local=(v-np.array(c['center']))@R
        max_outside=max(max_outside,float((abs(local)-np.array(c['half_size'])).max()));count+=len(v)
    clearance=min(r['prop_z_interval_m'][0] for r in info['rotors'])-landing['bounds_m'][1][2]
    standoff_gap=min(np.linalg.norm(np.array(r['axis_center_m'])[:2]-p)-r['prop_radius_m']-.003
                     for r in info['rotors'] for p in np.array([[.025,.025],[-.025,.025],[-.025,-.025],[.025,-.025]]))
    samples=[];bridge=SimArmBridge()
    for theta in [0,30,60,90,110,118]:
        for phi in range(0,360,45):
            data=mujoco.MjData(model);bridge.set_kinematic_shape(model,data,math.radians(theta),math.radians(phi))
            mujoco.mj_forward(model,data)
            contacts=[dict(a=model.geom(c.geom1).name,b=model.geom(c.geom2).name,distance_m=float(c.dist))
                      for c in data.contact[:data.ncon] if c.dist < -1e-6]
            samples.append(dict(theta_deg=theta,phi_deg=phi,self_collision_free=not contacts,contacts=contacts))
    body=info['runtime_geometry'];height=body['body_envelope_height_m'];cz=body['body_envelope_center_z_m']
    report=dict(platform_id=info['platform_id'],selected_airframe='SUPER-Hardware',
        assembly_status='registered development candidate; fasteners/load capacity not verified',
        dimensions=dict(prop_swept_radius_m=info['prop_sweep_body_radius_m'],rigid_height_m=height,
            rigid_center_offset_z_m=cz,rigid_z_interval_m=body['rigid_z_interval_m']),
        landing_component_count=len(landing['parts']),landing_vertices_checked=count,
        largest_obb_outside_m=max_outside,rotor_to_landing_min_vertical_gap_m=clearance,
        standoff_to_rotor_sweep_min_horizontal_gap_m=float(standoff_gap),
        coverage_passed=bool(max_outside<=1e-8 and clearance>=.005-1e-8 and standoff_gap>0),
        pose_samples=samples,self_collision_free_samples=sum(x['self_collision_free'] for x in samples),
        warning='Sampled pose results are not a continuous swept-volume certificate. Planner self-collision integration remains required.')
    (ROOT/'revision/data/super_geometry_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='pose_samples'},indent=2))
    for x in samples:
        if not x['self_collision_free']:print('BLOCKED',x['theta_deg'],x['phi_deg'],x['contacts'][:2])
    assert report['coverage_passed']

if __name__=='__main__':main()
