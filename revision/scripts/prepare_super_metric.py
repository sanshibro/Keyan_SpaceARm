"""Build a separately identified metric SUPER asset set; leave source CAD intact.

The archived Collada is a display assembly. Match its carbon board to the
official 226 mm STEP board before using it for geometry. This does not identify
mass, inertia, thrust or the real modification's mechanical interfaces.
"""
from pathlib import Path
import argparse, hashlib, json, math
import numpy as np

def obj_vertices(path):
    return np.array([[float(x) for x in ln.split()[1:4]]
                     for ln in path.read_text().splitlines() if ln.startswith('v ')])

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main(root):
    root=Path(root).resolve(); source=root/'mujoco_closed_loop/vendor'
    converted=source/'super_converted'; assembled=converted/'full_assembly'
    out=root/'revision/generated/super_metric';out.mkdir(parents=True,exist_ok=True)
    checks=[]
    upstream=json.loads((root/'revision/data/super_hardware_upstream_tree.json').read_text())
    for item in upstream['tree']:
        if item['type']!='blob':continue
        p=source/'SUPER-Hardware'/item['path']
        if not p.exists():continue
        data=p.read_bytes(); gitsha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        checks.append(dict(path=item['path'],match=gitsha==item['sha'],sha256=digest(p)))
    if not checks or not all(x['match'] for x in checks):raise ValueError('Upstream hardware content mismatch')
    step=json.loads((converted/'provenance.json').read_text())
    display=json.loads((assembled/'provenance.json').read_text())
    for name,part in display['parts'].items():
        if digest(assembled/(name+'.obj'))!=part['sha256']:raise ValueError('Mesh hash mismatch: '+name)
    board=obj_vertices(converted/'main_board.obj');carbon=obj_vertices(assembled/'carbon.obj')
    step_span=np.ptp(board,axis=0)[[0,2]];display_span=np.ptp(carbon,axis=0)[:2]
    ratio=step_span/display_span;scale=float(ratio.mean())
    if abs(ratio[0]-ratio[1])>2e-5:raise ValueError('Anisotropic board scale mismatch')
    motor_all=obj_vertices(assembled/'motor.obj')
    # The Collada 'motor' MATERIAL also contains 2725 central electronics
    # vertices; it is not a semantic component. Select the four peripheral
    # motor regions before fitting axes, rather than treating its AABB as one.
    peripheral_threshold=float(.5*min(display_span))
    motor=motor_all[np.linalg.norm(motor_all[:,:2],axis=1)>peripheral_threshold]
    xycenter=(motor.min(0)+motor.max(0))/2
    centers=[]
    for sx,sy in [(1,1),(-1,1),(-1,-1),(1,-1)]:
        part=motor[((motor[:,0]-xycenter[0])*sx>0)&((motor[:,1]-xycenter[1])*sy>0)]
        centers.append((part.min(0)+part.max(0))/2)
    centers=np.array(centers);origin=centers.mean(0)
    centers=(centers-origin)*scale
    allpts=[];parts={}
    for p in sorted(assembled.glob('*.obj')):
        vertices=obj_vertices(p);v=(vertices-origin)*scale;allpts.append(v)
        target=out/p.name;lines=[];i=0
        for ln in p.read_text().splitlines():
            if ln.startswith('v '):ln='v '+' '.join('%.9g'%x for x in v[i]);i+=1
            lines.append(ln)
        target.write_text('\n'.join(lines)+'\n')
        parts[p.stem]=dict(file=str(target.relative_to(root)),source_sha256=digest(p),sha256=digest(target),
            bounds_min=v.min(0).tolist(),bounds_max=v.max(0).tolist(),vertices=len(v))
    prop=obj_vertices(out/'prop.obj');indices=np.argmin(np.linalg.norm(prop[:,None,:2]-centers[None,:,:2],axis=2),axis=1)
    rotors=[]
    for i,center in enumerate(centers):
        v=prop[indices==i];radius=float(np.linalg.norm(v[:,:2]-center[:2],axis=1).max())
        rotors.append(dict(index=i,axis_center_m=center.tolist(),prop_radius_m=radius,
            prop_z_interval_m=[float(v[:,2].min()),float(v[:,2].max())],
            yaw_torque_per_thrust_m=.02*(-1)**i,
            yaw_torque_status='legacy provisional coefficient and spin ordering; requires verification'))
    allpts=np.vstack(allpts);sweep_radius=max(np.linalg.norm(np.array(x['axis_center_m'])[:2])+x['prop_radius_m'] for x in rotors)
    report=dict(schema_version=1,platform_id='super_qingyu_metric_candidate_v1',target_confirmed='SUPER airframe plus user continuum arm and modifications',
        upstream_url='https://github.com/hku-mars/SUPER-Hardware',upstream_commit=upstream['sha'],
        source_assembly='mujoco_closed_loop/vendor/SUPER-Hardware/SolidWorks/yunque.SLDASM',
        source_license='MIT; original preserved in vendor/SUPER-Hardware/LICENSE',upstream_files_verified=checks,
        units='m',frame='U: right-handed; +X front (archived front LEDs), +Y left, +Z up',
        origin='mean of four motor-axis AABB centres in the archived assembly',
        transform_from_display=dict(uniform_scale=scale,origin_display_m=origin.tolist()),
        calibration=dict(method='uniform in-plane carbon-board span match to official STEP',step_span_m=step_span.tolist(),
            display_span_m=display_span.tolist(),per_axis_scale=ratio.tolist(),
            residual_span_m=(display_span*scale-step_span).tolist(),
            physical_measurement=False,scope='CAD/display registration; component geometry and actual build still require cross-checks'),
        rotor_axis_extraction=dict(method='peripheral motor vertices; four quadrant AABB axes',
            peripheral_threshold_display_m=peripheral_threshold,selected_vertices=len(motor),
            central_material_vertices_excluded=len(motor_all)-len(motor)),
        source_sha256=digest(assembled/'provenance.json'),parts=parts,rotors=rotors,
        bounds_m=[allpts.min(0).tolist(),allpts.max(0).tolist()],prop_sweep_body_radius_m=float(sweep_radius),
        flexible_arm=dict(geometry_source='Qingyu visualization config',arc_length_m=.28,base_length_m=.0704,physical_length_verified=False),
        dynamics=dict(calibrated=False,mass_source='existing development approximation',rotor_max_N=12.,
            rotor_max_status='provisional, not F90 thrust calibration'),
        cad16=dict(identity='different tube-frame/M2810 platform; retained as reference and modification asset source',do_not_use_as_SUPER_airframe=True))
    path=root/'revision/data/super_metric_geometry.json';path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:report[k] for k in ['platform_id','calibration','bounds_m','prop_sweep_body_radius_m']},indent=2))
    print('Verified upstream files:',len(checks),'rotors:',centers.tolist())

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',default=str(Path(__file__).resolve().parents[2]));main(ap.parse_args().root)
