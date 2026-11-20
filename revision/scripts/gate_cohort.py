"""Check raw evidence and summarize paired, independently flown single-gate trials."""
import argparse
import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path
import tarfile
import zipfile

import numpy as np

METHODS = ('vision', 'strain', 'fixed', 'proposed', 'oracle')
CONDITIONS = [(m, v) for m in ('nominal', 'differential') for v in ('natural', 'occluded')]


def finite(value):
    return float(value) if np.isfinite(value) else None


def span(values):
    values = [x for x in values if x is not None]
    if not values:
        return None
    return {'min': float(min(values)), 'median': float(np.median(values)), 'max': float(max(values))}


def bend_degrees(vector):
    if not np.isfinite(vector).all():
        return None
    return float(np.rad2deg(np.linalg.norm(vector)))


def phase_optics(z, spec):
    start, end = (spec['window'][k] for k in ('blackout_start', 'blackout_end'))
    with np.load(BytesIO(z.read('optics/observations.npz')), allow_pickle=False) as obs:
        capture = obs['t']
        delivery = obs['arrival_t']
        valid = obs['valid'].astype(bool)
        if not np.allclose(delivery-capture,spec['camera']['delay_s'], rtol=0,atol=1.e-7):
            raise ValueError('Camera transport delay changed')
        during = (capture >= start-1.e-8) & (capture < end-1.e-8)
        after = capture >= end-1.e-8
        arrivals = np.flatnonzero(after & valid)
        return dict(frames=len(valid), valid_shutter_exposures=int(np.sum(valid & during)),
                    shutter_exposures=int(during.sum()),
                    post_window_first_valid_exposure_s=float(capture[arrivals[0]]) if len(arrivals) else None,
                    post_window_first_valid_delivery_s=float(delivery[arrivals[0]]) if len(arrivals) else None,
                    valid_in_transit_at_shutter_start=int(np.sum(valid & (capture<start-1.e-8) & (delivery>=start-1.e-8))))


def image_references(stage, protocol):
    from PIL import Image
    k=next(r['case'] for r in protocol['cases'] if r['seed']==protocol['representative_seed'] and
           (r['method'],r['material'],r['vision'])==('proposed','differential','occluded'))
    optics=stage/'formal'/k/'optics'
    frames=[json.loads(s) for s in (optics/'index.jsonl').read_text().splitlines()]
    row=next(r for r in frames if abs(r['t']-30.)<1.e-7)
    refs={}
    for key in ('rgb','depth'):
        asset=optics/row[key]
        with Image.open(asset) as im:
            if np.asarray(im).max()!=0:raise ValueError('Reference shutter image contains a return')
        refs[key]=hashlib.sha256(asset.read_bytes()).hexdigest()
        if refs[key]!=row[key+'_sha256']:raise IOError('Reference shutter image hash')
    with tarfile.open(stage/'records/registered-cad-assets.tgz','r:gz') as a:
        refs['model']=hashlib.sha256(a.extractfile('source-model.xml').read()).hexdigest()
    return refs


def evidence(stage, case, m, item, protocol, design, refs):
    path = stage/'evidence'/item['file']
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    if path.stat().st_size != item['bytes'] or checksum != item['sha256']:
        raise ValueError('Archive fingerprint: '+case)
    with zipfile.ZipFile(path) as z:
        files = json.loads(z.read('evidence-hashes.json'))
        if set(z.namelist()) != set(files) | {'evidence-hashes.json'}:
            raise ValueError('Unlisted evidence: '+case)
        for name, expect in files.items():
            h = hashlib.sha256()
            with z.open(name) as f:
                for block in iter(lambda: f.read(1<<20), b''):
                    h.update(block)
            if h.hexdigest()!=expect:
                raise IOError(case+'/'+name)
        if any(files[n]!=v for n,v in m['artifacts'].items()):
            raise ValueError('Flight and cohort fingerprints differ: '+case)
        if json.loads(z.read('metrics.json')) != m or json.loads(z.read('protocol.json')) != protocol:
            raise ValueError('Score or flight protocol differs: '+case)
        run = json.loads(z.read('run.json'))
        if run['error'] is not None or json.loads(z.read('control-summary.json'))['mechanics_only']:
            raise ValueError('Unqualified flight: '+case)
        pr=json.loads(z.read('provenance.json'));mission=pr['mission']
        registered=design['inputs']
        if (pr['plant_xml_sha256']!=refs['model'] or files['model.xml']!=refs['model'] or
            pr['protocol_sha256']!=files['protocol.json'] or
            mission['perception_lock_sha256']!=design['perception_lock'] or
            mission['controller_sha256']!=registered['revision/scripts/gate_feedback_flight.py'] or
            mission['model_builder_sha256']!=registered['revision/scripts/gate_feedback_model.py'] or
            m['scorer_sha256']!=registered['revision/scripts/gate_feedback_scores.py'] or
            mission['case']!=m['case']):
            raise ValueError('Registered plant, feedback, protocol, or scorer identity changed: '+case)
        with z.open('flight.ulg') as f:
            if not f.read(8).startswith(b'ULog\x01\x12\x35'):
                raise IOError('Native ULog signature: '+case)
        with z.open('flight.bag') as f:
            if f.readline()!=b'#ROSBAG V2.0\n':
                raise IOError('Native bag signature: '+case)
        op = phase_optics(z,protocol)
        frames = [json.loads(line) for line in z.read('optics/index.jsonl').splitlines()]
        if len(frames)!=op['frames']:
            raise ValueError('Pixel and optical track counts differ: '+case)
        s,e = (protocol['window'][k] for k in ('blackout_start','blackout_end'))
        masked=[row for row in frames if s-1.e-8<=row['t']<e-1.e-8]
        black=sum(row['mask_fraction']>=1-1.e-8 for row in masked)
        if black != (len(masked) if m['case']['vision']=='occluded' else 0):
            raise ValueError('Unexpected physical image occlusion: '+case)
        if black and op['valid_shutter_exposures']:
            raise ValueError('Native geometry accepted a full black image: '+case)
        if black and any(row['rgb_sha256']!=refs['rgb'] or row['depth_sha256']!=refs['depth'] for row in masked):
            raise ValueError('Exposure does not match true optical black and zero depth: '+case)
        op['black_exposures']=black
        with np.load(BytesIO(z.read('score-trace.npz')),allow_pickle=False) as q:
            t=q['t']
            nearest=int(np.argmin(q['arm_tool_clearance']))
            truth=bend_degrees(q['true_b'][nearest])
            estimate=bend_degrees(q['est_b'][nearest])
            command=bend_degrees(q['des_b'][nearest])
            geo=dict(time_s=float(t[nearest]),body_z_m=float(q['body_pose'][nearest,2]),
                     tool_z_m=float(q['tip'][nearest,2]),theta_true_deg=truth,theta_est_deg=estimate,
                     theta_est_minus_true_deg=estimate-truth if estimate is not None else None,
                     theta_des_minus_true_deg=command-truth if command is not None else None,
                     nearest_robot_m=q['arm_near_robot'][nearest].tolist(),
                     nearest_gate_m=q['arm_near_gate'][nearest].tolist(),pair=m['arm_nearest']['geoms'])
            if abs(float(t[nearest])-m['arm_nearest']['t'])>1.e-6:
                raise ValueError('Closest point time differs')
        a=json.loads(z.read('optics/complete.json'))
        if not a['frozen_sources_unchanged']:
            raise ValueError('Perception manifest changed')
        timing=next((i for i,n in enumerate(a['timing_columns']) if n=='total'),None)
        if timing is not None:
            op['per_image_wall_median_ms']=a['processing_wall_median_ms'][timing]
            op['per_image_wall_p95_ms']=a['processing_wall_p95_ms'][timing]
        return dict(archive_sha256=checksum,member_hashes=len(files),ulog_sha256=files['flight.ulg'],
                    bag_sha256=files['flight.bag'],physics_sha256=files['physics.npz'],optics=op,
                    arm_at_minimum=geo,score_sha256=files['metrics.json'],physical_xml_sha256=refs['model'],
                    perception_lock_sha256=mission['perception_lock_sha256'],px4_executable_sha256=pr['binary_sha256'],
                    physical_dt_s=pr['physics_dt_s'],hil_dt_s=pr['hil_dt_s'],mass_kg=pr['mass_kg'],
                    mujoco_version=pr['mujoco'])


def group_metrics(members):
    vals=list(members.values())
    agg={}
    for k,scale in [('body_min_m',100),('arm_tool_min_m',100),('self_min_m',1000),('tip_rmse_m',100),
                    ('tip_peak_m',100),('axis_rmse_deg',1),('axis_peak_deg',1),('dynamic_theta_range_deg',1),
                    ('dynamic_phi_range_deg',1),('theta_blackout_rmse_deg',1),('phi_blackout_rmse_deg',1),
                    ('theta_rmse_deg',1),('phi_rmse_deg',1),('hold_s',1),('feedback_hold_s',1),
                    ('clearance_hold_s',1),('completion_s',1),('true_max_total_curvature_deg',1)]:
        unit='cm' if scale==100 else 'mm' if scale==1000 else None
        name=k[:-1]+unit if unit else k
        agg[name]=span([v[k]*scale if v[k] is not None else None for v in vals])
    for k in ('theta_rmse_deg','phi_rmse_deg','theta_peak_deg','phi_peak_deg'):
        agg['vision_hold_'+k]=span([v['vision_hold_blackout'][k] for v in vals])
    for k in ('whole_angle','blackout_angle'):
        agg[k+'_counts']={s:sum(v[k][s] for v in vals) for s in ('samples','available','azimuth_samples')}
    return dict(n=len(vals),passed=sum(v['success'] for v in vals),safe=sum(v['safe'] for v in vals),
                crossed=sum(v['crossed'] for v in vals),dwelled=sum(v['dwell_reached'] for v in vals),
                arm_below_red=sum(v['arm_tool_min_m']<.06 for v in vals),
                body_below_red=sum(v['body_min_m']<.06 for v in vals),
                self_zero_or_negative=sum(v['self_min_m']<=0 for v in vals),
                physical_contact_flights=sum(v['physical_contact_steps']>0 for v in vals),
                metrics=agg)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',type=Path)
    p.add_argument('--partial',action='store_true',help='Write a separately named progress audit')
    args=p.parse_args();stage=args.stage
    p0=json.loads((stage/'protocol.json').read_bytes())
    design=json.loads((stage/'records/formal-design.json').read_bytes())
    d=json.loads((stage/'decisions/metrics.json').read_bytes());cases=d['cases'];items=d['evidence']
    expected={c['case']:c for c in p0['cases']}
    if set(items)!=set(cases) or not set(cases)<=set(expected) or design['design_commit']!=d['design_commit']:
        raise ValueError('Cohort identity mismatch')
    if not args.partial and set(cases)!=set(expected):
        raise ValueError('All 60 registered flights are required')
    audit={}; rows=[];refs=image_references(stage,p0)
    for case,m in cases.items():
        if m['case']!=expected[case]:
            raise ValueError(case)
        a=evidence(stage,case,m,items[case],p0,design,refs);audit[case]=a
        rows.append({'case':case,**m['case'], 'success':m['success'],'safe':m['safe'],'crossed':m['crossed'],
                     'dwell_reached':m['dwell_reached'], 'body_min_cm':m['body_min_m']*100,
                     'arm_min_cm':m['arm_tool_min_m']*100,'self_min_mm':m['self_min_m']*1000,
                     'contact_steps':m['physical_contact_steps'],'tip_rmse_cm':m['tip_rmse_m']*100,
                     'axis_rmse_deg':m['axis_rmse_deg'],'total_hold_s':m['hold_s'],
                     'feedback_hold_s':m['feedback_hold_s'],'clearance_hold_s':m['clearance_hold_s'],
                     'arm_minimum_time_s':a['arm_at_minimum']['time_s'],
                     'est_minus_true_at_arm_minimum_deg':a['arm_at_minimum']['theta_est_minus_true_deg'],
                     'true_undershoot_at_arm_minimum_deg':a['arm_at_minimum']['theta_des_minus_true_deg'],
                     'tool_z_at_minimum_m':a['arm_at_minimum']['tool_z_m'],
                     'body_z_at_minimum_m':a['arm_at_minimum']['body_z_m'],
                     **a['optics']})
    groups={}
    for material,visual in CONDITIONS:
        groups[material+'-'+visual]={}
        for method in METHODS:
            sel={k:v for k,v in cases.items() if (v['case']['material'],v['case']['vision'],v['case']['method'])==(material,visual,method)}
            groups[material+'-'+visual][method]=group_metrics(sel)
    out=stage/('qa' if args.partial else 'decisions');out.mkdir(exist_ok=True,parents=True)
    incidents=[json.loads(p.read_bytes()) for p in sorted((stage/'records/interruptions').glob('*.json'))
               if p.name.startswith(tuple(str(seed)+'-' for seed in p0['seeds']))]
    for item in incidents:
        if item['design_commit']!=d['design_commit'] or item['case'] not in expected:
            raise ValueError('Unregistered interruption record')
    summary=dict(design_commit=d['design_commit'],approved_perception=p0['approved'],trials=len(cases),
                 cases_with_verified_raw_members=len(audit),total_raw_members=sum(v['member_hashes'] for v in audit.values()),
                 different_phystraces=len(set(v['physics_sha256'] for v in audit.values())),
                 different_ulogs=len(set(v['ulog_sha256'] for v in audit.values())),
                 different_rosbags=len(set(v['bag_sha256'] for v in audit.values())),
                 verified_black_rgbd_exposures=sum(v['optics']['black_exposures'] for v in audit.values()),
                 shared_model_xml=refs['model'],perception_lock=design['perception_lock'],
                 distinct_px4_executables=len({v['px4_executable_sha256'] for v in audit.values()}),
                 interrupted_acquisitions=[{'case':v['case'],'frames_saved':v['rgb_images'],
                                             'reason':v['reason'],'archive':v['retained_folder'],
                                             'registered_flight_scored':v['case'] in cases} for v in incidents],
                 conditions=groups)
    for filename,value in [('cohort-summary.json',summary),('case-evidence-audit.json',audit)]:
        (out/filename).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    with (out/'gate-physical-audit.csv').open('w',newline='',encoding='utf-8-sig') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(len(cases),'flights;',summary['total_raw_members'],'verified members')

if __name__=='__main__':
    main()
