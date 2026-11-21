"""Release the industrial M2 campaign only from measured, reproducible pilot evidence."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
from statistics import median

ROOT=Path(__file__).resolve().parent.parent
DEV=(26112207,26112209)
FORMAL=(26112241,26112247,26112253)
METHODS=('straight','fixed','sequential','unregularized','coupled','stop','hard')
LOCK='2e8bb8e132355e2ac2e92be9a354a0617a773c819cb8282a2b0595a330656235'
SCENE=Path(os.environ.get('M2_SCENE',ROOT/'scene'))
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(test,why):
    if not test:raise ValueError(why)
def identity():
    import importlib.util
    source=ROOT/'source'
    pending=['m2_flight','m2_gate'];modules=set()
    while pending:
        name=pending.pop()
        if name in modules:continue
        modules.add(name)
        tree=ast.parse((source/(name+'.py')).read_text(encoding='utf8'))
        for node in ast.walk(tree):
            imports=([node.module or ''] if isinstance(node,ast.ImportFrom) else
                     [a.name for a in node.names] if isinstance(node,ast.Import) else [])
            pending.extend(k for k in imports if k.startswith('m2_') and k not in modules)
    runtime=[source/(name+'.py') for name in modules]
    runtime.extend(sorted(source.glob('m2_*.cc')))
    runtime.append(source/'m2_rolling_inspection.json')
    hil=Path(importlib.util.find_spec('round_f_sitl').origin)
    return dict(source={p.name:digest(p) for p in sorted(runtime)},hil_sender=digest(hil),scene=read(SCENE/'manifest.json')['scene_sha256'],
                registry=digest(source/'m2_self_registry.py'),perception=LOCK)
def check_static(path):
    d=read(path);manifest=read(SCENE/'manifest.json')
    require(d['scene_sha256']==manifest['scene_sha256'],'static scene digest')
    require(d['registry_sha256']==digest(ROOT/'source/m2_self_registry.py'),'static self registry digest')
    from m2_scene_lock import verify
    verify(SCENE,ROOT/'qa/registered-collision-scene.json')
    require(d['launch_preflight']['passed'] and d['launch_preflight']['zero_physical_contact'],'release-to-hover geometry')
    optics=d['target_optics']
    require(optics['scene_sha256']==d['scene_sha256'] and optics['stand_off_pass'] and optics['depth_center_pass'],'shared inspection optics')
    match=read(ROOT/'qa/static/native-crosscheck.json')
    require(match['passed'] and match['scene_sha256']==d['scene_sha256'] and match['native_launch_zero_contact'],'cross-platform geometry and optics')
    rows=d['gate_witnesses'];require({r['gate'] for r in rows}=={'gate1','gate2','gate3'} and len(rows)==3,'three physical witnesses')
    for r in rows:
        require(r['external_lower_m']>=.105 and r['hazard_lower_m']>=.015 and r['fit_lower_m']>=0,'static geometry margin at '+r['gate'])
    return d
def authorize(strategy,seed,development,gate_path):
    require(strategy in METHODS,'unknown method')
    require(seed in (DEV if development else FORMAL),'unregistered seed or campaign')
    path=Path(gate_path);d=read(path)
    if development:
        check_static(path);return dict(kind='development',static_sha256=digest(path),identity=identity())
    require(d.get('kind')=='qualified' and d['identity']==identity(),'pilot identity or qualification')
    for name,h in d['evaluation_sources'].items():
        require(digest(ROOT/'source'/name)==h,'pilot audit code changed: '+name)
    require(d['formal_seeds']==list(FORMAL) and d['methods']==list(METHODS),'campaign registration')
    require(d['static_sha256']==digest(path.parent/d['static_receipt']),'changed static preflight')
    check_static(path.parent/d['static_receipt'])
    for record in d['development']:
        require(digest(path.parent/record['path'])==record['sha256'],'pilot evidence changed')
    return dict(kind='formal',receipt_sha256=digest(path),identity=d['identity'])
def certify(static,metrics,out):
    static=Path(static).resolve();out=Path(out).resolve()
    check_static(static);ident=identity();runs={}
    for f in metrics:
        f=Path(f).resolve();m=read(f);k=(m['strategy'],m['seed'])
        require(k not in runs and m['identity']==ident and m['development'] is True,'unpaired or stale pilot')
        runs[k]=(f,m)
    for seed in DEV:
        m=runs['coupled',seed][1];w=m['solve_wall_s']
        require(m['success'] and m['gripper_open_command'] and m['synchronized_rgbd'] and m['native_logs_verified'],'pilot physical inspection')
        require(len(w)>=3 and m['activated_replans']>=3 and max(w)<=.95 and median(w)<1.,'rolling solve/activation')
        require(m['async_coverage'] and m['rolling_verified'] and m['causal_map_verified'],'rolling and map audit')
        require(m['external_lower_m']>=.062 and m['hazard_lower_m']>=.005 and m['fit_lower_m']>=0 and m['contact_steps']==0,'pilot signed clearances')
        require(m['task_depth_dwell_s']>=1.-1e-7 and m['motion_to_inspect_s'] is not None and 24.<=m['motion_to_inspect_s']<=34. and m['abnormal_stall_s']==0 and m['planner_pause_s']==0,'pilot motion, dock or stall')
    # The same first development seed supplies all seven physically matched contrasts.
    seed=DEV[0];full=runs['coupled',seed][1]
    for strategy in METHODS:
        v=runs[strategy,seed][1]
        require(v['causal_map_verified'] and v['synchronized_rgbd'] and v['native_logs_verified'],'missing comparison evidence')
    u=runs['unregularized',seed][1];stop=runs['stop',seed][1];hard=runs['hard',seed][1]
    require(full['position_tracking_peak_m']<u['position_tracking_peak_m'] or full['thrust_diff_rms_N']<u['thrust_diff_rms_N'],'coupling contrast')
    stop_duration=stop['motion_to_inspect_s'] if stop['motion_to_inspect_s'] is not None else stop['elapsed_motion_observation_s']
    require(stop['planner_pause_s']>0 and stop_duration>full['motion_to_inspect_s'],'stop-and-go contrast')
    require(hard['handoff_accel_jump_m_s2']>full['handoff_accel_jump_m_s2'] or hard['handoff_brate_jump_rad_s']>full['handoff_brate_jump_rad_s'],'hard-switch contrast')
    for strategy in ('straight','fixed','sequential'):
        m=runs[strategy,seed][1]
        require(m['planning_diagnosis'] and (not m['success'] or m['external_lower_m']<full['external_lower_m']),'restricted geometry contrast: '+strategy)
    relative=lambda p:os.path.relpath(p,out.parent)
    evaluation={name:digest(ROOT/'source'/name) for name in ('m2_score.py','m2_rolling_audit.py')}
    require(all(m['evaluation_sources']==evaluation for f,m in runs.values()),'pilot results must use the same independent evaluation')
    result=dict(kind='qualified',identity=ident,evaluation_sources=evaluation,static_receipt=relative(static),static_sha256=digest(static),development=[
        dict(method=k[0],seed=k[1],path=relative(f),sha256=digest(f)) for k,(f,m) in sorted(runs.items())],methods=list(METHODS),formal_seeds=list(FORMAL))
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():require(read(out)==result,'existing pilot qualification is immutable')
    else:out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('static',type=Path);p.add_argument('output',type=Path);p.add_argument('metrics',type=Path,nargs='+')
    a=p.parse_args();certify(a.static,a.metrics,a.output)
