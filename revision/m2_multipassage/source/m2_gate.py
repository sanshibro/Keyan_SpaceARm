"""Release the industrial M2 campaign only from measured, reproducible pilot evidence."""
import argparse
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
    source=ROOT/'source'
    runtime=[p for p in source.glob('m2_*.py') if p.stem.startswith(('m2_scene','m2_plant','m2_geometry','m2_rolling','m2_planner','m2_handoff','m2_servo','m2_flight','m2_self_registry','m2_gate','m2_score')) and 'preflight' not in p.stem]
    return dict(source={p.name:digest(p) for p in sorted(runtime)},scene=read(SCENE/'manifest.json')['scene_sha256'],
                registry=digest(source/'m2_self_registry.py'),perception=LOCK)
def check_static(path):
    d=read(path);manifest=read(SCENE/'manifest.json')
    require(d['scene_sha256']==manifest['scene_sha256'],'static scene digest')
    require(d['registry_sha256']==digest(ROOT/'source/m2_self_registry.py'),'static self registry digest')
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
        require(len(w)>=3 and m['activated_replans']>=3 and max(w)<=.95 and median(w)<1.,'rolling solve/activation')
        require(m['async_coverage'] and m['rolling_verified'] and m['causal_map_verified'],'rolling and map audit')
        require(m['external_lower_m']>=.062 and m['hazard_lower_m']>=.005 and m['fit_lower_m']>=0 and m['contact_steps']==0,'pilot signed clearances')
        require(m['task_depth_dwell_s']>=1. and 24.<=m['motion_to_inspect_s']<=34. and m['abnormal_stall_s']==0 and m['planner_pause_s']==0,'pilot motion, dock or stall')
    # The same first development seed supplies all seven physically matched contrasts.
    seed=DEV[0];full=runs['coupled',seed][1]
    for strategy in METHODS:
        v=runs[strategy,seed][1]
        require(v['causal_map_verified'] and v['synchronized_rgbd'] and v['native_logs_verified'],'missing comparison evidence')
    u=runs['unregularized',seed][1];stop=runs['stop',seed][1];hard=runs['hard',seed][1]
    require(full['position_tracking_peak_m']<u['position_tracking_peak_m'] or full['thrust_diff_rms_N']<u['thrust_diff_rms_N'],'coupling contrast')
    require(stop['planner_pause_s']>0 and stop['motion_to_inspect_s']>full['motion_to_inspect_s'],'stop-and-go contrast')
    require(hard['handoff_accel_jump_m_s2']>full['handoff_accel_jump_m_s2'] or hard['handoff_brate_jump_rad_s']>full['handoff_brate_jump_rad_s'],'hard-switch contrast')
    for strategy in ('straight','fixed','sequential'):
        m=runs[strategy,seed][1]
        require(m['planning_diagnosis'] and (not m['success'] or m['external_lower_m']<full['external_lower_m']),'restricted geometry contrast: '+strategy)
    relative=lambda p:os.path.relpath(p,out.parent)
    result=dict(kind='qualified',identity=ident,static_receipt=relative(static),static_sha256=digest(static),development=[
        dict(method=k[0],seed=k[1],path=relative(f),sha256=digest(f)) for k,(f,m) in sorted(runs.items())],methods=list(METHODS),formal_seeds=list(FORMAL))
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():require(read(out)==result,'existing pilot qualification is immutable')
    else:out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('static',type=Path);p.add_argument('output',type=Path);p.add_argument('metrics',type=Path,nargs='+')
    a=p.parse_args();certify(a.static,a.metrics,a.output)
