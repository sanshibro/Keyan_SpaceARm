"""Truncate sealed native outputs, pixels and arrived PX4 records; audit the deterministic optical layer."""
import argparse,json,os
from pathlib import Path
import numpy as np
import round_aug_vis as vis
from round_aug_vis_cov import finalize

def audit(case,cond,n):
    original=vis.OUT/'vision_inputs'/case/cond
    prefix=vis.OUT/'vision_prefix_inputs'/case/(cond+'-'+str(n));prefix.mkdir(parents=True,exist_ok=True)
    lines=(original/'frames.txt').read_text().splitlines()[:n];last_t=float(lines[-1].split()[0])
    for line in lines:
        for name in line.split()[1:]:
            if not (prefix/name).exists():os.link(original/name,prefix/name)
    (prefix/'frames.txt').write_text('\n'.join(lines)+'\n')
    if not (prefix/'settings.yaml').exists():os.link(original/'settings.yaml',prefix/'settings.yaml')
    sourcecsv=vis.OUT/'vision'/case/(cond+'-native.csv')
    csv=prefix/'sealed-native-prefix.csv';csv.write_text('\n'.join(sourcecsv.read_text().splitlines()[:n+1])+'\n')
    flight=vis.sources(case)[2];px4=prefix/'px4';px4.mkdir(exist_ok=True)
    rows=[json.loads(s) for s in (flight/'timing.jsonl').read_text().splitlines()]
    rows=[dict(t=e['t'],rpy_frd_ned=e['rpy_frd_ned']) for e in rows
          if e.get('t') is not None and 'rpy_frd_ned' in e and e['t']<=last_t+.2+1e-8]
    (px4/'timing.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in rows))
    original_materialize=vis.materialize
    vis.materialize=lambda c,k:(prefix,csv,px4)
    try:vis.process(case,cond,n)
    finally:vis.materialize=original_materialize
    finalize(case,cond+'-prefix')
    folder=vis.OUT/'vision'/case
    full=np.load(str(folder/(cond+'.npz')));part=np.load(str(folder/(cond+'-prefix.npz')))
    result={}
    for key in ['t','b','R','valid','locked','ready','quality']:
        a=full[key][:n];b=part[key];same=np.array_equal(a,b,equal_nan=True)
        finite=np.isfinite(a)&np.isfinite(b)
        result[key]=dict(exact_equal=bool(same),max_abs_difference=float(np.max(abs(a[finite].astype(float)-b[finite].astype(float)))) if finite.any() else 0.)
    report=dict(case=case,condition=cond,frames=n,last_exposure_s=last_t,
       px4_latest_available_s=float(max(e['t'] for e in rows)),exposure_delivery_delay_s=.2,
       passed=all(r['exact_equal'] for r in result.values()),fields=result,
       scope='sealed online native ORB CSV -> RGB-D anchors, orientation, covariance and ready; native multithreaded tracker is not rerun',
       prefix_frames_sha256=vis.sha(prefix/'frames.txt'),prefix_native_sha256=vis.sha(csv),
       prefix_px4_sha256=vis.sha(px4/'timing.jsonl'),future_pixel_files_linked=False,
       prefix_source_directory=str(prefix.relative_to(vis.ROOT)))
    vis.savej(folder/(cond+'-prefix-audit.json'),report);print('PREFIX',json.dumps(report),flush=True)
    assert report['passed']
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--case',default=vis.PILOT);a.add_argument('--conditions',nargs='+',default=['natural','occluded']);a.add_argument('--n',type=int,default=650);v=a.parse_args()
    for cond in v.conditions:audit(v.case,cond,v.n)

