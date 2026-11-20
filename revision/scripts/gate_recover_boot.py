"""Preserve an interrupted acquisition from a prior Linux boot and resume the registered matrix."""
import fcntl
import hashlib
import json
from pathlib import Path
import time

from gate_feedback_campaign import design
from gate_feedback_spec import OUT,dump


def previous_boot():
    locked=design()
    progress=OUT/'campaign-progress.json'
    if not progress.is_file():return
    p=json.loads(progress.read_bytes());name=p.get('active')
    allowed={c['case'] for c in locked['cases']}
    if name is None:return
    if name not in allowed:raise ValueError('Unknown active flight')
    results=json.loads((OUT/'decisions/metrics.json').read_bytes())
    if name in results['cases']:return
    folder=(OUT/'formal'/name).resolve()
    if folder.parent!=(OUT/'formal').resolve() or not folder.exists() or (folder/'run.json').exists():return
    started=int(next(line.split()[1] for line in Path('/proc/stat').read_text().splitlines() if line.startswith('btime ')))
    files=[m for m in folder.rglob('*') if not m.is_symlink() and m.is_file()]
    last=max((p.stat().st_mtime for p in files),default=folder.stat().st_mtime)
    if last>=started:
        raise RuntimeError('Incomplete flight belongs to current boot; keep it for direct inspection')
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime(started))
    root=(OUT/'interrupted').resolve();root.mkdir(exist_ok=True)
    dest=(root/(name+'-before-'+stamp)).resolve()
    if dest.parent!=root or dest.exists():raise RuntimeError('Interruption archive collision')
    hashes={}
    for rel in ('flight.bag','hil.csv','progress.json','protocol.json','model.xml','provenance.json',
                'optics/index.jsonl','optics/native.csv','optics/optical.jsonl'):
        f=folder/rel
        if f.exists():hashes[rel]=hashlib.sha256(f.read_bytes()).hexdigest()
    info=dict(case=name,reason='Host reboot interrupted image acquisition before the PX4/MuJoCo run closed.',
              design_commit=locked['design_commit'],completed_before_restart=len(results['cases']),
              current_boot_id=boot,current_boot_utc=stamp,last_write_utc=time.strftime('%FT%TZ',time.gmtime(last)),
              rgb_images=len(list((folder/'optics/pixels').glob('r*.jpg'))),
              depth_images=len(list((folder/'optics/pixels').glob('d*.png'))),
              retained_files=len(files),total_bytes=sum(v.stat().st_size for v in files),hashes=hashes,
              retained_folder=dest.relative_to(OUT).as_posix())
    folder.rename(dest)
    dump(dest/'interruption.json',info)
    dump(OUT/'interruptions'/(name+'-'+stamp+'.json'),info)
    print('Retained interrupted acquisition',dest,'RGB-D frames',info['rgb_images'],flush=True)


if __name__=='__main__':
    with (OUT/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous_boot()
