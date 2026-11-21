"""Verify native flight/scorer provenance and deterministic paired sensor inputs, without running a flight."""
import argparse,hashlib,inspect,json
from pathlib import Path
import numpy as np
import m2_flight as flight
import m2_score
from m2_gate import authorize,DEV,FORMAL,LOCK
from gate_live_shape import FrozenAssets,StreamingBox


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def check(receipt,record):
    if flight.verify_frozen()!=LOCK:raise RuntimeError('Mismatched perception frontend')
    reg=authorize('coupled',DEV[0],True,receipt)
    try:authorize('coupled',FORMAL[0],False,receipt)
    except ValueError:rejected=True
    else:raise RuntimeError('Formal launch accepted a static-only receipt')
    assets=FrozenAssets()
    def waveform(seed):
        box=StreamingBox(assets.spec,assets.profile,assets.noise,str(seed),'nominal',4.)
        return dict(timestamps=digest(box.all_t),readout=digest(box.all_noise),samples=len(box.all_t),
                    gains=digest(box.h),after_support=digest(np.random.default_rng([seed,1]).normal(size=64)))
    seeds={str(seed):waveform(seed) for seed in DEV+FORMAL}
    if len({v['readout'] for v in seeds.values()})!=len(seeds):raise AssertionError('Repeated sensor input')
    if waveform(DEV[0])!=seeds[str(DEV[0])]:raise AssertionError('Seed is not reproducible')
    source=inspect.getsource(flight.run)
    for term in ('default_rng(noise_seed)','default_rng([noise_seed,1])','translation_feedforward','2048','yaw_enu'):
        if term not in source:raise RuntimeError('Unexpected native PX4 path: '+term)
    run=json.loads((record/'run.json').read_text())
    bag=m2_score.logged_inputs(record,run)
    if not bag:raise RuntimeError('Missing raw native flight logs')
    with np.load(record/'physics.npz') as s,np.load(record/'control.npz') as c:
        stamps=run['release_us']+np.rint(c['t']*1e6).astype(np.int64)
        if not np.array_equal(s['us'][np.searchsorted(s['us'],stamps)],stamps):raise AssertionError('Clock alignment')
        ticks=np.unique(np.diff(s['us']),return_counts=True)
    return dict(development_receipt=reg['static_sha256'],scene=reg['identity']['scene'],frozen=LOCK,
        formal_rejects_development_receipt=rejected,paired_noise_example_digests=seeds,clock_intervals_us=list(zip(ticks[0].tolist(),ticks[1].tolist())),
        reference_native_bag_and_ULog=bag,reference_run=run['tag'],controller=inspect.getsourcefile(flight.fly),
        hil_sender=inspect.getsourcefile(flight.run),native_sensor_source=inspect.getsourcefile(StreamingBox))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('receipt',type=Path);p.add_argument('flight',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();a.output.write_text(json.dumps(check(a.receipt,a.flight),indent=2)+'\n')
