"""Cross-check the causal batch pose query against the single-pose convex oracle."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from m2_plant import build
from m2_geometry import Geometry,RangeMap
from m2_body_batch import BodyBatch


def check():
    p=build();g=Geometry(p);sensor=RangeMap(p)
    origin=np.array([2.08,.05,1.98,0.,1.4,0.])
    g.set_snapshot(sensor.reveal(0.,origin))
    states=np.asarray([[2.75,.347,1.937,-55.7,110.,90.],[2.75,.36,1.94,-60.,110.,90.],
                       [2.75,.32,1.94,-60.,110.,90.],[2.9,.39,1.945,65.,90.,90.]])
    states[:,3:]=np.deg2rad(states[:,3:]);batch=BodyBatch(g)
    oracle=[g.convex_map.query(q,groups=(0,),tilt=0.)[0] for q in states]
    value=batch.query(states)
    # Both algorithms produce a lower certificate; the upper witnesses bound their difference.
    for v,ref in zip(value,oracle):
        if not (ref['lower']-1.1e-5<=v<=ref['upper']+1e-8):
            raise AssertionError((v,ref))
    grid=np.repeat(states,72,axis=0);grid[:,2]+=np.tile(np.linspace(-.035,.035,72),4)
    elapsed=[]
    for k in range(4):
        t=time.perf_counter();out=batch.query(grid);elapsed.append(time.perf_counter()-t)
    rng=np.random.default_rng(26112207)
    pool=np.zeros((640,6));pool[:,:3]=rng.uniform([2.63,-.50,1.78],[3.03,.60,2.14],(len(pool),3))
    pool[:,3]=np.deg2rad(rng.choice(np.arange(-180,180,10),len(pool)))
    exact=batch.query(pool);tic=time.perf_counter()
    screened=batch.query(pool,reject_below=.085);screen_time=time.perf_counter()-tic
    if not np.array_equal(screened>=.085,exact>=.085):
        raise AssertionError('An admissible pose was culled by the landmark screen')
    kept=np.isfinite(screened)
    if not np.allclose(screened[kept],exact[kept],atol=1.01e-5,rtol=0.):
        raise AssertionError('Surviving rows must retain their nominal GJK certificate')
    g.set_snapshot(sensor.reveal(0.1,origin))
    try:batch.query(states)
    except ValueError:pass
    else:raise AssertionError('Snapshot identity check failed')
    return dict(scene=p.scene_digest,n=len(grid),oracle=oracle,batch_m=value.tolist(),
                batch_seconds=elapsed,qualifying_count=int(sum(out>=.085)),random_pool=len(pool),
                landmark_seconds=screen_time,random_safes=int(sum(exact>=.085)),screened_out=int(sum(~kept)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
    text=json.dumps(check(),indent=2)+'\n'
    if args.output:args.output.write_text(text)
    print(text)
