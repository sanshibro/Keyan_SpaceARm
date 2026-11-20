"""Observable covariance finalization, independent of bend truth or commands."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'revision/data/round_augmented_fusion/vision'
def finalize(case,cond):
    dest=OUT/case;p=dest/(cond+'.npz')
    z=dict(np.load(str(p)));diag=json.loads((dest/(cond+'-diagnostics.json')).read_text())
    original=z.get('R_geometry',z['R']).copy();R=original.copy();quality=np.zeros(len(z['t']));ready=np.zeros(len(z['t']),bool)
    lock_at=-1e6;stable=0;last_map=None
    for i,d in enumerate(diag):
        if not z['valid'][i]:stable=0;continue
        if d.get('relocked',False) or (last_map is not None and d['map']!=last_map):lock_at=z['t'][i]
        last_map=d['map']
        depth=float(d.get('depth_ratio',.2));rate=float(d.get('angular_rate_deg_s',0.))
        extent=min(1.,min(np.array(d.get('extent',[75.,48.]))/np.array([260.,180.])))
        # The plane information uses relative support: absolute image coverage is restored here.
        factor=1./max(.08,depth)*(1+(rate/90.)**2)
        if d.get('plane_rank',0)<3:factor*=1.+max(0.,.3-extent)/.3
        factor*=1.+3.*max(0.,1.-(z['t'][i]-lock_at)/.32)
        R[i]=original[i]*factor
        variance=float(np.linalg.eigvalsh(R[i]).max())
        quality[i]=np.clip(np.deg2rad(.85)**2/variance,0,1)
        stable=stable+1 if quality[i]>.20 and d.get('age_s',100)<.3 else 0
        ready[i]=stable>=4
    z.update(R_geometry=original,R=R,quality=quality,ready=ready)
    np.savez_compressed(str(p),**z)
    manifest=dict(frames=len(quality),covariance='geometric information, absolute depth support, unobservable axes, angular rate, causal relock transition',
       invalid_updates=int((~z['valid']).sum()),ready=int(ready.sum()),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       output_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    (dest/(cond+'-ready.json')).write_text(json.dumps(manifest,indent=2)+'\n')
    print('READY',case,cond,manifest,flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--cases',nargs='+',required=True);a.add_argument('--conditions',nargs='+',required=True);v=a.parse_args()
    for case in v.cases:
        for cond in v.conditions:finalize(case,cond)

