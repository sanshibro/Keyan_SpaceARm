"""Read-only signal audit and pilot-only accuracy, never held-out truth."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'revision/data/round_augmented_fusion'
CASES=['241118-online_pause','26112011-straight','26112012-left','26112013-right']
CONDITIONS=['natural','graded','occluded']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    report={}
    for case in CASES:
        report[case]={}
        for cond in CONDITIONS:
            d=OUT/'vision'/case;z=np.load(str(d/(cond+'.npz')));t=z['t'];v=z['valid']
            assert len(t)==850 and np.allclose(np.diff(t),.04)
            assert np.array_equal(z['locked'],v)
            assert np.all(np.isfinite(z['b'][v])) and np.all(np.linalg.eigvalsh(z['R'][v])>0)
            assert np.all(z['quality'][~v]==0) and np.all((z['quality']>=0)&(z['quality']<=1))
            assert not z['ready'][~v].any()
            black=(t>=31)&(t<35);rest=np.flatnonzero((t>=35)&v)
            work=OUT/'vision_inputs'/case/cond
            entries=json.loads((work/'pixels.json').read_text())
            assert len(entries)==850
            item=dict(frames=850,valid=int(v.sum()),blackout_valid=int(v[black].sum()) if cond=='occluded' else None,
                      restore_delay_s=float(t[rest[0]]-35) if cond=='occluded' and len(rest) else None,
                      final_npz_sha256=sha(d/(cond+'.npz')),pixels_manifest_sha256=sha(work/'pixels.json'))
            for label,(a,b) in {'clear':(14,25),'partial':(28,31),'recovered':(36,46)}.items():
                ix=(t>=a)&(t<b)&v;item[label+'_quality_median']=float(np.median(z['quality'][ix])) if ix.any() else None
            if cond=='occluded':
                assert item['blackout_valid']==0
                assert sum(e['modified_fraction']==1 for e in entries)==100
                assert item['restore_delay_s']<.6
            if case==CASES[0]:
                truth_path=ROOT/'revision/data/round_shape_clearance/runs'/case/'scoring_truth.npz'
                y=np.load(str(truth_path));ix=np.searchsorted(y['t'],t-1e-7);assert np.max(abs(y['t'][ix]-t))<1e-5
                target=y['mocap_b'][ix];error=np.linalg.norm(z['b'],axis=1)-np.linalg.norm(target,axis=1)
                use=(t>=14)&(t<46)&v
                item['pilot_only_theta_rmse_at_exposure_deg']=float(np.rad2deg(np.sqrt(np.mean(error[use]**2))))
                late=(t>=36)&(t<46)&v;item['pilot_only_recovered_theta_rmse_deg']=float(np.rad2deg(np.sqrt(np.mean(error[late]**2))))
            report[case][cond]=item
    folder=OUT/'vision_sources';folder.mkdir(exist_ok=True)
    for p in [ROOT/'mujoco_closed_loop/anchored_rgbd.py',*list((ROOT/'revision/scripts').glob('round_aug_vis*.py'))]:
        dest=folder/sha(p);dest.mkdir(exist_ok=True);(dest/p.name).write_bytes(p.read_bytes())
    binary=Path('/tmp/g1_wrist_orb_20261119/track');lib=Path.home()/'桌面/Test/ORB_SLAM3/lib/libORB_SLAM3.so'
    result=dict(native_binary_sha256=sha(binary),native_library_sha256=sha(lib),conditions=report,
                calibration='archived fixed relative extrinsics and neutral pose only',held_out_truth_opened=False)
    (OUT/'vision/audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()

