"""Independent pilot scoring only; held-out flight truth is never accepted."""
from pathlib import Path
import json,numpy as np
ROOT=Path(__file__).resolve().parents[2];case='241118-online_pause'
z=np.load(ROOT/'revision/data/round_augmented_fusion/vision'/case/'natural.npz')
y=np.load(ROOT/'revision/data/round_shape_clearance/runs'/case/'scoring_truth.npz')
ix=np.searchsorted(y['t'],z['t']-1e-7);assert np.max(abs(y['t'][ix]-z['t']))<1e-5
truth=y['mocap_b'][ix];ref=np.linalg.norm(truth,axis=1);est=np.linalg.norm(z['b'],axis=1)
report={}
for name,(lo,hi) in {'all':(14,46),'before':(14,26),'transition':(26,36),'recovered':(36,46)}.items():
 use=(z['t']>=lo)&(z['t']<hi);valid=use&z['valid']
 report[name]={'frames':int(use.sum()),'valid':int(valid.sum()),'theta_rmse_deg':float(np.rad2deg(np.sqrt(np.mean((est[valid]-ref[valid])**2)))), 'vector_rmse_deg':float(np.rad2deg(np.sqrt(np.mean(np.sum((z['b'][valid]-truth[valid])**2,axis=1)))))}
print(json.dumps(report,indent=2));(ROOT/'revision/data/round_augmented_fusion/vision'/case/'pilot-score.json').write_text(json.dumps(report,indent=2))
