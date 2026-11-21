"""Read the saved native model for figure generation on another workstation."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,os
import numpy as np
import mujoco
def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def writej(path,obj):Path(path).write_text(json.dumps(clean(obj),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def option():
    o=mujoco.MjvOption();o.geomgroup[:]=[1,1,1,0,0,0];o.sitegroup[:]=0;return o
def build():
    replay=os.environ.get('M1_REPLAY_DIR')
    if not replay:
        from m1_plant import build as native
        return native()
    root=Path(replay);meta=json.loads((root/'native-task.json').read_text())
    path=root/'model.mjb';assert meta['mujoco']==mujoco.__version__
    with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==meta['mjb_sha256']
    relative=os.path.relpath(path)
    if relative.isascii():m=mujoco.MjModel.from_binary_path(relative)
    else:m=mujoco.MjModel.from_binary_path('registered-model.mjb',assets={'registered-model.mjb':path.read_bytes()})
    assert m.ngeom==498 and m.nq==26
    uid=m.body('uav').id
    return SimpleNamespace(m=m,d=mujoco.MjData(m),base=uid,mass=m.body_subtreemass[uid],scene_digest=meta['scene'],task=meta['task'])
