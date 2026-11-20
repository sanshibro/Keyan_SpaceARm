"""Sensor simulation boundary: export only four surface path lengths and the release clock."""
import hashlib,json
from pathlib import Path
import numpy as np
import mujoco
from round_g1_model import make_plant,ROOT
from slim_sensor import SlimGaugeRig

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def run():
    cfg=json.loads((ROOT/'revision/protocols/round_aug_contract.json').read_bytes())
    specs=json.loads((ROOT/'revision/protocols/round_slim_protocol.json').read_bytes())
    plant=make_plant();m,d=plant.m,plant.d;rig=SlimGaugeRig(m,specs,.3)
    target=ROOT/'revision/data/round_augmented_fusion/geometry';target.mkdir(parents=True,exist_ok=True)
    meta={}
    for case in [cfg['pilot']]+cfg['cases']:
        base='round_g1_online' if case==cfg['pilot'] else 'round_strain_health'
        origin=ROOT/'revision/data'/base/'runs'/case;run_info=json.loads((origin/'run.json').read_bytes())
        with np.load(origin/'physics.npz') as f:
            t=(f['us']-run_info['release_us'])/1e6
            sel=(t>=0)&(t<46.35);t=t[sel];q=f['qpos'][sel]
        data=[]
        for state in q:
            d.qpos[:]=state;mujoco.mj_kinematics(m,d);data.append(rig.lengths(d))
        out=target/(case+'.npz');np.savez_compressed(out,t=t,length_m=np.asarray(data))
        meta[case]=dict(physics_sha256=digest(origin/'physics.npz'),npz_sha256=digest(out),samples=len(t),channels=4)
    (target/'provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('Surface geometry released:',list(meta),flush=True)
if __name__=='__main__':run()
