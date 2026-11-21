"""Compare independent native and portable static certificates for the locked scene."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from m2_scene_lock import verify

def read(p):
    return json.loads(p.read_text(encoding='utf8'))

def fingerprint(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def compare(root,native):
    qa=root/'qa/static';local=qa/'certified.json';a,b=map(read,(local,native))
    lock=verify(root/'scene',root/'qa/registered-collision-scene.json')
    checks=[];rows=[]
    def require(condition,message):
        if not condition:raise ValueError(message)
        checks.append(message)
    require(b['model_source']=='native build','native simulator model')
    require(a['scene_sha256']==b['scene_sha256']==lock['scene_sha256'],'identical visual and physical scene')
    require(a['registry_sha256']==b['registry_sha256'],'identical self-collision registry')
    for x,y in zip(a['gate_witnesses'],b['gate_witnesses']):
        require(x['gate']==y['gate'] and np.allclose(x['xyz'],y['xyz'],rtol=0,atol=1e-7)
            and x['theta_deg']==y['theta_deg'] and x['phi_deg']==y['phi_deg'],'registered witness '+x['gate'])
        keys=('external_lower_m','hazard_lower_m','fit_lower_m')
        delta=max(abs(x[k]-y[k]) for k in keys)
        require(delta<.00005 and min(x['external_lower_m'],y['external_lower_m'])>=.105,'paired clearance '+x['gate'])
        rows.append(dict(gate=x['gate'],external_m=y['external_lower_m'],hazard_m=y['hazard_lower_m'],fit_m=y['fit_lower_m'],largest_difference_m=delta))
    require(len(rows)==3,'three passage centres')
    for d in (a,b):
        launch=d['launch_preflight'];require(launch['passed'] and launch['zero_physical_contact'],'zero static launch contacts')
    l=a['target_optics'];r=b['target_optics']
    require(abs(l['optical_depth_m']-.168)<1e-6 and abs(r['optical_depth_m']-.168)<1e-6,'168 mm optical datum')
    require(abs(l['native_depth_at_face_m']-r['native_depth_at_face_m'])<.0001 and all(v['depth_center_pass'] for v in [l,r]),'native and local wrist depth agree')
    record=dict(passed=True,checks=checks,scene_sha256=lock['scene_sha256'],registered_solids=lock['identical_registered_collision_geoms'],
        witnesses=rows,native_launch_samples=b['launch_preflight']['sample_count'],native_launch_zero_contact=b['launch_preflight']['zero_physical_contact'],
        native_optics=r,local_optics=l,local_evidence_sha256=fingerprint(local),native_evidence_sha256=fingerprint(native))
    (qa/'native-crosscheck.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
    (qa/'collision-lock.json').write_text(json.dumps(lock,indent=2)+'\n',encoding='utf8')
    return record

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('native',type=Path);a=p.parse_args()
    d=compare(Path(__file__).resolve().parents[1],a.native)
    print(d['scene_sha256'],d['witnesses'])
