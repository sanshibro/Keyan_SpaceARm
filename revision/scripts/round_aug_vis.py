"""Produce causal anchor-checked optical observations from archived RGB-D and PX4 attitude only."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
import cv2,numpy as np,pandas as pd
from scipy.spatial.transform import Rotation,Slerp
from anchored_rgbd import AnchoredRGBD
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'revision/data/round_augmented_fusion'
HEALTH=ROOT/'revision/data/round_strain_health';SHAPE=ROOT/'revision/data/round_shape_clearance'
PILOT='241118-online_pause';CASES=['26112011-straight','26112012-left','26112013-right']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def savej(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sources(case):
    if case==PILOT:return SHAPE/'rgbd-inputs/textured'/case,SHAPE/'slam/textured'/case,ROOT/'revision/data/round_g1_online/runs'/case
    return HEALTH/'inputs/textured'/case,HEALTH/'artifacts/slam/textured'/case,HEALTH/'runs'/case
def frames(folder):
    return [(float(s[0]),folder/s[1],folder/s[2]) for s in [line.split() for line in (folder/'frames.txt').read_text().splitlines()]]
def materialize(case,cond):
    src,old,flight=sources(case);work=OUT/'vision_inputs'/case/cond;work.mkdir(parents=True,exist_ok=True)
    lines=[];records=[]
    for i,(t,rgb,dep) in enumerate(frames(src)):
        fraction=.68*np.clip((t-26.)/2.,0,1) if 26<=t<31 else 0.
        black=cond=='occluded' and 31<=t<35
        fraction=1. if black else fraction if cond!='natural' else 0.
        rp=work/('r%04d.png'%i if fraction else rgb.name);dp=work/dep.name
        if not rp.exists() or not dp.exists():
            if fraction:
                image=cv2.imread(str(rgb));depth=cv2.imread(str(dep),cv2.IMREAD_UNCHANGED);n=round(image.shape[1]*fraction)
                image[:,:n]=0;depth[:,:n]=0
                assert cv2.imwrite(str(rp),image) and cv2.imwrite(str(dp),depth)
            else:
                for new,oldp in ((rp,rgb),(dp,dep)):
                    if not new.exists():os.link(oldp,new)
        lines.append('%.6f %s %s'%(t,rp.name,dp.name))
        records.append(dict(frame=i,t=t,rgb=str(rp.relative_to(ROOT)),depth=str(dp.relative_to(ROOT)),
                            rgb_sha256=sha(rp),depth_sha256=sha(dp),modified_fraction=float(fraction)))
    (work/'frames.txt').write_text('\n'.join(lines)+'\n')
    if not (work/'settings.yaml').exists():os.link(src/'settings.yaml',work/'settings.yaml')
    savej(work/'pixels.json',records)
    dest=OUT/'vision'/case;dest.mkdir(parents=True,exist_ok=True);csv=dest/(cond+'-native.csv')
    if not csv.exists():
        if cond=='natural':
            csv.write_bytes((old/'natural.csv').read_bytes())
        else:
            exe=Path('/tmp/g1_wrist_orb_20261119/track');orb=Path.home()/'桌面/Test/ORB_SLAM3'
            with (dest/(cond+'-native.log')).open('w') as log:
                r=subprocess.run([str(exe),str(orb/'Vocabulary/ORBvoc.txt'),str(work/'settings.yaml'),str(work),'natural',str(csv)],
                                 cwd=work,stdout=log,stderr=subprocess.STDOUT,timeout=420)
            if r.returncode:raise RuntimeError('native ORB failed')
    df=pd.read_csv(csv);assert len(df)==len(records) and np.allclose(df.t,[r['t'] for r in records])
    savej(dest/(cond+'-input.json'),dict(case=case,condition=cond,frames=len(df),native_csv_sha256=sha(csv),
          pixels_manifest=str((work/'pixels.json').relative_to(ROOT)),pixels_manifest_sha256=sha(work/'pixels.json'),
          settings_sha256=sha(work/'settings.yaml'),native_mode='natural; degradation physically materialized before TrackRGBD'))
    return work,csv,flight
def body_rotation(rpy):
    r,p,y=rpy;cr,sr=np.cos(r),np.sin(r);cp,sp=np.cos(p),np.sin(p);cy,sy=np.cos(y),np.sin(y)
    a=np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],[sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]])
    return np.array([[0,1,0],[1,0,0],[0,0,-1]])@a@np.diag([1.,-1.,-1.])
def bend(R):
    tangent=R[:,2];a=np.arctan2(np.linalg.norm(tangent[:2]),tangent[2])
    return a*tangent[:2]/max(1e-12,np.linalg.norm(tangent[:2]))
def process(case,cond,limit=None):
    work,csv,flight=materialize(case,cond);tab=pd.read_csv(csv)
    if limit is not None:tab=tab.iloc[:limit].copy()
    with np.load(str(SHAPE/'pilot'/PILOT/'packets.npz')) as c:
        UB=c['T_UB'][:3,:3].copy();TC=c['T_TC'][:3,:3].copy()
    # Archive stores fixed surveyed extrinsics; no model constructor or physical state is opened.
    ev=[json.loads(s) for s in (flight/'timing.jsonl').read_text().splitlines()]
    ev=[e for e in ev if e.get('t') is not None and 'rpy_frd_ned' in e]
    stamps=np.array([e['t'] for e in ev]);rots=np.stack([body_rotation(e['rpy_frd_ned']) for e in ev])
    t=tab.t.to_numpy();k=np.searchsorted(stamps,t)
    assert np.all(stamps[k]<=t+.200001)
    base=Slerp(stamps,Rotation.from_matrix(rots))(t).as_matrix()
    fs=cv2.FileStorage(str(work/'settings.yaml'),cv2.FILE_STORAGE_READ);f=lambda k:fs.getNode(k).real()
    K=np.array([[f('Camera1.fx'),0,f('Camera1.cx')],[0,f('Camera1.fy'),f('Camera1.cy')],[0,0,1.]])
    estimator=AnchoredRGBD(K,f('RGBD.DepthMapFactor'));fs.release()
    pixels=frames(work);b=np.full((len(tab),2),np.nan);R=np.full((len(tab),2,2),np.nan)
    valid=np.zeros(len(tab),bool);locked=valid.copy();ready=valid.copy();quality=np.zeros(len(tab))
    nativepose=tab[['r'+str(i) for i in range(12)]].to_numpy().reshape(-1,3,4);WM=None;diagnostics=[];stable=0
    for i,row in enumerate(tab.itertuples()):
        cv2.setRNGSeed(31071+i)
        _,rp,dp=pixels[i];im=cv2.imread(str(rp),0);dep=cv2.imread(str(dp),cv2.IMREAD_UNCHANGED)
        feat=estimator.features(im,dep)
        native=np.eye(4);native[:3]=nativepose[i]
        ok=bool(row.state==2 and row.matched>=20 and np.isfinite(row.reproj_px) and row.reproj_px<3. and row.map_id>=0)
        corrected,q,diag=estimator.update(row.t,native,int(row.map_id),ok,feat,i)
        if WM is None and estimator.initial is not None:
            first_t,first_T,_=estimator.initial
            # Calibrated neutral tool axes coincide with mount axes at initialization.
            WM=base[i]@UB@TC@first_T[:3,:3]
        if corrected is not None:
            BT=(base[i]@UB).T@WM@corrected[:3,:3].T@TC.T
            b[i]=bend(BT);quality[i]=q;valid[i]=locked[i]=True
            sigma=np.deg2rad(diag['sigma_deg']);J=np.empty((2,3))
            for a in range(3):
                v=np.zeros(3);v[a]=1e-5;J[:,a]=(bend(Rotation.from_rotvec(v).as_matrix()@BT)-b[i])/1e-5
            if 'cov_map' in diag:
                A=(base[i]@UB).T@WM;cov=A@np.array(diag['cov_map'])@A.T
                R[i]=J@cov@J.T+np.eye(2)*np.deg2rad(.2)**2
            else:R[i]=sigma**2*(J@J.T)+np.eye(2)*np.deg2rad(.15)**2
            stable=stable+1 if q>.12 and diag['age_s']<.25 and diag['angular_rate_deg_s']<80 else 0
            ready[i]=stable>=4
        else:stable=0
        diag.update(ready=bool(ready[i]));diagnostics.append(diag)
        if i%100==0:print(case,cond,i,'valid',int(valid[:i+1].sum()),'bank',len(estimator.bank),flush=True)
    dest=OUT/'vision'/case;name=cond if limit is None else cond+'-prefix'
    np.savez_compressed(dest/(name+'.npz'),t=t,b=b,R=R,valid=valid,locked=locked,ready=ready,quality=quality,
                        native_state=tab.state.to_numpy(),native_map=tab.map_id.to_numpy())
    savej(dest/(name+'-diagnostics.json'),diagnostics)
    meta=dict(case=case,condition=cond,frames=len(t),valid=int(valid.sum()),ready=int(ready.sum()),
              initialized=WM is not None,calibration_source='archived T_UB/T_TC; calibrated neutral tool=mount axes',
              px4_attitude_only=True,uses_simulator_or_scene_or_commands_or_strain=False,
              source_sha256={p.name:sha(p) for p in [Path(__file__),ROOT/'mujoco_closed_loop/anchored_rgbd.py']},
              output_sha256=sha(dest/(name+'.npz')))
    savej(dest/(name+'-summary.json'),meta);print('DONE',meta,flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--cases',nargs='+',default=[PILOT]);a.add_argument('--conditions',nargs='+',default=['natural'])
    a.add_argument('--limit',type=int);a.add_argument('--inputs-only',action='store_true');v=a.parse_args()
    for case in v.cases:
        for cond in v.conditions:
            if v.inputs_only:materialize(case,cond)
            else:process(case,cond,v.limit)

