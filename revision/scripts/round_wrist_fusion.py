"""Same-clock, marker-free comparison. Ground truth is opened only by the evaluator."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation,Slerp
from gate_shape_timing import enu_body_rotation
from round_g1_model import make_plant
from round_shape_pixels import OUTPUT,EVIDENCE,calibration,matrix
from round_shape_sensors import features,predict,vector_angles,wrap_deg
from estimation import orientation_shape

DELAY=.2
DT=.02
Q_ACCEL=18.
METHODS=('strain','strain_filter','slam','fusion')
PHASES={'all':(14,46),'before':(14,31.2),'during':(31.2,35.2),'after':(35.2,46)}
CASES=[str(seed)+'-'+policy+'_pause' for seed in (241119,241120) for policy in ('clock','online')]
def readj(path):return json.loads(path.read_bytes())
def savej(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def as_b(theta,phi):
    return np.deg2rad(theta)[...,None]*np.stack((np.cos(np.deg2rad(phi)),np.sin(np.deg2rad(phi))),axis=-1)

def visual_packets(case,csv,plant,evidence=EVIDENCE):
    tab=pd.read_csv(csv);t=tab.t.to_numpy()
    assert np.allclose(np.diff(t),.04,atol=1e-7)
    folder=evidence/'runs'/case
    events=[json.loads(s) for s in (folder/'timing.jsonl').read_text().splitlines()]
    events=[e for e in events if 'rpy_frd_ned' in e and e.get('t') is not None]
    stamps=np.array([e['t'] for e in events])
    bodyrot=np.stack([enu_body_rotation(e['rpy_frd_ned']) for e in events])
    # Both bracketing PX4 packets have arrived by the delayed RGB-D update.
    k=np.searchsorted(stamps,t);assert np.all(stamps[k]<=t+DELAY+1e-6)
    base=Slerp(stamps,Rotation.from_matrix(bodyrot))(t).as_matrix()
    ub,tc=calibration(plant)
    home_bt=plant.d.body('mount').xmat.reshape(3,3).T@plant.d.body('tool').xmat.reshape(3,3)
    candidate=(tab.state.to_numpy()==2)&(tab.matched.to_numpy()>=15)&(tab.reproj_px.to_numpy()<=3)&(tab.map_id.to_numpy()>=0)
    ready=np.flatnonzero(candidate&(t<13))
    b=np.full((len(t),2),np.nan);confidence=np.zeros(len(t))
    if not len(ready):return t,b,confidence,dict(anchored=False)
    first=ready[0];anchor=int(tab.map_id[first]);good=candidate&(tab.map_id.to_numpy()==anchor)
    pose=tab[['r'+str(i) for i in range(12)]].to_numpy().reshape(-1,3,4)[:,:,:3]
    # Neutral CAD assembly and vehicle attitude set the world-to-map alignment once.
    # No fitted trajectory / ground-truth orientations are used for alignment or resetting.
    R_WM=base[first]@ub[:3,:3]@home_bt@tc[:3,:3]@pose[first]
    for i in np.flatnonzero(good):
        R_BC=(base[i]@ub[:3,:3]).T@R_WM@pose[i].T
        b[i]=orientation_shape(matrix(R_BC),tc)
    n=tab.matched.to_numpy();rp=tab.reproj_px.to_numpy()
    confidence[good]=np.clip(n[good]/100.,.05,1.)*np.exp(-.5*(rp[good]/1.5)**2)
    return t,b,confidence,dict(anchored=True,initial_map=anchor,anchor_s=float(t[first]),state_ok=int((tab.state==2).sum()),
                              original_map_accepted=int(good.sum()),new_map_frames=int((candidate&~good).sum()),
                              processing_median_ms=float(np.median(tab.track_ms)),processing_p95_ms=float(np.percentile(tab.track_ms,95)))

def filter_bend(t,bs,capture,bv,quality,sigma,strain_fresh=None):
    """Update at exposure, replay the intervening strain samples, release only the current output."""
    ts,ps=vector_angles(bs)
    strain_fresh=np.ones(len(t),dtype=bool) if strain_fresh is None else np.asarray(strain_fresh,dtype=bool)
    if strain_fresh.shape!=(len(t),):raise ValueError('Incorrect strain clock length')
    tv,pv=vector_angles(bv)
    ci=np.searchsorted(t,capture-1e-7);di=np.searchsorted(t,capture+DELAY-1e-7)
    assert np.allclose(t[ci],capture,atol=1e-6)
    delivered={int(d):(int(c),j) for j,(c,d) in enumerate(zip(ci,di)) if d<len(t)}
    seen={};n=len(t);x=np.zeros((n,2,2));P=np.zeros((n,2,2,2));has_plane=np.zeros(n,bool)
    output=np.empty((n,2));direction=np.zeros(n,bool);wv_now=np.zeros(n)
    f=np.array([[1.,DT],[0.,1.]])
    a=np.array([.5*DT**2,DT]);Q=Q_ACCEL**2*np.outer(a,a)+np.diag([.08**2,1e-5])
    assert np.allclose(np.diff(t),DT,atol=1e-7)
    for i in range(n):
        start=i
        if i in delivered:
            c,j=delivered[i]
            if quality[j]>0 and np.isfinite(tv[j]):
                seen[c]=j;start=c
                present_precision=(1./sigma[0]**2 if strain_fresh[c] else 0.)
                wv_now[i]=(quality[j]/2.2**2)/(present_precision+quality[j]/2.2**2)
        for k in range(start,i+1):
            if k:
                xk=x[k-1]@f.T;pk=f@P[k-1]@f.T+Q;plane=has_plane[k-1]
            else:
                xk=np.array([[ts[0],0.],[ps[0],0.]])
                pk=np.array([np.diag([25.,25.]),np.diag([180.**2,40.**2])]);plane=False
            j=seen.get(k)
            ws=np.array([1./sigma[0]**2 if strain_fresh[k] else 0.,0.])
            if strain_fresh[k] and ts[k]>=5:ws[1]=(max(.12,np.sin(np.deg2rad(ts[k])))/sigma[1])**2
            wv=np.zeros(2)
            if j is not None:
                wv[0]=quality[j]/2.2**2
                if tv[j]>=5:wv[1]=wv[0]*max(.12,np.sin(np.deg2rad(tv[j])))**2
            obs=np.array([ts[k],ps[k]])
            if wv[0]:obs[0]=(ws[0]*ts[k]+wv[0]*tv[j])/(ws[0]+wv[0])
            if wv[1]:
                resultant=ws[1]*np.exp(1j*np.deg2rad(ps[k]))+wv[1]*np.exp(1j*np.deg2rad(pv[j]))
                if abs(resultant)>1e-8*(ws[1]+wv[1]):obs[1]=np.rad2deg(np.angle(resultant))
                else:ws[1]=wv[1]=0.
            w=ws+wv
            if w[1] and not plane:
                xk[1]=[obs[1],0.];plane=True
            for h in (0,1):
                if not w[h]:continue
                R=1./w[h];gain=pk[h,:,0]/(pk[h,0,0]+R)
                delta=obs[h]-xk[h,0]
                if h:delta=wrap_deg(delta)
                xk[h]+=gain*delta
                A=np.eye(2);A[:,0]-=gain
                pk[h]=A@pk[h]@A.T+np.outer(gain,gain)*R
            x[k],P[k],has_plane[k]=xk,pk,plane
        theta=np.clip(x[i,0,0],0.,120.)
        direction[i]=has_plane[i] and theta>=5.
        angle=x[i,1,0] if direction[i] else ps[i]
        output[i]=as_b(theta,angle)
    return output,direction,wv_now

def evaluate(case,condition,plant):
    dest=OUTPUT/'slam/textured'/case
    model=readj(OUTPUT/'bench/predictor.json')
    val=readj(OUTPUT/'bench/scores.json')['validation']
    # Fixed from whole validation trajectories before SLAM scoring.
    sigma=np.array([max(.5,val['theta_rmse_deg']),max(1.,val['phi_rmse_deg'])])
    with np.load(str(OUTPUT/'runs'/case/'observations.npz')) as z:
        t=z['t'];bs=predict(model,features(z['strain_adc'],t)[0])
    cap,bv,q,diag=visual_packets(case,dest/(condition+'.csv'),plant)
    last=np.searchsorted(cap+DELAY,t+1e-7,side='right')-1
    use=(last>=0)&(t-cap[np.maximum(0,last)]<DELAY+.061)
    v=np.full_like(bs,np.nan);v[use]=bv[last[use]]
    sf,sp,_=filter_bend(t,bs,cap,bv,np.zeros_like(q),sigma)
    fu,fp,w=filter_bend(t,bs,cap,bv,q,sigma)
    b=np.stack([bs,sf,v,fu]);theta,phi=vector_angles(b)
    identifiable=np.stack([theta[0]>=5,sp,theta[2]>=5,fp])
    # The scorer alone opens the independent virtual optical marker data.
    truth=np.load(str(OUTPUT/'runs'/case/'scoring_truth.npz'))['mocap_b']
    check=np.searchsorted(t,cap-1e-7)
    np.savez_compressed(dest/(condition+'-estimate.npz'),t=t,b=b,truth=truth,identifiable=identifiable,
                        methods=METHODS,vision_weight_on_delivery=w,capture=cap,visual_b=bv,truth_at_capture=truth[check],quality=q)
    valid=np.flatnonzero((cap>=35)&(q>0))
    diag['first_valid_after_restore_s']=float(cap[valid[0]]-35) if len(valid) else None
    diag['raw_valid_in_occlusion']=int(np.sum((cap>=31)&(cap<35)&(q>0)))
    if condition=='occluded':
        assert diag['raw_valid_in_occlusion']==0
        assert np.all(w[(t>=31.22)&(t<35.20)]==0)
    return diag

def metrics(b,ref,ident):
    n=len(ref);good=np.isfinite(b).all(1);ts,ps=vector_angles(b);tr,pr=vector_angles(ref)
    eligible=tr>=5;plane=good&eligible&ident
    def rmse(delta):return None if not len(delta) else float(np.sqrt(np.mean(delta**2)))
    return dict(total=n,n=int(good.sum()),coverage=float(good.mean()) if n else None,
                theta_rmse_deg=rmse(ts[good]-tr[good]),vector_rmse_deg=rmse(np.rad2deg(np.linalg.norm(b[good]-ref[good],axis=1))),
                phi_rmse_deg=rmse(wrap_deg(ps[plane]-pr[plane])),phi_n=int(plane.sum()),phi_truth_eligible=int(eligible.sum()))

def summarize():
    output={'cases':CASES,'phase_s':PHASES,'conditions':{},'per_case':{}}
    for cond in ('natural','occluded'):
        parts=[dict(np.load(str(OUTPUT/'slam/textured'/case/(cond+'-estimate.npz')))) for case in CASES]
        phases={}
        for phase,(lo,hi) in PHASES.items():
            sets=[(z['t']>=lo)&(z['t']<hi) for z in parts]
            ref=np.concatenate([z['truth'][s] for z,s in zip(parts,sets)])
            methods={}
            all_visible=np.concatenate([np.isfinite(z['b'][2,s]).all(1) for z,s in zip(parts,sets)])
            for h,method in enumerate(METHODS):
                b=np.concatenate([z['b'][h,s] for z,s in zip(parts,sets)])
                ident=np.concatenate([z['identifiable'][h,s] for z,s in zip(parts,sets)])
                methods[method]=dict(all=metrics(b,ref,ident),shared_visual=metrics(b[all_visible],ref[all_visible],ident[all_visible]))
            phases[phase]=methods
        output['conditions'][cond]=phases
        for case,z in zip(CASES,parts):
            sel=(z['t']>=14)&(z['t']<46)
            cap=z['capture']>=14
            measure=metrics(z['visual_b'][cap],z['truth_at_capture'][cap],vector_angles(z['visual_b'][cap])[0]>=5)
            output['per_case'].setdefault(case,{})[cond]=dict(at_capture=measure,all={
                m:metrics(z['b'][h,sel],z['truth'][sel],z['identifiable'][h,sel]) for h,m in enumerate(METHODS)},
                tracking=readj(OUTPUT/'slam/textured'/case/(cond+'-diagnostics.json')))
    savej(OUTPUT/'slam/markerless-scores.json',output)
    for cond in ('natural','occluded'):
        for phase in ('all','during','after'):
            print(cond,phase,[(m,output['conditions'][cond][phase][m]['all']['theta_rmse_deg'],
                              output['conditions'][cond][phase][m]['all']['coverage']) for m in METHODS],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cases',nargs='*',default=CASES);p.add_argument('--conditions',nargs='+',default=['natural','occluded'])
    p.add_argument('--summary',action='store_true');a=p.parse_args();plant=make_plant()
    for case in a.cases:
        for cond in a.conditions:
            out=OUTPUT/'slam/textured'/case
            done=readj(out/(cond+'.json'));assert done['exit']==0 and done['processed']==done['expected']
            diag=evaluate(case,cond,plant);savej(out/(cond+'-diagnostics.json'),diag);print(case,cond,diag,flush=True)
    if a.summary:summarize()
