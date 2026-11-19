"""Causal 50 Hz strain and delayed pixel-pose fusion. No command or mocap enters fuse()."""
import argparse
import json
from types import SimpleNamespace

import cv2
import mujoco
import numpy as np

from fiducials import FiducialLocalization
from estimation import orientation_shape
from round_shape_pixels import OUTPUT,pixel_rig,matrix
from round_shape_sensors import vector_angles,wrap_deg,features,predict
from round_shape_theta_reference import run_ekf
from sensors import RGBDSensor

DEFAULTS=SimpleNamespace(strain_sigma=3.,vision_sigma=2.2,process_accel_sigma=18.,process_theta_sigma=.08,
                         consistency_gate_deg=7.,reject_vision_diff_deg=13.,min_vision_confidence=.08,
                         rejected_vision_confidence=.02)


def bend_vectors(theta,phi):
    return np.deg2rad(theta)[:,None]*np.column_stack([np.cos(np.deg2rad(phi)),np.sin(np.deg2rad(phi))])


def deliver(t,packets,blackout):
    # The capture clock is motion time. Exactly one visual update per delivered packet.
    incoming=np.full((len(t),2),np.nan);held=incoming.copy()
    quality=np.zeros(len(t));age=np.full(len(t),np.nan);fresh=np.zeros(len(t),bool)
    j=0;latest=None
    for i,now in enumerate(t):
        while j<len(packets) and packets[j,1]+.2<=now+1e-7:
            row=packets[j];j+=1
            blocked=blackout is not None and blackout[0]<=row[1]<blackout[1]
            if blocked or not row[4]:
                latest=None
                continue
            latest=row
            incoming[i]=row[-2:];fresh[i]=True
            # Pose is already gated on depth consistency and reprojection; account for quality and capture age.
            quality[i]=(np.clip((row[5]-12)/90,0,1)*np.exp(-(row[6]/1.5)**2-(now-row[1])/.4))**(1/3)
        if latest is not None and now-latest[1]<=.41+1e-7:
            held[i]=latest[-2:];age[i]=max(0.,now-latest[1])
    return incoming,held,quality,age,fresh


def fuse(t,strain_b,packets,phi_sigma,blackout=None):
    incoming,held,quality,age,fresh=deliver(t,packets,blackout)
    ts,ps=vector_angles(strain_b);tv,pv=vector_angles(incoming)
    legacy=run_ekf(t,ts,tv,fresh,DEFAULTS)
    # Manuscript angular observation: inverse-variance weighting on the unit circle,
    # wrapped innovations in a constant-rate model and no new azimuth near straight.
    phi=np.zeros(len(t));plane_weight=np.zeros(len(t))
    state=np.zeros(2);P=np.diag([180.**2,40.**2]);I=np.eye(2)
    for i,now in enumerate(t):
        dt=.02 if not i else now-t[i-1]
        F=np.array([[1.,dt],[0.,1.]])
        a=np.array([.5*dt*dt,dt])
        state=F@state;P=F@P@F.T+18.**2*np.outer(a,a)+np.diag([.08**2,1e-5])
        ws=(1./phi_sigma**2 if ts[i]>=5 else 0.)
        wv=(quality[i]/2.2**2 if fresh[i] and tv[i]>=5 else 0.)
        if ws+wv>0.:
            obs=ws*np.exp(1j*np.deg2rad(ps[i]))
            if wv:obs+=wv*np.exp(1j*np.deg2rad(pv[i]))
            if abs(obs)>1e-7*(ws+wv):
                z=np.rad2deg(np.angle(obs));R=1./(ws+wv);K=P[:,0]/(P[0,0]+R)
                state+=K*wrap_deg(z-state[0])
                H=np.array([1.,0.]);A=I-np.outer(K,H)
                P=A@P@A.T+np.outer(K,K)*R
        phi[i]=wrap_deg(state[0]);plane_weight[i]=wv/(ws+wv) if ws+wv else 0.
    return dict(strain=strain_b,vision=held,fusion=bend_vectors(np.maximum(legacy['theta'],0.),phi),
                fresh=fresh,vision_age_s=age,w_theta=legacy['vision_weight'],w_phi=plane_weight)


def bench_pixels():
    from round_g1_model import make_plant
    from round_shape_bench import errors
    rig=pixel_rig();m,d,ub,tc,items=rig
    native=make_plant();ids=native.arm.qpos_ids.copy()
    cam=RGBDSensor(m,'wrist_rgbd',480,360)
    try:
        model=json.loads((OUTPUT/'bench/predictor.json').read_bytes())
        summary={}
        for f in sorted((OUTPUT/'bench').glob('test-*.npz')):
            raw=np.load(str(f));t=raw['t']
            loc=FiducialLocalization(np.eye(4),refine=True,robust_depth=True,items=items)
            WU=matrix(np.eye(3),np.array([0.,0.,2.1]))
            rows=[];snapshots=[]
            for j in range(0,len(t),10):
                d.qpos[:]=m.qpos0;d.qpos[:3]=WU[:3,3];d.qpos[ids]=raw['plant_joint'][j]
                d.time=t[j];mujoco.mj_forward(m,d)
                packet=cam.read(d);vis=loc.update(packet);b=np.full(2,np.nan)
                if vis.valid:b=orientation_shape(np.linalg.solve(WU@ub,vis.T_WC),tc)
                rows.append([j,t[j],int(round(1e6*t[j])),int(round(1e6*t[j])),vis.valid,vis.inliers,vis.reprojection_px,*b])
                if j%100==0:snapshots.append(cv2.resize(packet.rgb,(240,180)))
            rows=np.array(rows);sb=predict(model,features(raw['counts'],t)[0])
            est=fuse(t,sb,rows,model['strain_phi_sigma_deg'])
            out=OUTPUT/'bench-vision';out.mkdir(exist_ok=True)
            np.savez_compressed(out/f.name,t=t,packets=rows,sample_rgb=snapshots,mocap=raw['mocap_b'],**est)
            summary[f.stem]={}
            for name in ('strain','vision','fusion'):
                valid=np.isfinite(est[name]).all(axis=1)
                summary[f.stem][name]=dict(coverage=float(valid.mean()),**errors(est[name][valid],raw['mocap_b'][valid])) if valid.any() else dict(coverage=0.)
        (OUTPUT/'bench-vision/scores.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    finally:cam.close()


if __name__=='__main__':bench_pixels()
