"""Generate independent bench motions, acquire four gauge channels, and freeze a strain predictor."""
import argparse
import json
import os
from pathlib import Path

import mujoco
import numpy as np

from arm_command import ArmCommand, MujocoArmBackend
from round_e1_model import build,neutral,ROOT
from round_shape_pixels import OUTPUT
from round_shape_sensors import Channels,features,fit_model,mocap_bend,predict,vector_angles,wrap_deg

SPLITS={'train':[0,45,90,135,180,225,270,315], 'validation':[65,205], 'test':[22.5,127.5,247.5]}


def motion(t,index,split,az):
    speed={'train':1.,'validation':.91,'test':1.13}[split]*(1.+.035*(index%3))
    clock=max(0.,t-.85)/speed
    levels=[0,69+5*(index%5),27,85-3*(index%4),47,0]
    angles=np.deg2rad(az+np.array([0,0,37,-26,18,18]))
    targets=np.deg2rad(levels)[:,None]*np.column_stack((np.cos(angles),np.sin(angles)))
    times=np.array([0.,2.6,5.2,7.7,10.1,12.6])
    n=min(len(times)-2,max(0,np.searchsorted(times,clock,side='right')-1))
    u=np.clip((clock-times[n])/(times[n+1]-times[n]),0.,1.)
    u=u**3*(10.-15.*u+6.*u*u)
    return (1.-u)*targets[n]+u*targets[n+1]


def observe(model,label,heading,index,split):
    data=neutral(model)
    sensor=Channels(model,data,label)
    arm=MujocoArmBackend(model,data,8)
    raw=[];markers=[];joints=[];cmd=[]
    for k in range(801):
        t=.02*k
        vector=motion(t,index,split,heading)
        q=ArmCommand(k,t,float(np.linalg.norm(vector)),float(np.arctan2(vector[1],vector[0])))
        arm.apply(q)
        mujoco.mj_forward(model,data)
        a,marks=sensor.read(data,t)
        raw.append(a);markers.append(marks)
        joints.append(data.qpos[arm.qpos_ids].copy());cmd.append(vector)
        if k<800:
            for _ in range(10):mujoco.mj_step(model,data)
    return dict(t=np.arange(801)*.02,counts=raw,markers=markers,plant_joint=joints,command_b=cmd)


def errors(est,ref):
    theta,phi=vector_angles(est);rt,rp=vector_angles(ref)
    good=(rt>=5)&np.isfinite(phi)
    return {'n':len(ref),'vector_rmse_deg':float(np.rad2deg(np.sqrt(np.mean(np.sum((est-ref)**2,axis=1))))),
            'theta_rmse_deg':float(np.sqrt(np.mean((theta-rt)**2))),
            'phi_rmse_deg':float(np.sqrt(np.mean(wrap_deg(phi[good]-rp[good])**2))) if good.any() else None,
            'phi_n':int(good.sum())}


def main():
    target=OUTPUT/'bench'
    target.mkdir(parents=True,exist_ok=True)
    m,_=build(8,.002)
    # Dedicated free-standing material/vision training rig; no obstacle/contact result taken from it.
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    Xs={};Ys={};results={}
    for split,angles in SPLITS.items():
        for i,az in enumerate(angles):
            label='%s-%02d'%(split,i)
            raw=observe(m,label,az,i,split)
            t=raw['t'];Y=np.stack([mocap_bend(marks) for marks in raw['markers']])
            X,tare=features(raw['counts'],t)
            np.savez_compressed(target/(label+'.npz'),**raw,mocap_b=Y,tare=tare)
            Xs[label]=X;Ys[label]=Y
            print(label,'acquired',len(Y),'frames',flush=True)
    train=[k for k in Xs if k.startswith('train-')]
    model=fit_model([Xs[k] for k in train],[Ys[k] for k in train])
    model['training_sessions']=train
    val=[k for k in Xs if k.startswith('validation-')]
    model['strain_phi_sigma_deg']=max(1.,errors(np.vstack([predict(model,Xs[k]) for k in val]),np.vstack([Ys[k] for k in val]))['phi_rmse_deg'])
    (target/'predictor.json').write_text(json.dumps(model,indent=2,allow_nan=False))
    for split in SPLITS:
        keys=[k for k in Xs if k.startswith(split+'-')]
        results[split]=errors(np.vstack([predict(model,Xs[k]) for k in keys]),np.vstack([Ys[k] for k in keys]))
    (target/'scores.json').write_text(json.dumps(results,indent=2,allow_nan=False))
    print(results,flush=True)


if __name__=='__main__':main()
