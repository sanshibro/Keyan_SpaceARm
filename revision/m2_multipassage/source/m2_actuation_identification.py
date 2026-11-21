"""Paired development-only check of actuator rate lag and native yaw-rate delivery."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter


def samples(attempt):
    p=attempt/'coupled-26112207';c=np.load(p/'flight/control.npz');s=np.load(p/'audit/series.npz')
    t=c['t'];actual=s['actual_b'];issued=c['issued_b'];dt=np.median(np.diff(t))
    smooth=savgol_filter(actual,17,3,axis=0);speed=savgol_filter(actual,17,3,axis=0,deriv=1,delta=dt)
    valid=(c['phase']=='execute')&(c['projection']=='nominal')&(np.linalg.norm(speed,axis=1)<np.deg2rad(90.))
    theta=np.linalg.norm(smooth,axis=1)
    design=np.stack((np.ones((len(t),2)),smooth,np.column_stack((smooth[:,0]*theta,smooth[:,1]*theta)),speed),axis=2)
    return design[valid],(issued-smooth)[valid],speed[valid]


def lag(train,test):
    X,y,_=samples(train);H,z,v=samples(test);checks=[]
    for k in range(2):
        xc=X[:,k];yc=y[:,k]
        w=np.ones(len(xc));coef=np.zeros(4)
        for iteration in range(8):
            sw=np.sqrt(w);coef=np.linalg.lstsq(xc*sw[:,None],yc*sw,rcond=None)[0]
            e=yc-xc@coef;w=np.minimum(1.,np.deg2rad(1.5)/np.maximum(1.e-6,abs(e)))
        static=np.linalg.lstsq(xc[:,:3],yc,rcond=None)[0]
        mv=abs(v[:,k])>np.deg2rad(5)
        rmse=lambda e:float(np.rad2deg(np.sqrt(np.mean(e[mv]**2)))) if mv.any() else None
        checks.append(dict(channel=k,training_samples=len(y),test_samples=int(mv.sum()),
            time_constant_s=float(coef[-1]),validation_rms_deg=rmse(z[:,k]-H[:,k]@coef),
            validation_static_rms_deg=rmse(z[:,k]-H[:,k,:3]@static)))
    return dict(fit=train.name,holdout=test.name,channels=checks)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('first',type=Path);ap.add_argument('second',type=Path)
    ap.add_argument('output',type=Path);args=ap.parse_args()
    result=[lag(args.first,args.second),lag(args.second,args.first)]
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
