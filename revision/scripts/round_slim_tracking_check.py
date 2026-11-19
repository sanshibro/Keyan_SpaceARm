"""Contrast command tracking with estimation accuracy on the same output ticks."""
import numpy as np
import pandas as pd
from round_shape_pixels import EVIDENCE
from round_slim_box import OUT
from round_wrist_fusion import CASES,readj,savej
def main():
    out={}
    for case in CASES:
        folder=EVIDENCE/'runs'/case;df=pd.read_csv(folder/'hil.csv')
        z=np.load(OUT/'primary'/case/'natural.npz');t=z['t'];angle=np.rad2deg(np.linalg.norm(z['truth'],axis=-1))
        stamp=(df.us.to_numpy()-readj(folder/'run.json')['release_us'])/1e6
        idx=np.searchsorted(stamp,t,side='right')-1;sel=(t>=14)&(t<46);assert np.all(idx[sel]>=0)
        q={'mocap_max_deg':float(angle[sel].max())}
        for name in ('request_deg','issued_deg'):
            e=df[name].to_numpy()[idx[sel]]-angle[sel]
            q[name]={'rmse_deg':float(np.sqrt(np.mean(e**2))),'peak_abs_deg':float(abs(e).max())}
        out[case]=q
    savej(OUT/'tracking-check.json',out);print(out)
if __name__=='__main__':main()
