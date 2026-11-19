"""Replay the new flights' actual wrist RGB-D streams through native ORB-SLAM3."""
from pathlib import Path
import numpy as np
import pandas as pd
from round_health_flights import OUT,SPEC
from round_g1_model import make_plant
from round_patent_camera import OUT as CALIBRATION
from round_wrist_fusion import readj,visual_packets
from round_wrist_render import run as render
from round_wrist_run import run as track,build
from round_patent_relock import process
from round_patent_inputs import quality

def main():
    folder=OUT/'inputs';optical=OUT/'artifacts'
    source=optical/'slam/textured'
    orb=Path.home()/'桌面/Test/ORB_SLAM3'
    executable=build(orb,Path('/tmp/g1_wrist_orb_20261119'))
    profile=readj(CALIBRATION/'visual-calibration.json')
    plant=make_plant()
    for entry in SPEC['flight']:
        case=entry['case']
        render(case,False,folder,evidence=OUT,output=optical,material_seed=91105)
        dest=OUT/'records'/case;dest.mkdir(parents=True,exist_ok=True)
        for cond in ('natural','occluded'):
            track(orb,folder,case,'textured',cond,output=optical,executable=executable)
            process(case,cond,output=OUT,old=source,input_root=folder/'textured')
            table=OUT/'camera'/case/(cond+'-registered.csv');df=pd.read_csv(table)
            cap,b,old,diag=visual_packets(case,table,plant,evidence=OUT)
            ok=np.isfinite(b).all(1);cv=quality(df,ok,profile)
            repro=df.reproj_px.to_numpy();good=np.zeros(len(cap),bool);count=0
            for j in range(len(cap)):
                recent=repro[max(0,j-4):j+1]
                stable=cv[j]>=profile['teacher_threshold'] and np.isfinite(recent).all() and np.max(recent)<=1.35*profile['teacher_pixel_max']
                count=count+1 if stable else 0;good[j]=count>=4
            bridged=np.load(OUT/'camera'/case/(cond+'-map.npz'))['bridged']
            for i in np.flatnonzero(bridged&~np.r_[False,bridged[:-1]]):good[i:i+8]=False
            np.savez_compressed(dest/(cond+'.npz'),capture=cap,visual_b=b,cv=cv,stable_teacher=good,bridge=bridged,native_q=old)
            print('Visual stream',case,cond,len(cap),int(ok.sum()),flush=True)
if __name__=='__main__':main()
