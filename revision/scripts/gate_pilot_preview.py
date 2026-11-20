"""Release-clock checks from the first instrumented physical flight."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def build(case,output):
    s=json.loads((case/'control-summary.json').read_text())
    c=np.load(case/'control.npz'); w=np.load(case/'physics.npz')
    u=json.loads((case/'run.json').read_text())
    t=c['t']; pt=(w['us']-u['release_us'])*1e-6
    fig,axes=plt.subplots(3,1,figsize=(7.25,5.15),sharex=True)
    plt.rcParams.update({'pdf.fonttype':42,'font.family':'DejaVu Sans'})
    fig.subplots_adjust(top=.88,bottom=.11,left=.12,right=.97,hspace=.28)
    for ax in axes:
        ax.axvspan(28,32,color='#e6e9ed',lw=0)
        if s['exit_time'] is not None:ax.axvline(s['exit_time'],color='#8a8c8d',ls=':',lw=.8)
        ax.spines['right'].set_visible(False);ax.spines['top'].set_visible(False)
        ax.spines['left'].set_color('#b6c0c1');ax.spines['bottom'].set_color('#b6c0c1')
        ax.tick_params(labelsize=8,length=2,color='#899093');ax.grid(axis='y',color='#ebeeee',lw=.6)
    for k,style,col,lbl in [('des_b','--','#303e46','Target'),('est_b','-','#318578','Oracle feedback')]:
        b=c[k];th=np.rad2deg(np.linalg.norm(b,axis=1));ph=(np.rad2deg(np.arctan2(b[:,1],b[:,0]))+360)%360
        ph[th<5]=np.nan
        axes[0].plot(t,th,ls=style,color=col,lw=1.55,label=lbl)
        axes[1].plot(t,ph,ls=style,color=col,lw=1.55)
    err=100*np.linalg.norm(w['tcp']-s['goal_xyz'],axis=1)
    near=(pt>=s['exit_time'])&(pt<=55)
    axes[2].plot(pt[near],err[near],lw=1.65,color='#516f91',label='Tool position error')
    axes[2].axhline(3.5,c='#af6655',ls='--',lw=1,label='3.5 cm threshold')
    axes[0].legend(loc='upper left',fontsize=7,frameon=False,ncol=2)
    axes[2].legend(loc='upper left',fontsize=7,frameon=False,ncol=2)
    axes[0].set_ylabel('Bend θ (deg)',fontsize=9);axes[1].set_ylabel('Azimuth φ (deg)',fontsize=9)
    axes[2].set_ylabel('TCP error (cm)',fontsize=9);axes[2].set_xlabel('Time since launch support release (s)',fontsize=9)
    axes[2].set_xlim(12,55)
    fig.text(.12,.975,'Oblique gate task  |  physical reference flight',fontweight='bold',fontsize=12,color='#2d3a41',va='top')
    fig.text(.12,.924,'Independent development seed 26112090  ·  shaded interval: scheduled wrist occlusion',fontsize=8,color='#52616a')
    fig.savefig(output,dpi=170,facecolor='white');plt.close(fig)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('case',type=Path);ap.add_argument('output',type=Path)
    args=ap.parse_args();build(args.case,args.output)
