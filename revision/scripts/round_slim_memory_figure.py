"""Display one controlled length cycle and a held-out hardware transfer test."""
import numpy as np
import matplotlib.pyplot as plt
from round_slim_visuals import C,INK,MUTED,title,finish
from round_slim_box import OUT
from round_wrist_fusion import readj
def main():
    z=np.load(OUT/'memory/controlled-loop.npz');r=readj(OUT/'memory-scores.json')
    fig=plt.figure(figsize=(10.8,3.35));g=fig.add_gridspec(1,3,left=.072,right=.98,bottom=.21,top=.85,width_ratios=[1.03,1.13,1.03],wspace=.43)
    a,b,c=[fig.add_subplot(g[i]) for i in range(3)]
    gain=readj(OUT/'box-profile.json')['slope_counts_per_unit_ratio'][0]
    x=z['eps']*100;t=z['t'];delta=(z['combined']-z['static'])*gain
    go=(t>=2)&(t<=8);back=(t>=16)&(t<=22)
    a.plot(x[go],100*z['static'][go],':',lw=1.6,color='#576468',label='Static')
    a.plot(x[go],100*z['combined'][go],color=C[0],lw=1.8,label='Loading')
    a.plot(x[back],100*z['combined'][back],color=C[1],lw=1.8,label='Unloading')
    for q in [go,back]:
        ix=np.flatnonzero(q)[int(.5*np.sum(q))]
        a.plot(x[ix],100*z['combined'][ix],'o',ms=3.8,color=C[0] if q is go else C[1])
    title(a,'a   At the same length');a.set(xlabel='Strip strain (%)',ylabel='Resistance change (%)',xticks=[30,35,40,45,50])
    a.legend(loc='upper left',fontsize=7.2,frameon=False,handlelength=1.6)
    for lo,hi in [(8,16),(22,29)]:b.axvspan(lo,hi,color='#eef2f0',lw=0)
    b.axhline(0,color='#a6b4ae',lw=.7);b.plot(t,(z['loop']-z['static'])*gain,color=C[1],lw=1.45,label='Loop only')
    b.plot(t,(z['relax']-z['static'])*gain,color=C[2],lw=1.35,label='Relaxation only')
    b.plot(t,delta,color=C[0],lw=1.15,ls='--',label='Combined')
    title(b,'b   Error hidden by scale');b.set(xlabel='Cycle time (s)',ylabel='Change from static (counts)',xlim=(0,29),xticks=[0,8,16,22,29])
    b.legend(loc='upper left',frameon=False,fontsize=7.1,handlelength=1.6)
    rows=r['historical_check']['heldout'];k=np.arange(3);fst=np.array([v['static_rmse_counts'] for v in rows]);sec=np.array([v['memory_rmse_counts'] for v in rows]);w=.32
    c.bar(k-w/2,fst,width=w,color='#a4afb1',label='Static');c.bar(k+w/2,sec,width=w,color=C[0],label='With memory')
    for i in range(3):
        for off,n in [(-w/2,fst[i]),(w/2,sec[i])]:c.text(i+off,n+1.1,'%.0f'%n,ha='center',fontsize=7.4,color=INK)
    c.set_ylim(0,max(fst.max(),sec.max())*1.32);c.set_xticks(k);c.set_xticklabels(['CH1','CH2','CH3'])
    c.set_xlabel('Original acquisition box');c.set_ylabel('Held-out RMSE (counts)')
    title(c,'c   Transfer to another session');c.legend(ncol=2,fontsize=7.2,frameon=False,loc='upper left',columnspacing=.8,handlelength=1.5)
    for ax in (a,b,c):ax.grid(axis='y',color='#dce4e0',lw=.48);ax.set_axisbelow(True)
    fig.text(.073,.027,'a,b  Prescribed length and illustrative memory parameters.',fontsize=8,color=MUTED)
    fig.text(.633,.027,'c  Separate physical session; angle-based strain proxy.',fontsize=8,color=MUTED)
    finish(fig,'memory')
if __name__=='__main__':main()
