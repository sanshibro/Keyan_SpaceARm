"""Development-only diagnosis. Render directly from a scored M2 pilot and its native logs."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot(folder):
    audit=folder/'audit';flight=folder/'flight'
    m=json.loads((audit/'metrics.json').read_text());release=json.loads((flight/'run.json').read_text())['release_us']
    with np.load(flight/'control.npz') as c,np.load(audit/'series.npz') as s,np.load(flight/'physics.npz') as p:
        t=c['t'];tp=(p['us']-release)*1.e-6;ix=np.searchsorted(p['us'],release+np.rint(1e6*t).astype('int64'))
        event=m['certificates']['body']['t_s'];g1=m['certificates']['arm']['t_s']
        contact_idx=np.flatnonzero(np.diff(p['contact_steps'],prepend=0)>0)
        if not len(contact_idx):return
        plt.rcParams.update({'font.family':'serif','font.serif':['STIX Two Text','DejaVu Serif'],'mathtext.fontset':'stix','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'svg.fonttype':'none','pdf.fonttype':42})
        fig,axes=plt.subplots(2,2,figsize=(10.2,6.0),constrained_layout=True)
        ink='#303e41';teal='#217375';rust='#af6340';gray='#879192'
        def title(ax,s):ax.set_title(s,loc='left',fontweight='semibold',fontsize=11,pad=8);ax.grid(axis='y',alpha=.16);ax.tick_params(width=.65)
        a,b,d,e=axes.ravel();sel=(t>=g1-3)&(t<=g1+2.8);ss=(t>=event-2.5)&(t<=event+2.4)
        title(a,'a   Arm folding at the high portal')
        a.plot(t[sel],np.rad2deg(np.linalg.norm(c['des_b'][sel],axis=1)),color=gray,ls='--',label='Planner command')
        a.plot(t[sel],np.rad2deg(np.linalg.norm(c['est_b'][sel],axis=1)),color=rust,lw=1.15,label='Fused estimate')
        a.plot(t[sel],np.rad2deg(s['theta'][sel]),color=teal,lw=1.4,label='Physical shape')
        a.axvline(g1,color=ink,ls=':',lw=.85);a.set_ylabel('Bending magnitude (deg)');a.legend(frameon=False,fontsize=8)
        title(b,'b   Vertical tracking at the portal')
        for z,label,color,style in [(c['ref_world'][:,2],'Reference',gray,'--'),(p['qpos'][ix,2],'Measured base',teal,'-')]:
            b.plot(t[sel],z[sel],color=color,ls=style,label=label)
        b.axvline(g1,color=ink,ls=':',lw=.85);b.set_ylabel('World height (m)');b.legend(frameon=False,fontsize=8)
        title(d,'c   Passage exit at the cross-pipe')
        d.plot(c['ref_world'][ss,0],c['ref_world'][ss,1],'--',color=gray,label='Issued base reference')
        d.plot(p['qpos'][ix[ss],0],p['qpos'][ix[ss],1],color=teal,label='Measured base')
        d.scatter(p['qpos'][contact_idx,0],p['qpos'][contact_idx,1],s=18,color=rust,zorder=3,label='Contact interval');d.set_xlabel('World x (m)');d.set_ylabel('World y (m)');d.legend(frameon=False,fontsize=8)
        title(e,'d   Residual speed at the end of the local plan')
        e.plot(t[ss],np.linalg.norm(c['plan_dq'][ss,:3],axis=1),'--',color=gray,label='Commanded speed')
        e.plot(t[ss],np.linalg.norm(p['qvel'][ix[ss],:3],axis=1),color=teal,label='Measured speed')
        e.axvspan(tp[contact_idx[0]]-.02,tp[contact_idx[-1]],color=rust,alpha=.25,label='Contact samples')
        e.set_ylabel('Base speed (m/s)');e.legend(frameon=False,fontsize=8)
        for ax in [a,b,e]:ax.set_xlabel('Time after release (s)')
        for ext in ['png','pdf']:fig.savefig(folder/('engineering-pilot.'+ext),dpi=240,facecolor='white')
        plt.close(fig)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);plot(p.parse_args().folder)
