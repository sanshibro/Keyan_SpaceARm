"""Publication-scale vector plots from retained development logs."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[2]
out=ROOT/'revision/figures';out.mkdir(exist_ok=True,parents=True)
data=ROOT/'revision/data'
plt.rcParams.update({'font.family':'serif','font.serif':['Liberation Serif','DejaVu Serif'],
    'font.size':10,'axes.labelsize':10,'axes.titlesize':10,'axes.linewidth':.65,
    'xtick.labelsize':9,'ytick.labelsize':9,'pdf.fonttype':42,'ps.fonttype':42,
    'savefig.facecolor':'white','figure.facecolor':'white'})
fixed=np.genfromtxt(data/'super_backend/fixed/states.csv',delimiter=',',names=True)
hover=np.genfromtxt(data/'super_backend/hover/states.csv',delimiter=',',names=True)
audit=json.loads((data/'super_geometry_audit.json').read_text())
fig,ax=plt.subplots(1,2,figsize=(8.0,3.05),gridspec_kw={'width_ratios':[1.03,1]})
a=ax[0];a.set_title('(a)  Integrated bend response',loc='left',pad=9)
a.plot(fixed['time_s'],fixed['command_theta_deg'],c='#252525',lw=1.2,ls='--',label='Command')
a.plot(fixed['time_s'],fixed['total_curvature_deg'],c='#286383',lw=1.4,label='Fixed base')
a.plot(hover['time_s'],hover['total_curvature_deg'],c='#ba704b',lw=1.05,ls=':',label='Hover')
a.axhline(120,c='#aa554a',lw=.8,ls=(0,(4,3)));a.text(.1,123,'120° bound',color='#985045',fontsize=8)
a.set(xlabel='Time (s)',ylabel='Total bending (deg)',xlim=(0,14),ylim=(-3,136),xticks=[0,4,8,12],yticks=[0,30,60,90,120])
a.legend(loc='lower center',bbox_to_anchor=(.5,-.38),frameon=False,ncol=3,fontsize=8,handlelength=1.8,columnspacing=1.1)
for side in ['top','right']:a.spines[side].set_visible(False)
a.grid(axis='y',color='#e1e3e4',lw=.5)
b=ax[1];b.set_title('(b)  Self-collision at sampled shapes',loc='left',pad=9)
thetas=[0,30,60,90,110,118];phis=list(range(0,360,45))
for row in audit['pose_samples']:
    ok=row['self_collision_free'];x=phis.index(row['phi_deg']);y=thetas.index(row['theta_deg'])
    b.scatter(x,y,s=45,marker='o' if ok else 'x',color='#337c75' if ok else '#b86148',linewidths=1.25,zorder=3)
b.set(xticks=range(8),xticklabels=phis,yticks=range(6),yticklabels=thetas,
    xlabel=r'Bending plane, $\varphi$ (deg)',ylabel=r'Bend command, $\theta$ (deg)',xlim=(-.5,7.5),ylim=(-.5,5.5))
b.grid(color='#e5e6e7',lw=.5)
for side in ['top','right']:b.spines[side].set_visible(False)
handles=[Line2D([0],[0],marker='o',ls='',color='#337c75',label='Clear sample (30)'),Line2D([0],[0],marker='x',ls='',color='#b86148',label='Collision (18)')]
b.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,-.38),frameon=False,ncol=2,fontsize=8)
fig.subplots_adjust(left=.075,right=.985,top=.88,bottom=.29,wspace=.36)
for ext in ['pdf','png']:fig.savefig(out/('super_backend_checks.'+ext),dpi=240)
plt.close(fig)
print(out/'super_backend_checks.pdf')
