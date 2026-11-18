"""Build the research meeting plots and an independent one-page candidate from integrated data."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator
import numpy as np
from round_e1_run import DEFAULT

ink='#213641';teal='#227676';copper='#ae623c';grey='#62737a'
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'axes.labelcolor':ink,'text.color':ink,
    'xtick.color':ink,'ytick.color':ink,'axes.spines.top':False,'axes.spines.right':False,
    'axes.edgecolor':'#607680','axes.linewidth':.65,'pdf.fonttype':42,'savefig.facecolor':'white'})

def chart(fig,layout,x,letters):
    a,b=[fig.add_subplot(s) for s in layout]
    t=x['t_s'];xs=[0,2,4,6,8,10,12,14]
    for ax in (a,b):
        ax.axvspan(7,9,color='#e9efed',zorder=-2)
        ax.grid(axis='y',color='#dde4e5',linewidth=.5);ax.set_axisbelow(True)
        ax.set_xticks(xs);ax.set_xlim(0,14);ax.set_xlabel('Simulation time (s)')
        ax.tick_params(length=3,width=.6,labelsize=8)
    a.plot(t,x['bend_8_deg'],color=teal,lw=1.8,label='Integrated, 8 elements')
    a.plot(t,x['command_deg'],color=ink,lw=1,ls='--',label='Issued command')
    a.set_ylim(-5,126);a.set_yticks([0,30,60,90,120]);a.set_ylabel('Bend angle (deg)')
    a.set_title('(%s)  Command and response'%letters[0],loc='left',size=10)
    a.legend(loc='lower center',bbox_to_anchor=(.55,.01),fontsize=7,frameon=False,handlelength=2)
    b.plot(t,x['n8_n24_tcp_mm'],color=copper,lw=1.8,label='8 vs 24')
    b.plot(t,x['n16_n24_tcp_mm'],color=teal,lw=1.4,ls='--',label='16 vs 24')
    b.set_ylim(-.05,1.32);b.set_yticks([0,.3,.6,.9,1.2]);b.set_ylabel('TCP separation (mm)')
    b.set_title('(%s)  Effect of spatial refinement'%letters[1],loc='left',size=10)
    b.legend(loc='upper left',fontsize=7,frameon=False,handlelength=2.4)
    k=int(np.argmax(x['n8_n24_tcp_mm']))
    b.scatter([t[k]],[x['n8_n24_tcp_mm'][k]],s=13,color=copper,zorder=3)
    b.annotate('%.2f mm'%x['n8_n24_tcp_mm'][k],(t[k],x['n8_n24_tcp_mm'][k]),xytext=(8,5),textcoords='offset points',fontsize=7,color=copper)

def figures():
    s=json.loads((DEFAULT/'summary.json').read_text())
    x=np.genfromtxt(DEFAULT/'timeline.csv',delimiter=',',names=True,encoding='utf8')
    f,axes=plt.subplots(1,2,figsize=(7.7,2.6));[ax.remove() for ax in axes]
    grid=gridspec.GridSpec(1,2,figure=f,left=.08,right=.985,bottom=.19,top=.87,wspace=.38)
    chart(f,[grid[0],grid[1]],x,'ab')
    for extension in ('png','pdf'):f.savefig(DEFAULT/('curves.'+extension),dpi=220)
    plt.close(f)
    f=plt.figure(figsize=(8.27,10.5))
    f.text(.08,.951,'Spatial convergence of the continuum-arm model',size=16,weight='bold')
    f.text(.08,.925,'Fixed SUPER airframe   /   Qingyu geometry   /   numerical development study',size=9,color=grey)
    f.add_artist(plt.Line2D([.08,.925],[.907,.907],transform=f.transFigure,color='#8da0a6',lw=.65))
    f.text(.08,.881,'What changes if the same arm is integrated with more elements?',size=11,style='italic')
    f.text(.08,.850,'The 0.28 m arm bends, holds and returns in a single plane under gravity. All models share the\n'
                      'same 0.25 kg arm, 0.20 kg distal assembly, neutral inertia and equivalent drive parameters.',size=9,linespacing=1.45)
    for j,(t,phase) in enumerate([(0,'Neutral'),(4,'Bending'),(8,'Peak hold')]):
        ax=f.add_axes([.08+j*.288,.588,.265,.224])
        ax.imshow(plt.imread(str(DEFAULT/('cad_t%02d.png'%t))));ax.axis('off')
        ax.text(0,1.02,'(%s)  %s'%('abc'[j],phase),transform=ax.transAxes,size=10)
        i=int(np.argmin(np.abs(x['t_s']-t)))
        ax.text(0,-.04,'%.1f s    |    integrated %.1f°'%(t,x['bend_8_deg'][i]),transform=ax.transAxes,color=grey,size=8)
    grid=gridspec.GridSpec(1,2,figure=f,left=.107,right=.924,bottom=.37,top=.548,wspace=.37)
    chart(f,[grid[0],grid[1]],x,'de')
    f.text(.08,.309,'Maximum differences over the 14 s cycle',size=11,weight='bold')
    cols=[.083,.357,.51,.657,.915]
    headers=['Elements','Arm end (mm)','TCP (mm)','Centerline (mm)','Budget (mm)']
    for j,(pos,key) in enumerate(zip(cols,headers)):f.text(pos,.276,key,ha='left' if j==0 else 'right',fontsize=8,color=grey)
    for i,c in enumerate(s['comparisons'][:3]):
        values=['%d vs %d'%(c['N_a'],c['N_b']),'%.3f'%c['backbone_end_mm'],'%.3f'%c['tcp_mm'],'%.3f'%c['centerline_mm'],'%.1f'%(1.5 if i==2 else 3.)]
        for j,(pos,v) in enumerate(zip(cols,values)):f.text(pos,.246-i*.026,v,ha='left' if j==0 else 'right',fontsize=9)
    for y in (.265,.181):
        f.add_artist(plt.Line2D([.08,.925],[y,y],transform=f.transFigure,color='#a8b8bc',lw=.65))
    time_max=max(c['tcp_mm'] for c in s['comparisons'][3:])
    f.text(.08,.149,'The worst 8-to-24 TCP difference is %.2f mm. Halving the step from 1 to 0.5 ms changes the\n'
                      'TCP by at most %.4f mm. The 8-element model meets the registered budgets for this motion.'%(s['comparisons'][1]['tcp_mm'],time_max),size=9,linespacing=1.5)
    f.text(.08,.087,'The three CAD frames use integrated 8-element states. The shaded interval is the 118° hold.\n'
                      'TCP denotes the gripper tool center point. Physical stiffness and flight performance require separate measurements.',size=8,color=grey,linespacing=1.45)
    f.savefig(DEFAULT/'review.pdf');plt.close(f)
if __name__=='__main__': figures()
