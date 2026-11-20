"""Registered CAD views and time traces of the independent development seed."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gatereview_visuals import load_flight,NativeViews,angle

def build(stage):
    case=stage/'development/native-qualification-04'
    info=json.loads((case/'control-summary.json').read_text())
    flight=load_flight(case,info['case'])
    protocol=json.loads((stage/'protocol.json').read_text())
    camera=NativeViews({'proposed':flight},protocol['review_cameras'],'gate_portable_model',(640,480))
    goal=np.array(info['goal_xyz']);target=np.array(info['goal_axis']);z=flight.z;t=flight.t
    figure,axes=plt.subplots(2,3,figsize=(10.6,5.35),gridspec_kw={'height_ratios':[1.48,1.]})
    figure.subplots_adjust(left=.065,right=.98,bottom=.13,top=.875,hspace=.34,wspace=.26)
    try:
        for j,(stamp,title) in enumerate([(27.8,'Approach'),(31.4,'Dynamic occlusion'),(52.,'Oblique alignment')]):
            i=flight.index(stamp);ax=axes[0,j]
            if j<2:
                pts=np.stack([z['arm_near_robot'][i],z['arm_near_gate'][i]])
                region=([.75,.015,1.99],1.16)
            else:
                pts=np.stack([goal,goal+.16*target,z['tip'][i],z['tip'][i]+.16*z['tool_axis'][i]])
                region=([1.84,.15,1.97],1.09)
            im,uv=camera.image('proposed',stamp,'oblique',pts,region)
            ax.imshow(im);ax.set_xlim(-.5,im.width-.5);ax.set_ylim(im.height-.5,-.5)
            ax.set_axis_off();ax.set_title(f'{chr(97+j)}   {title}    {stamp:.1f} s',fontsize=10,loc='left',weight='semibold')
            if j<2:ax.plot(uv[:,0],uv[:,1],color='#ad6255',lw=1.15,marker='o',ms=3)
            else:
                ax.plot(uv[:2,0],uv[:2,1],'--',color='#ad793d',lw=1.5);ax.scatter(*uv[0],marker='+',s=40,c='#ad793d')
                ax.plot(uv[2:,0],uv[2:,1],color='#24776d',lw=1.1)
    finally:camera.close()
    cut=(t>=24)&(t<=39);blk=z['blackout']
    for j in (0,1):
        ax=axes[1,j]
        ax.plot(t[cut],angle(z['true_b'])[j][cut],lw=1.65,color='#283d46',label='Physical')
        ax.plot(t[cut],angle(z['est_b'])[j][cut],lw=1.45,color='#278678',ls='--',label='Fused')
        ax.plot(t[blk],angle(z['vision_hold_b'])[j][blk],lw=1.2,color='#a36f3f',ls='-.',label='Held vision')
        ax.set_ylabel(('Bend θ (deg)','Azimuth φ (deg)')[j]);ax.set_xlim(24,39)
    ax=axes[1,2];near=(t>=28)&(t<=40)
    ax.plot(t[near],100*z['body_clearance'][near],color='#66829b',lw=1.3,label='Body')
    ax.plot(t[near],100*z['arm_tool_clearance'][near],color='#278678',lw=1.6,label='Arm and tool')
    ax.axhline(6,color='#ad6255',ls=':',lw=1.1,label='6.0 cm');ax.set_xlim(28,40);ax.set_ylim(4,27)
    ax.set_ylabel('Gate clearance (cm)')
    for j,ax in enumerate(axes[1]):
        ax.axvspan(28,32,color='#e9edef',zorder=-4)
        ax.set_xlabel('Time since release (s)');ax.spines['right'].set_visible(False);ax.spines['top'].set_visible(False)
        ax.grid(axis='y',lw=.5,color='#e0e5e5');ax.tick_params(labelsize=8,length=2)
        if j!=1:ax.legend(loc='upper center',bbox_to_anchor=(.5,-.32),ncol=3,frameon=False,fontsize=7)
    figure.text(.065,.965,'Two-axis gate flight with frozen augmented feedback',fontsize=15,weight='bold',color='#25393d')
    figure.text(.065,.915,'Development seed 26112090   ·   4 s optical occlusion   ·   registered SUPER–Qingyu geometry',fontsize=9,color='#51656a')
    figure.savefig(stage/'pilot/单门闭环_受限通过与斜向对准.png',dpi=170,facecolor='white');plt.close(figure)

if __name__=='__main__':
    import sys
    build(Path(sys.argv[1]))
