"""Show paired differences due to the pre-entry strain offset under the same 4 s blackout."""
import argparse
import json
from pathlib import Path
import numpy as np
from matplotlib.lines import Line2D
from gatereview_visuals import METHODS,NAMES,COLORS,INK,MUTED,PALE,RED,style,save


def build(stage):
    s=json.loads((stage/'protocol.json').read_bytes())
    metrics=json.loads((stage/'decisions/metrics.json').read_bytes())['cases']
    if set(metrics)!={r['case'] for r in s['cases']}:
        raise ValueError('Paired outcome diagram requires the complete registered set')
    cells={(v['case']['seed'],v['case']['method'],v['case']['material']):v
           for v in metrics.values() if v['case']['vision']=='occluded'}
    seeds=s['seeds'];plt=style()
    fig,axes=plt.subplots(1,3,figsize=(9.6,3.95),width_ratios=[1.12,1.14,.91])
    fig.subplots_adjust(left=.126,right=.99,top=.78,bottom=.22,wspace=.29)
    keys=[('arm_tool_min_m',100,6.0,'Minimum arm–frame clearance (cm)'),
          ('axis_rmse_deg',1,3.,'Terminal tool-axis RMSE (deg)')]
    for j,(key,factor,threshold,label) in enumerate(keys):
        a=axes[j]
        all_values=[v[key]*factor for v in cells.values()]
        if j==0:
            low=min(threshold-.75,min(all_values)-.35);high=max(max(all_values)+.40,threshold+.75)
        else:
            low=-.18;high=max(max(all_values)+.35,threshold+.6)
        a.set_xlim(low,high)
        if j==0:a.axvspan(low,threshold,color='#f7eeeb',zorder=-4)
        a.axvline(threshold,ls='--',color=RED,lw=.9)
        a.text(threshold,.985,(f'{threshold:.1f} cm minimum' if j==0 else '3.0° axis tolerance'),
               transform=a.get_xaxis_transform(),color=RED,fontsize=7,ha='center',va='top',
               bbox=dict(facecolor='white',edgecolor='none',pad=.7))
        for mi,method in enumerate(METHODS):
            for rank,seed in enumerate(seeds):
                y=mi+(.17*(rank-1));x0=cells[(seed,method,'nominal')][key]*factor
                x1=cells[(seed,method,'differential')][key]*factor
                a.plot([x0,x1],[y,y],lw=.9,color='#b6c2c2',zorder=1)
                a.scatter(x0,y,s=22,facecolor='white',edgecolor=COLORS[method],linewidth=.9,zorder=2)
                a.scatter(x1,y,s=13,facecolor=COLORS[method],edgecolor=COLORS[method],linewidth=.6,zorder=3)
        a.set_xlabel(label,fontsize=8.4)
        a.set_ylim(4.59,-.83);a.set_yticks(range(5),[NAMES[m] for m in METHODS] if j==0 else ['']*5)
        a.tick_params(axis='y',length=0,pad=7);a.spines['left'].set_visible(False)
        a.grid(axis='x',color=PALE,lw=.4,zorder=-6);a.set_axisbelow(True)
        for yi in range(4):a.axhline(yi+.5,lw=.4,color=PALE,zorder=-5)
    table=axes[2];table.axis('off');table.set_xlim(0,1);table.set_ylim(4.59,-.83)
    table.text(.5,-.63,'Complete task',ha='center',color=INK,fontsize=9.7,weight='bold')
    table.text(.23,-.28,'Nominal',ha='center',color=MUTED,fontsize=7.8);table.text(.76,-.28,'Offset',ha='center',color=MUTED,fontsize=7.8)
    for mi,m in enumerate(METHODS):
        for u,mat in ((.23,'nominal'),(.76,'differential')):
            count=sum(cells[(seed,m,mat)]['success'] for seed in seeds)
            table.text(u,mi+.13,f'{count} / 3',ha='center',va='center',color=COLORS[m],weight='bold',fontsize=10)
        if mi<4:table.axhline(mi+.5,lw=.45,color=PALE)
    fig.text(.02,.955,'Effect of a pre-entry strain offset during visual occlusion',color=INK,size=12.8,weight='bold')
    fig.text(.02,.893,'Each horizontal segment pairs two independent flights with the same seed, controller and 4 s shutter.',fontsize=8.4,color=MUTED)
    handles=[Line2D([],[],marker='o',mfc='white',mec=INK,ls='none',label='Baseline material hysteresis'),
             Line2D([],[],marker='o',mfc=INK,mec=INK,ls='none',label='Hysteresis + pre-entry differential offset')]
    fig.legend(handles=handles,loc='lower left',bbox_to_anchor=(.111,.074),ncol=2,frameon=False,fontsize=8)
    fig.text(.126,.035,'Rows within a method follow seeds 26112131, 26112137, 26112143.  Axis RMSE is measured over 49–55 s; task counts use the registered joint 1 s dwell.',size=7.3,color=MUTED)
    save(fig,stage/'review','gate_paired_outcomes');plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',type=Path);args=p.parse_args();build(args.stage.resolve())
