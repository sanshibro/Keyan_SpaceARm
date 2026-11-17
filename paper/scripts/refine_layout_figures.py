"""Re-layout existing vector artwork; no raw samples or new results are generated.

The selected source PDF paths are transformed affinely into their original axis
coordinates. This preserves the source's plotted polylines, including its existing
path simplification. It is not recovery or reanalysis of raw experimental data.
"""
from pathlib import Path
import json
import fitz
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
doc = fitz.open(ROOT/'figures/rolling_revised.pdf')
draw = doc[0].get_drawings()
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,
    'axes.titlesize':9,'axes.labelsize':8.5,'xtick.labelsize':8,
    'ytick.labelsize':8,'axes.linewidth':.6,'pdf.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(7.12,3.0))
fig.subplots_adjust(left=.077,right=.986,bottom=.31,top=.88,wspace=.29)
indices=[0,3,5]
def points(d):
    assert all(item[0]=='l' for item in d['items'])
    return np.array([tuple(d['items'][0][1])]+[tuple(v[2]) for v in d['items']])
def transform(p,panel):
    if panel==0:
        return np.column_stack(((p[:,0]-draw[61]['rect'].x0)/((draw[53]['rect'].x0-draw[61]['rect'].x0)/10),
            .5+(draw[63]['rect'].y0-p[:,1])/((draw[63]['rect'].y0-draw[71]['rect'].y0)/2)))
    return np.column_stack(((p[:,0]-draw[94]['rect'].x0)/((draw[96]['rect'].x0-draw[94]['rect'].x0)/2.5),
        (draw[102]['rect'].y0-p[:,1])/((draw[102]['rect'].y0-draw[114]['rect'].y0)/120)))
audit=[]
for i in range(47,53):
    rr=draw[i]['rect'];v=transform(np.array([[rr.x0,rr.y1],[rr.x1,rr.y0]]),0)
    axes[0].add_patch(Rectangle(v[0],*(v[1]-v[0]),facecolor='#e2e4e6',edgecolor='#94999e',linewidth=.6,zorder=0))
for panel,base in enumerate([73,116]):
    ax=axes[panel]
    for k in indices:
        d=draw[base+k];xy=transform(points(d),panel)
        ax.plot(*xy.T,color=d['color'],lw=1.05,ls=(0,(4,2)),label=f'Window {k+1}',zorder=3)
        ax.plot(*xy[0],marker='o',ms=3,color=d['color'],zorder=6)
        audit.append({'panel':panel,'source_path_index':base+k,'vertices':len(xy)})
    d=draw[base+7];xy=transform(points(d),panel)
    ax.plot(*xy.T,color='#16191c',lw=1.7,label='Executed',zorder=5)
    audit.append({'panel':panel,'source_path_index':base+7,'vertices':len(xy)})
    ax.set_xlim(-.45,10 if panel==0 else 9.8)
    ax.set_xticks([0,2,4,6,8,10] if panel==0 else [0,2,4,6,8])
    ax.set_xlabel('Forward progress, $-y$ (m)')
    ax.grid(color='#e6e8eb',lw=.45,zorder=0)
    ax.spines[['top','right']].set_visible(False)
axes[0].set_ylim(.2,2.5);axes[0].set_ylabel('World height, $z$ (m)')
axes[0].set_title('(a) Vehicle plans through the passages',loc='left',pad=8)
axes[1].set_ylim(-5,125);axes[1].set_ylabel(r'Bending angle, $\theta$ (deg)')
axes[1].set_title('(b) Planned bending versus execution',loc='left',pad=8)
handles,labels=axes[1].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.060),ncol=4,frameon=False,handlelength=2.9,columnspacing=2.6)
fig.text(.5,.014,'Dashed: window plans (may be superseded)    Solid black: execution',ha='center',fontsize=8,color='#444b52')
fig.savefig(ROOT/'figures/rolling_clear.pdf')
(ROOT/'scripts/rolling_artwork_audit.json').write_text(json.dumps(audit,indent=2))
print('Generated rolling_clear.pdf from unchanged source plot paths.')

# Remove the obsolete source-panel letter without changing plotted geometry.
geometry = fitz.open(ROOT/"figures/geometry_revised.pdf")
for rect in geometry[0].search_for("(b)"):
    geometry[0].add_redact_annot(rect, fill=False)
geometry[0].apply_redactions(images=0, graphics=0, text=0)
geometry.save(ROOT/"figures/geometry_clean.pdf")
