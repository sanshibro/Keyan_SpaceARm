"""A held-vision diagnostic on the recorded trajectory of the registered fusion flight."""
import argparse
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from PIL import Image

from gatereview_visuals import load_flight, angle, style


def build(stage):
    protocol=json.loads((stage/'protocol.json').read_bytes())
    case=next(row for row in protocol['cases'] if row['seed']==protocol['representative_seed'] and
              (row['method'],row['material'],row['vision'])==('proposed','differential','occluded'))
    folder=stage/'formal'/case['case']
    flight=load_flight(folder,case)
    spec=protocol['window'];on,off=spec['blackout_start'],spec['blackout_end']
    rows=[json.loads(s) for s in (folder/'optics/index.jsonl').read_text().splitlines() if s]
    by_stamp={round(r['t'],4):r for r in rows}
    with zipfile.ZipFile(stage/'evidence'/(case['case']+'.zip')) as archive:
        with np.load(io.BytesIO(archive.read('optics/observations.npz'))) as observations:
            t=observations['t']
            valid=observations['valid']
            ids=[np.flatnonzero(valid & (t<spec['partial_start']-1.e-7))[-1],
                 int(np.argmin(abs(t-(on+off)/2))),
                 np.flatnonzero(valid & (t>=off-1.e-7))[0]]
            snapshots=[by_stamp[round(float(t[i]),4)] for i in ids]
    metrics=json.loads((folder/'metrics.json').read_bytes())
    plt=style()
    fig=plt.figure(figsize=(8.8,4.95),layout='constrained')
    gs=fig.add_gridspec(2,6,height_ratios=[.98,1.15],hspace=.13,wspace=.12)
    for j,(name,row) in enumerate(zip(('Before obstruction','Complete occlusion','First recovered pose'),snapshots)):
        ax=fig.add_subplot(gs[0,2*j:2*j+2])
        with Image.open(folder/'optics'/row['rgb']) as im:
            ax.imshow(im)
        ax.set_xticks([]);ax.set_yticks([])
        for key in ax.spines:
            ax.spines[key].set_visible(True);ax.spines[key].set_color('#a4b2b2');ax.spines[key].set_linewidth(.55)
        ax.set_title(name,loc='left',fontsize=9.5,pad=4)
        ax.set_xlabel(f"Exposure {row['t']:.2f} s   ·   delivery {row['arrival_t']:.2f} s",fontsize=8,labelpad=3)
    for i,name in enumerate(('Bending magnitude θ (deg)','Bending plane φ (deg)')):
        ax=fig.add_subplot(gs[1,3*i:3*i+3])
        for key,label,color,ls,lw in [('true_b','Physical','#263a45','-',1.9),('est_b','Proposed','#238572','--',1.5),
                                      ('vision_hold_b','Last visual pose','#a56940','-.',1.35)]:
            values=angle(flight.z[key])[i];ax.plot(flight.t,values,color=color,ls=ls,lw=lw,label=label)
        ax.set_xlim(on-2,off+2);ax.set_ylim((28,85) if i==0 else (157,197))
        ax.axvspan(on,off,color='#e6ebec',zorder=-4)
        ax.grid(axis='y',color='#dfe5e4',lw=.5)
        ax.set_ylabel(name);ax.set_xlabel('Time since release (s)')
        if i==0:ax.legend(fontsize=7.7,frameon=False,loc='upper left')
    out=stage/'review';out.mkdir(exist_ok=True)
    main=out/'gate_wrist_window.png';fig.savefig(main,dpi=220,facecolor='white');plt.close(fig)
    with Image.open(main) as im:
        im.thumbnail((1250,800),Image.Resampling.LANCZOS)
        im.convert('RGB').save(out/'gate_wrist_window.webp','WEBP',quality=88,method=6)
    note=dict(case=case['case'],exposures=[r['t'] for r in snapshots],deliveries=[r['arrival_t'] for r in snapshots],
              rgb_sha256=[r['rgb_sha256'] for r in snapshots],blackout=metrics['blackout_angle'],
              vision_hold=metrics['vision_hold_blackout'],true_theta_range_deg=metrics['dynamic_theta_range_deg'],
              true_phi_range_deg=metrics['dynamic_phi_range_deg'])
    (out/'wrist-window.json').write_text(json.dumps(note,indent=2)+'\n',encoding='utf-8')
    print(str(main))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',type=Path);args=p.parse_args()
    build(args.stage.resolve())
