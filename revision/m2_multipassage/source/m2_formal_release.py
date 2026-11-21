"""Create the paired M2 paper figures and two-page review from a qualified campaign."""
import argparse
import json
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from m2_gate import FORMAL, METHODS
from m2_review_data import COLORS, METHODS as NAMES, FlightRecord, read, sha

GROUP_A=('straight','fixed','sequential','unregularized','coupled')
GROUP_B=('stop','hard','coupled')
INK='#26383d'
TEAL='#326f78'
RUST='#ab6548'

def matrix(path):
    cases={}
    for item in read(path)['cases']:
        f=path.parent/item['flight'];a=path.parent/item['audit'];m=read(a/'metrics.json');key=(m['strategy'],m['seed'])
        if (key in cases or m['development'] or not m['native_logs_verified'] or
            not m['synchronized_rgbd'] or not m['causal_map_verified']):raise ValueError('Unpaired or unaudited formal case '+str(key))
        cases[key]=(f,a,m)
    if set(cases)!={(k,s) for k in METHODS for s in FORMAL}:raise ValueError('Expected 21 distinct formal flights')
    ids={json.dumps(m['identity'],sort_keys=True) for _,_,m in cases.values()}
    if len(ids)!=1:raise ValueError('Nonuniform formal software or scene')
    if len({m['raw_ulog_sha256'] for _,_,m in cases.values()})!=21:raise ValueError('Repeated PX4 flight evidence')
    return cases

def summarize(cases,group):
    result=[]
    for k in group:
        m=[cases[k,s][2] for s in FORMAL];times=[q['motion_to_inspect_s'] for q in m if q['success'] and q['motion_to_inspect_s'] is not None]
        solves=[v for q in m for v in q['solve_wall_s']]
        frontier=[]
        for seed in FORMAL:
            with np.load(cases[k,seed][1]/'series.npz') as r:frontier.append(float(np.max(r['qpos'][:,0])))
        result.append(dict(method=k,runs=3,successes=sum(q['success'] for q in m),furthest_x_m=median(frontier),
              worst_external_mm=1000*min(q['external_lower_m'] for q in m),
              worst_self_mm=1000*min(q['hazard_lower_m'] for q in m),
              movement_s=median(times) if times else None,
              rotor_moment_rms_Nm=median(q['rotor_roll_pitch_rms_Nm'] for q in m if q['rotor_roll_pitch_rms_Nm'] is not None) if any(q['rotor_roll_pitch_rms_Nm'] is not None for q in m) else None,
              solver_median_s=median(solves) if solves else None,solver_max_s=max(solves) if solves else None,
              pause_s=median(q['planner_pause_s'] for q in m),
              activation_count=median(q['activated_replans'] for q in m),
              accel_jump_m_s2=median(q['handoff_accel_jump_m_s2'] for q in m),
              bendrate_jump_deg_s=float(np.degrees(median(q['handoff_brate_jump_rad_s'] for q in m))),
              trials=[dict(seed=q['seed'],success=q['success'],raw_ulog_sha256=q['raw_ulog_sha256'],
                       metrics_sha256=sha(cases[k,q['seed']][1]/'metrics.json')) for q in m]))
    return result

def text_value(v,decimals):
    return '—' if v is None else ('%.*f'%(decimals,v))

def table(group,rows,savepath):
    if group=='A':
        columns=[('Furthest x\n(m)','furthest_x_m',2),('Min. external\ngap (mm)','worst_external_mm',1),
                 ('Min. self\ngap (mm)','worst_self_mm',1),('Rotor moment\nRMS (N m)','rotor_moment_rms_Nm',3),
                 ('To inspection\n(s)','movement_s',2)]
        caption='Geometry and dynamic coupling';foot='Clearances are the worst over all three flights. Other continuous values are medians. Time uses successful inspections.'
    else:
        columns=[('Switches','activation_count',0),('Solve time\nmedian (s)','solver_median_s',2),
                 ('Solve time\nmax. (s)','solver_max_s',2),('Planning\npause (s)','pause_s',2),
                 ('Acceleration\njump (m/s²)','accel_jump_m_s2',2),('Bend-rate\njump (°/s)','bendrate_jump_deg_s',2),('To inspection\n(s)','movement_s',2)]
        caption='Rolling execution and handoff';foot='Solve times include rejected requests. Each jump is the largest per flight, followed by the median across seeds.'
    h=1.55+.29*len(rows);fig,ax=plt.subplots(figsize=(9.7,h));ax.set_axis_off()
    labels=['Planner','Pass']+[a[0] for a in columns]
    cells=[[NAMES[r['method']],str(r['successes'])+'/3']+[text_value(r[k],d) for _,k,d in columns] for r in rows]
    widths=[.208,.075]+[(1.-.283)/len(columns)]*len(columns)
    t=ax.table(cellText=cells,colLabels=labels,colWidths=widths,loc='center',cellLoc='center',bbox=(0,.20,1,.69))
    t.auto_set_font_size(False);t.set_fontsize(8)
    for (i,j),c in t.get_celld().items():
        c.visible_edges='TB' if i==0 else 'B' if i==len(rows) else ''
        c.set_linewidth(.65 if i==0 else .38);c.set_edgecolor('#859598')
        if j==0:c.set_text_props(ha='left')
        if i==0:c.set_text_props(weight='semibold',color=INK)
        elif rows[i-1]['method']=='coupled':c.set_text_props(weight='semibold',color=TEAL)
    ax.text(0,.985,'Table '+group+'    '+caption,transform=ax.transAxes,va='top',ha='left',weight='semibold',fontsize=11,color=INK)
    ax.text(0,.07,foot,transform=ax.transAxes,fontsize=7.7,color='#526568')
    fig.subplots_adjust(left=.02,right=.98,top=.98,bottom=.02)
    fig.savefig(savepath);plt.close(fig)

def window_events(r):
    events=r.events();sub={e['id']:e for e in events if e['event']=='submit'}
    activations=[e for e in events if e['event']=='activate']
    return sub,activations

def rolling(records,file):
    fig=plt.figure(figsize=(9.8,6.5),constrained_layout=True);gs=fig.add_gridspec(3,3,height_ratios=(1.0,1.,1.1))
    a=fig.add_subplot(gs[0,:]);b=fig.add_subplot(gs[1,:],sharex=a)
    full=records['coupled'];start=full.control_summary['motion_start_s'];t=full.series['t']-start
    submissions,events=window_events(full)
    for j,e in enumerate(events):
        request=submissions[e['id']];s=request['submit_release_s']-start
        next_s=events[j+1]['release_s']-start if j+1<len(events) else float(t[-1])
        a.hlines(request['map']['known_frontier'],max(0,s),next_s,color='#b5c6c9',lw=2.2)
        a.vlines(max(0,s),request['map']['origin'][0],request['map']['known_frontier'],lw=.6,color='#96b0b4',ls=':')
        a.plot(e['release_s']-start,request['map']['known_frontier'],'|',color=TEAL,ms=7)
    a.plot(t,full.series['qpos'][:,0],color=TEAL,lw=1.4,label='Recorded vehicle position')
    a.plot([],[],color='#b5c6c9',lw=2,label='Frontier of the accepted local map')
    a.set(ylabel='Corridor x (m)',xlim=(0,max(r.control['t'][-1]-start for r in records.values())))
    a.set_title('a   Observed corridor and active planning windows',loc='left');a.legend(frameon=False,ncol=2,fontsize=8)
    for k,r in records.items():
        s=r.series;v=np.linalg.norm(np.gradient(s['qpos'][:,:3],s['t'],axis=0),axis=1)
        b.plot(s['t']-start,v,color=COLORS[k],lw=1.2,label=NAMES[k])
    b.set(xlabel='Time from motion onset (s)',ylabel='Body speed (m/s)')
    b.set_title('b   Measured progress through successive handoffs',loc='left');b.legend(frameon=False,ncol=3,fontsize=8)
    selected=[e for e in events if e['source_id']>=0 and e['handoff']['kind']=='C2_quintic'][:3]
    for j in range(3):
        ax=fig.add_subplot(gs[2,j])
        if j>=len(selected):
            ax.axis('off');ax.text(.5,.45,'No additional C² handoff',ha='center',transform=ax.transAxes,color='#657779');continue
        event=selected[j];stamp=event['release_s'];pos=np.interp(stamp,full.series['t'],full.series['qpos'][:,0])
        for k,r in records.items():
            _,items=window_events(r)
            items=[e for e in items if e['source_id']>=0]
            if not items:continue
            e=min(items,key=lambda h:abs(np.interp(h['release_s'],r.series['t'],r.series['qpos'][:,0])-pos))
            t=r.series['t']-e['release_s'];ids=(t>=-.45)&(t<=.80)
            moment=np.linalg.norm(r.series['rotor_moment_flu'][:,:2],axis=1)
            ax.plot(t[ids],moment[ids],color=COLORS[k],lw=1.15)
        ax.axvline(0,color='#889b9e',lw=.65,ls=':');ax.set(xlabel='Local switch time (s)',xlim=(-.45,.80))
        if j==0:ax.set_ylabel('Rotor moment (N m)')
        ax.set_title(chr(99+j)+'   Handoff near x = %.2f m'%pos,loc='left')
    for ax in fig.axes:ax.grid(axis='y',alpha=.15,lw=.5)
    for ext in ('pdf','png','svg'):fig.savefig(file.with_suffix('.'+ext),dpi=250,facecolor='white')
    plt.close(fig)

def coupled_curves(a,b,path):
    fig,(l,r)=plt.subplots(1,2,figsize=(9.7,1.75),constrained_layout=True)
    for record in (a,b):
        s=record.series;t=s['t']-record.control_summary['motion_start_s'];sel=t>=0
        axes=(l,r);ys=(s['tilt'],np.linalg.norm(s['mount_moment'],axis=1))
        for ax,y in zip(axes,ys):ax.plot(t[sel],y[sel],lw=.9,color=COLORS[record.method],label=NAMES[record.method])
    l.set(ylabel='Body tilt (°)',xlabel='Motion time (s)');r.set(ylabel='Mount moment (N m)',xlabel='Motion time (s)')
    for ax in (l,r):ax.grid(axis='y',alpha=.15)
    l.legend(frameon=False,fontsize=6.8,loc='upper right');fig.savefig(path);plt.close(fig)

def review(out,seed):
    import fitz
    pdf=fitz.open()
    def place(page,file,rect):
        with fitz.open(str(out/file)) as doc:page.show_pdf_page(fitz.Rect(*rect),doc,0)
    for i,(title,subtitle) in enumerate([
        ('Whole-body passage and manipulator loading','Three orthogonal restrictions and a surveyed optical inspection task.'),
        ('Online replanning and verified handoff','Finite sensing horizon and measured reactions at spatially matched switches.')]):
        page=pdf.new_page(width=600,height=850)
        page.insert_text((26,35),title,fontname='Times-Bold',fontsize=20,color=(.13,.20,.21))
        page.insert_text((27,53),subtitle,fontname='Times-Roman',fontsize=10)
        page.draw_line((26,64),(574,64),color=(.5,.61,.62),width=.6)
        if i==0:
            place(page,'measured-corridor.pdf',(23,72,577,465));place(page,'coupling-curves.pdf',(26,472,574,586))
            place(page,'group-A.pdf',(23,593,577,802))
        else:
            place(page,'rolling-handoffs.pdf',(24,80,576,492));place(page,'group-B.pdf',(23,495,577,690))
            page.insert_textbox((29,710,570,782),
                'The three close-ups align each method at the activation nearest the full method\'s longitudinal station. '
                'Time zero is the activation recorded in that flight. The full method is shared across the two comparisons.',
                fontsize=10.1,fontname='Times-Roman',lineheight=1.3,color=(.18,.25,.26))
        page.insert_text((27,823),'Paired closed-loop flight experiments   |   Representative seed '+str(seed),fontsize=8,fontname='Times-Roman')
        page.insert_text((560,823),str(i+1),fontname='Times-Roman',fontsize=9)
    pdf.save(str(out/'M2_三关全身规划与滚动切换_两页审阅.pdf'),garbage=4,deflate=True);pdf.close()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('manifest',type=Path);ap.add_argument('out',type=Path)
    args=ap.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    cases=matrix(args.manifest);seed=FORMAL[0]
    plt.rcParams.update({'font.family':'serif','font.serif':['STIX Two Text','DejaVu Serif'],'axes.spines.top':False,'axes.spines.right':False,
        'axes.labelsize':8,'axes.titlesize':10,'font.size':8,'pdf.fonttype':42,'svg.fonttype':'none'})
    for name,group in [('A',GROUP_A),('B',GROUP_B)]:
        rows=summarize(cases,group);table(name,rows,out/('group-'+name+'.pdf'))
        (out/('group-'+name+'.json')).write_text(json.dumps(dict(seeds=list(FORMAL),summary=rows),indent=2)+'\n',encoding='utf8')
    selected={m:FlightRecord(*cases[m,seed][:2]) for m in ('coupled','stop','hard','unregularized')}
    from m2_measured_scene import make
    make(selected['coupled'],out);rolling({k:selected[k] for k in GROUP_B},out/'rolling-handoffs')
    coupled_curves(selected['unregularized'],selected['coupled'],out/'coupling-curves.pdf')
    review(out,seed)
    return out

if __name__=='__main__':main()
