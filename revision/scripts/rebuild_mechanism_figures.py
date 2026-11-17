#!/usr/bin/env python3
"""Rebuild the two mechanism figures from the frozen representative run.

No planner/simulator is executed. Geometry is the vendored e62b6f2 geometry;
measurements and commands are read separately. Run from any directory.
"""
import argparse, csv, hashlib, json, math, sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'revision/frozen_geometry'))
from uav_continuum_whole_body_v2.geometry import world_arm_points_batch, world_arm_sections_batch, world_body_points_batch
from uav_continuum_whole_body_v2.environment import BoxWorld
from uav_continuum_whole_body_v2.types import ArmModel, Limits

BLUE='#285f7f'; ORANGE='#b45d39'; GREEN='#327568'; GRAY='#737d83'; INK='#20272d'
DATA=ROOT/'revision/data'; FIG=ROOT/'paper/figures'; LIVE=ROOT/'revision/live'

def status(stage,detail,progress,**kw):
    p=LIVE/'status.json'; d=json.loads(p.read_text()) if p.exists() else {}
    d.update(stage=stage,detail=detail,progress=progress,updated=datetime.now(timezone.utc).isoformat(),**kw)
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(d,ensure_ascii=False,indent=2));temp.replace(p)
    print(stage+': '+detail,flush=True)

def read_jsonl(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def yaw(q):
    x,y,z,w=q;return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))

def series(rows,start,finish):
    valid=[r for r in rows if start<=r['stamp']<=finish and r.get('arm_deg') and None not in r['arm_deg']]
    # Stable sort; keep the last sample at a repeated stamp, no smoothing.
    unique={float(r['stamp']):r for r in valid}
    ordered=[unique[t] for t in sorted(unique)]
    stamps=np.array([r['stamp'] for r in ordered])
    states=np.array([list(r['position'])+[yaw(r['orientation'])]+list(np.deg2rad(r['arm_deg'])) for r in ordered])
    return stamps,states,{'raw_valid':len(valid),'unique':len(ordered),'duplicate_policy':'last sample at identical stamp','max_gap_s':float(np.max(np.diff(stamps)))}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,required=True);a=parser.parse_args()
    source=a.source_root.resolve();DATA.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
    formal=source/'xtdrone_plan/experiments/method_comparison/v21_e3_final_450_commite62b6f2'
    runs=list(formal.glob('multi_passage_hard_s011__*/closed_loop_runs/multi_passage_hard_s011__rolling_wb_v2__*'))
    assert len(runs)==1, 'Expected exactly the previously selected representative run'
    run=runs[0];manifest=json.loads((run/'manifest.json').read_text())
    def resolve_source(old):
        p=Path(old);marker='xtdrone_plan/';return source/marker/old.split(marker,1)[1]
    oracle_path=resolve_source(manifest['oracle_file']);oracle=yaml.safe_load(oracle_path.read_text())
    ev=read_jsonl(run/'events.jsonl')
    start=min(e['actual_start'] for e in ev if e['state']=='START_SCHEDULED')
    finish=max(e['stamp'] for e in ev if e['state']=='TASK_COMPLETE')
    status('提取原始日志','按实际下发命令、实际状态、窗口事件分别重建。',25)
    t,states,tracking_audit=series(read_jsonl(run/'tracking.jsonl'),start,finish)
    ct,commands,command_audit=series(read_jsonl(run/'commands.jsonl'),start,finish)
    arm=ArmModel();limits=Limits(body_radius=.28,body_height=.25,body_margin=.10,arm_margin=.07)
    world=BoxWorld.from_dicts(oracle['boxes'])
    points=world_arm_points_batch(states,arm);body=world_body_points_batch(states,limits)
    bc,_=world.minimum_distances_grouped(body);ac,_=world.minimum_distances_grouped(points);ac-=arm.radius
    gate_y=sorted(set(float(b.center[1]) for b in world.boxes),reverse=True)
    indices=np.array([np.argmin(abs(states[:,1]-v)) for v in gate_y])
    straight=states[indices].copy();straight[:,4:6]=0
    sp=world_arm_points_batch(straight,arm);sc,_=world.minimum_distances_grouped(sp);sc-=arm.radius
    pairs=[];gate_rows=[]
    straight_drop=-world_arm_points_batch(np.zeros((1,6)),arm)[0,:,2].min()
    for k,(yy,idx) in enumerate(zip(gate_y,indices)):
        boxes=[b for b in world.boxes if abs(b.center[1]-yy)<1e-8]
        lower=next(b for b in boxes if b.name.endswith('_lower'));upper=next(b for b in boxes if b.name.endswith('_upper'))
        lo=float(lower.center[2]+lower.half[2]);hi=float(upper.center[2]-upper.half[2])
        zmin=lo+.02+straight_drop+arm.radius+limits.arm_margin
        zmax=hi-.02-limits.body_height/2-limits.body_margin
        normal=np.array([np.sin(lower.yaw),-np.cos(lower.yaw),0.])
        center=np.array([lower.center[0],lower.center[1],0.])
        projected=(np.concatenate([points,body],axis=1)-center)@normal
        overlap=(projected.min(axis=1)<=lower.half[1])&(projected.max(axis=1)>=-lower.half[1])
        pairs.append(dict(name=f'G{k+1}',y=yy,center=center,normal=normal,lower=lo,upper=hi,half=float(lower.half[1]),yaw=float(lower.yaw),zmin=float(zmin),zmax=float(zmax)))
        gate_rows.append(dict(gate=f'G{k+1}',sample_rule='nearest recorded UAV world-y to gate center; not an interpolated crossing',stamp=float(t[idx]),elapsed_s=float(t[idx]-start),x_m=states[idx,0],y_m=states[idx,1],z_m=states[idx,2],yaw_rad=states[idx,3],theta_deg=np.rad2deg(states[idx,4]),phi_deg=np.rad2deg(states[idx,5]),arm_clearance_m=ac[idx],straight_arm_clearance_m=sc[k],body_clearance_m=bc[idx],straight_uav_min_m=zmin,body_uav_max_m=zmax,altitude_interval_gap_m=zmin-zmax,passage_overlap_first_s=float(t[overlap][0]-start),passage_overlap_last_s=float(t[overlap][-1]-start),passage_overlap_arm_min_m=float(ac[overlap].min()),passage_overlap_body_min_m=float(bc[overlap].min())))
    # Check against the already published audit, independent of plotting choices.
    np.testing.assert_allclose(ac[indices],[.2158669267,.2338839745,.1905800964],atol=1e-8,rtol=0)
    np.testing.assert_allclose(sc,[-.0474646754,-.0325092682,-.0483666152],atol=1e-8,rtol=0)
    np.testing.assert_allclose([bc.min(),ac.min()],[.1888515485,.1886649478],atol=1e-8,rtol=0)
    safe={};switch={}
    for e in ev:
        if e['state']=='SAFE_TRAJECTORY' and e.get('trajectory_id'):
            i=int(e['trajectory_id'])
            if i not in safe or len(e)>len(safe[i]):safe[i]=e
        if e['state']=='SWITCHED_ROLLING' and e.get('trajectory_id'):
            i=int(e['trajectory_id'])
            if i not in switch or len(e)>len(switch[i]):switch[i]=e
    metadata={int(p.stem.split('_')[-1]):json.loads(p.read_text()) for p in run.glob('trajectory_*.json')}
    ids=sorted(metadata);assert ids==list(range(1,8)) and set(safe)==set(ids) and set(switch)==set(ids[1:])
    active=[start]+[switch[i]['actual_start'] for i in ids[1:]]
    window_rows=[]
    for k,i in enumerate(ids):
        e=safe[i];sw=switch.get(i,{});bridge=sw.get('switch_bridge',{});prev=metadata.get(i-1)
        window_rows.append(dict(window=i,safe_elapsed_s=e['stamp']-start,active_elapsed_s=active[k]-start,end_elapsed_s=(active[k+1] if k+1<len(active) else finish)-start,compute_wall_s=e['total_planning_time'],seed_wall_s=e['seed_time'],refine_validate_wall_s=e['optimize_validate_time'],nominal_metadata_reserve_s=(prev['execution_start_time']+prev['duration']-e['stamp']) if prev else '',handoff='initial' if i==1 else ('C2 bridge' if bridge.get('status')=='SAFE' else 'direct C1'),bridge_duration_s=bridge.get('duration_s',0),matched_elapsed_s=sw.get('matched_elapsed',''),preserved_phase_rate=sw.get('preserved_phase_rate','')))
    # ZOH associates measurements with the latest logged issued command. No future command interpolation.
    j=np.searchsorted(ct,t,side='right')-1;valid=j>=0;j=np.maximum(j,0);age=t-ct[j];valid &= age<=.15
    issued=commands[j];err=np.linalg.norm(states[:,:3]-issued[:,:3],axis=1);err[~valid]=np.nan
    time_rows=[]
    for k in range(len(t)):
        time_rows.append(dict(elapsed_s=t[k]-start,stamp=t[k],x_m=states[k,0],y_m=states[k,1],z_m=states[k,2],theta_actual_deg=np.rad2deg(states[k,4]),phi_actual_deg=np.rad2deg(states[k,5]),theta_issued_deg=np.rad2deg(issued[k,4]) if valid[k] else '',phi_issued_deg=np.rad2deg(issued[k,5]) if valid[k] else '',issued_age_s=age[k],association_valid=bool(valid[k]),position_error_m=err[k] if valid[k] else '',body_clearance_m=bc[k],arm_clearance_m=ac[k]))
    write_csv(DATA/'gate_samples.csv',gate_rows);write_csv(DATA/'window_events.csv',window_rows);write_csv(DATA/'issued_actual_series.csv',time_rows)
    np.savez_compressed(DATA/'representative_series.npz',time=t-start,states=states,command_time=ct-start,commands=commands,points=points,body_clearance=bc,arm_clearance=ac,gate_indices=indices)
    inputs=[run/'manifest.json',run/'tracking.jsonl',run/'commands.jsonl',run/'events.jsonl',oracle_path]+list(run.glob('trajectory_*.json'))+list((ROOT/'revision/frozen_geometry').rglob('*.py'))
    audit={'run_name':run.name,'algorithm_commit':manifest['commit'],'task_start_stamp':start,'task_finish_stamp':finish,'duration_s':finish-start,'arm':asdict(arm),'limits':asdict(limits),'tracking':tracking_audit,'command':command_audit,'timebase':'ROS/simulation stamp for execution; total_planning_time is wall-clock duration, shown only by window index','clearance_definition':'Frozen yaw-based sampled body envelope and PCC/base/tool centerline minus arm radius; post-hoc oracle geometry. Not exact CAD surface distance.','gate_sample_rule':gate_rows[0]['sample_rule'],'command_association':'Last issued command at or before measurement stamp, maximum age 0.15 s; no smoothing or time shifting.','nominal_reserve_definition':'Previous trajectory metadata start + duration - successor SAFE stamp; not phase-governor corrected physical buffer countdown.','body_min_m':float(bc.min()),'arm_min_m':float(ac.min()),'gate_count':len(gate_rows),'window_count':len(window_rows),'verified_published_numbers':True,'valid_command_association_fraction':float(valid.mean()),'inputs':[{'path':str(p.relative_to(source)) if source in p.parents else str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs]}
    (DATA/'geometry_and_timebase.json').write_text(json.dumps(audit,indent=2))
    replay={'gates':[{'name':p['name'],'s':-(p['y']-states[0,1]),'lower':p['lower'],'upper':p['upper']} for p in pairs],'frames':[{'t':float(t[k]-start),'s':float(-(states[k,1]-states[0,1])),'z':float(states[k,2]),'theta':float(np.rad2deg(states[k,4])),'arm':np.column_stack((-(points[k,:,1]-states[0,1]),points[k,:,2])).round(5).tolist()} for k in range(0,len(t),2)]}
    (LIVE/'replay.json').write_text(json.dumps(replay,separators=(',',':')))
    status('原数据核对通过','三门净空和全程最小净空与原审计一致；正在排版机制图。',45,replay='replay.json',items=['三门关键状态已核对','七窗按 trajectory_id 关联','实际指令与状态分开读取'])
    style();make_geometry(t-start,states,points,body,indices,sp,pairs,gate_rows)
    status('几何机制图已生成','正在生成窗口接续、下发指令和净空时序。',55,geometry='passage_mechanism_v2.png')
    make_rolling(t-start,states,ct-start,commands,bc,ac,window_rows,gate_rows)
    status('两张图初稿已生成','正在检查论文尺寸下的文字、局部图与事件标记。',65,rolling='rolling_execution_v2.png')
    print(json.dumps({'body_min':float(bc.min()),'arm_min':float(ac.min()),'gates':gate_rows,'windows':window_rows},indent=2))

def style():
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8.3,'axes.labelsize':8.3,'axes.titlesize':9,'xtick.labelsize':7.6,'ytick.labelsize':7.6,'legend.fontsize':7.5,'axes.linewidth':.65,'lines.linewidth':1.3,'pdf.fonttype':42,'ps.fonttype':42,'mathtext.fontset':'dejavuserif','axes.spines.top':False,'axes.spines.right':False,'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK})

def save(fig,name):
    for ext in ['pdf','svg','png']:
        fig.savefig(FIG/(name+'.'+ext),dpi=220,facecolor='white')
    (LIVE/(name+'.png')).write_bytes((FIG/(name+'.png')).read_bytes())
    plt.close(fig)

def title(ax,label,text):ax.set_title(label+'  '+text,loc='left',fontweight='bold',pad=8)

def make_geometry(time,states,points,body,indices,sp,pairs,rows):
    fig=plt.figure(figsize=(7.16,5.3));gs=fig.add_gridspec(3,3,height_ratios=[.86,1.4,.83],hspace=.65,wspace=.30,left=.075,right=.985,top=.945,bottom=.10)
    ax=fig.add_subplot(gs[0,:]);title(ax,'a','Recorded passage and morphology')
    y0=states[0,1];progress=-(states[:,1]-y0)
    for p in pairs:
        s=-(p['y']-y0)
        for low,high in [(0,p['lower']),(p['upper'],2.48)]:
            ax.add_patch(Rectangle((s-.12,low),.24,high-low,facecolor='#e0e3e5',edgecolor='#91999f',lw=.6))
        ax.text(s,2.55,p['name'],ha='center',fontsize=8)
    ax.plot(progress,states[:,2],color=BLUE,lw=1,label='UAV trajectory')
    selected=[int(np.argmin(abs(progress-.55)))]+indices.tolist()+[int(np.argmin(abs(progress-8.7)))]
    for k in selected:
        xx=progress[k];zz=states[k,2]
        ax.add_patch(Rectangle((xx-.28,zz-.125),.56,.25,facecolor='#dbe7ed',edgecolor=BLUE,lw=.7,zorder=4))
        ax.plot(-(points[k,:,1]-y0),points[k,:,2],color=GREEN,lw=1.9,zorder=5)
        ax.plot(-(points[k,-1,1]-y0),points[k,-1,2],'o',ms=2.5,color=INK,zorder=6)
    ax.set_xlim(-.2,10);ax.set_ylim(.55,2.7);ax.set_yticks([1,2]);ax.set_ylabel('World z (m)');ax.set_xlabel('Forward progress (m)',labelpad=1);ax.grid(axis='y',alpha=.12)
    for k,(p,idx,row) in enumerate(zip(pairs,indices,rows)):
        ax=fig.add_subplot(gs[1,k]);title(ax,chr(ord('b')+k),p['name']+' local section')
        for low,high in [(.45,p['lower']),(p['upper'],1.94)]:
            ax.add_patch(Rectangle((-p['half'],low),2*p['half'],high-low,fc='#e2e4e6',ec='#8f979d',lw=.7,zorder=0))
        project=lambda pts:(pts-p['center'])@p['normal']
        actual_x=project(points[idx]);straight_x=project(sp[k]);bx=float(project(states[idx,:3]))
        ax.add_patch(Rectangle((bx-.28,states[idx,2]-.125),.56,.25,fc='#e1ebf0',ec=BLUE,lw=.9,zorder=2))
        ax.plot(actual_x,points[idx,:,2],color=GREEN,lw=2.2,zorder=4)
        ax.plot(straight_x,sp[k,:,2],color=ORANGE,lw=1.25,ls=(0,(3,2)),zorder=5)
        ax.scatter([straight_x[-1]],[sp[k,-1,2]],marker='x',s=22,color=ORANGE,zorder=6)
        ax.text(.96,.95,r'$\theta=%.1f^\circ$'%row['theta_deg'],transform=ax.transAxes,ha='right',va='top',fontsize=7.7)
        ax.set_xlim(-.63,.63);ax.set_ylim(.58,1.92);ax.set_aspect('equal');ax.set_xticks([-.5,0,.5]);ax.set_xlabel('Gate-normal distance (m)',labelpad=2)
        if k==0:ax.set_ylabel('World z (m)')
        else:ax.tick_params(labelleft=False)
        ax.grid(alpha=.12)
    ax=fig.add_subplot(gs[2,:]);title(ax,'e','Arm clearance at the three recorded gate samples')
    for k,row in enumerate(rows):
        yy=2-k;a=100*row['arm_clearance_m'];b=100*row['straight_arm_clearance_m']
        ax.plot([b,a],[yy,yy],color='#b9c3c9',lw=1,zorder=1);ax.scatter(a,yy,color=GREEN,s=27,label='Recorded bent morphology' if k==0 else None,zorder=3);ax.scatter(b,yy,color=ORANGE,marker='s',s=25,label='Straight at the same UAV pose' if k==0 else None,zorder=3)
        ax.text(a+.6,yy,f'+{a:.1f}',va='center',fontsize=8,color=GREEN);ax.text(b-.6,yy,f'{b:.1f}',va='center',ha='right',fontsize=8,color=ORANGE)
    ax.axvspan(-8,0,color=ORANGE,alpha=.045);ax.axvline(0,color=INK,lw=.8);ax.axvline(7,color=GRAY,lw=.75,ls='--');ax.text(7.3,2.65,'7 cm planning margin',fontsize=7.3,color=GRAY)
    ax.set_yticks([2,1,0]);ax.set_yticklabels(['G1','G2','G3']);ax.set_ylim(-.5,2.6);ax.set_xlim(-8,27);ax.set_xticks([-5,0,5,10,15,20,25]);ax.set_xlabel('Signed model-based clearance (cm)',labelpad=2);ax.grid(axis='x',alpha=.12)
    fig.legend(*ax.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.52,.005),frameon=False,ncol=2,columnspacing=2)
    save(fig,'passage_mechanism_v2')

def make_rolling(time,states,ct,commands,bc,ac,windows,gates):
    fig=plt.figure(figsize=(7.16,5.5));gs=fig.add_gridspec(4,1,height_ratios=[.85,1.1,1.05,.95],hspace=.84,left=.10,right=.98,top=.945,bottom=.085)
    axes=[fig.add_subplot(gs[i]) for i in range(4)];end=time[-1]
    palette=['#92aebd','#7ca0b2','#608ca2','#3f758e','#357c80','#5c9385','#89a99b']
    ax=axes[0];title(ax,'a','Accepted windows and actual activation')
    for k,w in enumerate(windows):
        begin=w['active_elapsed_s'];stop=w['end_elapsed_s'];y=6-k
        ax.plot([max(0,w['safe_elapsed_s']),begin],[y,y],color=GRAY,lw=.9,ls=':')
        ax.broken_barh([(begin,stop-begin)],(y-.32,.64),facecolors=palette[k])
        ax.text((begin+stop)/2,y,'W'+str(w['window']),ha='center',va='center',fontsize=7.2,color=INK)
        if w['bridge_duration_s']:
            ax.broken_barh([(begin,w['bridge_duration_s'])],(y-.32,.64),facecolors=ORANGE)
    ax.set_ylim(-.6,6.6);ax.set_yticks([]);ax.set_xlim(0,end);ax.tick_params(labelbottom=False);ax.set_ylabel('Active window');ax.text(.99,1.01,'Dotted: accepted, waiting   Orange: C² bridge',transform=ax.transAxes,ha='right',fontsize=7.1,color=GRAY)
    ax=axes[1];title(ax,'b','Issued morphology and measured response')
    ax.plot(ct,np.rad2deg(commands[:,4]),color=BLUE,lw=1.25,ls='--',label='Issued command')
    ax.plot(time,np.rad2deg(states[:,4]),color=INK,lw=1.1,label='Measured state')
    ax.set_ylabel(r'Bending $\theta$ (deg)');ax.set_ylim(-5,130);ax.set_yticks([0,60,120]);ax.set_xlim(0,end);ax.tick_params(labelbottom=False);ax.legend(frameon=False,ncol=2,loc='upper right',bbox_to_anchor=(1,1.12));ax.grid(axis='y',alpha=.15)
    ax=axes[2];title(ax,'c','Post-hoc clearance in the critical range')
    ax.plot(time,bc*100,color=BLUE,label='Body');ax.plot(time,ac*100,color=GREEN,label='Arm')
    ax.axhline(6,color=BLUE,lw=.7,ls='--');ax.axhline(2,color=GREEN,lw=.7,ls='--');ax.set_ylim(0,50)
    ax.set_ylabel('Clearance (cm)');ax.set_xlim(0,end);ax.set_xlabel('Elapsed execution time (s)',labelpad=1);ax.legend(frameon=False,ncol=2,loc='upper right');ax.grid(axis='y',alpha=.15)
    ax.text(.015,.17,'Minimum: body 18.9 cm / arm 18.9 cm',transform=ax.transAxes,fontsize=7,color=GRAY)
    ax.text(.015,.04,'Dashed: audit thresholds (6 / 2 cm)',transform=ax.transAxes,fontsize=7,color=GRAY)
    for ax in axes[:3]:
        for w in windows[1:]:ax.axvline(w['active_elapsed_s'],color='#849097',lw=.45,ls=':',alpha=.7,zorder=0)
        for g in gates:ax.axvline(g['elapsed_s'],color='#a8b9b3',lw=.65,ls=(0,(5,3)),alpha=.8,zorder=0)
    for g in gates:axes[1].text(g['elapsed_s']+.8,10,g['gate'],fontsize=7,color=GREEN)
    ax=axes[3];title(ax,'d','Computation and nominal reserve by window')
    xx=np.arange(1,8);seed=np.array([w['seed_wall_s'] for w in windows]);rv=np.array([w['refine_validate_wall_s'] for w in windows]);reserve=[np.nan if w['nominal_metadata_reserve_s']=='' else w['nominal_metadata_reserve_s'] for w in windows]
    ax.bar(xx,seed,width=.46,color='#9eb5c1',label='Seed');ax.bar(xx,rv,bottom=seed,width=.46,color=BLUE,label='Refine + validate');ax.scatter(xx,reserve,marker='D',s=17,color=GREEN,label='Nominal metadata reserve')
    ax.axhline(15,color=GRAY,lw=.7,ls='--');ax.text(7.4,15,'15 s\ndeadline',fontsize=7,va='center',color=GRAY)
    for x,w in zip(xx,windows):ax.text(x,w['compute_wall_s']+.45,f"{w['compute_wall_s']:.2f}",ha='center',fontsize=7)
    ax.set_ylim(0,28);ax.set_xlim(.5,8.2);ax.set_xticks(xx);ax.set_xticklabels(['W'+str(x) for x in xx]);ax.set_ylabel('Duration (s)');ax.grid(axis='y',alpha=.12);ax.legend(frameon=False,ncol=3,loc='upper left',bbox_to_anchor=(0,1.06),fontsize=6.8)
    save(fig,'rolling_execution_v2')

if __name__=='__main__':main()
