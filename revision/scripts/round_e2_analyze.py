"""Compute paired sensitivity contrasts and numerical controls from all archived integrations."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from round_e1_compare import crossing
from round_e2_run import DEFAULT,F,P,ROOT,cases,save_json,tag

def write_table(file,rows):
    with file.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def largest_distance(a,b):
    return float(1000*np.linalg.norm(a-b,axis=1).max())

def time_shift(t,signal,u,rise):
    try:return float(1000*(crossing(t,signal,rise)-crossing(t,u,rise)))
    except ValueError:return None

def analyze(out):
    config=json.loads((out/'provenance.json').read_text())
    states={};runs={};matrix=[];logs=[]
    nominal=(8,.0005,1.,1.,0)
    for cell in cases():
        key=tag(*cell);p=out/(key+'.npz');doc=json.loads((out/(key+'.json')).read_text())
        assert doc['id']==key and hashlib.sha256(p.read_bytes()).hexdigest()==doc['state_sha256']
        with np.load(str(p),allow_pickle=False) as src:
            x={k:src[k] for k in ('t','net','command','curvature','tcp','end')}
            assert src['state_t'].shape[0]==src['qpos'].shape[0]==src['qvel'].shape[0]==src['ctrl'].shape[0]
        states[cell]=x;logs.append(doc)
    ref=states[nominal]
    with np.load(str(ROOT/'revision/data/round_e1/n08_dt0500us.npz')) as old:
        np.testing.assert_allclose(ref['t'],old['t'],rtol=0,atol=1e-10)
        reproduction=dict(max_tcp_mm=largest_distance(ref['tcp'],old['tcp']),
            max_net_deg=float(max(abs(ref['net']-old['net']))))
    reproduction['passed']=reproduction['max_tcp_mm']<=.001 and reproduction['max_net_deg']<=.0001
    for cell,doc in zip(cases(),logs):
        n,dt,k,c,g=cell;x=states[cell];t=x['t'];v=x['net'];u=x['command']
        mask=(t>=8.5)&(t<=9);hold=(t>=7)&(t<=9)
        assert np.array_equal(t,ref['t']) and len(t)==2801, 'incomplete run: '+doc['id']
        z=1000*float(np.mean(ref['tcp'][mask,2]-x['tcp'][mask,2]))
        row=dict(id=doc['id'],N=n,dt_ms=dt*1000,k_scale=k,d_scale=c,load_g=g,
                 K_Nm_rad=doc['K_Nm_rad'],D_Nms_rad=doc['D_Nms_rad'],hold_deficit_deg=float(np.mean(118-v[mask])),
                 rise_delay_ms=time_shift(t,v,u,True),fall_delay_ms=time_shift(t,v,u,False),
                 hold_overshoot_deg=float(max(0,np.max(v[hold])-118)),
                 max_tcp_vs_nominal_mm=largest_distance(ref['tcp'],x['tcp']),hold_tcp_down_vs_nominal_mm=z,
                 peak_total_deg=doc['peak_total_deg'],peak_torque_Nm=doc['peak_torque_Nm'],
                 contact_steps=doc['native_contact_steps'],saturated_steps=doc['saturated_steps'],
                 warnings=doc['solver_warnings'],over120_steps=doc['over120_steps'],physics_steps=doc['steps'],valid=doc['valid'])
        runs[cell]=row
        if n==8 and dt==.0005:matrix.append(row)
    wkeys=('hold_deficit_deg','rise_delay_ms','fall_delay_ms','hold_overshoot_deg')
    factors=[('stiffness',2,F['stiffness_scale']),('drive_damping',3,F['drive_damping_scale']),('added_payload',4,F['added_payload_g'])]
    paired=[];anchors={}
    for label,i,levels in factors:
        for low in cases():
            if low[:2]!=(8,.0005) or low[i]!=levels[0]: continue
            high=tuple(levels[-1] if j==i else v for j,v in enumerate(low))
            a,b=runs[low],runs[high];xa,xb=states[low],states[high]
            pair=dict(factor=label,low=a['id'],high=b['id'],matched_valid=a['valid'] and b['valid'])
            for metric in wkeys:
                pair['delta_'+metric]=(b[metric]-a[metric]) if a[metric] is not None and b[metric] is not None else None
            pair['max_paired_tcp_mm']=largest_distance(xa['tcp'],xb['tcp'])
            paired.append(pair)
            if all(v==nominal[j] for j,v in enumerate(low) if j!=i):anchors[label]=pair
    assert len(matrix)==27 and len(paired)==27 and len(anchors)==3
    checks=[]
    for cfg in P['crosschecks']:
        k,c,g=cfg['stiffness_scale'],cfg['drive_damping_scale'],cfg['added_payload_g']
        ca=(8,.0005,k,c,g);cb=(cfg['N'],cfg['dt_s'],k,c,g)
        x,y=states[ca],states[cb];a,b=runs[ca],runs[cb]
        name='coarse_fine' if cb[0]!=8 else 'time_halving';budget=P['limits'][name]
        row=dict(test=name,base=a['id'],comparison=b['id'],tcp_mm=largest_distance(x['tcp'],y['tcp']),
            backbone_end_mm=largest_distance(x['end'],y['end']),net_deg=float(max(abs(x['net']-y['net']))),
            t50_delta_ms=max(abs(a[metric]-b[metric]) for metric in ('rise_delay_ms','fall_delay_ms')))
        row['passed']=bool(a['valid'] and b['valid'] and all(row[q]<=v for q,v in budget.items()))
        checks.append(row)
    def item(k,c,g):return runs[(8,.0005,k,c,g)]
    load_effects={str(k):item(k,1.,100)['hold_deficit_deg']-item(k,1.,0)['hold_deficit_deg'] for k in (.75,1.,1.25)}
    spans={}
    for label,_,_ in factors:
        subset=[p for p in paired if p['factor']==label]
        spans[label]={}
        for name in ['delta_hold_deficit_deg','delta_rise_delay_ms','max_paired_tcp_mm']:
            a=np.array([p[name] for p in subset if p[name] is not None])
            spans[label][name]={'min':float(a.min()),'median':float(np.median(a)),'max':float(a.max())}
    summary=dict(matrix_size=len(matrix),all_runs=len(runs),physics_steps=sum(r['physics_steps'] for r in runs.values()),
        all_valid=all(r['valid'] for r in runs.values()),peak_total_deg=max(r['peak_total_deg'] for r in runs.values()),
        summed_contact_steps=sum(r['contact_steps'] for r in runs.values()),summed_saturated_steps=sum(r['saturated_steps'] for r in runs.values()),
        summed_over120_steps=sum(r['over120_steps'] for r in runs.values()),summed_warnings=sum(r['warnings'] for r in runs.values()),
        nominal_rerun=reproduction,numerical_controls=checks,anchored_contrasts=anchors,paired_spans=spans,
        load_effect_hold_deg_by_K=load_effects,load_stiffness_interaction_deg=load_effects['0.75']-load_effects['1.25'],
        worst_tcp_cell=max(matrix,key=lambda z:z['max_tcp_vs_nominal_mm']),nominal=item(1.,1.,0),heavy=item(1.,1.,100),
        low_damping=item(1.,.6,0),high_damping=item(1.,1.4,0),code=config['git'])
    save_json(out/'summary.json',summary);save_json(out/'model_checks.json',logs)
    write_table(out/'matrix.csv',list(runs.values()));write_table(out/'pairs.csv',paired);write_table(out/'checks.csv',checks)
    trajectories=[]
    for i,t in enumerate(ref['t']):
        row=dict(t_s=t,command_deg=ref['command'][i])
        for name,cell in [('nominal',nominal),('heavy',(8,.0005,1.,1.,100)),('soft',(8,.0005,.75,1.,0)),
                          ('damp_low',(8,.0005,1.,.6,0)),('damp_high',(8,.0005,1.,1.4,0)),('soft_heavy',(8,.0005,.75,1.,100))]:
            x=states[cell];row[name+'_net_deg']=x['net'][i];row[name+'_tcp_sep_mm']=1000*np.linalg.norm(x['tcp'][i]-ref['tcp'][i])
            for ax,val in zip('xyz',x['tcp'][i]):row[name+'_tcp_'+ax+'_m']=val
        trajectories.append(row)
    write_table(out/'aligned_trajectories.csv',trajectories)
    print('Nominal E1 rerun:',reproduction,'; all valid:',summary['all_valid'])
    for p in anchors.values(): print(p['factor'],p['delta_hold_deficit_deg'],p['delta_rise_delay_ms'],p['max_paired_tcp_mm'])
    print('controls',[(r['test'],r['tcp_mm'],r['passed']) for r in checks],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=DEFAULT);analyze(p.parse_args().out)
