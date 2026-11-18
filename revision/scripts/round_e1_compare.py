"""Evaluate the predeclared metrics from synchronized integrated states, retaining all cells."""
import csv
import json
import numpy as np
from round_e1_run import DEFAULT, P

def save_table(name,rows):
    with (DEFAULT/name).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)

def crossing(t,v,rising):
    idx=np.flatnonzero((v[:-1]<59)&(v[1:]>=59) if rising else (v[:-1]>=59)&(v[1:]<59))
    if len(idx)!=1: raise ValueError('expected exactly one half-amplitude crossing per side')
    k=int(idx[0])
    return float(t[k]+(59-v[k])*(t[k+1]-t[k])/(v[k+1]-v[k]))

def evaluate():
    data={};singles=[]
    for n in P['segments']:
        for dt in P['step_s']:
            label='n%02d_dt%04dus'%(n,round(dt*1e6))
            with np.load(str(DEFAULT/(label+'.npz'))) as src:
                x={k:src[k] for k in ('t','command','net','curvature','backbone','tcp','base_world','fractions')}
            t=x['t'];v=x['net'];u=x['command'];hold=(t>=7)&(t<=9)
            log=json.loads((DEFAULT/(label+'.json')).read_text())
            assert len(t)==2801 and np.all(np.isfinite(x['backbone'])) and np.all(np.isfinite(x['tcp']))
            valid=all(log[k]==0 for k in ('over120_steps','saturated_steps','contact_steps','solver_warnings','command_limited_steps'))
            s=dict(N=n,dt_ms=dt*1e3,peak_curvature_deg=log['peak_raw_curvature_deg'],
                rise_delay_ms=1000*(crossing(t,v,True)-crossing(t,u,True)),
                fall_delay_ms=1000*(crossing(t,v,False)-crossing(t,u,False)),
                hold_overshoot_deg=float(max(0.,max(v[hold])-118)),
                hold_bias_deg=float(np.mean(v[(t>=8.5)&(t<=9)]-118)),release_residual_deg=float(v[-1]),
                peak_torque_Nm=log['max_arm_torque_Nm'],contact_steps=log['contact_steps'],valid=bool(valid))
            singles.append(s);data[n,dt]=(x,s)
    comparisons=[];series=[]
    for tag,na,da,nb,db in [('coarse_vs_finer',8,.0005,16,.0005),('coarse_vs_finer',8,.0005,24,.0005),
                             ('fine_vs_fine',16,.0005,24,.0005)]+[('time_halving',n,.001,n,.0005) for n in (8,16,24)]:
        x,a=data[na,da];y,b=data[nb,db]
        np.testing.assert_allclose(x['t'],y['t'],atol=1e-12)
        np.testing.assert_allclose(x['fractions'],y['fractions'])
        gaps=1000*np.linalg.norm(x['backbone']-y['backbone'],axis=2)
        tcp=1000*np.linalg.norm(x['tcp']-y['tcp'],axis=1)
        ang=np.abs(x['net']-y['net'])
        kt,ks=np.unravel_index(np.argmax(gaps),gaps.shape)
        row=dict(test=tag,N_a=na,dt_a_ms=1000*da,N_b=nb,dt_b_ms=1000*db,backbone_end_mm=float(max(gaps[:,-1])),
                 tcp_mm=float(max(tcp)),centerline_mm=float(gaps[kt,ks]),worst_t_s=float(x['t'][kt]),
                 worst_s_over_L=float(x['fractions'][ks]),angle_deg=float(max(ang)),
                 rise_delta_ms=abs(a['rise_delay_ms']-b['rise_delay_ms']),fall_delta_ms=abs(a['fall_delay_ms']-b['fall_delay_ms']),
                 overshoot_delta_deg=abs(a['hold_overshoot_deg']-b['hold_overshoot_deg']))
        budget=P['selection'][tag]
        row['passed']=bool(a['valid'] and b['valid'] and max(row[k] for k in ('backbone_end_mm','tcp_mm','centerline_mm'))<=budget['position_mm']
                           and row['angle_deg']<=budget['angle_deg'] and max(row['rise_delta_ms'],row['fall_delta_ms'])<=budget['delay_ms']
                           and row['overshoot_delta_deg']<=budget['overshoot_deg'])
        comparisons.append(row)
        if tag!='time_halving':
            series.append((na,nb,np.max(gaps,axis=1),gaps[:,-1],tcp,ang))
    verdict='retain_8' if all(r['passed'] for r in comparisons) else ('consider_16' if all(r['passed'] for r in comparisons if r['test']=='fine_vs_fine' or (r['test']=='time_halving' and r['N_a'] in (16,24))) else 'hold')
    save_table('runs.csv',singles);save_table('comparisons.csv',comparisons)
    rows=[]
    for i,t in enumerate(data[8,.0005][0]['t']):
        row={'t_s':t,'command_deg':data[8,.0005][0]['command'][i]}
        for n in (8,16,24):
            x=data[n,.0005][0]
            row['bend_%d_deg'%n]=x['net'][i]
            for pt,arr in [('tcp',x['tcp'][i]),('end',x['backbone'][i,-1])]:
                for axis,v,v0 in zip('xyz',arr,x['base_world']):
                    row['%s_%d_world_%s_m'%(pt,n,axis)]=v
                    row['%s_%d_from_arm_origin_world_%s_m'%(pt,n,axis)]=v-v0
        for na,nb,cl,end,tcp,ang in series:
            for key,vals in [('centerline_mm',cl),('end_mm',end),('tcp_mm',tcp),('bend_difference_deg',ang)]:
                row['n%d_n%d_%s'%(na,nb,key)]=vals[i]
        rows.append(row)
    save_table('timeline.csv',rows)
    summary=dict(decision=verdict,runs=singles,comparisons=comparisons,scope='Fixed SUPER airframe, 0.28 m arm, simulated 0.20 kg tool assembly, no grasped object, phi=180 deg.')
    (DEFAULT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(verdict)
    for c in comparisons:print(c['test'],c['N_a'],c['N_b'],'end / TCP / line, mm',*[round(c[k],4) for k in ('backbone_end_mm','tcp_mm','centerline_mm')],c['passed'])
if __name__=='__main__': evaluate()
