"""Fresh post-freeze MuJoCo bench motions: labels are for independent scoring only.

This entry point never fits or modifies the inverse. Four predetermined headings,
amplitudes, holds, reversals and plane changes exercise new complete trajectories.
"""
import argparse
from pathlib import Path
import numpy as np
import mujoco
from round_e1_model import build,neutral
from arm_command import ArmCommand,MujocoArmBackend
from round_shape_sensors import marker_cluster,mocap_bend
from slim_sensor import SlimGaugeRig
from continuum_strain_inverse import ContinuousStrainInverse
from round_aug_strain_calibrate import make_record,stats
from round_aug_strain_common import DESIGN,environment,savej,readj,sha,seed

def run(args):
    out=Path(args.output);model=out/'strain-model/model.json'
    frozen=readj(out/'strain-model/freeze.json')
    if sha(model)!=frozen['model_sha256']:raise ValueError('Frozen model hash changed')
    inverse=ContinuousStrainInverse(model);spec,profile,noise=environment(args.slim,args.protocol)
    m,_=build(8,.002);m.opt.disableflags|=int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    rig=SlimGaugeRig(m,spec,.3);rows={};estimates=[];references=[];covariances=[];prefix_max=0.
    protocol=dict(headings_deg=[13.,107.,221.,307.],peak_deg=[58.,77.,96.,105.],
        observation='New virtual mount/tool marker clusters at 25 micrometer noise; score only.',
        freeze_sha256=sha(model),use='Fresh post-freeze test; no refit or selected removal.')
    savej(out/'strain-model/independent-bench-protocol.json',protocol)
    for index,(heading,peak) in enumerate(zip(protocol['headings_deg'],protocol['peak_deg'])):
        name='test-independent-%02d'%index;folder=out/'strain-bench-independent'/name
        folder.mkdir(parents=True,exist_ok=True)
        rng=np.random.default_rng(seed('aug-new-bench:'+name));marks=np.random.default_rng(seed('aug-new-markers:'+name))
        knot=np.array([0.,1.4,4.8,7.8,10.7,13.4,16.9,20.2,23.6,26.1,29.7,32.8,36.])*(.90+.07*index)
        amp=peak*np.array([0,0,1,1,.42,.42,.77,.77,.20,.20,.88,0,0])
        phi=np.deg2rad(heading+np.array([0,0,0,0,27,27,-32,-32,39,39,-10,0,0]))
        bknots=np.deg2rad(amp)[:,None]*np.column_stack((np.cos(phi),np.sin(phi)))
        t=np.arange(0,knot[-1]+.005,.02);d=neutral(m);arm=MujocoArmBackend(m,d,8)
        lengths=[];truth=[]
        for k,stamp in enumerate(t):
            j=np.clip(np.searchsorted(knot,stamp,side='right')-1,0,len(knot)-2)
            f=np.clip((stamp-knot[j])/(knot[j+1]-knot[j]),0,1);f=f**3*(10-15*f+6*f*f)
            bend=(1-f)*bknots[j]+f*bknots[j+1]
            arm.apply(ArmCommand(k,stamp,np.linalg.norm(bend),np.arctan2(bend[1],bend[0])))
            mujoco.mj_forward(m,d);lengths.append(rig.lengths(d))
            markers=np.stack([marker_cluster(d,m.body(n).id) for n in ('mount','tool')])
            markers+=marks.normal(0,.000025,markers.shape);truth.append(mocap_bend(markers))
            if k+1<len(t):
                for _ in range(10):mujoco.mj_step(m,d)
        np.savez_compressed(folder/'reference.npz',t=t,length_m=lengths,mocap_b=truth)
        rows[name]={}
        for variation,label in [(.06,'nominal'),(.15,'material_transfer')]:
            rec=make_record(folder,spec,profile,noise,variation);raw=rec['raw']
            z=inverse.estimate(raw['t'],raw['adc'],raw['tare']);rows[name][label]=stats(z['b'],rec['truth'])
            np.savez_compressed(folder/(label+'-observation.npz'),t=raw['t'],adc=raw['adc'],tare=raw['tare'],**z)
            np.savez_compressed(folder/(label+'-scoring.npz'),t=raw['t'],mocap_b=rec['truth'])
            if label=='nominal':
                estimates.append(z['b']);references.append(rec['truth']);covariances.append(z['R'])
                cut=len(raw['t'])//2;p=inverse.estimate(raw['t'][:cut],raw['adc'][:cut],raw['tare'])
                prefix_max=max(prefix_max,float(np.max(abs(p['b']-z['b'][:cut]))),float(np.max(abs(p['R']-z['R'][:cut]))))
        rows[name]['geometry_sha256']=sha(folder/'reference.npz')
        print('New bench',name,rows[name]['nominal'],flush=True)
    est,ref,R=map(np.vstack,(estimates,references,covariances));e=est-ref
    nees=np.einsum('ni,nij,nj->n',e,np.linalg.inv(R),e)
    report=dict(model_sha256=sha(model),pooled_nominal=stats(est,ref),motions=rows,
        nominal_joint95_coverage=float(np.mean(nees<=5.991464547)),causal_prefix_max_abs=prefix_max,
        scope='Fresh post-freeze simulated bench motions with empirical box acquisition noise. '
              'Previous test-00..02 were development diagnostics after detecting a converter-rail bug. '
              'These four new motions did not train shape, material, or covariance parameters.')
    savej(out/'strain-model/independent-bench-validation.json',report)
    print('INDEPENDENT',report['pooled_nominal'],'prefix',prefix_max,flush=True)

if __name__=='__main__':
    root=Path(__file__).resolve().parents[2];p=argparse.ArgumentParser()
    p.add_argument('--output',default=root/'revision/data/round_augmented_fusion')
    p.add_argument('--slim',default=root/'revision/data/round_slim_sensor')
    p.add_argument('--protocol',default=root/'revision/protocols/round_slim_protocol.json')
    run(p.parse_args())
