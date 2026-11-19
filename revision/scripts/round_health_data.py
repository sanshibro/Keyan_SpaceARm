"""Independent bend/hold/return motions and paired simulated four-channel box readings."""
import argparse,json
import mujoco
import numpy as np
from round_health_flights import OUT,SPEC
from round_g1_model import make_plant
from round_e1_model import build,neutral
from arm_command import ArmCommand,MujocoArmBackend
from round_shape_sensors import stable_seed,marker_cluster,mocap_bend
from round_slim_box import OUT as SLIM,SPEC as SLIM_SPEC
from round_wrist_fusion import readj
from slim_sensor import SlimGaugeRig,SlimReadout

STATES=('healthy','offset','memory','mixed')
FLIGHT_STATES=STATES+('differential',)
PROFILE=readj(SLIM/'box-profile.json')
def motions():
    return {split:[(f'{split}-{i:02d}',a,i) for i,a in enumerate(SPEC['bench'][key])] for split,key in
        [('train','new_train_headings_deg'),('validation','validation'),('test','test')]}
def box_source(key,t,lengths,state,flight=False):
    rng=np.random.default_rng(stable_seed('health-install-'+key))
    # Paired distortions have the same stationary box noise and acquisition intervals.
    noise=np.load(SLIM/'stationary-box.npz')['events']
    box=SlimReadout(SLIM_SPEC,PROFILE,noise,stable_seed('health-box-'+key))
    bias=np.asarray(PROFILE['offset_stress_counts'])*rng.uniform(.65,1.35,4)
    if not flight and rng.uniform()<.5:bias=-bias
    if state=='differential':
        g=np.asarray(PROFILE['slope_counts_per_unit_ratio']);v=rng.uniform(.024,.036)
        bias=g*v*np.array([1,-.25,-1,.25])
    start=rng.uniform(28.,29.6) if flight else rng.uniform(.24,.38)*t[-1]
    ramp=rng.uniform(1.5,4.5)
    def injected(ts):
        return np.clip((ts-start)/ramp,0.,1.)[:,None]*bias
    memory=dict(half_width=float(rng.uniform(.014,.028)),tau_s=float(rng.uniform(1.3,2.8)),
        play_weight=float(rng.uniform(.34,.68)),transient_weight=float(rng.uniform(.012,.031)))
    d=box.sample(t,lengths,.3,memory=memory if state in ('memory','mixed') else None,
                 offset_profile=injected if state in ('offset','mixed','differential') else None)
    if d['overrange'].any():raise ValueError('Out-of-range strain must be handled before scoring')
    frac=np.clip((d['t']-start)/ramp,0,1) if state in ('offset','mixed','differential') else np.zeros(len(d['t']))
    label=np.full(len(frac),2 if state in ('memory','mixed') else 0,dtype=int)
    label[frac>=.85]+=1
    transition=(frac>.02)&(frac<.85)
    return d,label,transition,dict(onset_s=start,ramp_s=ramp,offset_codes=bias.tolist(),material=memory)
def geometry(model,d,rig,rng):
    mk=np.stack([marker_cluster(d,model.body(name).id) for name in ('mount','tool')])
    tracking=mk+rng.normal(0,.000025,mk.shape)
    return rig.lengths(d),tracking,mocap_bend(tracking)
def persist(folder,key,t,paths,markers,b,flight):
    folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/'reference.npz',t=t,length_m=paths,markers_m=markers,mocap_b=b)
    meta={}
    for name in FLIGHT_STATES if flight else STATES:
        d,lab,transition,info=box_source(key,t,paths,name,flight);meta[name]=info
        np.savez_compressed(folder/(name+'.npz'),t=d['t'],adc=d['adc'],tare=d['tare'],initial_adc=d['initial_adc'],rail=d['rail'])
        np.savez_compressed(folder/(name+'-labels.npz'),classes=lab,transition=transition)
    (folder/'generator.json').write_text(json.dumps(meta,indent=2))
def bench():
    m,_=build(8,.002);m.opt.disableflags|=int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    rig=SlimGaugeRig(m,SLIM_SPEC,.3)
    for split,lst in motions().items():
        for key,az,_ in lst:
            dest=OUT/'bench'/key
            if (dest/'generator.json').exists():continue
            rand=np.random.default_rng(stable_seed('health-motion-'+key));marks=np.random.default_rng(stable_seed('marker-'+key))
            d=neutral(m);arm=MujocoArmBackend(m,d,8)
            high=rand.uniform(62,94);middle=rand.uniform(33,48);low=rand.uniform(12,24)
            amp=np.array([0,0,high,high,middle,middle,low,high-6,high-6,middle,middle,low,low,0,0.])
            variation=rand.uniform(18,38)
            phi=np.deg2rad(az+variation*np.array([0,0,0,0,.5,.5,.75,-.3,-.3,.5,.5,.75,.75,0,0]))
            end=np.c_[amp*np.cos(phi),amp*np.sin(phi)]
            knots=np.array([0.,1.2,3.7,5.7,8.4,10.1,12.1,15.5,18.6,21.3,23.2,26.,28.4,31.3,34.])*rand.uniform(.82,1.18)
            t=np.arange(0,knots[-1]+.005,.02);paths=[];clusters=[];truth=[]
            for k,ts in enumerate(t):
                j=np.clip(np.searchsorted(knots,ts,side='right')-1,0,len(knots)-2)
                u=np.clip((ts-knots[j])/(knots[j+1]-knots[j]),0,1);u=u**3*(10-15*u+6*u*u)
                v=np.deg2rad((1-u)*end[j]+u*end[j+1])
                arm.apply(ArmCommand(k,ts,np.linalg.norm(v),np.arctan2(v[1],v[0])))
                mujoco.mj_forward(m,d);length,marker,b=geometry(m,d,rig,marks)
                paths.append(length);clusters.append(marker);truth.append(b)
                if k+1<len(t):
                    for _ in range(10):mujoco.mj_step(m,d)
            persist(dest,key,t,np.array(paths),clusters,truth,False)
            print('Bench motion',key,len(t),flush=True)
def flight():
    plant=make_plant();m,d=plant.m,plant.d;rig=SlimGaugeRig(m,SLIM_SPEC,.3)
    for entry in SPEC['flight']:
        key=entry['case'];dest=OUT/'records'/key/'strain'
        if (dest/'generator.json').exists():continue
        src=OUT/'runs'/key;run=readj(src/'run.json')
        with np.load(src/'physics.npz') as phys:
            t=(phys['us']-run['release_us'])/1e6;sel=(t>=0)&(t<46.31)
            t=t[sel];q=phys['qpos'][sel]
        marks=np.random.default_rng(stable_seed('marker-'+key));paths=[];clusters=[];truth=[]
        for qpos in q:
            d.qpos[:]=qpos;mujoco.mj_kinematics(m,d);length,marker,b=geometry(m,d,rig,marks)
            paths.append(length);clusters.append(marker);truth.append(b)
        persist(dest,key,t,np.array(paths),clusters,truth,True)
        print('Held-out strain packets',key,len(t),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('part',choices=['bench','flight']);o=ap.parse_args()
    (bench if o.part=='bench' else flight)()
