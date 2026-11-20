"""Streaming adapters for frozen 881a799 sensing and bend estimation.

No fitting or controller logic lives here. The physical layer accepts instantaneous
surface lengths; the inference layer accepts ADC and arrived visual packets only.
All clocks are release-based seconds; b is radians and covariance is radians^2.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from continuum_strain_inverse import ContinuousStrainInverse,polynomial
from augmented_bend_kf import BendFilter,Settings,RAD
from slim_sensor import SlimGaugeRig,SlimReadout
from round_aug_strain_common import DESIGN,material_for,seed


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def readj(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def writej(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


class FrozenAssets:
    def __init__(self,root=None,data_dir=None,slim_dir=None,protocol_path=None,verify=True):
        self.root=Path(root) if root else Path(__file__).resolve().parents[2]
        self.data=Path(data_dir) if data_dir else self.root/'revision/data/round_augmented_fusion'
        self.slim=Path(slim_dir) if slim_dir else self.root/'revision/data/round_slim_sensor'
        self.protocol=Path(protocol_path) if protocol_path else self.root/'revision/protocols/round_slim_protocol.json'
        self.freeze=readj(self.data/'frozen-settings.json');self.model_path=self.data/'strain-model/model.json'
        if digest(self.model_path)!=self.freeze['strain_model']:raise ValueError('Frozen strain-model hash mismatch')
        if verify:
            for relative,expected in self.freeze['software'].items():
                if digest(self.root/relative)!=expected:raise ValueError('Frozen software changed: '+relative)
        self.settings=Settings(**self.freeze['settings'])
        self.spec=readj(self.protocol);self.profile=readj(self.slim/'box-profile.json')
        with np.load(self.slim/'stationary-box.npz',allow_pickle=False) as f:self.noise=f['events'].copy()
        self.model=readj(self.model_path)
        self.hashes=dict(frozen_settings=digest(self.data/'frozen-settings.json'),
                         strain_model=digest(self.model_path),protocol=digest(self.protocol),
                         box_profile=digest(self.slim/'box-profile.json'),
                         box_noise=digest(self.slim/'stationary-box.npz'))


class StreamingStrainInverse:
    """One-sample translation of frozen estimate(), with its states persisted.

The original class supplies parameter parsing, static_shape and solve_shape.
Arithmetic/state transitions below follow its two loops in the same order.
"""
    def __init__(self,model,tare):
        self.f=ContinuousStrainInverse(model);self.tare=np.asarray(tare,float)
        self.prev=np.full(4,self.f.neutral);self.play=self.prev.copy();self.transient=np.zeros(4)
        self.last_t=None;self.smoothed=None;self.last_velocity=np.zeros(2);self.reversal=0.

    def step(self,t,adc):
        f=self.f;adc=np.asarray(adc,float)
        if adc.shape!=(4,) or not np.isfinite(adc).all():raise ValueError('Expected four finite ADC codes')
        if self.last_t is not None and t<=self.last_t:raise ValueError('ADC clock moved backward')
        dt=t-self.last_t if self.last_t is not None else float(f.model.get('nominal_sample_s',.025))
        measured=f.neutral+(adc-self.tare)/f.gain
        decay=np.exp(-dt/f.tau);c=-np.expm1(-dt/f.tau)*f.tau/dt
        slope=1.-f.wp+f.wt*c;base=f.wt*(decay*self.transient-c*self.prev)
        center=(measured-f.wp*self.play-base)/slope
        lower=(measured-f.wp*f.h-base)/(1.+f.wt*c);upper=(measured+f.wp*f.h-base)/(1.+f.wt*c)
        value=np.where(center<self.play-f.h,lower,np.where(center>self.play+f.h,upper,center))
        value=np.clip(value,f.curve_r[0]-.10,f.curve_r[-1]+.10)
        e=np.interp(value,f.curve_r,f.curve_e);usable=(adc>1)&(adc<4094)
        shape=f.solve_shape(e,usable)
        if f.geometry_coef is not None and not np.all(usable):
            expected_e=polynomial(shape[None,:])[0]@f.geometry_coef
            expected_r=np.interp(expected_e,f.curve_e,f.curve_r)
            value=np.where(usable,value,expected_r)
        self.play=np.clip(self.play,value-f.h,value+f.h)
        self.transient=decay*self.transient+c*(value-self.prev);self.prev=value
        bend,common,strain=f.static_shape(value[None,:]);bend=shape if f.geometry_coef is not None else bend[0]
        common=common[0];strain=strain[0];point=f.static_shape(measured[None,:])[0][0]
        if self.smoothed is None:self.smoothed=bend.copy()
        velocity=(bend-self.smoothed)/.18;speed=np.linalg.norm(velocity)
        if velocity@self.last_velocity<0 and np.linalg.norm(self.last_velocity)>.025:self.reversal=1.
        self.reversal*=np.exp(-dt/.35)
        self.smoothed+=-np.expm1(-dt/.18)*(bend-self.smoothed);self.last_velocity=velocity
        residual_size=np.linalg.norm(common/f.common_scale)
        rate_factor=min(4.,(speed/.7)**2);common_factor=min(24.,max(0.,residual_size-2.)**2)
        edge_factor=max(0.,(np.linalg.norm(bend)-np.deg2rad(85.))/np.deg2rad(15.))**2
        rail=bool(np.any((adc<=1)|(adc>=4094)))
        factor=1.+.30*rate_factor+.5*self.reversal+.35*common_factor+2*edge_factor
        if rail:factor=max(factor,100.)
        self.last_t=float(t)
        return dict(t=float(t),b=bend.copy(),R=f.cov0*factor,point_b=point,
                    common_residual=common,material_corrected_ratio=value.copy(),
                    speed_rad_s=float(speed),reversal_score=float(self.reversal),estimated_strain=strain)


class StreamingBox:
    """SlimReadout.sample on a bounded preallocated native acquisition schedule.

Only sampling/noise are preallocated. Each output uses the two most recent actual
integrated geometric states bracketing its acquisition time, never a future path.
"""
    def __init__(self,spec,profile,events,seed_key,condition,horizon_s=100.,t0=0.,drift_window=(18.,22.)):
        if condition not in ('nominal','differential'):raise ValueError('This stage permits nominal or differential only')
        self.spec=spec;self.profile=profile;self.t0=float(t0);self.horizon=float(horizon_s)
        self.condition=condition;self.drift_window=tuple(map(float,drift_window));self.key=str(seed_key)
        if self.drift_window[1]<=self.drift_window[0]:raise ValueError('Invalid pre-gate ramp')
        self.material=material_for('flight:'+self.key,DESIGN['heldout_material_fractional_variation'])
        self.h,self.tau,self.wp,self.wt=[np.asarray(self.material[k]) for k in ('half_width','tau_s','play_weight','transient_weight')]
        self.reader=SlimReadout(spec,profile,events,seed('aug-box:'+self.key))
        self.gain=np.asarray(profile['slope_counts_per_unit_ratio']);self.zero=np.asarray(profile['neutral_codes'])
        self.free=spec['free_active_length_m'];self.preload=DESIGN['preload'];self.base=self.reader.ratio(self.preload)
        self.offset=np.asarray(DESIGN['offset_ratio'][condition])*self.gain
        rng=self.reader.rng;chunks=[];timestamps=[];elapsed=self.t0-.72
        while elapsed<=self.horizon:
            start=int(rng.integers(0,len(events)-32));extra=int(rng.integers(0,len(events)-32))
            donor=int(rng.integers(1,4));block=events[start:start+32]
            times=elapsed+np.cumsum(block[:,0]);elapsed=times[-1]
            timestamps.extend(times);chunks.append(np.column_stack((block[:,1:4],events[extra:extra+32,donor])))
        self.all_t=np.asarray(timestamps);self.all_noise=np.concatenate(chunks)
        keep=self.all_t<=self.horizon+1e-9;self.all_t=self.all_t[keep];self.all_noise=self.all_noise[keep]
        prior=self.all_t<self.t0-1e-8
        pre_codes=np.rint(np.clip(self.zero+self.all_noise[prior],0,4095)).astype(np.int16)
        self.tare=np.median(pre_codes,axis=0);self.initial_adc=pre_codes[-1]
        self.index=int(np.sum(prior));self.prev_t=None;self.prev_length=None;self.prev_ratio=None
        self.play=None;self.transient=np.zeros(4);self.prev_correction=np.zeros(4)
        self.geometry=[];self.records=[]

    def observe_lengths(self,t,length_m):
        t=float(t);L=np.asarray(length_m,float)
        if L.shape!=(4,) or not np.isfinite(L).all():raise ValueError('Four finite physical surface spans required')
        if t>self.horizon+1e-9:raise ValueError('Acquisition schedule exhausted; declare a longer horizon before run')
        if self.prev_t is not None and t<=self.prev_t:raise ValueError('Physical clock moved backward')
        ratio=self.reader.ratio(L/self.free-1.)
        if not np.isfinite(ratio).all() or np.any(L<self.free-1e-8):raise ValueError('Physical gauge outside tensile operating range')
        self.geometry.append((t,L.copy()))
        if self.prev_t is None:
            if abs(t-self.t0)>1e-7:raise ValueError('Supply release-zero geometry before advancing physics')
            self.play=ratio.copy();correction=np.zeros(4)
        else:
            dt=t-self.prev_t;delta=ratio-self.prev_ratio
            self.play=np.clip(self.play,ratio-self.h,ratio+self.h)
            decay=np.exp(-dt/self.tau)
            self.transient=decay*self.transient+delta*(-np.expm1(-dt/self.tau))*self.tau/dt
            correction=(self.play-ratio)*self.wp+self.transient*self.wt
        result=[]
        while self.index<len(self.all_t) and self.all_t[self.index]<=t+1e-12:
            stamp=self.all_t[self.index]
            if self.prev_t is None:sample_L=L;sample_c=correction
            else:
                u=(stamp-self.prev_t)/(t-self.prev_t)
                sample_L=self.prev_length+u*(L-self.prev_length)
                sample_c=self.prev_correction+u*(correction-self.prev_correction)
            extension=sample_L/self.free-1.;relative=np.maximum(0.,self.reader.ratio(extension)+sample_c)
            lo,hi=self.drift_window;z=np.clip((stamp-lo)/(hi-lo),0.,1.);z=z**3*(10.-15.*z+6.*z*z)
            analog=self.zero+(relative-self.base)*self.gain+self.all_noise[self.index]+z*self.offset
            adc=np.rint(np.clip(analog,0,4095)).astype(np.int16)
            record=dict(t=float(stamp),arrival_t=t,adc=adc,tare=self.tare.copy(),initial_adc=self.initial_adc.copy(),
                        rail=(adc==0)|(adc==4095))
            result.append(record);self.records.append(record);self.index+=1
        self.prev_t=t;self.prev_length=L.copy();self.prev_ratio=ratio;self.prev_correction=correction
        return result


class LiveGauge:
    def __init__(self,plant,seed_key,condition,assets=None,horizon_s=100.,drift_window=(18.,22.)):
        self.assets=assets or FrozenAssets();a=self.assets
        self.rig=SlimGaugeRig(plant.m,a.spec,DESIGN['preload'])
        self.box=StreamingBox(a.spec,a.profile,a.noise,seed_key,condition,horizon_s,0.,drift_window)
        self.inverse=StreamingStrainInverse(a.model,self.box.tare);self.observations=[]
    def sample(self,t,plant):
        events=self.box.observe_lengths(t,self.rig.lengths(plant.d))
        for event in events:
            event.update(self.inverse.step(event['t'],event['adc']));self.observations.append(event.copy())
        return events
    def save(self,folder):
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        t,L=zip(*self.box.geometry);np.savez_compressed(folder/'strain_geometry.npz',t=t,length_m=L)
        _save_rows(folder/'strain_observations.npz',self.observations)
        writej(folder/'strain_generation.json',dict(seed_key=self.box.key,condition=self.box.condition,
            material=self.box.material,offset_codes=self.box.offset.tolist(),ramp_s=self.box.drift_window,
            assets=self.assets.hashes,adapter_sha256=digest(__file__)))


def _advance(f,t,events,index,paired=0.):
    while index<len(events) and events[index]['t']<=t+1e-9:
        packet=events[index];f.strain(packet['t'],packet['b'],packet['R'],paired);index+=1
    f.predict(t);return index


class StreamingBend:
    def __init__(self,c,augmented,t0):
        self.c=c;self.augmented=augmented;self.master=BendFilter(c,augmented,t0);self.live=self.master.copy()
        self.ms=self.ls=self.ci=self.previous_end=0;self.last_visual=None
    def step(self,now,strain,arrived):
        got=False;nis=np.nan;used=False;gv=0.;cv=0.;gs=0.;cs=0.
        for packet in arrived:
            exp=packet['t'];quality=float(packet['quality']);reliable=bool(packet.get('ready',packet['valid']))
            support=bool(self.augmented and reliable and packet['valid'] and self.ci and self.master.used_camera
                         and exp-self.last_visual['t']<.085)
            prior=self.master.copy() if support else None;old_ms=self.ms
            self.ms=_advance(self.master,exp,strain,self.ms,float(support))
            self.master.camera(exp,packet['b'],packet['R'],bool(packet['valid']),quality,reliable)
            if support and not self.master.used_camera:
                self.master=prior;self.ms=_advance(self.master,exp,strain,old_ms)
                self.master.camera(exp,packet['b'],packet['R'],True,quality,reliable)
            nis=self.master.nis;used=self.master.used_camera;gv=self.master.gain_v
            cv=quality if packet['valid'] else 0.
            self.ci+=1;got=True;self.last_visual=packet
        if got:self.live=self.master.copy();self.ls=self.ms
        self.ls=_advance(self.live,now,strain,self.ls)
        bend=np.full(2,np.nan);cov=np.zeros((2,2));bias=np.zeros(2)
        if self.ls:
            bend=self.live.x[:2].copy();cov=self.live.P[:2,:2].copy()
            if self.augmented:bias=self.live.x[4:].copy()
            if self.previous_end<self.ls:gs=self.live.gain_s
            sr=float(np.trace(self.live.last_sr)/2);cs=1./(1.+sr/(RAD**2))
        if not got and self.last_visual and self.last_visual['valid'] and now-self.last_visual['arrival']<.065:
            cv=float(self.last_visual['quality'])
        self.previous_end=self.ls
        return dict(b=bend,cov=cov,bias=bias,state=self.live.x.copy(),P=self.live.P.copy(),
                    bias_cov=self.live.P[4:,4:].copy() if self.augmented else np.zeros((2,2)),
                    gain_v=gv,gain_s=gs,confidence_v=cv,confidence_s=cs,accepted=used,nis=nis)


class LiveEstimator:
    def __init__(self,settings=None,t0=0.,assets=None):
        self.assets=assets
        if settings is None:
            self.assets=assets or FrozenAssets();settings=self.assets.settings
        self.c=settings;self.fixed=StreamingBend(settings,False,t0);self.proposed=StreamingBend(settings,True,t0)
        self.strain=[];self.visual=[];self.rows={k:[] for k in ('vision','strain','fixed','proposed','vision_hold')}
        self.last_now=None;self.optical_rate=np.zeros(2);self.last_valid_t=-np.inf;self.last_valid=None
    def step(self,t,strain_events=(),arrived_visual_events=()):
        t=float(t)
        if self.last_now is not None and t<=self.last_now:raise ValueError('Output clock moved backward')
        for original in strain_events:
            p={k:original[k] for k in ('t','b','R')};p['arrival_t']=original.get('arrival_t',p['t'])
            _validate_observation(p,True)
            if p['t']>t+1e-9 or p['arrival_t']>t+1e-9:raise ValueError('Future strain event')
            if self.strain and p['t']<=self.strain[-1]['t']:raise ValueError('Strain packet duplicate or reorder')
            self.strain.append(p)
        arrived=[]
        for original in arrived_visual_events:
            p={k:original[k] for k in ('t','b','R','valid','quality')}
            p['ready']=original.get('ready',p['valid'])
            p['arrival']=float(original.get('arrival',p['t']+self.c.delay))
            _validate_observation(p,bool(p['valid']))
            if abs(p['arrival']-p['t']-self.c.delay)>1e-8:raise ValueError('Visual arrival must equal frozen 0.20s delay')
            if p['arrival']>t+1e-9:raise ValueError('Future optical event')
            if self.visual and p['t']<=self.visual[-1]['t']:raise ValueError('Optical packet duplicate or reorder')
            previous=self.visual[-1] if self.visual else None
            rate=np.zeros(2)
            if p['valid']:
                if previous is not None and p['t']-self.last_valid_t<.085:
                    delta=(p['b']-previous['b'])/(p['t']-self.last_valid_t);speed=np.linalg.norm(delta)
                    delta*=min(1.,np.deg2rad(65)/max(speed,1e-10))
                    alpha=-np.expm1(-(p['t']-self.last_valid_t)/.12)
                    rate=(1-alpha)*self.optical_rate+alpha*delta
                self.last_valid_t=p['t'];self.last_valid=p
            else:self.last_valid_t=-np.inf
            self.optical_rate=rate;self.visual.append(p);arrived.append(p)
        source={}
        for name,filter_ in [('fixed',self.fixed),('proposed',self.proposed)]:source[name]=filter_.step(t,self.strain,arrived)
        last_s=self.strain[-1] if self.strain else None;last_v=self.visual[-1] if self.visual else None
        sa=t-last_s['t'] if last_s else np.inf
        va=t-self.last_valid['t'] if self.last_valid is not None else np.inf
        arrival_age=t-last_v['arrival'] if last_v else np.inf
        vis_ok=bool(last_v and last_v['valid'] and arrival_age<.065)
        source['strain']=dict(b=last_s['b'].copy() if last_s else np.full(2,np.nan),cov=last_s['R'].copy() if last_s else np.full((2,2),np.nan))
        source['vision']=dict(b=last_v['b']+self.optical_rate*(t-last_v['t']) if vis_ok else np.full(2,np.nan),
                              cov=last_v['R'].copy() if vis_ok else np.full((2,2),np.nan))
        source['vision_hold']=dict(b=self.last_valid['b'].copy() if self.last_valid is not None else np.full(2,np.nan),
                                  cov=self.last_valid['R'].copy() if self.last_valid is not None else np.full((2,2),np.nan))
        for name,z in source.items():
            if name=='strain':sample=last_s['t'] if last_s else -np.inf;arrival=last_s['arrival_t'] if last_s else -np.inf
            elif name in ('vision','vision_hold'):
                p=last_v if name=='vision' else self.last_valid
                sample=p['t'] if p is not None else -np.inf;arrival=p['arrival'] if p is not None else -np.inf
            else:
                sample=max(last_s['t'] if last_s else -np.inf,self.last_valid['t'] if self.last_valid is not None else -np.inf)
                arrival=max(last_s['arrival_t'] if last_s else -np.inf,self.last_valid['arrival'] if self.last_valid is not None else -np.inf)
            z.update(t=t,valid=bool(np.isfinite(z['b']).all()),age=t-sample,sample_t=sample,arrival_t=arrival,
                     strain_age=sa,vision_age=va,vision_arrival_age=arrival_age,
                     last_valid_visual_t=self.last_valid['t'] if self.last_valid is not None else -np.inf)
            self.rows[name].append({k:v.copy() if isinstance(v,np.ndarray) else v for k,v in z.items()})
        self.last_now=t
        return source
    def save(self,folder):
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        for name,rows in self.rows.items():_save_rows(folder/('estimate_'+name+'.npz'),rows)
        _save_rows(folder/'filter_strain_events.npz',self.strain)
        _save_rows(folder/'filter_visual_events.npz',self.visual)
        writej(folder/'estimator_provenance.json',dict(adapter_sha256=digest(__file__),
            assets=self.assets.hashes if self.assets else {},settings=self.c.__dict__))


def _save_rows(path,rows):
    if not rows:np.savez_compressed(path,t=np.empty(0));return
    keys=set.intersection(*(set(x) for x in rows))
    np.savez_compressed(path,**{k:np.asarray([r[k] for r in rows]) for k in sorted(keys)})


def _validate_observation(packet,valid):
    packet['t']=float(packet['t']);packet['b']=np.asarray(packet['b'],float).copy()
    packet['R']=np.asarray(packet['R'],float).copy()
    if not np.isfinite(packet['t']) or packet['b'].shape!=(2,) or packet['R'].shape!=(2,2):
        raise ValueError('Expected finite time, two-dimensional bend and 2x2 covariance')
    if valid:
        R=packet['R']
        if not np.isfinite(packet['b']).all() or not np.isfinite(R).all():raise ValueError('Nonfinite valid observation')
        if not np.allclose(R,R.T) or np.min(np.linalg.eigvalsh(R))<=0:raise ValueError('Covariance is not symmetric positive definite')
