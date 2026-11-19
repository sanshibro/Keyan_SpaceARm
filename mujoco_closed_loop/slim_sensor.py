"""Four tensile Elastreme Sense Slim gauges and a replayable multichannel readout.

The rig takes MuJoCo body poses only. The instrument takes path lengths and sampled
readout characteristics only. Neither consumes arm commands, camera angles or mocap.
"""
from pathlib import Path
import numpy as np

class SlimGaugeRig:
    def __init__(self,model,spec,preload=None):
        self.free=spec['free_active_length_m']
        self.preload=spec['preload_primary'] if preload is None else float(preload)
        self.center=spec['mount_center_m']
        self.installed=self.free*(1.+self.preload)
        self.radius=spec['arm_radius_m']
        self.ids=[model.body('seg%d'%i).id for i in range(8)]+[model.body('tool').id]
        station=np.arange(9)*.035
        lo,hi=self.center+np.array([-.5,.5])*self.installed
        if lo<0 or hi>station[-1]:raise ValueError('Gauge extends beyond flexible segment')
        samples=np.r_[lo,station[(station>lo)&(station<hi)],hi]
        self.link=np.clip(np.searchsorted(station,samples,side='right')-1,0,7)
        self.frac=(samples-station[self.link])/.035
        angles=np.arange(4)*np.pi/2
        self.offsets=self.radius*np.column_stack((np.cos(angles),np.sin(angles),np.zeros(4)))
    def paths(self,data):
        rot=data.xmat[self.ids].reshape(-1,3,3)
        rings=data.xpos[self.ids,None,:]+self.offsets[None,:,:]@rot.transpose(0,2,1)
        f=self.frac[:,None,None]
        return rings[self.link]*(1.-f)+rings[self.link+1]*f
    def lengths(self,data):
        return np.linalg.norm(np.diff(self.paths(data),axis=0),axis=-1).sum(axis=0)

class SlimReadout:
    def __init__(self,spec,profile,baseline_events,seed):
        self.spec=spec;self.profile=profile;self.reference=baseline_events
        self.rng=np.random.default_rng(seed)
        curve=spec['resistance_curve_25C']
        self.strain=np.array(curve['engineering_strain']);self.resistance=np.array(curve['relative_resistance_change'])
    def ratio(self,strain):
        """Unloaded plateau for slack; operation above the rated maximum is invalid."""
        e=np.asarray(strain,float)
        out=np.interp(np.maximum(e,0.),self.strain,self.resistance)
        return np.where((e<=1+1e-12)&np.isfinite(e),out,np.nan)
    def sample(self,physics_t,lengths,preload,gain_scale=1.,offset=None):
        physics_t=np.asarray(physics_t);lengths=np.asarray(lengths)
        if np.any(np.diff(physics_t)<=0) or lengths.shape!=(len(physics_t),4):
            raise ValueError('Expected four synchronized paths')
        # Use contiguous stationary residual/interval blocks to preserve code discreteness and timing.
        events=self.reference
        chunks=[];elapsed=float(physics_t[0])-.72
        timestamps=[];free=self.spec['free_active_length_m']
        while elapsed<=physics_t[-1]:
            start=int(self.rng.integers(0,len(events)-32));extra=int(self.rng.integers(0,len(events)-32))
            donor=int(self.rng.integers(1,4));block=events[start:start+32]
            times=elapsed+np.cumsum(block[:,0]);elapsed=times[-1]
            timestamps.extend(times);chunks.append(np.column_stack((block[:,1:4],events[extra:extra+32,donor])))
        t=np.asarray(timestamps);keep=t<=physics_t[-1]+1e-9;t=t[keep];noise=np.concatenate(chunks)[keep]
        # The neutral preload is fixed during the pre-roll; the neutral button sets the box baseline.
        L=np.column_stack([np.interp(t,physics_t,lengths[:,j],left=free*(1.+preload)) for j in range(4)])
        extension=L/free-1.;relative=self.ratio(extension)
        gain=np.asarray(self.profile['slope_counts_per_unit_ratio'])*gain_scale
        box_zero=np.array(self.profile['neutral_codes'])
        analog=box_zero+(relative-self.ratio(preload))*gain+noise
        if offset is not None:
            analog+=np.clip((t-29.)/4.,0,1)[:,None]*np.asarray(offset)
        ok=np.isfinite(analog)
        codes=np.full(analog.shape,-1,dtype=np.int16)
        codes[ok]=np.rint(np.clip(analog[ok],0,4095)).astype(np.int16)
        tared=(t<physics_t[0]-1e-8);tare=np.median(codes[tared],axis=0)
        visible=~tared
        return dict(t=t[visible],adc=codes[visible],tare=tare,initial_adc=codes[tared][-1],strain=extension[visible],dr_over_r=relative[visible],
                    slack=(extension[visible]<-1e-8),overrange=(~ok[visible]),
                    rail=((codes[visible]==0)|(codes[visible]==4095)))
