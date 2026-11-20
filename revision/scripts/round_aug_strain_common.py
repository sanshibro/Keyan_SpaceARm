"""Strain-only experiment configuration and physical readout generation.

Never imported by the deployed inverse. Plant material/environment parameters are
scoring metadata, not inference inputs. Event strengths/times are fixed a priori.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from slim_sensor import SlimReadout

CONDITIONS = ('nominal', 'common', 'differential', 'single', 'late')
PLANT_MEMORY = dict(half_width=[.018,.020,.022,.019], tau_s=[1.7,2.,2.3,1.9],
                    play_weight=[.46,.50,.54,.48], transient_weight=[.018,.020,.022,.019])
DESIGN = dict(schema=1, preload=.30, temperature_C=25.,
    nominal_material=PLANT_MEMORY, calibration_material_fractional_variation=.06,
    heldout_material_fractional_variation=.15,
    formal_ramp_s=[18.,22.], late_ramp_s=[29.,33.], visual_occlusion_s=[31.,35.],
    visual_delay_s=.20,
    offset_ratio=dict(nominal=[0.,0.,0.,0.],common=None,
        differential=[.018,-.0045,-.018,.0045],single=None,
        late=[.018,-.0045,-.018,.0045]),
    adc_offset_counts=dict(common=[40.,40.,40.,40.],single=[40.,0.,0.,0.]),
    rationale='Fixed before target scoring. Ratio drift times installed gain gives ADC counts. '
        'Ramps are smooth quintics; all regular faults reach plateau nine seconds before occlusion. '
        'Late repeats the differential amplitude across the blackout boundary as a stress control.',
    scope='Four simulated SLIM channels use an empirical three-channel box noise/timing pool. '
        'Dynamic material law is a conditional simulator model, not a newly measured four-strip law.')

def seed(key):
    return int(hashlib.sha256(key.encode()).hexdigest()[:8],16)

def readj(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def savej(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def environment(slim,protocol):
    slim=Path(slim)
    return readj(protocol),readj(slim/'box-profile.json'),np.load(slim/'stationary-box.npz')['events']

def material_for(key,variation):
    rng=np.random.default_rng(seed('independent-material:'+key))
    return {k:(np.asarray(v)*(1.+rng.uniform(-variation,variation,4))).tolist()
            for k,v in PLANT_MEMORY.items()}

def sample_geometry(key,t,length_m,spec,profile,noise,condition='nominal',material=None):
    if condition not in CONDITIONS:raise ValueError(condition)
    material=PLANT_MEMORY if material is None else material
    gain=np.asarray(profile['slope_counts_per_unit_ratio'])
    code_offset=(np.asarray(DESIGN['adc_offset_counts'][condition]) if condition in ('common','single')
                 else np.asarray(DESIGN['offset_ratio'][condition])*gain)
    lo,hi=DESIGN['late_ramp_s'] if condition=='late' else DESIGN['formal_ramp_s']
    def drift(ts):
        z=np.clip((ts-lo)/(hi-lo),0.,1.);smooth=z**3*(10.-15.*z+6.*z*z)
        return smooth[:,None]*code_offset
    # Same key intentionally uses identical physical sample intervals and box noise
    # across all conditions; condition changes only the prescribed offset function.
    box=SlimReadout(spec,profile,noise,seed('aug-box:'+key))
    d=box.sample(t,length_m,DESIGN['preload'],memory=material,offset_profile=drift)
    if np.any(d['overrange']) or np.any(d['slack']):
        raise ValueError('Physical strip left the prescribed tensile operating range')
    return d,dict(material=material,temperature_C=DESIGN['temperature_C'],
                  onset_s=lo,plateau_s=hi,offset_codes=code_offset.tolist(),
                  offset_ratio=(code_offset/gain).tolist(),noise_seed=seed('aug-box:'+key))
