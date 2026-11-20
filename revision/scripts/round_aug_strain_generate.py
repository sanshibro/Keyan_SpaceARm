"""Generate held-out ADC streams from sanitized length/time and run frozen inverse.

This script refuses any geometry NPZ containing extra fields, including labels.
There is no scoring or tuning on target flight data in this entry point.
"""
import argparse
from pathlib import Path
import numpy as np
from continuum_strain_inverse import ContinuousStrainInverse
from round_aug_strain_common import CONDITIONS,DESIGN,environment,material_for,readj,savej,sha,sample_geometry

def generate(args):
    out=Path(args.output);geometry=Path(args.geometry)
    model=Path(args.model);inverse=ContinuousStrainInverse(model)
    spec,profile,noise=environment(args.slim,args.protocol)
    frozen=readj(out/'strain-model'/'freeze.json')
    if sha(model)!=frozen['model_sha256']:raise ValueError('Frozen model hash changed')
    if DESIGN != readj(out/'strain-model'/'generation-design.json'):raise ValueError('Design changed after bench freeze')
    files=sorted(geometry.glob('*.npz')) if not args.case else [geometry/(args.case+'.npz')]
    for path in files:
        with np.load(path,allow_pickle=False) as stream:
            if set(stream.files) != {'t','length_m'}:
                raise ValueError('Only sanitized t,length_m geometry is permitted: '+str(path))
            t=stream['t'];length_m=stream['length_m']
        metadata={}
        material=material_for('flight:'+path.stem,DESIGN['heldout_material_fractional_variation'])
        for condition in CONDITIONS:
            raw,physical=sample_geometry(path.stem,t,length_m,spec,profile,noise,condition,material=material)
            result=inverse.estimate(raw['t'],raw['adc'],raw['tare'],raw['initial_adc'])
            folder=out/'strain'/path.stem;folder.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(folder/(condition+'.npz'),t=raw['t'],adc=raw['adc'],
                tare=raw['tare'],initial_adc=raw['initial_adc'],rail=raw['rail'],**result)
            metadata[condition]=physical
            print(path.stem,condition,len(raw['t']),'ADC events',flush=True)
        # Scoring metadata is a separate branch, not co-located in observation NPZ.
        savej(out/'scoring-metadata'/'strain'/(path.stem+'.json'),dict(
            geometry_sha256=sha(path),model_sha256=sha(model),generation=metadata,
            config_sha256=sha(out/'strain-model'/'generation-design.json')))

if __name__=='__main__':
    root=Path(__file__).resolve().parents[2]
    p=argparse.ArgumentParser();p.add_argument('--output',default=root/'revision/data/round_augmented_fusion')
    p.add_argument('--geometry',default=root/'revision/data/round_augmented_fusion/geometry')
    p.add_argument('--model',default=root/'revision/data/round_augmented_fusion/strain-model/model.json')
    p.add_argument('--slim',default=root/'revision/data/round_slim_sensor')
    p.add_argument('--protocol',default=root/'revision/protocols/round_slim_protocol.json')
    p.add_argument('--case');generate(p.parse_args())
