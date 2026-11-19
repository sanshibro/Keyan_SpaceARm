"""Isolated PX4 SITL flight on the frozen one-gate mission."""
import argparse
from functools import partial
import json
from pathlib import Path
from round_f_sitl import run, digest
from round_g1_model import DATA, P, PROTOCOL, make_plant
from round_g1_plan import reference, wide_reference


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('seed',type=int)
    ap.add_argument('mode',choices=['wide','straight','fixed','variable_nominal','variable_timed'])
    a=ap.parse_args()
    plans=DATA/'plans.json'
    case='%d-%s'%(a.seed,a.mode)
    iswide=a.mode=='wide'
    fn=wide_reference() if iswide else partial(reference,a=json.loads(plans.read_text())['selected'][a.mode],mode=a.mode)
    scripts=[Path(__file__).with_name('round_g1_'+n+'.py') for n in ('model','plan','run')]
    mission=dict(seed=a.seed,mode=a.mode,plan_sha256=digest(plans),phase_seed=[a.seed,1],
                 wrapper_sha256={p.name:digest(p) for p in scripts})
    run(DATA/'runs'/case,case,.4,reference_fn=fn,score_end=39. if iswide else P['deadline_s'],
        noise_seed=a.seed,protocol_name=PROTOCOL.name,mission_record=mission,
        plant_factory=partial(make_plant,'wide' if iswide else 'study'))


if __name__=='__main__':
    main()
