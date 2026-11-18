"""Use the native F SITL transport and a fixed-position shape schedule."""
import argparse
import functools
import json
import math
from pathlib import Path

import numpy as np

from round_f_plant import ROOT, bending_command
from round_f_sitl import digest, run

PROTOCOL = ROOT / 'revision/protocols/round_fp.json'
P = json.loads(PROTOCOL.read_text())
TARGET = np.array(P['target_ENU_m'])


def setpoint(t, variant):
    if t is None:
        return 'preflight', TARGET, 0.
    if t < 9.:
        return 'liftoff', TARGET, 0.
    if t < 16.:
        return 'hover', TARGET, 0.
    start = P['return_start_s']
    if t < start:
        phase = 'bend' if t < 22. else 'hold'
        angle = 0. if variant == 'straight' else bending_command(t - 15.)
        return phase, TARGET, angle
    duration = P['durations_s'][variant]
    if duration and t < start + duration:
        u = (t - start) / duration
        return 'retract', TARGET, 59. * (1. + math.cos(math.pi * u))
    return 'recover', TARGET, 0.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    parser.add_argument('seed', type=int, choices=[int(x) for x in P['seed_order']])
    parser.add_argument('variant', choices=P['durations_s'])
    args = parser.parse_args()
    tag = '%d-%s' % (args.seed, args.variant)
    mission = dict(protocol=PROTOCOL.name, seed=args.seed, variant=args.variant,
                   wrapper_sha256=digest(Path(__file__)), release_noise_stream=[args.seed, 1])
    run(args.out.resolve(), tag, .4, reference_fn=functools.partial(setpoint, variant=args.variant),
        score_end=P['score_end_s'], noise_seed=args.seed, protocol_name=PROTOCOL.name,
        mission_record=mission)


if __name__ == '__main__':
    main()
