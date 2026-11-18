"""Supplementary outward-speed trial, using the preregistered F+ hover baseline."""
import argparse
from pathlib import Path

from round_fp_run import P, setpoint
from round_f_plant import bending_command
from round_f_sitl import digest, run


def fast_outward(t):
    phase, target, angle = setpoint(t, 'nominal')
    if t is not None and 16. <= t < 22.:
        angle = bending_command(1. + 2. * (t - 16.)) if t < 19. else 118.
        phase = 'bend' if t < 19. else 'hold'
    return phase, target, angle


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    parser.add_argument('seed', type=int, choices=[int(s) for s in P['seed_order']])
    args = parser.parse_args()
    mission = dict(seed=args.seed, variant='fastbend', release_noise_stream=[args.seed, 1],
                   wrapper_sha256=digest(Path(__file__)),
                   common_schedule_sha256=digest(Path(__file__).with_name('round_fp_run.py')))
    run(args.out.resolve(), '%d-fastbend' % args.seed, .4, reference_fn=fast_outward,
        score_end=P['score_end_s'], noise_seed=args.seed, protocol_name='round_fp_flex.json',
        mission_record=mission)
