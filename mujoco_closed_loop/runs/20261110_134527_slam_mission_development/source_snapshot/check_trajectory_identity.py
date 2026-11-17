"""V2 spline recovery must preserve geometry and derivatives between samples."""
import numpy as np
from trajectory_tools import reconstruct_spline
from uav_continuum_whole_body_v2.trajectory import SplineTrajectory
rng=np.random.default_rng(72)
for stationary in (True,False):
    controls=rng.normal(0,.2,(8,6));durations=rng.uniform(.4,1.4,5)
    original=SplineTrajectory(controls,durations,stationary_start=stationary,stationary_end=True)
    sampled=original.sample(.05,metadata=dict(span_durations=durations.tolist(),conditioned_initial_rate=None if stationary else [0]*6))
    recovered=reconstruct_spline(sampled);times=np.linspace(0,original.duration,200)
    for order in range(4):
        assert np.allclose(original.evaluate(times,order),recovered.evaluate(times,order),atol=1e-8,rtol=1e-8)
print('PASS: original geometry, velocity, acceleration and jerk at 200 off-grid times, both endpoint modes')
