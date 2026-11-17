"""Recover the exact reused V2 spline from its full-rank sampled representation."""
import numpy as np
from planner_adapter import ROOT
from scipy.interpolate import BSpline
from uav_continuum_whole_body_v2.trajectory import SplineTrajectory

def reconstruct_spline(trajectory):
    durations=np.asarray(trajectory.metadata['span_durations'],dtype=float)
    count=len(durations)+3
    stationary=trajectory.metadata.get('conditioned_initial_rate') is None
    template=SplineTrajectory(np.zeros((count,6)),durations,stationary_start=stationary,stationary_end=True)
    phase=template._phase(trajectory.times)[0]
    basis=BSpline(template.knots,np.eye(count),3,extrapolate=False)(phase)
    controls,residuals,rank,_=np.linalg.lstsq(basis,trajectory.states,rcond=None)
    if rank!=count:raise ValueError('Cannot recover a unique V2 spline')
    spline=SplineTrajectory(controls,durations,stationary_start=stationary,stationary_end=True)
    for order,expected in enumerate((trajectory.states,trajectory.velocities,trajectory.accelerations,trajectory.jerks)):
        if not np.allclose(spline.evaluate(trajectory.times,order),expected,atol=1e-7,rtol=1e-7):
            raise ValueError('Recovered V2 spline derivative mismatch: '+str(order))
    return spline
