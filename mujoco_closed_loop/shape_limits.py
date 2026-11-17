"""Common bending envelope for new simulation runs (radians, no state clipping)."""
import math
import numpy as np
HARD_THETA_MAX = math.radians(120.)
COMMAND_THETA_MAX = math.radians(118.)
PLAN_THETA_MAX = COMMAND_THETA_MAX

def bounded_command(theta, phi, kinematic=False):
    if not all(math.isfinite(v) for v in (theta,phi)) or not 0 <= theta <= HARD_THETA_MAX+1e-12:
        raise ValueError('Bending request must be finite and within [0, 120] degrees')
    return min(theta,HARD_THETA_MAX if kinematic else COMMAND_THETA_MAX),phi

def joint_bending_metrics(pairs):
    pairs=np.asarray(pairs,dtype=float)
    if pairs.ndim!=2 or pairs.shape[1]!=2 or not np.isfinite(pairs).all():
        raise ValueError('Expected finite bending-joint pairs')
    # Total discrete curvature is conservative when segments bend in different planes.
    total=float(np.sum(np.linalg.norm(pairs,axis=1)))
    net=float(np.linalg.norm(np.sum(pairs,axis=0)))
    return dict(total_curvature_deg=math.degrees(total),net_bending_deg=math.degrees(net),
                within_hard_limit=total<=HARD_THETA_MAX+1e-9)
