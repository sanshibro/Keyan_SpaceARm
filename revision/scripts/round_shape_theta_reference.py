"""Original project theta fusion kernel, copied verbatim from Test/dual_strain_ai/run_confidence_ekf_fusion.py.
Upstream SHA-256 f327c72717dd8c0c9efecb2c9186d60ee488196843cb542b1be18a605bb6de58.
Command-based evaluation and plotting deliberately remain outside the kernel.
"""
import math
import numpy as np


def ekf_update(x, p, z, r):
    h = np.array([[1.0, 0.0]])
    y = float(z - h @ x)
    s = float(h @ p @ h.T + r)
    k = (p @ h.T) / s
    x = x + (k[:, 0] * y)
    p = (np.eye(2) - k @ h) @ p
    return x, p, y, s


def run_ekf(t, strain, vision, fresh, args):
    n = len(t)
    theta = np.full(n, np.nan)
    theta_dot = np.full(n, np.nan)
    vision_weight = np.zeros(n)
    strain_weight = np.zeros(n)
    innovation_vision = np.full(n, np.nan)
    innovation_strain = np.full(n, np.nan)

    first = np.where(np.isfinite(strain))[0]
    if len(first) == 0:
        raise ValueError("No finite strain prediction samples found.")
    i0 = int(first[0])
    x = np.array([float(strain[i0]), 0.0])
    p = np.diag([25.0, 25.0])
    prev_t = float(t[i0])

    r_strain = args.strain_sigma ** 2
    r_vision_base = args.vision_sigma ** 2
    q_accel = args.process_accel_sigma ** 2

    for i in range(i0, n):
        dt = float(t[i] - prev_t) if i > i0 and np.isfinite(t[i]) and np.isfinite(prev_t) else 0.02
        dt = min(max(dt, 0.001), 0.5)
        f = np.array([[1.0, dt], [0.0, 1.0]])
        q = q_accel * np.array(
            [[0.25 * dt ** 4, 0.5 * dt ** 3], [0.5 * dt ** 3, dt ** 2]]
        )
        x = f @ x
        p = f @ p @ f.T + q + np.diag([args.process_theta_sigma ** 2, 1e-5])

        # Strain is always available after baseline, but is assigned a larger noise.
        if np.isfinite(strain[i]):
            x, p, innovation_strain[i], _ = ekf_update(x, p, strain[i], r_strain)
            strain_weight[i] = 1.0 / r_strain

        # Vision confidence uses tracking freshness and consistency with strain.
        if fresh[i] and np.isfinite(vision[i]):
            diff = abs(vision[i] - strain[i]) if np.isfinite(strain[i]) else 0.0
            consistency = math.exp(-0.5 * (diff / max(args.consistency_gate_deg, 1e-6)) ** 2)
            confidence = max(args.min_vision_confidence, consistency)
            if diff > args.reject_vision_diff_deg:
                confidence = args.rejected_vision_confidence
            r_vision = r_vision_base / max(confidence, 1e-4)
            x, p, innovation_vision[i], _ = ekf_update(x, p, vision[i], r_vision)
            vision_weight[i] = 1.0 / r_vision

        denom = vision_weight[i] + strain_weight[i]
        if denom > 0:
            vision_weight[i] /= denom
            strain_weight[i] /= denom

        theta[i] = float(np.clip(x[0], -5.0, 110.0))
        theta_dot[i] = float(x[1])
        prev_t = float(t[i])

    return {
        "theta": theta,
        "theta_dot": theta_dot,
        "vision_weight": vision_weight,
        "strain_weight": strain_weight,
        "innovation_vision": innovation_vision,
        "innovation_strain": innovation_strain,
    }
