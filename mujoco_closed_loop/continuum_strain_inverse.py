"""Causal four-channel continuous material inverse; runtime inputs are ADC only.

Calibration labels belong to independent bench motions. There are no flight labels,
condition selectors, discrete health experts, or simulator parameters in estimate().
All angles/covariances are radians/radian squared.
"""
import json
from pathlib import Path
import numpy as np


def polynomial(d):
    u, v = np.asarray(d).T
    return np.column_stack((np.ones(len(d)), u, v, u*u, u*v, v*v,
                            u**3, u*u*v, u*v*v, v**3))


def fit_polynomial(x, y, ridge=1e-7):
    scale = np.maximum(np.std(x, axis=0), 1e-8)
    scale[0] = 1.
    z = x / scale
    reg = np.eye(z.shape[1]) * len(z) * ridge
    reg[0, 0] = 0.
    return np.linalg.solve(z.T @ z + reg, z.T @ y) / scale[:, None]


class ContinuousStrainInverse:
    def __init__(self, model):
        if isinstance(model, (str, Path)):
            model = json.loads(Path(model).read_text())
        self.model = model
        self.gain = np.asarray(model['gains_counts_per_ratio'], float)
        self.curve_e = np.asarray(model['curve_strain'], float)
        self.curve_r = np.asarray(model['curve_ratio'], float)
        self.neutral = float(np.interp(model['preload'], self.curve_e, self.curve_r))
        mat = model['calibrated_memory']
        self.h, self.tau, self.wp, self.wt = [np.broadcast_to(np.asarray(mat[k], float), (4,)).copy()
            for k in ('half_width', 'tau_s', 'play_weight', 'transient_weight')]
        if np.any(self.wp >= .95) or np.any(self.tau <= 0):
            raise ValueError('Inverse needs positive material slope and relaxation time')
        self.coef = np.asarray(model['shape_coefficients'], float)
        self.even_coef = np.asarray(model['even_coefficients'], float)
        self.cov0 = np.asarray(model['base_covariance_rad2'], float)
        self.common_scale = np.asarray(model['common_residual_scale'], float)
        self.geometry_coef = (np.asarray(model['geometry_forward_coefficients'], float)
                              if 'geometry_forward_coefficients' in model else None)

    def solve_shape(self, e, usable):
        d = .5*(e[:2]-e[2:])
        bend = polynomial(d[None,:])[0] @ self.coef
        if self.geometry_coef is None:
            return bend
        # Four physical surface strains overdetermine two bend coordinates.
        # A converter rail is censored, not a measured shortening of that strip.
        for _ in range(4):
            u,v = bend
            pred = polynomial(bend[None,:])[0] @ self.geometry_coef
            jac = np.array([[0,1,0,2*u,v,0,3*u*u,2*u*v,v*v,0],
                            [0,0,1,0,u,2*v,0,u*u,2*u*v,3*v*v]]) @ self.geometry_coef
            J = jac[:,usable].T
            delta = np.linalg.solve(J.T@J+np.eye(2)*1e-8,J.T@(e-pred)[usable])
            bend += np.clip(delta,-.25,.25)
        return bend

    def static_shape(self, ratio):
        e = np.column_stack([np.interp(ratio[:, j], self.curve_r, self.curve_e) for j in range(4)])
        d = .5 * (e[:, :2] - e[:, 2:])
        common = .5 * (e[:, :2] + e[:, 2:])
        x = polynomial(d)
        return x @ self.coef, common - x @ self.even_coef, e

    def estimate(self, t, adc, tare, initial_adc=None):
        t, adc, tare = np.asarray(t, float), np.asarray(adc, float), np.asarray(tare, float)
        if t.ndim != 1 or adc.shape != (len(t), 4) or tare.shape != (4,) or not len(t):
            raise ValueError('Expected t[N], adc[N,4], tare[4]')
        if not np.all(np.diff(t) > 0) or not np.isfinite(adc).all():
            raise ValueError('Nonmonotonic acquisition times or invalid ADC')
        measured = self.neutral + (adc - tare) / self.gain
        # The material is equilibrated at the per-run neutral tare before acquisition.
        prev = np.full(4, self.neutral)
        play = prev.copy()
        transient = np.zeros(4)
        corrected = np.zeros_like(measured); shapes = np.zeros((len(t),2))
        for i, stamp in enumerate(t):
            dt = stamp - t[i-1] if i else float(self.model.get('nominal_sample_s', .025))
            decay = np.exp(-dt / self.tau)
            c = -np.expm1(-dt / self.tau) * self.tau / dt
            # Invert y=x+w_p*(play(x)-x)+w_t*transient(x) exactly in
            # each of its three monotone affine branches (same state as SlimReadout).
            slope = 1. - self.wp + self.wt*c
            base = self.wt*(decay*transient - c*prev)
            center = (measured[i] - self.wp*play - base) / slope
            lower = (measured[i] - self.wp*self.h - base) / (1.+self.wt*c)
            upper = (measured[i] + self.wp*self.h - base) / (1.+self.wt*c)
            value = np.where(center < play-self.h, lower,
                             np.where(center > play+self.h, upper, center))
            value = np.clip(value, self.curve_r[0]-.10, self.curve_r[-1]+.10)
            e = np.interp(value,self.curve_r,self.curve_e)
            usable = (adc[i]>1) & (adc[i]<4094)
            shapes[i] = self.solve_shape(e,usable)
            if self.geometry_coef is not None and not np.all(usable):
                expected_e = polynomial(shapes[i:i+1])[0] @ self.geometry_coef
                expected_r = np.interp(expected_e,self.curve_e,self.curve_r)
                value = np.where(usable,value,expected_r)
            play = np.clip(play, value-self.h, value+self.h)
            transient = decay*transient + c*(value-prev)
            corrected[i] = value
            prev = value
        bend, common, strain = self.static_shape(corrected)
        if self.geometry_coef is not None:bend=shapes
        point, _, _ = self.static_shape(measured)
        R = np.empty((len(t), 2, 2)); speed = np.zeros(len(t)); rev = np.zeros(len(t))
        smoothed = bend[0].copy(); last_velocity = np.zeros(2); reversal = 0.
        residual_size = np.linalg.norm(common / self.common_scale, axis=1)
        for i, stamp in enumerate(t):
            dt = stamp-t[i-1] if i else float(self.model.get('nominal_sample_s', .025))
            velocity = (bend[i]-smoothed) / .18
            speed[i] = np.linalg.norm(velocity)
            if velocity @ last_velocity < 0 and np.linalg.norm(last_velocity) > .025:
                reversal = 1.
            reversal *= np.exp(-dt/.35); rev[i] = reversal
            smoothed += -np.expm1(-dt/.18) * (bend[i]-smoothed)
            last_velocity = velocity
            rate_factor = min(4., (speed[i]/.7)**2)
            common_factor = min(24., max(0., residual_size[i]-2.)**2)
            edge_factor = max(0., (np.linalg.norm(bend[i])-np.deg2rad(85.))/np.deg2rad(15.))**2
            rail = bool(np.any((adc[i] <= 1) | (adc[i] >= 4094)))
            factor = 1. + .30*rate_factor + .5*reversal + .35*common_factor + 2*edge_factor
            if rail: factor = max(factor, 100.)
            R[i] = self.cov0 * factor
        return dict(b=bend, R=R, point_b=point, common_residual=common,
                    material_corrected_ratio=corrected, speed_rad_s=speed,
                    reversal_score=rev, estimated_strain=strain)
