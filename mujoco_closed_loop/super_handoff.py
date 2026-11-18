"""Geometry qualification of a direct gap or actual whole-body bending-vector quintic.
The round-D API does not schedule a bridge or command an aircraft.
"""

import math
import numpy as np
from async_planner import GATES
from uav_continuum_whole_body_v2.execution import (
    whole_body_switch_deltas,
    whole_body_switch_gate,
    bending_vector,
    bending_vector_rate,
    bending_state_from_vector,
    evaluate_quintic_bridge,
)
from uav_continuum_whole_body_v2.types import ValidationResult
from super_planning_geometry import wrap


class DirectGap:
    def __init__(self, a, b):
        self.a = np.asarray(a, float)
        self.delta = np.asarray(b, float) - self.a
        self.delta[3] = wrap(self.delta[3])
        self.delta[5] = wrap(self.delta[5])
        self.duration = 1.0
        self.span_boundaries = np.array([0.0, 1.0])

    def evaluate(self, t, derivative=0):
        if derivative != 0:
            raise ValueError("direct geometric gap has no time derivative")
        return self.a + self.delta * np.asarray(t)[:, None]


class QuinticBendingBridge:
    def __init__(self, coefficients, duration):
        self.c = np.array(coefficients, dtype=float, copy=True)
        self.duration = float(duration)
        self.span_boundaries = np.array([0.0, self.duration])
        if (
            self.c.shape != (6, 6)
            or not np.isfinite(self.c).all()
            or not 0 < self.duration <= 900.0
        ):
            raise ValueError("invalid bridge parameters")

    def evaluate(self, t, derivative=0):
        if derivative:
            raise ValueError("use bridge-coordinate derivatives for boundary matching")
        q = evaluate_quintic_bridge(self.c, t, 0)
        return np.array([[*a[:4], *bending_state_from_vector(a[4:])] for a in q])


def coordinates(q):
    return np.r_[q[:4], bending_vector(*q[4:6])]


def rates(q, v):
    return np.r_[v[:4], bending_vector_rate(q[4], q[5], v[4], v[5])]


def accelerations(q, v, a):
    theta, phi = q[4:]
    theta_rate, phi_rate = v[4:]
    theta_acc, phi_acc = a[4:]
    axis = np.array([math.cos(phi), math.sin(phi)])
    tangent = np.array([-axis[1], axis[0]])
    curvature_acc = (theta_acc - theta * phi_rate**2) * axis + (
        2 * theta_rate * phi_rate + theta * phi_acc
    ) * tangent
    return np.r_[a[:4], curvature_acc]


def review(adapter, state, rate, spline, bridge=None, start_acceleration=None):
    target = spline.evaluate([0.0], 0)[0]
    vr = spline.evaluate([0.0], 1)[0]

    def reject(why, details, i=0, t=0.0):
        return ValidationResult(False, "handoff_rejected", details, i, t, why)

    if any(np.shape(x) != (6,) or not np.isfinite(x).all() for x in (state, rate, target, vr)):
        return reject("invalid_handoff_state", {})
    if bridge is None:
        gaps = whole_body_switch_deltas(state, rate, target, vr)
        issues = whole_body_switch_gate(gaps, GATES)
        if issues:
            return reject("direct_gate", dict(violations=issues, gaps=gaps))
        path = DirectGap(state, target)
        kind = "direct_gap"
    else:
        if not isinstance(bridge, QuinticBendingBridge):
            raise ValueError("expected actual bending-vector quintic")
        if np.shape(start_acceleration) != (6,) or not np.isfinite(start_acceleration).all():
            return reject("missing_bridge_acceleration", {})
        for t, q, v, a in [
            (0.0, state, rate, start_acceleration),
            (bridge.duration, target, vr, spline.evaluate([0.0], 2)[0]),
        ]:
            gap = evaluate_quintic_bridge(bridge.c, [t], 0)[0] - coordinates(q)
            gap[3] = wrap(gap[3])
            if (
                np.max(np.abs(gap)) > 1.0e-6
                or np.max(np.abs(evaluate_quintic_bridge(bridge.c, [t], 1)[0] - rates(q, v)))
                > 1.0e-6
            ):
                return reject("bridge_boundary_mismatch", dict(time=t))
            if (
                np.max(
                    np.abs(evaluate_quintic_bridge(bridge.c, [t], 2)[0] - accelerations(q, v, a))
                )
                > 1.0e-6
            ):
                return reject("bridge_acceleration_mismatch", dict(time=t))
        path = bridge
        kind = "quintic_geometry_only"
    result = adapter.fine.proof(path)
    if not result["accepted"]:
        w = result["first"]
        return reject(w["reason"], {"route": kind, "transition": result}, w["index"], w["time_s"])
    proof = adapter.validator.validate(spline)
    proof.metrics["handoff_geometry"] = {
        "route": kind,
        "transition": result,
        "map_hash": adapter.snapshot.map_hash,
    }
    return proof
