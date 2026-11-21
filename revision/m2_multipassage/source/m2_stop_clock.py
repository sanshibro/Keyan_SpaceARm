"""C² braking at the rolling commit station, on the already certified spatial path."""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class StopClock:
    begin: float
    ramp: float

    @property
    def duration(self):
        return self.begin + self.ramp

    def derivatives(self, seconds):
        t = np.asarray(seconds, dtype=float)
        u = np.clip((t - self.begin) / self.ramp, 0., 1.)
        phase = self.begin + self.ramp * (u - u**3 + .5*u**4)
        rate = 1. - 3.*u**2 + 2.*u**3
        acceleration = (-6.*u + 6.*u**2) / self.ramp
        return np.where(t < self.begin, t, phase), rate, acceleration

    def sample(self, curve, seconds):
        t, rate, acceleration = self.derivatives([seconds])
        q = curve.evaluate(t, 0)[0]
        v = curve.evaluate(t, 1)[0]
        a = curve.evaluate(t, 2)[0]
        return q, v*rate[0], a*rate[0]**2 + v*acceleration[0]


def commit_clock(curve, distance):
    t = np.linspace(0., curve.duration, int(np.ceil(curve.duration/.04)) + 1)
    x = curve.evaluate(t, 0)[:, 0]
    station = x[0] + distance
    crossed = np.flatnonzero(x >= station)
    if not len(crossed):
        return None
    i = int(crossed[0])
    hit = brentq(lambda s: curve.evaluate([s], 0)[0, 0] - station, t[i-1], t[i])
    for ramp in (.9, 1.2, 1.5, 1.9, 2.4, 3., 3.8):
        begin = hit - .5*ramp
        if begin < 0.:
            break
        clock = StopClock(begin, ramp)
        tt = np.linspace(begin, clock.duration, int(np.ceil(ramp/.01)) + 1)
        phase, speed, acc = clock.derivatives(tt)
        q = curve.evaluate(phase, 0)
        v = curve.evaluate(phase, 1)
        a = curve.evaluate(phase, 2)
        jerk = curve.evaluate(phase, 3)
        u = (tt-begin)/ramp
        jclock = (-6.+12.*u)/ramp**2
        world_a = a * speed[:, None]**2 + v * acc[:, None]
        world_j = jerk * speed[:, None]**3 + 3.*a*(speed*acc)[:, None] + v*jclock[:, None]
        angular = np.abs(world_a[:, 3:]);angular[q[:, 4] <= .03, 2] = 0.
        if (np.linalg.norm(world_a[:, :3], axis=1).max() <= 1. and
                np.linalg.norm(world_j[:, :3], axis=1).max() <= 3. and
                (angular <= np.deg2rad([90., 100., 120.])).all()):
            return clock
    raise ValueError('No admissible stop profile at the registered rolling commit station')
