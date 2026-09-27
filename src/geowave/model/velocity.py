from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from geowave.util.mathx import lerp

@dataclass(frozen=True)
class VelocityPick:
    time_s: float
    velocity_mps: float
    confidence: float = 1.0

@dataclass
class VelocityFunction:
    x: float
    y: float
    picks: list[VelocityPick]

    def __post_init__(self):
        self.picks = sorted(self.picks, key=lambda p: p.time_s)
        if any((p.velocity_mps <= 0 for p in self.picks)):
            raise ValueError('velocity must be positive')

    def at(self, t: float) -> float:
        if not self.picks:
            raise ValueError('empty velocity function')
        if t <= self.picks[0].time_s:
            return self.picks[0].velocity_mps
        if t >= self.picks[-1].time_s:
            return self.picks[-1].velocity_mps
        for a, b in zip(self.picks, self.picks[1:]):
            if a.time_s <= t <= b.time_s:
                return lerp(a.time_s, a.velocity_mps, b.time_s, b.velocity_mps, t)
        return self.picks[-1].velocity_mps

@dataclass
class VelocityModel1D:
    depths_m: np.ndarray
    velocities_mps: np.ndarray

    def __post_init__(self):
        self.depths_m = np.asarray(self.depths_m, float)
        self.velocities_mps = np.asarray(self.velocities_mps, float)
        if self.depths_m.ndim != 1 or self.depths_m.shape != self.velocities_mps.shape:
            raise ValueError('shape mismatch')
        if np.any(np.diff(self.depths_m) < 0):
            raise ValueError('depths must increase')
        if np.any(self.velocities_mps <= 0):
            raise ValueError('nonpositive velocity')

    def velocity_at_depth(self, z: float) -> float:
        return float(np.interp(z, self.depths_m, self.velocities_mps))

    def one_way_time(self, z: float, steps: int=512) -> float:
        if z <= 0:
            return 0.0
        zs = np.linspace(0, z, steps)
        vs = np.interp(zs, self.depths_m, self.velocities_mps)
        integrator = getattr(np, 'trapezoid', np.trapz)
        return float(integrator(1.0 / vs, zs))
