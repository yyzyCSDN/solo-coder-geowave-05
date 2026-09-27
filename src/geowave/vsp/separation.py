from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from geowave.spectral.fk import fk_transform, velocity_fan_mask, apply_fk_mask

@dataclass(frozen=True)
class Wavefields:
    downgoing: np.ndarray
    upgoing: np.ndarray
    method: str

def running_median(x: np.ndarray, half_width: int) -> np.ndarray:
    """Column-wise median over a sliding depth window (boundaries shrink)."""
    n = x.shape[0]
    out = np.empty_like(x)
    for i in range(n):
        lo, hi = max(0, i - half_width), min(n, i + half_width + 1)
        out[i] = np.median(x[lo:hi], axis=0)
    return out

def median_separate(flattened: np.ndarray, half_width: int = 3) -> Wavefields:
    """Median-filter separation on first-break-flattened data.

    After flattening the downgoing direct arrival is horizontal and passes
    the median filter; dipping upgoing reflections are rejected. Downgoing =
    median panel, upgoing = residual.
    """
    if half_width < 1:
        raise ValueError('half_width must be >= 1')
    down = running_median(flattened, half_width)
    return Wavefields(down, flattened - down, f'median({2 * half_width + 1})')

def fk_separate(data: np.ndarray, dt_s: float, dz_m: float,
                down_apparent_velocity_mps: float,
                slowness_half_width_spm: float | None = None) -> Wavefields:
    """Fan-filter separation on the original (unflattened) depth-time panel.

    The downgoing first break follows t = z/V (positive time-depth slope),
    which maps to positive wavenumber for positive frequency. A narrow fan
    around +1/V is kept as downgoing and removed; everything else is
    upgoing. ``slowness_half_width_spm`` widens the fan in slowness to retain
    the source ghost; a default 15% band around 1/V is used when omitted.
    """
    panel = fk_transform(data, dt_s, dz_m)
    p0 = 1.0 / down_apparent_velocity_mps
    hp = slowness_half_width_spm if slowness_half_width_spm is not None else 0.15 * p0
    v_hi = 1.0 / max(p0 - hp, 1e-6)
    v_lo = 1.0 / (p0 + hp)
    F, K = np.meshgrid(panel.frequencies_hz, panel.wavenumbers_per_m)
    fan = velocity_fan_mask(panel, v_lo, v_hi) & (np.sign(F) == np.sign(K))
    fan[:, 0] = False  # do not swallow DC
    down = apply_fk_mask(data, dt_s, dz_m, fan)
    return Wavefields(down, data - down, 'fk-fan')
